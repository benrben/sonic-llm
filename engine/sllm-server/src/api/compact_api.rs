//! HTTP side of compaction and the extracted-fields record ([`crate::compaction`]).
//!
//! * `POST /v1/agents/{id}/compact`: replace the agent's history between its system prompt and its recent tail with a
//!   summary. Body (all optional): `summary` (text, tokenized without special tokens added, or an id array),
//!   `summary_ids`, `tail_tokens` (default 1024) or `tail_from` (a context position), `prefix_tokens` (default: the system
//!   prompt), `summary_tokens` (what the summarizer is asked for when no summary is given; the summarizer is the
//!   first-tokens STUB until the helper's summarize endpoint is merged). Answers when the summary and the tail are in
//!   the cache, with the [`crate::compaction::Report`] (`object: "agent.compaction"`). 409 while the agent runs a turn or
//!   compacts, 422 when there is nothing to replace, 501 for the contiguous cache.
//! * `GET /v1/agents/{id}/compaction`, `PUT` (a [`crate::compaction::Policy`]: `window_tokens`, `every_tokens`,
//!   `tail_tokens`, `summary_tokens`), `DELETE`: the policy that compacts the agent after a turn on its own.
//! * `GET /v1/agents/{id}/fields`, `PUT` (replace), `PATCH` (merge; `null` removes a key), `DELETE
//!   /v1/agents/{id}/fields/{key}`: the extracted-fields record. Bodies are a JSON object of scalars, or
//!   `{"fields": {...}}`.

use super::*;
use crate::compaction::{FieldsOp, Policy, Spec, Trigger};
use serde_json::Map;

pub(super) fn routes() -> Router<St> {
    Router::new()
        .route("/v1/agents/{id}/compact", post(compact))
        .route("/v1/agents/{id}/compaction", get(policy_get).put(policy_put).delete(policy_delete))
        .route("/v1/agents/{id}/fields", get(fields_get).put(fields_put).patch(fields_patch))
        .route("/v1/agents/{id}/fields/{key}", delete(field_delete))
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct CompactReq {
    summary: Option<Prompt>,
    summary_ids: Option<Vec<i32>>,
    tail_tokens: Option<usize>,
    tail_from: Option<usize>,
    prefix_tokens: Option<usize>,
    summary_tokens: Option<usize>,
}

async fn compact(State(st): State<St>, Path(id): Path<String>, body: Bytes) -> Result<Json<Value>, ApiError> {
    let r: CompactReq = parse_body(&body, true)?;
    if r.tail_tokens.is_some() && r.tail_from.is_some() {
        return Err(ApiError::invalid("give tail_tokens or tail_from, not both"));
    }
    let summary = match (r.summary, r.summary_ids) {
        (Some(_), Some(_)) => return Err(ApiError::invalid("give summary or summary_ids, not both")),
        (Some(Prompt::Ids(v)), None) | (None, Some(v)) => Some(v),
        // a summary goes inside the conversation: no BOS in front of it
        (Some(Prompt::Text(t)), None) => Some(encode(&st, t, false, true).await?),
        (None, None) => None,
    };
    let spec = Spec { summary, tail_tokens: r.tail_tokens, tail_from: r.tail_from, prefix_tokens: r.prefix_tokens, summary_tokens: r.summary_tokens, trigger: Trigger::Api, summarized_by: None, context: None, frame: None };
    let report = ask(&st, |reply| Command::AgentCompact { id, spec, reply }).await??;
    let mut v = serde_json::to_value(report).unwrap_or(Value::Null);
    v["object"] = json!("agent.compaction");
    Ok(Json(v))
}

/// Read (`None`), set (`Some(Some(p))`) or clear (`Some(None)`) an agent's compaction policy; the policy now in force.
async fn agent_policy(st: &St, id: String, policy: Option<Option<Policy>>) -> Result<Json<Value>, ApiError> {
    let info = ask(st, |reply| Command::AgentPolicy { id, policy, reply }).await??;
    Ok(Json(info.compaction))
}

async fn policy_get(State(st): State<St>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    agent_policy(&st, id, None).await
}

async fn policy_put(State(st): State<St>, Path(id): Path<String>, body: Bytes) -> Result<Json<Value>, ApiError> {
    let p: Policy = parse_body(&body, false)?;
    agent_policy(&st, id, Some(Some(p))).await
}

async fn policy_delete(State(st): State<St>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    agent_policy(&st, id, Some(None)).await
}

/// A fields body: the object itself, or `{"fields": {...}}`.
fn fields_body(body: &Bytes) -> Result<Map<String, Value>, ApiError> {
    let mut m: Map<String, Value> = parse_body(body, false)?;
    if m.len() == 1 {
        if let Some(Value::Object(inner)) = m.get("fields") {
            let inner = inner.clone();
            m = inner;
        }
    }
    Ok(m)
}

async fn fields(st: &St, id: String, op: FieldsOp) -> Result<Json<Value>, ApiError> {
    Ok(Json(ask(st, |reply| Command::AgentFields { id, op, reply }).await??))
}

async fn fields_get(State(st): State<St>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    fields(&st, id, FieldsOp::Get).await
}

async fn fields_put(State(st): State<St>, Path(id): Path<String>, body: Bytes) -> Result<Json<Value>, ApiError> {
    let m = fields_body(&body)?;
    fields(&st, id, FieldsOp::Replace(m)).await
}

async fn fields_patch(State(st): State<St>, Path(id): Path<String>, body: Bytes) -> Result<Json<Value>, ApiError> {
    let m = fields_body(&body)?;
    fields(&st, id, FieldsOp::Merge(m)).await
}

async fn field_delete(State(st): State<St>, Path((id, key)): Path<(String, String)>) -> Result<Json<Value>, ApiError> {
    fields(&st, id, FieldsOp::Remove(key)).await
}

/// `POST /v1/agents` with `compaction` or `fields`: checked before the agent is made.
pub(super) fn check_create(policy: Option<&Policy>, fields: Option<&Map<String, Value>>, system_len: usize) -> Result<(), ApiError> {
    if let Some(p) = policy {
        p.validate(system_len).map_err(ApiError::invalid)?;
    }
    if let Some(f) = fields {
        crate::compaction::Fields::default().apply(FieldsOp::Replace(f.clone())).map_err(ApiError::invalid)?;
    }
    Ok(())
}

/// ... and applied once it exists; `info` is updated.
pub(super) async fn apply_create(st: &St, info: &mut AgentInfo, policy: Option<Policy>, fields: Option<Map<String, Value>>) -> Result<(), ApiError> {
    if let Some(f) = fields {
        let id = info.id.clone();
        let v = ask(st, |reply| Command::AgentFields { id, op: FieldsOp::Replace(f), reply }).await??;
        info.fields = v["fields"].clone();
    }
    if let Some(p) = policy {
        let id = info.id.clone();
        *info = ask(st, |reply| Command::AgentPolicy { id, policy: Some(Some(p)), reply }).await??;
    }
    Ok(())
}
