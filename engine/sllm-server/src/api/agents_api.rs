//! The agent routes: `/v1/agents`, a turn, an outside event, cancel, suspend and wake.

use super::*;

#[derive(Deserialize, Default)]
struct AgentCreateReq {
    #[serde(default)]
    lane: Option<Lane>,
    #[serde(default, alias = "system_ids")]
    system_prompt_ids: Vec<i32>,
    /// Fork: the new agent starts with a copy of this idle agent's context (its cells shared, nothing read again).
    #[serde(default)]
    from_agent: Option<String>,
    /// The compaction policy (phase 3; default: the server's).
    #[serde(default)]
    compaction: Option<crate::compaction::Policy>,
    /// The first extracted fields (phase 3).
    #[serde(default)]
    fields: Option<serde_json::Map<String, Value>>,
}

pub(super) async fn agent_create(State(st): State<St>, body: Bytes) -> Result<Response, ApiError> {
    let r: AgentCreateReq = parse_body(&body, true)?;
    let lane = r.lane.unwrap_or(Lane::Interactive);
    compact_api::check_create(r.compaction.as_ref(), r.fields.as_ref(), r.system_prompt_ids.len())?;
    if r.from_agent.is_some() && !r.system_prompt_ids.is_empty() {
        return Err(ApiError::invalid("from_agent and system_prompt_ids: one or the other"));
    }
    let mut info = ask(&st, |reply| Command::AgentCreate { lane, system_ids: r.system_prompt_ids, from: r.from_agent, reply }).await??;
    compact_api::apply_create(&st, &mut info, r.compaction, r.fields).await?;
    Ok((StatusCode::CREATED, Json(info)).into_response())
}

pub(super) async fn agent_list(State(st): State<St>) -> Result<Json<Value>, ApiError> {
    let list = ask(&st, |reply| Command::AgentList { reply }).await?;
    Ok(Json(json!({"object": "list", "data": list})))
}

pub(super) async fn agent_get(State(st): State<St>, Path(id): Path<String>) -> Result<Json<AgentInfo>, ApiError> {
    Ok(Json(ask(&st, |reply| Command::AgentGet { id, reply }).await??))
}

pub(super) async fn agent_delete(State(st): State<St>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    let (info, freed) = ask(&st, |reply| Command::AgentDelete { id, reply }).await??;
    Ok(Json(json!({"id": info.id, "object": "agent", "deleted": true, "context_tokens_freed": freed})))
}

pub(super) async fn agent_cancel(State(st): State<St>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    let cancelled = ask(&st, |reply| Command::AgentCancel { id: id.clone(), reply }).await??;
    Ok(Json(json!({"id": id, "cancelled": cancelled})))
}

#[derive(Deserialize, Default)]
struct SuspendReq {
    #[serde(default)]
    tier: Option<crate::lifecycle::Tier>,
}

/// Suspend an agent (phase 3 lifecycle): `{"tier": "ssd"}` (default) moves its private KV to an SSD snapshot and
/// frees its pages; `{"tier": "history"}` drops its KV (the next turn re-prefills the token history). On a busy agent
/// it applies when the running turn ends (`"deferred": true`).
pub(super) async fn agent_suspend(State(st): State<St>, Path(id): Path<String>, body: Bytes) -> Result<Json<Value>, ApiError> {
    let r: SuspendReq = parse_body(&body, true)?;
    let tier = r.tier.unwrap_or(crate::lifecycle::Tier::Ssd);
    Ok(Json(ask(&st, |reply| Command::AgentSuspend { id, tier, reply }).await??))
}

/// Start restoring a suspended agent now (a tool result is about to arrive): its next turn finds the KV in RAM.
pub(super) async fn agent_wake(State(st): State<St>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    Ok(Json(ask(&st, |reply| Command::AgentWake { id, reply }).await??))
}

/// A turn's request must not name stop strings (see [`refuse_stop_on_agents`]).
fn refuse_stops(req: &GenReq) -> Result<(), ApiError> {
    if stops_of(&req.stop).is_empty() {
        Ok(())
    } else {
        Err(refuse_stop_on_agents())
    }
}

/// A turn: append `prompt` (ids) to the agent's context and generate `n_predict` tokens. Streams by default, in the same
/// chunk format as `/completion`; the final chunk adds `agent_id` and `context_tokens`.
pub(super) async fn agent_turn(State(st): State<St>, Path(id): Path<String>, body: Bytes) -> Result<Response, ApiError> {
    let mut req: GenReq = parse_gen_body(&body, true)?;
    refuse_stops(&req)?;
    if req.prompt.is_none() {
        req.prompt = Some(Prompt::Ids(vec![]));
    }
    run_gen(&st, req, Some(id), true).await
}


#[derive(Deserialize)]
struct EventReq {
    /// The tool the event comes from (`synthetic`: the made-up call's name).
    name: String,
    /// The event: a string goes in as it is, any other JSON as compact JSON.
    #[serde(default)]
    content: Value,
    /// The made-up call's arguments (`synthetic` only).
    #[serde(default)]
    arguments: Option<Value>,
    /// `synthetic` (default): the agent's last reply gets a tool call made up for the event, and the event is its result.
    /// `tool_result`: the last reply ended with the agent's own tool call; the event is its result.
    #[serde(default, rename = "as")]
    framing: Option<String>,
    #[serde(default)]
    call_id: Option<String>,
    /// Only return the framed text and its tokens; no turn runs.
    #[serde(default)]
    dry_run: bool,
    #[serde(flatten)]
    gen: GenReq,
}

/// An event's `as`: `synthetic` (the default, true) or `tool_result` (false).
fn synthetic_framing(framing: Option<&str>) -> Result<bool, ApiError> {
    match framing {
        None | Some("synthetic") => Ok(true),
        Some("tool_result") => Ok(false),
        Some(o) => Err(ApiError::invalid(format!("`as` must be synthetic or tool_result, not {o:?}"))),
    }
}

/// An event's `content` as text: a string as it is, null as nothing, any other JSON as compact JSON.
fn event_content(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        v => v.to_string(),
    }
}

/// An outside event for an agent: framed as the result of a tool call in the model's own chat format and run as the
/// agent's next turn (queued behind a running turn, waking a suspended agent), so the agent reads it as it reads any
/// tool result. The response is the turn's, with `X-Event-Framing: tool|user` (user: the template has no tool role).
pub(super) async fn agent_event(State(st): State<St>, Path(id): Path<String>, body: Bytes) -> Result<Response, ApiError> {
    let mut r = parse_gen_body::<EventReq>(&body, false)?;
    refuse_stops(&r.gen)?;
    let synthetic = synthetic_framing(r.framing.as_deref())?;
    if r.gen.prompt.is_some() {
        return Err(ApiError::invalid("an event takes `content`, not `prompt`"));
    }
    check_after_reply(&st, &id).await?;
    let (text, framing) = frame_event(&st, &mut r, synthetic)?;
    let ids = encode(&st, text.clone(), false, true).await?;
    if r.dry_run {
        return Ok(Json(json!({"agent_id": id, "framing": framing.name(), "text": text, "tokens": ids})).into_response());
    }
    r.gen.prompt = Some(Prompt::Ids(ids));
    let mut resp = run_gen(&st, r.gen, Some(id), true).await?;
    resp.headers_mut().insert("x-event-framing", HeaderValue::from_static(framing.name()));
    Ok(resp)
}

/// An event follows a reply: refused for an agent with no turn yet (none done, running or queued).
async fn check_after_reply(st: &St, id: &str) -> Result<(), ApiError> {
    let info = { let id = id.to_string(); ask(st, |reply| Command::AgentGet { id, reply }).await?? };
    if info.turns == 0 && !info.busy && info.queued_turns == 0 {
        return Err(ApiError::new(409, "conflict_error", format!("agent {id} has no turn yet: an event follows a reply")));
    }
    Ok(())
}

/// The event's text as the result of a tool call in the model's chat template, and how it was framed.
fn frame_event(st: &St, r: &mut EventReq, synthetic: bool) -> Result<(String, crate::chat::Framing), ApiError> {
    let content = event_content(&r.content);
    let args = r.arguments.take().unwrap_or_else(|| json!({}));
    let call_id = r.call_id.take().unwrap_or_else(|| format!("call_{}", st.next_req.load(Relaxed)));
    let tmpl = chat_template(st)?;
    let ev = crate::chat::Event { name: &r.name, arguments: &args, content: &content, synthetic, call_id: &call_id };
    tmpl.event_text(&ev).map_err(|e| ApiError::new(501, "not_supported_error", e))
}
