//! The fields llama-server puts on its replies that the routes share: `system_fingerprint`, `timings`, the usage detail, the
//! model name, and `logprobs`.

use super::*;

/// llama-server's `system_fingerprint` is its build; this server's is its version and commit.
pub(in crate::api) fn system_fingerprint() -> String {
    format!("sllm-{}-{}", crate::CRATE_VERSION, crate::GIT_COMMIT)
}

/// The model name of a reply whose request names none: `--alias`, else the loaded model's path (llama-server's `model_name`).
pub(in crate::api) fn default_model(st: &St) -> String {
    st.compat.alias.clone().unwrap_or_else(|| st.shared.info.get().map(|i| i.model.clone()).unwrap_or_default())
}

/// OpenAI `usage` with llama-server's `prompt_tokens_details.cached_tokens`.
pub(in crate::api) fn oai_usage(d: &Done) -> Value {
    let mut u = usage(d);
    u["prompt_tokens_details"] = json!({"cached_tokens": d.cached_n + d.shared_n});
    u
}

/// `system_fingerprint` on a reply object or chunk.
pub(in crate::api) fn stamp(mut v: Value) -> Value {
    v["system_fingerprint"] = json!(system_fingerprint());
    v
}

/// `timings` on a reply object or on the last chunk of a stream (llama-server's `timings`).
pub(in crate::api) fn with_timings(mut v: Value, d: &Done) -> Value {
    v["timings"] = timings_json(d);
    v
}

/// What a request asked of token probabilities: the entries of `top_logprobs` it wants beside the chosen token's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::api) struct LogprobsAsk {
    pub top: usize,
}

impl LogprobsAsk {
    /// The engine's `n_probs`: it gives the chosen token's probability from 1 on.
    pub fn n_probs(self) -> usize {
        self.top.max(1)
    }
}

/// The most `top_logprobs` the engine serves.
const MAX_TOP: u64 = 2;

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError::new(400, "invalid_request_error", msg)
}

/// The probabilities the engine can serve, or the 400 that says it cannot.
fn require_probs(st: &St, stream: bool) -> Result<(), ApiError> {
    if !chat_api::engine_has_probs(st) {
        return Err(bad("unsupported field `logprobs`: this engine does not serve token probabilities (only --engine next does)"));
    }
    if stream {
        return Err(bad("unsupported: `logprobs` with `stream`: the probabilities come with the whole reply (send stream false)"));
    }
    Ok(())
}

/// The value of a field when it asks for something (null, "", [], {}, false and 0 do not).
fn asked<'a>(v: Option<&'a Value>, field: &str) -> Option<&'a Value> {
    v.filter(|x| !is_not_asked_for(field, x))
}

fn top_of(v: Option<&Value>, what: &str) -> Result<u64, ApiError> {
    let n = match v {
        None | Some(Value::Null) => 0,
        Some(x) => x.as_u64().ok_or_else(|| bad(format!("`{what}` must be a non-negative integer")))?,
    };
    if n > MAX_TOP {
        return Err(bad(format!("unsupported `{what}` {n}: the engine serves the best {MAX_TOP} tokens (use {MAX_TOP} or less)")));
    }
    Ok(n)
}

/// `logprobs` / `top_logprobs` of a chat request (`logprobs` true, `top_logprobs` 0 to 2): `None` when not asked for. 400 when the
/// engine has no probabilities, when `stream` is on, or when `top_logprobs` is more than the engine serves.
pub(in crate::api) fn chat_logprobs(logprobs: Option<&Value>, top_logprobs: Option<&Value>, st: &St, stream: bool) -> Result<Option<LogprobsAsk>, ApiError> {
    let on = match asked(logprobs, "logprobs") {
        None => false,
        Some(Value::Bool(true)) => true,
        Some(_) => return Err(bad("`logprobs` must be a boolean on /v1/chat/completions")),
    };
    let top = top_of(asked(top_logprobs, "top_logprobs"), "top_logprobs")?;
    if !on {
        return if top > 0 { Err(bad("`top_logprobs` needs `logprobs: true`")) } else { Ok(None) };
    }
    require_probs(st, stream)?;
    Ok(Some(LogprobsAsk { top: top as usize }))
}

/// `logprobs: N` of a text completion (OpenAI's legacy integer; 0 is not asked for), N up to 2.
pub(in crate::api) fn text_logprobs(logprobs: Option<&Value>, st: &St, stream: bool) -> Result<Option<LogprobsAsk>, ApiError> {
    let Some(v) = asked(logprobs, "logprobs") else { return Ok(None) };
    let n = top_of(Some(v), "logprobs")?;
    require_probs(st, stream)?;
    Ok(Some(LogprobsAsk { top: n as usize }))
}

/// `chat_template_kwargs`: only `enable_thinking` (a boolean) is read, as the template's thinking switch; any other key is 400.
/// Null, `{}` and absent leave thinking off.
pub(in crate::api) fn enable_thinking(kwargs: &Option<Value>) -> Result<bool, ApiError> {
    let Some(o) = kwargs.as_ref().filter(|v| !is_not_asked_for("chat_template_kwargs", v)) else { return Ok(false) };
    let Some(o) = o.as_object() else { return Err(bad("`chat_template_kwargs` must be an object")) };
    if let Some(k) = o.keys().find(|k| *k != "enable_thinking") {
        return Err(bad(format!("unsupported `chat_template_kwargs` key `{k}`: only `enable_thinking` is read (see /caps chat)")));
    }
    o.get("enable_thinking").and_then(Value::as_bool).ok_or_else(|| bad("`chat_template_kwargs.enable_thinking` must be a boolean"))
}

fn tok_json(t: &TokProb) -> Value {
    json!({"id": t.id, "token": t.token, "bytes": t.bytes, "logprob": t.logprob})
}

/// One generated token's entry of `logprobs.content`: the token with its log-probability and the best tokens' (at most `top`).
pub(in crate::api) fn logprob_entry(p: &TokProbs, top: usize) -> Value {
    let mut v = tok_json(&p.chosen);
    v["top_logprobs"] = Value::Array(p.top.iter().take(top).map(tok_json).collect());
    v
}

/// `logprobs` of a reply: `{"content": [...]}`, one entry per generated token (llama-server's shape).
pub(in crate::api) fn logprobs_json(entries: Vec<Value>) -> Value {
    json!({"content": entries})
}
