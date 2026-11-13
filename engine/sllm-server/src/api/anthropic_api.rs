//! `POST /v1/messages`, `POST /v1/messages/count_tokens`, `GET /v1/models` and `GET /v1/models/{id}`: Anthropic's Messages
//! API over the executor, for Claude Code and the Anthropic SDKs.
//!
//! The request becomes chat-template input in [`crate::anthropic::convert`] (system as a string, blocks or an in-messages
//! entry; `tool_use` and `tool_result` blocks; the canonicalization rules that keep the rendered prompt byte-stable across
//! turns), is rendered by the same [`ChatTemplate::render_chat`] as `/v1/chat/completions`, tokenized by the same
//! [`prompt_ids`], and runs like any `/v1/completions` request. `count_tokens` calls the same [`prepare`], so its count is
//! the length of the prompt a message with that body prefills. The reply text goes through the chat route's
//! [`Pipe`]: stop strings, then (when the request has `thinking`) [`crate::thinking`], then the tool-call parser. Stream and
//! non-stream are one code path: the non-stream reply is the stream's events folded into a message, so they agree.
//!
//! Honoured: `model` (echoed), `max_tokens` (required), `messages`, `system`, `tools` (custom tools with an `input_schema`),
//! `tool_choice` `auto` and `none` (`any` and `tool` need grammar-constrained decoding: 400), `stop_sequences`, `stream`,
//! `thinking` (`enabled` and `adaptive` turn on `enable_thinking`; the budget is not enforced), `temperature` (validated,
//! decoding is greedy). `top_p`, `top_k`, `metadata`, `disable_parallel_tool_use` and other fields are accepted and ignored.
//! An image or any non-text block is 400.
//!
//! Wire: a `message` with `content` blocks (`text`, `thinking`, `tool_use`), `stop_reason` (`end_turn`, `max_tokens`,
//! `stop_sequence`, `tool_use`) and `usage` (`input_tokens`, `output_tokens`, `cache_read_input_tokens`,
//! `cache_creation_input_tokens`). `cache_read_input_tokens` is the prompt tokens served from the cache (`cached_n` +
//! `shared_n`); the server does not tell creation from reads, so creation is 0. `input_tokens` is the whole prompt, as
//! `count_tokens` counts it, except for a request that uses `cache_control` (Claude Code's does): there the cache reads are
//! taken out of it, as Anthropic's usage is disjoint and Claude Code adds the three. Streaming sends
//! `event: <type>\ndata: {...}` frames: `message_start`, per block `content_block_start`, one `ping` after the first block
//! starts, `content_block_delta`s (`text_delta`, `thinking_delta` then a `signature_delta` that is a fixed placeholder,
//! `input_json_delta` with the whole arguments in one chunk: a call is parsed whole, never partial), `content_block_stop`,
//! `message_delta` (stop reason, final usage), `message_stop`. A `ping` also goes out whenever no event has for `ping_ms`
//! (a long prefill). An empty reply is one empty text block.
//!
//! Errors: `{"type": "error", "error": {"type": ..., "message": ...}}` with Anthropic's statuses: 400 `invalid_request_error`
//! (also what the OpenAI routes call 422 and 501; a prompt that does not fit the context says `prompt is too long: N tokens
//! > M maximum`, the words Claude Code reads), 404 `not_found_error`, 429 `rate_limit_error`, 500 `api_error`, 503 as 529
//! `overloaded_error`. After a stream has started an error is an `error` event.
//!
//! Tool results too large for the context are cut by [`crate::anthropic::ToolResultLimit`] (`--tool-result-limit`).

use super::chat_api::{prompt_ids, tool_parser, Pipe, ReplyWriter, Streaming};
use super::*;
use crate::anthropic::{cut_tool_results, ToolResultLimit, CAPS};
use crate::chat::{ChatTemplate, RenderError};
use crate::thinking::{prompt_opens_thinking, ThinkParser};
use crate::toolcall::{Part, ToolCall, ToolFormat, ToolParser};
use std::time::Duration;

mod reply;
use reply::{collect, stream_reply, Blocks, MessageWriter, Meta};
#[cfg(test)]
use reply::{stop_reason, usage_json};

pub(super) fn routes() -> Router<St> {
    Router::new()
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .route("/v1/models", get(models))
        .route("/v1/models/{id}", get(model))
}

pub(super) const ENDPOINTS: [&str; 4] = ["/v1/messages", "/v1/messages/count_tokens", "/v1/models", "/v1/models/{id}"];

/// `/caps` `anthropic`: what the routes honour and refuse.
pub(super) fn caps(st: &St) -> Value {
    let format = chat_template_cached(st).ok().and_then(|t| t.tool_format()).map(ToolFormat::name);
    json!({
        "routes": ENDPOINTS,
        "honoured": ["model", "max_tokens", "messages", "system", "tools", "tool_choice", "stop_sequences", "stream", "thinking", "temperature",
                     "top_p", "top_k", "metadata"],
        "tool_choice": ["auto", "none"],
        "tool_choice_refused": "any and tool need grammar-constrained decoding: 400",
        "tool_call_format": format,
        "content_blocks": "text, tool_use, tool_result, thinking (earlier turns, dropped); an image or any other block is 400",
        "thinking": "enabled (or adaptive) turns on enable_thinking and splits <think>...</think> into thinking blocks; the budget is not enforced",
        "usage": "input_tokens, output_tokens, cache_read_input_tokens (cached + shared prefix), cache_creation_input_tokens (always 0)",
        "tool_result_limit": st.anthropic.tool_result_limit.name(),
        "ping_ms": st.anthropic.ping_ms,
    })
}

// -- errors ------------------------------------------------------------------------------------------------------

/// An [`ApiError`] on its way out in Anthropic's shape.
pub(super) struct AErr(ApiError);

impl From<ApiError> for AErr {
    fn from(e: ApiError) -> Self {
        AErr(e)
    }
}

impl From<EngineError> for AErr {
    fn from(e: EngineError) -> Self {
        AErr(e.into())
    }
}

fn bad(message: impl Into<String>) -> AErr {
    AErr(ApiError::new(400, "invalid_request_error", message))
}

/// (status, Anthropic's error type, message) for an engine error.
fn wire_error(e: &EngineError) -> (u16, &'static str, String) {
    match (e.status, e.kind) {
        (_, "exceed_context_size_error") => {
            (400, "invalid_request_error", e.context.map_or_else(|| e.message.clone(), |(n, ctx)| format!("prompt is too long: {n} tokens > {ctx} maximum")))
        }
        (404, _) => (404, "not_found_error", e.message.clone()),
        (429, _) => (429, "rate_limit_error", e.message.clone()),
        (503, _) => (529, "overloaded_error", e.message.clone()),
        (s, _) if s >= 500 && s != 501 => (500, "api_error", e.message.clone()),
        _ => (400, "invalid_request_error", e.message.clone()),
    }
}

fn error_body(e: &EngineError) -> (u16, Value) {
    let (status, kind, message) = wire_error(e);
    (status, json!({"type": "error", "error": {"type": kind, "message": message}}))
}

impl IntoResponse for AErr {
    fn into_response(self) -> Response {
        let (status, body) = error_body(&self.0 .0);
        let mut r = (StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), Json(body)).into_response();
        if let Some(ms) = self.0 .0.retry_after_ms {
            r.headers_mut().insert("retry-after", HeaderValue::from(ms.div_ceil(1000).max(1)));
        }
        r
    }
}

// -- the request -------------------------------------------------------------------------------------------------

#[derive(Deserialize)]
struct MsgReq {
    model: Option<String>,
    max_tokens: Option<i64>,
    messages: Option<Value>,
    system: Option<Value>,
    tools: Option<Value>,
    tool_choice: Option<Value>,
    stop_sequences: Option<Vec<String>>,
    temperature: Option<f32>,
    #[serde(default)]
    stream: bool,
    thinking: Option<Value>,
}

#[derive(PartialEq)]
enum ToolChoice {
    Auto,
    None,
}

fn tool_choice(v: &Option<Value>) -> Result<ToolChoice, AErr> {
    let kind = match v {
        None | Some(Value::Null) => return Ok(ToolChoice::Auto),
        Some(Value::String(s)) => s.as_str(),
        Some(o) => o.get("type").and_then(Value::as_str).ok_or_else(|| bad("tool_choice.type: Field required"))?,
    };
    match kind {
        "auto" => Ok(ToolChoice::Auto),
        "none" => Ok(ToolChoice::None),
        "any" | "tool" => Err(bad(format!(
            "unsupported `tool_choice` type {kind:?}: forcing a tool call needs grammar-constrained decoding, which the messages route does not have yet; use \"auto\" or \"none\""
        ))),
        k => Err(bad(format!("tool_choice.type: expected \"auto\", \"any\", \"tool\" or \"none\", got {k:?}"))),
    }
}

fn thinking_enabled(v: &Option<Value>) -> Result<bool, AErr> {
    match v {
        None | Some(Value::Null) => Ok(false),
        Some(o) => match o.get("type").and_then(Value::as_str) {
            Some("enabled") | Some("adaptive") => Ok(true),
            Some("disabled") => Ok(false),
            _ => Err(bad("thinking.type: expected \"enabled\", \"adaptive\" or \"disabled\"")),
        },
    }
}

/// A request ready to run: the prompt's token ids and the parsers for the reply.
struct Prepared {
    ids: Vec<i32>,
    tools: Option<ToolParser>,
    think: Option<ThinkParser>,
    cache_control: bool,
    /// Tool results cut to fit the context.
    cut: usize,
}

fn render_error(e: RenderError) -> AErr {
    bad(e.to_string())
}

/// The request's `messages`: a list of at least one message.
fn request_messages(r: &MsgReq) -> Result<&Vec<Value>, AErr> {
    match &r.messages {
        None | Some(Value::Null) => Err(bad("messages: Field required")),
        Some(Value::Array(a)) if a.is_empty() => Err(bad("messages: at least one message is required")),
        Some(Value::Array(a)) => Ok(a),
        Some(_) => Err(bad("messages: Input should be a valid list")),
    }
}

/// `--tool-result-limit auto` for a prompt the engine would refuse: the largest cap that leaves room for a reply, else the
/// smallest that fits at all. The (prompt, ids, results cut) of the best attempt, None when no cap cuts anything.
async fn auto_cut<R>(st: &St, tmpl: &ChatTemplate, render: &R, messages: &[Value], n_ctx: usize) -> Result<Option<(String, Vec<i32>, usize)>, AErr>
where
    R: Fn(&[Value]) -> Result<String, AErr> + Sync,
{
    let reserve = (n_ctx / 16).clamp(256, n_ctx / 2);
    let mut fallback = None;
    for cap in CAPS {
        let (m, n) = cut_tool_results(messages, cap);
        if n == 0 {
            continue;
        }
        let t = render(&m)?;
        let i = prompt_ids(st, tmpl, t.clone()).await?;
        let fits = i.len() + reserve <= n_ctx;
        if fits || i.len() < n_ctx {
            fallback = Some((t, i, n));
        }
        if fits {
            break;
        }
    }
    Ok(fallback)
}

/// Debugging aid (step 8), off unless SLLM_DUMP_PROMPTS is set: the rendered prompt and its ids of every request, to find
/// where two prompts first differ.
fn dump_prompt(text: &str, ids: &[i32]) {
    if let Some(dir) = std::env::var_os("SLLM_DUMP_PROMPTS") {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let dir = std::path::PathBuf::from(dir);
        let _ = std::fs::write(dir.join(format!("{n:03}.txt")), text);
        let _ = std::fs::write(dir.join(format!("{n:03}.ids")), ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("\n"));
    }
}

/// A validated request as chat-template input: the template, the converted messages and tools, and the tool-call parser.
struct ChatInput {
    tmpl: Arc<ChatTemplate>,
    messages: Vec<Value>,
    tools: Vec<Value>,
    tool_parser: Option<ToolParser>,
    think_on: bool,
    cache_control: bool,
}

fn chat_input(st: &St, r: &MsgReq) -> Result<ChatInput, AErr> {
    let messages = request_messages(r)?;
    let choice = tool_choice(&r.tool_choice)?;
    let think_on = thinking_enabled(&r.thinking)?;
    let conv = crate::anthropic::convert(r.system.as_ref(), messages, r.tools.as_ref()).map_err(bad)?;
    let tmpl: Arc<ChatTemplate> = chat_template_cached(st).map_err(|e| ApiError::new(501, "not_supported_error", format!("the messages route needs the model's chat template: {e}")))?;
    // tool_choice none: the model is not told about the tools and its reply is not searched for calls
    let tools = if choice == ToolChoice::None { vec![] } else { conv.tools };
    let tool_parser = tool_parser(&tmpl, &tools, r#"{"type": "none"}"#)?;
    Ok(ChatInput { tmpl, messages: conv.messages, tools, tool_parser, think_on, cache_control: conv.cache_control })
}

/// `--tool-result-limit`: the prompt (`text`, `ids`) again with its tool results cut, when the limit says so; and how many
/// results were cut.
async fn fit_tool_results<R>(st: &St, tmpl: &ChatTemplate, render: &R, messages: &[Value], text: String, ids: Vec<i32>) -> Result<(String, Vec<i32>, usize), AErr>
where
    R: Fn(&[Value]) -> Result<String, AErr> + Sync,
{
    let n_ctx = st.shared.info.get().map_or(usize::MAX, |i| i.n_ctx);
    match st.anthropic.tool_result_limit {
        ToolResultLimit::Off => Ok((text, ids, 0)),
        ToolResultLimit::Chars(cap) => {
            let (m, n) = cut_tool_results(messages, cap);
            if n == 0 {
                return Ok((text, ids, 0));
            }
            let text = render(&m)?;
            let ids = prompt_ids(st, tmpl, text.clone()).await?;
            Ok((text, ids, n))
        }
        // only a prompt the engine would refuse is touched
        ToolResultLimit::Auto if ids.len() >= n_ctx => Ok(auto_cut(st, tmpl, render, messages, n_ctx).await?.unwrap_or((text, ids, 0))),
        ToolResultLimit::Auto => Ok((text, ids, 0)),
    }
}

/// Everything `/v1/messages` and `/v1/messages/count_tokens` share: validation, the prompt (rendered, tokenized, tool results
/// cut when the prompt would not fit) and the reply parsers.
async fn prepare(st: &St, r: &MsgReq) -> Result<Prepared, AErr> {
    let ChatInput { tmpl, messages, tools, tool_parser, think_on, cache_control } = chat_input(st, r)?;
    let render = |msgs: &[Value]| tmpl.render_chat(msgs, &tools, true, think_on).map_err(render_error);
    let text = render(&messages)?;
    let ids = prompt_ids(st, &tmpl, text.clone()).await?;
    let (text, ids, cut) = fit_tool_results(st, &tmpl, &render, &messages, text, ids).await?;
    dump_prompt(&text, &ids);
    let opens = prompt_opens_thinking(&text);
    let think = (think_on || opens).then(|| ThinkParser::new(opens));
    Ok(Prepared { ids, tools: tool_parser, think, cache_control, cut })
}

async fn count_tokens(State(st): State<St>, body: Bytes) -> Result<Json<Value>, AErr> {
    let r = parse_body::<MsgReq>(&body, false)?;
    let p = prepare(&st, &r).await?;
    Ok(Json(json!({"input_tokens": p.ids.len()})))
}

/// The request's `max_tokens`: required, at least 1.
fn max_tokens(r: &MsgReq) -> Result<i64, AErr> {
    match r.max_tokens {
        None => Err(bad("max_tokens: Field required")),
        Some(n) if n < 1 => Err(bad("max_tokens: Input should be greater than or equal to 1")),
        Some(n) => Ok(n),
    }
}

async fn messages(State(st): State<St>, body: Bytes) -> Result<Response, AErr> {
    let r: MsgReq = parse_body(&body, false)?;
    let max_tokens = max_tokens(&r)?;
    let params = GenReq { n_predict: Some(max_tokens), temperature: r.temperature, ..Default::default() }.params()?;
    let p = prepare(&st, &r).await?;
    let model = r.model.clone().unwrap_or_else(|| model_id(&st));
    let input_tokens = p.ids.len();
    let pump = submit(&st, Gen { agent: None, ids: p.ids, params, lane: None, deadline_ms: None }).await?;
    let req_id = pump.guard.id;
    let m = Meta { id: format!("msg_{req_id}"), req_id, model, input_tokens, cache_control: p.cache_control };
    let ping = (st.anthropic.ping_ms > 0 && r.stream).then(|| Duration::from_millis(st.anthropic.ping_ms));
    let run = Streaming {
        pump,
        pipe: Pipe::new(r.stop_sequences.clone().unwrap_or_default(), p.think, p.tools).with_vocab(st.tokenizer.vocab()),
        n: 0,
        w: MessageWriter { m, blocks: Blocks::default(), started: false, ping, error: None },
    };
    let mut resp = if r.stream { stream_reply(run).into_response() } else { collect(run).await? };
    if p.cut > 0 {
        resp.headers_mut().insert("x-sllm-tool-results-cut", HeaderValue::from(p.cut));
    }
    Ok(with_request_id(resp, req_id))
}

// -- models ------------------------------------------------------------------------------------------------------

/// The loaded model's id: `--alias`, else the GGUF's file name without `.gguf`.
fn model_id(st: &St) -> String {
    llama_compat::model_name(st)
}

/// `secs` since the epoch as RFC 3339 UTC.
fn rfc3339(secs: u64) -> String {
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    let (y, m, d) = crate::chat::civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

fn model_json(st: &St) -> Value {
    let id = model_id(st);
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    llama_compat::model_object(st, json!({"type": "model", "id": id, "display_name": id, "created_at": rfc3339(now.saturating_sub(st.started.elapsed().as_secs()))}))
}

fn loaded(st: &St) -> Result<(), AErr> {
    if st.shared.info.get().is_none() {
        return Err(ApiError::new(503, "unavailable_error", "Loading model").into());
    }
    Ok(())
}

/// `GET /v1/models` and `GET /models`: Anthropic's list (`data`, `has_more`, `first_id`, `last_id`) and llama-server's in one
/// object (`object`, `models`, and in each entry `object`, `created`, `owned_by`, `meta`, ...).
pub(super) async fn models(State(st): State<St>) -> Result<Json<Value>, AErr> {
    loaded(&st)?;
    let m = model_json(&st);
    Ok(Json(json!({"object": "list", "data": [m], "models": llama_compat::models_array(&st), "has_more": false, "first_id": m["id"], "last_id": m["id"]})))
}

async fn model(State(st): State<St>, Path(id): Path<String>) -> Result<Json<Value>, AErr> {
    loaded(&st)?;
    let m = model_json(&st);
    let full = st.shared.info.get().map(|i| i.model.clone()).unwrap_or_default();
    if id != m["id"].as_str().unwrap_or_default() && id != full {
        return Err(ApiError::new(404, "not_found_error", format!("model: {id}")).into());
    }
    Ok(Json(m))
}

#[cfg(test)]
mod tests;
