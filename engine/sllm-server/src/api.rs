//! The HTTP API (axum). Everything here is a thin layer over the executor's command channel.
//!
//! * `POST /completion`: llama-server compatible (token-id prompts, SSE chunks, `timings`). The shape was captured from
//!   llama-server b11669 (`tests/fixtures/`). A field that would change what the output means but is not implemented
//!   (`json_schema`, `lora`, `tools`, `response_format`, ...: [`REFUSED_FIELDS`]) is refused with 400 on
//!   every generation route; fields that cannot change a greedy output ([`HARMLESS_FIELDS`]) are accepted and ignored.
//! * `POST /v1/completions`: OpenAI compatible, text prompt through the GGUF tokenizer.
//! * `POST /v1/chat/completions`: OpenAI chat, messages through the GGUF's chat template, `tools` parsed out of the reply
//!   (`api/chat_api.rs`). `POST /v1/messages`, `/v1/messages/count_tokens` and `GET /v1/models[/{id}]`: Anthropic's Messages
//!   API for Claude Code (`api/anthropic_api.rs`, errors in Anthropic's shape). `stop` (a string or a list) works on all three text routes, not on agent turns (`stop.rs`).
//! * `POST /v1/agents`, `GET /v1/agents[/{id}]`, `POST /v1/agents/{id}/turn`, `POST /v1/agents/{id}/cancel` (also
//!   `DELETE /v1/agents/{id}/request`), `DELETE /v1/agents/{id}`.
//! * `POST /v1/agents/{id}/event`: an outside event becomes the agent's next turn, framed by the model's chat template as
//!   the result of a tool call ([`crate::chat`]); it queues and wakes like any turn.
//! * `DELETE /v1/requests/{id}`: cancel a queued or running request by the id in its `X-Request-Id` response header
//!   (phase 2 task 8). A client that disconnects is cancelled the same way, at the next step boundary.
//!
//! Phase 2 scheduling fields on `/completion`, `/v1/completions` and turns: `lane` (`interactive`, `background` or
//! `helper`; default: the agent's lane, background for a plain completion) and `deadline_ms` (stop the request this long
//! after it arrives, `stop_type` "deadline"). A lane whose slots and queue are full answers 503 with
//! `{"error": {"type": "deferred", "retry_after_ms": N, ...}, "deferred": {lane, running, queued, ...}}` and a
//! `Retry-After` header.
//! * `POST /tokenize` (llama-server compatible; the playground adapter needs it), `GET /health`, `/caps`, `/metrics`; and the rest of
//!   llama-server's surface in [`llama_compat`] (`/props`, `/slots`, `/detokenize`, `/apply-template`, path aliases, `--api-key`).
//!
//! Errors are llama-server shaped: `{"error": {"code": N, "message": "...", "type": "..."}}`. Malformed JSON and a body that does not fit the
//! request's fields (a missing or mistyped field) are 400, a request whose values are not valid is 422, an unknown route or id is 404, a busy agent is 409, a full command queue
//! or a model still loading is 503.

use crate::engine::*;
use crate::executor::*;
use crate::stop::{stops_of, StopFilter, StopSpec};
use crate::tokenizer::Tokenizer;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use futures_util::stream::{self, Stream, StreamExt};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, oneshot};

pub struct ApiState {
    pub cmd: mpsc::Sender<Command>,
    pub queue_cap: usize,
    pub shared: Arc<Shared>,
    pub next_req: AtomicU64,
    pub tokenizer: Tokenizer,
    pub started: Instant,
    pub http_threads: usize,
    /// The thread plan reported in `/caps`.
    pub threads: Value,
    pub version: Value,
    /// The helper lane (phase 3, `--helper-model`): its own model, thread and page pool; `None` without one.
    pub helper: Option<Arc<crate::helper::Helper>>,
    /// The model's chat template, read from the GGUF at the first event (`POST /v1/agents/{id}/event`).
    pub chat: std::sync::OnceLock<Result<Arc<crate::chat::ChatTemplate>, String>>,
    /// The Anthropic routes (`--tool-result-limit`).
    pub anthropic: crate::anthropic::AnthropicConfig,
    /// The llama-server compatibility layer: `--alias`, `--api-key`, `--reasoning-format` ([`llama_compat`]).
    pub compat: llama_compat::CompatConfig,
    /// The request slots the scheduler runs at once, as `/props` `total_slots` and the entries of `/slots`.
    pub slots: usize,
}

pub type St = Arc<ApiState>;

pub fn router(state: St) -> Router {
    Router::new()
        .route("/completion", post(completion))
        .route("/v1/completions", post(oai_completions))
        .route("/v1/agents", post(agent_create).get(agent_list))
        .route("/v1/agents/{id}", get(agent_get).delete(agent_delete))
        .route("/v1/agents/{id}/turn", post(agent_turn))
        .route("/v1/agents/{id}/cancel", post(agent_cancel))
        .route("/v1/agents/{id}/suspend", post(agent_suspend))
        .route("/v1/agents/{id}/wake", post(agent_wake))
        .route("/v1/agents/{id}/event", post(agent_event))
        .route("/v1/agents/{id}/request", delete(agent_cancel))
        .route("/v1/requests/{id}", delete(request_cancel))
        .merge(compact_api::routes())
        .merge(chat_api::routes())
        .merge(anthropic_api::routes())
        .merge(llama_compat::routes())
        .route("/health", get(health))
        .route("/caps", get(caps))
        .route("/metrics", get(metrics))
        .merge(crate::helper::api::routes())
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(DefaultBodyLimit::max(64 << 20))
        .layer(axum::middleware::from_fn_with_state(state.clone(), llama_compat::auth::require_key))
        .with_state(state)
}

/// Compaction and the extracted-fields record (phase 3): `POST /v1/agents/{id}/compact`, `/v1/agents/{id}/compaction`
/// (the policy) and `/v1/agents/{id}/fields`.
mod compact_api;

/// `POST /v1/chat/completions`: the chat template, tool calls, OpenAI chunks.
mod chat_api;

/// `POST /v1/messages`, `/v1/messages/count_tokens`, `GET /v1/models`: Anthropic's Messages API over the same template, parsers and executor.
mod anthropic_api;

/// `POST /completion` (llama-server wire) and `DELETE /v1/requests/{id}`.
mod completion_api;

/// `POST /v1/completions` (OpenAI wire).
mod oai_api;

/// llama-server compatibility: path aliases, `/props`, `/slots`, `/detokenize`, `/apply-template`, the 501s, `--api-key`.
pub mod llama_compat;

/// The agent routes.
mod agents_api;

use agents_api::{agent_cancel, agent_create, agent_delete, agent_event, agent_get, agent_list, agent_suspend, agent_turn, agent_wake};
pub use completion_api::timings_json;
use completion_api::*;
use oai_api::*;

// -- errors ------------------------------------------------------------------------------------------------------

#[derive(Debug)]
pub struct ApiError(pub EngineError);

impl ApiError {
    pub fn new(status: u16, kind: &'static str, message: impl Into<String>) -> Self {
        ApiError(EngineError::new(status, kind, message))
    }
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(422, "invalid_request_error", message)
    }
}

impl From<EngineError> for ApiError {
    fn from(e: EngineError) -> Self {
        ApiError(e)
    }
}

pub fn error_json(e: &EngineError) -> Value {
    let mut v = json!({"error": {"code": e.status, "message": e.message, "type": e.kind}});
    if let Some(b) = e.shortfall_bytes {
        v["error"]["shortfall_bytes"] = json!(b);
    }
    if let Some(ms) = e.retry_after_ms {
        v["error"]["retry_after_ms"] = json!(ms);
    }
    if let Some((n_prompt_tokens, n_ctx)) = e.context {
        v["error"]["n_prompt_tokens"] = json!(n_prompt_tokens);
        v["error"]["n_ctx"] = json!(n_ctx);
    }
    if let Some(d) = &e.deferred {
        v["deferred"] = d.clone();
    }
    v
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.0.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let mut r = (status, Json(error_json(&self.0))).into_response();
        if let Some(ms) = self.0.retry_after_ms {
            r.headers_mut().insert("retry-after", HeaderValue::from(ms.div_ceil(1000).max(1)));
        }
        r
    }
}

fn queue_full() -> ApiError {
    ApiError(EngineError::new(503, "unavailable_error", "the executor queue is full").with_retry_after(100))
}

async fn not_found() -> ApiError {
    ApiError::new(404, "not_found_error", "File Not Found")
}

async fn method_not_allowed() -> ApiError {
    ApiError::new(405, "invalid_request_error", "Method Not Allowed")
}

/// JSON body parsing: unparsable JSON and JSON of the wrong shape are 400. An empty body is `{}`
/// for requests whose fields are all optional.
fn parse_body<T: DeserializeOwned>(body: &Bytes, empty_ok: bool) -> Result<T, ApiError> {
    parse_body_checked(body, empty_ok, |_| Ok(()))
}

/// [`parse_body`] for the generation routes: the request is also checked by [`reject_unsupported`] before it is typed,
/// because the typed structs drop the fields they do not know.
fn parse_gen_body<T: DeserializeOwned>(body: &Bytes, empty_ok: bool) -> Result<T, ApiError> {
    parse_body_checked(body, empty_ok, reject_unsupported)
}

fn parse_body_checked<T: DeserializeOwned>(body: &Bytes, empty_ok: bool, check: impl Fn(&Value) -> Result<(), ApiError>) -> Result<T, ApiError> {
    if body.iter().all(|b| b.is_ascii_whitespace()) {
        if empty_ok {
            return serde_json::from_str("{}").map_err(|e| ApiError::invalid(e.to_string()));
        }
        return Err(ApiError::new(400, "invalid_request_error", "empty request body"));
    }
    // Two steps so that the message is right: text that is not JSON is "invalid JSON", JSON of the wrong shape names the field.
    // (A single typed parse reports "[1,2" as a type error, not as truncated input.)
    let v: Value = serde_json::from_slice(body).map_err(|e| ApiError::new(400, "invalid_request_error", format!("invalid JSON: {e}")))?;
    check(&v)?;
    // a body that does not fit the request's type is 400, as in llama-server
    serde_json::from_value(v).map_err(|e| ApiError::new(400, "invalid_request_error", e.to_string()))
}

/// Request fields that change what the output means (a constraint, an adapter, tool calls, stop strings, a logit or
/// penalty change, extra output) and that sllm-server does not implement: a non-empty value is refused with 400 on
/// `/completion`, `/v1/completions`, `/v1/agents/{id}/turn` and `/v1/agents/{id}/event`. A route that does implement one of them
/// lets it through (`reject_fields`'s `except`): `json_schema` on the two text routes and on chat (it becomes the grammar),
/// `response_format` and `logprobs` / `top_logprobs` on chat, `logprobs` on `/v1/completions`, `n_probs` on `/completion`. Null, `""`, `[]`, `{}`, `false`
/// and `0` (and `1.0` for `repeat_penalty`) are "not asked for" and stay accepted. Listed in `/caps` under `completion.refused`.
pub const REFUSED_FIELDS: &[&str] = &[
    "json_schema", "lora", "tools", "tool_choice", "response_format", "functions", "function_call",
    "logit_bias", "logprobs", "top_logprobs", "n_probs", "echo", "suffix", "best_of", "mirostat", "repeat_penalty",
    "presence_penalty", "frequency_penalty", "dry_multiplier", "xtc_probability", "images", "image_data", "chat_template_kwargs",
];

/// Fields that are accepted and cannot change the output: the engine is greedy (`temperature` and the truncation
/// samplers do not alter an argmax), or they are bookkeeping and wire options of llama-server and OpenAI clients.
/// Fields in neither list are not known to the server and are ignored. Listed in `/caps` under `completion.ignored_harmless`.
pub const HARMLESS_FIELDS: &[&str] = &[
    "top_k", "top_p", "min_p", "typical_p", "top_n_sigma", "seed", "samplers", "dynatemp_range", "dynatemp_exponent",
    "repeat_last_n", "dry_base", "dry_allowed_length", "dry_penalty_last_n", "dry_sequence_breakers", "xtc_threshold",
    "min_keep", "n_keep", "id_slot", "slot_id", "timings_per_token", "return_progress", "post_sampling_probs", "model", "user",
];

fn is_not_asked_for(field: &str, v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Bool(b) => !b,
        Value::Number(n) => n.as_f64() == Some(if field == "repeat_penalty" { 1.0 } else { 0.0 }),
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
    }
}

fn reject_unsupported(v: &Value) -> Result<(), ApiError> {
    reject_fields(v, &[])
}

/// [`reject_unsupported`] except for the fields in `except` (the chat route takes `tools` and `tool_choice`).
/// `n_probs`: an integer; the empty values every refused field may carry (null, "", [], {}, false) mean it was not asked for.
fn n_probs_or_empty<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<i64>, D::Error> {
    let v = <Value as serde::Deserialize>::deserialize(d)?;
    match &v {
        Value::Null | Value::Bool(false) => Ok(None),
        Value::String(x) if x.is_empty() => Ok(None),
        Value::Array(x) if x.is_empty() => Ok(None),
        Value::Object(x) if x.is_empty() => Ok(None),
        Value::Number(n) => n.as_i64().map(Some).ok_or_else(|| serde::de::Error::custom("n_probs must be an integer")),
        _ => Err(serde::de::Error::custom("n_probs must be an integer")),
    }
}

fn reject_fields(v: &Value, except: &[&str]) -> Result<(), ApiError> {
    let Some(o) = v.as_object() else { return Ok(()) };
    for f in REFUSED_FIELDS.iter().filter(|f| !except.contains(f)) {
        if o.get(*f).is_some_and(|x| !is_not_asked_for(f, x)) {
            return Err(ApiError::new(
                400,
                "invalid_request_error",
                format!("unsupported field `{f}`: sllm-server does not implement it and would silently ignore it (see /caps completion.refused)"),
            ));
        }
    }
    Ok(())
}

// -- plain endpoints ---------------------------------------------------------------------------------------------

async fn health(State(st): State<St>) -> Response {
    // the helper lane, when configured, is part of being ready
    if let Some(h) = &st.helper {
        match h.state() {
            Err(msg) => return ApiError::new(500, "server_error", format!("the helper failed to start: {msg}")).into_response(),
            Ok(false) => return ApiError::new(503, "unavailable_error", "Loading helper model").into_response(),
            Ok(true) => {}
        }
    }
    if st.shared.ready.load(Relaxed) {
        return Json(json!({"status": "ok"})).into_response();
    }
    if let Some(msg) = st.shared.init_error.lock().unwrap().clone() {
        return ApiError::new(500, "server_error", format!("engine failed to start: {msg}")).into_response();
    }
    ApiError::new(503, "unavailable_error", "Loading model").into_response()
}

async fn caps(State(st): State<St>) -> Response {
    let Some(caps) = st.shared.caps.get() else {
        return ApiError::new(503, "unavailable_error", "Loading model").into_response();
    };
    let info = st.shared.info.get().cloned();
    let mut honoured = vec!["prompt", "n_predict", "temperature", "ignore_eos", "stream", "return_tokens", "lane", "deadline_ms", "cache_prompt", "grammar", "stop", "json_schema"];
    if is_next_engine(&st) {
        honoured.push("n_probs");
    }
    Json(json!({
        "server": st.version,
        "engine": caps,
        // which implementation runs the model: "native" (sllm-core: qwen3, gemma3), "llama" (libllama: any GGUF) or "fake"
        "engine_kind": caps.get("kind").cloned().unwrap_or_else(|| caps.get("engine").cloned().unwrap_or(Value::Null)),
        "arch": caps.get("arch"),
        "model": info,
        "tokenizer": st.tokenizer.name(),
        "threads": st.threads,
        "endpoints": ["/completion", "/v1/completions", "/v1/agents", "/v1/agents/{id}", "/v1/agents/{id}/turn",
                      "/v1/agents/{id}/cancel", "/v1/agents/{id}/request", "/v1/requests/{id}", "/tokenize", "/health",
                      "/caps", "/metrics", "/v1/helper", "/v1/helper/classify", "/v1/helper/extract", "/v1/helper/compress",
                      "/v1/helper/summarize", "/v1/agents/{id}/compact", "/v1/agents/{id}/compaction",
                      "/v1/agents/{id}/fields", "/v1/agents/{id}/fields/{key}", "/v1/agents/{id}/event", "/v1/chat/completions",
                      "/v1/messages", "/v1/messages/count_tokens", "/v1/models", "/v1/models/{id}",
                      "/v1/health", "/completions", "/chat/completions", "/models", "/props", "/slots", "/lora-adapters", "/detokenize",
                      "/apply-template"],
        // llama-server's routes that answer 501 not_supported_error (this server generates text only)
        "not_supported": ["/embedding", "/embeddings", "/v1/embeddings", "/rerank", "/reranking", "/v1/rerank", "/v1/reranking", "POST /props"],
        "chat": chat_api::caps(&st),
        "anthropic": anthropic_api::caps(&st),
        // the helper lane (phase 3): null without --helper-model
        "helper": st.helper.as_ref().map(|h| json!({"config": h.cfg.to_json(), "info": h.shared.info.get(),
            "brain_lane_info": h.brain_shared.as_ref().and_then(|b| b.info.get())})),
        // compaction (phase 3): the default policy and the summarizer (`helper-summarize` with --helper-model, else `stub-first-n`)
        "compaction": st.shared.compaction.get(),
        "scheduler": st.shared.sched_config.get(),
        // `--prefix-sharing on|off` (phase 2), as the engine reports it (null for an engine without it)
        "prefix_sharing": caps.get("prefix_sharing"),
        // HOOK(profile): the generic model profile, the memory budget derived from it and the startup calibration (null without them)
        "profile": caps.get("profile"),
        "budget": caps.get("budget"),
        "calibration": caps.get("calibration"),
        "completion": {
            "prompt": "array of token ids (a string is tokenized with the GGUF tokenizer)",
            "honoured": honoured,  // temperature: accepted and validated, but decoding is greedy
            "request_id": "the X-Request-Id response header; DELETE /v1/requests/{id} cancels it",
            "cache_prompt": "false keeps the request off the prompt cache on disk; sharing with live sequences is unchanged",
            "n_probs": "the next engine only, on /completion only (listed under refused elsewhere): completion_probabilities, one entry per generated token {id, token, bytes, logprob, top_logprobs}; logprob is the log-softmax of the raw scores (temperature 1); top_logprobs has min(n_probs, 2) entries, the best tokens of the whole vocabulary; values above 2 are served as 2",
            "stop": "a string or a list, on /completion and /v1/completions (not on agent turns): matched on the streamed text across token boundaries; the stop string is not output; stop_type \"word\" with stopping_word, finish_reason \"stop\"",
            "json_schema": "/completion and /v1/completions (not agent turns): the schema becomes the grammar (a subset: pattern, allOf and integer bounds are 400; both a grammar and a schema is 400); the chat route also takes response_format",
            "logprobs": "/v1/completions logprobs 1 or 2 (not with stream) and /v1/chat/completions logprobs with top_logprobs 0 to 2: on an engine with token probabilities (--engine next); /completion has n_probs",
            "sampling": "greedy: temperature and the other sampler fields do not change the output in phase 1",
            // fields that would change the output and are not implemented: HTTP 400 on /completion, /v1/completions,
            // /v1/agents/{id}/turn and /v1/agents/{id}/event when non-empty (null, "", [], {}, false, 0 are accepted)
            "refused": REFUSED_FIELDS.iter().filter(|f| **f != "json_schema").collect::<Vec<_>>(),
            "refused_on_agent_turns_and_events": ["json_schema"],
            // accepted and without effect on the output (greedy engine, bookkeeping); fields in neither list are ignored
            "ignored_harmless": HARMLESS_FIELDS,
        },
    }))
    .into_response()
}

async fn metrics(State(st): State<St>) -> Json<Value> {
    let depth = st.cmd.max_capacity() - st.cmd.capacity();
    let mut v = st.shared.metrics_json(depth, st.cmd.max_capacity(), st.http_threads, st.started.elapsed().as_secs_f64());
    v["helper"] = st.helper.as_ref().map_or(Value::Null, |h| h.metrics());
    Json(v)
}

async fn encode(st: &St, text: String, add_special: bool, parse_special: bool) -> Result<Vec<i32>, ApiError> {
    let st = st.clone();
    tokio::task::spawn_blocking(move || st.tokenizer.encode(&text, add_special, parse_special))
        .await
        .map_err(|e| ApiError::new(500, "server_error", format!("tokenizer task failed: {e}")))?
        .map_err(|e| ApiError::new(500, "server_error", format!("tokenizer: {e}")))
}

// -- submitting a request to the executor ------------------------------------------------------------------------

/// Cancels the request when dropped while armed: the client went away (stream dropped, handler future dropped).
struct CancelGuard {
    id: ReqId,
    cmd: mpsc::Sender<Command>,
    armed: bool,
}

impl CancelGuard {
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.cmd.try_send(Command::Cancel(self.id));
        }
    }
}

struct Gen {
    agent: Option<AgentId>,
    ids: Vec<i32>,
    params: SamplingParams,
    lane: Option<Lane>,
    deadline_ms: Option<u64>,
}

enum Item {
    Token { id: i32, text: String, probs: Option<Box<TokProbs>> },
    Done(Done),
    Error(EngineError),
}

/// One admitted request's event stream.
struct Pump {
    rx: mpsc::Receiver<ReqEvent>,
    guard: CancelGuard,
    prompt_n: usize,
    finished: bool,
}

impl Pump {
    async fn next(&mut self) -> Option<Item> {
        if self.finished {
            return None;
        }
        loop {
            match self.rx.recv().await {
                Some(ReqEvent::Started { .. }) => continue,
                Some(ReqEvent::Token { id, text, probs }) => return Some(Item::Token { id, text, probs }),
                Some(ReqEvent::Done(d)) => {
                    self.finished = true;
                    self.guard.disarm();
                    return Some(Item::Done(d));
                }
                Some(ReqEvent::Error(e)) => {
                    self.finished = true;
                    self.guard.disarm();
                    return Some(Item::Error(e));
                }
                None => {
                    self.finished = true;
                    return Some(Item::Error(EngineError::new(500, "server_error", "the executor dropped the request")));
                }
            }
        }
    }
}

impl Pump {
    /// Stop the request now (a stop string matched) and wait for its `Done`; tokens already on the way are dropped.
    /// `None`: the executor failed the request or dropped it.
    async fn cancel_and_drain(&mut self) -> Option<Done> {
        if !self.finished {
            let _ = self.guard.cmd.send(Command::Cancel(self.guard.id)).await;
        }
        loop {
            match self.next().await? {
                Item::Done(d) => return Some(d),
                Item::Error(_) => return None,
                Item::Token { .. } => {}
            }
        }
    }

    /// `cancel_and_drain` for a stop string matched after `n` tokens: the final report, counting those tokens; the error when
    /// the executor failed or dropped the request.
    async fn stop_after(&mut self, n: usize) -> Result<Done, EngineError> {
        let mut d = self.cancel_and_drain().await.ok_or_else(|| EngineError::new(500, "server_error", "the executor dropped the request"))?;
        d.predicted_n = n;
        Ok(d)
    }
}

/// Send the request and wait until the executor has admitted or refused it, so a refusal is still an HTTP status.
async fn submit(st: &St, g: Gen) -> Result<Pump, ApiError> {
    if !st.shared.ready.load(Relaxed) {
        return Err(ApiError::new(503, "unavailable_error", "Loading model"));
    }
    let id = st.next_req.fetch_add(1, Relaxed);
    let (tx, mut rx) = mpsc::channel(EVENT_CAP);
    let cmd = StartCmd { id, agent: g.agent, ids: g.ids, params: g.params, lane: g.lane, deadline_ms: g.deadline_ms, events: tx };
    st.cmd.try_send(Command::Start(Box::new(cmd))).map_err(|_| queue_full())?;
    let mut guard = CancelGuard { id, cmd: st.cmd.clone(), armed: true };
    match rx.recv().await {
        Some(ReqEvent::Started { prompt_n, .. }) => Ok(Pump { rx, guard, prompt_n, finished: false }),
        Some(ReqEvent::Error(e)) => {
            guard.disarm();
            Err(e.into())
        }
        _ => Err(ApiError::new(500, "server_error", "the executor dropped the request")),
    }
}

async fn ask<T>(st: &St, make: impl FnOnce(oneshot::Sender<T>) -> Command) -> Result<T, ApiError> {
    let (tx, rx) = oneshot::channel();
    st.cmd.try_send(make(tx)).map_err(|_| queue_full())?;
    rx.await.map_err(|_| ApiError::new(500, "server_error", "the executor is gone"))
}

// -- /completion (llama-server wire) -----------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(untagged)]
enum Prompt {
    Ids(Vec<i32>),
    Text(String),
}

#[derive(Deserialize, Default)]
struct GenReq {
    #[serde(alias = "ids")]
    prompt: Option<Prompt>,
    #[serde(alias = "n", alias = "max_tokens")]
    n_predict: Option<i64>,
    temperature: Option<f32>,
    ignore_eos: Option<bool>,
    stream: Option<bool>,
    return_tokens: Option<bool>,
    lane: Option<Lane>,
    deadline_ms: Option<u64>,
    cache_prompt: Option<bool>,
    /// GBNF; null and "" are no grammar (an invalid one is refused by the engine with 400 before the request is queued).
    grammar: Option<String>,
    /// A string or a list; `""`, `[]` and null are no stop string.
    stop: Option<StopSpec>,
    /// Silent thinking steps instead of text (see `SamplingParams::latent_steps`).
    latent_steps: Option<usize>,
    /// A JSON schema the output must follow (llama-server's `json_schema`); it becomes the grammar ([`llama_compat::grammar_for`]).
    /// `/completion` and `/v1/completions` only: agent turns still refuse it ([`REFUSED_FIELDS`]).
    json_schema: Option<Value>,
    /// llama-server's `n_probs` (only `/completion` of the next engine lets a non-zero value through; see [`REFUSED_FIELDS`]):
    /// 0 or less is off, values above 2 are served as 2.
    #[serde(default, deserialize_with = "n_probs_or_empty")]
    n_probs: Option<i64>,
}

impl GenReq {
    fn params(&self) -> Result<SamplingParams, ApiError> {
        let n_predict = self.n_predict.unwrap_or(-1);
        if n_predict < -1 {
            return Err(ApiError::invalid("n_predict must be -1 (until EOS or the context is full) or >= 0"));
        }
        let temperature = self.temperature.unwrap_or(0.8);
        if !(temperature >= 0.0) {
            return Err(ApiError::invalid("temperature must be >= 0"));
        }
        Ok(SamplingParams { n_predict, temperature, ignore_eos: self.ignore_eos.unwrap_or(false), cache_prompt: self.cache_prompt.unwrap_or(true),
            grammar: llama_compat::grammar_for(&self.grammar, &self.json_schema, &None)?, latent_steps: self.latent_steps.unwrap_or(0),
            n_probs: self.n_probs.unwrap_or(0).clamp(0, 2) as usize })
    }
}

#[derive(Deserialize, Default)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

fn chat_template_cached(st: &St) -> Result<Arc<crate::chat::ChatTemplate>, String> {
    st.chat
        .get_or_init(|| match st.tokenizer.gguf_path() {
            Some(p) => crate::chat::ChatTemplate::from_gguf(p).map(Arc::new),
            None => Ok(Arc::new(crate::chat::ChatTemplate::chatml())),
        })
        .clone()
}

fn chat_template(st: &St) -> Result<Arc<crate::chat::ChatTemplate>, ApiError> {
    chat_template_cached(st).map_err(|e| ApiError::new(501, "not_supported_error", format!("events need the model's chat template: {e}")))
}
