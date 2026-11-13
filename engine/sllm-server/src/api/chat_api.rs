//! `POST /v1/chat/completions`: OpenAI chat over the executor.
//!
//! The messages and `tools` are rendered by the GGUF's chat template ([`crate::chat::ChatTemplate::render_chat`]; without
//! a GGUF, `--fake`, the ChatML stand-in), tokenized, and run like any `/v1/completions` request. The reply text goes
//! through [`Pipe`]: stop strings first ([`crate::stop`]), then the tool-call parser ([`crate::toolcall`]) when the
//! request has `tools` and `tool_choice` is `auto`. Stream and non-stream read the reply through the same `Pipe`, so they
//! agree.
//!
//! Honoured: `messages`, `tools`, `tool_choice` (`auto`, `none`), `stop`, `stream`, `stream_options.include_usage`,
//! `max_tokens` / `max_completion_tokens` (default: until EOS or the context is full), `temperature` (validated, decoding is
//! greedy), `ignore_eos`, `lane`, `deadline_ms`, `model` (echoed), `n` = 1. A forced `tool_choice` (`required`, a named
//! function) needs grammar-constrained decoding and is 400. `response_format`, `json_schema`, `chat_template_kwargs` and the
//! other [`REFUSED_FIELDS`] stay 400. Thinking is off (`enable_thinking` false). Errors: a body that is not JSON is 400, a
//! request that is not valid (no messages, a bad role, an image part) is 422, a template that raises is 400, a template with
//! no tool-call parser given `tools` is 501.
//!
//! Wire: `chat.completion` with `choices[0].message` (`content` null when the reply is only tool calls, `tool_calls` with
//! `arguments` a JSON string) and `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`,
//! `prompt_tokens_details.cached_tokens`). `finish_reason` is `stop`, `length` or `tool_calls`. Streaming sends
//! `chat.completion.chunk`s: a first delta with the role, content deltas, each tool call whole in one delta (never
//! partial JSON), a chunk with `finish_reason` and `usage`, a trailing usage chunk with empty `choices` when
//! `include_usage` is set, and `data: [DONE]`.

use super::*;
use crate::chat::{ChatTemplate, RenderError};
use crate::thinking::{prompt_opens_thinking, ThinkParser, ThinkPart};
use crate::toolcall::{Part, ToolCall, ToolFormat, ToolParser};

/// The answer levels (`"level"` in the request): sonic, fast, normal, deep.
mod answer;

/// Reading a reply: the collected message and the streaming driver (shared with the Anthropic route).
mod collect;
use collect::{collect_reply, next_token, Collected, Next};
pub(super) use collect::{ReplyWriter, Streaming};
pub(super) use answer::engine_has_probs;

pub(super) fn routes() -> Router<St> {
    Router::new().route("/v1/chat/completions", post(chat)).route("/chat/completions", post(chat))
}

/// `/caps` `chat`: what the route honours and refuses.
pub(super) fn caps(st: &St) -> Value {
    let format = chat_template_cached(st).ok().and_then(|t| t.tool_format()).map(ToolFormat::name);
    json!({
        "route": "/v1/chat/completions",
        "honoured": ["messages", "model", "tools", "tool_choice", "stop", "stream", "stream_options", "max_tokens", "max_completion_tokens",
                     "temperature", "ignore_eos", "lane", "deadline_ms", "n", "level", "grammar", "json_schema", "response_format", "reasoning_format", "chat_template_kwargs"],
        "honoured_with_conditions": {
            "logprobs": "logprobs true with top_logprobs 0 to 2, not with stream, on an engine with token probabilities (--engine next); one entry per generated token in choices[0].logprobs.content",
            "response_format": "text, json_object (any object) or json_schema; the schema becomes a grammar (a subset of JSON schema: pattern, allOf and integer bounds are 400); not with tools, grammar or level",
            "chat_template_kwargs": "only {\"enable_thinking\": true|false} (the template's thinking switch, off by default here, on by default in llama-server); any other key is 400",
            "reasoning_format": "auto and deepseek move <think>...</think> text to message.reasoning_content (delta.reasoning_content when streaming); none leaves it in the content; the default is --reasoning-format (auto)",
        },
        "levels": ["off", "sonic", "fast", "normal", "deep"],
        "refused": REFUSED_FIELDS.iter().filter(|f| !matches!(**f, "tools" | "tool_choice" | "json_schema" | "response_format" | "logprobs" | "top_logprobs" | "chat_template_kwargs")).collect::<Vec<_>>(),
        "tool_choice": ["auto", "none"],
        "tool_choice_refused": "required and a named function need grammar-constrained decoding: 400",
        "tool_call_format": format,
        "n": "must be 1",
        "content_parts": "text only; an image part is 422",
        "thinking": "off (enable_thinking false) unless chat_template_kwargs says {\"enable_thinking\": true}; a model that writes <think> text anyway has it split off as reasoning_content",
        "stop": "a string or a list; the stop string is not output; finish_reason stop",
        "usage": "on the chunk with finish_reason (and a trailing chunk with empty choices with stream_options.include_usage)",
    })
}

#[derive(Deserialize)]
struct ChatReq {
    messages: Option<Value>,
    model: Option<String>,
    tools: Option<Value>,
    tool_choice: Option<Value>,
    #[serde(alias = "max_completion_tokens")]
    max_tokens: Option<i64>,
    temperature: Option<f32>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    stop: Option<StopSpec>,
    ignore_eos: Option<bool>,
    n: Option<u32>,
    lane: Option<Lane>,
    deadline_ms: Option<u64>,
    /// "sonic" | "fast" | "normal" | "deep" (see [`answer`]); absent, null or "off": the plain path.
    level: Option<Value>,
    /// GBNF for the reply (llama-server's `grammar`); `json_schema` and `response_format` become one ([`llama_compat::grammar_for`]).
    grammar: Option<String>,
    json_schema: Option<Value>,
    response_format: Option<Value>,
    /// OpenAI's `logprobs` (true) and `top_logprobs` (0 to 2): the probability of each token of the reply (not with `stream`).
    logprobs: Option<Value>,
    top_logprobs: Option<Value>,
    /// llama-server's `chat_template_kwargs`: only `enable_thinking` (a boolean) is read; any other key is 400.
    chat_template_kwargs: Option<Value>,
    /// llama-server's `reasoning_format`: `auto` / `deepseek` split `<think>` text into `reasoning_content`, `none` leaves it.
    reasoning_format: Option<String>,
}

#[derive(PartialEq)]
enum ToolChoice {
    Auto,
    None,
}

fn tool_choice(v: &Option<Value>) -> Result<ToolChoice, ApiError> {
    match v {
        None | Some(Value::Null) => Ok(ToolChoice::Auto),
        Some(Value::String(s)) if s == "auto" => Ok(ToolChoice::Auto),
        Some(Value::String(s)) if s == "none" => Ok(ToolChoice::None),
        Some(Value::String(s)) if s == "required" => Err(forced()),
        Some(Value::Object(_)) => Err(forced()),
        Some(other) => Err(ApiError::invalid(format!("tool_choice must be \"auto\", \"none\", \"required\" or a function, not {other}"))),
    }
}

fn forced() -> ApiError {
    ApiError::new(400, "invalid_request_error", "unsupported `tool_choice`: \"required\" and a named function need grammar-constrained decoding, which the chat route does not have yet; use \"auto\" or \"none\"")
}

fn tool_list(v: &Option<Value>) -> Result<Vec<Value>, ApiError> {
    let Some(v) = v.as_ref().filter(|v| !v.is_null()) else { return Ok(vec![]) };
    let a = v.as_array().ok_or_else(|| ApiError::invalid("tools must be a list"))?;
    for t in a {
        let ok = t.get("type").and_then(Value::as_str) == Some("function") && t.pointer("/function/name").and_then(Value::as_str).is_some_and(|n| !n.is_empty());
        if !ok {
            return Err(ApiError::invalid("each tool must be {\"type\": \"function\", \"function\": {\"name\": ..., ...}}"));
        }
    }
    Ok(a.clone())
}

/// The prompt's token ids, as llama-server makes them: the vocabulary adds BOS itself when it wants one, so a BOS the
/// template wrote is cut from the front first (never two); a vocabulary that adds none keeps the template's.
pub(super) async fn prompt_ids(st: &St, tmpl: &ChatTemplate, text: String) -> Result<Vec<i32>, ApiError> {
    let add_bos = st.tokenizer.vocab().is_some_and(|v| v.add_bos());
    let text = match text.strip_prefix(tmpl.bos()) {
        Some(rest) if add_bos && !tmpl.bos().is_empty() => rest.to_string(),
        _ => text,
    };
    encode(st, text, add_bos, true).await
}

/// Stop strings, then (the Anthropic route) `<think>` reasoning, then tool calls, over the reply text.
pub(super) struct Pipe {
    pub(super) stop: StopFilter,
    pub(super) think: Option<ThinkParser>,
    pub(super) tools: Option<ToolParser>,
    /// The stop string that matched.
    pub(super) hit: Option<String>,
    /// The vocabulary, to print the control tokens a tool format's markers are made of.
    vocab: Option<Arc<llama_ref::Vocab>>,
}

impl Pipe {
    pub(super) fn new(stop: Vec<String>, think: Option<ThinkParser>, tools: Option<ToolParser>) -> Pipe {
        Pipe { stop: StopFilter::new(stop), think, tools, hit: None, vocab: None }
    }

    pub(super) fn with_vocab(mut self, vocab: Option<Arc<llama_ref::Vocab>>) -> Pipe {
        self.vocab = vocab;
        self
    }

    /// The text of token `id`. A control token the tool format is written with (`<|tool_call>`, `<|channel|>`) prints as
    /// nothing by default; with a tool parser its text is shown, so the parser can see the markers.
    fn shown(&self, id: i32, text: &str) -> String {
        if let (true, Some(tools), Some(v)) = (text.is_empty(), &self.tools, &self.vocab) {
            let piece = String::from_utf8_lossy(&v.piece_special(id)).into_owned();
            if tools.format().preserves(&piece) {
                return piece;
            }
        }
        text.to_string()
    }

    /// The reply text that is not reasoning: tool calls out of it, or content.
    fn answer(&mut self, shown: String) -> Vec<Part> {
        match &mut self.tools {
            Some(t) => t.push(&shown),
            None if shown.is_empty() => vec![],
            None => vec![Part::Content(shown)],
        }
    }

    fn text(&mut self, shown: String) -> Vec<Part> {
        let Some(think) = &mut self.think else { return self.answer(shown) };
        let pieces = think.push(&shown);
        self.route(pieces)
    }

    fn route(&mut self, pieces: Vec<ThinkPart>) -> Vec<Part> {
        let mut parts = Vec::new();
        for p in pieces {
            match p {
                ThinkPart::Thinking(t) => parts.push(Part::Thinking(t)),
                ThinkPart::Text(t) => parts.extend(self.answer(t)),
            }
        }
        parts
    }

    /// (parts to send now, a stop string matched)
    pub(super) fn token(&mut self, id: i32, text: &str) -> (Vec<Part>, bool) {
        let text = self.shown(id, text);
        let (shown, hit) = self.stop.push(&text);
        let matched = hit.is_some();
        if matched {
            self.hit = hit;
        }
        (self.text(shown), matched)
    }

    /// The reply is over: what was held back (a possible stop string, a possible tag) was not one.
    pub(super) fn finish(&mut self) -> Vec<Part> {
        let tail = self.stop.finish();
        let mut parts = self.text(tail);
        if let Some(think) = &mut self.think {
            let pieces = think.finish();
            parts.extend(self.route(pieces));
        }
        if let Some(t) = &mut self.tools {
            parts.extend(t.finish());
        }
        parts
    }
}

#[derive(Clone)]
struct Meta {
    id: String,
    req_id: ReqId,
    created: u64,
    model: String,
    include_usage: bool,
}

fn call_json(m: &Meta, i: usize, c: &ToolCall, with_index: bool) -> Value {
    let mut v = json!({"id": format!("call_{}_{i}", m.req_id), "type": "function", "function": {"name": c.name, "arguments": c.arguments}});
    if with_index {
        v["index"] = json!(i);
    }
    v
}

fn finish_of(d: &Done, matched: bool, calls: usize) -> &'static str {
    match ending(d, matched, calls) {
        Ending::Length => "length",
        Ending::ToolCalls => "tool_calls",
        Ending::StopString | Ending::Natural => "stop",
    }
}

fn chat_usage(d: &Done) -> Value {
    llama_compat::oai_usage(d)
}

fn chunk(m: &Meta, delta: Value, finish: Option<&str>) -> Value {
    llama_compat::stamp(json!({"id": m.id, "object": "chat.completion.chunk", "created": m.created, "model": m.model,
           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}))
}

/// The trailing chunk of a stream with `stream_options.include_usage`: no choices, the usage.
fn usage_chunk(m: &Meta, d: &Done) -> Value {
    llama_compat::stamp(json!({"id": m.id, "object": "chat.completion.chunk", "created": m.created, "model": m.model, "choices": [], "usage": chat_usage(d)}))
}

/// The `messages` of the request (a non-empty list) and the check that `n` is 1: both come before anything else is looked at.
fn checked_messages(r: &ChatReq) -> Result<Vec<Value>, ApiError> {
    let messages = match &r.messages {
        None | Some(Value::Null) => return Err(ApiError::new(400, "invalid_request_error", "'messages' is required")),
        Some(Value::Array(a)) if a.is_empty() => return Err(ApiError::new(400, "invalid_request_error", "`messages` must contain at least one message")),
        Some(Value::Array(a)) => a.clone(),
        Some(_) => return Err(ApiError::new(400, "invalid_request_error", "Expected 'messages' to be an array")),
    };
    if r.n.unwrap_or(1) != 1 {
        return Err(ApiError::invalid("n must be 1"));
    }
    Ok(messages)
}

fn template_of(st: &St) -> Result<Arc<ChatTemplate>, ApiError> {
    chat_template_cached(st).map_err(|e| ApiError::new(501, "not_supported_error", format!("the chat route needs the model's chat template: {e}")))
}

/// The model name the reply echoes: the request's, else the loaded model's.
fn model_of(st: &St, r: &ChatReq) -> String {
    r.model.clone().unwrap_or_else(|| llama_compat::default_model(st))
}

fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// The tool-call parser of a request that has tools (none: no tools); 501 when the template has no tool-call syntax, saying how
/// the route spells `tool_choice` none (`none_choice`).
pub(super) fn tool_parser(tmpl: &ChatTemplate, tools: &[Value], none_choice: &str) -> Result<Option<ToolParser>, ApiError> {
    if tools.is_empty() {
        return Ok(None);
    }
    let format = tmpl.tool_format().ok_or_else(|| {
        ApiError::new(501, "not_supported_error", format!("tools: this model's chat template has no tool-call syntax (formats read: qwen-json, qwen-xml, gemma4, gpt-oss, mistral, llama31-json); send tool_choice {none_choice} or no tools"))
    })?;
    Ok(Some(ToolParser::new(format).with_tools(tools)))
}

/// The chat template's prompt text for the messages (and tools); a request the template refuses is 422, a template that raises 400.
fn render_prompt(tmpl: &ChatTemplate, messages: &[Value], tools: &[Value], thinking: bool) -> Result<String, ApiError> {
    tmpl.render_chat(messages, tools, true, thinking).map_err(|e| match e {
        RenderError::Request(m) => ApiError::invalid(m),
        RenderError::Template(_) => ApiError::new(400, "invalid_request_error", e.to_string()),
    })
}

async fn chat(State(st): State<St>, body: Bytes) -> Result<Response, ApiError> {
    let r: ChatReq = parse_body_checked(&body, false, |v| {
        answer::check_body(v)?;
        reject_fields(v, &["tools", "tool_choice", "logprobs", "top_logprobs", "json_schema", "response_format", "chat_template_kwargs"])
    })?;
    let messages = checked_messages(&r)?;
    if let Some(level) = answer::parse_level(r.level.as_ref())? {
        return answer_level(&st, r, messages, level).await;
    }
    chat_plain(&st, r, messages).await
}

/// The prompt of a plain chat request (the template renders the messages and the tools; `tool_choice` none leaves the tools out) and the
/// parser for the tool calls of its reply.
async fn plain_prompt(st: &St, messages: &[Value], choice: ToolChoice, tools: Vec<Value>, thinking: bool) -> Result<PlainPrompt, ApiError> {
    let tmpl = template_of(st)?;
    // tool_choice none: the model is not told about the tools and its reply is not searched for calls
    let tools = if choice == ToolChoice::None { vec![] } else { tools };
    let parser = tool_parser(&tmpl, &tools, "\"none\"")?;
    let prompt = render_prompt(&tmpl, messages, &tools, thinking)?;
    let opens_thinking = prompt_opens_thinking(&prompt);
    let ids = prompt_ids(st, &tmpl, prompt).await?;
    if ids.is_empty() {
        return Err(ApiError::invalid("the messages render to an empty prompt"));
    }
    Ok(PlainPrompt { ids, parser, opens_thinking })
}

/// A plain request's prompt, the parser for the tool calls of its reply, and whether the prompt ends inside `<think>`.
struct PlainPrompt {
    ids: Vec<i32>,
    parser: Option<ToolParser>,
    opens_thinking: bool,
}

/// The reasoning parser of a reply, when the request's `reasoning_format` (else `--reasoning-format`) splits it off.
fn think_parser(st: &St, r: &ChatReq, opens_thinking: bool) -> Result<Option<ThinkParser>, ApiError> {
    let format = match r.reasoning_format.as_deref() {
        None => st.compat.reasoning,
        Some(f) => llama_compat::ReasoningFormat::parse(f).map_err(|e| ApiError::invalid(format!("reasoning_format: {e}")))?,
    };
    Ok(format.splits().then(|| ThinkParser::new(opens_thinking)))
}

/// The sampling parameters of a plain request: the limits, the grammar (`grammar`, `json_schema` or `response_format`) and `logprobs`.
fn plain_params(r: &ChatReq, has_tools: bool, logprobs: Option<llama_compat::LogprobsAsk>) -> Result<SamplingParams, ApiError> {
    let mut params = GenReq { n_predict: Some(r.max_tokens.unwrap_or(-1)), temperature: r.temperature, ignore_eos: r.ignore_eos, ..Default::default() }.params()?;
    params.grammar = llama_compat::grammar_for(&r.grammar, &r.json_schema, &r.response_format)?;
    if params.grammar.is_some() && has_tools {
        return Err(ApiError::new(400, "invalid_request_error", "unsupported: `tools` together with `grammar`, `json_schema` or `response_format` (the constrained reply could not hold a tool call)"));
    }
    params.n_probs = logprobs.map_or(0, |l| l.n_probs());
    Ok(params)
}

/// The plain path: tools (when the request has them) and stop strings, no answer level.
async fn chat_plain(st: &St, r: ChatReq, messages: Vec<Value>) -> Result<Response, ApiError> {
    let choice = tool_choice(&r.tool_choice)?;
    let tools = tool_list(&r.tools)?;
    let logprobs = llama_compat::chat_logprobs(r.logprobs.as_ref(), r.top_logprobs.as_ref(), st, r.stream)?;
    let params = plain_params(&r, choice == ToolChoice::Auto && !tools.is_empty(), logprobs)?;

    let thinking = llama_compat::enable_thinking(&r.chat_template_kwargs)?;
    let PlainPrompt { ids, parser, opens_thinking } = plain_prompt(st, &messages, choice, tools, thinking).await?;
    let think = think_parser(st, &r, opens_thinking)?;
    let model = model_of(st, &r);
    let pump = submit(st, Gen { agent: None, ids, params, lane: r.lane, deadline_ms: r.deadline_ms }).await?;
    let req_id = pump.guard.id;
    let meta = Meta { id: format!("chatcmpl-{req_id}"), req_id, created: unix_now(), model, include_usage: r.stream_options.is_some_and(|o| o.include_usage) };
    let pipe = Pipe::new(stops_of(&r.stop), think, parser).with_vocab(st.tokenizer.vocab());
    let resp = if r.stream { Ok(chat_stream(pump, pipe, meta).into_response()) } else { chat_collect(pump, pipe, meta, logprobs).await };
    resp.map(|resp| with_request_id(resp, req_id))
}

/// The reply at an answer level ([`answer`]): the plain path never gets here.
async fn answer_level(st: &St, r: ChatReq, messages: Vec<Value>, level: crate::levels::Level) -> Result<Response, ApiError> {
    let tmpl = template_of(st)?;
    let model = model_of(st, &r);
    let plan = answer::Plan {
        level,
        messages,
        tmpl,
        stops: stops_of(&r.stop),
        max_tokens: r.max_tokens,
        temperature: r.temperature,
        ignore_eos: r.ignore_eos,
        // a level is a chat answer: the interactive lane unless the caller names one (the background lane has no drafting
        // and other kernels: Qwen3.6-35B-A3B, deep on 60 questions 7.6 s against 3.1 s a question, 15 texts differ)
        lane: r.lane.or(Some(Lane::Interactive)),
        deadline_ms: r.deadline_ms,
        n_ctx: st.shared.info.get().map(|i| i.n_ctx),
        want_probs: level.gated() && answer::engine_has_probs(st),
        include_usage: r.stream_options.is_some_and(|o| o.include_usage),
        model,
    };
    // the first stage is submitted here, so that its refusal (a full context, a template error) is an HTTP status even when streaming
    let first = plan.start(st, level.first_stage(), plan.want_probs).await?;
    let req_id = first.guard.id;
    let meta = Meta { id: format!("chatcmpl-{req_id}"), req_id, created: unix_now(), model: plan.model.clone(), include_usage: plan.include_usage };
    if !r.stream {
        let out = answer::Out { tx: None, meta: meta.clone() };
        let o = answer::run(st, &plan, first, &out).await?;
        let resp = Json(json!({
            "id": meta.id, "object": "chat.completion", "created": meta.created, "model": meta.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": o.text}, "finish_reason": o.finish}],
            "usage": o.usage, "level": o.level,
        }))
        .into_response();
        return Ok(with_request_id(resp, req_id));
    }
    let (tx, rx) = mpsc::channel::<Event>(64);
    let st = st.clone();
    tokio::spawn(async move {
        let out = answer::Out { tx: Some(tx.clone()), meta: meta.clone() };
        let _ = tx.send(sse_data(&chunk(&meta, json!({"role": "assistant", "content": ""}), None))).await;
        match answer::run(&st, &plan, first, &out).await {
            Ok(o) => {
                let mut last = chunk(&meta, json!({}), Some(o.finish));
                last["usage"] = o.usage.clone();
                last["level"] = o.level;
                if tx.send(sse_data(&last)).await.is_err() {
                    return;
                }
                if meta.include_usage {
                    let _ = tx.send(sse_data(&json!({"id": meta.id, "object": "chat.completion.chunk", "created": meta.created, "model": meta.model,
                                                     "choices": [], "usage": o.usage}))).await;
                }
            }
            Err(e) => {
                let _ = tx.send(sse_data(&error_json(&e.0))).await;
            }
        }
        let _ = tx.send(Event::default().data("[DONE]")).await;
    });
    let s = stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|e| (Ok::<_, Infallible>(e), rx)) });
    Ok(with_request_id(Sse::new(s).into_response(), req_id))
}

async fn chat_collect(mut pump: Pump, mut pipe: Pipe, m: Meta, logprobs: Option<llama_compat::LogprobsAsk>) -> Result<Response, ApiError> {
    let mut got = Collected { top_logprobs: logprobs.map(|l| l.top), ..Default::default() };
    let (d, matched) = collect_reply(&mut pump, &mut pipe, &mut got).await?;
    got.absorb(pipe.finish());
    let mut choice = json!({"index": 0, "message": got.message(&m), "finish_reason": finish_of(&d, matched, got.calls.len())});
    if logprobs.is_some() {
        choice["logprobs"] = llama_compat::logprobs_json(got.logprobs);
    }
    let reply = llama_compat::stamp(json!({
        "id": m.id, "object": "chat.completion", "created": m.created, "model": m.model,
        "choices": [choice],
        "usage": chat_usage(&d),
    }));
    Ok(Json(llama_compat::with_timings(reply, &d)).into_response())
}

/// The chat route's own state of a stream: its ids, whether the first chunk is out, and how many tool calls went out.
struct ChatWriter {
    m: Meta,
    started: bool,
    calls: usize,
}

impl ReplyWriter for ChatWriter {
    type Event = Event;

    fn parts(&mut self, parts: Vec<Part>, out: &mut Vec<Event>) {
        for p in parts {
            let delta = match p {
                Part::Content(t) => json!({"content": t}),
                Part::Call(c) => {
                    let v = json!({"tool_calls": [call_json(&self.m, self.calls, &c, true)]});
                    self.calls += 1;
                    v
                }
                Part::Thinking(t) => json!({"reasoning_content": t}),
            };
            out.push(sse_data(&chunk(&self.m, delta, None)));
        }
    }

    fn end(&mut self, d: &Done, hit: Option<&str>, out: &mut Vec<Event>) {
        let mut last = chunk(&self.m, json!({}), Some(finish_of(d, hit.is_some(), self.calls)));
        last["usage"] = chat_usage(d);
        // llama-server's `timings` ride on the last chunk of the stream
        out.push(sse_data(&if self.m.include_usage { last } else { llama_compat::with_timings(last, d) }));
        if self.m.include_usage {
            out.push(sse_data(&llama_compat::with_timings(usage_chunk(&self.m, d), d)));
        }
        out.push(Event::default().data("[DONE]"));
    }

    fn error(&mut self, e: EngineError, out: &mut Vec<Event>) {
        stream_error(out, &e);
    }
}

impl Streaming<ChatWriter> {
    /// The events for the next piece of the reply, and whether the stream is over.
    async fn step(&mut self) -> (Vec<Event>, bool) {
        let mut out = Vec::new();
        if !self.w.started {
            self.w.started = true;
            out.push(sse_data(&chunk(&self.w.m, json!({"role": "assistant", "content": ""}), None)));
        }
        let item = self.pump.next().await;
        let over = self.item(item, &mut out).await;
        (out, over)
    }
}

fn chat_stream(pump: Pump, pipe: Pipe, m: Meta) -> impl IntoResponse {
    let s = stream::unfold(Some(Streaming { pump, pipe, n: 0, w: ChatWriter { m, started: false, calls: 0 } }), |state| async move {
        let mut run = state?;
        let (events, done) = run.step().await;
        Some((stream::iter(events.into_iter().map(Ok::<_, Infallible>)), if done { None } else { Some(run) }))
    })
    .flatten();
    Sse::new(s)
}
