//! Answer levels on `POST /v1/chat/completions` (`"level": "sonic" | "fast" | "normal" | "deep"`; absent, null or "off": the
//! plain path of [`super::chat`], untouched). The orchestration of `answer_generate` in `playground/src/sllm_playground/levels.py`;
//! the pure pieces are in [`crate::levels`].
//!
//! A stage is one generation with the stage's system prompt (the caller's system message follows it) and a token cap, the chat
//! template's thinking off. sonic (ONE) and fast (CHAT, an ordinary chat answer) run one stage and stream it. normal runs
//! ONE, TIGHT and, when they do not agree, LONG; deep is for now the same as normal (a long-work level is being built). No
//! level uses token probabilities since 2026-11-13. Nothing of a short answer of normal/deep is streamed while it is written: a kept one is sent whole as one content delta, a LONG answer streams as it is written.
//!
//! The response `usage` counts the prompt and completion tokens of every stage that ran; `level` (top level, and on the final
//! stream chunk) is `{"name", "stages": [{"prompt", "tokens"}], "check"}`.

use super::*;
use crate::levels::{self, Level, Stage};

/// The request fields that make no sense next to a level (a level answers plain chat).
const CONFLICTS: &[&str] = &["tools", "tool_choice", "response_format", "json_schema", "grammar", "logprobs", "chat_template_kwargs"];

/// The level a request asks for: `None` for no level (absent, null, "off"); 422 for anything else that is not a level name.
pub(super) fn parse_level(v: Option<&Value>) -> Result<Option<Level>, ApiError> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s == "off" => Ok(None),
        Some(Value::String(s)) => Level::parse(s).map(Some).ok_or_else(|| ApiError::invalid(format!("unknown level {s:?}: use \"sonic\", \"fast\", \"normal\", \"deep\" or \"off\""))),
        Some(other) => Err(ApiError::invalid(format!("level must be a string (\"sonic\", \"fast\", \"normal\", \"deep\" or \"off\"), not {other}"))),
    }
}

/// The check on the raw body: a bad level is 422, a level together with tools / tool_choice / response_format / grammar is 400.
/// Does nothing without a level.
pub(super) fn check_body(v: &Value) -> Result<(), ApiError> {
    let Some(o) = v.as_object() else { return Ok(()) };
    let Some(level) = parse_level(o.get("level"))? else { return Ok(()) };
    for f in CONFLICTS {
        if o.get(*f).is_some_and(|x| !is_not_asked_for(f, x)) {
            return Err(ApiError::new(
                400,
                "invalid_request_error",
                format!("`level` ({}) cannot be combined with `{f}`: answer levels are for plain answers (no tools, no constrained output)", level.name()),
            ));
        }
    }
    Ok(())
}

/// Whether the engine serves token probabilities (`n_probs`): the next engine; the fake engine models them for tests.
pub(in crate::api) fn engine_has_probs(st: &St) -> bool {
    matches!(st.shared.caps.get().and_then(|c| c.get("engine")).and_then(Value::as_str), Some("next" | "fake"))
}

/// The text of a message's `content` (a string, or a list of text parts).
fn content_text(m: &Value) -> String {
    match m.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .map(|p| match p {
                Value::String(s) => s.as_str(),
                other => other.get("text").and_then(Value::as_str).unwrap_or(""),
            })
            .collect(),
        _ => String::new(),
    }
}

pub(super) struct Plan {
    pub level: Level,
    pub messages: Vec<Value>,
    pub tmpl: Arc<ChatTemplate>,
    pub stops: Vec<String>,
    /// The caller's `max_tokens`; negative (until the context is full) is no limit.
    pub max_tokens: Option<i64>,
    pub temperature: Option<f32>,
    pub ignore_eos: Option<bool>,
    pub lane: Option<Lane>,
    pub deadline_ms: Option<u64>,
    pub n_ctx: Option<usize>,
    pub want_probs: bool,
    pub include_usage: bool,
    pub model: String,
}

impl Plan {
    /// The messages of a stage: the stage prompt as the system message, the caller's system message (when it sent one first)
    /// after it ([`levels::stage_system`]).
    fn stage_messages(&self, stage: Stage) -> Vec<Value> {
        let mut msgs = self.messages.clone();
        let first_is_system = msgs.first().and_then(|m| m.get("role")).and_then(Value::as_str).is_some_and(|r| r == "system" || r == "developer");
        let caller = if first_is_system { content_text(&msgs[0]) } else { String::new() };
        let sys = json!({"role": "system", "content": levels::stage_system(stage.prompt(), &caller, "")});
        if first_is_system {
            msgs[0] = sys;
        } else {
            msgs.insert(0, sys);
        }
        msgs
    }

    /// Render, tokenize, cap and submit one stage.
    pub async fn start(&self, st: &St, stage: Stage, probs: bool) -> Result<Pump, ApiError> {
        let prompt = self.tmpl.render_chat(&self.stage_messages(stage), &[], true, false).map_err(|e| match e {
            RenderError::Request(m) => ApiError::invalid(m),
            RenderError::Template(_) => ApiError::new(400, "invalid_request_error", e.to_string()),
        })?;
        let ids = prompt_ids(st, &self.tmpl, prompt).await?;
        if ids.is_empty() {
            return Err(ApiError::invalid("the messages render to an empty prompt"));
        }
        // the stage's cap, the caller's limit, what the context leaves
        let mut cap = stage.cap();
        if let Some(m) = self.max_tokens.filter(|m| *m >= 0) {
            cap = cap.min(m);
        }
        if let Some(n_ctx) = self.n_ctx {
            cap = cap.min((n_ctx as i64 - ids.len() as i64).max(1));
        }
        let params = GenReq {
            n_predict: Some(cap),
            temperature: self.temperature,
            ignore_eos: self.ignore_eos,
            n_probs: probs.then_some(levels::N_PROBS),
            ..Default::default()
        }
        .params()?;
        submit(st, Gen { agent: None, ids, params, lane: self.lane, deadline_ms: self.deadline_ms }).await
    }
}

/// Where the content deltas of a streamed reply go (none: a whole reply is built).
pub(super) struct Out {
    pub tx: Option<mpsc::Sender<Event>>,
    pub meta: Meta,
}

impl Out {
    /// Send a content delta; false: the client went away.
    async fn content(&self, text: &str) -> bool {
        match &self.tx {
            Some(tx) if !text.is_empty() => tx.send(sse_data(&chunk(&self.meta, json!({"content": text}), None))).await.is_ok(),
            _ => true,
        }
    }
}

fn gone() -> ApiError {
    ApiError::new(499, "client_closed_request", "the client went away")
}

struct StageRun {
    stage: Stage,
    text: String,
    done: Done,
    matched: bool,
}

/// The text of a stage as it is read, and where it goes while it is written.
struct StageReader<'a> {
    pipe: Pipe,
    text: String,
    out: &'a Out,
    /// Its content goes to the client as it is written.
    stream: bool,
}

impl StageReader<'_> {
    /// The content in `parts` is added to the text; returned as what is new.
    fn take(&mut self, parts: Vec<Part>) -> String {
        let mut now = String::new();
        for p in parts {
            if let Part::Content(t) = p {
                now.push_str(&t);
            }
        }
        self.text.push_str(&now);
        now
    }

    /// Send what is new when the stage streams; an error when the client went away.
    async fn forward(&self, now: &str) -> Result<(), ApiError> {
        if self.stream && !self.out.content(now).await {
            return Err(gone());
        }
        Ok(())
    }

    /// One token through the pipe; whether a stop string matched.
    async fn token(&mut self, id: i32, piece: &str) -> Result<bool, ApiError> {
        let (parts, hit) = self.pipe.token(id, piece);
        let now = self.take(parts);
        self.forward(&now).await?;
        // a silent stage sends nothing, so a client that left would only be noticed at the first delta (real engine,
        // deep cut after 0.4 s: 8.5 s and 1,048 tokens more): the stream's receiver being gone ends the stage now
        if !self.stream && self.out.tx.as_ref().is_some_and(|t| t.is_closed()) {
            return Err(gone());
        }
        Ok(hit)
    }

    /// The reply is over: what the pipe held back.
    async fn finish(&mut self) -> Result<(), ApiError> {
        let parts = self.pipe.finish();
        let now = self.take(parts);
        self.forward(&now).await
    }
}

/// Read the stage's reply to its end through the reader; (the final report, a stop string matched).
async fn read_stage(pump: &mut Pump, rd: &mut StageReader<'_>) -> Result<(Done, bool), ApiError> {
    let mut n = 0usize;
    loop {
        match next_token(pump).await? {
            Next::Token(id, piece, _) => {
                n += 1;
                if rd.token(id, &piece).await? {
                    return Ok((pump.stop_after(n).await?, true));
                }
            }
            Next::Done(d) => return Ok((d, false)),
        }
    }
}

/// Read one stage to its end; `stream`: its content goes to the client as it is written.
async fn drain(mut pump: Pump, stage: Stage, stops: &[String], out: &Out, stream: bool) -> Result<StageRun, ApiError> {
    let mut rd = StageReader { pipe: Pipe::new(stops.to_vec(), None, None), text: String::new(), out, stream };
    let (done, matched) = read_stage(&mut pump, &mut rd).await?;
    rd.finish().await?;
    Ok(StageRun { stage, text: rd.text, done, matched })
}

/// What a level run produced.
pub(super) struct Outcome {
    pub text: String,
    pub finish: &'static str,
    pub usage: Value,
    pub level: Value,
}

/// Run the level. `first` is the already submitted first stage (so that a refusal of it is still an HTTP status): ONE for sonic,
/// normal and deep, CHAT for fast.
pub(super) async fn run(st: &St, plan: &Plan, first: Pump, out: &Out) -> Result<Outcome, ApiError> {
    let level = plan.level;
    let mut ran: Vec<StageRun> = Vec::new();
    let streams_first = matches!(level, Level::Sonic | Level::Fast);
    let first = drain(first, level.first_stage(), &plan.stops, out, streams_first).await?;
    let (shown, check): (usize, Option<String>);
    match level {
        Level::Sonic | Level::Fast => {
            ran.push(first);
            (shown, check) = (0, None);
        }
        Level::Normal | Level::Deep => {
            let tight = drain(plan.start(st, Stage::Tight, false).await?, Stage::Tight, &plan.stops, out, false).await?;
            let (same, why) = levels::answers_agree(&first.text, &tight.text);
            ran.push(first);
            ran.push(tight);
            (shown, check) = if same { keep(out, &ran[1], 1, why).await? } else { second(st, plan, out, &mut ran, why).await? };
        }
    }
    let shown = &ran[shown];
    let (mut prompt_n, mut completion_n, mut cached) = (0, 0, 0);
    for r in &ran {
        prompt_n += r.done.prompt_n;
        completion_n += r.done.predicted_n;
        cached += r.done.cached_n + r.done.shared_n;
    }
    Ok(Outcome {
        text: shown.text.clone(),
        finish: finish_of(&shown.done, shown.matched, 0),
        usage: json!({"prompt_tokens": prompt_n, "completion_tokens": completion_n, "total_tokens": prompt_n + completion_n,
                      "prompt_tokens_details": {"cached_tokens": cached}}),
        level: json!({"name": level.name(), "stages": ran.iter().map(|r| json!({"prompt": r.stage.name(), "tokens": r.done.predicted_n})).collect::<Vec<_>>(),
                      "check": check}),
    })
}

/// The short answer `ran[at]` is kept: sent whole as one content delta.
async fn keep(out: &Out, r: &StageRun, at: usize, why: String) -> Result<(usize, Option<String>), ApiError> {
    if !out.content(&r.text).await {
        return Err(gone());
    }
    Ok((at, Some(why)))
}

/// Answer again with LONG (streamed as it is written); an empty LONG answer keeps the last short one.
async fn second(st: &St, plan: &Plan, out: &Out, ran: &mut Vec<StageRun>, why: String) -> Result<(usize, Option<String>), ApiError> {
    let long = drain(plan.start(st, Stage::Long, false).await?, Stage::Long, &plan.stops, out, true).await?;
    let kept = ran.len() - 1;
    let empty = long.text.trim().is_empty();
    ran.push(long);
    if empty {
        keep(out, &ran[kept], kept, format!("{why} (the second answer was empty: kept the TIGHT answer)")).await
    } else {
        Ok((ran.len() - 1, Some(why)))
    }
}
