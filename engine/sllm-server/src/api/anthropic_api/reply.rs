//! The reply of `/v1/messages`: the stream's events, and the non-stream message folded from them.

use super::*;

pub(super) struct Meta {
    pub(super) id: String,
    pub(super) req_id: ReqId,
    pub(super) model: String,
    pub(super) input_tokens: usize,
    pub(super) cache_control: bool,
}

/// One server-sent event: its name (also the `type` of its data) and its data.
type Ev = (&'static str, Value);

fn sse_event((name, v): Ev) -> Event {
    Event::default().event(name).data(v.to_string())
}

pub(super) fn usage_json(m: &Meta, d: &Done) -> Value {
    let read = d.cached_n + d.shared_n;
    let input = if m.cache_control { d.prompt_n.saturating_sub(read) } else { d.prompt_n };
    json!({"input_tokens": input, "output_tokens": d.predicted_n, "cache_read_input_tokens": read, "cache_creation_input_tokens": 0})
}

pub(super) fn stop_reason(d: &Done, matched: bool, calls: usize) -> &'static str {
    match ending(d, matched, calls) {
        Ending::Length => "max_tokens",
        Ending::ToolCalls => "tool_use",
        Ending::StopString => "stop_sequence",
        Ending::Natural => "end_turn",
    }
}

/// A call's `input` as JSON text: the arguments object, compact; arguments that are not an object are kept under `arguments`.
fn call_input(c: &ToolCall) -> String {
    match serde_json::from_str::<Value>(&c.arguments) {
        Ok(v @ Value::Object(_)) => v.to_string(),
        _ => json!({"arguments": c.arguments}).to_string(),
    }
}

#[derive(PartialEq, Clone, Copy)]
enum Open {
    Text,
    Thinking,
}

/// The content blocks of a reply as events: a block of text or thinking stays open while the same kind goes on; a tool call
/// is a whole block of its own.
#[derive(Default)]
pub(super) struct Blocks {
    open: Option<Open>,
    next: usize,
    calls: usize,
    pinged: bool,
}

/// What a thinking block's `signature_delta` says: the server cannot sign reasoning, and drops thinking blocks that come back.
const SIGNATURE: &str = "sllm-unsigned";

impl Blocks {
    fn start(&mut self, block: Value, out: &mut Vec<Ev>) {
        out.push(("content_block_start", json!({"type": "content_block_start", "index": self.next, "content_block": block})));
        self.next += 1;
        if !self.pinged {
            self.pinged = true;
            out.push(("ping", json!({"type": "ping"})));
        }
    }

    fn delta(&self, delta: Value, out: &mut Vec<Ev>) {
        out.push(("content_block_delta", json!({"type": "content_block_delta", "index": self.next - 1, "delta": delta})));
    }

    fn stop(&self, out: &mut Vec<Ev>) {
        out.push(("content_block_stop", json!({"type": "content_block_stop", "index": self.next - 1})));
    }

    fn close(&mut self, out: &mut Vec<Ev>) {
        match self.open.take() {
            Some(Open::Thinking) => {
                self.delta(json!({"type": "signature_delta", "signature": SIGNATURE}), out);
                self.stop(out);
            }
            Some(Open::Text) => self.stop(out),
            None => {}
        }
    }

    /// `delta` into the open block of `kind`; another open block is closed and a new one (`block`) started first.
    fn write(&mut self, kind: Open, block: Value, delta: Value, out: &mut Vec<Ev>) {
        if self.open != Some(kind) {
            self.close(out);
            self.start(block, out);
            self.open = Some(kind);
        }
        self.delta(delta, out);
    }

    fn part(&mut self, p: Part, m: &Meta, out: &mut Vec<Ev>) {
        match p {
            Part::Content(t) => self.write(Open::Text, json!({"type": "text", "text": ""}), json!({"type": "text_delta", "text": t}), out),
            Part::Thinking(t) => self.write(Open::Thinking, json!({"type": "thinking", "thinking": ""}), json!({"type": "thinking_delta", "thinking": t}), out),
            Part::Call(c) => {
                self.close(out);
                self.start(json!({"type": "tool_use", "id": format!("toolu_{}_{}", m.req_id, self.calls), "name": c.name, "input": {}}), out);
                self.delta(json!({"type": "input_json_delta", "partial_json": call_input(&c)}), out);
                self.stop(out);
                self.calls += 1;
            }
        }
    }

    /// The reply is over: close the open block; a reply with no block gets one empty text block.
    fn end(&mut self, out: &mut Vec<Ev>) {
        self.close(out);
        if self.next == 0 {
            self.start(json!({"type": "text", "text": ""}), out);
            self.stop(out);
        }
    }
}

/// The message route's own state of a stream: its ids and counts, the open blocks, whether `message_start` is out, and when to ping.
pub(super) struct MessageWriter {
    pub(super) m: Meta,
    pub(super) blocks: Blocks,
    pub(super) started: bool,
    /// Send a `ping` when no event has come for this long.
    pub(super) ping: Option<Duration>,
    /// The error that ended the reply (also sent as an `error` event).
    pub(super) error: Option<EngineError>,
}

impl ReplyWriter for MessageWriter {
    type Event = Ev;

    fn parts(&mut self, parts: Vec<Part>, out: &mut Vec<Ev>) {
        for p in parts {
            self.blocks.part(p, &self.m, out);
        }
    }

    fn end(&mut self, d: &Done, hit: Option<&str>, out: &mut Vec<Ev>) {
        self.blocks.end(out);
        out.push((
            "message_delta",
            json!({"type": "message_delta", "delta": {"stop_reason": stop_reason(d, hit.is_some(), self.blocks.calls), "stop_sequence": hit},
                   "usage": usage_json(&self.m, d)}),
        ));
        out.push(("message_stop", json!({"type": "message_stop"})));
    }

    fn error(&mut self, e: EngineError, out: &mut Vec<Ev>) {
        out.push(("error", error_body(&e).1));
        self.error = Some(e);
    }
}

impl Streaming<MessageWriter> {
    /// The events for the next piece of the reply, and whether the reply is over.
    async fn step(&mut self) -> (Vec<Ev>, bool) {
        let mut out = Vec::new();
        let w = &mut self.w;
        if !w.started {
            // the first bytes go out before the prefill is done
            w.started = true;
            let usage = json!({"input_tokens": w.m.input_tokens, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0});
            out.push((
                "message_start",
                json!({"type": "message_start", "message": {"id": w.m.id, "type": "message", "role": "assistant", "model": w.m.model, "content": [],
                                                           "stop_reason": null, "stop_sequence": null, "usage": usage}}),
            ));
            return (out, false);
        }
        let next = match w.ping {
            Some(every) => match tokio::time::timeout(every, self.pump.next()).await {
                Ok(item) => item,
                Err(_) => return (vec![("ping", json!({"type": "ping"}))], false),
            },
            None => self.pump.next().await,
        };
        let over = self.item(next, &mut out).await;
        (out, over)
    }
}

pub(super) fn stream_reply(run: Streaming<MessageWriter>) -> impl IntoResponse {
    let s = stream::unfold(Some(run), |state| async move {
        let mut run = state?;
        let (events, done) = run.step().await;
        Some((stream::iter(events.into_iter().map(|e| Ok::<_, Infallible>(sse_event(e)))), if done { None } else { Some(run) }))
    })
    .flatten();
    Sse::new(s)
}

/// A `content_block_delta`'s `d` onto the block `b` it adds to (a call's arguments onto `partial`, parsed at its stop).
fn apply_delta(b: &mut Value, d: &Value, partial: &mut String) {
    let append = |b: &mut Value, key: &str, add: &Value| b[key] = json!(format!("{}{}", b[key].as_str().unwrap_or(""), add.as_str().unwrap_or("")));
    match d["type"].as_str() {
        Some("text_delta") => append(b, "text", &d["text"]),
        Some("thinking_delta") => append(b, "thinking", &d["thinking"]),
        Some("signature_delta") => b["signature"] = d["signature"].clone(),
        Some("input_json_delta") => partial.push_str(d["partial_json"].as_str().unwrap_or("")),
        _ => {}
    }
}

/// The content blocks the events build, as an SDK folds them.
fn fold(events: &[Ev]) -> Vec<Value> {
    let (mut content, mut partial) = (Vec::<Value>::new(), String::new());
    for (name, v) in events {
        match *name {
            "content_block_start" => {
                content.push(v["content_block"].clone());
                partial.clear();
            }
            "content_block_delta" => {
                let (Some(b), d) = (content.last_mut(), &v["delta"]) else { continue };
                apply_delta(b, d, &mut partial);
            }
            "content_block_stop" => {
                if let Some(b) = content.last_mut().filter(|b| b["type"] == "tool_use") {
                    b["input"] = serde_json::from_str(&partial).unwrap_or_else(|_| json!({}));
                }
            }
            _ => {}
        }
    }
    content
}

pub(super) async fn collect(mut run: Streaming<MessageWriter>) -> Result<Response, AErr> {
    let mut events = Vec::new();
    loop {
        let (e, done) = run.step().await;
        events.extend(e);
        if done {
            break;
        }
    }
    if let Some(e) = run.w.error.take() {
        return Err(e.into());
    }
    let Some(delta) = events.iter().find(|(n, _)| *n == "message_delta").map(|(_, v)| v.clone()) else {
        return Err(EngineError::new(500, "server_error", "the executor dropped the request").into());
    };
    Ok(Json(json!({
        "id": run.w.m.id, "type": "message", "role": "assistant", "model": run.w.m.model, "content": fold(&events),
        "stop_reason": delta["delta"]["stop_reason"], "stop_sequence": delta["delta"]["stop_sequence"], "usage": delta["usage"],
    }))
    .into_response())
}
