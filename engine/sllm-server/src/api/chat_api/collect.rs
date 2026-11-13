//! Reading a chat reply to its end or event by event: the pieces `/v1/chat/completions` and `/v1/messages` share.

use super::*;

pub(super) fn executor_dropped() -> ApiError {
    ApiError::new(500, "server_error", "the executor dropped the request")
}

/// The content, the reasoning and the tool calls a whole reply is made of, and (when asked for) the probability of each token.
#[derive(Default)]
pub(super) struct Collected {
    pub(super) content: String,
    pub(super) reasoning: String,
    pub(super) calls: Vec<ToolCall>,
    /// `logprobs.content`: one entry per generated token, with this many `top_logprobs` (`None`: not asked for).
    pub(super) logprobs: Vec<Value>,
    pub(super) top_logprobs: Option<usize>,
}

impl Collected {
    pub(super) fn absorb(&mut self, parts: Vec<Part>) {
        for p in parts {
            match p {
                Part::Content(t) => self.content.push_str(&t),
                Part::Call(c) => self.calls.push(c),
                Part::Thinking(t) => self.reasoning.push_str(&t),
            }
        }
    }

    /// The reply as `choices[0].message` (`content` null when the reply is only tool calls; `reasoning_content` when there was reasoning).
    pub(super) fn message(&self, m: &Meta) -> Value {
        let mut msg = json!({"role": "assistant", "content": if self.content.is_empty() && !self.calls.is_empty() { Value::Null } else { json!(self.content) }});
        if !self.reasoning.is_empty() {
            msg["reasoning_content"] = json!(self.reasoning);
        }
        if !self.calls.is_empty() {
            msg["tool_calls"] = Value::Array(self.calls.iter().enumerate().map(|(i, c)| call_json(m, i, c, false)).collect());
        }
        msg
    }

    fn note_probs(&mut self, probs: Option<Box<TokProbs>>) {
        if let (Some(top), Some(p)) = (self.top_logprobs, probs) {
            self.logprobs.push(llama_compat::logprob_entry(&p, top));
        }
    }
}

/// The next thing a whole (non-streamed) reply delivers.
pub(in crate::api) enum Next {
    Token(i32, String, Option<Box<TokProbs>>),
    Done(Done),
}

/// The reply's next token, or its final report; an engine error or a dropped request is the error.
pub(in crate::api) async fn next_token(pump: &mut Pump) -> Result<Next, ApiError> {
    match pump.next().await {
        Some(Item::Token { id, text, probs }) => Ok(Next::Token(id, text, probs)),
        Some(Item::Done(d)) => Ok(Next::Done(d)),
        Some(Item::Error(e)) => Err(e.into()),
        None => Err(executor_dropped()),
    }
}

/// Read the reply to its end through the pipe; (the final report, a stop string matched).
pub(super) async fn collect_reply(pump: &mut Pump, pipe: &mut Pipe, got: &mut Collected) -> Result<(Done, bool), ApiError> {
    let mut n = 0usize;
    loop {
        match next_token(pump).await? {
            Next::Token(id, text, probs) => {
                n += 1;
                got.note_probs(probs);
                let (parts, hit) = pipe.token(id, &text);
                got.absorb(parts);
                if hit {
                    return Ok((pump.stop_after(n).await?, true));
                }
            }
            Next::Done(d) => return Ok((d, false)),
        }
    }
}

/// How a route words a streamed reply: its parts, its end and its failure as that route's events.
pub(in crate::api) trait ReplyWriter {
    type Event;
    fn parts(&mut self, parts: Vec<Part>, out: &mut Vec<Self::Event>);
    /// The reply is over and its last parts are written: `d` is the final report, `hit` the stop string that cut it.
    fn end(&mut self, d: &Done, hit: Option<&str>, out: &mut Vec<Self::Event>);
    fn error(&mut self, e: EngineError, out: &mut Vec<Self::Event>);
}

/// A reply streamed from its pump through its pipe, worded by `w`.
pub(in crate::api) struct Streaming<W> {
    pub(in crate::api) pump: Pump,
    pub(in crate::api) pipe: Pipe,
    /// Tokens so far.
    pub(in crate::api) n: usize,
    pub(in crate::api) w: W,
}

impl<W: ReplyWriter> Streaming<W> {
    /// One item of the reply as events in `out`; whether the reply is over.
    pub(in crate::api) async fn item(&mut self, item: Option<Item>, out: &mut Vec<W::Event>) -> bool {
        match item {
            Some(Item::Token { id, text, .. }) => {
                self.n += 1;
                let (parts, hit) = self.pipe.token(id, &text);
                let stopped = if hit { Some(self.pump.stop_after(self.n).await) } else { None };
                self.w.parts(parts, out);
                let Some(end) = stopped else { return false };
                match end {
                    Ok(d) => self.finish(&d, out),
                    Err(e) => self.w.error(e, out),
                }
                true
            }
            Some(Item::Done(d)) => {
                self.finish(&d, out);
                true
            }
            Some(Item::Error(e)) => {
                self.w.error(e, out);
                true
            }
            None => true,
        }
    }

    /// The reply ended with `d`: what the pipe held back, then the route's ending.
    fn finish(&mut self, d: &Done, out: &mut Vec<W::Event>) {
        let parts = self.pipe.finish();
        self.w.parts(parts, out);
        self.w.end(d, self.pipe.hit.as_deref(), out);
    }
}

