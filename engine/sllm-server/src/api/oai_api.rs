//! `POST /v1/completions` (OpenAI wire).

use super::*;

#[derive(Deserialize)]
struct OaiReq {
    prompt: Value,
    #[serde(default)]
    model: Option<String>,
    max_tokens: Option<i64>,
    temperature: Option<f32>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    ignore_eos: Option<bool>,
    grammar: Option<String>,
    /// A JSON schema the text must follow (becomes the grammar).
    json_schema: Option<Value>,
    /// OpenAI's integer `logprobs` (1 or 2): the probability of each token, with that many best ones; not with `stream`.
    logprobs: Option<Value>,
    stop: Option<StopSpec>,
    n: Option<u32>,
    lane: Option<Lane>,
    deadline_ms: Option<u64>,
}

fn int_array(v: &[Value]) -> Option<Vec<i32>> {
    v.iter().map(|x| x.as_i64().and_then(|i| i32::try_from(i).ok())).collect()
}

fn bad_prompt() -> ApiError {
    ApiError::invalid("prompt must be a string, an array of token ids, or a one-element array of either")
}

async fn oai_prompt(st: &St, p: &Value) -> Result<Vec<i32>, ApiError> {
    match p {
        Value::String(s) => encode(st, s.clone(), true, true).await,
        Value::Array(a) => oai_prompt_list(st, a).await,
        _ => Err(bad_prompt()),
    }
}

/// A prompt given as a list: token ids, or a one-element list of a string or of token ids.
async fn oai_prompt_list(st: &St, a: &[Value]) -> Result<Vec<i32>, ApiError> {
    match a {
        [] => Err(bad_prompt()),
        a if a.iter().all(|x| x.is_i64() || x.is_u64()) => int_array(a).ok_or_else(bad_prompt),
        [one] => match one {
            Value::String(s) => encode(st, s.clone(), true, true).await,
            Value::Array(inner) => int_array(inner).ok_or_else(bad_prompt),
            _ => Err(bad_prompt()),
        },
        _ => Err(ApiError::invalid("batched prompts are not supported: send one prompt per request")),
    }
}

fn oai_chunk(id: &str, created: u64, model: &str, text: &str, finish: Option<&str>) -> Value {
    llama_compat::stamp(json!({"id": id, "object": "text_completion", "created": created, "model": model,
           "choices": [{"index": 0, "text": text, "logprobs": null, "finish_reason": finish}]}))
}

pub(super) fn finish_reason(r: StopReason) -> &'static str {
    match r {
        StopReason::Limit | StopReason::Context | StopReason::Deadline => "length",
        StopReason::Eos | StopReason::Cancelled => "stop",
    }
}

/// How a chat-style reply ended, before each route words it its own way.
pub(super) enum Ending {
    Length,
    ToolCalls,
    StopString,
    Natural,
}

/// A stop string that matched ended the reply even when the limit was reached too; a reply that called tools ended on them.
pub(super) fn ending(d: &Done, matched: bool, calls: usize) -> Ending {
    if !matched && finish_reason(d.stop) == "length" {
        Ending::Length
    } else if calls > 0 {
        Ending::ToolCalls
    } else if matched {
        Ending::StopString
    } else {
        Ending::Natural
    }
}

pub(super) fn usage(d: &Done) -> Value {
    json!({"prompt_tokens": d.prompt_n, "completion_tokens": d.predicted_n, "total_tokens": d.prompt_n + d.predicted_n})
}

/// What every chunk of a `/v1/completions` reply says: its id, when it was made and the model.
#[derive(Clone)]
struct OaiHead {
    id: String,
    created: u64,
    model: String,
}

impl OaiHead {
    fn chunk(&self, text: &str, finish: Option<&str>) -> Value {
        oai_chunk(&self.id, self.created, &self.model, text, finish)
    }

    /// The final chunk of a reply that does not stream: all its text, why it ended, the usage and llama-server's `timings`
    /// (and `logprobs` when asked for).
    fn whole(&self, text: &str, finish: &str, d: &Done, logprobs: Option<Value>) -> Value {
        let mut v = self.chunk(text, Some(finish));
        v["usage"] = llama_compat::oai_usage(d);
        if let Some(l) = logprobs {
            v["choices"][0]["logprobs"] = l;
        }
        llama_compat::with_timings(v, d)
    }
}

fn dropped() -> ApiError {
    ApiError::new(500, "server_error", "the executor dropped the request")
}

/// The reply of a request that does not stream: its text up to a stop string or the end.
async fn oai_collect(mut pump: Pump, mut stop: StopFilter, head: &OaiHead, top: Option<usize>) -> Result<Value, ApiError> {
    let (mut text, mut n, mut entries) = (String::new(), 0usize, Vec::new());
    loop {
        match pump.next().await {
            Some(Item::Token { text: t, probs, .. }) => {
                n += 1;
                entries.extend(top.zip(probs).map(|(top, p)| llama_compat::logprob_entry(&p, top)));
                let (shown, hit) = stop.push(&t);
                text.push_str(&shown);
                if hit.is_some() {
                    let Some(mut d) = pump.cancel_and_drain().await else { return Err(dropped()) };
                    d.predicted_n = n;
                    return Ok(head.whole(&text, "stop", &d, top.map(|_| llama_compat::logprobs_json(entries))));
                }
            }
            Some(Item::Done(d)) => {
                text.push_str(&stop.finish());
                return Ok(head.whole(&text, finish_reason(d.stop), &d, top.map(|_| llama_compat::logprobs_json(entries))));
            }
            Some(Item::Error(e)) => return Err(e.into()),
            None => return Err(dropped()),
        }
    }
}

/// The events that end a stream on an error: the error, then `[DONE]`.
pub(super) fn stream_error(out: &mut Vec<Event>, e: &EngineError) {
    out.push(sse_data(&error_json(e)));
    out.push(Event::default().data("[DONE]"));
}

/// A streamed token: its chunk (text held back as a possible stop string waits); the end of the reply when a stop string
/// matched (Err: the executor dropped the request, the stream ends on that error).
async fn oai_stream_token(pump: &mut Pump, stop: &mut StopFilter, n: &mut usize, head: &OaiHead, text: &str, out: &mut Vec<Event>) -> Result<Option<(Done, &'static str)>, ()> {
    *n += 1;
    let (shown, hit) = stop.push(text);
    if stop.stops().is_empty() || !shown.is_empty() {
        out.push(sse_data(&head.chunk(&shown, None)));
    }
    if hit.is_none() {
        return Ok(None);
    }
    match pump.cancel_and_drain().await {
        Some(mut d) => {
            d.predicted_n = *n;
            Ok(Some((d, "stop")))
        }
        None => {
            stream_error(out, &EngineError::new(500, "server_error", "the executor dropped the request"));
            Err(())
        }
    }
}

/// The last events of a stream: a chunk with the finish reason, the usage chunk when asked for, then `[DONE]`.
fn oai_stream_end(out: &mut Vec<Event>, head: &OaiHead, d: &Done, reason: &str, include_usage: bool) {
    let last = head.chunk("", Some(reason));
    // llama-server's `timings` ride on the last chunk of the stream
    out.push(sse_data(&if include_usage { last } else { llama_compat::with_timings(last, d) }));
    if include_usage {
        let usage_chunk = llama_compat::stamp(json!({"id": head.id, "object": "text_completion", "created": head.created, "model": head.model,
                                                     "choices": [], "usage": llama_compat::oai_usage(d)}));
        out.push(sse_data(&llama_compat::with_timings(usage_chunk, d)));
    }
    out.push(Event::default().data("[DONE]"));
}

/// The events for the next item of a streamed reply, and whether the stream goes on after them; None: the executor's
/// stream ended.
async fn oai_stream_step(pump: &mut Pump, stop: &mut StopFilter, n: &mut usize, head: &OaiHead, include_usage: bool) -> Option<(Vec<Event>, bool)> {
    let mut out: Vec<Event> = Vec::new();
    let final_done = match pump.next().await {
        Some(Item::Token { text, .. }) => match oai_stream_token(pump, stop, n, head, &text, &mut out).await {
            Ok(done) => done,
            Err(()) => return Some((out, false)),
        },
        Some(Item::Done(d)) => {
            let tail = stop.finish();
            if !tail.is_empty() {
                out.push(sse_data(&head.chunk(&tail, None)));
            }
            let reason = finish_reason(d.stop);
            Some((d, reason))
        }
        Some(Item::Error(e)) => {
            stream_error(&mut out, &e);
            return Some((out, false));
        }
        None => return None,
    };
    let Some((d, reason)) = final_done else { return Some((out, true)) };
    oai_stream_end(&mut out, head, &d, reason, include_usage);
    Some((out, false))
}

pub(super) async fn oai_completions(State(st): State<St>, body: Bytes) -> Result<Response, ApiError> {
    let r: OaiReq = parse_body_checked(&body, false, |v| reject_fields(v, &["json_schema", "logprobs"]))?;
    if r.n.unwrap_or(1) != 1 {
        return Err(ApiError::invalid("n must be 1"));
    }
    let max_tokens = r.max_tokens.unwrap_or(-1); // until the model stops, as chat (Ben 2026-11-09; was OpenAI's 16)
    let stops = stops_of(&r.stop);
    let g = GenReq { n_predict: Some(max_tokens), temperature: r.temperature, ignore_eos: r.ignore_eos, grammar: r.grammar.clone(), json_schema: r.json_schema.clone(), ..Default::default() };
    let logprobs = llama_compat::text_logprobs(r.logprobs.as_ref(), &st, r.stream)?;
    let mut params = g.params()?;
    params.n_probs = logprobs.map_or(0, |l| l.n_probs());
    let ids = oai_prompt(&st, &r.prompt).await?;
    if ids.is_empty() {
        return Err(ApiError::invalid("prompt must not be empty"));
    }
    let model = r.model.clone().unwrap_or_else(|| llama_compat::default_model(&st));
    let pump = submit(&st, Gen { agent: None, ids, params, lane: r.lane, deadline_ms: r.deadline_ms }).await?;
    let req_id = pump.guard.id;
    let id = format!("cmpl-{req_id}");
    let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let include_usage = r.stream_options.map(|o| o.include_usage).unwrap_or(false);
    let stop = StopFilter::new(stops);
    let head = OaiHead { id, created, model };
    if !r.stream {
        let v = oai_collect(pump, stop, &head, logprobs.map(|l| l.top)).await?;
        return Ok(with_request_id(Json(v).into_response(), req_id));
    }
    Ok(with_request_id(oai_stream(pump, stop, head, include_usage), req_id))
}

/// Streaming: token chunks, a final chunk with finish_reason, an optional usage chunk, then [DONE].
fn oai_stream(pump: Pump, stop: StopFilter, head: OaiHead, include_usage: bool) -> Response {
    let s = stream::unfold(Some((pump, stop, 0usize)), move |state| {
        let head = head.clone();
        async move {
            let (mut pump, mut stop, mut n) = state?;
            let (out, more) = oai_stream_step(&mut pump, &mut stop, &mut n, &head, include_usage).await?;
            Some((stream::iter(out.into_iter().map(Ok::<_, Infallible>)), more.then_some((pump, stop, n))))
        }
    })
    .flatten();
    Sse::new(s).into_response()
}
