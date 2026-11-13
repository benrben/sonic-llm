//! `POST /completion` (llama-server wire) and `DELETE /v1/requests/{id}`.

use super::*;

struct LlamaWire {
    n: usize,
    prompt_n: usize,
    return_tokens: bool,
    stream: bool,
    model: String,
    params: SamplingParams,
    /// (agent id) on turn responses: the final chunk also carries `agent_id` and `context_tokens`.
    agent: Option<String>,
    /// `stop` strings: the filter that holds text back, and the one that matched.
    stop: StopFilter,
    hit: Option<String>,
}

pub(super) fn stop_type(r: StopReason) -> &'static str {
    match r {
        StopReason::Limit | StopReason::Context => "limit",
        StopReason::Eos => "eos",
        StopReason::Cancelled => "cancelled",
        StopReason::Deadline => "deadline",
    }
}

pub(super) fn per(ms: f64, n: usize) -> (f64, f64) {
    let per_tok = if n > 0 { ms / n as f64 } else { 0.0 };
    let per_s = if ms > 0.0 { n as f64 * 1000.0 / ms } else { 0.0 };
    (per_tok, per_s)
}

/// Timings measured by the executor at completion (rule 008). As in llama-server, `cache_n` counts the prompt tokens
/// served from the cache (an agent's context before the turn, plus shared prefix pages) and `prompt_n` the prompt tokens
/// this request actually prefilled.
pub fn timings_json(d: &Done) -> Value {
    let processed = d.prompt_n.saturating_sub(d.shared_n);
    let (pt, ps) = per(d.prompt_ms, processed);
    let (gt, gs) = per(d.predicted_ms, d.predicted_n);
    json!({
        "cache_n": d.cached_n + d.shared_n,
        "prompt_n": processed, "prompt_ms": d.prompt_ms, "prompt_per_token_ms": pt, "prompt_per_second": ps,
        "predicted_n": d.predicted_n, "predicted_ms": d.predicted_ms, "predicted_per_token_ms": gt, "predicted_per_second": gs,
    })
}

/// llama-server's `generation_settings`, with the same keys. The sampler fields sllm-server does not implement are
/// reported at llama-server's defaults and ignored (phase 1 is greedy).
pub(super) fn generation_settings(p: &SamplingParams, stream: bool) -> Value {
    json!({
        "seed": 4294967295u64, "temperature": p.temperature, "dynatemp_range": 0.0, "dynatemp_exponent": 1.0,
        "top_k": 40, "top_p": 0.95, "min_p": 0.05, "top_n_sigma": -1.0, "xtc_probability": 0.0, "xtc_threshold": 0.1,
        "typical_p": 1.0, "repeat_last_n": 64, "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "dry_multiplier": 0.0, "dry_base": 1.75, "dry_allowed_length": 2, "dry_penalty_last_n": 64,
        "dry_sequence_breakers": ["\n", ":", "\"", "*"], "mirostat": 0, "mirostat_tau": 5.0, "mirostat_eta": 0.1,
        "adaptive_target": -1.0, "adaptive_decay": 0.9, "stop": [], "max_tokens": p.n_predict, "n_predict": p.n_predict,
        "n_keep": 0, "n_discard": 0, "ignore_eos": p.ignore_eos, "stream": stream, "logit_bias": [], "n_probs": p.n_probs,
        "min_keep": 0, "grammar": p.grammar.as_deref().unwrap_or(""), "grammar_lazy": false, "grammar_triggers": [], "preserved_tokens": [],
        "chat_format": "Content-only", "reasoning_format": "deepseek", "reasoning_in_content": false,
        "generation_prompt": "", "samplers": ["penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"],
        "speculative.types": "none", "timings_per_token": false, "post_sampling_probs": false, "backend_sampling": false,
        "lora": [],
    })
}

impl LlamaWire {
    fn token_chunk(&mut self, id: i32, text: String, probs: Option<&TokProbs>) -> Value {
        self.n += 1;
        let mut v = json!({
            // as in llama-server, a streamed chunk always carries the id of its token (`return_tokens` is for the final chunk)
            "index": 0, "content": text, "tokens": [id],
            "stop": false, "id_slot": -1, "tokens_predicted": self.n, "tokens_evaluated": self.prompt_n,
        });
        if self.params.n_probs > 0 {
            v["completion_probabilities"] = Value::Array(probs.map(prob_json).into_iter().collect());
        }
        v
    }

    /// The final chunk. `content`/`tokens` are empty when streaming and the whole output when not (llama-server).
    ///
    /// With `n_probs`, `probs` is `completion_probabilities`: one entry per token of the output (the whole output when not streaming).
    fn final_chunk(&self, d: &Done, content: String, tokens: Vec<i32>, probs: Vec<Value>) -> Value {
        let mut v = json!({
            "index": 0, "content": content, "tokens": tokens, "id_slot": 0, "stop": true, "model": self.model,
            "tokens_predicted": d.predicted_n, "tokens_evaluated": d.prompt_n,
            "generation_settings": generation_settings(&self.params, self.stream),
            "prompt": "", "has_new_line": false, "truncated": false, "stopping_word": self.hit.as_deref().unwrap_or(""),
            "stop_type": if self.hit.is_some() { "word" } else { stop_type(d.stop) },
            "tokens_cached": d.cached_n + d.prompt_n + d.predicted_n.saturating_sub(1),
            "timings": timings_json(d),
        });
        v["generation_settings"]["stop"] = json!(self.stop.stops());
        if self.params.n_probs > 0 {
            v["completion_probabilities"] = Value::Array(probs);
        }
        if let Some(a) = &self.agent {
            let o = v.as_object_mut().unwrap();
            o.insert("agent_id".into(), json!(a));
            o.insert("context_tokens".into(), json!(d.context_tokens));
            if let Some(w) = &d.wake {
                o.insert("wake".into(), json!(w));
            }
            if d.rollovers > 0 {
                o.insert("rollovers".into(), json!(d.rollovers));
                o.insert("rollover_ms".into(), json!(d.rollover_ms));
            }
        }
        v
    }
}

impl LlamaWire {
    /// The whole reply of a request that does not stream: the final chunk with all its text (its tokens when asked for).
    fn collected(&self, d: &Done, content: String, toks: Vec<i32>, probs: Vec<Value>) -> Response {
        let toks = if self.return_tokens { toks } else { vec![] };
        Json(self.final_chunk(d, content, toks, probs)).into_response()
    }
}

/// One `completion_probabilities` entry (llama-server's format): the token, its log-probability and the best tokens'.
fn prob_json(p: &TokProbs) -> Value {
    let one = |t: &TokProb| json!({"id": t.id, "token": t.token, "bytes": t.bytes, "logprob": t.logprob});
    let mut v = one(&p.chosen);
    v["top_logprobs"] = Value::Array(p.top.iter().map(one).collect());
    v
}

pub(super) fn sse_data(v: &Value) -> Event {
    Event::default().data(v.to_string())
}

type SseOut = Sse<std::pin::Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>>;

fn llama_stream(pump: Pump, wire: LlamaWire) -> SseOut {
    let s = stream::unfold(Some((pump, wire)), |state| async move {
        let (mut pump, mut wire) = state?;
        match pump.next().await? {
            Item::Token { id, text, probs } => {
                let (shown, hit) = wire.stop.push(&text);
                let Some(word) = hit else {
                    let v = wire.token_chunk(id, shown, probs.as_deref());
                    return Some((Ok(sse_data(&v)), Some((pump, wire))));
                };
                // a stop string matched: the request is cut here, and the final chunk carries the text before it
                wire.n += 1;
                wire.hit = Some(word);
                match pump.cancel_and_drain().await {
                    Some(mut d) => {
                        d.predicted_n = wire.n;
                        let v = wire.final_chunk(&d, shown, if wire.return_tokens { vec![id] } else { vec![] }, probs.iter().map(|p| prob_json(p)).collect());
                        Some((Ok(sse_data(&v)), None))
                    }
                    None => Some((Ok(sse_data(&error_json(&EngineError::new(500, "server_error", "the executor dropped the request")))), None)),
                }
            }
            Item::Done(d) => {
                // text held back as a possible stop string was not one
                let tail = wire.stop.finish();
                let v = wire.final_chunk(&d, tail, vec![], vec![]);
                Some((Ok(sse_data(&v)), None))
            }
            Item::Error(e) => Some((Ok(sse_data(&error_json(&e))), None)),
        }
    });
    Sse::new(s.boxed() as std::pin::Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>)
}

async fn llama_collect(mut pump: Pump, mut wire: LlamaWire) -> Result<Response, ApiError> {
    let (mut content, mut toks, mut probs) = (String::new(), Vec::new(), Vec::new());
    loop {
        match pump.next().await {
            Some(Item::Token { id, text, probs: p }) => {
                let (shown, hit) = wire.stop.push(&text);
                content.push_str(&shown);
                toks.push(id);
                // (a stop string's token stays in both lists, as in `tokens`)
                probs.extend(p.as_deref().map(prob_json));
                if let Some(word) = hit {
                    wire.hit = Some(word);
                    let d = pump.stop_after(toks.len()).await?;
                    return Ok(wire.collected(&d, content, toks, probs));
                }
            }
            Some(Item::Done(d)) => {
                content.push_str(&wire.stop.finish());
                return Ok(wire.collected(&d, content, toks, probs));
            }
            Some(Item::Error(e)) => return Err(e.into()),
            None => return Err(ApiError::new(500, "server_error", "the executor dropped the request")),
        }
    }
}

async fn resolve_prompt(st: &St, p: Option<Prompt>) -> Result<Vec<i32>, ApiError> {
    match p {
        Some(Prompt::Ids(v)) => Ok(v),
        // llama-server tokenizes completion prompts with add_special and parse_special on.
        Some(Prompt::Text(t)) => encode(st, t, true, true).await,
        None => Err(ApiError::new(400, "invalid_request_error", "missing field `prompt`")),
    }
}

pub(super) async fn run_gen(st: &St, req: GenReq, agent: Option<String>, default_stream: bool) -> Result<Response, ApiError> {
    let params = req.params()?;
    let stops = stops_of(&req.stop);
    if agent.is_some() && !stops.is_empty() {
        return Err(refuse_stop_on_agents());
    }
    let stream = req.stream.unwrap_or(default_stream);
    let return_tokens = req.return_tokens.unwrap_or(false);
    let ids = resolve_prompt(st, req.prompt).await?;
    if agent.is_none() && ids.is_empty() {
        return Err(ApiError::invalid("prompt must not be empty"));
    }
    let model = llama_compat::default_model(st);
    let g = Gen { agent: agent.clone(), ids, params: params.clone(), lane: req.lane, deadline_ms: req.deadline_ms };
    let pump = submit(st, g).await?;
    let id = pump.guard.id;
    let wire = LlamaWire { n: 0, prompt_n: pump.prompt_n, return_tokens, stream, model, params, agent, stop: StopFilter::new(stops), hit: None };
    let r = if stream { Ok(llama_stream(pump, wire).into_response()) } else { llama_collect(pump, wire).await };
    r.map(|r| with_request_id(r, id))
}

/// A turn keeps what it generated in the agent's context, so cutting it at a stop string would leave the cut-off tokens
/// there: `stop` is refused on agent routes.
pub(super) fn refuse_stop_on_agents() -> ApiError {
    ApiError::new(400, "invalid_request_error", "unsupported field `stop` on agent turns and events: the tokens after a stop string would stay in the agent's context (see /caps completion.stop)")
}

/// The request's id, for `DELETE /v1/requests/{id}` (a header: the chunk keys stay llama-server's).
pub(super) fn with_request_id(mut r: Response, id: ReqId) -> Response {
    r.headers_mut().insert("x-request-id", HeaderValue::from(id));
    r
}

pub(super) async fn request_cancel(State(st): State<St>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    let n: ReqId = id.trim_start_matches("cmpl-").parse().map_err(|_| ApiError::new(404, "not_found_error", format!("no request {id}")))?;
    let cancelled = ask(&st, |reply| Command::CancelReq { id: n, reply }).await?;
    if !cancelled {
        return Err(ApiError::new(404, "not_found_error", format!("no queued or running request {id}")));
    }
    Ok(Json(json!({"id": n, "object": "request", "cancelled": true})))
}

/// Whether the engine is the next engine (`--engine next`), the only one with token probabilities.
pub(super) fn is_next_engine(st: &St) -> bool {
    st.shared.caps.get().and_then(|c| c.get("engine")).and_then(Value::as_str) == Some("next")
}

pub(super) async fn completion(State(st): State<St>, body: Bytes) -> Result<Response, ApiError> {
    // `n_probs` is let through here, on the next engine only (every other route and engine refuses a non-zero value)
    let except: &[&str] = if is_next_engine(&st) { &["n_probs", "json_schema"] } else { &["json_schema"] };
    let req: GenReq = parse_body_checked(&body, false, |v| reject_fields(v, except))?;
    run_gen(&st, req, None, false).await
}
