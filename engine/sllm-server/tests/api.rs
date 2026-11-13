//! Endpoint tests: determinism, n_predict / ignore_eos, SSE timing, cancel mid-stream, agents lifecycle, OpenAI wire,
//! 404 / 422 / 400 / 409 / 503, /caps and /metrics.

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::fake::FakeConfig;
use sllm_server::tokenizer::Tokenizer;
use sllm_server::ServerConfig;
use std::time::{Duration, Instant};

fn ids(a: i32, n: i32) -> Vec<i32> {
    (a..a + n).collect()
}

fn toks(v: &Value) -> Vec<i64> {
    v["tokens"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}

fn err_of(v: &Value) -> (i64, String, String) {
    let e = &v["error"];
    (e["code"].as_i64().unwrap(), e["type"].as_str().unwrap().to_string(), e["message"].as_str().unwrap().to_string())
}

// -- determinism, n_predict, ignore_eos ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn fake_tokens_are_deterministic_and_depend_on_the_prompt_and_seed() {
    let a = spawn(fast());
    let b = spawn(fast());
    let c = spawn(FakeConfig { seed: 7, ..fast() });
    let p = ids(10, 20);
    let (x1, x2, y, z) = (a.complete(&p, 16, json!({})).await, a.complete(&p, 16, json!({})).await, b.complete(&p, 16, json!({})).await, c.complete(&p, 16, json!({})).await);
    assert_eq!(toks(&x1), toks(&x2));
    assert_eq!(toks(&x1), toks(&y), "same seed and prompt, different server");
    assert_ne!(toks(&x1), toks(&z), "different seed");
    let other = a.complete(&ids(11, 20), 16, json!({})).await;
    assert_ne!(toks(&x1), toks(&other), "different prompt");
    assert!(toks(&x1).iter().all(|&t| (0..151_936).contains(&t)));
}

#[tokio::test(flavor = "multi_thread")]
async fn n_predict_is_honoured_exactly() {
    let srv = spawn(fast());
    for n in [0, 1, 2, 5, 37] {
        let v = srv.complete(&ids(1, 4), n, json!({})).await;
        assert_eq!(v["tokens_predicted"], n);
        assert_eq!(toks(&v).len(), n as usize);
        assert_eq!(v["stop_type"], "limit");
        assert_eq!(v["timings"]["predicted_n"], n);
    }
    // a prefix of a longer run
    let long = srv.complete(&ids(1, 4), 10, json!({})).await;
    let short = srv.complete(&ids(1, 4), 4, json!({})).await;
    assert_eq!(toks(&long)[..4], toks(&short)[..]);
}

#[tokio::test(flavor = "multi_thread")]
async fn ignore_eos_controls_where_generation_stops() {
    let srv = spawn(FakeConfig { natural_len: Some(5), ..fast() });
    // the fake model wants to stop after 5 tokens
    let stops = srv.complete(&ids(1, 4), 50, json!({"ignore_eos": false})).await;
    assert_eq!(stops["stop_type"], "eos");
    assert_eq!(stops["tokens_predicted"], 5);
    let goes_on = srv.complete(&ids(1, 4), 50, json!({"ignore_eos": true})).await;
    assert_eq!(goes_on["stop_type"], "limit");
    assert_eq!(goes_on["tokens_predicted"], 50);
    assert_eq!(toks(&goes_on)[..5], toks(&stops)[..], "same tokens up to the stop");
    // n_predict below the natural length: limit wins
    let short = srv.complete(&ids(1, 4), 3, json!({"ignore_eos": false})).await;
    assert_eq!((short["stop_type"].as_str().unwrap(), short["tokens_predicted"].as_u64().unwrap()), ("limit", 3));
    // the EOS id is never emitted as a token
    assert!(!toks(&goes_on).contains(&151_645));
    // default natural length (from the prompt) is between 8 and 64
    let d = spawn(fast());
    let v = d.complete(&ids(3, 9), 500, json!({"ignore_eos": false})).await;
    assert_eq!(v["stop_type"], "eos");
    assert!((8..64).contains(&v["tokens_predicted"].as_u64().unwrap()));
}

#[tokio::test(flavor = "multi_thread")]
async fn context_limit_stops_with_limit() {
    let srv = spawn(FakeConfig { n_ctx: 32, ..fast() });
    let v = srv.complete(&ids(1, 20), 100, json!({})).await;
    assert_eq!(v["tokens_predicted"], 12);
    assert_eq!(v["stop_type"], "limit");
    let r = srv.post("/completion", &json!({"prompt": ids(1, 40), "n_predict": 4})).await;
    assert_eq!(r.status(), 400);
    let v: Value = r.json().await.unwrap();
    assert_eq!(err_of(&v).1, "exceed_context_size_error");
    // llama-server's fields of this error type
    assert_eq!((v["error"]["n_prompt_tokens"].clone(), v["error"]["n_ctx"].clone()), (json!(40), json!(32)));
}

/// An agent at its cell limit: a turn that would go past it is a 400 (llama-server's context error) that changes nothing,
/// and another agent's turns are unaffected (the fix-cellcap task: a full agent must never take the server down).
#[tokio::test(flavor = "multi_thread")]
async fn a_full_agent_is_refused_alone_and_the_others_go_on() {
    let srv = spawn(FakeConfig { n_ctx: 64, ..fast() });
    let new_agent = || async { srv.post("/v1/agents", &json!({})).await.json::<Value>().await.unwrap()["id"].as_str().unwrap().to_string() };
    let (full, other) = (new_agent().await, new_agent().await);
    // fill `full` to the brim: 40 + 30 asks for more than the 64 cells, so the turn stops at the context limit
    let v: Value = srv.post(&format!("/v1/agents/{full}/turn"), &json!({"prompt": ids(1, 40), "n_predict": 30, "stream": false, "ignore_eos": true})).await.json().await.unwrap();
    assert_eq!((v["tokens_predicted"].clone(), v["stop_type"].clone()), (json!(24), json!("limit")));
    let ctx = srv.get_json(&format!("/v1/agents/{full}")).await["context_tokens"].as_u64().unwrap();
    assert_eq!(ctx, 64);
    // a further turn (a one-token chunk or a long one) is refused, and the agent is unchanged and idle
    for n in [1, 32] {
        let r = srv.post(&format!("/v1/agents/{full}/turn"), &json!({"prompt": ids(100, n), "n_predict": 4, "stream": false})).await;
        assert_eq!(r.status(), 400, "turn of {n} tokens on a full agent");
        let e: Value = r.json().await.unwrap();
        assert_eq!(err_of(&e).1, "exceed_context_size_error");
        assert_eq!((e["error"]["n_prompt_tokens"].clone(), e["error"]["n_ctx"].clone()), (json!(64 + n), json!(64)));
        let info = srv.get_json(&format!("/v1/agents/{full}")).await;
        assert_eq!((info["context_tokens"].clone(), info["busy"].clone()), (json!(64), json!(false)));
    }
    // the other agent and plain completions go on as before
    let v: Value = srv.post(&format!("/v1/agents/{other}/turn"), &json!({"prompt": ids(1, 8), "n_predict": 5, "stream": false, "ignore_eos": true})).await.json().await.unwrap();
    assert_eq!(v["tokens_predicted"], 5);
    assert_eq!(srv.complete(&ids(1, 8), 5, json!({})).await["tokens_predicted"], 5);
    assert_eq!(srv.metrics().await["agents"]["active"], 2);
}

// -- streaming ---------------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn sse_chunks_arrive_as_tokens_are_produced() {
    let srv = spawn(FakeConfig { token_ms: 40.0, ..fast() });
    let mut r = srv.post("/completion", &json!({"prompt": ids(1, 4), "n_predict": 6, "stream": true, "return_tokens": true})).await;
    let t0 = Instant::now();
    let mut seen: Vec<(Duration, usize)> = vec![];
    let mut buf = String::new();
    while let Some(c) = r.chunk().await.unwrap() {
        buf.push_str(std::str::from_utf8(&c).unwrap());
        seen.push((t0.elapsed(), buf.matches("\n\n").count()));
    }
    let events = sse_events(&buf);
    assert_eq!(events.len(), 7);
    let first = seen.iter().find(|(_, n)| *n >= 1).unwrap().0;
    let last = seen.last().unwrap().0;
    assert!(first < Duration::from_millis(120), "first chunk after {first:?}: the stream is buffered");
    assert!(last - first > Duration::from_millis(100), "chunks must be spread over the decode: {first:?}..{last:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_mid_stream_when_the_client_disconnects() {
    let srv = spawn(FakeConfig { token_ms: 10.0, ..fast() });
    let mut r = srv.post("/completion", &json!({"prompt": ids(1, 4), "n_predict": 100000, "stream": true, "ignore_eos": true})).await;
    let mut got = String::new();
    while got.matches("\n\n").count() < 3 {
        got.push_str(std::str::from_utf8(&r.chunk().await.unwrap().expect("stream ended early")).unwrap());
    }
    assert_eq!(srv.metrics().await["active_requests"], 1);
    drop(r); // the client goes away mid-stream
    let m = srv.wait_metrics("request cancelled", |m| m["requests"]["cancelled"] == 1 && m["active_requests"] == 0).await;
    assert!(m["tokens_generated"].as_u64().unwrap() < 200, "generation went on after the disconnect: {m}");
    // the server is healthy and serves the next request
    let v = srv.complete(&ids(1, 4), 3, json!({})).await;
    assert_eq!(v["tokens_predicted"], 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn non_streaming_request_dropped_mid_flight_is_cancelled() {
    let srv = spawn(FakeConfig { token_ms: 10.0, ..fast() });
    let c = reqwest::Client::builder().timeout(Duration::from_millis(100)).no_proxy().build().unwrap();
    let res = c.post(srv.url("/completion")).json(&json!({"prompt": ids(1, 4), "n_predict": 100000, "ignore_eos": true})).send().await;
    assert!(res.is_err(), "the client times out");
    srv.wait_metrics("request cancelled", |m| m["requests"]["cancelled"] == 1 && m["active_requests"] == 0).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_requests_share_steps_and_keep_their_own_tokens() {
    let srv = spawn(FakeConfig { token_ms: 5.0, ..fast() });
    let solo: Vec<Vec<i64>> = {
        let mut v = vec![];
        for i in 0..4 {
            v.push(toks(&srv.complete(&ids(i * 10, 6), 12, json!({})).await));
        }
        v
    };
    let steps_before = srv.metrics().await["steps"].as_u64().unwrap();
    let futs = (0..4).map(|i| srv.complete_owned(ids(i * 10, 6), 12));
    let both = futures_util::future::join_all(futs).await;
    for (i, v) in both.iter().enumerate() {
        assert_eq!(toks(v), solo[i], "request {i} under concurrency");
    }
    let steps = srv.metrics().await["steps"].as_u64().unwrap() - steps_before;
    assert!(steps < 4 * 12, "four concurrent requests should share decode steps, took {steps}");
}

impl Srv {
    async fn complete_owned(&self, p: Vec<i32>, n: i64) -> Value {
        self.complete(&p, n, json!({})).await
    }
}

// -- agents ------------------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn agent_lifecycle_create_turn_turn_delete() {
    let srv = spawn(fast());
    let system = ids(500, 7);
    let r = srv.post("/v1/agents", &json!({"lane": "background", "system_prompt_ids": system})).await;
    assert_eq!(r.status(), 201);
    let a: Value = r.json().await.unwrap();
    let id = a["id"].as_str().unwrap().to_string();
    assert!(id.starts_with("ag_"));
    assert_eq!((a["lane"].as_str().unwrap(), a["context_tokens"].as_u64().unwrap(), a["turns"].as_u64().unwrap(), a["busy"].as_bool().unwrap()), ("background", 7, 0, false));

    // GET and list
    assert_eq!(srv.get_json(&format!("/v1/agents/{id}")).await["context_tokens"], 7);
    let list = srv.get_json("/v1/agents").await;
    assert_eq!(list["data"].as_array().unwrap().len(), 1);
    assert_eq!(srv.metrics().await["agents"]["active"], 1);

    // turn 1 streams in the /completion chunk format, and equals a plain completion over system + turn
    let t1 = ids(900, 5);
    let (raw, chunks) = srv.stream(&format!("/v1/agents/{id}/turn"), &json!({"prompt": t1, "n_predict": 6, "ignore_eos": true, "return_tokens": true})).await;
    assert!(!raw.contains("[DONE]"));
    assert_eq!(chunks.len(), 7);
    let fin = chunks.last().unwrap();
    assert_eq!((fin["stop"].clone(), fin["agent_id"].clone(), fin["context_tokens"].clone()), (json!(true), json!(id), json!(7 + 5 + 6)));
    assert_eq!(fin["tokens_evaluated"], 5, "only the turn's ids are evaluated");
    assert_eq!(fin["timings"]["cache_n"], 7, "the system prompt is reused");
    let g1 = tokens_of(&chunks);
    let mut full: Vec<i32> = system.clone();
    full.extend(&t1);
    let plain = srv.complete(&full, 6, json!({})).await;
    assert_eq!(g1, toks(&plain), "turn 1 == completion over the same context");

    // turn 2 (non-streaming this time) continues from system + t1 + generated
    let t2 = ids(300, 3);
    let r = srv.post(&format!("/v1/agents/{id}/turn"), &json!({"prompt": t2, "n_predict": 4, "stream": false, "ignore_eos": true, "return_tokens": true})).await;
    assert_eq!(r.status(), 200);
    let v2: Value = r.json().await.unwrap();
    assert_eq!(v2["context_tokens"], 7 + 5 + 6 + 3 + 4);
    full.extend(g1.iter().map(|&x| x as i32));
    full.extend(&t2);
    let plain2 = srv.complete(&full, 4, json!({})).await;
    assert_eq!(toks(&v2), toks(&plain2), "turn 2 == completion over system + t1 + g1 + t2");
    let info = srv.get_json(&format!("/v1/agents/{id}")).await;
    assert_eq!((info["turns"].clone(), info["context_tokens"].clone(), info["busy"].clone()), (json!(2), json!(25), json!(false)));

    // cancel while idle: nothing to cancel
    let c: Value = srv.post(&format!("/v1/agents/{id}/cancel"), &json!({})).await.json().await.unwrap();
    assert_eq!(c["cancelled"], false);

    // delete frees it; everything on it is 404 afterwards
    let d = srv.http.delete(srv.url(&format!("/v1/agents/{id}"))).send().await.unwrap();
    assert_eq!(d.status(), 200);
    let dv: Value = d.json().await.unwrap();
    assert_eq!((dv["deleted"].clone(), dv["context_tokens_freed"].clone()), (json!(true), json!(25)));
    assert_eq!(srv.get(&format!("/v1/agents/{id}")).await.status(), 404);
    assert_eq!(srv.post(&format!("/v1/agents/{id}/turn"), &json!({"prompt": [1], "n_predict": 1})).await.status(), 404);
    assert_eq!(srv.post(&format!("/v1/agents/{id}/cancel"), &json!({})).await.status(), 404);
    assert_eq!(srv.http.delete(srv.url(&format!("/v1/agents/{id}"))).send().await.unwrap().status(), 404);
    assert_eq!(srv.metrics().await["agents"]["active"], 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn an_event_runs_as_a_turn_framed_as_a_tool_result() {
    let srv = spawn(fast());
    let a: Value = srv.post("/v1/agents", &json!({"system_prompt_ids": ids(500, 7)})).await.json().await.unwrap();
    let id = a["id"].as_str().unwrap().to_string();
    let ev = json!({"name": "inbox", "content": "disk 91% full", "arguments": {"box": "ops"}, "call_id": "c1", "n_predict": 3, "stream": false, "ignore_eos": true});
    // an event follows a reply: none yet
    assert_eq!(srv.post(&format!("/v1/agents/{id}/event"), &ev).await.status(), 409);
    srv.post(&format!("/v1/agents/{id}/turn"), &json!({"prompt": ids(900, 5), "n_predict": 4, "stream": false, "ignore_eos": true})).await;
    // no GGUF: ChatML with Qwen's tool format; the byte tokenizer makes one token per byte
    let mut dry = ev.clone();
    dry["dry_run"] = json!(true);
    let d: Value = srv.post(&format!("/v1/agents/{id}/event"), &dry).await.json().await.unwrap();
    let text = "\n<tool_call>\n{\"name\": \"inbox\", \"arguments\": {\"box\": \"ops\"}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\ndisk 91% full\n</tool_response><|im_end|>\n<|im_start|>assistant\n";
    assert_eq!((d["framing"].as_str().unwrap(), d["text"].as_str().unwrap()), ("tool", text));
    assert_eq!(d["tokens"].as_array().unwrap().len(), text.len());
    assert_eq!(srv.get_json(&format!("/v1/agents/{id}")).await["turns"], 1, "a dry run runs nothing");
    let r = srv.post(&format!("/v1/agents/{id}/event"), &ev).await;
    assert_eq!((r.status().as_u16(), r.headers()["x-event-framing"].to_str().unwrap().to_string()), (200, "tool".to_string()));
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["context_tokens"], json!(7 + 5 + 4 + text.len() + 3));
    let bad = json!({"name": "x", "content": "y", "as": "other"});
    assert_eq!(srv.post(&format!("/v1/agents/{id}/event"), &bad).await.status(), 422);
    assert_eq!(srv.post("/v1/agents/ag_999/event", &ev).await.status(), 404);
}

#[tokio::test(flavor = "multi_thread")]
async fn agent_create_defaults_and_validation() {
    let srv = spawn(fast());
    // empty body: interactive lane, no system prompt
    let r = srv.http.post(srv.url("/v1/agents")).send().await.unwrap();
    assert_eq!(r.status(), 201);
    let a: Value = r.json().await.unwrap();
    assert_eq!((a["lane"].as_str().unwrap(), a["context_tokens"].as_u64().unwrap()), ("interactive", 0));
    // a body that does not fit the request's types is 400, as in llama-server
    assert_eq!(srv.post("/v1/agents", &json!({"lane": "vip"})).await.status(), 400);
    assert_eq!(srv.post("/v1/agents", &json!({"system_prompt_ids": "x"})).await.status(), 400);
    assert_eq!(srv.post("/v1/agents", &json!({"system_prompt_ids": [-3]})).await.status(), 422); // an invalid value, not a wrong type
    assert_eq!(srv.post_raw("/v1/agents", "{not json").await.status(), 400);
    // a turn with no ids is allowed (generate from the existing context)
    let id = a["id"].as_str().unwrap();
    let t = srv.post(&format!("/v1/agents/{id}/turn"), &json!({})).await;
    assert_eq!(t.status(), 200);
}

#[tokio::test(flavor = "multi_thread")]
async fn agent_turn_cancel_endpoint_ends_the_stream_and_busy_agent_queues() {
    let srv = spawn(FakeConfig { token_ms: 10.0, ..fast() });
    let a: Value = srv.post("/v1/agents", &json!({})).await.json().await.unwrap();
    let id = a["id"].as_str().unwrap().to_string();
    let mut r = srv.post(&format!("/v1/agents/{id}/turn"), &json!({"prompt": ids(1, 3), "n_predict": 100000, "ignore_eos": true})).await;
    let mut got = String::new();
    while got.matches("\n\n").count() < 2 {
        got.push_str(std::str::from_utf8(&r.chunk().await.unwrap().unwrap()).unwrap());
    }
    // a second turn on a busy agent waits in the agent's input queue (phase 3: no more 409 busy)
    let second = {
        let (http, url) = (srv.http.clone(), srv.url(&format!("/v1/agents/{id}/turn")));
        tokio::spawn(async move { http.post(url).json(&json!({"prompt": [1], "n_predict": 1, "stream": false})).send().await.unwrap() })
    };
    let t0 = std::time::Instant::now();
    while srv.get_json(&format!("/v1/agents/{id}")).await["queued_turns"] != 1 {
        assert!(t0.elapsed() < std::time::Duration::from_secs(5), "the second turn was not queued");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert_eq!(srv.get_json(&format!("/v1/agents/{id}")).await["busy"], true);
    // cancel: the open stream finishes with stop_type "cancelled"
    let c: Value = srv.post(&format!("/v1/agents/{id}/cancel"), &json!({})).await.json().await.unwrap();
    assert_eq!(c["cancelled"], true);
    while let Some(c) = r.chunk().await.unwrap() {
        got.push_str(std::str::from_utf8(&c).unwrap());
    }
    let chunks = sse_json(&got);
    let fin = chunks.last().unwrap();
    assert_eq!((fin["stop"].clone(), fin["stop_type"].clone()), (json!(true), json!("cancelled")));
    assert!(fin["tokens_predicted"].as_u64().unwrap() < 1000);
    // the queued turn runs once the first has ended, on the context the first left behind
    let second = second.await.unwrap();
    assert_eq!(second.status(), 200);
    let sv: Value = second.json().await.unwrap();
    assert!(sv["context_tokens"].as_u64().unwrap() > 3 + 1);
    let info = srv.get_json(&format!("/v1/agents/{id}")).await;
    assert_eq!((info["busy"].clone(), info["queued_turns"].clone(), info["state"].clone()), (json!(false), json!(0), json!("idle")));
    assert_eq!(srv.post(&format!("/v1/agents/{id}/turn"), &json!({"prompt": [1], "n_predict": 2, "stream": false})).await.status(), 200);
}

#[tokio::test(flavor = "multi_thread")]
async fn deleting_an_agent_mid_turn_cancels_the_turn() {
    let srv = spawn(FakeConfig { token_ms: 10.0, ..fast() });
    let a: Value = srv.post("/v1/agents", &json!({})).await.json().await.unwrap();
    let id = a["id"].as_str().unwrap().to_string();
    let mut r = srv.post(&format!("/v1/agents/{id}/turn"), &json!({"prompt": ids(1, 3), "n_predict": 100000, "ignore_eos": true})).await;
    let _ = r.chunk().await.unwrap().unwrap();
    assert_eq!(srv.http.delete(srv.url(&format!("/v1/agents/{id}"))).send().await.unwrap().status(), 200);
    let mut rest = String::new();
    while let Some(c) = r.chunk().await.unwrap() {
        rest.push_str(std::str::from_utf8(&c).unwrap());
    }
    assert!(rest.contains("\"stop_type\":\"cancelled\""));
    srv.wait_metrics("idle", |m| m["active_requests"] == 0).await;
}

// -- OpenAI wire -------------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn v1_completions_non_stream_and_usage() {
    let srv = spawn(fast());
    let r = srv.post("/v1/completions", &json!({"model": "m", "prompt": "hello", "max_tokens": 5, "temperature": 0, "ignore_eos": true})).await;
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["model"], "m");
    assert!(v["id"].as_str().unwrap().starts_with("cmpl-"));
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert!(v["choices"][0]["text"].as_str().unwrap().len() > 5);
    assert_eq!(v["usage"], json!({"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0}})); // bytes tokenizer: 5 bytes
    // ids as the prompt and the default max_tokens: none, the answer runs past OpenAI's 16 (Ben 2026-11-09: no cap)
    let v2: Value = srv.post("/v1/completions", &json!({"prompt": [1, 2, 3], "ignore_eos": true})).await.json().await.unwrap();
    assert!(v2["usage"]["completion_tokens"].as_u64().unwrap() > 16, "{v2}");
    // [[ids]] with one row, ["text"] with one element
    assert_eq!(srv.post("/v1/completions", &json!({"prompt": [[1, 2, 3]], "max_tokens": 1})).await.status(), 200);
    assert_eq!(srv.post("/v1/completions", &json!({"prompt": ["abc"], "max_tokens": 1})).await.status(), 200);
    // same tokens as /completion over the same ids
    let plain = srv.complete(&[104, 101, 108, 108, 111], 5, json!({})).await;
    let text = plain["content"].as_str().unwrap();
    assert_eq!(v["choices"][0]["text"], text);
}

#[tokio::test(flavor = "multi_thread")]
async fn v1_completions_stream_with_usage_chunk_and_done() {
    let srv = spawn(fast());
    let r = srv.post("/v1/completions", &json!({"prompt": "hi", "max_tokens": 3, "stream": true, "stream_options": {"include_usage": true}, "ignore_eos": true})).await;
    assert_eq!(r.status(), 200);
    assert!(r.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));
    let raw = r.text().await.unwrap();
    let ev = sse_events(&raw);
    assert_eq!(ev.last().unwrap(), "[DONE]");
    let chunks = sse_json(&raw);
    assert_eq!(chunks.len(), 3 + 1 + 1, "3 tokens, the finish chunk, the usage chunk");
    assert!(chunks[..3].iter().all(|c| c["choices"][0]["finish_reason"].is_null() && !c["choices"][0]["text"].as_str().unwrap().is_empty()));
    assert_eq!(chunks[3]["choices"][0]["finish_reason"], "length");
    assert_eq!(chunks[4]["choices"], json!([]));
    assert_eq!(chunks[4]["usage"], json!({"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5, "prompt_tokens_details": {"cached_tokens": 0}}));
    // without include_usage there is no usage chunk
    let raw = srv.post("/v1/completions", &json!({"prompt": "hi", "max_tokens": 2, "stream": true})).await.text().await.unwrap();
    assert_eq!(sse_json(&raw).len(), 3);
    assert!(!raw.contains("usage"));
}

#[tokio::test(flavor = "multi_thread")]
async fn v1_completions_validation() {
    let srv = spawn(fast());
    for (body, why) in [
        (json!({"prompt": ["a", "b"]}), "batched prompts"),
        (json!({"prompt": "x", "n": 2}), "n != 1"),
        (json!({"prompt": 5}), "prompt type"),
        (json!({"prompt": ""}), "empty prompt"),
        (json!({"prompt": "x", "max_tokens": -7}), "max_tokens"),
        (json!({"max_tokens": 3}), "missing prompt"),
    ] {
        let r = srv.post("/v1/completions", &body).await;
        // a missing field fails the body's type: 400; a value that is not valid is 422
        assert_eq!(r.status(), if why == "missing prompt" { 400 } else { 422 }, "{why}");
        assert_eq!(err_of(&r.json().await.unwrap()).1, "invalid_request_error");
    }
}

// -- 404 / 422 / 400 / 405 / 503 ---------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn not_found_and_unprocessable_and_bad_json() {
    let srv = spawn(fast());
    for p in ["/", "/v1", "/v1/agents/nope/turn/extra", "/infill", "/v1/responses"] {
        assert_eq!(srv.get(p).await.status(), 404, "{p}");
    }
    let r = srv.get("/nope").await;
    assert_eq!(r.status(), 404);
    assert_eq!(err_of(&r.json().await.unwrap()), (404, "not_found_error".into(), "File Not Found".into()));
    assert_eq!(srv.get("/v1/agents/ag_999").await.status(), 404);
    assert_eq!(srv.post("/v1/agents/ag_999/turn", &json!({"prompt": [1], "n_predict": 1})).await.status(), 404);
    assert_eq!(srv.post("/v1/agents/ag_999/cancel", &json!({})).await.status(), 404);
    assert_eq!(srv.http.delete(srv.url("/v1/agents/ag_999")).send().await.unwrap().status(), 404);
    // wrong method on a real route
    let m = srv.get("/completion").await;
    assert_eq!(m.status(), 405);
    assert_eq!(err_of(&m.json().await.unwrap()).0, 405);

    // 400 for a field missing or of the wrong type, 422 for valid JSON with an invalid value
    for (body, why) in [
        (json!({"n_predict": 4}), "missing prompt"),
        (json!({"prompt": [1, 2], "n_predict": "x"}), "n_predict type"),
        (json!({"prompt": [1, 2], "n_predict": -2}), "n_predict < -1"),
        (json!({"prompt": [1, 2], "temperature": -0.5}), "negative temperature"),
        (json!({"prompt": []}), "empty prompt"),
        (json!({"prompt": 5}), "prompt type"),
        (json!({"prompt": [1, "a"]}), "mixed prompt array"),
        (json!({"prompt": [1, 999_999_999]}), "token id out of vocabulary"),
        (json!({"prompt": [1, -4]}), "negative token id"),
    ] {
        let r = srv.post("/completion", &body).await;
        // a field missing or of the wrong type is 400 (as llama-server); a value that is not valid is 422
        let want = if matches!(why, "n_predict type" | "prompt type" | "mixed prompt array" | "missing prompt") { 400 } else { 422 };
        assert_eq!(r.status(), want, "{why}");
        let (code, kind, msg) = err_of(&r.json().await.unwrap());
        assert_eq!((code, kind.as_str()), (want as i64, "invalid_request_error"), "{why}");
        assert!(!msg.is_empty());
    }
    // 400: not JSON at all, or empty
    for body in ["{not json", "", "[1,2"] {
        let r = srv.post_raw("/completion", body).await;
        assert_eq!(r.status(), 400, "{body:?}");
        assert_eq!(err_of(&r.json().await.unwrap()).1, "invalid_request_error");
    }
    // none of the refusals left anything running
    let m = srv.metrics().await;
    assert_eq!(m["active_requests"], 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn full_command_queue_answers_503() {
    // Executor busy in a 300 ms step; the command queue holds one command.
    let srv = spawn_with(FakeConfig { token_ms: 300.0, ..fast() }, Tokenizer::Bytes, ServerConfig { port: 0, queue_cap: 1, ..Default::default() });
    let mut a = srv.post("/completion", &json!({"prompt": [1, 2], "n_predict": 3, "stream": true})).await;
    let _first = a.chunk().await.unwrap().unwrap(); // admitted; the executor is now inside the next step
    let hb = srv.http.post(srv.url("/completion")).json(&json!({"prompt": [3], "n_predict": 1})).send();
    let b = tokio::spawn(hb);
    tokio::time::sleep(Duration::from_millis(60)).await; // B sits in the queue
    let c = srv.post("/completion", &json!({"prompt": [4], "n_predict": 1})).await;
    assert_eq!(c.status(), 503);
    assert_eq!(err_of(&c.json().await.unwrap()).1, "unavailable_error");
    let _ = a.text().await;
    assert_eq!(b.await.unwrap().unwrap().status(), 200, "the queued request is served once the executor is free");
}

// -- text endpoints with the real GGUF tokenizer -----------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn gguf_tokenizer_matches_llama_servers_ids() {
    let Some(gguf) = qwen_gguf() else { return };
    let meta: Value = serde_json::from_slice(&std::fs::read(fixtures_dir().join("llama_capture_meta.json")).unwrap()).unwrap();
    let text = meta["text"].as_str().unwrap();
    let llama_ids = meta["request"]["prompt"].clone(); // llama-server's own /tokenize of this text
    let srv = spawn_with(fast(), Tokenizer::gguf(gguf), ServerConfig { port: 0, ..Default::default() });
    let t: Value = srv.post("/tokenize", &json!({"content": text})).await.json().await.unwrap();
    assert_eq!(t["tokens"], llama_ids, "same ids as llama-server /tokenize");
    // a text prompt on /v1/completions and on /completion goes through the same tokenizer
    let v: Value = srv.post("/v1/completions", &json!({"prompt": text, "max_tokens": 2})).await.json().await.unwrap();
    assert_eq!(v["usage"]["prompt_tokens"], 12);
    let c: Value = srv.post("/completion", &json!({"prompt": text, "n_predict": 2})).await.json().await.unwrap();
    assert_eq!(c["tokens_evaluated"], 12);
    // special tokens are only parsed on request
    let plain: Value = srv.post("/tokenize", &json!({"content": "<|im_start|>", "parse_special": false})).await.json().await.unwrap();
    let special: Value = srv.post("/tokenize", &json!({"content": "<|im_start|>", "parse_special": true})).await.json().await.unwrap();
    assert!(plain["tokens"].as_array().unwrap().len() > 1);
    assert_eq!(special["tokens"], json!([151644]));
}

#[tokio::test(flavor = "multi_thread")]
async fn tokenize_uses_one_cached_vocab_handle() {
    let Some(gguf) = qwen_gguf() else { return };
    let srv = spawn_with(fast(), Tokenizer::gguf(gguf), ServerConfig { port: 0, ..Default::default() });
    let t0 = Instant::now();
    for i in 0..200 {
        let t: Value = srv.post("/tokenize", &json!({"content": format!("hello world {i}")})).await.json().await.unwrap();
        assert!(t["tokens"].as_array().unwrap().len() >= 3);
    }
    // Reloading the vocabulary of a 150k-token model on every call costs tens of milliseconds each; the cached handle
    // answers 200 calls in well under a second.
    assert!(t0.elapsed() < Duration::from_secs(3), "200 /tokenize calls took {:?}", t0.elapsed());
}

#[tokio::test(flavor = "multi_thread")]
async fn byte_tokenizer_endpoint() {
    let srv = spawn(fast());
    let t: Value = srv.post("/tokenize", &json!({"content": "AB"})).await.json().await.unwrap();
    assert_eq!(t["tokens"], json!([65, 66]));
    assert_eq!(srv.post("/tokenize", &json!({})).await.status(), 400);
}

// -- /caps, /metrics, /health ------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn caps_and_metrics_documents() {
    let srv = spawn(fast());
    let caps = srv.get_json("/caps").await;
    assert_eq!(caps["engine"]["engine"], "fake");
    assert!(caps["engine"]["capabilities"].is_null(), "no device probe for the fake engine unless --probe-caps");
    assert_eq!(caps["server"]["name"], "sllm-server");
    assert_eq!(caps["server"]["ggml"], "b11669");
    assert!(!caps["server"]["commit"].as_str().unwrap().is_empty());
    assert!(caps["completion"]["cache_prompt"].as_str().unwrap().contains("prompt cache on disk"));
    assert_eq!(caps["model"]["n_vocab"], 151_936);

    srv.complete(&ids(1, 4), 5, json!({})).await;
    srv.post("/v1/agents", &json!({})).await;
    let m = srv.metrics().await;
    for k in ["uptime_s", "http_threads", "queue_depth", "queue_capacity", "active_requests", "requests", "agents", "steps", "tokens_generated",
              "prompt_tokens_evaluated", "pages_active", "pages_reserved", "scratch_bytes", "footprint_bytes", "last_step_us"] {
        assert!(m.get(k).is_some(), "metrics lacks {k}");
    }
    assert_eq!(m["steps"], 5);
    assert_eq!(m["tokens_generated"], 5);
    assert_eq!(m["prompt_tokens_evaluated"], 4);
    assert_eq!(m["requests"]["completed"], 1);
    assert_eq!(m["agents"]["created"], 1);
    assert_eq!(m["http_threads"], 2);
    assert!(m["pages_active"].is_null() && m["scratch_bytes"].is_null() && m["memory"].is_null(), "no pool and no ledger in the fake engine");
    assert!(m["footprint_bytes"].as_u64().unwrap() > 0, "the process footprint is live for every engine");
    assert_eq!(m["queue_depth"], 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn http_thread_count_is_a_setting() {
    let srv = spawn_with(fast(), Tokenizer::Bytes, ServerConfig { port: 0, http_threads: 5, ..Default::default() });
    assert_eq!(srv.metrics().await["http_threads"], 5);
}

#[tokio::test(flavor = "multi_thread")]
async fn health_is_503_until_the_engine_is_built() {
    use sllm_server::engine::Engine;
    let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
    let factory: sllm_server::executor::EngineFactory = Box::new(move || {
        go_rx.recv().unwrap();
        Ok(Box::new(sllm_server::fake::FakeEngine::new(fast())) as Box<dyn Engine>)
    });
    let run = sllm_server::start(ServerConfig { port: 0, ..Default::default() }, Tokenizer::Bytes, factory).unwrap();
    let c = reqwest::Client::builder().no_proxy().build().unwrap();
    let r = c.get(format!("{}/health", run.url())).send().await.unwrap();
    assert_eq!(r.status(), 503);
    assert_eq!(err_of(&r.json().await.unwrap()).2, "Loading model");
    let r = c.post(format!("{}/completion", run.url())).json(&json!({"prompt": [1]})).send().await.unwrap();
    assert_eq!(r.status(), 503);
    go_tx.send(()).unwrap();
    run.wait_ready(Duration::from_secs(5)).unwrap();
    assert_eq!(c.get(format!("{}/health", run.url())).send().await.unwrap().status(), 200);
}

#[tokio::test(flavor = "multi_thread")]
async fn engine_that_fails_to_build_reports_it() {
    let factory: sllm_server::executor::EngineFactory = Box::new(|| Err("no such device".to_string()));
    let run = sllm_server::start(ServerConfig { port: 0, ..Default::default() }, Tokenizer::Bytes, factory).unwrap();
    assert_eq!(run.wait_ready(Duration::from_secs(5)).unwrap_err(), "no such device");
    let r = reqwest::Client::builder().no_proxy().build().unwrap().get(format!("{}/health", run.url())).send().await.unwrap();
    assert_eq!(r.status(), 500);
}

// -- fields the server does not implement are refused, not ignored (step s1-honesty) ------------------------------

/// (field, a non-empty value) for every field that would change what the output means.
fn refused_cases() -> Vec<(&'static str, Value)> {
    vec![
        ("json_schema", json!({"type": "object"})),
        ("lora", json!([{"id": 0, "scale": 1.0}])),
        ("tools", json!([{"type": "function", "function": {"name": "f"}}])),
        ("tool_choice", json!("auto")),
        ("response_format", json!({"type": "json_object"})),
        ("logit_bias", json!({"15": -100})),
        ("logprobs", json!(5)),
        ("n_probs", json!(3)),
        ("echo", json!(true)),
        ("repeat_penalty", json!(1.3)),
        ("presence_penalty", json!(0.5)),
        ("mirostat", json!(2)),
    ]
}

/// `json_schema` is honoured on the two text routes (it becomes the grammar); agent turns and events still refuse it.
fn json_schema_is_honoured(path: &str, field: &str) -> bool {
    field == "json_schema" && (path == "/completion" || path == "/v1/completions")
}

/// The routes that generate, with a body that is valid on its own (200).
async fn generation_routes(srv: &Srv) -> Vec<(String, Value)> {
    let a: Value = srv.post("/v1/agents", &json!({})).await.json().await.unwrap();
    let id = a["id"].as_str().unwrap().to_string();
    let first = srv.post(&format!("/v1/agents/{id}/turn"), &json!({"prompt": ids(1, 4), "n_predict": 2, "stream": false})).await;
    assert_eq!(first.status(), 200);
    vec![
        ("/completion".into(), json!({"prompt": ids(1, 4), "n_predict": 2})),
        ("/v1/completions".into(), json!({"prompt": "hello", "max_tokens": 2})),
        (format!("/v1/agents/{id}/turn"), json!({"prompt": ids(1, 4), "n_predict": 2, "stream": false})),
        (format!("/v1/agents/{id}/event"), json!({"name": "check", "content": "x", "dry_run": true})), // dry_run: framed, no turn, so the agent's context does not fill up
    ]
}

#[tokio::test(flavor = "multi_thread")]
async fn unsupported_fields_are_refused_with_400_on_every_generation_route() {
    let srv = spawn(fast());
    for (path, base) in generation_routes(&srv).await {
        // negative control: the request without the field is accepted
        let r = srv.post(&path, &base).await;
        assert_eq!(r.status(), 200, "{path} without the field");
        r.bytes().await.unwrap();
        for (field, value) in refused_cases() {
            if json_schema_is_honoured(&path, field) {
                continue; // tested on its own (tests/llama_compat.rs): a schema becomes the grammar
            }
            let mut b = base.clone();
            b[field] = value;
            let r = srv.post(&path, &b).await;
            assert_eq!(r.status(), 400, "{path} with {field}");
            let (code, kind, msg) = err_of(&r.json().await.unwrap());
            assert_eq!((code, kind.as_str()), (400, "invalid_request_error"), "{path} {field}");
            assert!(msg.contains(&format!("`{field}`")), "the message names the field: {msg}");
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn empty_or_null_values_of_unsupported_fields_are_accepted() {
    let srv = spawn(fast());
    let empties = [json!(null), json!(""), json!([]), json!({}), json!(false), json!(0)];
    for (path, base) in generation_routes(&srv).await {
        for (field, _) in refused_cases() {
            for e in &empties {
                if field == "repeat_penalty" && *e == json!(0) {
                    continue; // its neutral value is 1.0; 0 is a real (nonsensical) penalty and stays refused
                }
                let mut b = base.clone();
                b[field] = e.clone();
                let r = srv.post(&path, &b).await;
                let st = r.status();
                assert_eq!(st, 200, "{path} with {field}={e}: {}", r.text().await.unwrap());
                r.bytes().await.unwrap();
            }
        }
        // the neutral value of repeat_penalty, and harmless fields (greedy engine, bookkeeping)
        let mut b = base.clone();
        for (k, v) in [("repeat_penalty", json!(1.0)), ("top_k", json!(40)), ("top_p", json!(0.9)), ("seed", json!(7)), ("min_p", json!(0.05)), ("id_slot", json!(-1))] {
            b[k] = v;
        }
        let r = srv.post(&path, &b).await;
        assert_eq!(r.status(), 200, "{path} with harmless fields");
        r.bytes().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn caps_lists_what_is_refused_and_what_is_ignored() {
    let srv = spawn(fast());
    let caps = srv.get_json("/caps").await;
    let list = |k: &str| -> Vec<String> { caps["completion"][k].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect() };
    let (refused, harmless, honoured) = (list("refused"), list("ignored_harmless"), list("honoured"));
    for (field, _) in refused_cases().into_iter().filter(|(f, _)| *f != "json_schema") {
        assert!(refused.iter().any(|f| f == field), "/caps completion.refused lacks {field}");
    }
    assert!(harmless.iter().any(|f| f == "top_p"));
    for f in &refused {
        assert!(!harmless.contains(f) && !honoured.contains(f), "{f} is listed twice");
    }
    // the lists match reality: every refused field is refused, every harmless one is accepted
    let base = json!({"prompt": ids(1, 4), "n_predict": 1});
    for f in &refused {
        let mut b = base.clone();
        b[f.as_str()] = json!(if f == "repeat_penalty" { json!(2.0) } else { json!(true) });
        assert_eq!(srv.post("/completion", &b).await.status(), 400, "caps says {f} is refused");
    }
    for f in &harmless {
        let mut b = base.clone();
        b[f.as_str()] = json!(1);
        let r = srv.post("/completion", &b).await;
        assert_eq!(r.status(), 200, "caps says {f} is ignored");
        r.bytes().await.unwrap();
    }
}

// -- grammar (GBNF, step 4) ---------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn grammar_is_honoured_not_refused_and_an_invalid_one_is_a_400_before_anything_is_queued() {
    let srv = spawn(fast());
    let caps = srv.get_json("/caps").await;
    let list = |k: &str| -> Vec<String> { caps["completion"][k].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect() };
    assert!(list("honoured").iter().any(|f| f == "grammar"));
    assert!(!list("refused").iter().any(|f| f == "grammar"));
    for f in ["lora", "tools", "response_format"] {
        assert!(list("refused").iter().any(|x| x == f), "{f} stays refused");
    }
    // json_schema is honoured on the text routes (tests/llama_compat.rs) and still refused on agent turns and events
    assert!(list("honoured").iter().any(|f| f == "json_schema") && !list("refused").iter().any(|f| f == "json_schema"));
    assert_eq!(caps["completion"]["refused_on_agent_turns_and_events"], json!(["json_schema"]));
    let ok = "root ::= \"a\" | \"b\"";
    for (path, base) in generation_routes(&srv).await {
        let before = srv.metrics().await;
        for bad in ["not a grammar", "root = \"a\""] {
            let mut b = base.clone();
            if path.ends_with("/event") {
                b["dry_run"] = json!(false); // a dry run frames the event and runs no turn: nothing builds a grammar
            }
            b["grammar"] = json!(bad);
            let r = srv.post(&path, &b).await;
            assert_eq!(r.status(), 400, "{path} {bad}");
            let (code, kind, msg) = err_of(&r.json().await.unwrap());
            assert_eq!((code, kind.as_str()), (400, "invalid_request_error"), "{path}");
            assert!(msg.contains("grammar"), "{msg}");
        }
        let after = srv.metrics().await;
        // refused at admission: both are failures of the submit, nothing ran or is waiting
        assert_eq!(after["requests"]["failed"].as_u64().unwrap() - before["requests"]["failed"].as_u64().unwrap(), 2, "{path}");
        assert_eq!(before["requests"]["completed"], after["requests"]["completed"], "{path}: a refused grammar ran");
        assert_eq!(after["active_requests"], 0, "{path}: a refused grammar was queued");
        // a valid grammar, an empty one and null are accepted
        for g in [json!(ok), json!(""), json!(null)] {
            let mut b = base.clone();
            b["grammar"] = g.clone();
            let r = srv.post(&path, &b).await;
            assert_eq!(r.status(), 200, "{path} grammar={g}");
            r.bytes().await.unwrap();
        }
    }
    // the echo in generation_settings is the grammar the request asked for
    let v = srv.complete(&ids(1, 4), 2, json!({"grammar": ok})).await;
    assert_eq!(v["generation_settings"]["grammar"], ok);
    let v = srv.complete(&ids(1, 4), 2, json!({})).await;
    assert_eq!(v["generation_settings"]["grammar"], "");
    // a number is not a grammar
    let r = srv.post("/completion", &json!({"prompt": ids(1, 4), "n_predict": 2, "grammar": 5})).await;
    assert_eq!(r.status(), 400);
}
