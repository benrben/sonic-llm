//! Compaction and the extracted-fields record over the HTTP API, on the fake engine (phase 3). The fake engine's tokens
//! are a pure function of every id in the context, so "the compacted agent continues exactly like a from-scratch agent
//! over [system + summary + tail]" is checkable without a GPU; the real engine's exactness is in `compaction_real.rs`
//! and sllm-core's `compaction_gpu.rs`.

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::compaction::{CompactionConfig, Policy};
use sllm_server::fake::FakeConfig;
use sllm_server::tokenizer::Tokenizer;
use sllm_server::ServerConfig;
use std::time::Duration;

fn ids(a: i32, n: i32) -> Vec<i32> {
    (a..a + n).collect()
}

fn toks(v: &Value) -> Vec<i32> {
    v["tokens"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect()
}

async fn create(srv: &Srv, body: Value) -> String {
    let r = srv.post("/v1/agents", &body).await;
    assert_eq!(r.status(), 201);
    r.json::<Value>().await.unwrap()["id"].as_str().unwrap().to_string()
}

/// A turn of `prompt` with `n` tokens (ignore_eos): the generated ids.
async fn turn(srv: &Srv, id: &str, prompt: &[i32], n: i64) -> Vec<i32> {
    let r = srv.post(&format!("/v1/agents/{id}/turn"), &json!({"prompt": prompt, "n_predict": n, "stream": false, "ignore_eos": true, "return_tokens": true})).await;
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    toks(&r.json().await.unwrap())
}

async fn compact(srv: &Srv, id: &str, body: Value) -> (u16, Value) {
    let r = srv.post(&format!("/v1/agents/{id}/compact"), &body).await;
    (r.status().as_u16(), r.json().await.unwrap())
}

/// Wait until the agent has made `n` compactions (a policy compaction runs right after the turn that triggered it).
async fn wait_compactions(srv: &Srv, id: &str, n: u64) -> Value {
    let t0 = std::time::Instant::now();
    loop {
        let info = srv.get_json(&format!("/v1/agents/{id}")).await;
        if info["compaction"]["compactions"].as_u64().unwrap() >= n && info["compaction"]["compacting"] == false {
            return info;
        }
        assert!(t0.elapsed() < Duration::from_secs(5), "no compaction: {info}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Turns of 300 ids and 20 generated tokens until the context holds at least `n` tokens; the context's ids.
async fn grow(srv: &Srv, id: &str, mut ctx: Vec<i32>, n: usize) -> Vec<i32> {
    let mut k = 0;
    while ctx.len() < n {
        let p = ids(1000 + 300 * k, 300);
        ctx.extend(&p);
        ctx.extend(turn(srv, id, &p, 20).await);
        k += 1;
    }
    ctx
}

#[tokio::test(flavor = "multi_thread")]
async fn a_compacted_agent_continues_like_a_from_scratch_agent_over_its_new_window() {
    let srv = spawn(FakeConfig { pool_pages: Some(64), ..fast() });
    let system = ids(1, 40);
    let a = create(&srv, json!({"system_prompt_ids": system})).await;
    let ctx = grow(&srv, &a, system.clone(), 3000).await;
    let info = srv.get_json(&format!("/v1/agents/{a}")).await;
    assert_eq!(info["context_tokens"], ctx.len());
    let turns = info["turns"].clone();
    let summary = ids(50_000, 120);
    let (st, rep) = compact(&srv, &a, json!({"summary_ids": summary, "tail_tokens": 500})).await;
    assert_eq!(st, 200, "{rep}");
    assert_eq!(rep["object"], "agent.compaction");
    assert_eq!((rep["prefix_tokens"].clone(), rep["summary_tokens"].clone(), rep["tail_tokens"].clone()), (json!(40), json!(120), json!(500)));
    assert_eq!(rep["replaced_tokens"], ctx.len() - 40 - 500);
    assert_eq!((rep["context_before"].clone(), rep["context_after"].clone()), (json!(ctx.len()), json!(660)));
    assert_eq!(rep["summary_source"], "client");
    assert_eq!(rep["trigger"], "api");
    assert_eq!((rep["pages_before"].clone(), rep["pages_after"].clone()), (json!(ctx.len().div_ceil(256)), json!(1)));
    assert_eq!(rep["pages_freed"], ctx.len().div_ceil(256) - 1);
    assert_eq!(rep["reprefill_tokens"], 620);
    let info = srv.get_json(&format!("/v1/agents/{a}")).await;
    assert_eq!((info["context_tokens"].clone(), info["busy"].clone()), (json!(660), json!(false)));
    assert_eq!((info["compaction"]["compactions"].clone(), info["compaction"]["context_at_last"].clone()), (json!(1), json!(660)));
    assert_eq!(info["turns"], turns, "a compaction is not a turn");

    // from scratch: [system] then [summary + tail] (no generation), then the same next turn
    let b = create(&srv, json!({"system_prompt_ids": system})).await;
    let mut window = summary.clone();
    window.extend(&ctx[ctx.len() - 500..]);
    turn(&srv, &b, &window, 0).await;
    let next = ids(90_000, 30);
    assert_eq!(turn(&srv, &a, &next, 16).await, turn(&srv, &b, &next, 16).await);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_stub_summary_is_the_start_of_the_replaced_history() {
    let srv = spawn(fast());
    let system = ids(1, 10);
    let a = create(&srv, json!({"system_prompt_ids": system})).await;
    let ctx = grow(&srv, &a, system.clone(), 1200).await;
    let (st, rep) = compact(&srv, &a, json!({"tail_tokens": 100, "summary_tokens": 64})).await;
    assert_eq!(st, 200, "{rep}");
    assert_eq!((rep["summary_source"].clone(), rep["summary_tokens"].clone()), (json!("stub-first-n"), json!(64)));
    assert!(rep["pages_freed"].is_null(), "the fake engine without a pool model has no pages");
    let b = create(&srv, json!({"system_prompt_ids": system})).await;
    let window: Vec<i32> = ctx[10..74].iter().chain(&ctx[ctx.len() - 100..]).copied().collect();
    turn(&srv, &b, &window, 0).await;
    assert_eq!(turn(&srv, &a, &[7, 8, 9], 12).await, turn(&srv, &b, &[7, 8, 9], 12).await);
    // text summaries are tokenized (the bytes tokenizer: one id per byte), without BOS
    let (st, rep) = compact(&srv, &a, json!({"summary": "the user wants a refund", "tail_from": 30})).await;
    assert_eq!(st, 200, "{rep}");
    assert_eq!(rep["summary_tokens"], "the user wants a refund".len());
}

#[tokio::test(flavor = "multi_thread")]
async fn compaction_requests_are_checked() {
    let srv = spawn(fast());
    let a = create(&srv, json!({"system_prompt_ids": ids(1, 10)})).await;
    turn(&srv, &a, &ids(100, 50), 10).await;
    let cases = [
        (json!({"tail_tokens": 1024}), 422, "nothing to compact"),
        (json!({"tail_tokens": 10, "tail_from": 20}), 422, "not both"),
        (json!({"summary": [1], "summary_ids": [2]}), 422, "not both"),
        (json!({"tail_from": 5}), 422, "inside the prefix"),
        (json!({"tail_from": 500}), 422, "past the end"),
        (json!({"summary_ids": [-1], "tail_tokens": 10}), 422, "outside"),
        (json!({"bogus": 1}), 400, "unknown field"), // a body that does not fit the request is 400
    ];
    for (body, status, msg) in cases {
        let (st, v) = compact(&srv, &a, body.clone()).await;
        assert_eq!(st, status, "{body}: {v}");
        assert!(v["error"]["message"].as_str().unwrap().contains(msg), "{body}: {v}");
    }
    assert_eq!(srv.get_json(&format!("/v1/agents/{a}")).await["context_tokens"], 70, "a refused compaction changes nothing");
    let (st, _) = compact(&srv, "ag_999", json!({})).await;
    assert_eq!(st, 404);
    // a window bigger than the context is refused before the cut
    let (st, v) = compact(&srv, &a, json!({"summary_ids": vec![5; 5000], "tail_tokens": 10})).await;
    assert_eq!((st, v["error"]["type"].as_str().unwrap()), (400, "exceed_context_size_error"));
    assert_eq!(srv.get_json(&format!("/v1/agents/{a}")).await["context_tokens"], 70);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_busy_agent_is_not_compacted_and_a_turn_waits_for_a_compaction() {
    // slow prefill: 0.2 ms a token, so a 2k-token re-prefill takes ~0.4 s
    let srv = spawn(FakeConfig { prefill_ms_per_token: 0.2, token_ms: 1.0, ..fast() });
    let a = create(&srv, json!({"system_prompt_ids": ids(1, 10)})).await;
    turn(&srv, &a, &ids(100, 3000), 4).await;
    // a turn in flight: 409
    let slow = {
        let (url, http) = (srv.url(&format!("/v1/agents/{a}/turn")), srv.http.clone());
        tokio::spawn(async move { http.post(url).json(&json!({"prompt": [1, 2], "n_predict": 300, "stream": false, "ignore_eos": true})).send().await.unwrap().status() })
    };
    tokio::time::sleep(Duration::from_millis(30)).await;
    let (st, v) = compact(&srv, &a, json!({"tail_tokens": 2000})).await;
    assert_eq!(st, 409, "{v}");
    assert_eq!(slow.await.unwrap(), 200);
    // a compaction in flight: the next turn is held and runs after it; a second turn meanwhile is refused
    let c = {
        let (url, http) = (srv.url(&format!("/v1/agents/{a}/compact")), srv.http.clone());
        tokio::spawn(async move { http.post(url).json(&json!({"summary_ids": [9, 9, 9], "tail_tokens": 2000})).send().await.unwrap().json::<Value>().await.unwrap() })
    };
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(srv.get_json(&format!("/v1/agents/{a}")).await["compaction"]["compacting"], true);
    let held = {
        let (url, http) = (srv.url(&format!("/v1/agents/{a}/turn")), srv.http.clone());
        tokio::spawn(async move { http.post(url).json(&json!({"prompt": [5], "n_predict": 3, "stream": false, "ignore_eos": true})).send().await.unwrap().json::<Value>().await.unwrap() })
    };
    tokio::time::sleep(Duration::from_millis(20)).await;
    let r = srv.post(&format!("/v1/agents/{a}/turn"), &json!({"prompt": [6], "n_predict": 3, "stream": false})).await;
    assert_eq!(r.status(), 409);
    let rep = c.await.unwrap();
    assert_eq!(rep["context_after"], 10 + 3 + 2000, "{rep}");
    assert!(rep["reprefill_ms"].as_f64().unwrap() > 100.0, "{rep}");
    let v = held.await.unwrap();
    assert_eq!(v["tokens_predicted"], 3, "{v}");
    assert_eq!(v["timings"]["cache_n"], 10 + 3 + 2000, "the held turn ran on the compacted window");
}

#[tokio::test(flavor = "multi_thread")]
async fn policy_triggers_compact_after_a_turn() {
    let srv = spawn(FakeConfig { pool_pages: Some(64), ..fast() });
    let system = ids(1, 10);
    let policy = json!({"window_tokens": 1500, "tail_tokens": 200, "summary_tokens": 50});
    let a = create(&srv, json!({"system_prompt_ids": system, "compaction": policy})).await;
    let info = srv.get_json(&format!("/v1/agents/{a}")).await;
    assert_eq!(info["compaction"]["policy"]["window_tokens"], 1500);
    // 10 + 4 x 320 = 1290 < 1500: no compaction yet; the fifth turn crosses it
    let mut ctx = system.clone();
    for k in 0..5 {
        let p = ids(1000 + 300 * k, 300);
        ctx.extend(&p);
        ctx.extend(turn(&srv, &a, &p, 20).await);
        if k < 4 {
            let info = srv.get_json(&format!("/v1/agents/{a}")).await;
            assert_eq!((info["compaction"]["compactions"].clone(), info["compaction"]["compacting"].clone()), (json!(0), json!(false)), "after turn {k}");
        }
    }
    // the compaction runs right after the turn that crossed the window
    let info = wait_compactions(&srv, &a, 1).await;
    assert_eq!(info["context_tokens"], 10 + 50 + 200);
    let last = &info["compaction"]["last"];
    assert_eq!((last["trigger"].clone(), last["summary_source"].clone(), last["context_before"].clone()), (json!("window"), json!("stub-first-n"), json!(ctx.len())));
    assert_eq!(last["pages_freed"], ctx.len().div_ceil(256) - 1);
    let b = create(&srv, json!({"system_prompt_ids": system})).await;
    let window: Vec<i32> = ctx[10..60].iter().chain(&ctx[ctx.len() - 200..]).copied().collect();
    turn(&srv, &b, &window, 0).await;
    assert_eq!(turn(&srv, &a, &[3, 4], 10).await, turn(&srv, &b, &[3, 4], 10).await);

    // every_tokens counts from the last compaction; the policy can be read, changed and cleared
    let r = srv.http.put(srv.url(&format!("/v1/agents/{a}/compaction"))).json(&json!({"every_tokens": 600, "tail_tokens": 100, "summary_tokens": 20})).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let before = srv.get_json(&format!("/v1/agents/{a}/compaction")).await["compactions"].as_u64().unwrap();
    turn(&srv, &a, &ids(7000, 300), 20).await; // +320 since the last compaction (and +12 from the check above)
    assert_eq!(srv.get_json(&format!("/v1/agents/{a}/compaction")).await["compactions"].as_u64().unwrap(), before);
    turn(&srv, &a, &ids(7300, 300), 20).await; // +652: due
    wait_compactions(&srv, &a, before + 1).await;
    let c = srv.get_json(&format!("/v1/agents/{a}/compaction")).await;
    assert_eq!((c["compactions"].as_u64().unwrap(), c["last"]["trigger"].clone()), (before + 1, json!("every")));
    assert_eq!(srv.get_json(&format!("/v1/agents/{a}")).await["context_tokens"], 10 + 20 + 100);
    let r = srv.http.delete(srv.url(&format!("/v1/agents/{a}/compaction"))).send().await.unwrap();
    assert!(r.json::<Value>().await.unwrap()["policy"].is_null());
    // invalid policies: a window a compaction cannot get under, no trigger, unknown fields
    for p in [json!({"window_tokens": 300}), json!({"tail_tokens": 100}), json!({"window_tokens": 9000, "x": 1})] {
        let want = if p.get("x").is_some() { 400 } else { 422 }; // an unknown field fails the body's type: 400
        let r = srv.http.put(srv.url(&format!("/v1/agents/{a}/compaction"))).json(&p).send().await.unwrap();
        assert_eq!(r.status(), want, "{p}");
        let r = srv.post("/v1/agents", &json!({"system_prompt_ids": system, "compaction": p})).await;
        assert_eq!(r.status(), want, "{p}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_server_default_policy_applies_to_new_agents() {
    let comp = CompactionConfig { default_policy: Some(Policy { window_tokens: Some(1000), every_tokens: None, tail_tokens: 100, summary_tokens: 30 }), ..Default::default() };
    let srv = spawn_with(fast(), Tokenizer::Bytes, ServerConfig { port: 0, compaction: comp, ..Default::default() });
    let caps = srv.get_json("/caps").await;
    assert_eq!((caps["compaction"]["summarizer"].clone(), caps["compaction"]["default_policy"]["window_tokens"].clone()), (json!("stub-first-n"), json!(1000)));
    assert!(caps["endpoints"].as_array().unwrap().contains(&json!("/v1/agents/{id}/compact")));
    let a = create(&srv, json!({"system_prompt_ids": ids(1, 10)})).await;
    grow(&srv, &a, ids(1, 10), 1000).await;
    let info = wait_compactions(&srv, &a, 1).await;
    assert_eq!((info["compaction"]["compactions"].clone(), info["context_tokens"].clone()), (json!(1), json!(140)));
    // a system prompt too long for the default policy: the agent gets none
    let b = create(&srv, json!({"system_prompt_ids": ids(1, 900)})).await;
    assert!(srv.get_json(&format!("/v1/agents/{b}")).await["compaction"]["policy"].is_null());
}

#[tokio::test(flavor = "multi_thread")]
async fn the_fields_record_lives_outside_the_window() {
    let srv = spawn(fast());
    let a = create(&srv, json!({"system_prompt_ids": ids(1, 10), "fields": {"customer": "C-42"}})).await;
    let info = srv.get_json(&format!("/v1/agents/{a}")).await;
    assert_eq!(info["fields"], json!({"customer": "C-42"}));
    let path = format!("/v1/agents/{a}/fields");
    let r = srv.http.patch(srv.url(&path)).json(&json!({"order_id": "A-1001", "email": "ann@example.com", "step": 2})).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.json::<Value>().await.unwrap()["fields"], json!({"customer": "C-42", "order_id": "A-1001", "email": "ann@example.com", "step": 2}));
    let r = srv.http.patch(srv.url(&path)).json(&json!({"fields": {"step": 3, "email": null}})).send().await.unwrap();
    assert_eq!(r.json::<Value>().await.unwrap()["fields"], json!({"customer": "C-42", "order_id": "A-1001", "step": 3}));
    let r = srv.http.delete(srv.url(&format!("{path}/customer"))).send().await.unwrap();
    assert_eq!(r.json::<Value>().await.unwrap()["found"], true);
    // fields do not count in the window and survive a compaction
    let ctx = srv.get_json(&format!("/v1/agents/{a}")).await["context_tokens"].clone();
    assert_eq!(ctx, 10);
    turn(&srv, &a, &ids(100, 600), 10).await;
    let (st, _) = compact(&srv, &a, json!({"summary_ids": [1, 2], "tail_tokens": 50})).await;
    assert_eq!(st, 200);
    assert_eq!(srv.get_json(&path).await["fields"], json!({"order_id": "A-1001", "step": 3}));
    // replace, and the checks
    let r = srv.http.put(srv.url(&path)).json(&json!({"state": "refund_pending"})).send().await.unwrap();
    assert_eq!(r.json::<Value>().await.unwrap()["fields"], json!({"state": "refund_pending"}));
    let r = srv.http.patch(srv.url(&path)).json(&json!({"doc": {"nested": true}})).send().await.unwrap();
    assert_eq!(r.status(), 422);
    let r = srv.http.patch(srv.url(&path)).body("[1, 2]").header("content-type", "application/json").send().await.unwrap();
    assert_eq!(r.status(), 400, "a body of the wrong type");
    let r = srv.post("/v1/agents", &json!({"fields": {"x": [1]}})).await;
    assert_eq!(r.status(), 422);
    assert_eq!(srv.get("/v1/agents/ag_999/fields").await.status(), 404);
    assert_eq!(srv.get_json(&format!("/v1/agents/{a}")).await["fields"], json!({"state": "refund_pending"}));
}

#[tokio::test(flavor = "multi_thread")]
async fn deleting_a_compacting_agent_answers_everyone() {
    let srv = spawn(FakeConfig { prefill_ms_per_token: 0.2, token_ms: 1.0, ..fast() });
    let a = create(&srv, json!({"system_prompt_ids": ids(1, 10)})).await;
    turn(&srv, &a, &ids(100, 3000), 4).await;
    let c = {
        let (url, http) = (srv.url(&format!("/v1/agents/{a}/compact")), srv.http.clone());
        tokio::spawn(async move { http.post(url).json(&json!({"tail_tokens": 2500})).send().await.unwrap().status().as_u16() })
    };
    tokio::time::sleep(Duration::from_millis(50)).await;
    let held = {
        let (url, http) = (srv.url(&format!("/v1/agents/{a}/turn")), srv.http.clone());
        tokio::spawn(async move { http.post(url).json(&json!({"prompt": [5], "n_predict": 3, "stream": false})).send().await.unwrap().status().as_u16() })
    };
    tokio::time::sleep(Duration::from_millis(20)).await;
    let r = srv.http.delete(srv.url(&format!("/v1/agents/{a}"))).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(c.await.unwrap(), 499);
    assert_eq!(held.await.unwrap(), 404);
    assert_eq!(srv.metrics().await["active_requests"], 0);
}

/// 08-tests "Helper reservation": the background lane is full (every slot decoding, no queue room) when a compaction is
/// due. The compaction runs on the helper lane at once (not deferred, not queued behind the workers) and ends in a few
/// steps, while a background turn arriving meanwhile is deferred.
#[tokio::test(flavor = "multi_thread")]
async fn a_compaction_does_not_wait_for_a_full_background_lane() {
    use sllm_server::sched::SchedConfig;
    let sched = SchedConfig { caps: [4, 4, 1], queue_limits: [16, 0, 4], ..Default::default() };
    let fake = FakeConfig { prefill_ms_per_token: 0.02, token_ms: 2.0, ..fast() };
    let srv = spawn_with(fake, Tokenizer::Bytes, ServerConfig { port: 0, sched, ..Default::default() });
    let make = |srv: &Srv| {
        let srv_url = srv.url("/v1/agents");
        let http = srv.http.clone();
        async move {
            let r = http.post(srv_url).json(&json!({"system_prompt_ids": ids(1, 300), "lane": "background"})).send().await.unwrap();
            r.json::<Value>().await.unwrap()["id"].as_str().unwrap().to_string()
        }
    };
    // the agent that will compact: a long history, made before the lane fills
    let c = make(&srv).await;
    turn(&srv, &c, &ids(1000, 3500), 8).await;
    // four background agents decode 1500 tokens each (~3 s at 2 ms a step): the lane is full, its queue has no room
    let mut workers = Vec::new();
    for _ in 0..4 {
        let w = make(&srv).await;
        let (url, http) = (srv.url(&format!("/v1/agents/{w}/turn")), srv.http.clone());
        workers.push(tokio::spawn(async move {
            http.post(url).json(&json!({"prompt": [7], "n_predict": 1500, "ignore_eos": true, "stream": false})).send().await.unwrap().status().as_u16()
        }));
    }
    srv.wait_metrics("a full background lane", |m| m["scheduler"]["lanes"]["background"]["running"] == 4).await;
    let idle = make(&srv).await;
    let r = srv.post(&format!("/v1/agents/{idle}/turn"), &json!({"prompt": [1], "n_predict": 4, "stream": false})).await;
    assert_eq!(r.status(), 503, "the background lane is full");
    // the compaction: system 300 + summary 300 + tail 1024 re-prefilled while the workers decode
    let (st, rep) = compact(&srv, &c, json!({"tail_tokens": 1024, "summary_tokens": 300})).await;
    assert_eq!(st, 200, "{rep}");
    let m = srv.metrics().await;
    assert_eq!(m["scheduler"]["lanes"]["background"]["running"], 4, "the workers are still decoding: {m}");
    assert!(rep["queued_ms"].as_f64().unwrap() < 20.0, "the helper slot was free at once: {rep}");
    assert!(rep["total_ms"].as_f64().unwrap() < 1000.0, "within the 1 s budget: {rep}");
    assert_eq!(rep["context_after"], 300 + 300 + 1024);
    for w in workers {
        assert_eq!(w.await.unwrap(), 200);
    }
}

/// A summarizer that writes its summary on another thread, after a delay (the shape of the helper summarizer).
struct SlowSummarizer;
struct SlowPending(std::time::Instant);

impl sllm_server::compaction::PendingSummary for SlowPending {
    fn poll(&mut self) -> Option<Result<Vec<i32>, String>> {
        (self.0.elapsed() >= Duration::from_millis(150)).then(|| Ok(vec![7, 8, 9]))
    }
}

impl sllm_server::compaction::Summarizer for SlowSummarizer {
    fn name(&self) -> &'static str {
        "slow-test"
    }
    fn summarize(&self, _: &str, _: &[i32], _: usize) -> Result<Vec<i32>, String> {
        Err("asynchronous".into())
    }
    fn start(&self, _: &str, _: &[i32], _: usize) -> Option<Box<dyn sllm_server::compaction::PendingSummary>> {
        Some(Box::new(SlowPending(std::time::Instant::now())))
    }
}

/// Regression (G3 loop hang): with an asynchronous summary pending and nothing else running, the executor must keep
/// polling it; it used to block waiting for a command, so `/compact` never returned.
#[tokio::test(flavor = "multi_thread")]
async fn an_asynchronous_summary_is_polled_when_the_executor_has_no_other_work() {
    let comp = CompactionConfig { summarizer: sllm_server::compaction::SummarizerHandle(std::sync::Arc::new(SlowSummarizer)), ..Default::default() };
    let srv = spawn_with(fast(), Tokenizer::Bytes, ServerConfig { port: 0, compaction: comp, ..Default::default() });
    let a = create(&srv, json!({"system_prompt_ids": ids(1, 10)})).await;
    turn(&srv, &a, &ids(100, 600), 10).await;
    let r = tokio::time::timeout(Duration::from_secs(5), compact(&srv, &a, json!({"tail_tokens": 50})))
        .await
        .expect("the compaction never returned (the executor did not poll the pending summary)");
    assert_eq!(r.0, 200, "{}", r.1);
    assert_eq!(r.1["summary_source"], "slow-test");
    assert_eq!(r.1["summary_tokens"], 3);
}
