//! The phase 2 scheduler through the HTTP API, on the FakeEngine: lanes and the step policy (task 2), admission and
//! deferral (task 7), cancellation and deadlines (task 8), the reserved helper slot (task 9).

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::engine::{ReqId, Work, WorkItem};
use sllm_server::fake::{FakeConfig, StepTrace};
use sllm_server::sched::SchedConfig;
use sllm_server::tokenizer::Tokenizer;
use sllm_server::ServerConfig;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn ids(a: i32, n: i32) -> Vec<i32> {
    (a..a + n).collect()
}

fn spawn_sched(fake: FakeConfig, sched: SchedConfig) -> Srv {
    spawn_with(fake, Tokenizer::Bytes, ServerConfig { port: 0, sched, ..Default::default() })
}

fn traced(fake: FakeConfig) -> (FakeConfig, StepTrace) {
    let t: StepTrace = Arc::new(Mutex::new(Vec::new()));
    (FakeConfig { trace: Some(t.clone()), ..fake }, t)
}

fn req_id(r: &reqwest::Response) -> ReqId {
    r.headers()["x-request-id"].to_str().unwrap().parse().unwrap()
}

/// POST a streaming /completion and wait until it is admitted (headers back).
async fn open(srv: &Srv, prompt: &[i32], n: i64, extra: Value) -> reqwest::Response {
    let mut b = json!({"prompt": prompt, "n_predict": n, "stream": true, "ignore_eos": true, "return_tokens": true});
    for (k, v) in extra.as_object().into_iter().flatten() {
        b[k] = v.clone();
    }
    srv.post("/completion", &b).await
}

/// Read a stream to its end: (tokens, final chunk).
async fn drain(r: reqwest::Response) -> (Vec<i64>, Value) {
    drain_after(String::new(), r).await
}

/// As [`drain`], with what was already read of the stream.
async fn drain_after(mut raw: String, mut r: reqwest::Response) -> (Vec<i64>, Value) {
    while let Some(c) = r.chunk().await.unwrap() {
        raw.push_str(std::str::from_utf8(&c).unwrap());
    }
    let chunks = sse_json(&raw);
    (tokens_of(&chunks), chunks.last().cloned().unwrap_or(Value::Null))
}

/// Read until `k` SSE events have arrived; returns the rest of the response.
async fn read_events(r: &mut reqwest::Response, k: usize) -> String {
    let mut got = String::new();
    while got.matches("\n\n").count() < k {
        got.push_str(std::str::from_utf8(&r.chunk().await.unwrap().expect("stream ended early")).unwrap());
    }
    got
}

fn lane<'a>(m: &'a Value, name: &str) -> &'a Value {
    &m["scheduler"]["lanes"][name]
}

fn steps_of(t: &StepTrace) -> Vec<Vec<WorkItem>> {
    t.lock().unwrap().clone()
}

fn has(step: &[WorkItem], req: ReqId, prefill: bool) -> bool {
    step.iter().any(|w| w.req == req && matches!(w.work, Work::Prefill { .. }) == prefill)
}

/// Task 2: an interactive request's prefill goes into the next step, its decode rows lead every step, and background
/// prefill waits while it decodes (until the period lets one chunk through); background decode rows keep running.
#[tokio::test(flavor = "multi_thread")]
async fn interactive_lane_goes_first_and_defers_background_prefill() {
    let (fake, trace) = traced(FakeConfig { token_ms: 2.0, ..fast() });
    let srv = spawn_sched(fake, SchedConfig { chunk: 64, step_tokens: 64, helper_tokens: 32, bg_prefill_period: 8, ..Default::default() });
    // a background request decoding, a second one with a long prompt still prefilling
    let mut b1 = open(&srv, &ids(1, 8), 400, json!({})).await;
    read_events(&mut b1, 2).await;
    let b2 = open(&srv, &ids(100, 2000), 50, json!({})).await;
    let b2_id = req_id(&b2);
    let mut i1 = open(&srv, &ids(7, 100), 40, json!({"lane": "interactive"})).await;
    let i1_id = req_id(&i1);
    read_events(&mut i1, 1).await;
    let (toks, last) = drain(i1).await;
    assert_eq!((toks.len(), last["stop_type"].as_str()), (39, Some("limit")));
    let steps = steps_of(&trace);
    let first = steps.iter().position(|s| s.iter().any(|w| w.req == i1_id)).unwrap();
    assert!(has(&steps[first], i1_id, true), "its first item is a prefill chunk");
    assert_eq!(steps[first].iter().find(|w| w.req == i1_id).unwrap().work, Work::Prefill { n: 64 }, "before background prefill");
    assert!(!has(&steps[first], b2_id, true), "interactive prefill takes the budget before background prefill");
    // while it decodes: it is the first row, background decode continues, background prefill only every 8th step
    let decoding: Vec<&Vec<WorkItem>> = steps.iter().filter(|s| s.first().is_some_and(|w| w.req == i1_id && w.work == Work::Decode)).collect();
    assert!(decoding.len() >= 38, "{} decode steps", decoding.len());
    let with_bg_prefill = decoding.iter().filter(|s| has(s, b2_id, true)).count();
    assert!(with_bg_prefill >= 3 && with_bg_prefill <= decoding.len() / 8 + 1, "background prefill in {with_bg_prefill} of {} steps", decoding.len());
    assert!(decoding.iter().all(|s| s.iter().any(|w| w.work == Work::Decode && w.req != i1_id)), "background rows ride along");
    let m = srv.metrics().await;
    assert!(m["scheduler"]["bg_prefill_deferred_steps"].as_u64().unwrap() >= 30, "{m}");
    assert!(m["scheduler"]["steps_with_interactive"].as_u64().unwrap() >= 38);
    assert_eq!(lane(&m, "interactive")["completed"], 1);
    drop(b1);
    drop(b2);
}

/// Task 2: tokens do not depend on the lane, on chunking, or on the neighbours of a step.
#[tokio::test(flavor = "multi_thread")]
async fn lanes_and_chunks_do_not_change_tokens() {
    let srv = spawn(fast());
    let solo: Vec<Vec<i64>> = {
        let mut v = vec![];
        for i in 0..6 {
            v.push(tokens_of(&srv.stream("/completion", &json!({"prompt": ids(i * 50, 300), "n_predict": 20, "stream": true, "ignore_eos": true, "return_tokens": true})).await.1));
        }
        v
    };
    let srv = spawn_sched(FakeConfig { token_ms: 1.0, ..fast() }, SchedConfig { chunk: 37, step_tokens: 100, helper_tokens: 20, ..Default::default() });
    let lanes = ["interactive", "background", "helper", "background", "interactive", "background"];
    let futs = (0..6).map(|i| {
        let srv = &srv;
        async move { drain(open(srv, &ids(i as i32 * 50, 300), 20, json!({"lane": lanes[i]})).await).await.0 }
    });
    let got = futures_util::future::join_all(futs).await;
    for i in 0..6 {
        assert_eq!(got[i], solo[i], "request {i} on the {} lane", lanes[i]);
    }
}

/// Task 7: past a lane's slots and queue a request is deferred (503, retry_after, the deferred object, Retry-After),
/// the queue never exceeds its limit, other lanes are unaffected, and /metrics reports it all.
#[tokio::test(flavor = "multi_thread")]
async fn a_full_lane_defers_with_retry_after_and_the_queue_stays_bounded() {
    let srv = spawn_sched(FakeConfig { token_ms: 3.0, ..fast() }, SchedConfig { caps: [4, 2, 1], queue_limits: [16, 2, 4], ..Default::default() });
    let mut open_ = vec![];
    for i in 0..4 {
        let r = open(&srv, &ids(i * 10, 5), 60, json!({})).await;
        assert_eq!(r.status(), 200, "request {i} fits the 2 slots + 2 queue places");
        open_.push(r);
    }
    for _ in 0..2 {
        let r = open(&srv, &ids(1, 5), 60, json!({})).await;
        assert_eq!(r.status(), 503);
        let ra: u64 = r.headers()["retry-after"].to_str().unwrap().parse().unwrap();
        assert!(ra >= 1);
        let e: Value = r.json().await.unwrap();
        assert_eq!(e["error"]["type"], "deferred", "{e}");
        assert!(e["error"]["retry_after_ms"].as_u64().unwrap() >= 10);
        let d = &e["deferred"];
        assert_eq!((d["lane"].as_str(), d["running"].as_u64(), d["queued"].as_u64(), d["queue_limit"].as_u64()), (Some("background"), Some(2), Some(2), Some(2)), "{e}");
        assert_eq!(d["reason"], "lane_queue_full");
    }
    // another lane still admits
    let i = srv.complete(&ids(3, 4), 5, json!({"lane": "interactive"})).await;
    assert_eq!(i["tokens_predicted"], 5);
    let m = srv.metrics().await;
    let bg = lane(&m, "background");
    assert_eq!((bg["running"].as_u64(), bg["queued"].as_u64(), bg["deferred"].as_u64()), (Some(2), Some(2), Some(2)), "{m}");
    assert!(bg["max_queued"].as_u64().unwrap() <= 2);
    assert_eq!(m["requests"]["deferred"], 2);
    assert_eq!(m["scheduler"]["deferred"], 2);
    // the queued ones run once slots free; everything completes
    for r in open_ {
        let (t, last) = drain(r).await;
        assert_eq!((t.len(), last["stop_type"].as_str()), (60, Some("limit")));
    }
    let m = srv.metrics().await;
    assert_eq!(lane(&m, "background")["completed"], 4);
    assert_eq!(lane(&m, "background")["queued"], 0);
    // /caps carries the scheduler settings
    let caps = srv.get_json("/caps").await;
    assert_eq!(caps["scheduler"]["caps"], json!([4, 2, 1]));
    assert_eq!(caps["scheduler"]["queue_limits"], json!([16, 2, 4]));
}

/// Task 7: queued prompts count against the memory budget: each admitted request (running or queued) reserves its worst
/// case pages at admission, so a pool that cannot also hold the next one does not admit it, although its lane queue has
/// room. Since 2026-11-09 that request waits for room (the executor's room queue) and runs when one ends, instead of a 503.
#[tokio::test(flavor = "multi_thread")]
async fn queued_requests_reserve_their_pages_so_admission_never_overcommits() {
    // 7 pages of 256 tokens; each request needs 500 + 100 = 3 pages worst case; one background slot.
    let fake = FakeConfig { token_ms: 3.0, pool_pages: Some(7), ..fast() };
    let srv = spawn_sched(fake, SchedConfig { caps: [4, 1, 1], queue_limits: [16, 8, 4], ..Default::default() });
    let a = open(&srv, &[5; 500], 100, json!({})).await;
    let b = open(&srv, &[6; 500], 100, json!({})).await;
    assert_eq!((a.status(), b.status()), (reqwest::StatusCode::OK, reqwest::StatusCode::OK));
    let m = srv.metrics().await;
    assert_eq!(lane(&m, "background")["queued"], 1, "{m}");
    let reserved = m["kv"]["sequences"].as_array().unwrap().iter().find(|s| s["id"] == "reserved").unwrap()["pages"].as_u64().unwrap();
    assert!(reserved >= 3, "the queued request's 3 pages are reserved before it runs: {reserved}");
    let c = open(&srv, &[7; 500], 100, json!({})).await;
    assert_eq!(c.status(), 200, "6 of 7 pages are promised: the third waits for room instead of a 503");
    let m = srv.metrics().await;
    assert_eq!(m["requests"]["failed"], 0, "{m}");
    let (ta, _) = drain(a).await;
    let (tb, _) = drain(b).await;
    let (tc, _) = drain(c).await;
    assert_eq!((ta.len(), tb.len(), tc.len()), (100, 100, 100));
    let m = srv.metrics().await;
    assert!(m["pages_peak"].as_u64().unwrap() <= 7);
    assert_eq!(m["pages_active"], 0);
}

/// Task 8: DELETE /v1/requests/{id} (queued or running), DELETE /v1/agents/{id}/request, and a client that goes away
/// while queued: each is removed at the next step boundary, its slot goes to the next queued request in the next step,
/// and the other requests' tokens are unchanged.
#[tokio::test(flavor = "multi_thread")]
async fn cancel_frees_the_slot_within_one_step_and_leaves_others_unchanged() {
    let solo = {
        let srv = spawn(fast());
        let mut v = vec![];
        for i in 0..4 {
            v.push(toks_of(&srv.complete(&ids(i * 10, 6), 30, json!({})).await));
        }
        v
    };
    let (fake, trace) = traced(FakeConfig { token_ms: 4.0, ..fast() });
    let srv = spawn_sched(fake, SchedConfig { caps: [4, 2, 1], queue_limits: [16, 8, 4], ..Default::default() });
    let mut r0 = open(&srv, &ids(0, 6), 30, json!({})).await;
    let mut r1 = open(&srv, &ids(10, 6), 400, json!({})).await; // will be cancelled mid-decode
    let r2 = open(&srv, &ids(20, 6), 30, json!({})).await; // queued
    let r3 = open(&srv, &ids(30, 6), 400, json!({})).await; // queued, cancelled while queued
    let (id1, id2, id3) = (req_id(&r1), req_id(&r2), req_id(&r3));
    let seen0 = read_events(&mut r0, 2).await;
    read_events(&mut r1, 3).await;
    // cancel the queued one, then the running one
    let c: Value = srv.http.delete(srv.url(&format!("/v1/requests/{id3}"))).send().await.unwrap().json().await.unwrap();
    assert_eq!(c["cancelled"], true);
    let steps_at_cancel = trace.lock().unwrap().len();
    let c = srv.http.delete(srv.url(&format!("/v1/requests/{id1}"))).send().await.unwrap();
    assert_eq!(c.status(), 200);
    // The DELETE is answered when the executor has removed the request, not when the next step is recorded: wait (short
    // timeout) until the steps after the cancel show a step without it and the step after that, then check as before.
    let wait = Instant::now();
    let after: Vec<Vec<WorkItem>> = loop {
        let a = steps_of(&trace)[steps_at_cancel..].to_vec();
        let settled = a.iter().position(|s| !s.iter().any(|w| w.req == id1)).is_some_and(|j| a.len() > j + 1 || a[j..].iter().any(|s| s.iter().any(|w| w.req == id2)));
        if settled || wait.elapsed() > Duration::from_secs(5) {
            break a;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    };
    let j = after.iter().position(|s| !s.iter().any(|w| w.req == id1)).unwrap();
    assert!(j <= 1, "the cancelled request left within one step");
    assert!(after[j..].iter().all(|s| !s.iter().any(|w| w.req == id1)));
    let k = after.iter().position(|s| s.iter().any(|w| w.req == id2)).expect("the queued request got the slot");
    assert!(k <= j + 1, "the freed slot is used in the next step ({j} -> {k})");
    assert!(!steps_of(&trace).iter().any(|s| s.iter().any(|w| w.req == id3)), "the cancelled queued request never ran");
    let (_, last1) = drain(r1).await;
    assert_eq!(last1["stop_type"], "cancelled");
    let (_, last3) = drain(r3).await;
    assert_eq!((last3["stop_type"].as_str(), last3["tokens_predicted"].as_u64()), (Some("cancelled"), Some(0)));
    assert_eq!(drain_after(seen0, r0).await.0, solo[0], "a neighbour of the cancelled request");
    assert_eq!(drain(r2).await.0, solo[2], "the request that took its slot");
    assert_eq!(srv.http.delete(srv.url(&format!("/v1/requests/{id1}"))).send().await.unwrap().status(), 404, "gone");

    // the agent form, and a client that disconnects while queued
    let a: Value = srv.post("/v1/agents", &json!({"lane": "background"})).await.json().await.unwrap();
    let aid = a["id"].as_str().unwrap().to_string();
    let mut t = srv.post(&format!("/v1/agents/{aid}/turn"), &json!({"prompt": ids(1, 5), "n_predict": 1000, "ignore_eos": true})).await;
    read_events(&mut t, 2).await;
    let c: Value = srv.http.delete(srv.url(&format!("/v1/agents/{aid}/request"))).send().await.unwrap().json().await.unwrap();
    assert_eq!(c["cancelled"], true);
    assert!(read_events(&mut t, 1).await.contains("cancelled") || drain(t).await.1["stop_type"] == "cancelled");
    let blocker1 = open(&srv, &ids(1, 4), 300, json!({})).await;
    let blocker2 = open(&srv, &ids(2, 4), 300, json!({})).await;
    let waiting = open(&srv, &ids(3, 4), 300, json!({})).await;
    let wid = req_id(&waiting);
    drop(waiting); // gone while queued
    let m = srv.wait_metrics("the queued request is swept", |m| lane(m, "background")["queued"] == 0 && m["active_requests"] == 2).await;
    assert!(lane(&m, "background")["cancelled"].as_u64().unwrap() >= 4, "{m}");
    assert!(!steps_of(&trace).iter().any(|s| s.iter().any(|w| w.req == wid)));
    drop((blocker1, blocker2));
}

fn toks_of(v: &Value) -> Vec<i64> {
    v["tokens"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}

/// Task 8: deadline_ms stops a running request ("deadline", tokens so far delivered) and expires a queued one; both are
/// counted, and a deadline that is not reached changes nothing.
#[tokio::test(flavor = "multi_thread")]
async fn deadlines_stop_running_and_queued_requests() {
    let srv = spawn_sched(FakeConfig { token_ms: 5.0, ..fast() }, SchedConfig { caps: [4, 1, 1], ..Default::default() });
    let t0 = Instant::now();
    let running = open(&srv, &ids(1, 4), 100_000, json!({"deadline_ms": 150})).await;
    let queued = open(&srv, &ids(2, 4), 10, json!({"deadline_ms": 60})).await; // waits behind the first, never runs
    let (t, last) = drain(running).await;
    let el = t0.elapsed();
    assert_eq!(last["stop_type"], "deadline");
    assert!(!t.is_empty() && t.len() < 100, "{} tokens", t.len());
    assert!(el >= Duration::from_millis(140) && el < Duration::from_millis(1500), "{el:?}");
    let (tq, lq) = drain(queued).await;
    assert_eq!((tq.len(), lq["stop_type"].as_str()), (0, Some("deadline")));
    let ok = srv.complete(&ids(5, 4), 5, json!({"deadline_ms": 60_000})).await;
    assert_eq!((ok["tokens_predicted"].as_u64(), ok["stop_type"].as_str()), (Some(5), Some("limit")));
    let m = srv.metrics().await;
    assert_eq!(m["requests"]["expired"], 2);
    assert_eq!(lane(&m, "background")["expired"], 2);
    assert_eq!(m["active_requests"], 0);
    // OpenAI wire: finish_reason "length"
    let v: Value = srv.post("/v1/completions", &json!({"prompt": [1, 2, 3], "max_tokens": 100000, "ignore_eos": true, "deadline_ms": 30})).await.json().await.unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "length");
}

/// Task 9: with the background lane full (slots and queue), a helper request is admitted at once, prefills in the next
/// step with its reserved share of the token budget even against interactive prefill, and completes.
#[tokio::test(flavor = "multi_thread")]
async fn helper_is_admitted_while_the_background_lane_is_full() {
    let (fake, trace) = traced(FakeConfig { token_ms: 3.0, ..fast() });
    let sched = SchedConfig { caps: [4, 2, 1], queue_limits: [16, 1, 2], step_tokens: 256, chunk: 256, helper_tokens: 64, ..Default::default() };
    let srv = spawn_sched(fake, sched);
    let mut bg = vec![];
    for i in 0..3 {
        bg.push(open(&srv, &ids(i * 10, 4), 500, json!({})).await);
    }
    let full = open(&srv, &ids(99, 4), 10, json!({})).await;
    assert_eq!(full.status(), 503, "the background lane is full");
    // an interactive prompt that alone would take the whole budget for several steps
    let inter = open(&srv, &ids(1, 2000), 4, json!({"lane": "interactive"})).await;
    let h = open(&srv, &ids(50, 200), 8, json!({"lane": "helper"})).await;
    assert_eq!(h.status(), 200);
    let hid = req_id(&h);
    let (ht, hl) = drain(h).await;
    assert_eq!((ht.len(), hl["stop_type"].as_str()), (8, Some("limit")));
    let steps = steps_of(&trace);
    let first = steps.iter().position(|s| s.iter().any(|w| w.req == hid)).unwrap();
    let w = steps[first].iter().find(|w| w.req == hid).unwrap();
    assert_eq!(w.work, Work::Prefill { n: 64 }, "the reserved share, next to the interactive chunk");
    let inter_chunk = steps[first].iter().find(|w| w.work != Work::Decode && w.req != hid).map(|w| w.work);
    assert_eq!(inter_chunk, Some(Work::Prefill { n: 192 }), "the interactive prompt leaves the helper its 64 tokens");
    let m = srv.metrics().await;
    assert_eq!(lane(&m, "helper")["completed"], 1);
    assert_eq!(lane(&m, "background")["deferred"], 1);
    drop(inter);
    drop(bg);
}

/// Agents get their lane from creation; a turn may name another; the busy check still holds.
#[tokio::test(flavor = "multi_thread")]
async fn agent_lanes_and_turn_lane_override() {
    let (fake, trace) = traced(fast());
    let srv = spawn_sched(fake, SchedConfig::default());
    let a: Value = srv.post("/v1/agents", &json!({"lane": "helper"})).await.json().await.unwrap();
    assert_eq!(a["lane"], "helper");
    let id = a["id"].as_str().unwrap();
    let r = srv.post(&format!("/v1/agents/{id}/turn"), &json!({"prompt": [1, 2], "n_predict": 3, "stream": false, "lane": "interactive"})).await;
    assert_eq!(r.status(), 200);
    let m = srv.metrics().await;
    assert_eq!(lane(&m, "interactive")["completed"], 1);
    assert_eq!(lane(&m, "helper")["completed"], 0);
    assert!(!steps_of(&trace).is_empty());
    let bad = srv.post("/completion", &json!({"prompt": [1], "lane": "vip"})).await;
    assert_eq!(bad.status(), 400); // a lane the body type does not know: 400
}
