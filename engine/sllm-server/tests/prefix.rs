//! Prefix sharing in the scheduler, through the HTTP API on the FakeEngine (its `prefix_page` model): a request whose
//! prefix another running request is prefilling waits on its pages (holds its slot, is not planned), attaches when they
//! are fed and prefills only its own part; cancellation and deadlines apply while it waits; a waiter whose writer leaves
//! falls back to a full prefill.

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::engine::{ReqId, Work, WorkItem};
use sllm_server::fake::{FakeConfig, StepTrace};
use sllm_server::sched::SchedConfig;
use sllm_server::tokenizer::Tokenizer;
use sllm_server::ServerConfig;
use std::sync::{Arc, Mutex};

const PAGE: usize = 16;
const PREFIX: i32 = 64;
const OWN: i32 = 20;

/// A prompt: the shared 64-token prefix (4 fake pages) and 20 tokens of its own.
fn prompt(i: i32) -> Vec<i32> {
    (1..=PREFIX).chain(1000 * (i + 1)..1000 * (i + 1) + OWN).collect()
}

/// Slow prefill (4 ms per token, 16-token chunks): the writer takes ~256 ms to feed the shared prefix.
fn slow(trace: &StepTrace, sharing: bool) -> Srv {
    let fake = FakeConfig {
        prefill_ms_per_token: 4.0,
        token_ms: 1.0,
        prefix_page: sharing.then_some(PAGE),
        trace: Some(trace.clone()),
        ..fast()
    };
    let sched = SchedConfig { chunk: 16, step_tokens: 64, helper_tokens: 16, ..Default::default() };
    spawn_with(fake, Tokenizer::Bytes, ServerConfig { port: 0, sched, ..Default::default() })
}

fn new_trace() -> StepTrace {
    Arc::new(Mutex::new(Vec::new()))
}

fn req_id(r: &reqwest::Response) -> ReqId {
    r.headers()["x-request-id"].to_str().unwrap().parse().unwrap()
}

async fn open(srv: &Srv, p: &[i32], n: i64, extra: Value) -> reqwest::Response {
    let mut b = json!({"prompt": p, "n_predict": n, "stream": true, "ignore_eos": true, "return_tokens": true});
    for (k, v) in extra.as_object().into_iter().flatten() {
        b[k] = v.clone();
    }
    let r = srv.post("/completion", &b).await;
    assert_eq!(r.status(), 200);
    r
}

async fn drain(mut r: reqwest::Response) -> (Vec<i64>, Value) {
    let mut raw = String::new();
    while let Some(c) = r.chunk().await.unwrap() {
        raw.push_str(std::str::from_utf8(&c).unwrap());
    }
    let chunks = sse_json(&raw);
    (tokens_of(&chunks), chunks.last().cloned().unwrap_or(Value::Null))
}

fn prefilled(steps: &[Vec<WorkItem>], req: ReqId) -> usize {
    steps.iter().flatten().filter(|w| w.req == req).map(|w| if let Work::Prefill { n } = w.work { n } else { 0 }).sum()
}

fn first_step(steps: &[Vec<WorkItem>], req: ReqId) -> Option<usize> {
    steps.iter().position(|s| s.iter().any(|w| w.req == req))
}

/// Solo tokens of each prompt (no sharing), the reference.
async fn solo(n: usize) -> Vec<Vec<i64>> {
    let srv = spawn(fast());
    let mut v = Vec::new();
    for i in 0..n as i32 {
        let c = srv.complete(&prompt(i), 10, json!({})).await;
        v.push(c["tokens"].as_array().unwrap().iter().map(|t| t.as_i64().unwrap()).collect());
    }
    v
}

/// Coalesced arrival: three requests arrive while the first prefills the shared prefix. They wait on its pages (held
/// slots, not planned, in /metrics), attach once the 64 tokens are fed, and prefill only their own 20 tokens.
#[tokio::test(flavor = "multi_thread")]
async fn waiters_hold_their_slot_attach_and_prefill_only_their_part() {
    let reference = solo(4).await;
    let trace = new_trace();
    let srv = slow(&trace, true);
    let w = open(&srv, &prompt(0), 10, json!({})).await;
    let wid = req_id(&w);
    srv.wait_metrics("the writer has begun its prefill", |m| m["scheduler"]["steps"].as_u64().unwrap() >= 1).await;
    // all three at once (the executor takes commands at step boundaries)
    let (p1, p2, p3) = (prompt(1), prompt(2), prompt(3));
    let (a, b, c) = tokio::join!(open(&srv, &p1, 10, json!({})), open(&srv, &p2, 10, json!({})), open(&srv, &p3, 10, json!({})));
    let rs = vec![a, b, c];
    let ids: Vec<ReqId> = rs.iter().map(req_id).collect();
    let m = srv.wait_metrics("three waiters", |m| m["scheduler"]["waiting_on_pages"] == 3).await;
    assert_eq!(m["scheduler"]["lanes"]["background"]["waiting"], 3, "{m}");
    assert_eq!(m["scheduler"]["lanes"]["background"]["running"], 4, "waiters hold their slots");
    assert_eq!(m["prefix"]["waiting"], 3);
    let (t0, f0) = drain(w).await;
    assert_eq!(t0, reference[0]);
    assert_eq!(f0["timings"]["cache_n"], 0);
    for (k, r) in rs.into_iter().enumerate() {
        let (t, f) = drain(r).await;
        assert_eq!(t, reference[k + 1], "waiter {k}: same tokens as alone");
        assert_eq!((f["timings"]["cache_n"].as_u64(), f["timings"]["prompt_n"].as_u64()), (Some(64), Some(20)), "{f}");
        assert_eq!(f["tokens_evaluated"], 84);
    }
    let steps = trace.lock().unwrap().clone();
    // the writer fed its whole prompt; each waiter only its own part, and none ran before the writer had fed the prefix
    assert_eq!(prefilled(&steps, wid), 84);
    let fed_prefix_at = {
        let mut fed = 0;
        steps.iter().position(|s| {
            fed += prefilled(std::slice::from_ref(s), wid);
            fed >= PREFIX as usize
        })
    }
    .unwrap();
    for &id in &ids {
        assert_eq!(prefilled(&steps, id), OWN as usize, "request {id} prefilled only its own tokens");
        assert!(first_step(&steps, id).unwrap() > fed_prefix_at, "request {id} was planned while it waited");
    }
    let m = srv.metrics().await;
    assert_eq!((m["prefix"]["waited_agents"].as_u64(), m["prefix"]["attached_agents"].as_u64()), (Some(3), Some(3)), "{m}");
    assert_eq!(m["scheduler"]["prefix_waited"], 3);
    assert_eq!(m["scheduler"]["prefix_shared_tokens"], 3 * 64);
    assert_eq!(m["scheduler"]["waiting_on_pages"], 0);
    let caps = srv.get_json("/caps").await;
    assert_eq!(caps["prefix_sharing"]["mode"], "on (modelled)");
    // a later request with the prefix while a holder is still decoding attaches at once (no wait)
    let hold = open(&srv, &prompt(0), 2000, json!({})).await;
    srv.wait_metrics("the holder decodes", |m| m["tokens_generated"].as_u64().unwrap() >= 41).await;
    let c = srv.complete(&prompt(5), 4, json!({})).await;
    assert_eq!(c["timings"]["cache_n"], 64);
    assert_eq!(srv.metrics().await["scheduler"]["prefix_waited"], 3, "it did not wait");
    drop(hold);
}

/// A waiter is cancelled (DELETE /v1/requests/{id}) or expires (deadline_ms) while it waits: it ends at the next step
/// boundary with no tokens, frees its slot, and the writer is unaffected. Neither was ever planned.
#[tokio::test(flavor = "multi_thread")]
async fn a_waiter_can_be_cancelled_or_expire_while_it_waits() {
    let reference = solo(1).await;
    let trace = new_trace();
    let srv = slow(&trace, true);
    let w = open(&srv, &prompt(0), 10, json!({})).await;
    srv.wait_metrics("the writer has begun its prefill", |m| m["scheduler"]["steps"].as_u64().unwrap() >= 1).await;
    let (p1, p2) = (prompt(1), prompt(2));
    let (c, d) = tokio::join!(open(&srv, &p1, 10, json!({})), open(&srv, &p2, 10, json!({"deadline_ms": 15})));
    let (cid, did) = (req_id(&c), req_id(&d));
    srv.wait_metrics("a waiter", |m| m["scheduler"]["waiting_on_pages"].as_u64().unwrap() >= 1).await;
    let r: Value = srv.http.delete(srv.url(&format!("/v1/requests/{cid}"))).send().await.unwrap().json().await.unwrap();
    assert_eq!(r["cancelled"], true);
    let (tc, fc) = drain(c).await;
    assert_eq!((tc.len(), fc["stop_type"].as_str()), (0, Some("cancelled")));
    let (td, fd) = drain(d).await;
    assert_eq!((td.len(), fd["stop_type"].as_str()), (0, Some("deadline")));
    // the scheduler snapshot is published at the next step boundary
    let m = srv.wait_metrics("no waiter left", |m| m["scheduler"]["waiting_on_pages"] == 0).await;
    assert_eq!(m["requests"]["expired"], 1);
    assert_eq!(m["scheduler"]["lanes"]["background"]["running"], 1, "only the writer holds a slot: {m}");
    let (tw, _) = drain(w).await;
    assert_eq!(tw, reference[0], "the writer is unaffected");
    let steps = trace.lock().unwrap().clone();
    assert!(first_step(&steps, cid).is_none() && first_step(&steps, did).is_none(), "a waiter is never planned");
    assert_eq!(srv.metrics().await["active_requests"], 0);
}

/// The writer leaves (cancelled) while a request waits on its pages: the waiter falls back to a full prefill of its own
/// and gives the same tokens as alone.
#[tokio::test(flavor = "multi_thread")]
async fn a_waiter_whose_writer_leaves_prefills_everything_itself() {
    let reference = solo(2).await;
    let trace = new_trace();
    let srv = slow(&trace, true);
    let w = open(&srv, &prompt(0), 10, json!({})).await;
    let wid = req_id(&w);
    srv.wait_metrics("the writer has begun its prefill", |m| m["scheduler"]["steps"].as_u64().unwrap() >= 1).await;
    let v = open(&srv, &prompt(1), 10, json!({})).await;
    let vid = req_id(&v);
    srv.wait_metrics("the waiter", |m| m["scheduler"]["waiting_on_pages"] == 1).await;
    let r: Value = srv.http.delete(srv.url(&format!("/v1/requests/{wid}"))).send().await.unwrap().json().await.unwrap();
    assert_eq!(r["cancelled"], true);
    let (_, fw) = drain(w).await;
    assert_eq!(fw["stop_type"], "cancelled");
    let (tv, fv) = drain(v).await;
    assert_eq!(tv, reference[1], "same tokens as alone");
    assert_eq!((fv["timings"]["cache_n"].as_u64(), fv["timings"]["prompt_n"].as_u64()), (Some(0), Some(84)), "{fv}");
    let steps = trace.lock().unwrap().clone();
    assert_eq!(prefilled(&steps, vid), 84, "a full prefill of its own");
    assert!(first_step(&steps, vid).unwrap() > steps.iter().rposition(|s| s.iter().any(|w| w.req == wid)).unwrap());
    let m = srv.metrics().await;
    assert_eq!((m["prefix"]["attached_agents"].as_u64(), m["scheduler"]["waiting_on_pages"].as_u64()), (Some(0), Some(0)), "{m}");
}

/// With sharing off (no prefix model) nothing waits and every request prefills its whole prompt: the same tokens.
#[tokio::test(flavor = "multi_thread")]
async fn without_sharing_nothing_waits() {
    let reference = solo(3).await;
    let trace = new_trace();
    let srv = slow(&trace, false);
    let mut rs = Vec::new();
    for i in 0..3 {
        rs.push(open(&srv, &prompt(i), 10, json!({})).await);
    }
    let ids: Vec<ReqId> = rs.iter().map(req_id).collect();
    for (k, r) in rs.into_iter().enumerate() {
        let (t, f) = drain(r).await;
        assert_eq!(t, reference[k]);
        assert_eq!(f["timings"]["cache_n"], 0);
    }
    let steps = trace.lock().unwrap().clone();
    assert!(ids.iter().all(|&id| prefilled(&steps, id) == 84));
    let m = srv.metrics().await;
    assert!(m["prefix"].is_null(), "no prefix counters without the model: {m}");
    assert_eq!(m["scheduler"]["prefix_waited"], 0);
}
