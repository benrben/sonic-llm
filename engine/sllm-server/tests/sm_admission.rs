//! Admission that waits for room (the executor's `room_wait`), on the fake engine with its test hooks: a request the engine
//! has no room for while others run waits and is asked again; it must be admitted exactly as it would have been the first
//! time (a History agent's rebuild once, a rollover clip once), keep its agent's order, and leave nothing behind when it is
//! refused. The fake's tokens are a pure function of the context, so "the same tokens as an agent that never waited"
//! checks the context.

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::compaction::{CompactionConfig, Rollover};
use sllm_server::fake::{FakeConfig, FakeHooks};
use sllm_server::tokenizer::Tokenizer;
use sllm_server::ServerConfig;
use std::sync::atomic::Ordering::SeqCst;
use std::time::Duration;

const SYSTEM: [i32; 4] = [1, 2, 3, 4];
const TURNS: [&[i32]; 2] = [&[5, 6, 7], &[8, 9]];

fn server(fake: FakeConfig, rollover: bool) -> Srv {
    let compaction = CompactionConfig { rollover: Rollover { on: rollover, ..Default::default() }, ..Default::default() };
    spawn_with(fake, Tokenizer::Bytes, ServerConfig { port: 0, compaction, ..Default::default() })
}

async fn create(srv: &Srv) -> String {
    let a: Value = srv.post("/v1/agents", &json!({"system_prompt_ids": SYSTEM})).await.json().await.unwrap();
    a["id"].as_str().unwrap().to_string()
}

fn turn_body(prompt: &[i32], n: i64) -> Value {
    json!({"prompt": prompt, "n_predict": n, "ignore_eos": true, "stream": false, "return_tokens": true})
}

async fn turn(srv: &Srv, id: &str, prompt: &[i32], n: i64) -> (u16, Value) {
    let r = srv.post(&format!("/v1/agents/{id}/turn"), &turn_body(prompt, n)).await;
    (r.status().as_u16(), r.json().await.unwrap())
}

/// A turn sent from a task of its own (it may wait).
fn turn_task(srv: &Srv, id: &str, prompt: &[i32], n: i64) -> tokio::task::JoinHandle<(u16, Value)> {
    let (http, url, body) = (srv.http.clone(), srv.url(&format!("/v1/agents/{id}/turn")), turn_body(prompt, n));
    tokio::spawn(async move {
        let r = http.post(url).json(&body).send().await.unwrap();
        (r.status().as_u16(), r.json().await.unwrap())
    })
}

fn toks(v: &Value) -> Vec<i64> {
    v["tokens"].as_array().unwrap_or_else(|| panic!("no tokens: {v}")).iter().map(|x| x.as_i64().unwrap()).collect()
}

/// A plain completion that is admitted but held off its slot (the fake's hook), so the executor has a request that holds
/// room; it ends once `release` lets it begin.
async fn holder(srv: &Srv, hooks: &FakeHooks) -> tokio::task::JoinHandle<u16> {
    hooks.hold_completions.store(true, SeqCst);
    let (http, url) = (srv.http.clone(), srv.url("/completion"));
    let h = tokio::spawn(async move { http.post(url).json(&json!({"prompt": [70, 71], "n_predict": 1, "stream": false})).send().await.unwrap().status().as_u16() });
    srv.wait_metrics("the holder admitted", |m| m["active_requests"] == 1).await;
    h
}

/// Refuse every start for room (the fake's hook) until a request waits for room; then the engine has room again.
async fn wait_for_room(srv: &Srv, hooks: &FakeHooks, send: impl FnOnce() -> tokio::task::JoinHandle<(u16, Value)>) -> tokio::task::JoinHandle<(u16, Value)> {
    hooks.refuse_starts.store(usize::MAX, SeqCst);
    let t = send();
    srv.wait_metrics("a request waiting for room", |m| m["scheduler"]["room_waiting"] == 1).await;
    hooks.refuse_starts.store(0, SeqCst);
    t
}

/// The tokens of the two turns on an agent that never waited.
async fn control(srv: &Srv) -> Vec<Vec<i64>> {
    let id = create(srv).await;
    let mut out = Vec::new();
    for t in TURNS {
        out.push(toks(&turn(srv, &id, t, 6).await.1));
    }
    out
}

/// Bug 1A: a History agent's turn is rewritten to re-prefill its history; waiting for room must not rewrite it twice.
#[tokio::test(flavor = "multi_thread")]
async fn a_history_agent_turn_that_waits_for_room_rebuilds_its_history_once() {
    let fake = FakeConfig { token_ms: 1.0, ..fast() };
    let hooks = fake.hooks.clone();
    let srv = server(fake, false);
    let want = control(&srv).await;
    let id = create(&srv).await;
    assert_eq!(toks(&turn(&srv, &id, TURNS[0], 6).await.1), want[0]);
    let r = srv.post(&format!("/v1/agents/{id}/suspend"), &json!({"tier": "history"})).await;
    assert_eq!(r.status(), 200);
    let h = holder(&srv, &hooks).await;
    let t = wait_for_room(&srv, &hooks, || turn_task(&srv, &id, TURNS[1], 6)).await;
    hooks.hold_completions.store(false, SeqCst);
    assert_eq!(h.await.unwrap(), 200);
    let (code, v) = t.await.unwrap();
    assert_eq!(code, 200, "{v}");
    assert_eq!(toks(&v), want[1], "the turn ran on system + turn 1 + its prompt, once: {v}");
    assert_eq!((v["wake"]["via"].clone(), v["wake"]["reprefill_tokens"].clone()), (json!("reprefill"), json!(4 + 3 + 6)), "{v}");
    let info = srv.get_json(&format!("/v1/agents/{id}")).await;
    assert_eq!(info["context_tokens"], 4 + 3 + 6 + 2 + 6, "{info}");
    assert_eq!(info["history_tokens"], info["context_tokens"], "{info}");
}

/// Bug 1C: a turn waiting for room holds its agent's order: a later turn of the same agent queues behind it.
#[tokio::test(flavor = "multi_thread")]
async fn a_turn_waiting_for_room_is_not_overtaken_by_the_next_turn_of_its_agent() {
    let fake = FakeConfig { token_ms: 1.0, ..fast() };
    let hooks = fake.hooks.clone();
    let srv = server(fake, false);
    let want = control(&srv).await;
    let id = create(&srv).await;
    let h = holder(&srv, &hooks).await;
    let first = wait_for_room(&srv, &hooks, || turn_task(&srv, &id, TURNS[0], 6)).await;
    let second = turn_task(&srv, &id, TURNS[1], 6);
    tokio::time::sleep(Duration::from_millis(50)).await;
    hooks.hold_completions.store(false, SeqCst);
    assert_eq!(h.await.unwrap(), 200);
    let (a, b) = (first.await.unwrap(), second.await.unwrap());
    assert_eq!((a.0, b.0), (200, 200), "{a:?} {b:?}");
    assert_eq!(vec![toks(&a.1), toks(&b.1)], want, "the turns ran in the order they arrived");
}

/// Bug 1B: a clipped rollover turn that waits for room keeps its whole reply budget (the clip is made once).
#[tokio::test(flavor = "multi_thread")]
async fn a_clipped_turn_that_waits_for_room_writes_everything_it_was_asked() {
    let fake = FakeConfig { token_ms: 0.02, ..fast() };
    let hooks = fake.hooks.clone();
    let srv = server(fake, true);
    let id = create(&srv).await;
    let h = holder(&srv, &hooks).await;
    let ids: Vec<i32> = (0..32).map(|i| 300 + i).collect();
    let t = wait_for_room(&srv, &hooks, || turn_task(&srv, &id, &ids, 4096 * 2)).await;
    hooks.hold_completions.store(false, SeqCst);
    assert_eq!(h.await.unwrap(), 200);
    let (code, v) = t.await.unwrap();
    assert_eq!(code, 200, "{v}");
    assert_eq!(v["tokens_predicted"], 4096 * 2, "every token asked for was written: {}", v["tokens_predicted"]);
    assert!(v["rollovers"].as_u64().unwrap() >= 1, "{}", v["rollovers"]);
}

/// Bug 1D: a clipped turn refused after its clip leaves no rollover record behind.
#[tokio::test(flavor = "multi_thread")]
async fn a_clipped_turn_that_is_refused_leaves_no_rollover_record() {
    let fake = fast();
    let hooks = fake.hooks.clone();
    let srv = server(fake, true);
    let id = create(&srv).await;
    hooks.refuse_starts.store(1, SeqCst);
    let (code, v) = turn(&srv, &id, &[300, 301, 302], 4096 * 2).await;
    assert_eq!(code, 503, "nothing runs, so no room is no room: {v}");
    let m = srv.metrics().await;
    assert_eq!(m["scheduler"]["rollover_turns"], 0, "{}", m["scheduler"]);
    // and one whose client leaves while it waits for room
    let h = holder(&srv, &hooks).await;
    let t = wait_for_room(&srv, &hooks, || turn_task(&srv, &id, &[300, 301, 302], 4096 * 2)).await;
    t.abort();
    tokio::time::sleep(Duration::from_millis(50)).await;
    hooks.hold_completions.store(false, SeqCst);
    assert_eq!(h.await.unwrap(), 200);
    let m = srv.wait_metrics("the room wait emptied", |m| m["scheduler"]["room_waiting"] == 0 && m["active_requests"] == 0).await;
    assert_eq!(m["scheduler"]["rollover_turns"], 0, "{}", m["scheduler"]);
}

/// LOW 29: a request waiting for room was asked again only when a request ended; a suspend that frees the room it waits
/// for must count too.
#[tokio::test(flavor = "multi_thread")]
async fn a_suspend_that_frees_room_lets_a_waiting_request_in() {
    // 4 pages of 256 tokens: an idle agent holds 2, the holder 1; a 300-token completion of 200 tokens needs 2 more
    let fake = FakeConfig { pool_pages: Some(4), token_ms: 1.0, ..fast() };
    let hooks = fake.hooks.clone();
    let srv = server(fake, false);
    let sys: Vec<i32> = (0..300).map(|i| i % 50 + 1).collect();
    let a: Value = srv.post("/v1/agents", &json!({"system_prompt_ids": sys})).await.json().await.unwrap();
    let id = a["id"].as_str().unwrap().to_string();
    let h = holder(&srv, &hooks).await;
    let (http, url) = (srv.http.clone(), srv.url("/completion"));
    let prompt: Vec<i32> = (0..300).map(|i| i % 40 + 60).collect();
    let c = tokio::spawn(async move { http.post(url).json(&json!({"prompt": prompt, "n_predict": 200, "ignore_eos": true})).send().await.unwrap().status().as_u16() });
    srv.wait_metrics("the completion waiting for room", |m| m["scheduler"]["room_waiting"] == 1).await;
    assert_eq!(srv.post(&format!("/v1/agents/{id}/suspend"), &json!({"tier": "ssd"})).await.status(), 200);
    srv.wait_metrics("the completion admitted after the suspend", |m| m["scheduler"]["room_waiting"] == 0).await;
    hooks.hold_completions.store(false, SeqCst);
    assert_eq!(h.await.unwrap(), 200);
    assert_eq!(c.await.unwrap(), 200);
}
