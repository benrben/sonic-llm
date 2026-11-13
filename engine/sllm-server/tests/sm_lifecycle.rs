//! The executor's agent lifecycle against the engine's snapshot events, on the fake engine with its test hooks: a snapshot
//! write that fails, while the agent is suspended (the next engine keeps its host copy) or while a wake is already reading
//! it back (libllama loses it and no restore event comes). Either way the agent must end where its next turn works: a
//! re-prefill of its history, with exactly the tokens of an agent that was never suspended.

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::fake::FakeConfig;
use sllm_server::lifecycle::LifecycleConfig;
use sllm_server::tokenizer::Tokenizer;
use sllm_server::ServerConfig;
use std::sync::atomic::Ordering::SeqCst;
use std::time::{Duration, Instant};

const TURNS: [&[i32]; 3] = [&[5, 6, 7], &[8, 9], &[10, 11, 12]];

fn spawn_life(fake: FakeConfig) -> Srv {
    spawn_with(fake, Tokenizer::Bytes, ServerConfig { port: 0, lifecycle: LifecycleConfig::default(), ..Default::default() })
}

async fn create(srv: &Srv) -> String {
    let a: Value = srv.post("/v1/agents", &json!({"system_prompt_ids": [1, 2, 3, 4]})).await.json().await.unwrap();
    a["id"].as_str().unwrap().to_string()
}

/// A turn, given at most 5 s (an agent stuck in a state never answers).
async fn turn(srv: &Srv, id: &str, prompt: &[i32]) -> (u16, Value) {
    let body = json!({"prompt": prompt, "n_predict": 6, "ignore_eos": true, "stream": false, "return_tokens": true});
    let r = tokio::time::timeout(Duration::from_secs(5), srv.post(&format!("/v1/agents/{id}/turn"), &body)).await;
    let r = r.unwrap_or_else(|_| panic!("the turn on {id} never answered (its agent is stuck)"));
    (r.status().as_u16(), r.json().await.unwrap())
}

fn toks(v: &Value) -> Vec<i64> {
    v["tokens"].as_array().unwrap_or_else(|| panic!("no tokens: {v}")).iter().map(|x| x.as_i64().unwrap()).collect()
}

async fn control(srv: &Srv) -> Vec<Vec<i64>> {
    let id = create(srv).await;
    let mut out = Vec::new();
    for t in TURNS {
        out.push(toks(&turn(srv, &id, t).await.1));
    }
    out
}

async fn state(srv: &Srv, id: &str) -> Value {
    srv.get_json(&format!("/v1/agents/{id}")).await["state"].clone()
}

async fn wait_state(srv: &Srv, id: &str, want: &str) {
    let t0 = Instant::now();
    while state(srv, id).await != want {
        assert!(t0.elapsed() < Duration::from_secs(5), "{id} never reached {want}: {}", state(srv, id).await);
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Bug 4B: a failed write leaves the next engine's agent suspended (it keeps the host copy); the executor moved the agent
/// to History without telling the engine, so every later turn was refused 409 agent_suspended.
#[tokio::test(flavor = "multi_thread")]
async fn a_failed_snapshot_write_leaves_an_agent_whose_next_turn_works() {
    let fake = fast();
    let hooks = fake.hooks.clone();
    let srv = spawn_life(fake);
    let want = control(&srv).await;
    let id = create(&srv).await;
    assert_eq!(toks(&turn(&srv, &id, TURNS[0]).await.1), want[0]);
    hooks.fail_writes.store(1, SeqCst);
    assert_eq!(srv.post(&format!("/v1/agents/{id}/suspend"), &json!({"tier": "ssd"})).await.status(), 200);
    wait_state(&srv, &id, "history").await;
    let (code, v) = turn(&srv, &id, TURNS[1]).await;
    assert_eq!(code, 200, "{v}");
    assert_eq!(toks(&v), want[1], "re-prefilled from the history: {v}");
    assert_eq!(v["wake"]["via"], "reprefill", "{v}");
    assert_eq!(srv.metrics().await["lifecycle"]["write_failures"], 1);
}

/// Bug 4A: a wake starts while the snapshot is still being written, then the write fails (libllama: the snapshot is gone
/// and no restore event comes). The agent was left Restoring for good, its queued turn never ran.
#[tokio::test(flavor = "multi_thread")]
async fn a_write_that_fails_under_a_wake_does_not_leave_the_agent_restoring() {
    let fake = FakeConfig { restore_ms: 5.0, ..fast() };
    let hooks = fake.hooks.clone();
    let srv = spawn_life(fake);
    let want = control(&srv).await;
    let id = create(&srv).await;
    assert_eq!(toks(&turn(&srv, &id, TURNS[0]).await.1), want[0]);
    hooks.hold_writes.store(true, SeqCst);
    assert_eq!(srv.post(&format!("/v1/agents/{id}/suspend"), &json!({"tier": "ssd"})).await.status(), 200);
    let w: Value = srv.post(&format!("/v1/agents/{id}/wake"), &json!({})).await.json().await.unwrap();
    assert_eq!(w["state"], "restoring", "{w}");
    hooks.fail_writes.store(1, SeqCst);
    hooks.hold_writes.store(false, SeqCst);
    let (code, v) = turn(&srv, &id, TURNS[1]).await;
    assert_eq!(code, 200, "{v}");
    assert_eq!(toks(&v), want[1], "re-prefilled from the history: {v}");
    assert_eq!(v["wake"]["via"], "reprefill", "{v}");
    // the wake that was called off left no clock running: the next wake measures its own restore
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(srv.post(&format!("/v1/agents/{id}/suspend"), &json!({"tier": "ssd"})).await.status(), 200);
    let (code, v) = turn(&srv, &id, TURNS[2]).await;
    assert_eq!(code, 200, "{v}");
    assert_eq!(toks(&v), want[2]);
    assert_eq!(v["wake"]["via"], "ssd", "{v}");
    assert!(v["wake"]["restore_ms"].as_f64().unwrap() < 250.0, "a stale wake clock: {v}");
}
