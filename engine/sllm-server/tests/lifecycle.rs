//! The agent lifecycle over the HTTP API with the fake engine (phase 3): suspend and wake, the history tier and the
//! re-prefill fallback, the idle limit, memory pressure, the per-agent input queue, the durable registry across a
//! restart. The fake's tokens are a pure function of every token an agent has seen, so "the same tokens as an agent
//! that was never suspended" checks that the context survived each path exactly.

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::fake::FakeConfig;
use sllm_server::lifecycle::LifecycleConfig;
use sllm_server::tokenizer::Tokenizer;
use sllm_server::ServerConfig;
use std::time::{Duration, Instant};

fn spawn_life(fake: FakeConfig, life: LifecycleConfig) -> Srv {
    spawn_with(fake, Tokenizer::Bytes, ServerConfig { port: 0, lifecycle: life, ..Default::default() })
}

async fn create(srv: &Srv, system: &[i32]) -> String {
    let a: Value = srv.post("/v1/agents", &json!({"system_prompt_ids": system, "lane": "background"})).await.json().await.unwrap();
    a["id"].as_str().unwrap().to_string()
}

async fn turn(srv: &Srv, id: &str, prompt: &[i32], n: i64) -> Value {
    let r = srv.post(&format!("/v1/agents/{id}/turn"), &json!({"prompt": prompt, "n_predict": n, "ignore_eos": true, "stream": false, "return_tokens": true})).await;
    assert_eq!(r.status(), 200, "turn on {id}");
    r.json().await.unwrap()
}

fn toks(v: &Value) -> Vec<i64> {
    v["tokens"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}

async fn info(srv: &Srv, id: &str) -> Value {
    srv.get_json(&format!("/v1/agents/{id}")).await
}

async fn suspend(srv: &Srv, id: &str, tier: &str) -> Value {
    let r = srv.post(&format!("/v1/agents/{id}/suspend"), &json!({"tier": tier})).await;
    assert_eq!(r.status(), 200);
    r.json().await.unwrap()
}

async fn wait_state(srv: &Srv, id: &str, state: &str) -> Value {
    let t0 = Instant::now();
    loop {
        let i = info(srv, id).await;
        if i["state"] == state {
            return i;
        }
        assert!(t0.elapsed() < Duration::from_secs(5), "{id} never reached {state}: {i}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn life(m: &Value) -> &Value {
    &m["lifecycle"]
}

const TURNS: [&[i32]; 4] = [&[5, 6, 7], &[8, 9], &[10, 11, 12, 13], &[14]];

/// The tokens of four turns on an agent that is never suspended.
async fn control(srv: &Srv) -> Vec<Vec<i64>> {
    let id = create(srv, &[1, 2, 3, 4]).await;
    let mut out = Vec::new();
    for t in TURNS {
        out.push(toks(&turn(srv, &id, t, 6).await));
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn suspend_to_ssd_and_wake_on_the_next_turn_is_exact() {
    let srv = spawn_life(FakeConfig { restore_ms: 20.0, ..fast() }, LifecycleConfig::default());
    let want = control(&srv).await;
    let id = create(&srv, &[1, 2, 3, 4]).await;
    for (k, t) in TURNS.iter().enumerate() {
        let r = turn(&srv, &id, t, 6).await;
        assert_eq!(toks(&r), want[k], "turn {k}");
        if k > 0 {
            assert_eq!(r["wake"]["via"], "ssd", "turn {k} found its KV restored from the SSD: {r}");
            assert!(r["wake"]["restore_ms"].as_f64().unwrap() >= 15.0);
        } else {
            assert_eq!(r["wake"]["via"], "ram");
        }
        let s = suspend(&srv, &id, "ssd").await;
        assert_eq!((s["state"].clone(), s["suspended"].clone()), (json!("suspended"), json!(true)));
        let i = info(&srv, &id).await;
        assert_eq!((i["state"].clone(), i["kv"].clone()), (json!("suspended"), json!("ssd")));
        assert_eq!(i["history_tokens"], i["context_tokens"]);
    }
    let m = srv.metrics().await;
    let l = life(&m);
    assert_eq!(l["agents"]["suspended"], 1);
    assert_eq!(l["suspends"]["api"], 4);
    assert_eq!(l["restores"], 3);
    assert_eq!(l["reprefills_avoided"], 3);
    assert_eq!(l["reprefills"], 0);
    assert!(l["restore_ms_p50"].as_f64().unwrap() >= 15.0 && l["restore_ms_p95"].as_f64().unwrap() >= l["restore_ms_p50"].as_f64().unwrap());
    assert!(l["bytes_written_per_suspend"].as_f64().unwrap() > 0.0);
    assert!(l["ssd_bytes"].as_u64().unwrap() > 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn history_tier_and_a_lost_snapshot_re_prefill_exactly() {
    for (fails, tier) in [(false, "history"), (true, "ssd")] {
        let srv = spawn_life(FakeConfig { restore_fails: fails, ..fast() }, LifecycleConfig::default());
        let want = control(&srv).await;
        let id = create(&srv, &[1, 2, 3, 4]).await;
        let mut hist = 4;
        for (k, t) in TURNS.iter().enumerate() {
            let r = turn(&srv, &id, t, 6).await;
            assert_eq!(toks(&r), want[k], "{tier} fails={fails}: turn {k}");
            if k > 0 {
                assert_eq!(r["wake"]["via"], "reprefill", "{r}");
                assert_eq!(r["wake"]["reprefill_tokens"], hist);
            }
            hist += t.len() + 6;
            assert_eq!(r["context_tokens"], hist);
            suspend(&srv, &id, tier).await;
            if tier == "history" {
                assert_eq!(info(&srv, &id).await["kv"], "none");
            }
        }
        let m = srv.metrics().await;
        assert_eq!(life(&m)["reprefills"], 3);
        if fails {
            assert_eq!(life(&m)["restore_failures"], 3);
        } else {
            assert_eq!(life(&m)["history_drops"], 4);
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn idle_agents_are_suspended_after_the_idle_limit_and_wake_restores() {
    let srv = spawn_life(FakeConfig { restore_ms: 30.0, ..fast() }, LifecycleConfig { idle_suspend_ms: Some(60), ..Default::default() });
    let id = create(&srv, &[1, 2, 3]).await;
    turn(&srv, &id, &[4], 3).await;
    assert_eq!(info(&srv, &id).await["state"], "idle");
    wait_state(&srv, &id, "suspended").await;
    assert_eq!(life(&srv.metrics().await)["suspends"]["idle"], 1);
    // a wake call starts the restore before any turn (the tool result is about to arrive)
    let w: Value = srv.post(&format!("/v1/agents/{id}/wake"), &json!({})).await.json().await.unwrap();
    assert_eq!(w["state"], "restoring");
    wait_state(&srv, &id, "idle").await;
    let r = turn(&srv, &id, &[5], 2).await;
    assert_eq!(r["wake"]["via"], "ssd");
}

#[tokio::test(flavor = "multi_thread")]
async fn memory_pressure_suspends_the_coldest_idle_agents() {
    // 12 pages of 256 tokens; each agent holds 2 pages (300 tokens)
    let fake = FakeConfig { pool_pages: Some(12), ..fast() };
    let srv = spawn_life(fake, LifecycleConfig { pressure_low: 0.0, pressure_high: 0.0, relieve_on_refusal: true, ..Default::default() });
    let sys: Vec<i32> = (0..300).map(|i| i % 50 + 1).collect();
    let mut ids = Vec::new();
    for k in 0..5 {
        let id = create(&srv, &sys).await;
        turn(&srv, &id, &[k + 1], 1).await;
        ids.push(id);
    }
    // 10 pages held by idle agents; a turn that needs 3 more pages makes room from the coldest (the first created)
    let id6 = create(&srv, &[]).await;
    let long: Vec<i32> = (0..600).map(|i| i % 40 + 1).collect();
    let r = turn(&srv, &id6, &long, 4).await;
    assert_eq!(r["wake"]["via"], "ram");
    assert_eq!(info(&srv, &ids[0]).await["state"], "suspended", "the coldest idle agent made room");
    assert_eq!(info(&srv, &ids[4]).await["state"], "idle", "the warmest stayed in RAM");
    let m = srv.metrics().await;
    assert!(life(&m)["suspends"]["pressure"].as_u64().unwrap() >= 1);
    // the suspended agent's next turn restores it (making room again if it must)
    let r = turn(&srv, &ids[0], &[9], 2).await;
    assert_eq!(r["wake"]["via"], "ssd");
    drop(srv);
    // the proactive policy: free share under 30% suspends idle agents until 50% is free
    let fake = FakeConfig { pool_pages: Some(12), ..fast() };
    let srv = spawn_life(fake, LifecycleConfig { pressure_low: 0.30, pressure_high: 0.50, ..Default::default() });
    let mut ids = Vec::new();
    for k in 0..5 {
        let id = create(&srv, &sys).await;
        turn(&srv, &id, &[k + 1], 1).await;
        ids.push(id);
    }
    let t0 = Instant::now();
    loop {
        let m = srv.metrics().await;
        let free = 12 - m["pages_active"].as_u64().unwrap();
        if free as f64 >= 0.5 * 12.0 {
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(5), "pressure never relieved: {}", life(&m));
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(info(&srv, &ids[0]).await["state"], "suspended");
}

#[tokio::test(flavor = "multi_thread")]
async fn input_queue_holds_turns_in_order_and_refuses_past_its_cap() {
    let srv = spawn_life(FakeConfig { token_ms: 5.0, restore_ms: 20.0, ..fast() }, LifecycleConfig { inbox_cap: 2, ..Default::default() });
    let want = control(&srv).await;
    let id = create(&srv, &[1, 2, 3, 4]).await;
    turn(&srv, &id, TURNS[0], 6).await;
    suspend(&srv, &id, "ssd").await;
    // three turns at once on a suspended agent: the first starts the restore and waits, the second queues behind it,
    // the third finds the queue full
    let mut hs = Vec::new();
    for t in &TURNS[1..3] {
        let (http, url) = (srv.http.clone(), srv.url(&format!("/v1/agents/{id}/turn")));
        let body = json!({"prompt": t, "n_predict": 6, "ignore_eos": true, "stream": false, "return_tokens": true});
        hs.push(tokio::spawn(async move { http.post(url).json(&body).send().await.unwrap().json::<Value>().await.unwrap() }));
        let t0 = Instant::now();
        let n = hs.len();
        while info(&srv, &id).await["queued_turns"].as_u64().unwrap() < n as u64 {
            assert!(t0.elapsed() < Duration::from_secs(5));
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }
    let full = srv.post(&format!("/v1/agents/{id}/turn"), &json!({"prompt": [1], "n_predict": 1})).await;
    assert_eq!(full.status(), 409);
    let mut got = Vec::new();
    for h in hs {
        got.push(toks(&h.await.unwrap()));
    }
    assert_eq!(got, want[1..3].to_vec(), "queued turns ran in arrival order on the restored context");
    let m = srv.metrics().await;
    assert_eq!((life(&m)["turns_queued"].clone(), life(&m)["inbox_full"].clone()), (json!(2), json!(1)));
    // deleting an agent refuses what waits for it
    suspend(&srv, &id, "ssd").await;
    let (http, url) = (srv.http.clone(), srv.url(&format!("/v1/agents/{id}/turn")));
    let h = tokio::spawn(async move { http.post(url).json(&json!({"prompt": [1], "n_predict": 1})).send().await.unwrap().status() });
    tokio::time::sleep(Duration::from_millis(2)).await;
    assert_eq!(srv.http.delete(srv.url(&format!("/v1/agents/{id}"))).send().await.unwrap().status(), 200);
    let st = h.await.unwrap();
    assert!(st == 404 || st == 200, "a turn queued on a deleted agent is refused (or ran first): {st}");
}

#[tokio::test(flavor = "multi_thread")]
async fn registry_survives_a_restart_and_the_history_is_the_truth() {
    let dir = std::env::temp_dir().join(format!("sllm-lifecycle-registry-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = || LifecycleConfig { registry_dir: Some(dir.join("registry")), ..Default::default() };
    let (want, id, hist) = {
        let srv = spawn_life(fast(), cfg());
        let want = control(&srv).await;
        let id = create(&srv, &[1, 2, 3, 4]).await;
        for t in &TURNS[..2] {
            turn(&srv, &id, t, 6).await;
        }
        let i = info(&srv, &id).await;
        suspend(&srv, &id, "ssd").await;
        srv.run.shutdown();
        (want, id, i["history_tokens"].clone())
    };
    let srv = spawn_life(fast(), cfg());
    let i = info(&srv, &id).await;
    assert_eq!((i["state"].clone(), i["history_tokens"].clone(), i["turns"].clone()), (json!("history"), hist.clone(), json!(2)), "{i}");
    let list = srv.get_json("/v1/agents").await;
    assert_eq!(list["data"].as_array().unwrap().len(), 2, "the control agent came back too");
    for (k, t) in TURNS.iter().enumerate().skip(2) {
        let r = turn(&srv, &id, t, 6).await;
        assert_eq!(toks(&r), want[k], "turn {k} after the restart");
        if k == 2 {
            assert_eq!((r["wake"]["via"].clone(), r["wake"]["reprefill_tokens"].clone()), (json!("reprefill"), hist.clone()));
        }
    }
    // new agents do not reuse a restored id
    let fresh = create(&srv, &[]).await;
    assert_ne!(fresh, id);
    assert_eq!(life(&srv.metrics().await)["registry"]["loaded"], 2);
    assert_eq!(srv.http.delete(srv.url(&format!("/v1/agents/{id}"))).send().await.unwrap().status(), 200);
    assert!(!dir.join("registry").join(format!("{id}.json")).exists(), "a deleted agent leaves the registry");
    srv.run.shutdown();
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn suspend_on_a_busy_agent_applies_when_its_turn_ends() {
    let srv = spawn_life(FakeConfig { token_ms: 5.0, ..fast() }, LifecycleConfig::default());
    let id = create(&srv, &[1]).await;
    let (http, url) = (srv.http.clone(), srv.url(&format!("/v1/agents/{id}/turn")));
    let h = tokio::spawn(async move { http.post(url).json(&json!({"prompt": [2], "n_predict": 20, "ignore_eos": true, "stream": false})).send().await.unwrap().status() });
    let t0 = Instant::now();
    while info(&srv, &id).await["busy"] != true {
        assert!(t0.elapsed() < Duration::from_secs(5));
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let s = suspend(&srv, &id, "ssd").await;
    assert_eq!(s["deferred"], true);
    assert_eq!(h.await.unwrap(), 200);
    wait_state(&srv, &id, "suspended").await;
    // unknown agents
    assert_eq!(srv.post("/v1/agents/ag_999/suspend", &json!({})).await.status(), 404);
    assert_eq!(srv.post("/v1/agents/ag_999/wake", &json!({})).await.status(), 404);
    assert_eq!(srv.post(&format!("/v1/agents/{id}/suspend"), &json!({"tier": "tape"})).await.status(), 400); // a value the body type does not know: 400
}

/// Compaction and the lifecycle (phase 3): a compaction rewrites the window, and the history follows it, so a
/// re-prefill or a snapshot holds the compacted window; a suspended agent is not compacted until it is back in RAM.
#[tokio::test(flavor = "multi_thread")]
async fn compacted_window_is_what_suspend_and_reprefill_keep() {
    let srv = spawn_life(fast(), LifecycleConfig::default());
    let sys: Vec<i32> = (1..=10).collect();
    let long: Vec<i32> = (0..3000).map(|i| i % 90 + 100).collect();
    let mut outs = Vec::new();
    for tier in ["none", "history", "ssd"] {
        let id = create(&srv, &sys).await;
        turn(&srv, &id, &long, 4).await;
        let r = srv.post(&format!("/v1/agents/{id}/compact"), &json!({"summary_ids": [9, 9, 9], "tail_tokens": 500})).await;
        assert_eq!(r.status(), 200);
        let rep: Value = r.json().await.unwrap();
        assert_eq!(rep["context_after"], 10 + 3 + 500, "{rep}");
        let i = info(&srv, &id).await;
        assert_eq!(i["history_tokens"], i["context_tokens"], "the history is the compacted window: {i}");
        if tier != "none" {
            suspend(&srv, &id, tier).await;
            let r = srv.post(&format!("/v1/agents/{id}/compact"), &json!({"summary_ids": [9], "tail_tokens": 100})).await;
            assert_eq!(r.status(), 409, "no compaction while the KV is not in RAM");
        }
        let t = turn(&srv, &id, &[7, 7], 6).await;
        match tier {
            "history" => assert_eq!((t["wake"]["via"].clone(), t["wake"]["reprefill_tokens"].clone()), (json!("reprefill"), json!(513))),
            "ssd" => assert_eq!(t["wake"]["via"], "ssd"),
            _ => assert_eq!(t["wake"]["via"], "ram"),
        }
        outs.push(toks(&t));
    }
    assert_eq!(outs[0], outs[1], "re-prefill of the compacted window");
    assert_eq!(outs[0], outs[2], "restore of the compacted window");
}
