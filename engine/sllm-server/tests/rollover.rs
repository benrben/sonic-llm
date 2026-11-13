//! Rollover over the HTTP API, on the fake engine: a turn is not stopped by the end of the window, a prompt longer than the
//! window is read, a turn that fits is the same turn as without rollover. What the notes say is in the real runs.

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::compaction::{CompactionConfig, Rollover};
use sllm_server::fake::FakeConfig;
use sllm_server::tokenizer::Tokenizer;
use sllm_server::ServerConfig;

const CTX: usize = 4096;

/// Tokens a client can keep up with: the server cancels a client that stops reading, whatever the window.
fn paced() -> FakeConfig {
    FakeConfig { token_ms: 0.02, ..fast() }
}

fn srv(on: bool) -> Srv {
    let cfg = ServerConfig { port: 0, compaction: CompactionConfig { rollover: Rollover { on, ..Default::default() }, ..Default::default() }, ..Default::default() };
    spawn_with(FakeConfig { n_ctx: CTX, ..paced() }, Tokenizer::Bytes, cfg)
}

async fn agent(s: &Srv) -> String {
    let r: Value = s.post("/v1/agents", &json!({"system_prompt_ids": [1, 2, 3, 4]})).await.json().await.unwrap();
    r["id"].as_str().unwrap().to_string()
}

async fn turn(s: &Srv, a: &str, ids: Vec<i32>, n: i64) -> (u16, Value) {
    let r = s.post(&format!("/v1/agents/{a}/turn"), &json!({"prompt": ids, "n_predict": n, "stream": false, "ignore_eos": true, "return_tokens": true})).await;
    let code = r.status().as_u16();
    (code, r.json().await.unwrap())
}

fn ids(n: usize) -> Vec<i32> {
    (0..n as i32).map(|i| 300 + i % 500).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_turn_writes_past_the_end_of_the_window() {
    let s = srv(true);
    let a = agent(&s).await;
    let (code, v) = turn(&s, &a, ids(32), (CTX * 3) as i64).await;
    assert_eq!(code, 200, "{v}");
    assert_eq!(v["tokens_predicted"], CTX * 3, "every token asked for was written: {v}");
    assert!(v["rollovers"].as_u64().unwrap() >= 3, "{v}");
    assert_eq!(v["tokens"].as_array().unwrap().len(), CTX * 3);
    assert!(v["context_tokens"].as_u64().unwrap() < CTX as u64);
    let info = s.get_json(&format!("/v1/agents/{a}")).await;
    assert_eq!(info["compaction"]["rollovers"], v["rollovers"], "{info}");
    assert_eq!(info["compaction"]["last"]["summary_source"], "brain", "{info}");
    assert_eq!(info["compaction"]["last"]["trigger"], "limit", "{info}");
    let m = s.metrics().await;
    assert_eq!(m["active_requests"], 0, "a rollover leaves no request behind: {m}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_turn_on_a_full_window_is_not_refused() {
    let s = srv(true);
    let a = agent(&s).await;
    for k in 0..6 {
        let (code, v) = turn(&s, &a, ids(900), 300).await;
        assert_eq!(code, 200, "turn {k}: {v}");
        assert_eq!(v["tokens_predicted"], 300, "turn {k}: {v}");
    }
    let info = s.get_json(&format!("/v1/agents/{a}")).await;
    assert!(info["compaction"]["rollovers"].as_u64().unwrap() >= 1, "{info}");
    assert_eq!(info["turns"].as_u64().unwrap() >= 6, true, "{info}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_prompt_longer_than_the_window_is_read() {
    let s = srv(true);
    let a = agent(&s).await;
    let (code, v) = turn(&s, &a, ids(CTX * 3 + 17), 40).await;
    assert_eq!(code, 200, "{v}");
    assert_eq!(v["tokens_predicted"], 40, "{v}");
    assert!(v["rollovers"].as_u64().unwrap() >= 3, "{v}");
    let m = s.metrics().await;
    assert_eq!(m["active_requests"], 0, "{m}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_turn_that_fits_is_the_same_turn() {
    let (on, off) = (srv(true), srv(false));
    let (a, b) = (agent(&on).await, agent(&off).await);
    for k in 0..3 {
        let (ca, x) = turn(&on, &a, ids(50 + k), 60).await;
        let (cb, y) = turn(&off, &b, ids(50 + k), 60).await;
        assert_eq!((ca, cb), (200, 200));
        assert_eq!(x["tokens"], y["tokens"], "turn {k}");
        assert!(x.get("rollovers").is_none(), "a turn that fits has llama-server's keys: {x}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn lines_the_window_lost_come_back_in_a_later_turn() {
    let s = srv(true);
    let a = agent(&s).await;
    let fact: Vec<i32> = "The code of project Wren is 2613.\n".bytes().map(i32::from).collect();
    let (code, v) = turn(&s, &a, fact.clone(), 8).await;
    assert_eq!(code, 200, "{v}");
    let (code, v) = turn(&s, &a, ids(32), (CTX * 2) as i64).await;
    assert_eq!(code, 200);
    assert!(v["rollovers"].as_u64().unwrap() >= 1, "{v}");
    let q: Vec<i32> = "What is the code of project Wren?".bytes().map(i32::from).collect();
    let (code, v) = turn(&s, &a, q.clone(), 4).await;
    assert_eq!(code, 200, "{v}");
    let info = s.get_json(&format!("/v1/agents/{a}")).await;
    assert_eq!(info["compaction"]["recalled"], 1, "{info}");
    assert!(v["tokens_evaluated"].as_u64().unwrap() as usize > q.len() + fact.len() - 2, "the line is in the prompt: {v}");
}

#[tokio::test(flavor = "multi_thread")]
async fn recalled_lines_never_push_a_turn_over_the_edge() {
    // a prompt that fits only just, in every position of the window, with lines to recall: every turn answers
    let s = srv(true);
    let a = agent(&s).await;
    let fact: Vec<i32> = "The code of project Wren is 2613.\n".bytes().map(i32::from).collect();
    let q: Vec<i32> = "What is the code of project Wren?".bytes().map(i32::from).collect();
    let (code, v) = turn(&s, &a, fact, 4).await;
    assert_eq!(code, 200, "{v}");
    for k in 0..200 {
        let prompt: Vec<i32> = ids(40 + 37 * (k % 90)).into_iter().chain(q.iter().copied()).collect();
        let (code, v) = turn(&s, &a, prompt, 4).await;
        assert_eq!(code, 200, "turn {k}: {v}");
    }
    let m = s.metrics().await;
    assert_eq!(m["active_requests"], 0, "{m}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_small_window_rolls_over_and_gets_on_with_the_work() {
    // the fake tokenizer makes a token of every byte, so this is the smallest window its long frames leave room in
    let cfg = ServerConfig { port: 0, compaction: CompactionConfig { rollover: Rollover { on: true, ..Default::default() }, ..Default::default() }, ..Default::default() };
    let s = spawn_with(FakeConfig { n_ctx: 2048, ..paced() }, Tokenizer::Bytes, cfg);
    let a = agent(&s).await;
    let (code, v) = turn(&s, &a, ids(32), 6000).await;
    assert_eq!(code, 200, "{v}");
    assert_eq!(v["tokens_predicted"], 6000, "{v}");
    assert!(v["rollovers"].as_u64().unwrap() >= 2, "{v}");
    let (code, v) = turn(&s, &a, ids(5000), 8).await;
    assert_eq!(code, 200, "{v}");
    assert_eq!(v["tokens_predicted"], 8, "{v}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_turn_never_writes_more_than_it_was_asked() {
    // windows where a rollover cannot get anywhere, and a cap on rollovers: the turn stops, it does not go round for ever
    for (n_ctx, max) in [(512, 64), (768, 64), (900, 64), (1100, 64), (1300, 64), (1600, 64), (4096, 1)] {
        let compaction = CompactionConfig { rollover: Rollover { on: true, max_per_turn: max, ..Default::default() }, ..Default::default() };
        let s = spawn_with(FakeConfig { n_ctx, ..paced() }, Tokenizer::Bytes, ServerConfig { port: 0, compaction, ..Default::default() });
        let a = agent(&s).await;
        let (code, v) = turn(&s, &a, ids(32), 20_000).await;
        assert_eq!(code, 200, "{n_ctx}: {v}");
        assert!(v["tokens_predicted"].as_u64().unwrap() <= 20_000, "{n_ctx}: {}", v["tokens_predicted"]);
        assert!(v["rollovers"].as_u64().unwrap_or(0) <= max as u64, "{n_ctx}: {}", v["rollovers"]);
        assert_eq!(s.metrics().await["active_requests"], 0, "{n_ctx}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn off_keeps_the_plain_limits() {
    let s = srv(false);
    let a = agent(&s).await;
    let (code, v) = turn(&s, &a, ids(CTX * 2), 8).await;
    assert_eq!(code, 400, "{v}");
    let (code, v) = turn(&s, &a, ids(32), (CTX * 2) as i64).await;
    assert_eq!(code, 200);
    assert!((v["tokens_predicted"].as_u64().unwrap() as usize) < CTX, "{v}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_window_too_small_to_roll_over_keeps_the_plain_limits() {
    let cfg = ServerConfig { port: 0, compaction: CompactionConfig { rollover: Rollover { on: true, ..Default::default() }, ..Default::default() }, ..Default::default() };
    let s = spawn_with(FakeConfig { n_ctx: 512, ..paced() }, Tokenizer::Bytes, cfg);
    let a = agent(&s).await;
    let (code, v) = turn(&s, &a, ids(32), 5000).await;
    assert_eq!(code, 200);
    assert!(v.get("rollovers").is_none(), "{v}");
    assert!((v["tokens_predicted"].as_u64().unwrap() as usize) < 512, "{v}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cancel_stops_a_turn_that_rolled_over() {
    let cfg = ServerConfig { port: 0, compaction: CompactionConfig { rollover: Rollover { on: true, ..Default::default() }, ..Default::default() }, ..Default::default() };
    let s = spawn_with(FakeConfig { n_ctx: CTX, ..paced() }, Tokenizer::Bytes, cfg);
    let a = agent(&s).await;
    let t = {
        let url = s.url(&format!("/v1/agents/{a}/turn"));
        tokio::spawn(async move {
            reqwest::Client::new().post(url).json(&json!({"prompt": ids(32), "n_predict": CTX * 40, "stream": false, "ignore_eos": true})).send().await.unwrap().json::<Value>().await.unwrap()
        })
    };
    let aid = a.clone();
    let t0 = std::time::Instant::now();
    loop {
        let i = s.get_json(&format!("/v1/agents/{aid}")).await;
        if i["compaction"]["rollovers"].as_u64().unwrap() >= 1 {
            break;
        }
        assert!(t0.elapsed().as_secs() < 20 && !t.is_finished(), "no rollover: {i}");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    s.post(&format!("/v1/agents/{a}/cancel"), &json!({})).await;
    let v = t.await.unwrap();
    assert_eq!(v["stop_type"], "cancelled", "{v}");
    let (code, v) = turn(&s, &a, ids(16), 8).await;
    assert_eq!(code, 200, "the agent works after the cancel: {v}");
}
