//! Compaction on the engine (libllama) behind the HTTP API (phase 3, 08-tests "Compaction correctness"). Metal: `#[ignore]`d;
//! run one at a time under the GPU lock:
//!
//!     python3 engine/scripts/with_gpu_lock.py --dev --mem-gb 8 --max-seconds 180 -- \
//!         cargo test --release -p sllm-server --test compaction_real -- --ignored --test-threads=1 --nocapture
//!
//! An agent with a system prompt, a long first turn and its generated tokens is compacted over the API to
//! `[system + summary + tail]`; its next turn's greedy tokens are those of a from-scratch agent over exactly that window
//! (a turn of `summary + tail` with `n_predict` 0, then the same next turn), on a fresh server and as a second agent of the
//! same server (other cells).

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::engine::Engine;
use sllm_server::llama_engine::{EngineConfig, LlamaEngine};
use sllm_server::sched::SchedConfig;
use sllm_server::tokenizer::Tokenizer;
use sllm_server::{start, ServerConfig};
use std::path::PathBuf;
use std::time::Duration;

const PROMPTS: &str = "/Users/benreich/Projects/crazyidea/sllm/results/prompts";
const QWEN3: &str = "/Users/benreich/Projects/crazyidea/research/inference_runtime/converted/qwen3-0.6b-Q4_0.gguf";
const GEMMA3: &str = "/Users/benreich/Projects/crazyidea/models/gguf/gemma-3-4b-it-qat-Q4_0.gguf";

fn spawn_real(model: &str) -> Srv {
    let _ = sllm_core::backend::registry();
    let mut cfg = EngineConfig { n_ctx: 8192, pool_pages: Some(96), ..EngineConfig::new(PathBuf::from(model)) };
    let sched = SchedConfig::default();
    cfg.max_rows = sched.max_rows();
    cfg.step_tokens = sched.step_tokens;
    let tok = Tokenizer::gguf(cfg.model.clone());
    let vocab = tok.vocab().unwrap();
    let factory: sllm_server::executor::EngineFactory = Box::new(move || LlamaEngine::new(cfg, vocab).map(|e| Box::new(e) as Box<dyn Engine>));
    let run = start(ServerConfig { port: 0, sched, ..Default::default() }, tok, factory).expect("start");
    run.wait_ready(Duration::from_secs(120)).expect("engine ready");
    Srv { run, http: reqwest::Client::builder().no_proxy().build().unwrap() }
}

fn ids(v: &Value) -> Vec<i32> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect()
}

fn load(pair: &str, file: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(format!("{PROMPTS}/{pair}/{file}")).unwrap()).unwrap()
}

async fn create(srv: &Srv, system: &[i32]) -> String {
    let r = srv.post("/v1/agents", &json!({"system_prompt_ids": system, "lane": "background"})).await;
    assert_eq!(r.status(), 201);
    r.json::<Value>().await.unwrap()["id"].as_str().unwrap().to_string()
}

async fn turn(srv: &Srv, id: &str, prompt: &[i32], n: i64) -> Vec<i32> {
    let r = srv.post(&format!("/v1/agents/{id}/turn"), &json!({"prompt": prompt, "n_predict": n, "stream": false, "ignore_eos": true, "return_tokens": true})).await;
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    ids(&r.json::<Value>().await.unwrap()["tokens"])
}

/// The from-scratch agent: `[system]`, then `window` with nothing generated, then `next`.
async fn from_scratch(srv: &Srv, system: &[i32], window: &[i32], next: &[i32]) -> Vec<i32> {
    let b = create(srv, system).await;
    turn(srv, &b, window, 0).await;
    let t = turn(srv, &b, next, 32).await;
    srv.http.delete(srv.url(&format!("/v1/agents/{b}"))).send().await.unwrap();
    t
}

async fn api_compaction_is_exact(model: &str, pair: &str) {
    let a8 = load(pair, "agents-8.json");
    let system = ids(&a8["prefix_ids"])[..600].to_vec();
    let summary = ids(&a8["agents"][0]["ids"])[2048..2248].to_vec();
    let doc = ids(&load(pair, "long-doc.json")["agents"][0]["ids"]);
    let next = ids(&a8["agents"][1]["ids"])[2048..2080].to_vec();

    let srv = spawn_real(model);
    let a = create(&srv, &system).await;
    let mut ctx = system.clone();
    ctx.extend(&doc[..2000]);
    ctx.extend(turn(&srv, &a, &doc[..2000], 24).await);
    ctx.extend(&doc[4000..4300]);
    ctx.extend(turn(&srv, &a, &doc[4000..4300], 24).await);
    let r = srv.post(&format!("/v1/agents/{a}/compact"), &json!({"summary_ids": summary, "tail_tokens": 700})).await;
    assert_eq!(r.status(), 200);
    let rep: Value = r.json().await.unwrap();
    eprintln!("{model}: {rep}");
    assert_eq!((rep["context_before"].as_u64().unwrap() as usize, rep["context_after"].as_u64().unwrap()), (ctx.len(), 600 + 200 + 700));
    let window: Vec<i32> = summary.iter().chain(&ctx[ctx.len() - 700..]).copied().collect();
    let got = turn(&srv, &a, &next, 32).await;
    // the same server: a second agent over the same window (its own cells)
    let same = from_scratch(&srv, &system, &window, &next).await;
    let result = json!({"test": "compaction_api_exactness", "model": model, "report": rep, "greedy_tokens": got.len(),
        "same_server_identical": got == same});
    drop(srv);
    // a fresh server
    let srv = spawn_real(model);
    let fresh = from_scratch(&srv, &system, &window, &next).await;
    assert_eq!(got, fresh, "{model}: the compacted agent's next turn differs from a from-scratch run on a fresh server");
    assert_eq!(got, same, "{model}: the compacted agent's next turn differs from a from-scratch agent of the same server");
    println!("{result}");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn qwen3_api_compaction_is_exact() {
    api_compaction_is_exact(QWEN3, "qwen3-0.6b-q4").await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn gemma3_api_compaction_is_exact() {
    api_compaction_is_exact(GEMMA3, "gemma-3-4b-it-qat-4bit").await;
}

/// A window policy on the engine: the turn that crosses it is followed by a compaction (stub summary) on the
/// helper lane; the next turn runs on `[system + stub summary + tail]` and equals a from-scratch agent over it.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn qwen3_policy_compaction_on_the_engine() {
    let pair = "qwen3-0.6b-q4";
    let a8 = load(pair, "agents-8.json");
    let system = ids(&a8["prefix_ids"])[..300].to_vec();
    let doc = ids(&load(pair, "long-doc.json")["agents"][0]["ids"]);
    let srv = spawn_real(QWEN3);
    let r = srv.post("/v1/agents", &json!({"system_prompt_ids": system, "compaction": {"window_tokens": 3000, "tail_tokens": 512, "summary_tokens": 128}})).await;
    assert_eq!(r.status(), 201);
    let a = r.json::<Value>().await.unwrap()["id"].as_str().unwrap().to_string();
    let mut ctx = system.clone();
    for k in 0..4 {
        let p = &doc[k * 800..(k + 1) * 800];
        ctx.extend(p);
        ctx.extend(turn(&srv, &a, p, 16).await);
    }
    // 300 + 4 x 816 = 3564 >= 3000: compacted after the 4th turn; the next turn is held behind it if it comes first
    let next = &doc[5000..5040];
    let got = turn(&srv, &a, next, 32).await;
    let info = srv.get_json(&format!("/v1/agents/{a}")).await;
    let last = &info["compaction"]["last"];
    eprintln!("policy compaction: {last}");
    assert_eq!((info["compaction"]["compactions"].clone(), last["trigger"].clone(), last["context_before"].as_u64().unwrap() as usize), (json!(1), json!("window"), ctx.len()));
    assert_eq!(last["context_after"], 300 + 128 + 512);
    let window: Vec<i32> = ctx[300..428].iter().chain(&ctx[ctx.len() - 512..]).copied().collect();
    let want = from_scratch(&srv, &system, &window, next).await;
    assert_eq!(got, want);
}

/// 08-tests "Helper reservation": a Gemma-3-4B brain whose background lane is full (4 running, 4 queued)
/// and a Qwen3-0.6B helper flooded with ordinary calls; an agent's compaction (the real helper summarizer, on the helper
/// lane's reserved slot) still completes within its 1 s budget, and the compaction waits on no slot: its queue time is
/// small, the background lane stays exactly as full as it was, every queued request finishes, and the compacted agent's
/// next turn runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "needs Metal and the GPU lock"]
async fn helper_reservation_compaction_with_the_background_lane_full() {
    use sllm_server::helper::HelperConfig;
    let _ = sllm_core::backend::registry();
    let pair = "gemma-3-4b-it-qat-4bit";
    let a8 = load(pair, "agents-8.json");
    let system = ids(&a8["prefix_ids"])[..300].to_vec();
    let doc = ids(&load(pair, "long-doc.json")["agents"][0]["ids"]);
    let tok = Tokenizer::gguf(PathBuf::from(GEMMA3));
    let vocab = tok.vocab().unwrap();
    let sched = SchedConfig { caps: [1, 4, 1], queue_limits: [4, 4, 4], ..SchedConfig::default() };
    let mut rc = EngineConfig { n_ctx: 4096, pool_pages: Some(96), ..EngineConfig::new(PathBuf::from(GEMMA3)) };
    rc.max_rows = sched.max_rows();
    rc.step_tokens = sched.step_tokens;
    let factory: sllm_server::executor::EngineFactory = Box::new(move || LlamaEngine::new(rc, vocab).map(|e| Box::new(e) as Box<dyn Engine>));
    let h = HelperConfig { slots: 3, pool_pages: 24, reserve_pages: 6, ..HelperConfig::new(PathBuf::from(QWEN3)) };
    let run = start(ServerConfig { port: 0, sched, helper: Some(h), ..Default::default() }, tok, factory).expect("start");
    run.wait_ready(Duration::from_secs(180)).expect("ready");
    let srv = std::sync::Arc::new(Srv { run, http: reqwest::Client::builder().no_proxy().build().unwrap() });
    assert_eq!(srv.get_json("/caps").await["compaction"]["summarizer"], "helper-summarize");

    // the agent that will be compacted: about 2,900 tokens of context
    let a = create(&srv, &system).await;
    let mut ctx = system.clone();
    for k in 0..2 {
        let p = &doc[k * 1200..k * 1200 + 1200];
        ctx.extend(p);
        ctx.extend(turn(&srv, &a, p, 16).await);
    }

    // the background lane full: 4 running, 4 queued, all long decodes
    let mut brain = Vec::new();
    for k in 0..8 {
        let s = srv.clone();
        brain.push(tokio::spawn(async move {
            s.post("/completion", &json!({"prompt": [2, 106 + k, 3, 4], "n_predict": 500, "ignore_eos": true, "lane": "background"})).await.status()
        }));
    }
    srv.wait_metrics("background lane full", |m| {
        let l = &m["scheduler"]["lanes"]["background"];
        l["running"].as_u64() == Some(4) && l["queued"].as_u64() == Some(4)
    })
    .await;
    // the helper flooded with ordinary calls (more than its ordinary slots)
    let text = "Tool call: run_tests() -> 1 failed\n".repeat(60);
    let mut flood = Vec::new();
    for _ in 0..8 {
        let (s, t) = (srv.clone(), text.clone());
        flood.push(tokio::spawn(async move { s.post("/v1/helper/summarize", &json!({"text": t, "max_tokens": 200, "ignore_eos": true})).await.status() }));
    }
    srv.wait_metrics("helper flooded", |m| m["helper"]["queued"].as_u64().unwrap_or(0) >= 3).await;

    let t0 = std::time::Instant::now();
    let r = srv.post(&format!("/v1/agents/{a}/compact"), &json!({"tail_tokens": 512, "summary_tokens": 128})).await;
    let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
    assert_eq!(r.status(), 200);
    let rep: Value = r.json().await.unwrap();
    let m = srv.metrics().await;
    let lane = m["scheduler"]["lanes"]["background"].clone();
    eprintln!("RESERVATION compaction wall {wall_ms:.0} ms, report {rep}, background lane {lane}, helper {}", m["helper"]);
    assert_eq!(rep["summary_source"], "helper-summarize", "{rep}");
    assert_eq!(rep["context_before"].as_u64().unwrap() as usize, ctx.len());
    assert!(rep["context_after"].as_u64().unwrap() < ctx.len() as u64, "{rep}");
    assert!(rep["queued_ms"].as_f64().unwrap() < 50.0, "the compaction waited for a slot: {rep}");
    // the bar under load (many agents at once): 2 s (Ben, G3 gate 2026-10-27); an unloaded compaction stays under 1 s
    assert!(wall_ms < 2000.0 && rep["total_ms"].as_f64().unwrap() < 2000.0, "compaction took {wall_ms:.0} ms: {rep}");
    // it was done while the lane was still full: no slot of it was taken or given
    assert_eq!(lane["running"], 4, "{lane}");

    for f in flood {
        assert_eq!(f.await.unwrap(), 200);
    }
    for b in brain {
        assert_eq!(b.await.unwrap(), 200, "a queued background request never got its slot");
    }
    // the compacted agent continues
    let t = turn(&srv, &a, &doc[3000..3032], 8).await;
    assert_eq!(t.len(), 8);
    println!("{}", json!({"test": "helper_reservation", "compaction_wall_ms": wall_ms, "report": rep}));
}
