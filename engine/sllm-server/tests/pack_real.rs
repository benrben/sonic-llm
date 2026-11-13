//! Sequence-id packing (`SLLM_SEQ_PACK`, default on) on a real model (libllama, per-sequence KV streams): eight agents of
//! different lengths finish at different times, so free ids open under the running ones and the engine moves running
//! sequences down into them (a KV stream copy). Every agent's tokens must equal the same prompt run alone. Metal:
//! `#[ignore]`d; run under the GPU lock:
//!
//!     engine/scripts/run_quiet.sh <owner> bash <script that runs>
//!         cargo test --release -p sllm-server --test pack_real -- --ignored --test-threads=1 --nocapture
//!
//! Model path from `SLLM_PACK_MODEL` (default: the Qwen3-0.6B Q8_0 the other real tests use). Skips (passes) when it is absent.
//! The holder test takes a sliding-window model, `SLLM_HOLDER_MODEL` (default: Gemma-4-E2B).

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::engine::Engine;
use sllm_server::llama_engine::{EngineConfig, LlamaEngine, RowExact};
use sllm_server::sched::SchedConfig;
use sllm_server::tokenizer::Tokenizer;
use sllm_server::{start, ServerConfig};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

const MODEL: &str = "/Users/benreich/Projects/crazyidea/research/inference_runtime/converted/qwen3-0.6b-Q8_0.gguf";

fn model_path() -> PathBuf {
    PathBuf::from(std::env::var("SLLM_PACK_MODEL").unwrap_or_else(|_| MODEL.to_string()))
}

fn spawn_real() -> Option<Srv> {
    let model = model_path();
    if !model.is_file() {
        eprintln!("SKIP: {} not found", model.display());
        return None;
    }
    let _ = sllm_core::backend::registry();
    let mut cfg = EngineConfig { n_ctx: 4096, pool_pages: Some(64), ..EngineConfig::new(model) };
    cfg.kv_unified = false; // per-sequence streams: the case packing is for
    cfg.row_exact = RowExact::On; // Q8_0 is not the exact kernel's fast type, so auto would leave it off
    let sched = SchedConfig::default();
    cfg.max_rows = sched.max_rows();
    cfg.step_tokens = sched.step_tokens;
    let tok = Tokenizer::gguf(cfg.model.clone());
    let vocab = tok.vocab().unwrap();
    let factory: sllm_server::executor::EngineFactory = Box::new(move || LlamaEngine::new(cfg, vocab).map(|e| Box::new(e) as Box<dyn Engine>));
    let run = start(ServerConfig { port: 0, sched, ..Default::default() }, tok, factory).expect("start");
    run.wait_ready(Duration::from_secs(120)).expect("engine ready");
    Some(Srv { run, http: reqwest::Client::builder().no_proxy().build().unwrap() })
}

async fn gen(srv: &Srv, prompt: &str, n: i64) -> Vec<i64> {
    let b = json!({"prompt": prompt, "n_predict": n, "temperature": 0.0, "ignore_eos": true, "return_tokens": true, "stream": false, "cache_prompt": false});
    let r = srv.post("/completion", &b).await;
    assert_eq!(r.status().as_u16(), 200);
    let v: Value = r.json().await.unwrap();
    v["tokens"].as_array().unwrap().iter().map(|t| t.as_i64().unwrap()).collect()
}

fn find(v: &Value, key: &str) -> Option<Value> {
    match v {
        Value::Object(o) => o.get(key).cloned().or_else(|| o.values().find_map(|x| find(x, key))),
        Value::Array(a) => a.iter().find_map(|x| find(x, key)),
        _ => None,
    }
}

const PROMPTS: [&str; 8] = [
    "Write a short story about a lighthouse keeper.\n",
    "List five facts about the planet Mars.\n",
    "Explain how a bicycle stays upright.\n",
    "Describe the taste of a ripe mango.\n",
    "Give a recipe for tomato soup.\n",
    "What are the rules of chess?\n",
    "Tell me about the history of the printing press.\n",
    "Why is the sky blue at noon and red at dusk?\n",
];
/// Tokens each agent generates: different, so they end at different steps and leave holes in the ids.
const LENS: [i64; 8] = [12, 90, 30, 140, 20, 110, 50, 70];

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn agents_that_finish_at_different_times_match_their_solo_runs_with_packing_on() {
    let Some(srv) = spawn_real() else { return };
    let srv = Arc::new(srv);
    // solo references first (one request at a time: the sequence never moves)
    let mut solo = Vec::new();
    for (p, n) in PROMPTS.iter().zip(LENS) {
        solo.push(gen(&srv, p, n).await);
    }
    let before = find(&srv.metrics().await, "stream_runs").unwrap_or(Value::Null);
    // all eight at once, twice over (ids are reused, so later waves see holes from the first)
    for wave in 0..2 {
        let mut set = tokio::task::JoinSet::new();
        for (i, (p, n)) in PROMPTS.iter().zip(LENS).enumerate() {
            let srv = srv.clone();
            set.spawn(async move { (i, gen(&srv, p, n).await) });
        }
        while let Some(r) = set.join_next().await {
            let (i, got) = r.unwrap();
            assert_eq!(got.len() as i64, LENS[i], "wave {wave} agent {i}: length");
            assert_eq!(got, solo[i], "wave {wave} agent {i}: tokens differ from the solo run");
        }
    }
    let after = find(&srv.metrics().await, "stream_runs").unwrap_or(Value::Null);
    let n = |v: &Value, k: &str| v[k].as_u64().unwrap_or(0);
    eprintln!("PACK stream_runs before {before} after {after}");
    assert_eq!(after["pack"], json!(true), "packing is on by default (SLLM_SEQ_PACK is not 0): {after}");
    // a run where no gap opened needs no move; a gap that stayed would show as a step of several runs
    let moved = n(&after, "moves") > n(&before, "moves");
    assert!(moved || n(&after, "max_runs") <= 1, "agents finished at different times, and a gap stayed under the running ones: {after}");
}

const HOLDER_MODEL: &str = "/Users/benreich/Projects/crazyidea/models/gguf/gemma-4-E2B-it-Q4_0.gguf";

/// Long prompts read once, then the same prompts on new agents, all at once, on a model that cannot copy a cache in part: each
/// agent's prompt makes a holder while the others run. Running sequences only batch on consecutive stream ids, so holders
/// between them would split every step into several graphs.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn holders_stay_out_of_the_way_of_the_running_agents() {
    let model = std::env::var("SLLM_HOLDER_MODEL").unwrap_or_else(|_| HOLDER_MODEL.to_string());
    if !PathBuf::from(&model).is_file() {
        eprintln!("SKIP: {model} not found");
        return;
    }
    std::env::set_var("SLLM_PACK_MODEL", &model);
    let Some(srv) = spawn_real() else { return };
    let srv = Arc::new(srv);
    let mut prompts = Vec::new();
    for i in 0..4 {
        let text = format!("Document {i}. You are one agent of a team that reads documents and keeps careful notes about them. ").repeat(100);
        prompts.push(srv.post("/tokenize", &json!({"content": text, "add_special": false})).await.json::<Value>().await.unwrap()["tokens"].clone());
    }
    for p in &prompts {
        let r = srv.post("/completion", &json!({"prompt": p, "n_predict": 1, "stream": false, "lane": "background", "cache_prompt": false})).await;
        assert_eq!(r.status().as_u16(), 200);
    }
    let before = find(&srv.metrics().await, "stream_runs").unwrap();
    let mut set = tokio::task::JoinSet::new();
    for p in prompts {
        let srv = srv.clone();
        let r = srv.post("/v1/agents", &json!({"lane": "background"})).await;
        assert_eq!(r.status().as_u16(), 201);
        let id = r.json::<Value>().await.unwrap()["id"].as_str().unwrap().to_string();
        set.spawn(async move {
            let b = json!({"prompt": p, "n_predict": 48, "stream": false, "ignore_eos": true});
            let r = srv.post(&format!("/v1/agents/{id}/turn"), &b).await;
            assert_eq!(r.status().as_u16(), 200);
        });
    }
    while let Some(r) = set.join_next().await {
        r.unwrap();
    }
    let m = srv.metrics().await;
    let (after, held) = (find(&m, "stream_runs").unwrap(), find(&m, "prefix_holders").unwrap());
    eprintln!("HOLDERS {held} stream_runs before {before} after {after}");
    let n = |v: &Value, k: &str| v[k].as_u64().unwrap_or(0);
    let (steps, split) = (n(&after, "steps") - n(&before, "steps"), n(&after, "split_steps") - n(&before, "split_steps"));
    assert!(n(&held, "created") >= 2, "the repeated prompts made too few holders: {held}");
    assert!(steps >= 48 && split <= 4, "holders split the running agents' steps: {split} of {steps}");
}
