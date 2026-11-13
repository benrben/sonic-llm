//! End-of-turn disk save (`SLLM_TURN_SAVE`, default on) on a real model (libllama, per-sequence KV streams): a two-turn
//! conversation of about 3,000 tokens runs, the server stops, a new one starts on the same prompt-cache directory, and turn 3
//! must load turn 2's finished state from the disk (most of its prompt comes from the cache) and give the tokens a cold run
//! gives. A third server on an empty directory is the cold run. Metal: `#[ignore]`d; run under the GPU lock:
//!
//!     engine/scripts/run_quiet.sh <owner> bash <script that runs>
//!         cargo test --release -p sllm-server --test turn_save_real -- --ignored --test-threads=1 --nocapture
//!
//! Model path from `SLLM_TURN_SAVE_MODEL` (default: the Qwen3-0.6B Q8_0 the other real tests use). Skips (passes) when it is
//! absent. `SLLM_TURN_SAVE=0` in the environment makes the restart assertions fail: that is the negative control.

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::engine::Engine;
use sllm_server::llama_engine::{EngineConfig, LlamaEngine, RowExact};
use sllm_server::sched::SchedConfig;
use sllm_server::tokenizer::Tokenizer;
use sllm_server::{start, ServerConfig};
use std::path::{Path, PathBuf};
use std::time::Duration;

const MODEL: &str = "/Users/benreich/Projects/crazyidea/research/inference_runtime/converted/qwen3-0.6b-Q8_0.gguf";

fn model_path() -> PathBuf {
    PathBuf::from(std::env::var("SLLM_TURN_SAVE_MODEL").unwrap_or_else(|_| MODEL.to_string()))
}

fn spawn_real(cache_dir: &Path) -> Option<Srv> {
    let model = model_path();
    if !model.is_file() {
        eprintln!("SKIP: {} not found", model.display());
        return None;
    }
    let _ = sllm_core::backend::registry();
    let mut cfg = EngineConfig { n_ctx: 8192, pool_pages: Some(64), ..EngineConfig::new(model) };
    cfg.kv_unified = false; // the prompt cache on disk is off with a unified pool
    cfg.row_exact = RowExact::On;
    cfg.prompt_cache = Some((cache_dir.to_path_buf(), 4 << 30));
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

/// `n` pseudo-random token ids (a fixed sequence per seed), away from the special ids.
fn ids(seed: u64, n: usize) -> Vec<i32> {
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1000 + (x % 20000) as i32
        })
        .collect()
}

/// A greedy turn: (the generated tokens, prompt tokens served from the cache).
async fn turn(srv: &Srv, prompt: &[i32], n: i64, cache_prompt: bool) -> (Vec<i32>, u64) {
    let v = srv.complete(prompt, n, json!({"cache_prompt": cache_prompt, "stream": false})).await;
    let toks = v["tokens"].as_array().unwrap().iter().map(|t| t.as_i64().unwrap() as i32).collect();
    (toks, v["timings"]["cache_n"].as_u64().unwrap_or(0))
}

fn find(v: &Value, key: &str) -> Option<Value> {
    match v {
        Value::Object(o) => o.get(key).cloned().or_else(|| o.values().find_map(|x| find(x, key))),
        Value::Array(a) => a.iter().find_map(|x| find(x, key)),
        _ => None,
    }
}

/// A counter of the engine's `prompt_cache` metrics.
fn pc(m: &Value, k: &str) -> u64 {
    find(m, "prompt_cache").and_then(|p| p[k].as_u64()).unwrap_or(0)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn turn_three_after_a_restart_loads_turn_twos_finished_state_and_matches_a_cold_run() {
    let root = std::env::temp_dir().join(format!("sllm-turn-save-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (warm_dir, cold_dir) = (root.join("warm"), root.join("cold"));

    // turn 1 (about 1,500 tokens), turn 2 (about 3,000): each prompt is the conversation so far plus new user tokens
    let mut conv = ids(1, 1500);
    {
        let Some(srv) = spawn_real(&warm_dir) else { return };
        let (r1, _) = turn(&srv, &conv, 48, true).await;
        conv.extend(&r1);
        conv.extend(ids(2, 1400));
        let (r2, _) = turn(&srv, &conv, 48, true).await;
        conv.extend(&r2);
        let m = srv.metrics().await;
        eprintln!("TURN_SAVE server 1: turn_saves {} skips {} entries {}", pc(&m, "turn_saves"), pc(&m, "turn_save_skips"), pc(&m, "entries"));
        assert!(pc(&m, "turn_saves") >= 1, "the idle engine saved no finished turn: {m}");
        srv.run.shutdown(); // joins the executor: the I/O thread has written everything
    }

    // turn 3 asks for the conversation so far plus a new user message
    conv.extend(ids(3, 120));
    let prompt_len = conv.len() as u64;

    let Some(srv) = spawn_real(&warm_dir) else { return };
    let (warm_tokens, cache_n) = turn(&srv, &conv, 32, true).await;
    let m = srv.metrics().await;
    eprintln!("TURN_SAVE after restart: prompt {prompt_len} cache_n {cache_n} hits {} hit_tokens {}", pc(&m, "hits"), pc(&m, "hit_tokens"));
    srv.run.shutdown();
    // a shared start is a whole number of 512-token blocks (row-exact mode), so at most one block of the saved turn is read again
    assert!(cache_n + 120 + 512 >= prompt_len, "after the restart only {cache_n} of {prompt_len} prompt tokens came from the cache");

    let Some(srv) = spawn_real(&cold_dir) else { return };
    let (cold_tokens, cold_cache) = turn(&srv, &conv, 32, false).await;
    srv.run.shutdown();
    assert_eq!(cold_cache, 0, "the cold run used a cache");
    assert_eq!(warm_tokens, cold_tokens, "the tokens after loading the saved turn differ from a cold run");
    let _ = std::fs::remove_dir_all(root);
}
