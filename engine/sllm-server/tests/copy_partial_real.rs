//! Partial stream copies (`SLLM_COPY_PARTIAL`, default on) on a real model (libllama, per-sequence KV streams): a stream
//! copy moves the rows its source uses instead of the whole stream. Requests that attach a shared prefix from a kept
//! sequence (a copy, then a cut back to the shared start), take a kept sequence over and cut it in the middle, and leave a
//! kept sequence behind (a copy) must produce the same greedy tokens with the partial copy on and off. Metal: `#[ignore]`d;
//! run under the GPU lock:
//!
//!     engine/scripts/with_gpu_lock.py --dev --owner NAME --max-seconds 170 --mem-gb 8 -- \
//!         cargo test --release -p sllm-server --test copy_partial_real -- --ignored --test-threads=1 --nocapture
//!
//! Model path from `SLLM_COPY_MODEL` (default: the Qwen3-0.6B Q4_0 the other real tests use). Skips (passes) when it is
//! absent. Needs the row-exact lib (`SLLM_GGML_METAL=rowexact`): with the official one the partial copy is off and the
//! test says so.

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::engine::Engine;
use sllm_server::llama_engine::{EngineConfig, LlamaEngine, RowExact};
use sllm_server::sched::SchedConfig;
use sllm_server::tokenizer::Tokenizer;
use sllm_server::{start, ServerConfig};
use std::path::PathBuf;
use std::time::Duration;

const MODEL: &str = "/Users/benreich/Projects/crazyidea/research/inference_runtime/converted/qwen3-0.6b-Q4_0.gguf";

fn model_path() -> PathBuf {
    PathBuf::from(std::env::var("SLLM_COPY_MODEL").unwrap_or_else(|_| MODEL.to_string()))
}

fn spawn_real() -> Option<Srv> {
    let model = model_path();
    if !model.is_file() {
        eprintln!("SKIP: {} not found", model.display());
        return None;
    }
    let _ = sllm_core::backend::registry();
    let mut cfg = EngineConfig { n_ctx: 8192, pool_pages: Some(64), ..EngineConfig::new(model) };
    cfg.kv_unified = false; // per-sequence streams: the case stream copies are for
    cfg.row_exact = RowExact::On;
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

/// A greedy request that may share a start with a kept sequence: (the generated tokens, prompt tokens served from the cache).
async fn turn(srv: &Srv, prompt: &[i32], n: i64) -> (Vec<i32>, u64) {
    let v = srv.complete(prompt, n, json!({"cache_prompt": true, "stream": false})).await;
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

/// What one engine did: every request's tokens, the start each one found in the cache, and the engine's counters.
struct Run {
    tokens: Vec<Vec<i32>>,
    cache_n: Vec<u64>,
    runs: Value,
    attached: u64,
}

/// Sequential requests. A request that shares a start with a kept sequence uses it by a copy of the kept stream cut back to
/// the shared start (the start is under half of the kept sequence: an attach), or takes the kept sequence over and cuts it
/// in the middle (the start is over half of it). Starts are whole 512-token blocks (row-exact mode), and the long tails keep
/// every new kept sequence over twice the start, so the first group attaches (copies) and the second cuts.
async fn scenario() -> Option<Run> {
    let srv = spawn_real()?;
    let p1 = ids(1, 2600);
    let mut prompts: Vec<Vec<i32>> = vec![p1.clone()];
    for seed in 2..5 {
        prompts.push([&p1[..1024], &ids(seed, 1200)[..]].concat()); // 1024 shared of 2300+ kept: an attach
    }
    let (mut tokens, mut cache_n) = (Vec::new(), Vec::new());
    for p in &prompts {
        let (t, c) = turn(&srv, p, 24).await;
        tokens.push(t);
        cache_n.push(c);
    }
    // over half of a kept sequence is shared: it is taken over and cut in the middle (at 2048)
    let cut = [&prompts[1][..2048], &ids(5, 200)[..]].concat();
    // the second request's prompt, its answer and new tokens, then another tail on the same start: the sequence it left
    // behind is cut, then attached from
    let again = [&prompts[1][..], &tokens[1][..], &ids(6, 100)[..]].concat();
    let other = [&prompts[1][..1024], &ids(7, 1300)[..]].concat();
    for p in [&cut, &again, &other] {
        let (t, c) = turn(&srv, p, 24).await;
        tokens.push(t);
        cache_n.push(c);
    }
    let m = srv.metrics().await;
    let runs = find(&m, "stream_runs").unwrap_or(Value::Null);
    let attached = find(&m, "attached_agents").and_then(|v| v.as_u64()).unwrap_or(0);
    srv.run.shutdown();
    Some(Run { tokens, cache_n, runs, attached })
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn attach_cut_and_kept_sequence_copies_give_the_same_tokens_with_partial_copies_on_and_off() {
    std::env::remove_var("SLLM_COPY_PARTIAL");
    let Some(on) = scenario().await else { return };
    std::env::set_var("SLLM_COPY_PARTIAL", "0");
    let off = scenario().await.expect("the model was there a moment ago");
    std::env::remove_var("SLLM_COPY_PARTIAL");
    let n = |r: &Run, k: &str| r.runs[k].as_u64().unwrap_or(0);
    eprintln!("COPY on : cache_n {:?} attached_agents {} runs {}", on.cache_n, on.attached, on.runs);
    eprintln!("COPY off: cache_n {:?} attached_agents {} runs {}", off.cache_n, off.attached, off.runs);

    // the tokens first: this is what a short copy would change
    for (i, (a, b)) in on.tokens.iter().zip(&off.tokens).enumerate() {
        assert_eq!(a.len(), 24, "request {i}: length");
        assert_eq!(a, b, "request {i}: tokens differ between partial copies on and off");
    }
    // the same prefix decisions either way, and the scenario really attached and cut
    assert_eq!(on.cache_n, off.cache_n, "the shared starts differ with partial copies on and off");
    assert!(on.cache_n.iter().skip(1).all(|&c| c >= 1024), "a request that shares a start did not use it: {:?}", on.cache_n);
    assert!(on.attached >= 3 && off.attached >= 3, "requests did not attach a shared start: {} / {}", on.attached, off.attached);
    // off: every copy is whole, and ggml-metal cut none (its counters are the process's: nothing added since the on run)
    assert_eq!(n(&off, "copies_partial"), 0, "SLLM_COPY_PARTIAL=0 still made a partial copy: {}", off.runs);
    assert!(n(&off, "copies_whole") >= 3, "too few stream copies ran with the switch off: {}", off.runs);
    assert_eq!(n(&off, "copies_cut"), n(&on, "copies_cut"), "ggml-metal cut a copy with the switch off: {}", off.runs);
    // on: the copies carry rows, which is what this test is for (the official lib has no hook: then none, say so)
    if on.runs["copy_partial"] == json!(false) {
        eprintln!("COPY: partial copies are not available here (the linked ggml has no hook): both runs were whole copies");
        return;
    }
    assert!(n(&on, "copies_partial") >= 3, "partial copies never ran: {}", on.runs);
    assert_eq!(n(&on, "copies_whole"), 0, "a copy fell back to the whole stream: {}", on.runs);
    // and ggml-metal really cut them (the hint reached the copies that libllama ran inside the decode): 2 tensors per layer
    assert!(n(&on, "copies_cut") >= 2 * n(&on, "copies_partial") && n(&on, "copy_bytes_saved") > 0, "ggml-metal cut too few copies: {}", on.runs);
    // the rows are the source's: under half of the 8192 cells of a stream on average here
    assert!(n(&on, "copy_rows") < 8192 / 2 * n(&on, "copies_partial"), "the copies carry too many rows: {}", on.runs);
}
