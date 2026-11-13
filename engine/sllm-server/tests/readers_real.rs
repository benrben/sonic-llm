//! The f16 readers (`--kv-read f16`) on a real model: a chat reader of `CHAT_READER_CELLS` cells and a long reader whose
//! context is made for a long read and freed after it (its state saved to the SSD). Every repeat must give the cold tokens:
//! a warm read, a read restored from the SSD after another long read, and a prompt read in the chat reader or the long one.
//! Metal: `#[ignore]`d; run under the GPU lock:
//!
//!     engine/scripts/with_gpu_lock.py --dev --owner NAME -- cargo test --release -p sllm-server --test readers_real -- --ignored --nocapture
//!
//! Model path from `SLLM_READERS_MODEL` (default: the Qwen3-0.6B Q4_0 the other real tests use). Skips (passes) when it is absent.

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::engine::Engine;
use sllm_server::llama_engine::{EngineConfig, KvType, LlamaEngine, RowExact, CHAT_READER_CELLS};
use sllm_server::sched::SchedConfig;
use sllm_server::tokenizer::Tokenizer;
use sllm_server::{start, ServerConfig};
use std::path::PathBuf;
use std::time::Duration;

const MODEL: &str = "/Users/benreich/Projects/crazyidea/research/inference_runtime/converted/qwen3-0.6b-Q4_0.gguf";

fn spawn_real(dir: &std::path::Path) -> Option<Srv> {
    let model = PathBuf::from(std::env::var("SLLM_READERS_MODEL").unwrap_or_else(|_| MODEL.to_string()));
    if !model.is_file() {
        eprintln!("SKIP: {} not found", model.display());
        return None;
    }
    let _ = sllm_core::backend::registry();
    let mut cfg = EngineConfig { n_ctx: 8192, pool_pages: Some(96), ..EngineConfig::new(model) };
    (cfg.kv_type, cfg.kv_read_f16) = (KvType::Q4_0, true);
    cfg.row_exact = RowExact::On;
    cfg.prompt_cache = Some((dir.to_path_buf(), 4 << 30));
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

/// 16 greedy tokens after `n` token ids (a fixed pseudo-random prompt, `seed` picks which) in `lane`.
async fn gen(srv: &Srv, seed: u64, n: usize, lane: &str) -> Vec<i64> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let ids: Vec<u64> = (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1000 + x % 29_000
        })
        .collect();
    let b = json!({"prompt": ids, "n_predict": 16, "temperature": 0.0, "ignore_eos": true, "return_tokens": true, "stream": false, "cache_prompt": false, "lane": lane});
    let r = srv.post("/completion", &b).await;
    assert_eq!(r.status().as_u16(), 200);
    let v: Value = r.json().await.unwrap();
    v["tokens"].as_array().unwrap().iter().map(|t| t.as_i64().unwrap()).collect()
}

async fn kv_read(srv: &Srv) -> Value {
    srv.metrics().await["kv_read"].clone()
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn a_long_read_frees_its_reader_and_every_repeat_gives_the_cold_tokens() {
    let dir = std::env::temp_dir().join(format!("sllm-readers-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let Some(srv) = spawn_real(&dir) else { return };
    let open_cells = |m: &Value| m["open_cells"].as_u64().unwrap();
    let idle = kv_read(&srv).await;
    assert_eq!(open_cells(&idle), CHAT_READER_CELLS as u64, "only the chat reader holds an f16 cache while idle: {idle}");

    let a = gen(&srv, 1, 6000, "background").await;
    srv.wait_metrics("the long reader is freed", |m| m["kv_read"]["released"].as_u64().unwrap_or(0) >= 1).await;
    let m = kv_read(&srv).await;
    eprintln!("READERS after a long read: {m}");
    assert_eq!(open_cells(&m), CHAT_READER_CELLS as u64, "the long reader gave its memory back: {m}");
    assert!(m["created"].as_u64().unwrap() >= 2, "the long reader's context was made for the read: {m}");

    // a warm repeat (read again from the SSD copy), then another long read, then the first once more
    assert_eq!(gen(&srv, 1, 6000, "background").await, a, "warm repeat");
    let b = gen(&srv, 2, 6000, "background").await;
    assert_ne!(a, b);
    assert_eq!(gen(&srv, 1, 6000, "background").await, a, "repeat after another long read");
    let m = kv_read(&srv).await;
    eprintln!("READERS after the repeats: {m}");
    assert!(m["restored"].as_u64().unwrap() >= 1 || m["reused_tokens"].as_u64().unwrap() > 0, "a repeat reuses the saved read: {m}");

    // a prompt the chat reader holds, and the same prompt in the long reader: the same cells
    let chat = gen(&srv, 3, 3000, "interactive").await;
    assert_eq!(gen(&srv, 3, 3000, "background").await, chat, "chat reader and long reader read alike");
    assert_eq!(gen(&srv, 3, 3000, "interactive").await, chat, "warm chat");
    // a chat that has outgrown the chat reader is read in the long one
    let long_chat = gen(&srv, 4, 5000, "interactive").await;
    assert_eq!(gen(&srv, 4, 5000, "interactive").await, long_chat, "warm long chat");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn the_long_reader_stays_open_for_reads_that_wait_and_is_freed_after_the_last() {
    let dir = std::env::temp_dir().join(format!("sllm-readers-wait-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let Some(srv) = spawn_real(&dir) else { return };
    let cold = gen(&srv, 5, 6000, "background").await;
    srv.wait_metrics("the long reader is freed", |m| m["kv_read"]["released"].as_u64().unwrap_or(0) >= 1).await;
    let before = kv_read(&srv).await["created"].as_u64().unwrap();

    let (x, y, z) = tokio::join!(gen(&srv, 5, 6000, "background"), gen(&srv, 5, 6000, "background"), gen(&srv, 5, 6000, "background"));
    assert!(x == cold && y == cold && z == cold, "three reads of one prompt give the cold tokens");
    let m = kv_read(&srv).await;
    eprintln!("READERS after three reads at once: {m}");
    assert_eq!(m["created"].as_u64().unwrap() - before, 1, "one context for all three reads: {m}");
    assert_eq!(m["states_out_max"].as_u64(), Some(1), "one long state in memory at a time: {m}");
    srv.wait_metrics("the long reader is freed after the last read", |m| m["kv_read"]["open_cells"].as_u64() == Some(CHAT_READER_CELLS as u64)).await;
    let _ = std::fs::remove_dir_all(&dir);
}
