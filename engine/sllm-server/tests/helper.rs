//! The helper lane (phase 3): Qwen3-0.6B beside the brain in one sllm-server, `/v1/helper/*`.
//!
//! The GPU tests load the helper on Metal, so they are `#[ignore]`d; run them under the GPU lock, one at a time:
//!
//!     python3 engine/scripts/with_gpu_lock.py --dev --mem-gb 6 -- \
//!         cargo test --release -p sllm-server --test helper -- --ignored --test-threads=1
//!
//! Without `--ignored` only the CPU test runs (no helper configured: the routes say so).

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::engine::Engine;
use sllm_server::helper::HelperConfig;
use sllm_server::llama_engine::{EngineConfig, LlamaEngine};
use sllm_server::sched::SchedConfig;
use sllm_server::tokenizer::Tokenizer;
use sllm_server::{start, ServerConfig};
use std::time::{Duration, Instant};

fn helper_cfg() -> HelperConfig {
    HelperConfig { pool_pages: 24, ..HelperConfig::new(qwen_gguf().expect("Qwen model required")) }
}

/// A fake brain (no GPU for the brain) and the real helper.
fn spawn_helper(h: HelperConfig) -> Srv {
    let _ = sllm_core::backend::registry();
    let srv = spawn_with(fast(), Tokenizer::Bytes, ServerConfig { port: 0, helper: Some(h), ..Default::default() });
    srv.run.wait_ready(Duration::from_secs(120)).expect("helper ready");
    srv
}

async fn call(srv: &Srv, op: &str, body: Value) -> Value {
    let r = srv.post(&format!("/v1/helper/{op}"), &body).await;
    let status = r.status();
    let v: Value = r.json().await.unwrap();
    assert_eq!(status, 200, "{op}: {v}");
    v
}

#[tokio::test]
async fn without_a_helper_model_the_routes_say_so() {
    let srv = spawn(fast());
    let r = srv.post("/v1/helper/classify", &json!({"text": "x"})).await;
    assert_eq!(r.status(), 404);
    let v: Value = r.json().await.unwrap();
    assert!(v["error"]["message"].as_str().unwrap().contains("--helper-model"), "{v}");
    assert!(srv.metrics().await["helper"].is_null());
    let caps = srv.get_json("/caps").await;
    assert!(caps["helper"].is_null());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs Metal and the GPU lock"]
async fn classify_scores_labels_without_generating() {
    let srv = spawn_helper(helper_cfg());
    let on = "Tool call: run_tests([\"tests/test_dates.py\"]) -> 1 failed: test_parse_date\nThought: day and month are swapped on line 42.\n\
              Tool call: edit_file(\"utils/dates.py\", line=42) -> ok\nTool call: run_tests([\"tests/test_dates.py\"]) -> 12 passed";
    let off = "Tool call: run_tests([\"tests/test_dates.py\"]) -> 1 failed\n".repeat(6);
    let goal = "fix the failing unit test test_parse_date";
    let a = call(&srv, "classify", json!({"text": on, "goal": goal})).await;
    let b = call(&srv, "classify", json!({"text": off, "goal": goal})).await;
    for v in [&a, &b] {
        let p: f64 = v["probs"].as_object().unwrap().values().map(|x| x.as_f64().unwrap()).sum();
        assert!((p - 1.0).abs() < 1e-4, "{v}");
        assert_eq!(v["generated_tokens"], 0);
        assert_eq!(v["method"], "first-token");
    }
    assert!(a["probs"]["on-track"].as_f64().unwrap() > b["probs"]["on-track"].as_f64().unwrap(), "{a} {b}");
    // caller labels; labels that share a first token are scored as whole sequences
    let c = call(&srv, "classify", json!({"text": "The weather today is sunny and warm.", "labels": ["weather report", "weather warning"],
                                          "instruction": "Is this a weather report or a weather warning?"})).await;
    assert_eq!(c["method"], "sequence");
    assert_eq!(c["label"], "weather report", "{c}");
    // yes/no checks over the same read: one fork each, P(yes) in [0, 1]
    let d = call(&srv, "classify", json!({"text": off, "goal": goal, "checks": ["Does the agent repeat the same action?", "Is the text in French?"]})).await;
    let checks = d["checks"].as_array().unwrap();
    assert_eq!(checks.len(), 2);
    for c in checks {
        let p = c["p_yes"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&p), "{d}");
    }
    assert_eq!(d["label"], b["label"], "the label question is answered the same with or without checks: {d} {b}");
    let m = srv.metrics().await;
    assert_eq!(m["helper"]["completed"]["classify"], 4);
    assert_eq!(m["helper"]["pages"]["used"], 0, "every sequence is freed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs Metal and the GPU lock"]
async fn extract_returns_only_spans_of_the_source() {
    let srv = spawn_helper(helper_cfg());
    let text = "Hi, my order A-77812 has not arrived. My email is jane.doe@example.org and my second order is B-1203.";
    let v = call(&srv, "extract", json!({"text": text, "fields": [{"name": "order_id", "what": "order ID"}, {"name": "email", "what": "email address"}]})).await;
    assert_eq!(v["valid"], true);
    let spans = v["spans"].as_array().unwrap();
    for s in spans {
        let (a, b) = (s["start"].as_u64().unwrap() as usize, s["end"].as_u64().unwrap() as usize);
        assert_eq!(&text[a..b], s["text"].as_str().unwrap(), "offsets point at the span");
    }
    let texts: Vec<&str> = spans.iter().map(|s| s["text"].as_str().unwrap()).collect();
    assert!(texts.contains(&"A-77812"), "{v}");
    assert!(texts.contains(&"jane.doe@example.org"), "{v}");
    // nothing to extract, and text with no IDs at all: whatever comes back is still in the source
    for t in ["Thanks, everything works now.", "Ünïcödé 文字 and an ID-like thing: zz-00", "a"] {
        let v = call(&srv, "extract", json!({"text": t, "what": "order ID"})).await;
        for s in v["spans"].as_array().unwrap() {
            let (a, b) = (s["start"].as_u64().unwrap() as usize, s["end"].as_u64().unwrap() as usize);
            assert_eq!(&t[a..b], s["text"].as_str().unwrap());
        }
        assert_eq!(v["valid"], true);
    }
    assert_eq!(srv.metrics().await["helper"]["pages"]["used"], 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs Metal and the GPU lock"]
async fn compress_keeps_the_relevant_chunk_within_the_budget() {
    let srv = spawn_helper(helper_cfg());
    let mut lines: Vec<String> = (0..40).map(|i| format!("2026-10-27 10:{i:02}:00 INFO worker-{} heartbeat ok, queue depth {}", i % 5, i * 3)).collect();
    lines[23] = "2026-10-27 10:23:00 ERROR db-primary connection refused: too many clients (max_connections=100)".into();
    let text = lines.join("\n");
    // query-likelihood scoring: the 0.6B helper's yes/no scores saturate (every chunk p ~ 0.9999, G3 gate 2026-10-27)
    let body = json!({"text": text, "question": "Why did the database fail?", "budget_tokens": 64, "chunk_tokens": 48, "method": "qlik"});
    let v = call(&srv, "compress", body).await;
    assert!(v["kept_tokens"].as_u64().unwrap() <= 64, "{v}");
    assert!(v["source_tokens"].as_u64().unwrap() > 400);
    assert!(v["text"].as_str().unwrap().contains("too many clients"), "{v}");
    assert_eq!(v["generated_tokens"], 0);
    let chunks = v["chunks"].as_array().unwrap();
    // the chunks cover the source in order
    let mut at = 0;
    for c in chunks {
        assert_eq!(c["start"].as_u64().unwrap() as usize, at);
        at = c["end"].as_u64().unwrap() as usize;
    }
    assert_eq!(at, text.len());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs Metal and the GPU lock"]
async fn summarize_generates_a_short_summary() {
    let srv = spawn_helper(helper_cfg());
    let hist = "User: our nightly ETL job now takes 2 h 10 min instead of 25 min.\nAgent: I ran EXPLAIN ANALYZE on staging.events; it does a \
                sequential scan because there is no index on the day column.\nAgent: I created index events_day_idx; the job step went from 41 \
                min to 3 min.\nUser: great, what about the orders step?\nAgent: orders still takes 52 min; next I will check its join on customer_id.";
    let v = call(&srv, "summarize", json!({"text": hist, "max_tokens": 200})).await;
    let n = v["generated_tokens"].as_u64().unwrap();
    assert!(n > 10 && n <= 200, "{v}");
    let s = v["summary"].as_str().unwrap();
    assert!(s.contains("index") || s.contains("ETL"), "{s}");
    let w = call(&srv, "summarize", json!({"text": hist, "max_tokens": 64, "ignore_eos": true})).await;
    assert_eq!(w["generated_tokens"], 64);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs Metal and the GPU lock"]
async fn malformed_and_oversized_calls_are_refused() {
    let srv = spawn_helper(HelperConfig { n_ctx: 512, ..helper_cfg() });
    let r = srv.post_raw("/v1/helper/classify", "{nope").await;
    assert_eq!(r.status(), 400);
    let r = srv.post("/v1/helper/classify", &json!({"text": "x", "labels": ["one"]})).await;
    assert_eq!(r.status(), 422);
    let r = srv.post("/v1/helper/extract", &json!({"text": "x"})).await;
    assert_eq!(r.status(), 422);
    let r = srv.post("/v1/helper/summarize", &json!({"text": "word ".repeat(2000)})).await;
    assert_eq!(r.status(), 400);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["error"]["type"], "exceed_context_size_error", "{v}");
}

/// Rule 073: the brain's background lane is full and the helper is flooded with ordinary calls; a compaction call still
/// runs at once (it does not wait for a slot held by the calls or workers it would free) and finishes within its budget.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "needs Metal and the GPU lock"]
async fn compaction_runs_while_the_background_lane_and_the_helper_are_full() {
    let _ = sllm_core::backend::registry();
    let model = qwen_gguf().unwrap();
    let tok = Tokenizer::gguf(model.clone());
    let vocab = tok.vocab().unwrap();
    // the brain: 4 background slots and a queue of 4, all of them taken by long decodes
    let sched = SchedConfig { caps: [1, 4, 1], queue_limits: [4, 4, 4], ..SchedConfig::default() };
    let mut rc = EngineConfig { n_ctx: 1024, pool_pages: Some(40), ..EngineConfig::new(model) };
    rc.max_rows = sched.max_rows();
    rc.step_tokens = sched.step_tokens;
    let factory: sllm_server::executor::EngineFactory = Box::new(move || LlamaEngine::new(rc, vocab).map(|e| Box::new(e) as Box<dyn Engine>));
    let h = HelperConfig { slots: 3, pool_pages: 24, reserve_pages: 6, ..helper_cfg() };
    let run = start(ServerConfig { port: 0, sched, helper: Some(h), ..Default::default() }, tok, factory).expect("start");
    run.wait_ready(Duration::from_secs(120)).expect("ready");
    let srv = std::sync::Arc::new(Srv { run, http: reqwest::Client::builder().no_proxy().build().unwrap() });
    let mut brain = Vec::new();
    for k in 0..8 {
        let s = srv.clone();
        brain.push(tokio::spawn(async move {
            s.post("/completion", &json!({"prompt": [1 + k, 2, 3, 4], "n_predict": 900, "ignore_eos": true, "lane": "background"})).await.status()
        }));
    }
    srv.wait_metrics("background lane full", |m| {
        let l = &m["scheduler"]["lanes"]["background"];
        l["running"].as_u64() == Some(4) && l["queued"].as_u64() == Some(4)
    })
    .await;
    let r = srv.post("/completion", &json!({"prompt": [9, 9], "n_predict": 4, "lane": "background"})).await;
    assert_eq!(r.status(), 503, "the background lane is full");
    // flood the helper: more ordinary calls than it has slots
    let text = "Tool call: run_tests() -> 1 failed\n".repeat(60);
    let mut flood = Vec::new();
    for _ in 0..8 {
        let (s, t) = (srv.clone(), text.clone());
        flood.push(tokio::spawn(async move { s.post("/v1/helper/classify", &json!({"text": t})).await.status() }));
    }
    srv.wait_metrics("helper flooded", |m| m["helper"]["queued"].as_u64().unwrap_or(0) >= 3).await;
    let hist = "User: the ETL job is slow.\nAgent: added index events_day_idx, step went from 41 to 3 min.\n".repeat(8);
    let t0 = Instant::now();
    let v = call(&srv, "summarize", json!({"text": hist, "max_tokens": 64, "ignore_eos": true, "priority": "compaction"})).await;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    assert_eq!(v["generated_tokens"], 64);
    assert!(v["timing"]["queued_ms"].as_f64().unwrap() < 50.0, "compaction waited for a slot: {v}");
    assert!(ms < 2000.0, "compaction took {ms:.0} ms: {v}"); // the bar under load (Ben, G3 gate 2026-10-27)
    for f in flood {
        assert_eq!(f.await.unwrap(), 200);
    }
    for b in brain {
        let _ = b.await;
    }
}

/// The helper's summarize is compaction's summarizer: `POST /v1/agents/{id}/compact` with no summary has the helper
/// write it (asynchronously, on its compaction slot); the agent continues on `[system + summary + tail]`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs Metal and the GPU lock"]
async fn compaction_uses_the_helper_summary() {
    let _ = sllm_core::backend::registry();
    let model = qwen_gguf().unwrap();
    let tok = Tokenizer::gguf(model.clone());
    let vocab = tok.vocab().unwrap();
    let sched = SchedConfig::default();
    let mut rc = EngineConfig { n_ctx: 4096, pool_pages: Some(32), ..EngineConfig::new(model) };
    rc.max_rows = sched.max_rows();
    rc.step_tokens = sched.step_tokens;
    let factory: sllm_server::executor::EngineFactory = Box::new(move || LlamaEngine::new(rc, vocab).map(|e| Box::new(e) as Box<dyn Engine>));
    let run = start(ServerConfig { port: 0, sched, helper: Some(helper_cfg()), ..Default::default() }, tok, factory).expect("start");
    run.wait_ready(Duration::from_secs(120)).expect("ready");
    let srv = Srv { run, http: reqwest::Client::builder().no_proxy().build().unwrap() };
    assert_eq!(srv.get_json("/caps").await["compaction"]["summarizer"], "helper-summarize");
    let r = srv.post("/v1/agents", &json!({"system_prompt_ids": [9707, 11, 498, 525, 264, 16585], "lane": "background"})).await;
    assert_eq!(r.status(), 201);
    let id = r.json::<Value>().await.unwrap()["id"].as_str().unwrap().to_string();
    let history = "User: our nightly ETL job now takes 2 h 10 min instead of 25 min.\nAgent: EXPLAIN ANALYZE on staging.events shows a \
                   sequential scan: no index on the day column. I created index events_day_idx; that step went from 41 to 3 min.\n"
        .repeat(12);
    let r = srv.post(&format!("/v1/agents/{id}/turn"), &json!({"prompt": history, "n_predict": 16, "stream": false})).await;
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    let (st, rep) = {
        let r = srv.post(&format!("/v1/agents/{id}/compact"), &json!({"tail_tokens": 64, "summary_tokens": 96})).await;
        let s = r.status().as_u16();
        (s, r.json::<Value>().await.unwrap())
    };
    assert_eq!(st, 200, "{rep}");
    assert_eq!(rep["summary_source"], "helper-summarize", "{rep}");
    let n = rep["summary_tokens"].as_u64().unwrap();
    assert!(n > 8 && n <= 192, "{rep}");
    assert!(rep["context_after"].as_u64().unwrap() < rep["context_before"].as_u64().unwrap(), "{rep}");
    assert!(rep["summarize_ms"].as_f64().unwrap() > 0.0);
    // the compacted agent takes another turn
    let r = srv.post(&format!("/v1/agents/{id}/turn"), &json!({"prompt": "\nUser: and what is next?\nAgent:", "n_predict": 8, "stream": false})).await;
    assert_eq!(r.status(), 200);
    assert_eq!(srv.metrics().await["helper"]["completed"]["summarize"], 1);
}

/// Classify on the brain: a Gemma-3-4B brain with the Qwen3-0.6B helper and `ClassifyOn::Brain` (a lane over the brain's
/// own GGUF, no second copy of its weights). The fixed 40-item on-track/off-track set (`data/ontrack_v1.jsonl`) with the
/// variant and threshold frozen in `data/classify_thresholds_frozen.json` before this set was run (`progress_yn`: off-track
/// when P(no) >= 0.05); the 0.6B helper called 45% of it right, so the bar is the brain's 38 of 40 (95%).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs Metal and the GPU lock"]
async fn classify_on_the_brain_meets_the_ontrack_bar() {
    use sllm_server::helper::ClassifyOn;
    const GEMMA3: &str = "/Users/benreich/Projects/crazyidea/models/gguf/gemma-3-4b-it-qat-Q4_0.gguf";
    const DATA: &str = "/Users/benreich/Projects/crazyidea/sllm/results/phase3-helper/data/ontrack_v1.jsonl";
    let _ = sllm_core::backend::registry();
    let model = std::path::PathBuf::from(GEMMA3);
    let tok = Tokenizer::gguf(model.clone());
    let vocab = tok.vocab().unwrap();
    let sched = SchedConfig::default();
    let mut rc = EngineConfig { n_ctx: 4096, pool_pages: Some(48), ..EngineConfig::new(model.clone()) };
    rc.max_rows = sched.max_rows();
    rc.step_tokens = sched.step_tokens;
    let factory: sllm_server::executor::EngineFactory = Box::new(move || LlamaEngine::new(rc, vocab).map(|e| Box::new(e) as Box<dyn Engine>));
    let h = HelperConfig { classify_on: ClassifyOn::Brain, brain_model: Some(model), ..helper_cfg() };
    let run = start(ServerConfig { port: 0, sched, helper: Some(h), ..Default::default() }, tok, factory).expect("start");
    run.wait_ready(Duration::from_secs(180)).expect("ready");
    let srv = Srv { run, http: reqwest::Client::builder().no_proxy().build().unwrap() };
    assert_eq!(srv.get_json("/v1/helper").await["config"]["classify_on"], "brain");
    let items: Vec<Value> = std::fs::read_to_string(DATA).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(items.len(), 40);
    let (mut correct, mut ms, mut wrong) = (0, Vec::new(), Vec::new());
    for it in &items {
        let instruction = format!(
            "The text is the recent transcript of an AI agent. Its goal: {} Is the agent making real progress toward this goal?",
            it["goal"].as_str().unwrap()
        );
        let v = call(&srv, "classify", json!({"text": it["transcript"], "labels": ["yes", "no"], "instruction": instruction})).await;
        let p_no = v["probs"]["no"].as_f64().unwrap();
        let pred = if p_no >= 0.05 { "off-track" } else { "on-track" };
        if pred == it["label"].as_str().unwrap() {
            correct += 1;
        } else {
            wrong.push((it["id"].clone(), p_no));
        }
        ms.push(v["timing"]["run_ms"].as_f64().unwrap());
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("{}", json!({"test": "classify_on_the_brain", "correct": correct, "of": 40, "accuracy": correct as f64 / 40.0, "wrong": wrong,
        "run_ms_p50": ms[20], "run_ms_mean": ms.iter().sum::<f64>() / 40.0}));
    let m = srv.metrics().await["helper"].clone();
    assert_eq!(m["brain_lane"]["completed"]["classify"], 40, "the calls ran on the brain lane: {m}");
    assert_eq!(m["completed"]["classify"], 0, "none on the helper model: {m}");
    assert!(correct >= 38, "{correct} of 40 correct");
}
