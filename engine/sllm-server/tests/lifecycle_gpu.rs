//! The agent lifecycle on the engine (phase 3): exactness of suspend and restore, restart recovery, and the
//! measurements (wake latency, capacity). They load a model on Metal, so they are `#[ignore]`d; run them one at a time
//! under the GPU lock (dev runs are for iteration):
//!
//!     python3 engine/scripts/with_gpu_lock.py --dev --mem-gb 6 --max-seconds 180 -- \
//!         cargo test --release -p sllm-server --test lifecycle_gpu -- --ignored --test-threads=1 <name>
//!
//! Model: `SLLM_LIFE_MODEL=qwen|gemma` (default qwen); KV cache type `SLLM_LIFE_KV=f16|q8_0|q4_0` (default f16). The
//! measurements append JSON lines to `SLLM_LIFE_OUT` when it is set. Snapshots and registries go under
//! `sllm/var/agents/<test>-<pid>` (created here, removed at the end).

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::engine::Engine;
use sllm_server::lifecycle::LifecycleConfig;
use sllm_server::llama_engine::{EngineConfig, KvType, LlamaEngine};
use sllm_server::sched::SchedConfig;
use sllm_server::tokenizer::Tokenizer;
use sllm_server::{start, ServerConfig};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const PROMPTS: &str = "/Users/benreich/Projects/crazyidea/sllm/results/prompts";
const VAR: &str = "/Users/benreich/Projects/crazyidea/sllm/var/agents";

struct Model {
    name: &'static str,
    path: PathBuf,
    pair: &'static str,
}

fn model() -> Model {
    match std::env::var("SLLM_LIFE_MODEL").as_deref() {
        Ok("gemma") => Model {
            name: "gemma-3-4b-it-qat-Q4_0",
            path: PathBuf::from("/Users/benreich/Projects/crazyidea/models/gguf/gemma-3-4b-it-qat-Q4_0.gguf"),
            pair: "gemma-3-4b-it-qat-4bit",
        },
        _ => Model {
            name: "qwen3-0.6b-Q4_0",
            path: PathBuf::from("/Users/benreich/Projects/crazyidea/research/inference_runtime/converted/qwen3-0.6b-Q4_0.gguf"),
            pair: "qwen3-0.6b-q4",
        },
    }
}

fn kv_type() -> KvType {
    match std::env::var("SLLM_LIFE_KV").as_deref() {
        Ok("q8_0") => KvType::Q8_0,
        Ok("q4_0") => KvType::Q4_0,
        _ => KvType::F16,
    }
}

fn ids_of(v: &Value) -> Vec<i32> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect()
}

/// The 2,048-token shared prefix of the agents preset and the 8,192 tokens of the long document.
fn prompts(pair: &str) -> (Vec<i32>, Vec<i32>) {
    let read = |f: &str| -> Value { serde_json::from_str(&std::fs::read_to_string(format!("{PROMPTS}/{pair}/{f}")).unwrap()).unwrap() };
    let prefix = ids_of(&read("agents-8.json")["prefix_ids"]);
    let long = ids_of(&read("long-doc.json")["agents"][0]["ids"]);
    (prefix, long)
}

fn run_dir(tag: &str) -> PathBuf {
    PathBuf::from(VAR).join(format!("{tag}-{}", std::process::id()))
}

fn spawn_life(model: &Model, n_ctx: usize, pages: usize, life: LifecycleConfig, spill: PathBuf) -> Srv {
    let _ = sllm_core::backend::registry();
    let tok = Tokenizer::gguf(model.path.clone());
    let vocab = tok.vocab().unwrap();
    let sched = SchedConfig::default();
    let mut cfg = EngineConfig { n_ctx, pool_pages: Some(pages), spill_dir: Some(spill), kv_type: kv_type(), ..EngineConfig::new(model.path.clone()) };
    cfg.max_rows = sched.max_rows();
    cfg.step_tokens = sched.step_tokens;
    let factory: sllm_server::executor::EngineFactory = Box::new(move || LlamaEngine::new(cfg, vocab).map(|e| Box::new(e) as Box<dyn Engine>));
    let run = start(ServerConfig { port: 0, sched, lifecycle: life, ..Default::default() }, tok, factory).expect("start");
    run.wait_ready(Duration::from_secs(180)).expect("engine ready");
    Srv { run, http: reqwest::Client::builder().no_proxy().build().unwrap() }
}

fn no_auto() -> LifecycleConfig {
    LifecycleConfig::default()
}

async fn create(srv: &Srv, system: &[i32], lane: &str) -> String {
    let r = srv.post("/v1/agents", &json!({"system_prompt_ids": system, "lane": lane})).await;
    assert_eq!(r.status(), 201);
    let a: Value = r.json().await.unwrap();
    a["id"].as_str().unwrap().to_string()
}

/// A streamed turn: (ms from the request to its first token, the tokens, the final chunk).
async fn turn(srv: &Srv, id: &str, prompt: &[i32], n: i64) -> (f64, Vec<i64>, Value) {
    let body = json!({"prompt": prompt, "n_predict": n, "ignore_eos": true, "stream": true, "return_tokens": true});
    let t0 = Instant::now();
    let mut r = srv.post(&format!("/v1/agents/{id}/turn"), &body).await;
    assert_eq!(r.status(), 200, "turn on {id}");
    let mut raw = String::new();
    let mut ttft = None;
    while let Some(c) = r.chunk().await.unwrap() {
        if ttft.is_none() {
            ttft = Some(t0.elapsed().as_secs_f64() * 1e3);
        }
        raw.push_str(std::str::from_utf8(&c).unwrap());
    }
    let chunks = sse_json(&raw);
    let fin = chunks.last().cloned().unwrap();
    assert!(fin.get("error").is_none(), "{fin}");
    (ttft.unwrap_or(f64::NAN), tokens_of(&chunks), fin)
}

async fn info(srv: &Srv, id: &str) -> Value {
    srv.get_json(&format!("/v1/agents/{id}")).await
}

async fn suspend(srv: &Srv, id: &str, tier: &str) -> Value {
    let r = srv.post(&format!("/v1/agents/{id}/suspend"), &json!({"tier": tier})).await;
    assert_eq!(r.status(), 200);
    r.json().await.unwrap()
}

/// Suspend to the SSD and wait until the snapshot is written (the write count moves).
async fn suspend_durable(srv: &Srv, id: &str) -> (Value, Value) {
    let before = srv.metrics().await["lifecycle"]["writes"].as_u64().unwrap();
    let s = suspend(srv, id, "ssd").await;
    let m = srv.wait_metrics("snapshot written", |m| m["lifecycle"]["writes"].as_u64().unwrap() > before).await;
    (s, m["lifecycle"].clone())
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    if s.is_empty() {
        return f64::NAN;
    }
    s[s.len() / 2]
}

fn p95(v: &[f64]) -> f64 {
    sllm_server::lifecycle::percentiles(v).map_or(f64::NAN, |p| p.1)
}

fn emit(rec: &Value) {
    println!("RESULT {rec}");
    if let Ok(path) = std::env::var("SLLM_LIFE_OUT") {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
        writeln!(f, "{rec}").unwrap();
    }
}

fn git_commit() -> String {
    sllm_server::GIT_COMMIT.to_string()
}

/// Greedy tokens after suspend and restore cycles equal those of an agent that is never suspended, on the same
/// server and prompts: SSD restores (with another agent reusing the freed cells in between, and a back-to-back
/// suspend with nothing new to write), and the history tier (re-prefill from the token history).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn suspend_restore_cycles_keep_the_greedy_tokens() {
    let m = model();
    let (prefix, long) = prompts(m.pair);
    let dir = run_dir("exact");
    let srv = spawn_life(&m, 4096, 64, no_auto(), dir.join("kv"));
    let system = &prefix[..600];
    let turns: Vec<Vec<i32>> = (0..5).map(|k| long[k * 300..k * 300 + 120 + 40 * k].to_vec()).collect();
    let n = 32;
    // control: never suspended
    let ctl = create(&srv, system, "background").await;
    let mut want = Vec::new();
    for t in &turns {
        want.push(turn(&srv, &ctl, t, n).await.1);
    }
    assert_eq!(srv.http.delete(srv.url(&format!("/v1/agents/{ctl}"))).send().await.unwrap().status(), 200);
    let mut report = Vec::new();
    for mode in ["ssd", "history"] {
        let id = create(&srv, system, "background").await;
        let mut got = Vec::new();
        let mut writes = Vec::new();
        for (k, t) in turns.iter().enumerate() {
            if k > 0 {
                if mode == "ssd" {
                    let (s, l) = suspend_durable(&srv, &id).await;
                    writes.push((l["last_suspend_bytes_written"].as_u64().unwrap(), s["bytes_freed"].as_u64().unwrap()));
                    if k == 2 {
                        // another agent takes the freed cells and leaves before the wake
                        let o = create(&srv, &long[4000..5000], "background").await;
                        turn(&srv, &o, &long[5000..5100], 8).await;
                        assert_eq!(srv.http.delete(srv.url(&format!("/v1/agents/{o}"))).send().await.unwrap().status(), 200);
                    }
                    if k == 3 {
                        // wake and suspend again with nothing new (the generic engine writes the whole sequence every time)
                        let w: Value = srv.post(&format!("/v1/agents/{id}/wake"), &json!({})).await.json().await.unwrap();
                        assert!(w["state"] == "restoring" || w["state"] == "idle", "{w}");
                        let t0 = Instant::now();
                        while info(&srv, &id).await["state"] != "idle" {
                            assert!(t0.elapsed() < Duration::from_secs(20));
                            tokio::time::sleep(Duration::from_millis(2)).await;
                        }
                        let (_, _l) = suspend_durable(&srv, &id).await;
                    }
                } else {
                    suspend(&srv, &id, "history").await;
                    assert_eq!(info(&srv, &id).await["kv"], "none");
                }
            }
            let (_, toks, fin) = turn(&srv, &id, t, n).await;
            if k > 0 {
                assert_eq!(fin["wake"]["via"], if mode == "ssd" { "ssd" } else { "reprefill" }, "{fin}");
            }
            got.push(toks);
        }
        let same: Vec<bool> = got.iter().zip(&want).map(|(a, b)| a == b).collect();
        let positions: usize = got.iter().zip(&want).map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x == y).count()).sum();
        let total: usize = want.iter().map(|w| w.len()).sum();
        let rec = json!({"test": "exactness", "model": m.name, "kv": kv_type().name(), "mode": mode, "turns": turns.len(),
                         "suspend_cycles": turns.len() - 1, "turns_identical": same, "positions_equal": positions, "positions": total,
                         "bytes_written_vs_private": writes, "commit": git_commit()});
        emit(&rec);
        report.push(rec);
        assert_eq!(srv.http.delete(srv.url(&format!("/v1/agents/{id}"))).send().await.unwrap().status(), 200);
    }
    srv.run.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
    for r in &report {
        assert!(r["turns_identical"].as_array().unwrap().iter().all(|x| x == true), "greedy tokens changed: {r}");
    }
}

/// A restarted server loads its agents from the durable registry and their next turn re-prefills the history: the
/// tokens equal those of an agent that lived through.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn restart_recovers_agents_from_the_registry() {
    let m = model();
    let (prefix, long) = prompts(m.pair);
    let dir = run_dir("restart");
    let life = || LifecycleConfig { registry_dir: Some(dir.join("registry")), ..no_auto() };
    let system = &prefix[..500];
    let turns: Vec<Vec<i32>> = (0..3).map(|k| long[k * 200..k * 200 + 150].to_vec()).collect();
    let (want, id) = {
        let srv = spawn_life(&m, 4096, 48, life(), dir.join("kv"));
        let ctl = create(&srv, system, "background").await;
        let mut want = Vec::new();
        for t in &turns {
            want.push(turn(&srv, &ctl, t, 24).await.1);
        }
        assert_eq!(srv.http.delete(srv.url(&format!("/v1/agents/{ctl}"))).send().await.unwrap().status(), 200);
        let id = create(&srv, system, "background").await;
        for (k, t) in turns.iter().enumerate().take(2) {
            assert_eq!(turn(&srv, &id, t, 24).await.1, want[k]);
        }
        suspend_durable(&srv, &id).await;
        srv.run.shutdown();
        (want, id)
    };
    let srv = spawn_life(&m, 4096, 48, life(), dir.join("kv"));
    let i = info(&srv, &id).await;
    assert_eq!(i["state"], "history", "{i}");
    let (ttft, toks, fin) = turn(&srv, &id, &turns[2], 24).await;
    assert_eq!(fin["wake"]["via"], "reprefill");
    emit(&json!({"test": "restart", "model": m.name, "kv": kv_type().name(), "identical": toks == want[2],
                 "reprefill_tokens": fin["wake"]["reprefill_tokens"], "ttft_ms": ttft, "commit": git_commit()}));
    assert_eq!(toks, want[2], "the turn after the restart");
    srv.run.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

/// Wake latency (measurement): an agent with a 2k shared prefix (held by a sibling) and about 6k own tokens.
/// Time to first token of a 16-token turn when the agent is resident, restored from its SSD snapshot, prefetched with
/// a wake call, and re-prefilled from its history.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn measure_wake_latency() {
    let m = model();
    let (prefix, long) = prompts(m.pair);
    let reps: usize = std::env::var("SLLM_LIFE_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let own: usize = std::env::var("SLLM_LIFE_OWN").ok().and_then(|s| s.parse().ok()).unwrap_or(5800);
    let dir = run_dir("wake");
    let n_ctx = 8448;
    let srv = spawn_life(&m, n_ctx, 8 + 2 * (n_ctx / 256) + 4, no_auto(), dir.join("kv"));
    let sib = create(&srv, &prefix, "background").await;
    let id = create(&srv, &prefix, "interactive").await;
    let t_build = Instant::now();
    turn(&srv, &id, &long[..own], 1).await;
    let build_ms = t_build.elapsed().as_secs_f64() * 1e3;
    let mut k = 0usize;
    let mut next = || {
        k += 1;
        long[own + k * 16..own + k * 16 + 16].to_vec()
    };
    turn(&srv, &id, &next(), 4).await; // warm
    let ctx0 = info(&srv, &id).await["context_tokens"].as_u64().unwrap();
    let kv = srv.metrics().await["kv"]["sequences"].clone();
    let (mut ram, mut ssd, mut pre, mut rep_ms, mut restore_ms) = (vec![], vec![], vec![], vec![], vec![]);
    let mut writes = vec![];
    let mut suspend_stage = vec![];
    let (mut ram_prompt_ms, mut ssd_prompt_ms) = (vec![], vec![]);
    for _ in 0..reps {
        let (t, _, fin) = turn(&srv, &id, &next(), 4).await;
        ram.push(t);
        ram_prompt_ms.push(fin["timings"]["prompt_ms"].as_f64().unwrap_or(f64::NAN));
        for _ in 0..2 {
            let (s, l) = suspend_durable(&srv, &id).await;
            suspend_stage.push(s["stage_ms"].as_f64().unwrap());
            writes.push(json!({"bytes_written": l["last_suspend_bytes_written"], "bytes_freed": s["bytes_freed"], "write_ms": l["engine"]["stats"]["last_write_ms"]}));
            let (t, _, fin) = turn(&srv, &id, &next(), 4).await;
            assert_eq!(fin["wake"]["via"], "ssd");
            ssd.push(t);
            ssd_prompt_ms.push(fin["timings"]["prompt_ms"].as_f64().unwrap_or(f64::NAN));
            restore_ms.push(fin["wake"]["restore_ms"].as_f64().unwrap());
        }
        // prefetch: the wake call comes ahead of the turn (the tool result is on its way)
        suspend_durable(&srv, &id).await;
        let t0 = Instant::now();
        srv.post(&format!("/v1/agents/{id}/wake"), &json!({})).await;
        while info(&srv, &id).await["state"] != "idle" {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let _ = t0;
        pre.push(turn(&srv, &id, &next(), 4).await.0);
        // re-prefill from the history
        suspend(&srv, &id, "history").await;
        let (t, _, fin) = turn(&srv, &id, &next(), 4).await;
        assert_eq!(fin["wake"]["via"], "reprefill");
        rep_ms.push(t);
    }
    let mt = srv.metrics().await;
    let rec = json!({
        "test": "wake_latency", "model": m.name, "kv": kv_type().name(), "commit": git_commit(), "dev_run": true,
        "context_tokens": ctx0, "shared_prefix_tokens": prefix.len(), "own_tokens_built": own, "build_ms": build_ms,
        "reps": reps,
        "ttft_ms": {
            "resident": {"median": median(&ram), "p95": p95(&ram), "samples": ram},
            "ssd_restore": {"median": median(&ssd), "p95": p95(&ssd), "samples": ssd},
            "prefetched_wake": {"median": median(&pre), "p95": p95(&pre), "samples": pre},
            "reprefill": {"median": median(&rep_ms), "p95": p95(&rep_ms), "samples": rep_ms},
        },
        "restore_ms": {"median": median(&restore_ms), "p95": p95(&restore_ms)},
        "turn_prompt_ms": {"resident": median(&ram_prompt_ms), "after_ssd_restore": median(&ssd_prompt_ms)},
        "suspend_stage_ms": {"median": median(&suspend_stage), "max": suspend_stage.iter().cloned().fold(0.0, f64::max)},
        "speedup_ttft_reprefill_over_ssd": median(&rep_ms) / median(&ssd),
        "writes": writes,
        "kv_sequences": kv,
        "lifecycle": mt["lifecycle"],
    });
    emit(&rec);
    let _ = sib;
    srv.run.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

/// Capacity (measurement): register `SLLM_LIFE_AGENTS` agents at 8k context (2k shared prefix plus about 6k own
/// tokens each) on a pool sized for `SLLM_LIFE_ACTIVE` resident agents; memory pressure spills the coldest to the SSD as
/// new ones arrive. Then the active set wakes together. Reports RAM and SSD bytes per agent.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn measure_capacity() {
    let m = model();
    let (prefix, long) = prompts(m.pair);
    let n_agents: usize = std::env::var("SLLM_LIFE_AGENTS").ok().and_then(|s| s.parse().ok()).unwrap_or(24);
    let n_active: usize = std::env::var("SLLM_LIFE_ACTIVE").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let own: usize = std::env::var("SLLM_LIFE_OWN").ok().and_then(|s| s.parse().ok()).unwrap_or(5800);
    let dir = run_dir("capacity");
    let n_ctx = 8192 + 256;
    let own_pages = (prefix.len() + own + 64).div_ceil(256) - prefix.len() / 256 + 1;
    let pages = prefix.len() / 256 + n_active * own_pages + 4;
    let srv = spawn_life(&m, n_ctx, pages, LifecycleConfig { relieve_on_refusal: true, max_restoring: n_active, ..Default::default() }, dir.join("kv"));
    let sib = create(&srv, &prefix, "background").await;
    let t0 = Instant::now();
    let mut ids = Vec::new();
    for k in 0..n_agents {
        let id = create(&srv, &prefix, "background").await;
        // each agent reads its own 6k slice of the document (rotated so the agents differ)
        let mut text: Vec<i32> = long[(k * 97) % 2000..].to_vec();
        text.extend_from_slice(&long[..]);
        turn(&srv, &id, &text[..own], 1).await;
        ids.push(id);
    }
    let register_ms = t0.elapsed().as_secs_f64() * 1e3;
    let m1 = srv.wait_metrics("snapshots written", |m| m["lifecycle"]["engine"]["io_in_flight"] == 0).await;
    let states = m1["lifecycle"]["agents"].clone();
    // the active set: the n_active oldest agents (all on the SSD) take a turn at once
    let t1 = Instant::now();
    let mut hs = Vec::new();
    for id in ids.iter().take(n_active) {
        let (http, url) = (srv.http.clone(), srv.url(&format!("/v1/agents/{id}/turn")));
        let body = json!({"prompt": &long[7000..7016], "n_predict": 16, "ignore_eos": true, "stream": false});
        hs.push(tokio::spawn(async move {
            let t = Instant::now();
            let v: Value = http.post(url).json(&body).send().await.unwrap().json().await.unwrap();
            (t.elapsed().as_secs_f64() * 1e3, v)
        }));
    }
    let mut lat = Vec::new();
    let mut via = Vec::new();
    for h in hs {
        let (ms, v) = h.await.unwrap();
        lat.push(ms);
        via.push(v["wake"]["via"].clone());
    }
    let burst_ms = t1.elapsed().as_secs_f64() * 1e3;
    let m2 = srv.metrics().await;
    let rec = json!({
        "test": "capacity", "model": m.name, "kv": kv_type().name(), "commit": git_commit(), "dev_run": true,
        "registered_agents": n_agents + 1, "pool_pages": pages, "pool_sized_for_active": n_active, "n_ctx": n_ctx,
        "own_tokens": own, "shared_prefix_tokens": prefix.len(), "register_ms": register_ms,
        "states_after_register": states,
        "ram_kv_bytes_used": m1["prefix"]["kv_bytes_used"], "ssd_bytes": m1["lifecycle"]["ssd_bytes"],
        "ssd_bytes_per_suspended_agent": m1["lifecycle"]["ssd_bytes"].as_f64().unwrap() / states["suspended"].as_f64().unwrap().max(1.0),
        "active_wake_turn_ms": {"median": median(&lat), "max": lat.iter().cloned().fold(0.0, f64::max), "samples": lat, "via": via},
        "active_burst_ms": burst_ms,
        "lifecycle_after": m2["lifecycle"],
        "footprint_bytes": m2["footprint_bytes"],
    });
    emit(&rec);
    let _ = sib;
    srv.run.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

/// A streamed turn with the arrival time (ms since the request) of every token event.
async fn turn_timed(srv: &Srv, id: &str, prompt: &[i32], n: i64, on_token: impl Fn(usize)) -> Vec<f64> {
    let body = json!({"prompt": prompt, "n_predict": n, "ignore_eos": true, "stream": true});
    let t0 = Instant::now();
    let mut r = srv.post(&format!("/v1/agents/{id}/turn"), &body).await;
    assert_eq!(r.status(), 200);
    let mut times = Vec::new();
    let mut buf = String::new();
    while let Some(c) = r.chunk().await.unwrap() {
        let now = t0.elapsed().as_secs_f64() * 1e3;
        buf.push_str(std::str::from_utf8(&c).unwrap());
        while let Some(i) = buf.find("\n\n") {
            let ev: String = buf.drain(..i + 2).collect();
            if ev.contains("\"stop\":false") {
                times.push(now);
                on_token(times.len());
            }
        }
    }
    times
}

fn itl(times: &[f64]) -> Vec<f64> {
    times.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Wake burst against the interactive lane (measurement): an interactive agent streams while 8 background agents
/// (2k shared prefix, about 6k own tokens each) wake at once, from their SSD snapshots or by re-prefilling their
/// history. Reports the interactive agent's inter-token latency without a burst and during each kind of burst, and
/// when the background turns finished.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn measure_wake_burst_against_interactive() {
    let m = model();
    let (prefix, long) = prompts(m.pair);
    let n_bg: usize = std::env::var("SLLM_LIFE_AGENTS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let own: usize = std::env::var("SLLM_LIFE_OWN").ok().and_then(|s| s.parse().ok()).unwrap_or(5800);
    let dir = run_dir("burst");
    let n_ctx = 8448;
    let pages = 8 + (n_bg + 2) * (n_ctx / 256) + 8;
    let srv = std::sync::Arc::new(spawn_life(&m, n_ctx, pages, no_auto(), dir.join("kv")));
    let fg = create(&srv, &long[7000..7200], "interactive").await;
    let sib = create(&srv, &prefix, "background").await;
    let mut bg = Vec::new();
    for k in 0..n_bg {
        let id = create(&srv, &prefix, "background").await;
        let mut text: Vec<i32> = long[(k * 131) % 2000..].to_vec();
        text.extend_from_slice(&long[..]);
        turn(&srv, &id, &text[..own], 1).await;
        bg.push(id);
    }
    let n_fg = 96;
    let mut k = 0usize;
    let mut fg_prompt = || {
        k += 1;
        long[7200 + k * 8..7200 + k * 8 + 8].to_vec()
    };
    turn(&srv, &fg, &fg_prompt(), 8).await; // warm
    let base = turn_timed(&srv, &fg, &fg_prompt(), n_fg, |_| {}).await;
    let mut out = serde_json::Map::new();
    for mode in ["ssd", "reprefill"] {
        for id in &bg {
            if mode == "ssd" {
                suspend_durable(&srv, id).await;
            } else {
                suspend(&srv, id, "history").await;
            }
        }
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let tx = std::sync::Mutex::new(Some(tx));
        let bgs = bg.clone();
        let srv2 = srv.clone();
        let body = long[7500..7516].to_vec();
        let burst = tokio::spawn(async move {
            let _ = rx.await;
            let t0 = Instant::now();
            let mut hs = Vec::new();
            for id in bgs {
                let (http, url) = (srv2.http.clone(), srv2.url(&format!("/v1/agents/{id}/turn")));
                let b = json!({"prompt": body, "n_predict": 8, "ignore_eos": true, "stream": false});
                hs.push(tokio::spawn(async move {
                    let v: Value = http.post(url).json(&b).send().await.unwrap().json().await.unwrap();
                    (t0.elapsed().as_secs_f64() * 1e3, v["wake"]["via"].clone())
                }));
            }
            let mut done = Vec::new();
            for h in hs {
                done.push(h.await.unwrap());
            }
            done
        });
        let times = turn_timed(&srv, &fg, &fg_prompt(), n_fg, |n| {
            if n == 8 {
                if let Some(t) = tx.lock().unwrap().take() {
                    let _ = t.send(());
                }
            }
        })
        .await;
        let done = burst.await.unwrap();
        let window: Vec<f64> = itl(&times[8..]);
        let finish: Vec<f64> = done.iter().map(|d| d.0).collect();
        out.insert(mode.into(), json!({
            "fg_itl_ms": {"median": median(&window), "p95": p95(&window), "max": window.iter().cloned().fold(0.0, f64::max)},
            "bg_turn_done_ms": {"median": median(&finish), "max": finish.iter().cloned().fold(0.0, f64::max)},
            "bg_via": done.iter().map(|d| d.1.clone()).collect::<Vec<_>>(),
        }));
    }
    let b = itl(&base[1..]);
    let rec = json!({
        "test": "wake_burst", "model": m.name, "kv": kv_type().name(), "commit": git_commit(), "dev_run": true,
        "background_agents": n_bg, "own_tokens": own, "shared_prefix_tokens": prefix.len(), "fg_tokens": n_fg,
        "fg_itl_ms_no_burst": {"median": median(&b), "p95": p95(&b)},
        "burst": out,
        "lifecycle": srv.metrics().await["lifecycle"],
    });
    emit(&rec);
    let _ = sib;
    std::sync::Arc::try_unwrap(srv).ok().map(|s| s.run.shutdown());
    let _ = std::fs::remove_dir_all(&dir);
}

/// The KV is a cache: a snapshot whose file is gone falls back to re-prefill from the history, with the same greedy tokens,
/// never to an error. (The native engine also detected a damaged snapshot with a checksum; libllama's sequence state has
/// none, so a flipped byte restores silently and is not tested for here.)
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn lost_snapshot_falls_back_to_reprefill() {
    let m = model();
    let (prefix, long) = prompts(m.pair);
    let dir = run_dir("lost");
    let kv = dir.join("kv");
    let srv = spawn_life(&m, 4096, 48, no_auto(), kv.clone());
    let system = &prefix[..700];
    let turns: Vec<Vec<i32>> = (0..3).map(|k| long[k * 250..k * 250 + 140].to_vec()).collect();
    let ctl = create(&srv, system, "background").await;
    let mut want = Vec::new();
    for t in &turns {
        want.push(turn(&srv, &ctl, t, 24).await.1);
    }
    assert_eq!(srv.http.delete(srv.url(&format!("/v1/agents/{ctl}"))).send().await.unwrap().status(), 200);
    let id = create(&srv, system, "background").await;
    assert_eq!(turn(&srv, &id, &turns[0], 24).await.1, want[0]);
    let snaps = || -> Vec<PathBuf> {
        std::fs::read_dir(&kv).map(|r| r.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "seq")).collect()).unwrap_or_default()
    };
    // gone
    suspend_durable(&srv, &id).await;
    let files = snaps();
    assert!(!files.is_empty(), "the snapshot is a file under {}", kv.display());
    for p in files {
        std::fs::remove_file(p).unwrap();
    }
    let (_, toks, fin) = turn(&srv, &id, &turns[1], 24).await;
    assert_eq!(fin["wake"]["via"], "reprefill", "{fin}");
    assert_eq!(toks, want[1], "after a lost snapshot");
    let l = srv.metrics().await["lifecycle"].clone();
    assert_eq!(l["restore_failures"], 1, "{l}");
    emit(&json!({"test": "lost_snapshot", "model": m.name, "kv": kv_type().name(), "identical": true, "restore_failures": l["restore_failures"], "commit": git_commit()}));
    srv.run.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

/// Compaction and suspend: a compacted agent's snapshot and re-prefill hold the compacted window
/// (greedy tokens equal an agent compacted the same way that never left RAM); compaction waits for the KV in RAM.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn compacted_agent_suspends_and_restores_its_compacted_window() {
    let m = model();
    let (prefix, long) = prompts(m.pair);
    let dir = run_dir("compact");
    let srv = spawn_life(&m, 4096, 64, no_auto(), dir.join("kv"));
    let system = &prefix[..512];
    let mut outs = Vec::new();
    for tier in ["none", "ssd", "history"] {
        let id = create(&srv, system, "background").await;
        turn(&srv, &id, &long[..2000], 8).await;
        let r = srv.post(&format!("/v1/agents/{id}/compact"), &json!({"summary_ids": &long[3000..3040], "tail_tokens": 300})).await;
        assert_eq!(r.status(), 200);
        let i = info(&srv, &id).await;
        assert_eq!(i["history_tokens"], i["context_tokens"], "{i}");
        if tier == "ssd" {
            suspend_durable(&srv, &id).await;
        } else if tier == "history" {
            suspend(&srv, &id, "history").await;
        }
        if tier != "none" {
            let r = srv.post(&format!("/v1/agents/{id}/compact"), &json!({"summary_ids": [1], "tail_tokens": 10})).await;
            assert_eq!(r.status(), 409);
        }
        let (_, toks, fin) = turn(&srv, &id, &long[2100..2116], 24).await;
        assert_eq!(fin["wake"]["via"], match tier { "ssd" => "ssd", "history" => "reprefill", _ => "ram" }, "{fin}");
        outs.push(toks);
    }
    emit(&json!({"test": "compaction_and_suspend", "model": m.name, "kv": kv_type().name(), "ssd_identical": outs[1] == outs[0], "reprefill_identical": outs[2] == outs[0], "commit": git_commit()}));
    assert_eq!(outs[1], outs[0], "restore of the compacted window");
    assert_eq!(outs[2], outs[0], "re-prefill of the compacted window");
    srv.run.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}
