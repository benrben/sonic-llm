//! Compaction and rollover against turns, the planner and room, on the fake engine with its test hooks: a turn that arrives
//! while the agent's summary is being written, a rollover between a summary's start and its end, a compaction whose
//! re-prefill finds no room after the cut, a rollover whose notes wait for room, a cancel while the notes are written.

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::compaction::{CompactionConfig, PendingSummary, Rollover, Summarizer, SummarizerHandle};
use sllm_server::fake::{FakeConfig, FakeHooks};
use sllm_server::tokenizer::Tokenizer;
use sllm_server::ServerConfig;
use std::sync::atomic::Ordering::SeqCst;
use std::sync::Arc;
use std::time::{Duration, Instant};

const CTX: usize = 4096;

/// A summarizer that writes its summary on another thread, after `ms` (the shape of the helper summarizer).
struct Slow(u64);
struct SlowPending(Instant, u64);

impl PendingSummary for SlowPending {
    fn poll(&mut self) -> Option<Result<Vec<i32>, String>> {
        (self.0.elapsed() >= Duration::from_millis(self.1)).then(|| Ok(vec![7, 8, 9]))
    }
}

impl Summarizer for Slow {
    fn name(&self) -> &'static str {
        "slow-test"
    }
    fn summarize(&self, _: &str, _: &[i32], _: usize) -> Result<Vec<i32>, String> {
        Err("asynchronous".into())
    }
    fn start(&self, _: &str, _: &[i32], _: usize) -> Option<Box<dyn PendingSummary>> {
        Some(Box::new(SlowPending(Instant::now(), self.0)))
    }
}

fn server(fake: FakeConfig, summary_ms: Option<u64>, rollover: bool) -> Srv {
    let mut compaction = CompactionConfig { rollover: Rollover { on: rollover, ..Default::default() }, ..Default::default() };
    if let Some(ms) = summary_ms {
        compaction.summarizer = SummarizerHandle(Arc::new(Slow(ms)));
    }
    spawn_with(FakeConfig { n_ctx: CTX, ..fake }, Tokenizer::Bytes, ServerConfig { port: 0, compaction, ..Default::default() })
}

async fn create(srv: &Srv, lane: &str) -> String {
    let a: Value = srv.post("/v1/agents", &json!({"system_prompt_ids": (1..=10).collect::<Vec<i32>>(), "lane": lane})).await.json().await.unwrap();
    a["id"].as_str().unwrap().to_string()
}

fn ids(a: i32, n: i32) -> Vec<i32> {
    (a..a + n).collect()
}

fn task(srv: &Srv, path: String, body: Value) -> tokio::task::JoinHandle<(u16, Value)> {
    let (http, url) = (srv.http.clone(), srv.url(&path));
    tokio::spawn(async move {
        let r = http.post(url).json(&body).send().await.unwrap();
        (r.status().as_u16(), r.json().await.unwrap_or(Value::Null))
    })
}

fn turn_task(srv: &Srv, id: &str, prompt: Vec<i32>, n: usize) -> tokio::task::JoinHandle<(u16, Value)> {
    task(srv, format!("/v1/agents/{id}/turn"), json!({"prompt": prompt, "n_predict": n, "ignore_eos": true, "stream": false, "return_tokens": true}))
}

async fn within<T>(what: &str, h: tokio::task::JoinHandle<T>) -> T {
    tokio::time::timeout(Duration::from_secs(10), h).await.unwrap_or_else(|_| panic!("{what} never ended")).unwrap()
}

async fn info(srv: &Srv, id: &str) -> Value {
    srv.get_json(&format!("/v1/agents/{id}")).await
}

/// A plain completion admitted but held off its slot (the fake's hook): a request that holds room until released.
async fn holder(srv: &Srv, hooks: &FakeHooks) -> tokio::task::JoinHandle<(u16, Value)> {
    hooks.hold_completions.store(true, SeqCst);
    let h = task(srv, "/completion".into(), json!({"prompt": [70, 71], "n_predict": 1, "stream": false}));
    srv.wait_metrics("the holder admitted", |m| m["active_requests"].as_u64() >= Some(1)).await;
    h
}

/// Bug 3: a background agent's turn arriving while its summary is written was admitted into the background lane, which the
/// planner keeps still while a summary is written, and the summary was not polled while the agent was busy: both hung.
#[tokio::test(flavor = "multi_thread")]
async fn a_turn_during_its_agents_summary_waits_for_the_compaction_and_nothing_hangs() {
    let srv = server(fast(), Some(150), false);
    let a = create(&srv, "background").await;
    within("the first turn", turn_task(&srv, &a, ids(100, 600), 10)).await;
    let c = task(&srv, format!("/v1/agents/{a}/compact"), json!({"tail_tokens": 50}));
    srv.wait_metrics("the summary pending", |m| m["scheduler"]["summaries_pending"] == 1).await;
    let t = turn_task(&srv, &a, vec![5, 6], 4);
    let (code, rep) = within("the compaction", c).await;
    assert_eq!(code, 200, "{rep}");
    let (code, v) = within("the turn during the summary", t).await;
    assert_eq!(code, 200, "{v}");
    assert_eq!(v["timings"]["cache_n"], 10 + 3 + 50, "the turn ran on the compacted window: {v}");
}

/// Bug 5: a rollover between a summary's start and its end changed the window the summary's cut points were fixed for.
#[tokio::test(flavor = "multi_thread")]
async fn a_summary_is_applied_to_the_window_it_was_written_for() {
    let srv = server(FakeConfig { token_ms: 0.02, ..fast() }, Some(150), true);
    let a = create(&srv, "interactive").await;
    within("the first turn", turn_task(&srv, &a, ids(100, 2000), 10)).await;
    let before = info(&srv, &a).await["context_tokens"].clone();
    let c = task(&srv, format!("/v1/agents/{a}/compact"), json!({"tail_tokens": 50}));
    srv.wait_metrics("the summary pending", |m| m["scheduler"]["summaries_pending"] == 1).await;
    let t = turn_task(&srv, &a, ids(300, 32), CTX * 2);
    let (code, rep) = within("the compaction", c).await;
    assert_eq!(code, 200, "{rep}");
    assert_eq!(rep["context_before"], before, "the summary replaced the window it was written for: {rep}");
    let (code, v) = within("the long turn", t).await;
    assert_eq!(code, 200, "{v}");
    assert_eq!(v["tokens_predicted"], CTX * 2, "{}", v["tokens_predicted"]);
}

/// Bug 6: a compaction's re-prefill that finds no room after the cut waited for room, but the compaction took that as "not
/// admitted" and failed, leaving the agent with its prefix only.
#[tokio::test(flavor = "multi_thread")]
async fn a_compaction_whose_prefill_waits_for_room_completes() {
    let fake = FakeConfig { token_ms: 1.0, ..fast() };
    let hooks = fake.hooks.clone();
    let srv = server(fake, None, false);
    let a = create(&srv, "interactive").await;
    within("the first turn", turn_task(&srv, &a, ids(100, 600), 10)).await;
    let h = holder(&srv, &hooks).await;
    hooks.refuse_starts.store(usize::MAX, SeqCst);
    let c = task(&srv, format!("/v1/agents/{a}/compact"), json!({"summary_ids": [9, 9, 9], "tail_tokens": 50}));
    srv.wait_metrics("the prefill waiting for room", |m| m["scheduler"]["room_waiting"] == 1).await;
    hooks.refuse_starts.store(0, SeqCst);
    hooks.hold_completions.store(false, SeqCst);
    assert_eq!(within("the holder", h).await.0, 200);
    let (code, rep) = within("the compaction", c).await;
    assert_eq!(code, 200, "{rep}");
    assert_eq!(rep["context_after"], 10 + 3 + 50, "{rep}");
    assert_eq!(info(&srv, &a).await["context_tokens"], 10 + 3 + 50);
}

/// A long turn whose first leg runs to the end of the window while every start is refused: its notes request waits for
/// room behind the holder. Returns the turn's task.
async fn notes_waiting(srv: &Srv, hooks: &FakeHooks, a: &str) -> tokio::task::JoinHandle<(u16, Value)> {
    let t = turn_task(srv, a, ids(300, 32), CTX * 2);
    srv.wait_metrics("the long turn running", |m| m["active_requests"].as_u64() >= Some(2)).await;
    hooks.refuse_starts.store(usize::MAX, SeqCst);
    srv.wait_metrics("the notes waiting for room", |m| m["scheduler"]["room_waiting"] == 1 && m["scheduler"]["notes_writing"] == 1).await;
    t
}

/// Bug 7: while a rollover's notes waited for room the agent was not busy, so a new turn of it ran in the middle of the
/// rollover (and the rollover then failed).
#[tokio::test(flavor = "multi_thread")]
async fn a_turn_arriving_while_the_notes_wait_for_room_runs_after_the_rolled_turn() {
    let fake = FakeConfig { token_ms: 0.02, ..fast() };
    let hooks = fake.hooks.clone();
    let srv = server(fake, None, true);
    let a = create(&srv, "interactive").await;
    let h = holder(&srv, &hooks).await;
    let long = notes_waiting(&srv, &hooks, &a).await;
    hooks.refuse_starts.store(0, SeqCst);
    let next = turn_task(&srv, &a, vec![5, 6], 4);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!next.is_finished(), "the new turn ran in the middle of the rollover: {:?}", next.await.unwrap());
    hooks.hold_completions.store(false, SeqCst);
    assert_eq!(within("the holder", h).await.0, 200);
    let (code, v) = within("the rolled turn", long).await;
    assert_eq!(code, 200, "{v}");
    assert_eq!(v["tokens_predicted"], CTX * 2, "{}", v["tokens_predicted"]);
    let (code, v) = within("the next turn", next).await;
    assert_eq!(code, 200, "{v}");
}

/// Bug 7: cancelling a turn held behind its rollover left the notes being written (the agent was compacted for a turn that
/// was gone); and while the notes waited for room the agent's cancel found nothing to cancel.
#[tokio::test(flavor = "multi_thread")]
async fn cancelling_a_turn_held_behind_its_rollover_calls_the_rollover_off() {
    let fake = FakeConfig { token_ms: 0.02, ..fast() };
    let hooks = fake.hooks.clone();
    let srv = server(fake, None, true);
    let a = create(&srv, "interactive").await;
    let h = holder(&srv, &hooks).await;
    let long = notes_waiting(&srv, &hooks, &a).await;
    let r: Value = srv.post(&format!("/v1/agents/{a}/cancel"), &json!({})).await.json().await.unwrap();
    assert_eq!(r["cancelled"], true, "{r}");
    let (_, v) = within("the cancelled turn", long).await;
    assert_eq!(v["stop_type"], "cancelled", "{v}");
    srv.wait_metrics("the notes called off", |m| m["scheduler"]["notes_writing"] == 0 && m["scheduler"]["held_turns"] == 0).await;
    hooks.refuse_starts.store(0, SeqCst);
    hooks.hold_completions.store(false, SeqCst);
    assert_eq!(within("the holder", h).await.0, 200);
    srv.wait_metrics("all ended", |m| m["active_requests"] == 0 && m["scheduler"]["room_waiting"] == 0).await;
    assert_eq!(info(&srv, &a).await["compaction"]["rollovers"], 0, "no rollover for a turn that is gone");
    let (code, v) = within("a turn after the cancel", turn_task(&srv, &a, vec![5, 6], 4)).await;
    assert_eq!(code, 200, "{v}");
}
