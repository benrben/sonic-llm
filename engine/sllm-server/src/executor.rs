//! The executor thread: owns the [`Engine`] and is the only thread that calls it. The HTTP side (tokio) talks to it through
//! a bounded command channel and gets events back on one bounded channel per request.
//!
//! # The step loop (phase 2: continuous batching)
//!
//! Every iteration: take every command waiting (blocking for one only when nothing is admitted), then at the step
//! boundary
//!
//! 1. **sweep**: a request whose client is gone (its event channel closed) is cancelled; one past its `deadline_ms`
//!    is stopped with [`StopReason::Deadline`]. Both are removed before the step, so their slot and pages are free for
//!    it (a cancel command is handled the same way, between two steps);
//! 2. **fill**: queued requests take free lane slots in arrival order ([`Engine::try_begin`]);
//! 3. **plan**: [`crate::sched::Planner`] picks the step's work (decode rows, interactive first; prefill chunks up to
//!    the step's token budget);
//! 4. **step**: one [`Engine::step`] over those items, and its events fanned out to the requests' channels.
//!
//! # Admission (phase 2 task 7)
//!
//! A request goes to a lane: its own `lane` field, else its agent's lane, else background (a plain `/completion`).
//! Each lane has a slot cap and a queue limit. A request that finds the lane's slots and queue full is **deferred**:
//! a 503 with `retry_after_ms` (and a `Retry-After` header) and a `deferred` object the agent layer can act on. An
//! admitted request is also admitted by the engine, which reserves its worst-case pages at once, so queued prompts count
//! against the memory budget before they run. Queue depth, deferrals and expiries are in `/metrics` `scheduler`.
//!
//! A request whose channel is nearly full (client stalled) is cancelled; the executor never blocks on a client.
//!
//! # Prefix sharing (phase 2)
//!
//! When a request is given a slot the engine says how its prompt is served ([`Engine::prefix_state`]): fully, with a
//! shared prefix already in the cache (only the rest is prefilled), or **waiting on pages** that another running request
//! is prefilling right now (coalesced prefill). A waiting request keeps its slot but is not planned; at every step
//! boundary it is asked again, and once the pages are committed it attaches and prefills only its own part. It is
//! swept, cancelled and expired like any other request. Waiting requests are in `/metrics` `scheduler`.

use crate::compaction::{self, is_compaction, CompactionConfig};
use crate::engine::*;
use crate::lifecycle::{AgentLife, LifeStats, LifecycleConfig, Registry, Tier, WakeInfo};
use crate::sched::{Plan, Planner, Running, SchedConfig};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, oneshot};

// The agent lifecycle on the executor thread (phase 3, `crate::lifecycle`).
mod life;
/// The lifecycle's work at every step boundary, and the idle loop's polling for it.
mod life_tick;
/// The compaction side of the executor (phase 3): begin a compaction, collect its end, the policy triggers.
mod compact;
/// The end of the window does not end a turn.
mod rollover;
/// Admission of a request: its lane, its agent, deferral.
mod admit;
/// Requests waiting for room in the engine.
mod room;
/// The step boundary: sweep, fill, prefix sharing, plan, the step's events.
mod boundary;
/// The commands from the HTTP side.
mod commands;
/// The step loop.
mod main_loop;
/// The counters and the `/metrics` document.
mod publish;

use main_loop::LoopRec;

/// Capacity of one request's event channel. The executor keeps one slot free for the final event.
pub const EVENT_CAP: usize = 1024;

#[derive(Debug, Clone)]
pub struct Done {
    pub stop: StopReason,
    /// Tokens evaluated for this request (prompt, or the turn's new ids).
    pub prompt_n: usize,
    pub predicted_n: usize,
    /// Prompt tokens served from existing context: the agent's context before the turn, 0 for a plain completion.
    pub cached_n: usize,
    /// Executor clock, measured at completion (rule 008): from the start of the prefill to the first token, and from
    /// there to the end. Time spent queued for a lane slot is not in either (it is `queued_ms`).
    pub prompt_ms: f64,
    pub predicted_ms: f64,
    pub queued_ms: f64,
    /// The agent's context size after the turn.
    pub context_tokens: Option<usize>,
    /// Prompt tokens served from shared prefix pages (not prefilled by this request).
    pub shared_n: usize,
    /// Time spent waiting on another request's prefill of the shared pages (coalesced prefill).
    pub wait_ms: f64,
    /// An agent turn: how its KV got back before it ran (resident, restored from SSD, re-prefilled; phase 3).
    pub wake: Option<WakeInfo>,
    /// How many times the turn rolled over, and how long it stood still for that.
    pub rollovers: u32,
    pub rollover_ms: f64,
}

#[derive(Debug, Clone)]
pub enum ReqEvent {
    /// The request was admitted (queued or running).
    Started { prompt_n: usize, lane: Lane },
    Token { id: i32, text: String, probs: Option<Box<TokProbs>> },
    Done(Done),
    Error(EngineError),
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentInfo {
    pub id: String,
    pub object: &'static str,
    pub lane: Lane,
    pub context_tokens: usize,
    pub turns: u64,
    pub busy: bool,
    pub created: u64,
    /// Lifecycle state (phase 3): active, idle, suspended, restoring, history.
    pub state: &'static str,
    /// Where the KV is: ram, ssd, none.
    pub kv: &'static str,
    /// Turns waiting in the agent's input queue.
    pub queued_turns: usize,
    /// Tokens of the agent's history (its context, the pending token included).
    pub history_tokens: usize,
    /// The extracted-fields record (phase 3): kept outside the window.
    pub fields: Value,
    /// Compaction state: prefix, policy, count, the last report ([`compaction::AgentState::to_json`]).
    pub compaction: Value,
}

#[derive(Clone)]
pub struct StartCmd {
    pub id: ReqId,
    pub agent: Option<AgentId>,
    pub ids: Vec<i32>,
    pub params: SamplingParams,
    /// The request's lane; `None` = the agent's lane, or background for a plain completion.
    pub lane: Option<Lane>,
    /// Stop the request (queued or running) this long after it arrives.
    pub deadline_ms: Option<u64>,
    pub events: mpsc::Sender<ReqEvent>,
}

pub enum Command {
    Start(Box<StartCmd>),
    /// The client went away (fire and forget).
    Cancel(ReqId),
    /// `DELETE /v1/requests/{id}`: true if the request was queued or running.
    CancelReq { id: ReqId, reply: oneshot::Sender<bool> },
    AgentCreate { lane: Lane, system_ids: Vec<i32>, from: Option<String>, reply: oneshot::Sender<Result<AgentInfo, EngineError>> },
    AgentGet { id: AgentId, reply: oneshot::Sender<Result<AgentInfo, EngineError>> },
    AgentList { reply: oneshot::Sender<Vec<AgentInfo>> },
    AgentDelete { id: AgentId, reply: oneshot::Sender<Result<(AgentInfo, usize), EngineError>> },
    AgentCancel { id: AgentId, reply: oneshot::Sender<Result<bool, EngineError>> },
    /// `POST /v1/agents/{id}/suspend` (phase 3).
    AgentSuspend { id: AgentId, tier: Tier, reply: oneshot::Sender<Result<Value, EngineError>> },
    /// `POST /v1/agents/{id}/wake`: start the restore now.
    AgentWake { id: AgentId, reply: oneshot::Sender<Result<Value, EngineError>> },
    /// Compaction (phase 3): cut the agent back to its prefix, prefill summary and tail; replies when that is done.
    AgentCompact { id: AgentId, spec: compaction::Spec, reply: oneshot::Sender<Result<compaction::Report, EngineError>> },
    /// Read (`policy: None`) or set the agent's compaction policy (`Some(None)` clears it).
    AgentPolicy { id: AgentId, policy: Option<Option<compaction::Policy>>, reply: oneshot::Sender<Result<AgentInfo, EngineError>> },
    /// Read or change the agent's extracted-fields record; replies with the record.
    AgentFields { id: AgentId, op: compaction::FieldsOp, reply: oneshot::Sender<Result<Value, EngineError>> },
    Shutdown,
}

/// Counters and state shared with the HTTP side. Everything /metrics and /health need is here, so neither waits for the
/// executor while it is in a step.
#[derive(Default)]
pub struct Shared {
    pub ready: AtomicBool,
    pub init_error: Mutex<Option<String>>,
    pub info: OnceLock<EngineInfo>,
    pub caps: OnceLock<Value>,
    /// The scheduler settings, for /caps.
    pub sched_config: OnceLock<Value>,
    /// The compaction settings (phase 3), for /caps.
    pub compaction: OnceLock<Value>,
    pub requests_started: AtomicU64,
    pub requests_completed: AtomicU64,
    pub requests_cancelled: AtomicU64,
    pub requests_failed: AtomicU64,
    /// Requests stopped by their deadline.
    pub requests_expired: AtomicU64,
    /// Requests refused with a deferral (lane queue full).
    pub requests_deferred: AtomicU64,
    /// Requests cancelled because their client stopped reading (event channel full).
    pub requests_stalled: AtomicU64,
    pub active_requests: AtomicU64,
    pub agents_active: AtomicU64,
    pub agents_created: AtomicU64,
    pub loop_iterations: AtomicU64,
    pub last_step_us: AtomicU64,
    pub max_step_us: AtomicU64,
    pub engine: Mutex<EngineCounters>,
    /// The `/metrics` `scheduler` object (lanes, queues, deferrals), published at every step boundary.
    pub sched: Mutex<Value>,
    /// The `/metrics` `lifecycle` object (phase 3), published at every step boundary.
    pub lifecycle: Mutex<Value>,
}

struct Slot {
    events: mpsc::Sender<ReqEvent>,
    lane: Lane,
    arrived: Instant,
    began: Option<Instant>,
    first: Option<Instant>,
    deadline: Option<Instant>,
    order: u64,
    prompt_n: usize,
    cached_n: usize,
    predicted_n: usize,
    /// Feed tokens still to prefill (0 = decoding).
    prefill_left: usize,
    n_predict: i64,
    agent: Option<AgentId>,
    /// Waiting on pages another request is writing: holds its slot, is not planned.
    waiting: bool,
    wait_since: Option<Instant>,
    wait_ms: f64,
    shared_n: usize,
}

struct AgentRec {
    lane: Lane,
    turns: u64,
    busy: Option<ReqId>,
    created: u64,
    /// Lifecycle state and token history (phase 3).
    life: AgentLife,
    /// Turns waiting for the running turn or the restore (phase 3: a queue instead of 409 busy).
    inbox: VecDeque<life::Queued>,
    comp: compaction::AgentState,
}

/// A compaction whose summary-and-tail prefill is running (an executor-owned request on the helper lane).
struct InFlight {
    agent: AgentId,
    rx: mpsc::Receiver<ReqEvent>,
    reply: Option<oneshot::Sender<Result<compaction::Report, EngineError>>>,
    report: compaction::Report,
    t0: Instant,
}

#[derive(Default, Clone, Copy, Serialize)]
struct LaneCounters {
    admitted: u64,
    deferred: u64,
    completed: u64,
    cancelled: u64,
    expired: u64,
    max_queued: usize,
}

#[derive(Default)]
struct SchedCounters {
    lanes: [LaneCounters; 3],
    steps: u64,
    steps_with_interactive: u64,
    bg_prefill_deferred_steps: u64,
    burst_skip_steps: u64,
    step_ms_ema: f64,
    last: Plan,
    /// Requests that waited on pages, and prompt tokens served from shared pages.
    prefix_waited: u64,
    prefix_shared_tokens: u64,
}

struct Executor {
    engine: Box<dyn Engine>,
    slots: HashMap<ReqId, Slot>,
    queues: [VecDeque<ReqId>; 3],
    running: [Vec<ReqId>; 3],
    planner: Planner,
    stats: SchedCounters,
    agents: HashMap<AgentId, AgentRec>,
    next_agent: u64,
    next_order: u64,
    shared: Arc<Shared>,
    /// The loop profile, written to `SLLM_LOOP_TRACE` at shutdown (task 12).
    trace: Option<Vec<LoopRec>>,
    /// When the counters were last published, and whether a change is waiting to be (see [`Executor::publish_due`]).
    last_publish: std::cell::Cell<Instant>,
    publish_dirty: std::cell::Cell<bool>,
    /// The request-set shape at the last publish ([`Executor::state_sig`]): a change publishes at once.
    last_sig: std::cell::Cell<[usize; 5]>,
    /// The agent lifecycle (phase 3).
    life_cfg: LifecycleConfig,
    life_stats: LifeStats,
    registry: Option<Registry>,
    /// Suspended agents whose restore waits for a slot of the wake budget.
    wake_queue: VecDeque<AgentId>,
    /// Compaction (phase 3): settings, compactions in flight, a turn held per compacting agent, agents whose turn just
    /// ended (policy check), the next compaction request id.
    comp_cfg: CompactionConfig,
    inflight: Vec<InFlight>,
    held: HashMap<AgentId, StartCmd>,
    /// Requests the engine had no room for (cache pages, sequence slots, recurrent states) while others run: they wait here
    /// and are asked again when a request ends, instead of failing (2026-11-09: 238 agents on Qwen3.5 failed on states).
    room_wait: VecDeque<room::RoomWait>,
    /// Requests ended (or agents freed, suspended or dropped to history) so far, and the count at the last retry of `room_wait`.
    room_gen: u64,
    room_tried: u64,
    /// Compactions whose summary an asynchronous summarizer is writing (the helper lane).
    summarizing: Vec<compact::Summarizing>,
    /// Agents writing their notes for a rollover, and the turns that were clipped.
    writing: Vec<rollover::Writing>,
    rolled: HashMap<ReqId, rollover::Turn>,
    /// While a summary is being written, background-lane steps wait so the helper has the GPU to itself: a compacting
    /// agent is waiting on it, background work is not (`SLLM_COMPACT_YIELD=0` turns it off). The brain's and the helper's
    /// steps otherwise take turns on the GPU, so each summary token cost a brain step too.
    compact_yield: bool,
    due: Vec<AgentId>,
    next_compact: u64,
}

pub type EngineFactory = Box<dyn FnOnce() -> Result<Box<dyn Engine>, String> + Send>;

/// Thread body: build the engine ON this thread (all Metal calls belong to it), publish its info, then loop.
/// The longest the idle executor waits for a command while admitted requests cannot run.
const IDLE_WAIT: Duration = Duration::from_millis(5);
/// The most often the step loop publishes the counters behind /metrics.
const PUBLISH_EVERY: Duration = Duration::from_millis(100);

pub fn run(factory: EngineFactory, rx: mpsc::Receiver<Command>, shared: Arc<Shared>, sched: SchedConfig) {
    run_with(factory, rx, shared, sched, CompactionConfig::default(), LifecycleConfig::default())
}

/// [`run`] with the compaction and agent lifecycle settings (phase 3).
pub fn run_with(factory: EngineFactory, mut rx: mpsc::Receiver<Command>, shared: Arc<Shared>, sched: SchedConfig, comp: CompactionConfig, life_cfg: LifecycleConfig) {
    let engine = match factory() {
        Ok(e) => e,
        Err(msg) => {
            *shared.init_error.lock().unwrap() = Some(msg);
            return;
        }
    };
    let _ = shared.info.set(engine.info());
    let _ = shared.caps.set(engine.caps());
    let _ = shared.sched_config.set(serde_json::to_value(&sched).unwrap_or(Value::Null));
    let _ = shared.compaction.set(json!({
        "default_policy": comp.default_policy,
        "summarizer": comp.summarizer.0.name(),
        "rollover": comp.rollover,
        "lane": "helper",
        "window": "[system prompt] + [summary] + [recent tail]: the prefix pages stay, the replaced history's pages are freed, summary and tail are prefilled after the prefix",
    }));
    let mut ex = Executor::new(engine, shared.clone(), sched, comp, life_cfg);
    if let Some(dir) = ex.life_cfg.registry_dir.clone() {
        match Registry::open(dir) {
            Ok(r) => ex.registry = Some(r),
            Err(e) => eprintln!("sllm-server: agent registry disabled: {e}"),
        }
    }
    ex.life_load();
    ex.publish_counters();
    shared.ready.store(true, Relaxed);
    ex.run_loop(&mut rx);
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

impl Executor {
    fn new(engine: Box<dyn Engine>, shared: Arc<Shared>, sched: SchedConfig, comp: CompactionConfig, life_cfg: LifecycleConfig) -> Executor {
        Executor {
            engine,
            slots: HashMap::new(),
            queues: Default::default(),
            running: Default::default(),
            planner: Planner::new(sched),
            stats: SchedCounters::default(),
            agents: HashMap::new(),
            next_agent: 1,
            next_order: 1,
            shared,
            trace: std::env::var_os("SLLM_LOOP_TRACE").map(|_| Vec::with_capacity(1 << 16)),
            last_publish: std::cell::Cell::new(Instant::now()),
            publish_dirty: std::cell::Cell::new(false),
            last_sig: std::cell::Cell::new([0; 5]),
            registry: None,
            life_cfg,
            life_stats: LifeStats::default(),
            wake_queue: VecDeque::new(),
            comp_cfg: comp,
            inflight: Vec::new(),
            held: HashMap::new(),
            room_wait: VecDeque::new(),
            room_gen: 0,
            room_tried: 0,
            summarizing: Vec::new(),
            writing: Vec::new(),
            rolled: HashMap::new(),
            compact_yield: std::env::var("SLLM_COMPACT_YIELD").map_or(true, |v| v != "0"),
            due: Vec::new(),
            next_compact: 0,
        }
    }

    fn cfg(&self) -> &SchedConfig {
        &self.planner.cfg
    }

    fn shutdown(&mut self) {
        let ids: Vec<ReqId> = self.slots.keys().copied().collect();
        for id in ids {
            self.stop_request(id, StopReason::Cancelled);
        }
        self.shared.ready.store(false, Relaxed);
        if let (Some(tr), Some(path)) = (self.trace.take(), std::env::var_os("SLLM_LOOP_TRACE")) {
            use std::io::Write;
            let write = || -> std::io::Result<()> {
                let mut f = std::io::BufWriter::new(std::fs::File::create(&path)?);
                for r in &tr {
                    serde_json::to_writer(&mut f, r)?;
                    f.write_all(b"\n")?;
                }
                f.flush()
            };
            if let Err(e) = write() {
                eprintln!("sllm-server: SLLM_LOOP_TRACE {path:?}: {e}");
            }
        }
    }

    /// Remove `id` from its lane's queue or slots.
    fn unlink(&mut self, id: ReqId, lane: Lane) {
        let l = lane.index();
        if let Some(i) = self.queues[l].iter().position(|&x| x == id) {
            self.queues[l].remove(i);
        }
        self.running[l].retain(|&x| x != id);
    }

    /// Stop a request from outside the engine (client gone, cancel endpoint, delete, deadline): tell the engine, then
    /// report the stop. Returns false for an unknown id.
    fn stop_request(&mut self, id: ReqId, reason: StopReason) -> bool {
        if !self.slots.contains_key(&id) {
            return false;
        }
        self.engine.cancel(id);
        self.finish(id, reason);
        true
    }

    fn fail(&mut self, id: ReqId, e: EngineError) {
        self.room_gen += 1;
        if std::env::var_os("SLLM_LOG_FAIL").is_some() {
            eprintln!("sllm-server: failed request {id:?}: {} {} {}", e.status, e.kind, e.message);
        }
        self.engine.cancel(id);
        let _ = self.engine.take_timings(id);
        if let Some(slot) = self.slots.remove(&id) {
            self.unlink(id, slot.lane);
            self.release_agent(&slot, false);
            self.rolled.remove(&id);
            self.shared.requests_failed.fetch_add(1, Relaxed);
            let _ = slot.events.try_send(ReqEvent::Error(e));
        }
    }

    fn release_agent(&mut self, slot: &Slot, completed_turn: bool) {
        if let Some(aid) = &slot.agent {
            if let Some(a) = self.agents.get_mut(aid) {
                a.busy = None;
                if completed_turn {
                    a.turns += 1;
                }
            }
            // a turn stopped before its first token did not happen: the engine rolled the context back
            self.life_turn_ended(aid, !completed_turn && slot.predicted_n == 0);
        }
    }

    fn finish(&mut self, id: ReqId, reason: StopReason) {
        self.room_gen += 1;
        let Some(slot) = self.slots.remove(&id) else { return };
        self.unlink(id, slot.lane);
        let now = Instant::now();
        let began = slot.began.unwrap_or(now);
        let first = slot.first.unwrap_or(now);
        // The engine knows when each request's prefill really started; otherwise the executor's clock (slot granted to
        // first token, first token to now) stands.
        let (prompt_ms, predicted_ms) = self.engine.take_timings(id).unwrap_or((
            first.saturating_duration_since(began).as_secs_f64() * 1e3,
            now.saturating_duration_since(first).as_secs_f64() * 1e3,
        ));
        let mut done = Done {
            stop: reason,
            prompt_n: slot.prompt_n,
            predicted_n: slot.predicted_n,
            cached_n: slot.cached_n,
            prompt_ms,
            predicted_ms,
            queued_ms: began.saturating_duration_since(slot.arrived).as_secs_f64() * 1e3,
            context_tokens: slot.agent.as_deref().and_then(|a| self.engine.agent_context_tokens(a)),
            shared_n: slot.shared_n,
            wait_ms: slot.wait_ms + slot.wait_since.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e3),
            wake: slot.agent.as_deref().and_then(|a| self.agents.get(a)).and_then(|a| a.life.wake.clone()),
            rollovers: 0,
            rollover_ms: 0.0,
        };
        if self.rollover_wanted(id, reason) {
            return self.rollover_continue(id, slot, &done);
        }
        self.rollover_totals(id, &mut done);
        let stopped = matches!(reason, StopReason::Cancelled | StopReason::Deadline);
        self.release_agent(&slot, !stopped && !is_compaction(id));
        if let Some(aid) = slot.agent.as_ref().filter(|_| !stopped && !is_compaction(id)) {
            if self.agents.get(aid).is_some_and(|a| a.comp.policy.is_some()) {
                self.due.push(aid.clone()); // the policy is checked after the step (compaction_tick)
            }
        }
        let lc = &mut self.stats.lanes[slot.lane.index()];
        match reason {
            StopReason::Cancelled => {
                lc.cancelled += 1;
                self.shared.requests_cancelled.fetch_add(1, Relaxed);
            }
            StopReason::Deadline => {
                lc.expired += 1;
                self.shared.requests_expired.fetch_add(1, Relaxed);
            }
            _ => {
                lc.completed += 1;
                self.shared.requests_completed.fetch_add(1, Relaxed);
            }
        }
        let _ = slot.events.try_send(ReqEvent::Done(done));
    }
}
