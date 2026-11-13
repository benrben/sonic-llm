//! The helper lane (phase 3): a small resident model (Qwen3-0.6B Q4_0 by default) beside the brain, in the same
//! process, for the context loop's four operations (plan section 5):
//!
//! * **classify**: read the text once and score the label tokens by their logits; no generation;
//! * **extract**: constrained span copying: the output can only be runs of the source's own tokens ([`spans`]), so a
//!   returned span always occurs in the source and an ID cannot be invented; spans come back with their offsets;
//! * **compress**: extractive: cut a tool result into chunks ([`chunk`]), score each chunk's relevance to a question
//!   classify-style (P(yes) against P(no)), keep the best within a token budget; no generation;
//! * **summarize**: generate a short summary of history (greedy, 200 to 500 tokens).
//!
//! # Shape
//!
//! * **Its own engine on its own thread.** The helper thread loads the helper's weights, its own Metal and CPU backends
//!   and its own small page pool (`--helper-kv-pages`), and is the only thread that touches them (D029 holds per model:
//!   one thread owns a model and encodes its Metal work). Its command buffers go to the device's queue beside the
//!   brain's, so a helper step runs while the brain's executor waits on its own step: helper calls fill the brain's
//!   bandwidth-bound slack instead of taking a slot or a step of the brain's lanes.
//! * **One batched step over every call.** The calls in flight are async jobs ([`run`]): straight-line code that awaits
//!   prefill and decode requests on its sequences. The thread's loop polls every job, gathers their requests (decode rows
//!   first, then prefill chunks up to `--helper-step-tokens`, compaction calls before the others) into ONE
//!   `BatchForward::step`, hands the results back and polls again. Forked sequences share the source's pages
//!   (copy-on-write), so the labels of a classify or the fields of an extract read the text once.
//! * **A reserved slot (rule 073).** Each call reserves its worst-case pages at admission. Ordinary calls may use all but
//!   one of the `--helper-slots` and must leave `--helper-reserve-pages` free; a call with `"priority": "compaction"`
//!   may use everything and goes ahead of the queue, so compaction never waits behind the calls (or the workers) it
//!   frees memory for.

pub mod api;
pub mod chunk;
pub mod exec;
mod run;
pub mod spans;
mod summarizer;

pub use summarizer::{HelperSummarizer, COMPACTION_SUMMARY_CAP};
pub(crate) use summarizer::brain_facts;

use llama_ref::Vocab;
use serde::Deserialize;
use serde_json::{json, Value};
use crate::llama_engine::KvType;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

use crate::engine::EngineError;

#[derive(Debug, Clone)]
pub struct HelperConfig {
    pub model: PathBuf,
    /// Most tokens one helper sequence (a call's prompt plus its output) may hold.
    pub n_ctx: usize,
    /// Pages of 256 cells of the helper's own KV cache.
    pub pool_pages: usize,
    pub kv_type: KvType,
    /// Calls running at once; one of them is held for compaction.
    pub slots: usize,
    /// Pages ordinary calls leave free for a compaction call.
    pub reserve_pages: usize,
    /// Prefill tokens per helper step.
    pub step_tokens: usize,
    /// Calls waiting for a slot before new ones are refused with 503.
    pub queue_cap: usize,
    /// CPU threads of the helper's context.
    pub cpu_threads: usize,
    /// Where classify and compress scoring run: on the helper model, or on the brain (a second lane over the brain's
    /// GGUF; see [`BrainLane`]).
    pub classify_on: ClassifyOn,
    /// The brain's GGUF (set by the server binary; needed for `ClassifyOn::Brain`).
    pub brain_model: Option<PathBuf>,
    /// Pages of 256 cells of the brain lane's own KV cache (`--helper-classify-kv-pages`).
    pub brain_pool_pages: usize,
    /// Prefill tokens per step of the brain lane (`--helper-classify-step`): small steps keep the brain's own decode fed.
    pub brain_step_tokens: usize,
    /// Calls the brain lane runs at once.
    pub brain_slots: usize,
}

/// Where the classify and compress-scoring operations run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifyOn {
    /// On the helper model (Qwen3-0.6B): small, but it failed the on-track classify bar (45% on the fixed set).
    Helper,
    /// On the brain model already in the process: a lane of its own (thread, page pool) over the brain's GGUF, whose
    /// mmapped weights are the same physical pages as the brain's (no second copy).
    Brain,
}

impl ClassifyOn {
    pub fn parse(s: &str) -> Result<Option<ClassifyOn>, String> {
        match s {
            "auto" => Ok(None),
            "brain" => Ok(Some(ClassifyOn::Brain)),
            "helper" => Ok(Some(ClassifyOn::Helper)),
            o => Err(format!("unknown --helper-classify {o:?} (brain, helper or auto)")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            ClassifyOn::Helper => "helper",
            ClassifyOn::Brain => "brain",
        }
    }
}

impl HelperConfig {
    pub fn new(model: PathBuf) -> HelperConfig {
        HelperConfig {
            model,
            n_ctx: 8192,
            pool_pages: 64,
            kv_type: KvType::F16,
            slots: 8,
            reserve_pages: 0,
            step_tokens: 2048,
            queue_cap: 256,
            cpu_threads: 1,
            classify_on: ClassifyOn::Helper,
            brain_model: None,
            brain_pool_pages: 32,
            brain_step_tokens: 512,
            brain_slots: 8,
        }
    }

    /// The configuration of the brain lane: this one with the brain's GGUF and the lane's own pool and step size.
    /// `None` unless classify runs on the brain.
    pub fn brain_lane(&self) -> Option<HelperConfig> {
        let model = self.brain_model.clone().filter(|_| self.classify_on == ClassifyOn::Brain)?;
        Some(HelperConfig {
            model,
            pool_pages: self.brain_pool_pages.max(1),
            step_tokens: self.brain_step_tokens.max(1),
            slots: self.brain_slots.max(1),
            reserve_pages: 0,
            ..self.clone()
        })
    }

    /// Pages held for compaction: by default one sequence of 4,096 tokens (16 pages), at most a quarter of the pool.
    pub fn reserve(&self) -> usize {
        if self.reserve_pages > 0 {
            self.reserve_pages.min(self.pool_pages)
        } else {
            16.min(self.pool_pages / 4)
        }
    }

    pub fn parse_kv_type(s: &str) -> Result<KvType, String> {
        KvType::parse(s).map_err(|e| format!("helper {e}"))
    }

    pub fn to_json(&self) -> Value {
        json!({
            "model": self.model.display().to_string(), "n_ctx": self.n_ctx, "pool_pages": self.pool_pages,
            "kv_type": self.kv_type.name(), "slots": self.slots, "reserve_pages": self.reserve(),
            "step_tokens": self.step_tokens, "queue_cap": self.queue_cap,
            "classify_on": self.classify_on.name(),
            "brain_lane": self.brain_lane().map(|b| json!({"model": b.model.display().to_string(), "pool_pages": b.pool_pages,
                "step_tokens": b.step_tokens, "slots": b.slots})),
        })
    }
}

/// How urgent a call is. `compaction` calls may take the reserved slot and pages and go first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    #[default]
    Normal,
    Compaction,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassifyReq {
    pub text: String,
    /// Candidate labels (default `["on-track", "off-track"]`).
    #[serde(default)]
    pub labels: Option<Vec<String>>,
    /// The question asked about the text (default: the on-track monitor's question).
    #[serde(default)]
    pub instruction: Option<String>,
    /// The agent's goal, for the default on-track question.
    #[serde(default)]
    pub goal: Option<String>,
    /// Yes/no questions asked about the same text (one read, one fork each): each comes back with P(yes).
    #[serde(default)]
    pub checks: Option<Vec<String>>,
    /// Score whole label sequences (teacher-forced, one forked sequence per label) even when the first tokens differ.
    #[serde(default)]
    pub full: bool,
    #[serde(default)]
    pub priority: Priority,
    #[serde(default)]
    pub deadline_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Field {
    pub name: String,
    /// What to copy, in words (default: the name).
    #[serde(default)]
    pub what: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractReq {
    pub text: String,
    /// One field, in words ("order IDs"); or `fields`.
    #[serde(default)]
    pub what: Option<String>,
    #[serde(default)]
    pub fields: Option<Vec<Field>>,
    /// Spans per field (default 16).
    #[serde(default)]
    pub max_spans: Option<usize>,
    /// Output tokens per field (default 96).
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub priority: Priority,
    #[serde(default)]
    pub deadline_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompressReq {
    pub text: String,
    pub question: String,
    /// Tokens of the source to keep (default 512).
    #[serde(default)]
    pub budget_tokens: Option<usize>,
    /// Chunk size in tokens (default 128).
    #[serde(default)]
    pub chunk_tokens: Option<usize>,
    /// How a chunk is scored: `yesno` (log-odds of "yes" against "no" after the question "does the passage help answer
    /// it?") or `qlik` (query likelihood: the mean log-probability of the question's tokens given the chunk).
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub priority: Priority,
    #[serde(default)]
    pub deadline_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SummarizeReq {
    pub text: String,
    /// Most output tokens (default 400, at most 1,024).
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// The length asked for in the instruction, in words (default 150).
    #[serde(default)]
    pub words: Option<usize>,
    /// Replaces the default instruction.
    #[serde(default)]
    pub instruction: Option<String>,
    /// Generate exactly `max_tokens` tokens (end-of-generation masked): for cost measurements.
    #[serde(default)]
    pub ignore_eos: bool,
    #[serde(default)]
    pub priority: Priority,
    #[serde(default)]
    pub deadline_ms: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum Op {
    Classify(ClassifyReq),
    Extract(ExtractReq),
    Compress(CompressReq),
    Summarize(SummarizeReq),
}

impl Op {
    pub fn name(&self) -> &'static str {
        match self {
            Op::Classify(_) => "classify",
            Op::Extract(_) => "extract",
            Op::Compress(_) => "compress",
            Op::Summarize(_) => "summarize",
        }
    }
    pub fn index(&self) -> usize {
        match self {
            Op::Classify(_) => 0,
            Op::Extract(_) => 1,
            Op::Compress(_) => 2,
            Op::Summarize(_) => 3,
        }
    }
    /// The scheduling fields every request carries: its priority and deadline.
    fn scheduling(&self) -> (Priority, Option<u64>) {
        match self {
            Op::Classify(r) => (r.priority, r.deadline_ms),
            Op::Extract(r) => (r.priority, r.deadline_ms),
            Op::Compress(r) => (r.priority, r.deadline_ms),
            Op::Summarize(r) => (r.priority, r.deadline_ms),
        }
    }
    fn priority(&self) -> Priority {
        self.scheduling().0
    }
    fn deadline_ms(&self) -> Option<u64> {
        self.scheduling().1
    }
}

impl From<ClassifyReq> for Op {
    fn from(r: ClassifyReq) -> Op {
        Op::Classify(r)
    }
}

impl From<ExtractReq> for Op {
    fn from(r: ExtractReq) -> Op {
        Op::Extract(r)
    }
}

impl From<CompressReq> for Op {
    fn from(r: CompressReq) -> Op {
        Op::Compress(r)
    }
}

impl From<SummarizeReq> for Op {
    fn from(r: SummarizeReq) -> Op {
        Op::Summarize(r)
    }
}

/// A seeded generator of numbers below `n` (a linear congruential generator), for the randomized tests of this module's files.
#[cfg(test)]
pub(crate) fn seeded(mut seed: u64) -> impl FnMut(usize) -> usize {
    move |n| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as usize) % n
    }
}

/// The most tokens one helper sequence may hold (`--helper-ctx`).
pub const MAX_CTX: usize = 32_768;

pub const OPS: [&str; 4] = ["classify", "extract", "compress", "summarize"];

pub struct Call {
    pub op: Op,
    pub arrived: Instant,
    pub reply: oneshot::Sender<Result<Value, EngineError>>,
}

pub(crate) enum Msg {
    Call(Box<Call>),
    Shutdown,
}

/// Counters the HTTP side reads without waiting for the helper thread.
#[derive(Default)]
pub struct HelperShared {
    pub ready: AtomicBool,
    pub init_error: Mutex<Option<String>>,
    /// Model and pool facts, set once the helper is built.
    pub info: OnceLock<Value>,
    /// The `/metrics` `helper` object, published at every step boundary.
    pub metrics: Mutex<Value>,
}

/// One lane's thread and the channel into it.
struct Lane {
    tx: mpsc::Sender<Msg>,
    shared: Arc<HelperShared>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Lane {
    fn start(name: &str, cfg: HelperConfig) -> std::io::Result<Lane> {
        let (tx, rx) = mpsc::channel(cfg.queue_cap.max(1) + 8);
        let shared = Arc::new(HelperShared::default());
        let thread = {
            let shared = shared.clone();
            std::thread::Builder::new().name(name.into()).spawn(move || run::run(cfg, rx, shared))?
        };
        Ok(Lane { tx, shared, thread: Mutex::new(Some(thread)) })
    }

    fn shutdown(&self) {
        stop_lane_thread(&self.tx, &self.thread);
    }
}

/// Tell a lane's thread to stop, and wait for it.
fn stop_lane_thread(tx: &mpsc::Sender<Msg>, thread: &Mutex<Option<std::thread::JoinHandle<()>>>) {
    let _ = tx.try_send(Msg::Shutdown);
    if let Some(h) = thread.lock().unwrap().take() {
        let _ = h.join();
    }
}

fn wait_lane(shared: &HelperShared, timeout: Duration, t0: Instant) -> Result<(), String> {
    loop {
        if shared.ready.load(Relaxed) {
            return Ok(());
        }
        if let Some(e) = shared.init_error.lock().unwrap().clone() {
            return Err(format!("helper: {e}"));
        }
        if t0.elapsed() > timeout {
            return Err(format!("helper not ready after {timeout:?}"));
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The running helper lane: its thread and the channel into it. With `ClassifyOn::Brain` there is a second thread, the
/// **brain lane**: the same machinery over the brain's own GGUF, which takes the classify and compress calls. Its weights
/// are the file's read-only shared mapping, the same physical pages the brain's executor uses (no second copy); it adds
/// its own small page pool and scratch, and its own Metal command queue beside the brain's.
pub struct Helper {
    tx: mpsc::Sender<Msg>,
    pub shared: Arc<HelperShared>,
    /// The brain lane's shared state (`info`, `metrics`, readiness), when classify runs on the brain.
    pub brain_shared: Option<Arc<HelperShared>>,
    pub cfg: HelperConfig,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    brain: Option<Lane>,
}

impl Helper {
    /// Start the helper thread; it loads the model (check [`Helper::wait_ready`]). Touch `sllm_core::backend::registry()`
    /// first in a process that also loads a vocabulary.
    pub fn start(cfg: HelperConfig) -> std::io::Result<Helper> {
        if cfg.classify_on == ClassifyOn::Brain && cfg.brain_model.is_none() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "classify on the brain needs the brain's GGUF (brain_model)"));
        }
        let (tx, rx) = mpsc::channel(cfg.queue_cap.max(1) + 8);
        let shared = Arc::new(HelperShared::default());
        let thread = {
            let (cfg, shared) = (cfg.clone(), shared.clone());
            std::thread::Builder::new().name("sllm-helper".into()).spawn(move || run::run(cfg, rx, shared))?
        };
        let brain = match cfg.brain_lane() {
            Some(bc) => Some(Lane::start("sllm-helper-brain", bc)?),
            None => None,
        };
        Ok(Helper { tx, shared, brain_shared: brain.as_ref().map(|l| l.shared.clone()), cfg, thread: Mutex::new(Some(thread)), brain })
    }

    pub fn wait_ready(&self, timeout: Duration) -> Result<(), String> {
        let t0 = Instant::now();
        wait_lane(&self.shared, timeout, t0)?;
        match &self.brain_shared {
            Some(b) => wait_lane(b, timeout, t0),
            None => Ok(()),
        }
    }

    /// Whether the helper (and its brain lane) is ready, or the first start error.
    pub fn state(&self) -> Result<bool, String> {
        for sh in std::iter::once(&self.shared).chain(self.brain_shared.iter()) {
            if let Some(e) = sh.init_error.lock().unwrap().clone() {
                return Err(e);
            }
        }
        Ok(std::iter::once(&self.shared).chain(self.brain_shared.iter()).all(|s| s.ready.load(Relaxed)))
    }

    /// The lane an operation runs on: classify and compress on the brain lane when there is one.
    fn lane_of(&self, op: &Op) -> (&mpsc::Sender<Msg>, &Arc<HelperShared>) {
        match (&self.brain, op) {
            (Some(b), Op::Classify(_) | Op::Compress(_)) => (&b.tx, &b.shared),
            _ => (&self.tx, &self.shared),
        }
    }

    /// Submit a call and wait for its result (async: from the HTTP handlers).
    pub async fn call(&self, op: Op) -> Result<Value, EngineError> {
        let (tx, shared) = self.lane_of(&op);
        if !shared.ready.load(Relaxed) {
            return Err(match shared.init_error.lock().unwrap().clone() {
                Some(e) => EngineError::new(500, "server_error", format!("the helper failed to start: {e}")),
                None => EngineError::new(503, "unavailable_error", "Loading helper model").with_retry_after(500),
            });
        }
        let (reply, rx) = oneshot::channel();
        let call = Box::new(Call { op, arrived: Instant::now(), reply });
        tx.try_send(Msg::Call(call)).map_err(|_| {
            EngineError::new(503, "unavailable_error", "the helper queue is full").with_retry_after(100)
        })?;
        rx.await.unwrap_or_else(|_| Err(EngineError::new(500, "server_error", "the helper dropped the call")))
    }

    /// Blocking variant for tests and tools outside a runtime.
    pub fn call_blocking(&self, op: Op) -> Result<Value, EngineError> {
        let (tx, _) = self.lane_of(&op);
        let (reply, rx) = oneshot::channel();
        let call = Box::new(Call { op, arrived: Instant::now(), reply });
        tx.blocking_send(Msg::Call(call)).map_err(|_| EngineError::new(500, "server_error", "the helper is gone"))?;
        rx.blocking_recv().unwrap_or_else(|_| Err(EngineError::new(500, "server_error", "the helper dropped the call")))
    }

    /// The helper's metrics; with a brain lane, its metrics are under `brain_lane`.
    pub fn metrics(&self) -> Value {
        let mut v = self.shared.metrics.lock().unwrap().clone();
        if let (Some(b), Value::Object(m)) = (&self.brain_shared, &mut v) {
            m.insert("brain_lane".into(), b.metrics.lock().unwrap().clone());
        }
        v
    }

    pub fn shutdown(&self) {
        stop_lane_thread(&self.tx, &self.thread);
        if let Some(b) = &self.brain {
            b.shutdown();
        }
    }
}

impl Drop for Helper {
    fn drop(&mut self) {
        // The helper's weights and Metal buffers must be gone before the process exits (ggml-metal aborts at exit with
        // live residency sets), so the thread is joined.
        self.shutdown();
    }
}

/// Load the helper's vocabulary (no weights, no GPU).
pub(crate) fn load_vocab(model: &std::path::Path) -> Result<Arc<Vocab>, String> {
    Vocab::load(model).map(Arc::new).map_err(|e| format!("{}: {e}", model.display()))
}
