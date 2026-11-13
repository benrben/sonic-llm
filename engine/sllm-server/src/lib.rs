//! sllm-server: the HTTP API in front of the SLLM engine (phase 1 task 5).
//!
//! One **executor thread** owns the [`engine::Engine`] and every call into it. A small tokio runtime (`http_threads` worker
//! threads) serves HTTP with axum and talks to the executor through bounded channels: one command channel in, one event
//! channel per request out. [`llama_engine::LlamaEngine`] (libllama: any GGUF) and [`fake::FakeEngine`] (CPU tests) implement
//! the engine trait.

// The json! of llama-server's generation_settings is long.
#![recursion_limit = "512"]

/// `variant_names!(Type { Variant => "name", ... })` gives `Type` a `name(self)`: the text of its variant. It is one `match`, so
/// the compiler insists that every variant has a name. With a complaint after the list (`"unknown {:?}"`, whose `{:?}` is
/// the text that named nothing) `Type` also gets `parse(&str) -> Result<Type, String>`, the other way round.
macro_rules! variant_names {
    ($ty:ident { $($variant:ident => $name:expr),+ $(,)? }) => {
        impl $ty {
            pub fn name(self) -> &'static str {
                match self {
                    $($ty::$variant => $name),+
                }
            }
        }
    };
    ($ty:ident { $($variant:ident => $name:expr),+ $(,)? }, $complaint:literal) => {
        variant_names!($ty { $($variant => $name),+ });

        impl $ty {
            pub fn parse(s: &str) -> Result<$ty, String> {
                [$($ty::$variant),+].into_iter().find(|v| v.name() == s).ok_or_else(|| format!($complaint, s))
            }
        }
    };
}

pub mod anthropic;
pub mod auto_mode;
pub mod api;
pub mod chat;
pub mod compaction;
pub mod engine;
pub mod exactness;
pub mod executor;
pub mod fake;
pub mod kvconv;
pub mod kvmem;
pub mod levels;
pub mod lifecycle;
pub mod llama_engine;
pub mod next_engine;
pub mod helper;
pub mod json_schema;
pub mod calibrate; // HOOK(profile): generic model profile + startup calibration (profile.rs, calibrate.rs)
pub mod profile;
pub mod prefix_tree;
pub mod prompt_cache;
pub mod sched;
pub mod shared_draft;
pub mod spec;
pub mod spill_io;
pub mod stop;
pub mod thinking;
pub mod threads;
pub mod tokenizer;
pub mod toolcall;

use axum::serve::ListenerExt;
use serde_json::{json, Value};
use std::net::{SocketAddr, TcpListener};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const GIT_COMMIT: &str = env!("SLLM_GIT_COMMIT");
pub const GGML_BUILD: &str = env!("SLLM_GGML_BUILD");

/// What `--version` prints: crate version, git commit and the ggml build the dylibs come from.
pub fn version_line() -> String {
    format!("Sonic LLM server (sllm-server) {CRATE_VERSION} (commit {GIT_COMMIT}) ggml {GGML_BUILD}")
}

pub fn version_json() -> Value {
    json!({"name": "sllm-server", "version": CRATE_VERSION, "commit": GIT_COMMIT, "ggml": GGML_BUILD})
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    /// 0 picks a free port (see [`Running::addr`]).
    pub port: u16,
    /// Tokio worker threads: the async runtime and the HTTP server are one runtime. Derived from the thread budget
    /// ([`threads`]); the library default is only for tests.
    pub http_threads: usize,
    /// The resolved thread plan, shown in `/caps` as `threads` (null when the caller did not resolve one).
    pub threads: Value,
    /// Capacity of the command channel into the executor. A full queue answers 503.
    pub queue_cap: usize,
    /// Stop on SIGINT/SIGTERM (the binary); tests leave this off.
    pub handle_signals: bool,
    /// The scheduler: lanes, queue limits, the step's token budget (phase 2).
    pub sched: sched::SchedConfig,
    /// The agent lifecycle (phase 3): suspend to SSD, wake queue, durable registry.
    pub lifecycle: lifecycle::LifecycleConfig,
    /// The helper lane (phase 3, `--helper-model`): a second model on its own thread with its own page pool.
    pub helper: Option<helper::HelperConfig>,
    /// Compaction: the default policy and the summarizer (phase 3).
    pub compaction: compaction::CompactionConfig,
    /// The Anthropic routes: the tool-result limit (`--tool-result-limit`) and the SSE keep-alive.
    pub anthropic: anthropic::AnthropicConfig,
    /// llama-server compatibility: `--alias`, `--api-key`, `--reasoning-format`.
    pub compat: api::llama_compat::CompatConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            host: "127.0.0.1".into(),
            port: 8080,
            http_threads: 2,
            queue_cap: 1024,
            handle_signals: false,
            threads: Value::Null,
            sched: sched::SchedConfig::default(),
            lifecycle: lifecycle::LifecycleConfig::default(),
            helper: None,
            compaction: compaction::CompactionConfig::default(),
            anthropic: anthropic::AnthropicConfig::default(),
            compat: Default::default(),
        }
    }
}

/// A started server: the HTTP thread and the executor thread.
pub struct Running {
    pub addr: SocketAddr,
    pub shared: Arc<executor::Shared>,
    /// The helper lane, when one was configured.
    pub helper: Option<Arc<helper::Helper>>,
    cmd: mpsc::Sender<executor::Command>,
    stop_http: Option<oneshot::Sender<()>>,
    http: Option<std::thread::JoinHandle<()>>,
    exec: Option<std::thread::JoinHandle<()>>,
}

impl Running {
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Wait until the engine (and the helper lane, if any) is built, or failed to build.
    pub fn wait_ready(&self, timeout: Duration) -> Result<(), String> {
        let t0 = Instant::now();
        while !self.shared.ready.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some(e) = self.shared.init_error.lock().unwrap().clone() {
                return Err(e);
            }
            if t0.elapsed() > timeout {
                return Err(format!("engine not ready after {timeout:?}"));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        match &self.helper {
            Some(h) => h.wait_ready(timeout.saturating_sub(t0.elapsed())),
            None => Ok(()),
        }
    }

    /// Block until the HTTP side stops (a signal, with `handle_signals`), then stop the executor.
    pub fn wait(mut self) {
        if let Some(h) = self.http.take() {
            let _ = h.join();
        }
        self.stop_executor();
    }

    pub fn shutdown(mut self) {
        if let Some(tx) = self.stop_http.take() {
            let _ = tx.send(());
        }
        if let Some(h) = self.http.take() {
            let _ = h.join();
        }
        self.stop_executor();
    }

    fn stop_executor(&mut self) {
        let _ = self.cmd.try_send(executor::Command::Shutdown);
        if let Some(h) = self.exec.take() {
            let _ = h.join();
        }
        if let Some(h) = &self.helper {
            h.shutdown();
        }
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        if let Some(tx) = self.stop_http.take() {
            let _ = tx.send(());
        }
        // The HTTP side detaches (it ends on its own). The executor is joined when the Shutdown could be queued: it answers
        // between two engine steps, and a real engine must be dropped (weights, Metal buffers) before the process exits or
        // ggml-metal aborts at exit with live residency sets.
        if self.cmd.try_send(executor::Command::Shutdown).is_ok() {
            if let Some(h) = self.exec.take() {
                let _ = h.join();
            }
        }
        // the helper's weights and Metal buffers go before the process exits too
        if let Some(h) = &self.helper {
            h.shutdown();
        }
    }
}

/// Bind, start the executor thread (which builds the engine with `factory`) and the HTTP runtime.
/// The rollover's words in the brain's tokens, in the frames of its chat template.
fn rollover_text(tokenizer: &tokenizer::Tokenizer) -> compaction::RolloverText {
    let template = match tokenizer.gguf_path() {
        Some(p) => chat::ChatTemplate::from_gguf(p),
        None => Ok(chat::ChatTemplate::chatml()),
    };
    let (frames, bos) = template.and_then(|t| Ok((t.turn_frames()?, t.bos().to_string()))).unwrap_or_else(|e| {
        eprintln!("sllm-server: rollover: plain frames: {e}");
        (compaction::TurnFrames::plain(), String::new())
    });
    match tokenizer.vocab() {
        Some(v) => {
            let (special, plain, text, f) = (v.clone(), v.clone(), v.clone(), v);
            compaction::RolloverText::new(
                &frames,
                &bos,
                Arc::new(move |t| special.tokenize(t, false, true).unwrap_or_default()),
                Arc::new(move |t| plain.tokenize(t, false, false).unwrap_or_default()),
                Arc::new(move |ids| String::from_utf8_lossy(&ids.iter().flat_map(|&t| text.piece(t)).collect::<Vec<u8>>()).into_owned()),
                Some(Arc::new(move |r, t, cap| helper::brain_facts(&f, r, t, cap))),
            )
        }
        None => {
            let bytes: Arc<dyn Fn(&str) -> Vec<i32> + Send + Sync> = Arc::new(|t| t.bytes().map(i32::from).collect());
            compaction::RolloverText::new(&frames, &bos, bytes.clone(), bytes, Arc::new(|ids| ids.iter().map(|&t| t as u8 as char).collect()), None)
        }
    }
}

pub fn start(cfg: ServerConfig, tokenizer: tokenizer::Tokenizer, factory: executor::EngineFactory) -> std::io::Result<Running> {
    let std_listener = TcpListener::bind((cfg.host.as_str(), cfg.port))?;
    std_listener.set_nonblocking(true)?;
    let addr = std_listener.local_addr()?;

    let shared = Arc::new(executor::Shared::default());
    let (cmd_tx, cmd_rx) = mpsc::channel(cfg.queue_cap.max(1));
    cfg.sched.validate().map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    cfg.lifecycle.validate().map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    // The helper lane: its own thread loads its model now, beside the executor's. With a brain vocabulary it also writes
    // compaction's summaries (asynchronously, on its reserved compaction slot) unless another summarizer was configured.
    let helper = match cfg.helper.clone() {
        Some(hc) => Some(Arc::new(helper::Helper::start(hc)?)),
        None => None,
    };
    let comp = compaction_config(&cfg, helper.as_ref(), &tokenizer);
    let exec = {
        let shared = shared.clone();
        let sched = cfg.sched.clone();
        let life = cfg.lifecycle.clone();
        std::thread::Builder::new().name("sllm-executor".into()).spawn(move || executor::run_with(factory, cmd_rx, shared, sched, comp, life))?
    };

    let state = Arc::new(api::ApiState {
        cmd: cmd_tx.clone(),
        queue_cap: cfg.queue_cap,
        shared: shared.clone(),
        next_req: AtomicU64::new(1),
        tokenizer,
        started: Instant::now(),
        http_threads: cfg.http_threads,
        threads: cfg.threads.clone(),
        version: version_json(),
        helper: helper.clone(),
        chat: Default::default(),
        anthropic: cfg.anthropic.clone(),
        compat: cfg.compat.clone(),
        slots: cfg.sched.max_rows(),
    });
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let http = spawn_http(std_listener, state, stop_rx, &cfg)?;

    Ok(Running { addr, shared, helper, cmd: cmd_tx, stop_http: Some(stop_tx), http: Some(http), exec: Some(exec) })
}

/// The compaction settings the executor runs with: the helper writes the summaries when there is one and a brain
/// vocabulary (unless another summarizer was configured), and rollover gets the model's text functions.
fn compaction_config(cfg: &ServerConfig, helper: Option<&Arc<helper::Helper>>, tokenizer: &tokenizer::Tokenizer) -> compaction::CompactionConfig {
    let mut comp = cfg.compaction.clone();
    if let (Some(h), Some(v)) = (helper, tokenizer.vocab()) {
        if comp.summarizer.0.name() == "stub-first-n" {
            comp.summarizer = compaction::SummarizerHandle(Arc::new(helper::HelperSummarizer::new(h.clone(), v)));
        }
    }
    if comp.rollover.on {
        comp.text = Some(rollover_text(tokenizer));
    }
    comp
}

/// The HTTP thread: its own tokio runtime of `http_threads` workers serves the API until `stop_rx` fires (or a signal,
/// with `handle_signals`).
fn spawn_http(std_listener: TcpListener, state: Arc<api::ApiState>, stop_rx: oneshot::Receiver<()>, cfg: &ServerConfig) -> std::io::Result<std::thread::JoinHandle<()>> {
    let handle_signals = cfg.handle_signals;
    let threads = cfg.http_threads.max(1);
    std::thread::Builder::new().name("sllm-http".into()).spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(threads)
            // spawn_blocking (tokenizing a text prompt) shares the budget: its pool is capped at the worker count.
            .max_blocking_threads(threads)
            .thread_name("sllm-http-worker")
            .enable_all()
            .build()
            .expect("tokio runtime");
        rt.block_on(async move {
            let listener = tokio::net::TcpListener::from_std(std_listener).expect("listener").tap_io(|tcp| {
                let _ = tcp.set_nodelay(true); // small SSE writes must not wait for Nagle
            });
            let serve = axum::serve(listener, api::router(state));
            tokio::select! {
                r = serve => { if let Err(e) = r { eprintln!("sllm-server: http error: {e}"); } }
                _ = stop_rx => {}
                _ = shutdown_signal(handle_signals) => {}
            }
        });
        rt.shutdown_timeout(Duration::from_secs(2));
    })
}

async fn shutdown_signal(enabled: bool) {
    if !enabled {
        return std::future::pending().await;
    }
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
    let mut int = signal(SignalKind::interrupt()).expect("SIGINT handler");
    tokio::select! { _ = term.recv() => {}, _ = int.recv() => {} }
}
