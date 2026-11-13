//! `sllm-server`: SLLM's HTTP API server.
//!
//! ```text
//! sllm-server --model GGUF [--auto-mode] [--agents N]
//!             [--kv-type auto|f16|q8_0|q4_0] [--kv-auto-4bit on|off] [--kv-read kv|f16] [--kv-pages N] [--ubatch N] [--working-set-bytes N]
//!             [--host H] [--port P] [--threads N] [--http-threads N] [--cpu-threads N] [--threads-config PATH]
//!             [--ctx N] [--queue-cap N] [--tool-result-limit auto|off|N] [--version]
//!             [--interactive-cap N] [--background-cap N] [--helper-cap N] [--interactive-queue N] [--background-queue N]
//!             [--helper-queue N] [--chunk N] [--step-tokens N] [--helper-tokens N] [--bg-prefill-interactive N]
//!             [--bg-prefill-period N] [--bg-rows-interactive N] [--chat-gap-ms N] [--bg-write-share F] [--bg-gap-ms N] [--prefill-fair 0|1] [--prefill-fair-min N]
//!             [--mixed-chunks N] [--burst-prefill N] [--burst-yield N] [--bg-prefill-forced N] [--front-prefill-solo 0|1]
//!             [--prefill-fair-granule 0|1]
//!             [--prefix-sharing on|off] [--kv-streams per-seq|unified]
//!             [--row-exact auto|on|off] [--spec off|auto|on] [--spec-k N] [--spec-n N] [--spec-rows N] [--spec-max-seqs N]
//!             [--agent-dir DIR] [--spill-dir DIR] [--idle-suspend-ms N] [--pressure-low F] [--pressure-high F]
//!             [--max-restoring N] [--ssd-budget-bytes N] [--inbox-cap N] [--pressure-relief on|off]
//!             [--prompt-cache on|off] [--prompt-cache-dir DIR] [--prompt-cache-gb N]
//!             [--helper-model GGUF] [--helper-ctx N] [--helper-kv-pages N] [--helper-kv-type f16|q8_0|q4_0]
//!             [--helper-slots N] [--helper-reserve-pages N] [--helper-step N]
//!             [--helper-classify brain|helper|auto] [--helper-classify-kv-pages N] [--helper-classify-step N] [--helper-classify-slots N]
//!             [--compact-window N] [--compact-every N] [--compact-tail N] [--compact-summary N]
//! sllm-server --fake [--model GGUF] [--fake-seed N] [--fake-prefill-ms F] [--fake-prefill-ms-per-token F]
//!             [--fake-token-ms F] [--fake-natural-len N] [--probe-caps] ...
//! ```
//!
//! `--auto-mode` sizes everything from this machine and this model (`auto_mode.rs`): give the job with `--agents N`, `--ctx N`,
//! both or neither; a line at start says what it chose, `/caps` `auto_mode` has the same, and with `--print-profile` it
//! prints that and stops, no GPU. It also turns `--pressure-relief` on and sets `--ssd-budget-bytes` to 64 GB. A flag you pass wins.
//! With `--model` and no `--fake` the engine runs the GGUF through libllama (`LlamaEngine`): any architecture llama.cpp
//! loads, one unified KV cache shared by every sequence, continuous batching over one `llama_decode` per step. `--ctx` is the
//! context of ONE request or agent; `--kv-pages` (pages of 256 cells) sizes the whole cache, and without it the cache holds the default
//! job (`profile::DEFAULT_AGENTS` agents and a chat at `--ctx` tokens each; more requests wait for a stream). `--kv-type` is the type of that
//! cache (auto by default: f16 unless the memory budget cannot hold what is asked for, then q8_0 when that holds more; 4-bit notes
//! only with `--kv-auto-4bit on`, because their quality cost depends on the model). The old model-specific native engine is gone (preserved at the git tag `native-engine-final`):
//! `--engine next` (Sonic LLM's own engine) is the default since 2026-11-13, `--engine llama` asks for the libllama engine, `--engine native` is refused, and so are the flags
//! only it had (`--kv-policy`, `--kv-layout`); `--kv paged` is accepted and ignored, `--kv contiguous` too with a warning.
//! The scheduler flags (see `sched.rs`): lane slots (`--interactive-cap`, default and most 256, `--background-cap`, default 256,
//! `--helper-cap`) and queue limits (`--*-queue`, past which a request is deferred with a 503 and `retry_after`), the
//! prefill `--chunk` (default `--ubatch`) and per-step token budget `--step-tokens` (default 2,048), split fairly over a
//! lane's prefilling requests (`--prefill-fair 1`, the default; 0 = in arrival order; no share below `--prefill-fair-min`,
//! default 256; a step of more than `--mixed-chunks` chunks, default 8, leaves background decode rows out), the helper's
//! reserved share `--helper-tokens`, and the interactive guards: background prefill tokens per step while an
//! interactive request decodes (`--bg-prefill-interactive`, default 0) with a full chunk every `--bg-prefill-period`
//! deferred steps (default 256), and a cap on background decode rows meanwhile (`--bg-rows-interactive`, default 32).
//! Burst: `--burst-prefill N` (default 0 = off) lets a burst of at least N prefilling background requests
//! finish its prefill before background decode rows ride along (prefill-first); `--burst-yield N` (default 0 = off;
//! batch mode: 8) lets N or more prefilling background requests take the whole step while an interactive request decodes
//! (its row stays); `--bg-prefill-forced N` (default: the
//! chunk) caps the background chunk forced through while an interactive request decodes; `--front-prefill-solo 0|1`
//! (default 1) defers background prefill while an interactive request prefills too; `--prefill-fair-granule 0|1`
//! (default 1) makes fair shares whole multiples of `--prefill-fair-min`.
//! `--prefix-sharing on|off` (default on): a sequence whose prompt starts with the same tokens as another's copies that
//! prefix (the cells are shared in the unified cache) and prefills only the rest; off for sliding-window and recurrent
//! models. `/caps` reports the mode.
//! `--working-set-bytes` replaces the device's recommended working set, of which the memory budget is 0.9; it exists to test
//! admission. `--fake` runs the deterministic FakeEngine (task 5a); `--model` then only sets its vocabulary size and the tokenizer.
//! `--threads N` is the one thread budget (tokio workers, which also serve HTTP, plus ggml CPU threads, the executor
//! thread among them); it defaults to the value swept for this machine class in `engine/config/threads.toml`, never to
//! the core count. `--http-threads` and `--cpu-threads` override their share. See `threads.rs` and `/caps`.
//! The agent lifecycle (phase 3, `lifecycle.rs`): `--agent-dir DIR` keeps a durable registry of the agents (token history
//! and metadata, `DIR/registry`) that a restarted server loads, and their SSD snapshots in `DIR/kv` (`--spill-dir`
//! overrides that; default a per-process directory under `$SLLM_SPILL_DIR` or the temp dir). `--idle-suspend-ms N`
//! suspends an agent idle in RAM that long (default off); a request the KV pool would refuse suspends the coldest idle
//! agents to make room with `--pressure-relief on` (default off: admission refuses as in phase 2; a restore always
//! makes room); `--pressure-low F` / `--pressure-high F` (default 0 = off / 0.20) also suspend them ahead of
//! time when a pool's free share drops under the low mark;
//! `--max-restoring N` (default 4) bounds the restores in flight; `--ssd-budget-bytes N` drops the longest-suspended
//! agents to history past that many snapshot bytes; `--inbox-cap N` (default: no cap) bounds the turns queued per agent.
//! `--helper-model GGUF` (phase 3) loads a second, small model (any GGUF; Qwen3-0.6B Q4_0 is the one measured) in this process
//! as its own libllama model and context on its own thread, with its own KV cache (`--helper-kv-pages`, default 64 pages of
//! 256 cells, `--helper-kv-type`, default f16), and serves the
//! context loop's operations on it: `POST /v1/helper/{classify,extract,compress,summarize}` (see `helper/mod.rs`).
//! `--helper-ctx` is the most tokens one helper call may hold (default 8,192), `--helper-slots` the calls run at once
//! (default 8; one is held for `"priority": "compaction"` calls, which also get `--helper-reserve-pages`, default 16),
//! `--helper-step` the prefill tokens of one helper step (default 2,048).
//! `--helper-classify brain|helper|auto` says where classify and compress scoring run: on the helper model, or on the
//! brain already in the process (a second lane over the brain's GGUF, whose mmapped weights are the same physical pages
//! as the brain's: no second copy; its own KV cache `--helper-classify-kv-pages`, default 32, steps of
//! `--helper-classify-step` tokens, default 512, `--helper-classify-slots`, default 8). `auto`, the default, picks the
//! brain when it is Gemma-3: the 0.6B helper called 45% of the fixed on-track set right, the brain 95%. Extract and
//! summarize always run on the helper model.
//! `--compact-window N` / `--compact-every N` (phase 3, default off) give every new agent a compaction policy: after a
//! turn that leaves N tokens or more in its window (or N tokens added since its last compaction), the window becomes
//! `[system prompt] + [summary of --compact-summary tokens, default 300] + [last --compact-tail tokens, default 1024]`.
//! With `--helper-model` the helper writes the summary (asynchronously, on its compaction slot: `helper-summarize`);
//! without it the summary is a STUB (the first tokens of the replaced history).
//! `--probe-caps` fills the fake engine's `/caps` with the task 1 capability record (the real engine always does).

//! `--bg-write-share F` (default 0.5, 0 = off) and `--bg-gap-ms N` (default 250, 0 = no limit): while background requests write
//! and others read, the writers keep this share of the time and a read step lasts at most about this long.
//! `--chat-gap-ms N` (default 40, 0 = off): while a chat decodes, background rows that make a step longer than this sit out
//! enough steps that the chat's tokens come this far apart on average, and ride along in full in the others.
//! `--rollover on|off` (default on): the end of the window does not end an agent's turn. It stops just short of the end,
//! the agent writes notes of up to `--rollover-summary N` tokens (default 128) in its own stream, the window becomes
//! `[system prompt] + [notes] + [facts] + [last --rollover-tail tokens, default 1024]` and the turn goes on, at most
//! `--rollover-max N` times (default: no cap). A prompt longer than the window is read in slices the same way.
//!
//! `--read-qmm on|off` (default on): prompt chunks of up to 4,096 rows take the qmm weight kernel instead of ggml's mul_mm
//! (`SLLM_QMM32_MAXN`). Gemma 4 E2B reads 7% faster with the same RAM and the same known-answer score; Qwen3-4B is within 2%.
//! The last bits change against mul_mm, so a few greedy tokens can differ (2 of 18 answers in the test).
//!
//! `--many-agents on|off` (default off): the setup for a server meant for more than 16 agents at once. One weight kernel for every
//! row count and exact attention up to 32 rows, so at 32 agents answers stay closest to a lone agent's (measured 95-97% the same over
//! 160 tokens, against 83-95%), at the price of 4-17% for an agent alone. An environment variable that is already set wins.
mod cli;

use cli::flags::{parse_args, Flags, USAGE};
use cli::setup::{apply_auto_mode, apply_calibration, configure_helper, load_tokenizer, print_model_profile, resolve_scheduler, resolve_thread_plan, set_kernel_env, validate_config};
use sllm_server::fake::{FakeConfig, FakeEngine};
use sllm_server::llama_engine::{self, EngineConfig, LlamaEngine};
use sllm_server::next_engine::NextConfig;
use sllm_server::threads;
use sllm_server::tokenizer::Tokenizer;
use sllm_server::{start, version_line, ServerConfig};
use std::time::Duration;

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("sllm-server: {msg}");
    std::process::exit(2)
}

/// The engine to run (sllm-next, libllama or the fake engine) and everything it is built from.
struct EngineChoice {
    is_fake: bool,
    use_next: bool,
    probe_caps: bool,
    real: EngineConfig,
    next: NextConfig,
    fake: FakeConfig,
}

impl EngineChoice {
    fn new(f: Flags) -> (EngineChoice, ServerConfig) {
        let Flags { cfg, fake, is_fake, probe_caps, use_next, real, .. } = f;
        // --engine next: a cache of --kv-pages (in 64-cell pages, taken as sequences grow), sequences of up to --ctx cells
        let next = NextConfig {
            model: real.model.clone(),
            n_ctx: real.n_ctx,
            // sequences are cheap (the cache is pages taken as they grow): one per --kv-pages page, 4 to 256
            slots: real.pool_pages.map_or(16, |p| p.clamp(4, 256)),
            // a step's prompt budget plus a write row for every lane slot (and drafts): with 64 rows of headroom, 74 writing agents
            // beside a full prompt chunk made a step of 1,089 rows above 1,088, and every request in it failed (2026-11-09)
            max_rows: cfg.sched.step_tokens.max(cfg.sched.quiet_chunk.unwrap_or(0)) + cfg.sched.max_rows() + 16,
            pages: real.pool_pages.map_or(0, |p| p * sllm_server::engine::PAGE_CELLS / 64),
            spill_dir: real.spill_dir.clone(),
        };
        (EngineChoice { is_fake, use_next, probe_caps, real, next, fake }, cfg)
    }

    /// The engine's name on the banner.
    fn name(&self, cfg: &ServerConfig, model_arch: Option<&str>) -> String {
        let (real, next_cfg) = (&self.real, &self.next);
        let mut engine_name = if self.is_fake { "fake".to_string() } else { format!("llama (arch {}, ctx {}, kv {})", model_arch.unwrap_or("?"), real.n_ctx, if real.kv_auto { "auto" } else { real.kv_type.name() }) };
        if let Some(h) = &cfg.helper {
            engine_name += &format!(", helper {} ({} pages {}), classify on the {}", h.model.display(), h.pool_pages, h.kv_type.name(), h.classify_on.name());
        }
        if self.use_next && !self.is_fake {
            engine_name = format!("next (arch {}, {} sequences of up to {} cells, {} cache pages)", model_arch.unwrap_or("?"), next_cfg.slots, next_cfg.n_ctx,
                                  if next_cfg.pages > 0 { next_cfg.pages.to_string() } else { "full windows".into() });
        }
        engine_name
    }

    /// What builds the engine on the executor's thread.
    fn factory(self, vocab: Option<std::sync::Arc<llama_ref::Vocab>>) -> sllm_server::executor::EngineFactory {
        let EngineChoice { is_fake, use_next, probe_caps, real, next: next_cfg, mut fake } = self;
        Box::new(move || {
            if use_next && !is_fake {
                let engine = sllm_server::next_engine::NextEngine::new(next_cfg, vocab.expect("a model implies a vocabulary"))?;
                return Ok(Box::new(engine) as Box<dyn sllm_server::engine::Engine>);
            }
            if !is_fake {
                let engine = LlamaEngine::new(real, vocab.expect("a model implies a vocabulary"))?;
                return Ok(Box::new(engine) as Box<dyn sllm_server::engine::Engine>);
            }
            if probe_caps {
                let caps = sllm_core::caps::Capabilities::probe().map_err(|e| format!("capability probe failed: {e}"))?;
                fake.capabilities = Some(serde_json::to_value(caps).map_err(|e| e.to_string())?);
            }
            Ok(Box::new(FakeEngine::new(fake)) as Box<dyn sllm_server::engine::Engine>)
        })
    }
}

/// Starts the server on the chosen engine, waits until it is ready, prints the banner and blocks until it stops.
fn run_server(f: Flags, tokenizer: Tokenizer, model_arch: Option<&str>, plan: threads::Plan) {
    let (choice, cfg) = EngineChoice::new(f);
    let vocab = tokenizer.vocab();
    let http_threads = cfg.http_threads;
    let engine_name = choice.name(&cfg, model_arch);
    let factory = choice.factory(vocab);
    let running = start(cfg, tokenizer, factory).unwrap_or_else(|e| die(format!("cannot start: {e}")));
    if let Err(e) = running.wait_ready(Duration::from_secs(120)) {
        die(format!("engine failed to start: {e}"));
    }
    println!("{}", version_line());
    println!(
        "Sonic LLM server: engine {engine_name}, http threads {http_threads}, ggml cpu threads {}, thread budget {} ({}), listening on http://{}",
        plan.cpu,
        plan.budget(),
        plan.source_label(),
        running.addr
    );
    use std::io::Write;
    let _ = std::io::stdout().flush();
    running.wait();
    println!("Sonic LLM server: stopped");
}

/// The configured server, from the thread plan on: the helper model, the vocabulary, then the engine and the HTTP server.
fn start_configured(mut f: Flags, model_arch: Option<&str>, plan: threads::Plan) {
    f.cfg.http_threads = plan.http;
    f.real.cpu_threads = plan.cpu;
    f.cfg.threads = plan.to_json();
    configure_helper(&mut f, model_arch);
    if !f.is_fake && f.model.is_none() {
        die(format!("--model GGUF is required (or --fake)\n{USAGE}"));
    }
    if !f.is_fake || f.probe_caps {
        // Initialise the ggml device registry BEFORE the vocabulary loads: libllama would otherwise create it first and the
        // capability record would miss ggml's Metal device log (sllm-core captures that log at registry init).
        let _ = sllm_core::backend::registry();
    }
    let tokenizer = load_tokenizer(&mut f);
    run_server(f, tokenizer, model_arch, plan);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(mut f) = parse_args(&args) else { return };
    // The architecture, read from the GGUF header (no GPU): reported at start, and picks where the helper classifies.
    let model_arch = f.model.as_deref().and_then(llama_engine::gguf_arch);
    // read once when the Metal device starts, so before anything below touches it; a variable already set wins
    set_kernel_env(&f);
    if !f.is_fake && f.model.is_some() {
        eprintln!("sllm-server: engine llama (libllama: arch {})", model_arch.as_deref().unwrap_or("unknown"));
    }
    // HOOK(profile): the generic model profile, printed and exposed in /caps `profile`; --print-profile stops here.
    if f.print_profile && !f.auto_mode {
        print_model_profile(&f);
        return;
    }
    let plan = resolve_thread_plan(&mut f);
    apply_calibration(&mut f);
    resolve_scheduler(&mut f);
    validate_config(&mut f);
    if apply_auto_mode(&mut f) {
        return;
    }
    start_configured(f, model_arch.as_deref(), plan);
}
