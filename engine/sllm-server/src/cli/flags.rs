//! The command line: every flag read into [`Flags`], flag by flag in the order given (`--help` prints [`USAGE`]).

use super::llama_flags;
use crate::die;
use sllm_server::compaction::Policy;
use sllm_server::fake::FakeConfig;
use sllm_server::llama_engine::{EngineConfig, KvType};
use sllm_server::threads::Overrides;
use sllm_server::{version_line, ServerConfig};
use std::path::PathBuf;

pub(crate) const USAGE: &str = "Sonic LLM server (sllm-server)\nusage: sllm-server --model GGUF [--auto-mode] [--agents N] [--kv-type auto|f16|q8_0|q4_0] [--kv-auto-4bit on|off] [--kv-read kv|f16] [--kv-pages N] \
[--ubatch N] [--working-set-bytes N] [--threads N] [--cpu-threads N] [--http-threads N] [--threads-config PATH] [--host H] [--port P] [--ctx N] [--queue-cap N] [--tool-result-limit auto|off|N] [--version] \
[--interactive-cap N] [--background-cap N] [--helper-cap N] [--interactive-queue N] [--background-queue N] [--helper-queue N] \
[--chunk N] [--step-tokens N] [--helper-tokens N] [--bg-prefill-interactive N] [--bg-prefill-period N] [--bg-rows-interactive N] [--chat-gap-ms N] [--bg-write-share F] [--bg-gap-ms N] [--prefill-fair 0|1] [--prefill-fair-min N] [--mixed-chunks N] \
[--burst-prefill N] [--burst-yield N] [--bg-prefill-forced N] [--front-prefill-solo 0|1] [--prefill-fair-granule 0|1] [--prefix-sharing on|off] \
[--agent-dir DIR] [--spill-dir DIR] [--idle-suspend-ms N] [--pressure-low F] [--pressure-high F] [--max-restoring N] [--ssd-budget-bytes N] [--inbox-cap N] [--pressure-relief on|off]\n       \
[--prompt-cache on|off] [--prompt-cache-dir DIR] [--prompt-cache-gb N]\n       \
[--helper-model GGUF] [--helper-ctx N] [--helper-kv-pages N] [--helper-kv-type f16|q8_0|q4_0] [--helper-slots N] [--helper-reserve-pages N] [--helper-step N]\n       \
[--helper-classify brain|helper|auto] [--helper-classify-kv-pages N] [--helper-classify-step N] [--helper-classify-slots N]\n       \
[--compact-window N] [--compact-every N] [--compact-tail N] [--compact-summary N] [--rollover on|off] [--rollover-summary N] [--rollover-tail N] [--rollover-max N] [--many-agents on|off] [--read-qmm on|off] [--calibrate on|off] [--print-profile]\n       \
sllm-server --fake [--model GGUF] [--fake-seed N] [--fake-prefill-ms F] [--fake-prefill-ms-per-token F] [--fake-token-ms F] \
[--fake-natural-len N] [--probe-caps]\n\
llama-server spellings accepted: -m/--model -c/--ctx-size -t/--threads -np/--parallel (= --agents) -ub/--ubatch-size -a/--alias NAME --api-key KEY --api-key-file FILE --reasoning-format auto|deepseek|none --host --port\n\
(accepted and ignored, with a note: -ngl --jinja -fa -b --mlock --no-mmap -kvu --slots --metrics -cb; and for old scripts: --kv paged)\n       [--engine next|llama] default next: Sonic LLM's own engine (a model it cannot run goes to the libllama engine)";


/// The flag being read and the rest of the command line, where its value comes from.
pub(crate) struct Arg<'a, 'b> {
    name: &'a str,
    rest: &'b mut std::slice::Iter<'a, String>,
}

impl Arg<'_, '_> {
    /// The value after the flag, or exit 2 when the command line ends.
    pub(crate) fn val(&mut self) -> String {
        self.rest.next().cloned().unwrap_or_else(|| die(format!("{} needs a value", self.name)))
    }

    pub(crate) fn num<T: std::str::FromStr>(&mut self) -> T {
        let v = self.val();
        v.parse().unwrap_or_else(|_| die(format!("{}: bad value {v:?}", self.name)))
    }

    pub(crate) fn path(&mut self) -> PathBuf {
        PathBuf::from(self.val())
    }

    /// The value read by `parse`; its error message is the whole complaint.
    fn parse_with<T, E: std::fmt::Display>(&mut self, parse: impl FnOnce(&str) -> Result<T, E>) -> T {
        parse(&self.val()).unwrap_or_else(|e| die(e))
    }

    /// The value read by `parse`; its error message follows the flag's name.
    pub(crate) fn parse_with_flag<T, E: std::fmt::Display>(&mut self, parse: impl FnOnce(&str) -> Result<T, E>) -> T {
        let name = self.name;
        self.parse_with(|v| parse(v).map_err(|e| format!("{name}: {e}")))
    }

    /// `--flag on|off`.
    fn on_off(&mut self) -> bool {
        match self.val().as_str() {
            "on" => true,
            "off" => false,
            v => die(format!("{}: expected on or off, got {v:?}", self.name)),
        }
    }
}

/// What the command line said, before anything is validated or started.
pub(crate) struct Flags {
    pub(crate) cfg: ServerConfig,
    pub(crate) fake: FakeConfig,
    pub(crate) model: Option<PathBuf>,
    pub(crate) is_fake: bool,
    pub(crate) probe_caps: bool,
    pub(crate) use_next: bool,
    pub(crate) swa_choice: String, // HOOK(profile): --swa-full auto|on|off
    pub(crate) ubatch_explicit: bool,
    pub(crate) calibrate_mode: sllm_server::calibrate::Mode, // HOOK(profile): --calibrate on|off, default off
    pub(crate) print_profile: bool, // HOOK(profile): --print-profile prints the model profile and budget as JSON and exits (no GPU)
    pub(crate) real: EngineConfig,
    // the prompt cache on disk (prompt_cache.rs): on by default, kept across restarts
    pub(crate) pcache_on: bool,
    pub(crate) pcache_dir: Option<PathBuf>,
    pub(crate) pcache_gb: u64,
    pub(crate) budget: Option<usize>,
    pub(crate) overrides: Overrides,
    pub(crate) threads_config: Option<PathBuf>,
    pub(crate) chunk: Option<usize>,
    pub(crate) step_tokens: Option<usize>,
    pub(crate) helper: sllm_server::helper::HelperConfig,
    pub(crate) helper_on: bool,
    pub(crate) classify_on: Option<sllm_server::helper::ClassifyOn>, // None: auto (the brain when it is Gemma-3)
    pub(crate) compact: Option<Policy>,
    pub(crate) rollover: sllm_server::compaction::Rollover,
    pub(crate) many_agents: bool,
    pub(crate) read_qmm: bool,
    pub(crate) auto_mode: bool,
    pub(crate) agents: Option<usize>,
    pub(crate) ctx_flag: bool,
    pub(crate) parallel: Option<usize>, // -np/--parallel (llama_flags.rs)
    pub(crate) relief_flag: bool,
}

impl Flags {
    fn new() -> Flags {
        let mut real = EngineConfig::new(PathBuf::new());
        real.kv_auto = true; // --kv-type auto: f16 unless memory is short
        real.kv_read_auto = true; // then 4-bit notes, read in f16, unless --kv-read says otherwise
        real.spec.mode = sllm_server::spec::SpecMode::Auto; // exact prompt lookup while one sequence decodes, where exact decode is on
        Flags {
            cfg: ServerConfig { handle_signals: true, ..Default::default() },
            fake: FakeConfig::default(),
            model: None,
            is_fake: false,
            probe_caps: false,
            use_next: true, // Sonic LLM's own engine is the default since 2026-11-13; `--engine llama` asks for the libllama one
            swa_choice: "auto".to_string(),
            ubatch_explicit: false,
            calibrate_mode: sllm_server::calibrate::Mode::Off,
            print_profile: false,
            real,
            pcache_on: true,
            pcache_dir: None,
            pcache_gb: 16,
            budget: None,
            overrides: Overrides::default(),
            threads_config: None,
            chunk: None,
            step_tokens: None,
            helper: sllm_server::helper::HelperConfig::new(PathBuf::new()),
            helper_on: false,
            classify_on: None,
            compact: None,
            rollover: sllm_server::compaction::Rollover { on: true, ..Default::default() },
            many_agents: false,
            read_qmm: true,
            auto_mode: false,
            agents: None,
            ctx_flag: false,
            parallel: None,
            relief_flag: false,
        }
    }
}


/// Reads the command line into `Flags`, flag by flag in the order given. `None`: `--version` or `--help` was printed, so the
/// process is done.
pub(crate) fn parse_args(args: &[String]) -> Option<Flags> {
    let mut f = Flags::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--version" | "-V" => {
                println!("{}", version_line());
                return None;
            }
            "--help" | "-h" => {
                println!("{USAGE}");
                return None;
            }
            _ => {}
        }
        if llama_flags::read_ignored(a, &mut it) {
            continue;
        }
        let Some((_, read)) = FLAGS.iter().find(|(name, _)| *name == llama_flags::long_name(a)) else {
            die(format!("unknown argument {a:?}\n{USAGE}"))
        };
        read(&mut Arg { name: a, rest: &mut it }, &mut f);
    }
    llama_flags::resolve_parallel(&mut f);
    Some(f)
}

/// Reads one flag (and its value, when it takes one) into `Flags`.
type ReadFlag = fn(&mut Arg, &mut Flags);

/// Every flag with how it is read, grouped by what it configures (each flag appears once).
static FLAGS: &[(&str, ReadFlag)] = &[
    // Switches without a value.
    ("--fake", |_, f| f.is_fake = true),
    ("--auto-mode", |_, f| f.auto_mode = true),
    ("--probe-caps", |_, f| f.probe_caps = true),
    ("--print-profile", |_, f| f.print_profile = true), // HOOK(profile)
    // Settings on the flags themselves.
    ("--swa-full", |a, f| f.swa_choice = a.val()), // HOOK(profile): auto|on|off
    ("--read-qmm", |a, f| f.read_qmm = a.on_off()),
    ("--many-agents", |a, f| f.many_agents = a.on_off()),
    ("--prompt-cache", |a, f| f.pcache_on = a.on_off()),
    ("--prompt-cache-gb", |a, f| f.pcache_gb = a.num()),
    // Settings stored straight into a field of a sub-configuration: a number, text or on/off.
    ("--fake-seed", |a, f| f.fake.seed = a.num()),
    ("--fake-prefill-ms", |a, f| f.fake.prefill_ms = a.num()),
    ("--fake-prefill-ms-per-token", |a, f| f.fake.prefill_ms_per_token = a.num()),
    ("--fake-token-ms", |a, f| f.fake.token_ms = a.num()),
    ("--host", |a, f| f.cfg.host = a.val()),
    ("--port", |a, f| f.cfg.port = a.num()),
    ("--queue-cap", |a, f| f.cfg.queue_cap = a.num()),
    ("--prefix-sharing", |a, f| f.real.prefix_sharing = a.on_off()),
    ("--kv-auto-4bit", |a, f| f.real.kv_allow_4bit = a.on_off()),
    ("--helper-ctx", |a, f| f.helper.n_ctx = a.num()),
    ("--helper-kv-pages", |a, f| f.helper.pool_pages = a.num()),
    ("--helper-slots", |a, f| f.helper.slots = a.num()),
    ("--helper-reserve-pages", |a, f| f.helper.reserve_pages = a.num()),
    ("--helper-step", |a, f| f.helper.step_tokens = a.num()),
    ("--helper-classify-kv-pages", |a, f| f.helper.brain_pool_pages = a.num()),
    ("--helper-classify-step", |a, f| f.helper.brain_step_tokens = a.num()),
    ("--helper-classify-slots", |a, f| f.helper.brain_slots = a.num()),
    ("--rollover", |a, f| f.rollover.on = a.on_off()),
    ("--rollover-summary", |a, f| f.rollover.summary_tokens = a.num()),
    ("--rollover-tail", |a, f| f.rollover.tail_tokens = a.num()),
    ("--rollover-max", |a, f| f.rollover.max_per_turn = a.num()),
    // The same, one level deeper.
    ("--helper-tokens", |a, f| f.cfg.sched.helper_tokens = a.num()),
    ("--prefill-fair-min", |a, f| f.cfg.sched.fair_min = a.num()),
    ("--mixed-chunks", |a, f| f.cfg.sched.max_mixed_chunks = a.num()),
    ("--bg-prefill-interactive", |a, f| f.cfg.sched.bg_prefill_interactive = a.num()),
    ("--bg-prefill-period", |a, f| f.cfg.sched.bg_prefill_period = a.num()),
    ("--burst-prefill", |a, f| f.cfg.sched.burst_min = a.num()),
    ("--burst-yield", |a, f| f.cfg.sched.burst_yield = a.num()),
    ("--chat-gap-ms", |a, f| f.cfg.sched.chat_gap_ms = a.num()),
    ("--bg-write-share", |a, f| f.cfg.sched.bg_write_share = a.num()),
    ("--bg-gap-ms", |a, f| f.cfg.sched.bg_gap_ms = a.num()),
    ("--pressure-low", |a, f| f.cfg.lifecycle.pressure_low = a.num()),
    ("--pressure-high", |a, f| f.cfg.lifecycle.pressure_high = a.num()),
    ("--max-restoring", |a, f| f.cfg.lifecycle.max_restoring = a.num()),
    ("--inbox-cap", |a, f| f.cfg.lifecycle.inbox_cap = a.num()),
    // Settings that stay unset (None) unless given.
    ("--model", |a, f| f.model = Some(a.path())),
    ("--agents", |a, f| f.agents = Some(a.num())),
    ("--threads", |a, f| f.budget = Some(a.num())),
    ("--threads-config", |a, f| f.threads_config = Some(a.path())),
    ("--chunk", |a, f| f.chunk = Some(a.num())),
    ("--step-tokens", |a, f| f.step_tokens = Some(a.num())),
    ("--prompt-cache-dir", |a, f| f.pcache_dir = Some(a.path())),
    ("--fake-natural-len", |a, f| f.fake.natural_len = Some(a.num())),
    ("--http-threads", |a, f| f.overrides.http_threads = Some(a.num())),
    ("--cpu-threads", |a, f| f.overrides.cpu_threads = Some(a.num())),
    ("--working-set-bytes", |a, f| f.real.working_set_bytes = Some(a.num())),
    ("--kv-pages", |a, f| f.real.pool_pages = Some(a.num())),
    ("--spill-dir", |a, f| f.real.spill_dir = Some(a.path())),
    ("--bg-prefill-forced", |a, f| f.cfg.sched.bg_prefill_forced = Some(a.num())),
    ("--bg-rows-interactive", |a, f| f.cfg.sched.bg_rows_interactive = Some(a.num())),
    ("--idle-suspend-ms", |a, f| f.cfg.lifecycle.idle_suspend_ms = Some(a.num())),
    ("--ssd-budget-bytes", |a, f| f.cfg.lifecycle.ssd_budget_bytes = Some(a.num())),
    // Switches given as 0 or 1.
    ("--prefill-fair", |a, f| f.cfg.sched.prefill_fair = a.num::<usize>() != 0),
    ("--prefill-fair-granule", |a, f| f.cfg.sched.fair_granule = a.num::<usize>() != 0),
    ("--front-prefill-solo", |a, f| f.cfg.sched.front_prefill_solo = a.num::<usize>() != 0),
    // The scheduler's three lanes: slots and queue limits.
    ("--interactive-cap", |a, f| f.cfg.sched.caps[0] = a.num()),
    ("--background-cap", |a, f| f.cfg.sched.caps[1] = a.num()),
    ("--helper-cap", |a, f| f.cfg.sched.caps[2] = a.num()),
    ("--interactive-queue", |a, f| f.cfg.sched.queue_limits[0] = a.num()),
    ("--background-queue", |a, f| f.cfg.sched.queue_limits[1] = a.num()),
    ("--helper-queue", |a, f| f.cfg.sched.queue_limits[2] = a.num()),
    // Speculation limits, held to a sane range.
    ("--spec-k", |a, f| f.real.spec.k = a.num::<usize>().clamp(1, 15)),
    ("--spec-n", |a, f| f.real.spec.n = a.num::<usize>().max(1)),
    ("--spec-rows", |a, f| f.real.spec.max_rows = a.num::<usize>().clamp(2, 16)),
    ("--spec-max-seqs", |a, f| f.real.spec.max_seqs = a.num::<usize>().max(1)),
    // The compaction policy (the first of its flags creates it).
    ("--compact-window", |a, f| f.compact.get_or_insert_with(Policy::untriggered).window_tokens = Some(a.num())),
    ("--compact-every", |a, f| f.compact.get_or_insert_with(Policy::untriggered).every_tokens = Some(a.num())),
    ("--compact-tail", |a, f| f.compact.get_or_insert_with(Policy::untriggered).tail_tokens = a.num()),
    ("--compact-summary", |a, f| f.compact.get_or_insert_with(Policy::untriggered).summary_tokens = a.num()),
    // Values read by the parser of their own type, and the flags kept for old scripts.
    ("--engine", |a, f| parse_engine_choice(&a.val(), f)),
    ("--calibrate", |a, f| f.calibrate_mode = a.parse_with(sllm_server::calibrate::Mode::parse)), // HOOK(profile)
    ("--tool-result-limit", |a, f| f.cfg.anthropic.tool_result_limit = a.parse_with_flag(sllm_server::anthropic::ToolResultLimit::parse)),
    ("--kv", |a, _| kv_layout_ignored(&a.val())),
    ("--kv-type", |a, f| (f.real.kv_type, f.real.kv_auto) = kv_type_choice(&a.val())),
    ("--kv-read", |a, f| (f.real.kv_read_f16, f.real.kv_read_auto) = kv_read_choice(&a.val())),
    ("--kv-policy", |a, _| kv_flag_removed(a.name)),
    ("--kv-layout", |a, _| kv_flag_removed(a.name)),
    ("--kv-streams", |a, f| f.real.kv_unified = kv_streams_unified(&a.val())),
    ("--row-exact", |a, f| f.real.row_exact = a.parse_with_flag(sllm_server::llama_engine::RowExact::parse)),
    ("--spec", |a, f| f.real.spec.mode = a.parse_with_flag(sllm_server::spec::SpecMode::parse)),
    ("--helper-kv-type", |a, f| f.helper.kv_type = a.parse_with(sllm_server::helper::HelperConfig::parse_kv_type)),
    ("--helper-classify", |a, f| f.classify_on = a.parse_with(sllm_server::helper::ClassifyOn::parse)),
    // Flags that set several fields at once.
    ("--ctx", read_ctx),
    ("--ubatch", read_ubatch),
    ("--agent-dir", read_agent_dir),
    ("--pressure-relief", read_pressure_relief),
    ("--helper-model", read_helper_model),
    // llama-server's own flags (llama_flags.rs).
    ("--alias", llama_flags::read_alias),
    ("--api-key", llama_flags::read_api_key),
    ("--api-key-file", llama_flags::read_api_key_file),
    ("--reasoning-format", llama_flags::read_reasoning_format),
    ("--parallel", llama_flags::read_parallel),
];

fn read_ctx(a: &mut Arg, f: &mut Flags) {
    f.ctx_flag = true;
    let n: usize = a.num();
    f.fake.n_ctx = n;
    f.real.n_ctx = n;
}

fn read_ubatch(a: &mut Arg, f: &mut Flags) {
    f.real.n_ubatch = a.num();
    f.ubatch_explicit = true; // HOOK(profile)
}

/// `--agent-dir DIR`: the registry and the KV spill directory live under it (an explicit `--spill-dir` wins).
fn read_agent_dir(a: &mut Arg, f: &mut Flags) {
    let d = a.path();
    f.cfg.lifecycle.registry_dir = Some(d.join("registry"));
    f.real.spill_dir = f.real.spill_dir.take().or(Some(d.join("kv")));
}

fn read_pressure_relief(a: &mut Arg, f: &mut Flags) {
    f.relief_flag = true;
    f.cfg.lifecycle.relieve_on_refusal = a.on_off()
}

fn read_helper_model(a: &mut Arg, f: &mut Flags) {
    f.helper.model = a.path();
    f.helper_on = true;
}

/// `--engine llama` (libllama, the default) or `next` (sllm-next, SLLM's own Metal kernels); the old native engine is
/// preserved at the tag `native-engine-final`.
fn parse_engine_choice(v: &str, f: &mut Flags) {
    match v {
        "llama" => f.use_next = false,
        "next" | "auto" => f.use_next = true,
        "native" => die("--engine native: the native engine was deleted; sllm-server runs every model through libllama now (the old engine is preserved at the git tag native-engine-final)"),
        v => die(format!("--engine: expected llama, auto or next, got {v:?}")),
    }
}

fn kv_flag_removed(flag: &str) -> ! {
    die(format!("{flag}: removed with the native engine (the generic engine has one KV cache type, --kv-type; the old engine is preserved at the git tag native-engine-final)"))
}

/// `--kv paged|contiguous`: accepted for old scripts and ignored (contiguous with a warning).
fn kv_layout_ignored(v: &str) {
    match v {
        "paged" => {}
        "contiguous" => eprintln!("sllm-server: warning: --kv contiguous is ignored: the engine has one unified KV cache"),
        v => die(format!("--kv: expected paged or contiguous, got {v:?}")),
    }
}

/// `--kv-type`: the cache type and whether it is chosen automatically.
fn kv_type_choice(v: &str) -> (KvType, bool) {
    match v {
        "auto" => (KvType::F16, true),
        v => (KvType::parse(v).unwrap_or_else(|e| die(format!("--kv-type: {e}"))), false),
    }
}

/// `--kv-read f16|kv`: (read the cache in f16, chosen automatically).
fn kv_read_choice(v: &str) -> (bool, bool) {
    match v {
        "f16" => (true, false),
        "kv" => (false, false),
        v => die(format!("--kv-read: expected f16 or kv, got {v:?}")),
    }
}

/// `--kv-streams per-seq|unified`: whether the cache is one unified stream.
fn kv_streams_unified(v: &str) -> bool {
    match v {
        "per-seq" => false,
        "unified" => true,
        v => die(format!("--kv-streams: expected per-seq or unified, got {v:?}")),
    }
}
