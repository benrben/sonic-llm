//! Opening the library's engine and reading the settings (environment variables) the engine runs with.

use super::*;

impl NextEngine {
    pub fn new(cfg: NextConfig, vocab: Arc<Vocab>) -> Result<NextEngine, String> {
        let path = CString::new(cfg.model.display().to_string()).map_err(|e| e.to_string())?;
        let mut buf = vec![0 as c_char; 512];
        let e = unsafe {
            sn_open(path.as_ptr(), KERNELS.as_ptr() as *const c_char, cfg.slots as c_int, cfg.n_ctx as c_int, cfg.max_rows as c_int, cfg.pages as c_int, buf.as_mut_ptr(), buf.len() as c_int)
        };
        if e.is_null() {
            return Err(unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned());
        }
        let n_vocab = unsafe { sn_n_vocab(e) } as usize;
        // a request that may not end (ignore_eos) has its argmax skip these on the GPU: no scores back to the CPU, and its
        // queued step is fed the token it takes (more than 8: such a row's end token is still replaced in `sample`)
        let eog = vocab.eog_ids();
        unsafe { sn_set_eog(e, eog.as_ptr(), if eog.len() <= 8 { eog.len() as c_int } else { 0 }) };
        let gpu_eog = eog.len() <= 8;
        let owners = SlotTable::new(cfg.slots);
        let cfg_slots = cfg.slots;
        let (io_tx, io_rx) = mpsc::channel();
        let prefix = PrefixCounters { sharing: true, ..Default::default() };
        let own_spill_dir = cfg.spill_dir.is_none();
        let spill_dir = cfg.spill_dir.clone().unwrap_or_else(|| {
            std::env::var_os("SLLM_SPILL_DIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir).join(format!("sllm-next-agents-{}", std::process::id()))
        });
        Ok(NextEngine {
            e, recurrent: unsafe { sn_recurrent(e) } != 0, wide: env_on("SLLM_NEXT_SPEC_WIDE") && unsafe { sn_recurrent(e) } == 0, cells: vec![Vec::new(); cfg.slots], cfg, vocab, n_vocab, owners, agents: HashMap::new(), jobs: HashMap::new(),
            counters: EngineCounters::default(), prefix, gpu_ms: 0.0, run_ms: 0.0, want_probs: false, probs: Vec::new(), io_tx, io_rx, events: Vec::new(), life: LifeCounters::default(),
            spill_dir, own_spill_dir, last_used: vec![0; cfg_slots], tick: 0, pages_peak: 0, chain: env_on("SLLM_NEXT_CHAIN"), gpu_eog,
            spec: spec_config(e),
            guess: env_on("SLLM_NEXT_GUESS") && unsafe { sn_guess_head(e) != 0 && sn_guess_on(e, 1) != 0 },
            guess_k: env_num::<usize>("SLLM_NEXT_GUESS_K", 3).clamp(1, 3),  // 3: 130 tok/s against 125 with 2 (40 answers, a guess is 0.54 ms)
            guess_steps: 0, guess_drafted: 0, guess_kept: 0, draft_guess: false,
            spec_steps: 0, spec_drafted: 0, spec_kept: 0,
            // (not a recurrent model: its background rows keep no state to go back to)
            bg_draft: env_on("SLLM_NEXT_BG_DRAFT") && unsafe { sn_recurrent(e) } == 0,
            coalesce: env_on("SLLM_NEXT_COALESCE"),
            // off by default: the unmap behind it held the GPU 9-15 s in the server (2 ms in a bare test), stalling requests
            trim: env_is_1("SLLM_NEXT_TRIM"),
            trim_at: std::time::Instant::now(),
            // off by default: without giving the memory back (SLLM_NEXT_TRIM) it only slows a follow-up (2026-11-10: a follow-up
            // 15 s after a 20-document swarm re-read everything, 24.6 -> 55.5 s, and the process kept its 10 GB)
            idle_drop: env_is_1("SLLM_NEXT_IDLE_DROP"),
            idle_since: None,
            trim_pending: false,
            pins: Vec::new(), recent: std::collections::VecDeque::new(), pin_made: 0, pin_used: 0, pin_tokens: 0,
            pin_max: env_num::<usize>("SLLM_NEXT_PINS", PIN_MAX),
            pin_min: env_num::<usize>("SLLM_NEXT_PIN_MIN", PIN_MIN).max(1),
            kept_state: env_num::<usize>("SLLM_NEXT_KEPT_STATE_MB", 1024) << 20,
            idle_after: std::time::Duration::from_secs_f64(env_num("SLLM_NEXT_IDLE_S", 5.0)),
            idle_cache: env_num::<usize>("SLLM_NEXT_IDLE_CACHE_MB", 1024) << 20,
            bg_rows: env_num("SLLM_NEXT_BG_ROWS", 16),
            small: unsafe { sn_small_rows(e) }.max(0) as usize,
            bg_k0: env_num("SLLM_NEXT_BG_K0", 2),
            bg_steps: 0, bg_drafted: 0, bg_kept: 0,
            shared: env_on("SLLM_NEXT_SHARED_DRAFT").then(|| SharedDraft::new(env_num("SLLM_NEXT_SHARED_TOKENS", 4 << 20), env_num("SLLM_NEXT_SHARED_N", 4))),
            shared_steps: 0, shared_drafted: 0, shared_kept: 0, draft_shared: false,
            emb_norm: 0.0, n_embd: unsafe { sn_n_embd(e) } as usize, realign: Vec::new(), soft: None, soft_gpu: false, lat_stop: Vec::new(), lat_mark: -1, lat_rows: 0,
        })
    }
}

/// A setting that is on unless its environment variable is "0".
fn env_on(name: &str) -> bool {
    std::env::var(name).map_or(true, |v| v != "0")
}

/// A setting that is off unless its environment variable is "1".
fn env_is_1(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

/// A number from an environment variable, or `default` when it is unset or not a number.
fn env_num<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Prompt lookup's settings.
///
/// On by default (SLLM_NEXT_SPEC=off turns it off): it costs nothing since a queued step is taken whether or not the caller
/// chains on, and steps queue unless the next lookup can match (Phi-3.5 after a document 89 -> 110 tok/s, Qwen3-4B level).
/// A recurrent model (Qwen3.5 / 3.6) takes back refused drafts through spare state entries (sn_draft_rows): one-row kernels
/// only (no wide checks), 2 drafts a step (a row costs it about a fifth of a step).
fn spec_config(e: *mut SnEngine) -> SpecConfig {
    let mode = if std::env::var("SLLM_NEXT_SPEC").map_or(true, |v| v != "off") { SpecMode::Auto } else { SpecMode::Off };
    // min_keep 0.2: Qwen3-4B after a document 99 -> 117 tok/s (0.4 sat out at its ~45% keep rate)
    // k 2: a 3-row step costs 1.2x a 1-row one, 4 rows 1.6-1.9x (step_bench)
    SpecConfig { mode, k: if env_on("SLLM_NEXT_SPEC_WIDE") && unsafe { sn_recurrent(e) } == 0 { 7 } else { 2 }, n: 3, max_rows: 8, max_seqs: 1, min_keep: 0.2 }
}
