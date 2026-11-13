//! Loading the model and making the context (`LlamaEngine::new`), and freeing them again.

use super::*;
use load_plan::{Choices, KvChoice, KvPlanner};

impl LlamaEngine {
    /// Load the model with every layer on the GPU and create the context. Runs on the executor thread.
    pub fn new(cfg: EngineConfig, vocab: Arc<Vocab>) -> Result<LlamaEngine, String> {
        let mut ld = Loaded::open(&cfg)?;
        let profile_check = ld.profile_check();
        let ch = Choices::make(&cfg, &ld).inspect_err(|_| ld.abandon())?;
        let readers = ld.make_readers(&cfg).inspect_err(|_| ld.abandon())?;
        let eog = Arc::new(vocab.eog_ids().to_vec());
        let eos_id = eog.first().copied().unwrap_or(-1);
        let info = EngineInfo { name: "llama".into(), model: cfg.model.display().to_string(), n_vocab: ld.n_vocab, n_ctx: ld.capacity, eos_id };
        let caps = ld.caps(&cfg, &ch, profile_check, &eog);
        let pcache = open_prompt_cache(&cfg, ld.capacity, ld.kv.kv_type, ch.row_exact, !readers.is_empty());
        let rcache = pcache.as_ref().filter(|_| !readers.is_empty()).and_then(|c| {
            let budget = cfg.prompt_cache.as_ref().map_or(0, |p| p.1);
            crate::prompt_cache::DiskCache::open(c.dir().join("reader-f16"), budget).ok()
        });
        let share_align = if ch.row_exact { cfg.n_ubatch.max(1) } else { 1 };
        let own_spill_dir = cfg.spill_dir.is_none();
        let spill_dir = cfg.spill_dir.clone().unwrap_or_else(|| {
            std::env::var_os("SLLM_SPILL_DIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir).join(format!("sllm-llama-agents-{}", std::process::id()))
        });
        let free_sids = (0..ld.n_seq as i32).rev().collect();
        let (model, unified, swa_full, share) = (ld.model, cfg.kv_unified, cfg.swa_full, ch.share);
        Ok(LlamaEngine {
            cfg,
            model,
            ctx: ld.ctx,
            batch: Batch::new(ld.n_batch),
            vocab,
            n_vocab: ld.n_vocab,
            eog,
            arch: ld.arch,
            info,
            caps,
            capacity: ld.capacity,
            swa_ring: !swa_full && unsafe { ffi::llama_model_n_swa(model) } > 0,
            total_cells: ld.total_cells,
            share,
            unified,
            free_sids,
            agents: HashMap::new(),
            jobs: HashMap::new(),
            counters: EngineCounters::default(),
            prefix: PrefixCounters { sharing: share || ch.holders_on, ..Default::default() },
            timings: HashMap::new(),
            last_bd: StepBreakdown::default(),
            events: Vec::new(),
            spill_dir,
            own_spill_dir,
            ssd_bytes: 0,
            ssd_writes: 0,
            ssd_reads: 0,
            prof: std::env::var_os("SLLM_LLAMA_PROF").map(|_| Vec::new()),
            io: SpillIo::new(),
            next_key: 1,
            stall: Stall::default(),
            next_order: 1,
            mem: MemStats::default(),
            job_failures: Vec::new(),
            ptree: std::cell::RefCell::new(PrefixTree::new()),
            holders: Vec::new(),
            holders_on: ch.holders_on,
            stream_runs: StreamRuns::default(),
            seq_pack: std::env::var_os("SLLM_SEQ_PACK").is_none_or(|v| v != "0"),
            // flash attention is on (see the context parameters in `make_context`), so V rows are cells like K rows
            copy_partial: share && !unified && ffi::has_copy_rows() && std::env::var_os("SLLM_COPY_PARTIAL").is_none_or(|v| v != "0"),
            copy_queue: CopyQueue::Idle,
            copy_holes: std::cell::Cell::new(false),
            turn_save: std::env::var_os("SLLM_TURN_SAVE").is_none_or(|v| v != "0"),
            read_pick: std::env::var_os("SLLM_KV_READ_PICK").is_none_or(|v| v != "0"),
            seen_prefix: HashMap::new(),
            recent_starts: std::collections::VecDeque::new(),
            holder_stats: HolderStats::default(),
            check_prefix: std::env::var_os("SLLM_PREFIX_CHECK").is_some(),
            no_suspend: false,
            spec: ch.spec,
            spec_why: ch.spec_why,
            row_exact: ch.row_exact,
            pcache,
            rcache,
            rstash: Vec::new(),
            handovers: Vec::new(),
            share_align,
            spec_state: HashMap::new(),
            spec_stats: SpecStats::default(),
            readers,
        })
    }
}

/// What loading found: the model, its main context, the KV cache the memory budget holds, and what libllama reports about
/// them. Owns the model and the context until the engine takes them ([`Loaded::abandon`] frees them when `new` fails).
pub(super) struct Loaded {
    model: *mut sys::llama_model,
    ctx: *mut sys::llama_context,
    /// libllama's log of making the context (its KV and compute-buffer lines).
    ctx_log: String,
    pub(super) arch: String,
    prof: crate::profile::Profile,
    kv: KvChoice,
    n_vocab: usize,
    /// Cells one sequence may hold (`--ctx` rounded up to 256).
    capacity: usize,
    pub(super) unified: bool,
    pub(super) recurrent: bool,
    pub(super) hybrid: bool,
    total_cells: usize,
    n_seq: usize,
    n_batch: usize,
    /// The model's sliding window (0: none).
    swa: i32,
}

impl Loaded {
    /// Load the model, plan the KV cache in the memory budget and make the context.
    fn open(cfg: &EngineConfig) -> Result<Loaded, String> {
        let path = CString::new(cfg.model.to_string_lossy().as_bytes()).map_err(|_| "model path contains NUL".to_string())?;
        let arch = gguf_arch(&cfg.model).unwrap_or_else(|| "unknown".into());
        let capacity = pad(cfg.n_ctx.max(1));
        let slots = cfg.max_rows.max(1);
        let unified = cfg.kv_unified;
        let model = load_model(cfg, &path, &arch)?;
        // SAFETY: plain libllama calls on the model just loaded.
        let n_vocab = unsafe {
            let vocab_p = sys::llama_model_get_vocab(model);
            sys::llama_vocab_n_tokens(vocab_p) as usize
        };
        // HOOK(profile): the unified cache is sized from the model profile (profile.rs): KV bytes per cell by layer kind
        // (full and sliding-window layers apart), the weights, libllama's own compute-buffer figure, the memory budget.
        // The cells are what --ctx x slots (or --kv-pages) asks for, up to what the budget holds.
        let prof = model_profile(cfg)?;
        let ws = cfg.working_set_bytes.or_else(|| sllm_core::caps::Capabilities::probe().ok().map(|c| c.recommended_working_set_bytes));
        // --kv-read f16: the readers' f16 caches of one sequence come out of the budget first
        // SAFETY: plain libllama calls on the model just loaded.
        let (recurrent, hybrid) = unsafe { (ffi::llama_model_is_recurrent(model), ffi::llama_model_is_hybrid(model)) };
        let can_read = !unified && !recurrent && !hybrid;
        let bandwidth = crate::profile::startup().map_or_else(|| crate::profile::Machine::detect().bandwidth_gbs, |s| s.machine.bandwidth_gbs);
        let kv = KvPlanner { cfg, prof: &prof, capacity, slots, ws, bandwidth }.choose(can_read);
        // per-sequence streams (the default): the profile sized `streams x capacity`, one stream per sequence id
        let total_cells = kv.budget.cells as usize;
        // a context the budget cannot hold is refused here, before libllama allocates anything
        if let Err(e) = kv.budget.check_fits(&prof, capacity as u64) {
            // SAFETY: the model was loaded above and nothing else holds it.
            unsafe { sys::llama_model_free(model) };
            return Err(e);
        }
        let n_seq = if unified { (slots * 2).clamp(MIN_SEQ_IDS, MAX_SEQ_IDS) } else { kv.budget.n_seq as usize };
        let (ctx, ctx_log) = make_context(cfg, model, kv.kv_type, total_cells, n_seq);
        if ctx.is_null() {
            // SAFETY: the model was loaded above and nothing else holds it.
            unsafe { sys::llama_model_free(model) };
            return Err(format!("llama_init_from_model failed (arch {arch}, {total_cells} cells, {n_seq} sequences)"));
        }
        // SAFETY: plain libllama calls on the context and the model just made.
        let (n_batch, swa) = unsafe { (sys::llama_n_batch(ctx) as usize, ffi::llama_model_n_swa(model)) };
        Ok(Loaded { model, ctx, ctx_log, arch, prof, kv, n_vocab, capacity, unified, recurrent, hybrid, total_cells, n_seq, n_batch, swa })
    }

    /// Partial copies of a sequence's cache are valid (no sliding window, no recurrent state).
    pub(super) fn can_share(&self) -> bool {
        self.swa == 0 && !self.recurrent && !self.hybrid
    }

    /// Free the context and the model: `new` failed after making them.
    fn abandon(&self) {
        // SAFETY: both were made by `open` and nothing else holds them.
        unsafe {
            sys::llama_free(self.ctx);
            sys::llama_model_free(self.model);
        }
    }

    /// HOOK(profile): the header-derived profile against the loaded model and against libllama's KV lines.
    fn profile_check(&mut self) -> Value {
        // SAFETY: the model is loaded.
        let diffs = unsafe { self.prof.verify_model(self.model) };
        let kv_lines = crate::profile::parse_kv_lines(&self.ctx_log);
        let kv_libllama: f64 = kv_lines.iter().map(|l| l.bytes()).sum();
        let kv_profile = self.kv.budget.kv_bytes(&self.prof, self.total_cells as u64) as f64;
        json!({
            "model_differences": diffs,
            "kv_bytes_libllama": kv_libllama, "kv_bytes_profile": kv_profile,
            "kv_matches": (kv_libllama - kv_profile).abs() <= 0.02 * 1048576.0 * kv_lines.len().max(1) as f64,
            "compute_buffer_bytes_libllama": crate::profile::parse_compute_bytes(&self.ctx_log),
            "compute_buffer_bytes_planned": self.kv.budget.compute_bytes,
            "prefix_sharing_safe_profile": self.prof.prefix_sharing_safe(), "prefix_sharing_safe_engine": self.can_share(),
        })
    }

    /// --kv-read f16 with a quantized cache: the readers (their cells came out of the budget in `open`); none otherwise.
    fn make_readers(&self, cfg: &EngineConfig) -> Result<Vec<Reader>, String> {
        let kv_type = self.kv.kv_type;
        if self.kv.read_f16 && kv_type != KvType::F16 && !self.unified && !self.recurrent && !self.hybrid {
            // SAFETY: the main context was just made and holds nothing yet.
            let made = unsafe { Reader::probe(self.ctx) }.and_then(|layout| {
                // the chat reader, then the long reader (its context is made for a read)
                reader_cells(self.capacity).iter().enumerate().map(|(i, &cells)| Reader::new(self.model, layout.clone(), cells, i > 0, cfg.n_ubatch.max(1), cfg.swa_full, cfg.cpu_threads, self.n_vocab)).collect::<Result<Vec<_>, _>>()
            });
            let mut readers = made?;
            readers.iter_mut().for_each(|r| r.ring = self.swa > 0);
            return Ok(readers);
        }
        if self.kv.read_f16 {
            let why = if kv_type == KvType::F16 { "the cache is f16 already" } else if self.unified { "unified KV cache" } else { "recurrent state" };
            eprintln!("sllm-server: --kv-read f16 not used: {why}");
        }
        Ok(Vec::new())
    }

    /// What `/caps` says about the model, the context and the engine's choices.
    fn caps(&self, cfg: &EngineConfig, ch: &Choices, profile_check: Value, eog: &[i32]) -> Value {
        let mut desc = vec![0 as c_char; 256];
        // SAFETY: `desc` has room for the 256 bytes libllama may write (its last one a NUL).
        let desc = unsafe {
            sys::llama_model_desc(self.model, desc.as_mut_ptr(), desc.len());
            std::ffi::CStr::from_ptr(desc.as_ptr()).to_string_lossy().into_owned()
        };
        let (kv_type, unified, recurrent, hybrid) = (self.kv.kv_type, self.unified, self.recurrent, self.hybrid);
        let kv_prof = crate::profile::KvType::parse(kv_type.name()).unwrap_or(crate::profile::KvType::F16);
        let bytes_per_cell = {
            let (f, s) = self.prof.kv_bytes_per_cell(kv_prof);
            f + s
        };
        json!({
            "engine": "llama-generic",
            "kind": "llama",
            "arch": self.arch,
            "model_desc": desc,
            "libllama": "b11669 (vendored)",
            "build": crate::exactness::backend().as_ref().map_or_else(|e| json!({"error": e}), |b| b.to_json()),
            "exact": crate::exactness::exact_json(&crate::exactness::ExactFacts {
                requested: match cfg.row_exact {
                    RowExact::Off => "off",
                    RowExact::Auto => "auto",
                    RowExact::On => "on",
                },
                engine_on: ch.row_exact,
                library_state: crate::exactness::library_switch(),
                unified,
                moe: ch.moe,
                recurrent_or_hybrid: recurrent || hybrid,
                scheduler_rows: cfg.max_rows,
                limit: crate::exactness::row_limit_here(),
            }),
            // SAFETY: plain libllama call on the loaded model.
            "n_ctx_train": unsafe { ffi::llama_model_n_ctx_train(self.model) },
            "kv": {
                "mode": if unified { "unified (libllama)" } else { "per-sequence streams (libllama)" }, "type": kv_type.name(), "type_choice": self.kv.why, "n_ctx": self.capacity, "total_cells": self.total_cells, "n_seq_max": self.n_seq,
                "n_ubatch": cfg.n_ubatch, "n_batch": self.n_batch, "bytes_per_cell_estimate": bytes_per_cell,
                "n_swa": self.swa, "recurrent": recurrent, "hybrid": hybrid,
            },
            // HOOK(profile): generic model profile, derived budgets, startup calibration (profile.rs, calibrate.rs)
            "profile": self.prof.to_json(),
            "budget": self.kv.budget.to_json(&self.prof),
            "profile_check": profile_check,
            "calibration": crate::profile::startup().map_or(Value::Null, |s| s.calibration.clone()),
            "auto_mode": crate::profile::auto_mode().cloned().unwrap_or(Value::Null),
            "batching": "one llama_batch per step over every running request (continuous batching)",
            "prefix_sharing": {
                "mode": if ch.share { "on" } else if ch.holders_on { "holders (sliding-window or recurrent model: whole-sequence copies of frozen prefixes)" } else { "off" },
                "unit": if ch.share { "token-exact common prefix, llama_memory_seq_cp" } else { "frozen system-prompt holder, full llama_memory_seq_cp" },
                "coalesced_prefill": "not supported",
            },
            "lifecycle": {"suspend": "llama_state_seq_get_data to the SSD", "restore": "llama_state_seq_set_data"},
            "eos_ids": eog,
            "sampling": "greedy (host argmax over the row's logits); ignore_eos masks the end-of-generation tokens",
            "grammar": "GBNF (llama.cpp syntax, root rule `root`) through libllama's grammar sampler; the argmax is checked first and the row masked only when the grammar refuses it; speculation is off in steps that decode a grammar request",
            "agents": "one sequence id per /v1/agents agent; its cells persist between turns",
            "unsupported": ["per-role kv precision policies and own-q8 (one cache type: --kv-type)", "coalesced prefill"],
        })
    }
}

/// Load the model with every layer on the GPU.
fn load_model(cfg: &EngineConfig, path: &CString, arch: &str) -> Result<*mut sys::llama_model, String> {
    // SAFETY: plain libllama calls; the model is checked before use and freed in Drop (or by `new` when it fails).
    let model = unsafe {
        let mut mp = sys::llama_model_default_params();
        mp.n_gpu_layers = 999;
        sys::llama_model_load_from_file(path.as_ptr(), mp)
    };
    if model.is_null() {
        return Err(format!("llama_model_load_from_file failed for {} (arch {arch})", cfg.model.display()));
    }
    Ok(model)
}

/// The model's profile: the one the server derived at startup, else read from the model file.
fn model_profile(cfg: &EngineConfig) -> Result<crate::profile::Profile, String> {
    Ok(match crate::profile::startup() {
        Some(st) => st.profile.clone(),
        None => crate::profile::load_profile(&cfg.model).map_err(|e| format!("model profile: {e}"))?,
    })
}

/// The main context: `total_cells` cells of `kv_type` for `n_seq` sequences, and libllama's log of making it.
fn make_context(cfg: &EngineConfig, model: *mut sys::llama_model, kv_type: KvType, total_cells: usize, n_seq: usize) -> (*mut sys::llama_context, String) {
    let slots = cfg.max_rows.max(1);
    // SAFETY: plain libllama calls on the model just loaded; the caller checks the context before use.
    unsafe {
        let mut cp = sys::llama_context_default_params();
        cp.n_ctx = total_cells as u32;
        cp.n_seq_max = n_seq as u32;
        cp.n_ubatch = cfg.n_ubatch.max(1) as u32;
        cp.n_batch = (cfg.step_tokens + slots).max(cfg.n_ubatch).max(1) as u32;
        cp.flash_attn_type = sys::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.kv_unified = cfg.kv_unified;
        cp.swa_full = cfg.swa_full; // HOOK(profile)
        cp.type_k = kv_type.ggml();
        cp.type_v = kv_type.ggml();
        if cfg.cpu_threads > 0 {
            cp.n_threads = cfg.cpu_threads as i32;
            cp.n_threads_batch = cfg.cpu_threads as i32;
        }
        // HOOK(profile): libllama's own KV / compute-buffer lines of this context, checked against the profile (`profile_check`)
        crate::profile::capture_log(|| sys::llama_init_from_model(model, cp))
    }
}

/// The prompt cache: a directory per model file and context layout (a state only loads into the same layout).
fn open_prompt_cache(cfg: &EngineConfig, capacity: usize, kv_type: KvType, row_exact: bool, read_f16: bool) -> Option<crate::prompt_cache::DiskCache> {
    cfg.prompt_cache.as_ref().filter(|_| !cfg.kv_unified).and_then(|(root, budget)| {
        let meta = std::fs::metadata(&cfg.model).ok()?;
        let mtime = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
        // the build and settings that make the bits (exactness.rs): a saved state only loads into the kernels that made it
        let build = crate::exactness::backend().as_ref().map_err(|e| eprintln!("sllm-server: prompt cache off: {e}")).ok()?;
        let key = format!(
            "{}|{}|{mtime}|{}|{capacity}|{}|{row_exact}|ubatch{}|{}{}",
            cfg.model.display(),
            meta.len(),
            kv_type.name(),
            cfg.swa_full,
            cfg.n_ubatch,
            build.key(),
            if read_f16 { "|read-f16" } else { "" }
        );
        let dir = root.join(format!("{:016x}", crate::prompt_cache::hash_tokens(&key.bytes().map(i32::from).collect::<Vec<_>>())));
        crate::prompt_cache::DiskCache::open(dir, *budget).map_err(|e| eprintln!("sllm-server: prompt cache off: {e}")).ok()
    })
}

impl Drop for LlamaEngine {
    fn drop(&mut self) {
        if let Some(p) = self.prof.take() {
            // medians per row count: (rows, decodes, wait-for-GPU us, argmax us)
            let mut by: std::collections::BTreeMap<u32, Vec<(u64, u64)>> = Default::default();
            for (s, a, n) in p {
                by.entry(n).or_default().push((s, a));
            }
            for (n, v) in by {
                let med = |f: fn(&(u64, u64)) -> u64| {
                    let mut x: Vec<u64> = v.iter().map(f).collect();
                    x.sort();
                    x[x.len() / 2]
                };
                eprintln!("SLLM_LLAMA_PROF rows={n} decodes={} first_logits_wait_us={} argmax_us={}", v.len(), med(|x| x.0), med(|x| x.1));
            }
        }
        self.jobs.clear(); // the grammar samplers
        self.readers.clear(); // their contexts before the model
        unsafe {
            if !self.ctx.is_null() {
                sys::llama_free(self.ctx);
            }
            if !self.model.is_null() {
                sys::llama_model_free(self.model);
            }
        }
        self.io.finish(); // pending removals run before the directory goes
        if self.own_spill_dir {
            let _ = std::fs::remove_dir(&self.spill_dir); // only if empty: agents' snapshots are removed with the agents
        }
    }
}
