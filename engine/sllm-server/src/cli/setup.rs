//! From the flags to a configuration: thread plan, calibration, scheduler sizes, the checks of flag combinations,
//! auto-mode, the helper model and the tokenizer.

use super::flags::Flags;
use crate::die;
use sllm_server::llama_engine::{self, KvType};
use sllm_server::threads::{self, MachineClass, ThreadsFile};
use sllm_server::tokenizer::Tokenizer;
use std::path::PathBuf;

/// Environment variables the Metal device reads once when it starts, so this runs before anything touches it; a variable that is
/// already set wins.
pub(crate) fn set_kernel_env(f: &Flags) {
    let mut env: Vec<(&str, &str)> = Vec::new();
    if f.read_qmm {
        env.push(("SLLM_QMM32_MAXN", "4096"));
    }
    if f.many_agents {
        env.extend([("SLLM_QMM32_N1", "1"), ("SLLM_QMM32_MAXN", "256"), ("SLLM_ROW_EXACT_MAXN", "32")]);
    }
    for (k, v) in env {
        if std::env::var_os(k).is_none() {
            std::env::set_var(k, v);
        }
    }
}

/// HOOK(profile): `--print-profile`: the model profile and budget as JSON on stdout (no GPU).
pub(crate) fn print_model_profile(f: &Flags) {
    let m = f.model.as_deref().unwrap_or_else(|| die("--print-profile needs --model GGUF"));
    let prof = sllm_server::profile::load_profile(m).unwrap_or_else(|e| die(e));
    let machine = sllm_server::profile::Machine::detect();
    let ws = f.real.working_set_bytes.or_else(|| sllm_core::caps::Capabilities::probe().ok().map(|c| c.recommended_working_set_bytes));
    let (budget, _, _) = sllm_server::profile::plan_engine(
        &prof,
        &sllm_server::profile::PlanArgs {
            n_ctx: f.real.n_ctx,
            slots: f.cfg.sched.max_rows(),
            pool_pages: f.real.pool_pages,
            n_ubatch: f.real.n_ubatch,
            step_tokens: f.step_tokens.unwrap_or(f.cfg.sched.step_tokens),
            kv_type: sllm_server::profile::KvType::parse(f.real.kv_type.name()).unwrap_or(sllm_server::profile::KvType::F16),
            unified: f.real.kv_unified,
            max_seq_ids: 256,
            swa_full: match f.swa_choice.as_str() {
                "on" => true,
                "off" => false,
                _ => f.compact.is_some(),
            },
            working_set: ws,
            bandwidth_gbs: machine.bandwidth_gbs,
        },
    );
    // what the engine would refuse before it allocates: a context the budget cannot hold
    let refusal = budget.check_fits(&prof, f.real.n_ctx.max(1).div_ceil(256) as u64 * 256).err();
    let out = serde_json::json!({"profile": prof.to_json(), "budget": budget.to_json(&prof), "refusal": refusal, "machine": {"brand": machine.brand, "bandwidth_gbs": machine.bandwidth_gbs, "source": machine.bandwidth_source}});
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}

/// One budget: --threads, else the stored sweep for this machine class (never the core count).
pub(crate) fn resolve_thread_plan(f: &mut Flags) -> threads::Plan {
    let config_path = f.threads_config.take().or_else(|| std::env::var_os("SLLM_THREADS_CONFIG").map(PathBuf::from)).unwrap_or_else(threads::default_config_path);
    let plan = threads::resolve(threads::Inputs {
        flag: f.budget,
        overrides: f.overrides,
        machine: MachineClass::detect(),
        file: Some((config_path.clone(), ThreadsFile::load(&config_path))),
    })
    .unwrap_or_else(|e| die(e));
    if let threads::Source::Fallback { reason } = &plan.source {
        eprintln!("sllm-server: warning: no swept thread budget ({reason}); using the fallback {}", threads::FALLBACK_BUDGET);
    }
    if plan.over_budget() {
        eprintln!("sllm-server: warning: --cpu-threads/--http-threads give {} threads, over the budget of {}", plan.http + plan.cpu, plan.budget());
    }
    plan
}

/// HOOK(profile): model profile (startup line, /caps `profile`) and the startup calibration; it sets the prefill chunk,
/// n_ubatch and the step budget unless a flag did. The values after it then see them as if they had been passed.
pub(crate) fn apply_calibration(f: &mut Flags) {
    if !f.is_fake {
        if let Some(m) = &f.model {
            if !m.is_file() {
                die(format!("GGUF not found: {}", m.display()));
            }
            let explicit = sllm_server::calibrate::Explicit { chunk: f.chunk.is_some(), ubatch: f.ubatch_explicit, step_tokens: f.step_tokens.is_some() };
            let tuned = sllm_server::calibrate::startup(m, f.calibrate_mode, explicit).unwrap_or_else(|e| die(e));
            if let Some(u) = tuned.n_ubatch {
                f.real.n_ubatch = u;
            }
            f.chunk = f.chunk.or(tuned.chunk);
            f.step_tokens = f.step_tokens.or(tuned.step_tokens);
        }
    }
}

/// The scheduler's prefill chunk and step budget, from the flags, the calibration and the engine choice.
pub(crate) fn resolve_scheduler(f: &mut Flags) {
    // The prefill chunk defaults to --ubatch (512: llama-server's n_ubatch, so a single prompt is chunked as llama.cpp
    // chunks it), the step's prefill budget to 2,048 tokens (four chunks; at least one).
    f.cfg.sched.chunk = f.chunk.unwrap_or(f.real.n_ubatch);
    // --engine next runs the architectures and weight types sllm-next has kernels for; any other GGUF runs on the libllama
    // engine (said once on stderr), so --engine next works for every model
    let mut next_rows = 1024;
    if let (true, false, Some(m)) = (f.use_next, f.is_fake, f.model.as_ref()) {
        match sllm_server::next_engine::check(m) {
            Ok(rows) => next_rows = rows,
            Err(why) => {
                eprintln!("sllm-server: --engine next: {why}; the libllama engine runs this model");
                f.use_next = false;
            }
        }
    }
    // --engine next reads a prompt in steps of up to 1,024 tokens (the read speed of 2,048, half the working buffers), or
    // 512 (llama.cpp's ubatch) for a model with sliding-window layers (they hold a window plus a step: Gemma 4 31B 0.42 GB
    // less) or wide layers (Mistral 7B's step buffers: one agent's RAM at ggml's), as sn_check says; reads within 2%
    if f.use_next {
        // (a MoE: bigger steps while no chat is about, unless the chunk or the budget was set by hand)
        if let (false, None, None, Some(m)) = (f.is_fake, f.chunk, f.step_tokens, f.model.as_ref()) {
            f.cfg.sched.quiet_chunk = sllm_server::next_engine::quiet_rows(m).filter(|&q| q > next_rows);
        }
        f.cfg.sched.chunk = f.chunk.unwrap_or(next_rows);
        f.step_tokens = f.step_tokens.or(Some(next_rows));
    }
    let step_default = (sllm_server::sched::DEFAULT_STEP_TOKENS, "default");
    f.cfg.sched.step_tokens = f.step_tokens.unwrap_or(step_default.0.max(f.cfg.sched.chunk));
    if f.step_tokens.is_none() && !f.is_fake {
        eprintln!("sllm-server: step budget {} tokens ({})", f.cfg.sched.step_tokens, step_default.1);
    }
    f.cfg.sched.helper_tokens = f.cfg.sched.helper_tokens.min(f.cfg.sched.step_tokens);
}

/// Checks the combinations of flags, in order: compaction, scheduler and lifecycle settings, --swa-full, then --agents and
/// --auto-mode against the rest. Also hands the rollover and compaction settings to the server and the row and step sizes to the engine.
pub(crate) fn validate_config(f: &mut Flags) {
    validate_policies(f);
    // HOOK(profile): sliding-window layers keep a ring of n_swa + n_ubatch cells (llama-server's default) unless compaction is
    // on: agent_cut drops a token range, which a ring cannot do.
    f.real.swa_full = match f.swa_choice.as_str() {
        "on" => true,
        "off" => false,
        "auto" => f.cfg.compaction.default_policy.is_some(),
        v => die(format!("--swa-full: expected auto, on or off, got {v:?}")),
    };
    f.real.max_rows = f.cfg.sched.max_rows();
    f.real.step_tokens = f.cfg.sched.step_tokens;
    validate_auto_mode_flags(f);
}

/// The rollover and compaction settings to the server (the compaction policy checked), then the scheduler's and the
/// lifecycle's own checks.
fn validate_policies(f: &mut Flags) {
    f.cfg.compaction.rollover = f.rollover;
    if let Some(p) = f.compact.take() {
        p.validate(0).unwrap_or_else(|e| die(format!("--compact-*: {e}")));
        f.cfg.compaction.default_policy = Some(p);
    }
    if let Err(e) = f.cfg.sched.validate() {
        die(e);
    }
    if let Err(e) = f.cfg.lifecycle.validate() {
        die(e);
    }
}

/// --agents and --auto-mode against the rest.
fn validate_auto_mode_flags(f: &Flags) {
    if f.agents.is_some() && !f.auto_mode {
        die("--agents needs --auto-mode");
    }
    if f.auto_mode {
        if f.is_fake || f.model.is_none() {
            die("--auto-mode needs --model GGUF (a real engine)");
        }
        if f.real.kv_unified {
            die("--auto-mode sizes per-sequence streams, not --kv-streams unified");
        }
        if f.agents.is_some() && f.real.pool_pages.is_some() {
            die("--agents and --kv-pages both set the size: drop one");
        }
    }
}

/// `--auto-mode`: sizes context, KV pool and cache types from the model profile and the machine. Returns true when
/// `--print-profile` printed the result and the process is done.
pub(crate) fn apply_auto_mode(f: &mut Flags) -> bool {
    if !f.auto_mode {
        return false;
    }
    let (cfg, real) = (&mut f.cfg, &mut f.real);
    let st = sllm_server::profile::startup().unwrap_or_else(|| die("no model profile"));
    let prof = &st.profile;
    use sllm_server::auto_mode::{Ask, Sized};
    use sllm_server::profile::KvType as Kv;
    let can_read = !prof.recurrent && !prof.hybrid;
    let explicit = Kv::parse(real.kv_type.name()).unwrap_or(Kv::F16);
    let notes = sllm_server::auto_mode::notes_for(real.kv_auto, real.kv_read_auto, real.kv_read_f16, explicit, can_read, real.kv_allow_4bit);
    let four_bit_off = real.kv_auto && !real.kv_allow_4bit;
    let base = sllm_server::profile::PlanArgs {
        n_ctx: real.n_ctx,
        slots: cfg.sched.max_rows(),
        pool_pages: real.pool_pages,
        n_ubatch: real.n_ubatch,
        step_tokens: cfg.sched.step_tokens,
        kv_type: notes[0].kv,
        unified: false,
        max_seq_ids: llama_engine::MAX_SEQ_IDS,
        swa_full: real.swa_full,
        working_set: real.working_set_bytes.or_else(|| sllm_core::caps::Capabilities::probe().ok().map(|c| c.recommended_working_set_bytes)),
        bandwidth_gbs: st.machine.bandwidth_gbs,
    };
    let ask = Ask {
        agents: f.agents,
        ctx: (f.ctx_flag && real.pool_pages.is_none()).then_some(real.n_ctx),
        default_ctx: real.n_ctx,
        top_ctx: if prof.n_ctx_train > 0 { prof.n_ctx_train as usize } else { usize::MAX },
        notes,
        max_streams: llama_engine::MAX_SEQ_IDS,
    };
    let size = Sized::new(prof, &base, &ask)
        .unwrap_or_else(|e| die(format!("--auto-mode: {e}{}", if four_bit_off { ". 4-bit notes are off (--kv-auto-4bit on lets auto-mode try them; their quality cost depends on the model)" } else { "" })));
    if size.choice.is_some() {
        real.n_ctx = size.ctx;
        real.pool_pages = size.kv_pages;
        real.kv_type = KvType::parse(size.notes.kv.name()).unwrap_or(real.kv_type);
        (real.kv_auto, real.kv_read_auto, real.kv_read_f16) = (false, false, size.notes.read_f16);
    }
    cfg.lifecycle.relieve_on_refusal |= !f.relief_flag;
    cfg.lifecycle.ssd_budget_bytes = cfg.lifecycle.ssd_budget_bytes.or(Some(sllm_server::auto_mode::SSD_BUDGET_BYTES));
    let name = prof.name.clone().unwrap_or_else(|| prof.arch.clone());
    let line = size.line(&name, cfg.lifecycle.relieve_on_refusal, cfg.lifecycle.ssd_budget_bytes);
    eprintln!("sllm-server: auto-mode: {line}");
    let mut json = size.to_json(&line, f.agents);
    if four_bit_off {
        json["kv_note"] = serde_json::json!("--kv-type auto: 4-bit notes are off (their quality cost depends on the model); --kv-auto-4bit on allows them");
    }
    if f.print_profile {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({"auto_mode": json})).unwrap());
        return true;
    }
    sllm_server::profile::set_auto_mode(json);
    false
}

/// The helper model, when `--helper-model` was given: checks it and decides where classification runs.
pub(crate) fn configure_helper(f: &mut Flags, model_arch: Option<&str>) {
    if f.helper_on {
        let mut helper = std::mem::replace(&mut f.helper, sllm_server::helper::HelperConfig::new(PathBuf::new()));
        check_helper(&helper);
        // Classify and compress scoring run on the brain when it is Gemma-3 (the 0.6B helper failed the on-track bar: 45%
        // against the brain's 95% on the fixed set) unless --helper-classify helper says otherwise.
        if let (Some(m), false) = (&f.model, f.is_fake) {
            let brain_is_gemma3 = || model_arch == Some("gemma3");
            helper.classify_on = match f.classify_on {
                Some(c) => c,
                None if brain_is_gemma3() => sllm_server::helper::ClassifyOn::Brain,
                None => sllm_server::helper::ClassifyOn::Helper,
            };
            helper.brain_model = Some(m.clone());
        } else if f.classify_on == Some(sllm_server::helper::ClassifyOn::Brain) {
            die("--helper-classify brain needs --model GGUF (a real brain)");
        }
        f.cfg.helper = Some(helper);
    }
}

/// The helper's GGUF exists and its sizes are in range.
fn check_helper(helper: &sllm_server::helper::HelperConfig) {
    if !helper.model.is_file() {
        die(format!("helper GGUF not found: {}", helper.model.display()));
    }
    if helper.n_ctx > sllm_server::helper::MAX_CTX || helper.pool_pages == 0 || helper.slots == 0 || helper.step_tokens == 0 {
        die(format!("--helper-ctx must be at most {} and --helper-kv-pages, --helper-slots, --helper-step at least 1", sllm_server::helper::MAX_CTX));
    }
}

/// Loads the vocabulary shared by /tokenize, the text endpoints and the engine, and fills in the model path and the prompt cache.
pub(crate) fn load_tokenizer(f: &mut Flags) -> Tokenizer {
    let tokenizer = match &f.model {
        Some(p) => {
            if !p.is_file() {
                die(format!("GGUF not found: {}", p.display()));
            }
            // The vocabulary is loaded once here and shared by /tokenize, the text endpoints and the engine.
            let t = Tokenizer::try_gguf(p.clone()).unwrap_or_else(|e| die(format!("cannot read the vocabulary of {e}")));
            if let Some(v) = t.vocab() {
                f.fake.n_vocab = v.n_vocab();
            }
            f.fake.model = p.display().to_string();
            t
        }
        None => Tokenizer::Bytes,
    };
    f.real.model = f.model.clone().unwrap_or_default();
    if f.pcache_on {
        let root = f.pcache_dir.take().unwrap_or_else(|| {
            let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
            home.join("Library/Caches/sllm/prompt-cache")
        });
        f.real.prompt_cache = Some((root, f.pcache_gb << 30));
    }
    tokenizer
}
