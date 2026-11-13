//! The choices `LlamaEngine::new` makes: the KV cache type in the memory budget (`--kv-type auto`), and exact decode,
//! speculation and prefix sharing for the loaded model.

use super::*;
use crate::profile::Budget;
use load::Loaded;

/// The cache the engine settled on: `--kv-type`, or what `--kv-type auto` took, and why (`/caps` `kv.type_choice`).
pub(super) struct KvChoice {
    pub(super) kv_type: KvType,
    pub(super) budget: Budget,
    /// Prompts are read in f16 readers (`--kv-read f16`, or auto's 4-bit notes read in f16).
    pub(super) read_f16: bool,
    pub(super) why: String,
}

/// The memory plans the cache type is chosen from: the model's profile in the budget, for a cache type and a working set.
pub(super) struct KvPlanner<'a> {
    pub(super) cfg: &'a EngineConfig,
    pub(super) prof: &'a crate::profile::Profile,
    pub(super) capacity: usize,
    pub(super) slots: usize,
    /// The recommended working set the budget is carved from.
    pub(super) ws: Option<u64>,
    pub(super) bandwidth: Option<f64>,
}

/// The smaller caches `--kv-type auto` may take when f16 cannot hold the cells asked for.
struct ShortOptions {
    b8: Budget,
    /// q4_0 read in f16 (the readers' memory comes out of the budget first).
    c4r: Option<Budget>,
    /// Plain q4_0.
    c4: Option<Budget>,
}

impl KvPlanner<'_> {
    /// The plan of a `kv` cache in the working set `ws`.
    fn plan(&self, kv: KvType, ws: Option<u64>) -> Budget {
        let cfg = self.cfg;
        let kv = crate::profile::KvType::parse(kv.name()).unwrap_or(crate::profile::KvType::F16);
        crate::profile::plan_engine(
            self.prof,
            &crate::profile::PlanArgs { n_ctx: cfg.n_ctx, slots: self.slots, pool_pages: cfg.pool_pages, n_ubatch: cfg.n_ubatch, step_tokens: cfg.step_tokens, kv_type: kv, unified: cfg.kv_unified, max_seq_ids: MAX_SEQ_IDS, swa_full: cfg.swa_full, working_set: ws, bandwidth_gbs: self.bandwidth },
        )
        .0
    }

    /// The working set without the readers' f16 caches and compute buffers as they are allocated (reader_reserve_bytes),
    /// asked only when a plan needs them.
    fn ws_read(&self) -> Option<u64> {
        let cfg = self.cfg;
        self.ws.map(|w| w.saturating_sub(if cfg.kv_unified { 0 } else { reader_reserve_bytes(self.prof, self.capacity, cfg.n_ubatch.max(1), cfg.swa_full) }))
    }

    /// --kv-type auto: f16 while the budget holds the cache asked for (--kv-pages), else q8_0 when that holds more. 4-bit
    /// notes (read in f16, or plain q4_0 when q8_0 is still short too) only with --kv-auto-4bit on: they cost quality
    /// in a way that depends on the model, so the engine does not choose them alone. Without --kv-pages the cache
    /// is the default job, grows into use on demand (idle agents go to the SSD), and is f16.
    pub(super) fn choose(&self, can_read: bool) -> KvChoice {
        let cfg = self.cfg;
        let budget = self.plan(cfg.kv_type, if cfg.kv_read_f16 { self.ws_read() } else { self.ws });
        let why = if cfg.kv_auto { "auto: f16 (no --kv-pages: the cache is the default job, sized from the budget)".to_string() } else { "--kv-type".to_string() };
        let choice = KvChoice { kv_type: cfg.kv_type, budget, read_f16: cfg.kv_read_f16, why };
        match (cfg.kv_auto, cfg.kv_type, cfg.pool_pages) {
            (true, KvType::F16, Some(pg)) => self.choose_for_pages(choice, (pg * CELL_PAD) as u64, can_read),
            _ => choice,
        }
    }

    /// --kv-type auto with --kv-pages: f16 while the budget holds the `want_cells` asked for.
    fn choose_for_pages(&self, mut c: KvChoice, want_cells: u64, can_read: bool) -> KvChoice {
        c.why = format!("auto: f16, the memory budget holds the {want_cells} cells asked for");
        if c.budget.cells >= want_cells {
            return c;
        }
        let cfg = self.cfg;
        let b8 = self.plan(KvType::Q8_0, self.ws);
        let c4r = Some(self.plan(KvType::Q4_0, self.ws_read())).filter(|b| cfg.kv_read_auto && can_read && b.cells > c.budget.cells && b.cells >= want_cells.min(b8.cells));
        // plain q4_0 only when q8_0 is still short too (perplexity +0.85% on Gemma-3 4B, unchanged on Qwen3-4B, 20 to 83 on Qwen3-0.6B)
        let c4 = if b8.cells < want_cells { Some(self.plan(KvType::Q4_0, self.ws)).filter(|c4| c4.cells > b8.cells) } else { None };
        ShortOptions { b8, c4r, c4 }.pick(c, want_cells, cfg.kv_allow_4bit)
    }
}

impl ShortOptions {
    /// What auto did not take, for /caps.
    fn held_back(&self, allow_4bit: bool) -> String {
        match (&self.c4r, &self.c4, allow_4bit) {
            (Some(b), _, false) | (None, Some(b), false) => format!("; 4-bit notes would hold {} cells but are off (they cost quality that depends on the model): --kv-auto-4bit on allows them", b.cells),
            _ => String::new(),
        }
    }

    /// Memory is short for the f16 cache `c`: the option that holds more, 4-bit notes only when `allow_4bit`.
    fn pick(self, mut c: KvChoice, want_cells: u64, allow_4bit: bool) -> KvChoice {
        let (b4r, b4) = if allow_4bit { (self.c4r.clone(), self.c4.clone()) } else { (None, None) };
        let held_back = self.held_back(allow_4bit);
        let f16_cells = c.budget.cells;
        if let Some(b4r) = b4r {
            c.why = format!("auto: q4_0 read in f16, memory is short (f16 holds {}, 4-bit with the readers {} of the {want_cells} cells asked for; --kv-auto-4bit on)", f16_cells, b4r.cells);
            eprintln!("sllm-server: KV cache {}", c.why);
            (c.kv_type, c.budget, c.read_f16) = (KvType::Q4_0, b4r, true);
        } else if let Some(b4) = b4 {
            c.why = format!("auto: q4_0, memory is short (f16 holds {}, q8_0 {}, q4_0 {} of the {want_cells} cells asked for; --kv-auto-4bit on)", f16_cells, self.b8.cells, b4.cells);
            eprintln!("sllm-server: KV cache {}", c.why);
            (c.kv_type, c.budget) = (KvType::Q4_0, b4);
        } else if self.b8.cells > f16_cells {
            c.why = format!("auto: q8_0, memory is short (f16 holds {} of the {want_cells} cells asked for, q8_0 {}){held_back}", f16_cells, self.b8.cells);
            eprintln!("sllm-server: KV cache {}", c.why);
            (c.kv_type, c.budget) = (KvType::Q8_0, self.b8);
        } else {
            c.why = format!("auto: f16, memory is short (it holds {} of the {want_cells} cells asked for) but q8_0 holds no more{held_back}", f16_cells);
            if !held_back.is_empty() {
                eprintln!("sllm-server: KV cache {}", c.why);
            }
        }
        c
    }
}

/// The engine's choices for the loaded model: the row-exact switch and speculation (spec.rs), and prefix sharing.
pub(super) struct Choices {
    pub(super) spec: SpecConfig,
    /// Why speculation is not available here ("" when it is).
    pub(super) spec_why: String,
    pub(super) row_exact: bool,
    pub(super) moe: bool,
    pub(super) share: bool,
    pub(super) holders_on: bool,
}

impl Choices {
    /// The row-exact switch and speculation: speculation needs the switch, per-sequence streams, a cache that can drop a
    /// tail (no recurrent state) and dense layers (the MoE mat-mul has no row-exact path). An error when what was asked for
    /// (`--row-exact on`, `--spec on`) cannot be had.
    pub(super) fn make(cfg: &EngineConfig, ld: &Loaded) -> Result<Choices, String> {
        let can_share = ld.can_share();
        let (moe, fast_exact) = weights_kind(cfg, &ld.arch);
        let mut ch = Choices { spec: cfg.spec.clone(), spec_why: String::new(), row_exact: false, moe, share: cfg.prefix_sharing && can_share, holders_on: cfg.prefix_sharing && !can_share };
        let want_exact = want_row_exact(cfg.row_exact, moe, ld.unified, fast_exact);
        // --spec auto follows the exact rule: where exact decode is off (weights not mostly Q4_0, MoE, a unified pool) it
        // does not turn the switch on (the switch alone would cost ~20% there), so speculation stays off
        if ch.spec.mode == SpecMode::Auto && !want_exact && cfg.row_exact != RowExact::On {
            ch.spec_why = "exact decode is off for this model (--row-exact auto)".into();
        }
        if ch.wants_switch(want_exact) {
            ch.switch_row_exact(cfg.row_exact)?;
        }
        if ch.spec.mode != SpecMode::Off {
            ch.settle_spec(ld, want_exact)?;
        }
        Ok(ch)
    }

    /// Exact decode is wanted, or speculation (which needs it) is on or may be.
    fn wants_switch(&self, want_exact: bool) -> bool {
        want_exact || self.spec.mode == SpecMode::On || (self.spec.mode == SpecMode::Auto && self.spec_why.is_empty())
    }

    /// Turn the row-exact switch on; an error when `--row-exact on` asked for it and the linked ggml has none.
    fn switch_row_exact(&mut self, requested: RowExact) -> Result<(), String> {
        if ffi::set_row_exact(true) {
            self.row_exact = true;
        } else if requested == RowExact::On {
            return Err("--row-exact: the linked ggml has no row-exact switch (build with SLLM_GGML_METAL=rowexact)".into());
        } else if self.spec.mode != SpecMode::Off {
            self.spec_why = "the linked ggml has no row-exact switch (build with SLLM_GGML_METAL=rowexact)".into();
        }
        Ok(())
    }

    /// Speculation is asked for: off (with the switch, unless exact decode wants it) where it cannot run; an error when
    /// `--spec on` asked for it.
    fn settle_spec(&mut self, ld: &Loaded, want_exact: bool) -> Result<(), String> {
        if self.spec_why.is_empty() {
            self.spec_why = spec_blocker(ld.unified, ld.recurrent || ld.hybrid, self.moe).into();
        }
        if self.spec_why.is_empty() {
            return Ok(());
        }
        let msg = format!("--spec {}: not available here: {}", self.spec.mode.name(), self.spec_why);
        if self.spec.mode == SpecMode::On {
            return Err(msg);
        }
        eprintln!("sllm-server: {msg}; speculation off");
        self.spec.mode = SpecMode::Off;
        if !want_exact && self.row_exact {
            ffi::set_row_exact(false);
            self.row_exact = false;
        }
        Ok(())
    }
}

/// `--row-exact auto`: only where the guarantee holds (mixture-of-experts mat-muls have no row-exact path; a unified pool's
/// cell layout changes the attention blocks) and where it costs no speed (`fast_exact`).
fn want_row_exact(requested: RowExact, moe: bool, unified: bool, fast_exact: bool) -> bool {
    match requested {
        RowExact::Off => false,
        RowExact::Auto => !moe && !unified && fast_exact,
        RowExact::On => true,
    }
}

/// Why speculation cannot run on this model and cache ("" when it can).
fn spec_blocker(unified: bool, recurrent_or_hybrid: bool, moe: bool) -> &'static str {
    if unified {
        "unified KV cache (use --kv-streams per-seq)"
    } else if recurrent_or_hybrid {
        "recurrent state cannot drop a draft's cells"
    } else if moe {
        "mixture-of-experts mat-muls have no row-exact path"
    } else {
        ""
    }
}

/// From the GGUF: the model is a mixture of experts, and most of its block weights' bytes are Q4_0.
fn weights_kind(cfg: &EngineConfig, arch: &str) -> (bool, bool) {
    let gguf = sllm_core::gguf::GgufFile::open(&cfg.model).ok();
    let moe = gguf.as_ref().and_then(|g| g.get_u32(&format!("{arch}.expert_count"))).is_some_and(|n| n > 1);
    // the one-row exact kernel (kernel_sllm_mme) covers Q4_0; other weight types decode one row on the few-row MMA
    // kernel, ~20% slower (Mistral Q4_K_M chat 0.77x), so auto wants most of the block weights' bytes in Q4_0
    let fast_exact = gguf.as_ref().is_some_and(|g| {
        let blk = g.tensors().iter().filter(|t| t.name.starts_with("blk.") && t.name.ends_with(".weight") && t.ne[1] > 1);
        let (q4, all) = blk.fold((0u64, 0u64), |(q, a), t| (q + if t.type_name == "q4_0" { t.nbytes } else { 0 }, a + t.nbytes));
        all > 0 && q4 * 4 >= all * 3
    });
    (moe, fast_exact)
}
