//! Budgets: the memory budget carved into weights, compute buffers and KV, and the engine configuration it is planned for.

use super::*;

/// The memory budget of a model, and what it buys.
#[derive(Debug, Clone)]
pub struct Budget {
    pub swa_full: bool,
    pub working_set: u64,
    pub util: f64,
    pub memory_budget: u64,
    pub weights: u64,
    pub compute_bytes: u64,
    pub compute_source: String,
    pub headroom: u64,
    pub kv_budget: u64,
    pub kv_type: KvType,
    /// Sequence ids (unified pool) or KV streams (per-sequence streams: one per sequence id).
    pub n_seq: u64,
    pub n_ubatch: u64,
    /// `Some(cells per stream)` with per-sequence KV streams (the default): the cache is `n_seq` streams of that many cells.
    pub stream_cells: Option<u64>,
    /// Cells of the cache that fit the KV budget (libllama's SWA sizing included).
    pub max_cells: u64,
    /// Cells to allocate.
    pub cells: u64,
    pub cells_reason: String,
    pub bandwidth_gbs: Option<f64>,
}

pub struct BudgetInputs {
    pub swa_full: bool,
    pub working_set: u64,
    /// Share of the working set the process may use (the engine's 0.9).
    pub util: f64,
    pub headroom: u64,
    pub kv_type: KvType,
    pub n_seq: u64,
    pub n_ubatch: u64,
    pub compute_bytes: u64,
    pub compute_source: String,
    /// Cells the configuration wants (the default job, or `--kv-pages` x 256) and the hard cap.
    pub want_cells: u64,
    /// What asked for `want_cells`, for the reason text (`Budget::cells_reason`).
    pub asked_by: String,
    pub min_cells: u64,
    pub max_total_cells: u64,
    /// Per-sequence streams: cells of one stream (the context of a sequence). `None` = one unified pool.
    pub stream_cells: Option<u64>,
    /// Most sequence ids / streams.
    pub max_seq_ids: u64,
    /// Share of the KV budget the default allocation may take.
    pub alloc_share: f64,
    pub bandwidth_gbs: Option<f64>,
}

/// Bytes libllama allocates for a KV cache of `cells` cells: one unified pool, or `cells / stream_cells` streams of
/// `stream_cells` cells each (the SWA ring of a stream is `n_swa + n_ubatch` cells, without `swa_full`).
pub fn kv_cache_bytes(p: &Profile, ty: KvType, cells: u64, n_seq: u64, n_ubatch: u64, swa_full: bool, stream_cells: Option<u64>) -> u64 {
    match stream_cells {
        Some(sc) => (cells / sc.max(1)) * p.kv_alloc_bytes(ty, sc, 1, n_ubatch, swa_full),
        None => p.kv_alloc_bytes(ty, cells, n_seq, n_ubatch, swa_full),
    }
}

/// The cells a budget allocates: what fits the whole KV budget, what is allocated, the sequence ids, and why.
struct CellPlan {
    max_cells: u64,
    cells: u64,
    n_seq: u64,
    cells_reason: String,
}

/// The reason for a cell count the KV budget limited: the allocation's share of it when that is below the whole.
fn share_reason(i: &BudgetInputs, why: &str) -> String {
    if i.alloc_share < 1.0 { format!("limited to {:.0}% of the KV budget", i.alloc_share * 100.0) } else { why.to_string() }
}

/// Per-sequence streams: how many streams of `sc` cells fit, one stream per sequence id.
fn stream_plan(p: &Profile, i: &BudgetInputs, kv_budget: u64, alloc_budget: u64, sc: u64) -> CellPlan {
    let per_stream = p.kv_alloc_bytes(i.kv_type, sc, 1, i.n_ubatch, i.swa_full).max(1);
    let max_streams = (kv_budget / per_stream).min(i.max_seq_ids);
    // at least two streams whatever the share says, but never more than the whole KV budget holds: the allocation
    // never goes past `kv_budget` (0 streams when not even one fits; the engine refuses that context)
    let by_budget = (alloc_budget / per_stream).max(2).min(max_streams);
    let want = (i.want_cells / sc).max(1);
    let cap = (i.max_total_cells / sc).min(i.max_seq_ids).max(1);
    let streams = want.min(by_budget).min(cap);
    let cells_reason = if max_streams == 0 {
        format!("not enough memory for one stream of {sc} cells")
    } else if streams == cap && want > streams {
        "capped at the engine maximum".to_string()
    } else if want > by_budget {
        share_reason(i, "limited to what the KV budget holds")
    } else {
        i.asked_by.clone()
    };
    CellPlan { max_cells: max_streams * sc, cells: streams * sc, n_seq: streams, cells_reason }
}

/// Largest padded cell count of one unified pool whose allocation fits a byte budget (the allocation is monotone in the cells).
fn max_unified_cells(p: &Profile, i: &BudgetInputs, budget: u64) -> u64 {
    let fits = |cells: u64| p.kv_alloc_bytes(i.kv_type, cells, i.n_seq, i.n_ubatch, i.swa_full) <= budget;
    let (mut lo, mut hi) = (0u64, 1u64 << 24);
    while lo < hi {
        let mid = ((lo + hi).div_ceil(2).div_ceil(CELL_PAD) * CELL_PAD).max(lo + CELL_PAD).min(hi);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - CELL_PAD;
        }
    }
    lo
}

/// One unified pool of cells.
fn unified_plan(p: &Profile, i: &BudgetInputs, kv_budget: u64, alloc_budget: u64) -> CellPlan {
    let max_cells = max_unified_cells(p, i, kv_budget);
    // The engine's rule: what is wanted, but only a share of what fits (`alloc_share`: the rest stays for compute
    // buffers that grow with the step, the helper lane, and the OS), at least two contexts whatever the share says,
    // not past the cap, and never past what the whole KV budget holds (`max_cells`: below one context the engine
    // refuses the configuration).
    let by_budget = max_unified_cells(p, i, alloc_budget).max(i.min_cells * 2);
    let cells = (i.want_cells.min(by_budget).min(i.max_total_cells) / CELL_PAD * CELL_PAD).max(i.min_cells).min(max_cells);
    let cells_reason = if cells < i.min_cells {
        format!("not enough memory for one context of {} cells", i.min_cells)
    } else if cells == i.max_total_cells / CELL_PAD * CELL_PAD && i.want_cells > cells {
        "capped at the engine maximum".to_string()
    } else if i.want_cells > by_budget {
        share_reason(i, "limited to what the KV budget holds")
    } else if i.want_cells > cells {
        "limited to what the KV budget holds".to_string()
    } else {
        i.asked_by.clone()
    };
    CellPlan { max_cells, cells, n_seq: i.n_seq, cells_reason }
}

impl Budget {
    pub fn plan(p: &Profile, i: &BudgetInputs) -> Budget {
        let memory_budget = (i.working_set as f64 * i.util) as u64;
        let weights = p.weights_total;
        let kv_budget = memory_budget.saturating_sub(weights + i.compute_bytes + i.headroom);
        let alloc_budget = (kv_budget as f64 * i.alloc_share) as u64;
        let CellPlan { max_cells, cells, n_seq, cells_reason } = match i.stream_cells {
            Some(sc) => stream_plan(p, i, kv_budget, alloc_budget, sc),
            None => unified_plan(p, i, kv_budget, alloc_budget),
        };
        Budget {
            swa_full: i.swa_full,
            working_set: i.working_set,
            util: i.util,
            memory_budget,
            weights,
            compute_bytes: i.compute_bytes,
            compute_source: i.compute_source.clone(),
            headroom: i.headroom,
            kv_budget,
            kv_type: i.kv_type,
            n_seq,
            n_ubatch: i.n_ubatch,
            stream_cells: i.stream_cells,
            max_cells,
            cells,
            cells_reason,
            bandwidth_gbs: i.bandwidth_gbs,
        }
    }

    /// Bytes the KV cache of `cells` cells takes with this budget's geometry.
    pub fn kv_bytes(&self, p: &Profile, cells: u64) -> u64 {
        kv_cache_bytes(p, self.kv_type, cells, self.n_seq, self.n_ubatch, self.swa_full, self.stream_cells)
    }

    /// The refusal of a configuration whose cache cannot be allocated: the budget holds less than one context of `capacity`
    /// cells, or (an accounting fault) the planned cells take more than the KV budget. Checked before libllama allocates.
    pub fn check_fits(&self, p: &Profile, capacity: u64) -> Result<(), String> {
        let one = self.kv_bytes(p, capacity);
        let mib = |b: u64| format!("{:.0} MiB", b as f64 / 1048576.0);
        if self.cells < capacity {
            return Err(format!(
                "the memory budget cannot hold one context of {capacity} tokens: {} of KV cache would take {} and {} is left (memory budget {}, weights {}, compute buffer {}, headroom {}); \
                 use a shorter --ctx, a smaller --kv-type or a smaller model",
                self.kv_type.name(),
                mib(one),
                mib(self.kv_budget),
                mib(self.memory_budget),
                mib(self.weights),
                mib(self.compute_bytes),
                mib(self.headroom),
            ));
        }
        let planned = self.kv_bytes(p, self.cells);
        if planned > self.kv_budget {
            return Err(format!("the planned KV cache ({}) is over the KV budget ({}): a sizing fault, not allocated", mib(planned), mib(self.kv_budget)));
        }
        Ok(())
    }

    /// KV bytes of one agent holding a context of `ctx` tokens (a stream of that size with per-sequence streams).
    fn agent_bytes(&self, p: &Profile, ctx: u64) -> u64 {
        p.kv_alloc_bytes(self.kv_type, pad(ctx.max(1)), 1, self.n_ubatch, self.swa_full).max(1)
    }

    /// Agents of `ctx` tokens each that fit the KV budget, before the sequence-id limit.
    pub fn agents_at(&self, p: &Profile, ctx: u64) -> u64 {
        match self.stream_cells {
            Some(_) => self.kv_budget / self.agent_bytes(p, ctx),
            None => self.max_cells / pad(ctx.max(1)),
        }
    }

    /// Longest context (a multiple of 256) `agents` agents can each hold.
    pub fn ctx_for(&self, p: &Profile, agents: u64) -> u64 {
        let agents = agents.max(1);
        match self.stream_cells {
            Some(_) => {
                let (mut lo, mut hi) = (0u64, p.n_ctx_train.max(CELL_PAD) / CELL_PAD);
                while lo < hi {
                    let mid = (lo + hi).div_ceil(2);
                    if agents * self.agent_bytes(p, mid * CELL_PAD) <= self.kv_budget {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }
                lo * CELL_PAD
            }
            None => self.max_cells / agents / CELL_PAD * CELL_PAD,
        }
    }

    /// Decode roofline at `batch` rows with `ctx` tokens each: `(step ms, tokens/s)`.
    pub fn roofline(&self, p: &Profile, batch: u64, ctx: u64) -> Option<(f64, f64)> {
        let bw = self.bandwidth_gbs? * 1e9;
        let bytes = p.decode_step_bytes(self.kv_type, batch, ctx) as f64;
        let s = bytes / bw;
        Some((s * 1e3, batch as f64 / s))
    }

    pub fn to_json(&self, p: &Profile) -> Value {
        let ctxs = [1024u64, 2048, 4096, 8192, 16384, 32768, p.n_ctx_train];
        let table: Vec<Value> = ctxs
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| json!({"ctx": c, "agents": self.agents_at(p, c), "agents_capped_by_seq_ids": self.agents_at(p, c).min(256)}))
            .collect();
        let by_agents: Vec<Value> = [1u64, 4, 8, 16, 32, 64].iter().map(|&a| json!({"agents": a, "ctx": self.ctx_for(p, a).min(p.n_ctx_train.max(1))})).collect();
        let roof = |b: u64, c: u64| match self.roofline(p, b, c) {
            Some((ms, tps)) => json!({"batch": b, "ctx": c, "bytes_per_step": p.decode_step_bytes(self.kv_type, b, c), "step_ms": ms, "tokens_per_s": tps}),
            None => json!({"batch": b, "ctx": c, "bytes_per_step": p.decode_step_bytes(self.kv_type, b, c), "step_ms": null, "tokens_per_s": null}),
        };
        json!({
            "working_set_bytes": self.working_set, "utilisation": self.util, "memory_budget_bytes": self.memory_budget,
            "weights_bytes": self.weights, "compute_buffer_bytes": self.compute_bytes, "compute_buffer_source": self.compute_source,
            "headroom_bytes": self.headroom, "kv_budget_bytes": self.kv_budget, "kv_type": self.kv_type.name(),
            "kv_layout": if self.stream_cells.is_some() { "per-sequence streams" } else { "unified pool" },
            "swa_full": self.swa_full,
            "kv_cells": {"max_fitting": self.max_cells, "allocated": self.cells, "stream_cells": self.stream_cells, "reason": self.cells_reason},
            "kv_alloc_bytes": self.kv_bytes(p, self.cells),
            "n_seq_max": self.n_seq,
            "max_agents_by_context": table, "max_context_by_agents": by_agents,
            "bandwidth_gbs": self.bandwidth_gbs,
            "roofline": [roof(1, 512), roof(8, 512), roof(32, 512)],
        })
    }
}

/// Upper bound on the cells of per-sequence streams. The memory budget is the real limit; this one only keeps a typing error
/// from asking libllama for a huge cache (2M cells ran 256 streams of 2,816 cells).
pub const MAX_TOTAL_CELLS: u64 = 2_097_152;
/// Upper bound on the cells of the unified cache: one pool, and a search for free cells walks it.
pub const MAX_UNIFIED_CELLS: u64 = 262_144;
/// The default job: with no `--kv-pages` and no `--agents` the cache is sized for this many agents. Metal wires a whole
/// buffer at its first GPU use, so a cache sized as a share of the budget wired about 8 GB for one plain chat. Not a
/// final value: it is the one place to change it.
pub const DEFAULT_AGENTS: u64 = 4;
/// Requests that run at once for `agents` agents: one more, so a chat beside the agents does not wait for an agent to end
/// (`--agents N` under `--auto-mode` sizes the same way).
pub const fn requests_for(agents: u64) -> u64 {
    agents + 1
}
/// Memory kept free beside weights, compute buffers and KV (the engine's former fixed reserve).
pub const HEADROOM_BYTES: u64 = 1 << 30;

/// The engine configuration a budget is planned for.
#[derive(Debug, Clone)]
pub struct PlanArgs {
    /// Context of one sequence (`--ctx`).
    pub n_ctx: usize,
    /// Lane slots (the scheduler's `max_rows`).
    pub slots: usize,
    /// `--kv-pages`: pages of 256 cells.
    pub pool_pages: Option<usize>,
    pub n_ubatch: usize,
    pub step_tokens: usize,
    pub kv_type: KvType,
    /// `--kv-streams unified`: one shared pool; otherwise one stream of `n_ctx` cells per sequence id.
    pub unified: bool,
    /// Most sequence ids the engine can hold.
    pub max_seq_ids: usize,
    /// libllama's `swa_full` (see `--swa-full`).
    pub swa_full: bool,
    pub working_set: Option<u64>,
    pub bandwidth_gbs: Option<f64>,
}

/// Cells the configuration asks for, and what asked: `--kv-pages` x 256, else the default job of [`DEFAULT_AGENTS`] agents
/// (and a chat) at `capacity` tokens each.
pub(super) fn wanted(pool_pages: Option<usize>, capacity: u64) -> (u64, String) {
    match pool_pages {
        Some(pg) => (pg as u64 * CELL_PAD, "--kv-pages".to_string()),
        None => {
            let n = requests_for(DEFAULT_AGENTS);
            (capacity * n, format!("the default job ({n} requests of --ctx tokens: {DEFAULT_AGENTS} agents and a chat; --kv-pages or --auto-mode --agents N change it)"))
        }
    }
}

/// The compute buffer of the configuration at `probe_cells` cells and `probe_seq` sequences, from libllama (a `no_alloc`
/// context), and where the figure came from; 1 GiB when the probe fails.
fn compute_buffer(p: &Profile, a: &PlanArgs, probe_cells: u64, probe_seq: u64, n_batch: usize) -> (u64, String) {
    match probe_memory(
        &p.path,
        &ProbeParams {
            cells: probe_cells as u32,
            n_seq: probe_seq as u32,
            n_ubatch: a.n_ubatch as u32,
            n_batch: n_batch as u32,
            kv_type: a.kv_type,
            kv_unified: a.unified,
            swa_full: a.swa_full,
            ..Default::default()
        },
    ) {
        Ok(pr) if pr.compute_bytes > 0 => (pr.compute_bytes, format!("libllama no_alloc context at {probe_cells} cells, {probe_seq} sequences, ubatch {}, batch {n_batch}", a.n_ubatch)),
        Ok(_) => (HEADROOM_BYTES, "default 1 GiB (the probe printed no compute buffer)".to_string()),
        Err(e) => (HEADROOM_BYTES, format!("default 1 GiB ({e})")),
    }
}

/// Sequence ids, batch size and the memory budget of an engine configuration: the one place the LlamaEngine sizes its cache.
/// Per-sequence streams (the default): how many streams of `n_ctx` cells fit, one per sequence id, so `streams x ctx` cells.
/// Unified pool: the same cells in one pool, 2x slots sequence ids. Without `--kv-pages` the cells are those of the default
/// job ([`DEFAULT_AGENTS`]), up to what fits; a burst beyond them waits for a stream (per-sequence) or for cells (unified).
/// The compute buffer is read from libllama itself (a `no_alloc` context at the planned shape), 1 GiB when that fails.
pub fn plan_engine(p: &Profile, a: &PlanArgs) -> (Budget, usize, usize) {
    let capacity = pad(a.n_ctx.max(1) as u64);
    let n_seq_unified = (a.slots * 2).clamp(64, 256);
    let n_batch = (a.step_tokens + a.slots).max(a.n_ubatch).max(1);
    let (want, asked_by) = wanted(a.pool_pages, capacity);
    let stream_cells = (!a.unified).then_some(capacity);
    // Two passes: the compute buffer depends on the cells (the attention mask), the cells on the compute buffer.
    let max_total_cells = if a.unified { MAX_UNIFIED_CELLS } else { MAX_TOTAL_CELLS };
    let mut probe_cells = want.min(max_total_cells).max(capacity);
    let mut probe_seq = n_seq_unified as u64;
    if let Some(sc) = stream_cells {
        probe_seq = (probe_cells / sc).clamp(1, a.max_seq_ids as u64);
        probe_cells = probe_seq * sc;
    }
    let mut budget = None;
    for _ in 0..2 {
        let (compute_bytes, compute_source) = compute_buffer(p, a, probe_cells, probe_seq, n_batch);
        let b = Budget::plan(
            p,
            &BudgetInputs {
                swa_full: a.swa_full,
                // no device figure: a 2 GiB cache budget, as before
                working_set: a.working_set.unwrap_or(p.weights_total + compute_bytes + HEADROOM_BYTES + (2 << 30)),
                util: if a.working_set.is_some() { 0.9 } else { 1.0 },
                headroom: HEADROOM_BYTES,
                kv_type: a.kv_type,
                n_seq: n_seq_unified as u64,
                n_ubatch: a.n_ubatch as u64,
                compute_bytes,
                compute_source,
                want_cells: want,
                asked_by: asked_by.clone(),
                min_cells: capacity,
                max_total_cells,
                stream_cells,
                max_seq_ids: a.max_seq_ids as u64,
                // an explicit --kv-pages is honoured up to what fits; otherwise the default job up to a quarter of what fits
                // (the rest for compute buffers, the helper lane and the OS)
                alloc_share: if a.working_set.is_some() && a.pool_pages.is_none() { 0.25 } else { 1.0 },
                bandwidth_gbs: a.bandwidth_gbs,
            },
        );
        // a budget that holds less than one context is final: the engine refuses it (`Budget::check_fits`)
        let done = (b.cells == probe_cells && b.n_seq == probe_seq) || b.cells < capacity;
        probe_cells = b.cells.max(capacity);
        probe_seq = b.n_seq.max(1);
        budget = Some(b);
        if done {
            break;
        }
    }
    let b = budget.expect("two passes");
    let n_seq = b.n_seq as usize;
    (b, n_seq, n_batch)
}
