//! The helper's executor over libllama: one model, one `llama_context` with a unified KV cache, and sequences in it.
//!
//! This is the generic replacement of the native batched executor the helper used to run on. The helper loop
//! ([`super::run`]) sees the same small interface: add a sequence, fork one (the fork shares every cell of the source:
//! `llama_memory_seq_cp` on a unified cache adds a sequence id to the cells, nothing is copied), truncate, remove, and run ONE
//! batched step over many sequences' prefill chunks and decode tokens (one `llama_decode`). Any GGUF libllama loads works.
//!
//! The cache is `pool_pages` pages of 256 cells in total (`--helper-kv-pages`). The helper loop does its own admission over
//! those pages (each call reserves its worst case), so the cache is never asked for more than it has; [`Exec::used_pages`]
//! is the figure `/metrics` shows.
//!
//! One thread owns an [`Exec`] and makes every libllama call on it (the same rule as the brain's executor).

use crate::engine::PAGE_CELLS;
use crate::llama_engine::{argmax, ffi, KvType};
use llama_ref::{sys, Batch};
use std::collections::HashMap;
use std::ffi::{c_char, CString};
use std::path::PathBuf;

/// A sequence of the executor (its libllama sequence id).
pub type SeqId = i32;

/// Which rows of an item produce outputs (logits, and a token).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputPolicy {
    /// No output row (an intermediate prefill chunk).
    None,
    /// The item's last row (a decode token, or the final prefill chunk).
    Last,
    /// Every row of the item.
    All,
}

impl OutputPolicy {
    /// Whether row `j` of an item of `n` rows is an output row.
    fn outputs(self, j: usize, n: usize) -> bool {
        match self {
            OutputPolicy::None => false,
            OutputPolicy::Last => j + 1 == n,
            OutputPolicy::All => true,
        }
    }
}

/// How the next token is chosen from the logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampler {
    /// The argmax of each output row (the lowest id on a tie); no logits come back.
    Greedy,
    /// The argmax and the full f32 logits `[n_vocab]` of each output row.
    HostLogits,
}

/// One sequence's work in a step.
#[derive(Debug, Clone, Copy)]
pub struct StepItem<'t> {
    pub agent: SeqId,
    /// One token (decode) or one prefill chunk.
    pub tokens: &'t [i32],
    pub output: OutputPolicy,
    pub sampler: Sampler,
}

impl<'t> StepItem<'t> {
    pub fn decode(agent: SeqId, token: &'t [i32], sampler: Sampler) -> StepItem<'t> {
        StepItem { agent, tokens: token, output: OutputPolicy::Last, sampler }
    }
    pub fn prefill(agent: SeqId, tokens: &'t [i32], output: OutputPolicy, sampler: Sampler) -> StepItem<'t> {
        StepItem { agent, tokens, output, sampler }
    }
}

/// What a step returns for one item.
#[derive(Debug, Clone, Default)]
pub struct ItemOut {
    /// One token per output row: the argmax of the row's logits.
    pub tokens: Vec<i32>,
    /// Logits of the output rows, `[rows, n_vocab]` row-major (host-logits items only).
    pub logits: Option<Vec<f32>>,
}

impl ItemOut {
    /// The token of the last output row.
    pub fn token(&self) -> Option<i32> {
        self.tokens.last().copied()
    }
}

/// How to build an executor.
#[derive(Debug, Clone)]
pub struct ExecOpts {
    pub model: PathBuf,
    /// Cells of the unified cache (`pool_pages` x 256).
    pub cells: usize,
    /// Sequence ids (the most sequences alive at once).
    pub max_seqs: usize,
    /// The most tokens one step may carry.
    pub max_tokens: usize,
    /// The most output rows one step may carry.
    pub max_outputs: usize,
    pub kv_type: KvType,
    pub cpu_threads: usize,
}

struct SeqState {
    /// Tokens in the cache.
    len: usize,
    /// Most tokens the sequence may hold.
    capacity: usize,
    /// Cells shared with the sequence it was forked from (counted once in [`Exec::used_pages`]).
    base: usize,
}

pub struct Exec {
    model: *mut sys::llama_model,
    ctx: *mut sys::llama_context,
    batch: Batch,
    cap: usize,
    n_vocab: usize,
    seqs: HashMap<SeqId, SeqState>,
    free: Vec<SeqId>,
    /// Facts about the model for the helper's `info`.
    pub facts: ModelFacts,
}

/// What `/v1/helper` shows about the loaded model.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelFacts {
    pub desc: String,
    pub n_layer: i32,
    pub n_embd: i32,
    pub n_vocab: usize,
    pub n_ctx_train: i32,
    pub weights_bytes: u64,
    pub cells: usize,
    pub kv_type: &'static str,
}

// SAFETY: the executor is built on the helper thread and used only from it; the raw pointers are owned by it.
unsafe impl Send for Exec {}

impl Exec {
    /// Load the model with every layer on the GPU and create the context.
    pub fn load(o: &ExecOpts) -> Result<Exec, String> {
        let path = CString::new(o.model.to_string_lossy().as_bytes()).map_err(|_| "model path contains NUL".to_string())?;
        let n_batch = o.max_tokens.max(1);
        let n_ubatch = n_batch.min(512);
        // SAFETY: plain libllama calls; every pointer is checked before use and freed in Drop (or on the error paths).
        unsafe {
            let mut mp = sys::llama_model_default_params();
            mp.n_gpu_layers = 999;
            let model = sys::llama_model_load_from_file(path.as_ptr(), mp);
            if model.is_null() {
                return Err(format!("llama_model_load_from_file failed for {}", o.model.display()));
            }
            let mut cp = sys::llama_context_default_params();
            cp.n_ctx = o.cells as u32;
            cp.n_seq_max = o.max_seqs.clamp(1, 256) as u32;
            cp.n_batch = n_batch as u32;
            cp.n_ubatch = n_ubatch as u32;
            cp.n_outputs_max = o.max_outputs.clamp(1, n_ubatch) as u32;
            cp.flash_attn_type = sys::LLAMA_FLASH_ATTN_TYPE_ENABLED;
            cp.kv_unified = true;
            cp.type_k = o.kv_type.ggml();
            cp.type_v = o.kv_type.ggml();
            if o.cpu_threads > 0 {
                cp.n_threads = o.cpu_threads as i32;
                cp.n_threads_batch = o.cpu_threads as i32;
            }
            let ctx = sys::llama_init_from_model(model, cp);
            if ctx.is_null() {
                sys::llama_model_free(model);
                return Err(format!("llama_init_from_model failed ({} cells, {} sequences)", o.cells, cp.n_seq_max));
            }
            let n_vocab = sys::llama_vocab_n_tokens(sys::llama_model_get_vocab(model)) as usize;
            let mut desc = vec![0 as c_char; 256];
            sys::llama_model_desc(model, desc.as_mut_ptr(), desc.len());
            let desc = std::ffi::CStr::from_ptr(desc.as_ptr()).to_string_lossy().into_owned();
            let facts = ModelFacts {
                desc,
                n_layer: ffi::llama_model_n_layer(model),
                n_embd: sys::llama_model_n_embd(model),
                n_vocab,
                n_ctx_train: ffi::llama_model_n_ctx_train(model),
                weights_bytes: ffi::llama_model_size(model),
                cells: o.cells,
                kv_type: o.kv_type.name(),
            };
            let cap = sys::llama_n_batch(ctx) as usize;
            let free = (0..cp.n_seq_max as i32).rev().collect();
            Ok(Exec { model, ctx, batch: Batch::new(cap), cap, n_vocab, seqs: HashMap::new(), free, facts })
        }
    }

    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    fn mem(&self) -> sys::llama_memory_t {
        unsafe { sys::llama_get_memory(self.ctx) }
    }

    /// A new, empty sequence that may hold `capacity` tokens.
    pub fn add_agent(&mut self, capacity: usize) -> Result<SeqId, String> {
        let id = self.free.pop().ok_or("no free sequence id (too many helper sequences alive)")?;
        self.seqs.insert(id, SeqState { len: 0, capacity, base: 0 });
        Ok(id)
    }

    /// A new sequence sharing every cell of `src`: nothing is copied, a later token of either gets its own cell.
    pub fn fork_agent(&mut self, src: SeqId) -> Result<SeqId, String> {
        let (len, capacity) = self.seqs.get(&src).map(|s| (s.len, s.capacity)).ok_or_else(|| format!("no sequence {src}"))?;
        let id = self.free.pop().ok_or("no free sequence id (too many helper sequences alive)")?;
        if len > 0 {
            // SAFETY: both ids are in range; the unified cache adds `id` to the cells of `src`.
            unsafe { ffi::llama_memory_seq_cp(self.mem(), src, id, 0, -1) };
        }
        self.seqs.insert(id, SeqState { len, capacity, base: len });
        Ok(id)
    }

    /// Drop a sequence and free its cells (cells another sequence shares stay).
    pub fn remove_agent(&mut self, id: SeqId) -> Result<(), String> {
        if self.seqs.remove(&id).is_none() {
            return Err(format!("no sequence {id}"));
        }
        // SAFETY: plain call; -1 / -1 is the whole sequence.
        unsafe { ffi::llama_memory_seq_rm(self.mem(), id, -1, -1) };
        self.free.push(id);
        Ok(())
    }

    pub fn agent_len(&self, id: SeqId) -> Result<usize, String> {
        self.seqs.get(&id).map(|s| s.len).ok_or_else(|| format!("no sequence {id}"))
    }

    /// Tokens the sequence can still take.
    pub fn agent_room(&self, id: SeqId) -> Result<usize, String> {
        self.seqs.get(&id).map(|s| s.capacity.saturating_sub(s.len)).ok_or_else(|| format!("no sequence {id}"))
    }

    /// Cut the sequence back to its first `len` tokens.
    pub fn truncate(&mut self, id: SeqId, len: usize) -> Result<(), String> {
        let cur = self.seqs.get(&id).map(|s| s.len).ok_or_else(|| format!("no sequence {id}"))?;
        if len >= cur {
            return Ok(());
        }
        // SAFETY: plain call.
        if !unsafe { ffi::llama_memory_seq_rm(self.mem(), id, len as i32, -1) } {
            return Err("this model's cache cannot drop a token range".into());
        }
        if let Some(s) = self.seqs.get_mut(&id) {
            s.len = len;
            s.base = s.base.min(len);
        }
        Ok(())
    }

    pub fn n_agents(&self) -> usize {
        self.seqs.len()
    }

    /// Pages of 256 cells in use, a cell shared by forks counted once.
    pub fn used_pages(&self) -> usize {
        let cells: usize = self.seqs.values().map(|s| s.len - s.base.min(s.len)).sum();
        cells.div_ceil(PAGE_CELLS)
    }

    /// Run one batched step: every item's tokens in ONE `llama_decode`. The items' lengths grow by their token counts.
    pub fn step(&mut self, items: &[StepItem]) -> Result<Vec<ItemOut>, String> {
        let n_total: usize = items.iter().map(|i| i.tokens.len()).sum();
        if n_total == 0 {
            return Ok(items.iter().map(|_| ItemOut::default()).collect());
        }
        if n_total > self.cap {
            return Err(format!("a step of {n_total} tokens is more than the batch size {}", self.cap));
        }
        let pos0 = self.start_positions(items)?;
        let rows = self.fill_batch(items, &pos0);
        let rc = unsafe { sys::llama_decode(self.ctx, self.batch.b) };
        if rc != 0 {
            return Err(self.decode_failed(items, &pos0, rc));
        }
        let mut outs = Vec::with_capacity(items.len());
        for (it, r) in items.iter().zip(&rows) {
            if let Some(s) = self.seqs.get_mut(&it.agent) {
                s.len += it.tokens.len();
            }
            outs.push(self.item_out(it, r)?);
        }
        Ok(outs)
    }

    /// A failed decode: the cells it may have written are dropped again; the reason.
    fn decode_failed(&self, items: &[StepItem], pos0: &[usize], rc: i32) -> String {
        for (it, &p0) in items.iter().zip(pos0) {
            unsafe { ffi::llama_memory_seq_rm(self.mem(), it.agent, p0 as i32, -1) };
        }
        match rc {
            1 => "llama_decode: no KV slot for the batch (the helper cache is full)".to_string(),
            2 => "llama_decode: aborted".to_string(),
            n => format!("llama_decode returned {n}"),
        }
    }

    /// Each item's first position: its sequence's length, checked to have room for the item's tokens.
    fn start_positions(&self, items: &[StepItem]) -> Result<Vec<usize>, String> {
        let mut pos0 = Vec::with_capacity(items.len());
        for it in items {
            let s = self.seqs.get(&it.agent).ok_or_else(|| format!("no sequence {}", it.agent))?;
            if s.len + it.tokens.len() > s.capacity {
                return Err(format!("sequence {} is full ({} + {} > {})", it.agent, s.len, it.tokens.len(), s.capacity));
            }
            pos0.push(s.len);
        }
        Ok(pos0)
    }

    /// The items' tokens into the batch; returns, per item, the batch indices of its output rows.
    fn fill_batch(&mut self, items: &[StepItem], pos0: &[usize]) -> Vec<Vec<i32>> {
        // rows[i]: batch indices of item i's output rows
        let mut rows: Vec<Vec<i32>> = Vec::with_capacity(items.len());
        // the batch holds `cap` tokens and n_total <= cap
        let b = &mut self.batch;
        let mut k = 0usize;
        for (it, &p0) in items.iter().zip(pos0) {
            let mut r = Vec::new();
            for (j, &t) in it.tokens.iter().enumerate() {
                let out = it.output.outputs(j, it.tokens.len());
                b.set(k, t, (p0 + j) as i32, it.agent, out);
                if out {
                    r.push(k as i32);
                }
                k += 1;
            }
            rows.push(r);
        }
        // libllama wants at least one output in a batch
        if rows.iter().all(|r| r.is_empty()) {
            b.want_logits(k - 1);
        }
        b.b.n_tokens = k as i32;
        rows
    }

    /// An item's output after the decode: the argmax of each of its output rows `r`, and their logits for host logits.
    fn item_out(&self, it: &StepItem, r: &[i32]) -> Result<ItemOut, String> {
        let mut out = ItemOut::default();
        let mut logits = (it.sampler == Sampler::HostLogits && !r.is_empty()).then(|| Vec::with_capacity(r.len() * self.n_vocab));
        for &idx in r {
            let p = unsafe { sys::llama_get_logits_ith(self.ctx, idx) };
            if p.is_null() {
                return Err(format!("llama_get_logits_ith({idx}) returned NULL"));
            }
            // SAFETY: libllama keeps n_vocab floats per output row until the next decode.
            let row = unsafe { std::slice::from_raw_parts(p, self.n_vocab) };
            out.tokens.push(argmax(row));
            if let Some(l) = &mut logits {
                l.extend_from_slice(row);
            }
        }
        out.logits = logits;
        Ok(out)
    }
}

impl Drop for Exec {
    fn drop(&mut self) {
        // SAFETY: both were returned by libllama and are freed once; the batch is freed by its own Drop.
        unsafe { llama_ref::free_context_and_model(self.ctx, self.model) }
    }
}
