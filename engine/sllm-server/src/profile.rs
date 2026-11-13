//! Generic model profile: everything the per-family native engine hard-coded, computed at runtime for ANY GGUF.
//!
//! No per-architecture code. The profile is read from the GGUF header with ggml's own `gguf_*` API (the metadata keys
//! llama.cpp itself reads, `<arch>.block_count`, `<arch>.attention.head_count_kv` (scalar or per layer), `key_length`,
//! `sliding_window`, `expert_count`, ... and the tensor table) and cross-checked against a loaded `llama_model`
//! ([`Profile::verify_model`]) and against libllama's own KV accounting ([`probe_memory`], a `no_alloc` context: nothing is
//! allocated on the GPU).
//!
//! * [`Profile`]: layers, heads, KV heads, head dims, embedding size, vocabulary, training context, sliding window and the
//!   SWA layers, experts, weight bytes (total and per decode step: for a mixture of experts only the experts a step
//!   touches), KV bytes per token per sequence (full and SWA layers apart, by KV type), whether prefix sharing is safe.
//! * [`Budget`]: the memory budget carved into weights, compute buffers and KV; the cells to allocate; the agents x context
//!   table; the decode roofline (bytes per step / machine bandwidth).
//! * [`Machine`]: the machine's memory bandwidth, a config value (`engine/config/machine.toml`).
//!
//! The startup calibration that measures the real curves lives in [`crate::calibrate`].

use llama_ref::sys;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

mod budget;
mod gguf;
mod machine;
mod probe;
mod read;

pub use budget::{kv_cache_bytes, plan_engine, requests_for, Budget, BudgetInputs, PlanArgs, DEFAULT_AGENTS, HEADROOM_BYTES, MAX_TOTAL_CELLS, MAX_UNIFIED_CELLS};
pub use gguf::{Gguf, TensorInfo};
pub use machine::{default_machine_config_path, parse_machine_bandwidth, Machine};
pub(crate) use machine::sysctl;
pub use probe::{capture_log, parse_compute_bytes, parse_kv_lines, probe_memory, KvLine, Probe, ProbeParams, LLAMA_LOG_GUARD};
pub use read::fingerprint;
use gguf::ffi;
#[cfg(test)]
use budget::wanted;
#[cfg(test)]
use read::period_pattern;

/// ggml tensor types the KV cache can use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvType {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q4_0,
}

impl KvType {
    pub fn ggml(self) -> i32 {
        match self {
            KvType::F32 => sys::GGML_TYPE_F32 as i32,
            KvType::F16 => sys::GGML_TYPE_F16 as i32,
            KvType::Bf16 => 30,
            KvType::Q8_0 => sys::GGML_TYPE_Q8_0 as i32,
            KvType::Q4_0 => sys::GGML_TYPE_Q4_0 as i32,
        }
    }
    pub fn parse(s: &str) -> Option<KvType> {
        [KvType::F32, KvType::F16, KvType::Bf16, KvType::Q8_0, KvType::Q4_0].into_iter().find(|t| t.name() == s)
    }
    /// Bytes of a row of `n` elements.
    pub fn row_bytes(self, n: u64) -> u64 {
        unsafe { ffi::ggml_row_size(self.ggml(), n as i64) as u64 }
    }
}

variant_names!(KvType { F32 => "f32", F16 => "f16", Bf16 => "bf16", Q8_0 => "q8_0", Q4_0 => "q4_0" });

// ---------------------------------------------------------------------------------------------------------------------
// The profile
// ---------------------------------------------------------------------------------------------------------------------

/// Cells of the SWA cache and of the unified cache are padded to this (libllama).
pub const CELL_PAD: u64 = 256;

fn pad(n: u64) -> u64 {
    n.div_ceil(CELL_PAD) * CELL_PAD
}

/// What one layer stores per KV cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerKv {
    pub n_head_kv: u32,
    /// Per-head key and value dimensions (the SWA layers' own when the model has them).
    pub k_dim: u32,
    pub v_dim: u32,
    pub swa: bool,
}

impl LayerKv {
    pub fn k_elems(&self) -> u64 {
        self.n_head_kv as u64 * self.k_dim as u64
    }
    pub fn v_elems(&self) -> u64 {
        self.n_head_kv as u64 * self.v_dim as u64
    }
}

/// Where the SWA layer map came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwaSource {
    /// No sliding window in this model.
    None,
    /// `<arch>.attention.sliding_window_pattern` as a per-layer array.
    MetadataArray,
    /// `<arch>.attention.sliding_window_pattern` as a period N (every N-th layer is a full-attention layer).
    MetadataPeriod,
    /// Not in the metadata: the number of SWA layers was read from libllama's own cache (a `no_alloc` context) and laid
    /// out with llama.cpp's generic period rule.
    Libllama,
    /// A sliding window with no layer map and no libllama probe: every layer is treated as SWA (an upper bound on the
    /// savings, a lower bound on the bytes is NOT guaranteed; `notes` says so).
    Assumed,
}

variant_names!(SwaSource {
    None => "none",
    MetadataArray => "metadata: sliding_window_pattern array",
    MetadataPeriod => "metadata: sliding_window_pattern period",
    Libllama => "libllama: no_alloc context (layer counts), llama.cpp period rule",
    Assumed => "assumed (no layer map in the metadata, no probe)",
});

#[derive(Debug, Clone)]
pub struct Profile {
    pub path: PathBuf,
    pub arch: String,
    pub name: Option<String>,
    pub file_bytes: u64,
    /// Size plus a hash of sampled blocks: cheap, and changes when the file does.
    pub fingerprint: String,
    pub n_layer: usize,
    pub n_embd: u64,
    /// Attention heads and KV heads: the maximum over the layers (layers differ in hybrid models).
    pub n_head: u32,
    pub n_head_kv: u32,
    /// Per-head key / value dimension of the full-attention layers.
    pub head_dim_k: u32,
    pub head_dim_v: u32,
    pub n_vocab: u64,
    pub n_ctx_train: u64,
    pub n_swa: u32,
    pub layers: Vec<LayerKv>,
    pub swa_source: SwaSource,
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub n_expert_shared: u32,
    pub recurrent: bool,
    pub hybrid: bool,
    /// MLA: the cache holds one latent per token and no V.
    pub mla: bool,
    pub weights_total: u64,
    /// Read in full by every decode step whatever the batch: everything but the expert stacks and the embedding table.
    pub weights_dense: u64,
    /// The expert stacks (`*_exps*` tensors): a step reads the experts its tokens route to.
    pub weights_experts: u64,
    /// Embedding tables that are only gathered (one row per token). A tied embedding is also the output head and is then
    /// counted in `weights_dense`.
    pub weights_gather: u64,
    /// Bytes of one embedding row (token_embd).
    pub embd_row_bytes: u64,
    pub tied_head: bool,
    pub notes: Vec<String>,
}

impl Profile {
    pub fn is_moe(&self) -> bool {
        self.n_expert > 1 && self.weights_experts > 0
    }

    pub fn n_swa_layers(&self) -> usize {
        self.layers.iter().filter(|l| l.swa).count()
    }

    pub fn n_full_layers(&self) -> usize {
        self.layers.iter().filter(|l| !l.swa && l.n_head_kv > 0).count()
    }

    /// Bytes one KV cell costs across the full-attention layers and across the SWA layers, for a cache type.
    pub fn kv_bytes_per_cell(&self, ty: KvType) -> (u64, u64) {
        let (mut full, mut swa) = (0u64, 0u64);
        for l in &self.layers {
            if l.n_head_kv == 0 {
                continue;
            }
            let b = ty.row_bytes(l.k_elems()) + if l.v_dim > 0 { ty.row_bytes(l.v_elems()) } else { 0 };
            if l.swa {
                swa += b;
            } else {
                full += b;
            }
        }
        (full, swa)
    }

    /// Cells of libllama's SWA cache for a unified cache of `cells` base cells.
    /// With `swa_full` (libllama's default, which LlamaEngine keeps) the SWA cache is as large as the base cache.
    pub fn swa_cells(&self, cells: u64, n_seq: u64, n_ubatch: u64, swa_full: bool) -> u64 {
        if self.n_swa_layers() == 0 {
            return 0;
        }
        if swa_full {
            return cells;
        }
        pad(cells.min(self.n_swa as u64 * n_seq + n_ubatch))
    }

    /// Bytes libllama allocates for the KV cache of `cells` unified cells (its `llama_kv_cache_iswa` sizing: the SWA cache
    /// holds `min(cells, n_swa * n_seq + n_ubatch)` cells, padded to 256).
    pub fn kv_alloc_bytes(&self, ty: KvType, cells: u64, n_seq: u64, n_ubatch: u64, swa_full: bool) -> u64 {
        let (full, swa) = self.kv_bytes_per_cell(ty);
        full * cells + swa * self.swa_cells(cells, n_seq, n_ubatch, swa_full)
    }

    /// Bytes of KV one sequence of `ctx` tokens needs when the SWA layers keep only their window.
    pub fn kv_bytes_per_seq(&self, ty: KvType, ctx: u64, n_ubatch: u64) -> u64 {
        let (full, swa) = self.kv_bytes_per_cell(ty);
        full * ctx + swa * ctx.min(self.n_swa as u64 + n_ubatch)
    }

    /// Fraction of an MoE's expert bytes a step touches with `batch` tokens, if routing were uniform at random:
    /// `1 - (1 - used/experts)^batch`. A lower bound on the expert traffic of real (skewed) routing at batch > 1.
    pub fn expert_fraction(&self, batch: u64) -> f64 {
        if !self.is_moe() || self.n_expert_used == 0 {
            return 1.0;
        }
        let p = (self.n_expert_used as f64 / self.n_expert as f64).min(1.0);
        1.0 - (1.0 - p).powi(batch.min(1 << 20) as i32)
    }

    /// Weight bytes one decode step reads for `batch` tokens.
    pub fn weights_per_step(&self, batch: u64) -> u64 {
        let experts = (self.weights_experts as f64 * self.expert_fraction(batch)) as u64;
        // a tied embedding is read in full as the output head (already in `weights_dense`); otherwise one row per token
        let gathered = if self.tied_head { 0 } else { self.embd_row_bytes * batch };
        self.weights_dense + experts + gathered
    }

    /// Bytes of KV one decode row reads at context length `ctx`.
    pub fn kv_read_bytes(&self, ty: KvType, ctx: u64) -> u64 {
        let (full, swa) = self.kv_bytes_per_cell(ty);
        full * ctx + swa * ctx.min(self.n_swa as u64)
    }

    /// Bytes one decode step of `batch` rows at context `ctx` reads: the roofline's numerator.
    pub fn decode_step_bytes(&self, ty: KvType, batch: u64, ctx: u64) -> u64 {
        self.weights_per_step(batch) + batch * self.kv_read_bytes(ty, ctx)
    }

    /// The prefix-sharing rule the generic engine applies: a partial prefix copy is valid for plain attention only.
    pub fn prefix_sharing_safe(&self) -> bool {
        self.n_swa == 0 && !self.recurrent && !self.hybrid
    }

    /// The `/caps` `profile` object.
    pub fn to_json(&self) -> Value {
        let ty = |t: KvType| {
            let (full, swa) = self.kv_bytes_per_cell(t);
            json!({"full_layers": full, "swa_layers": swa, "total_once": full + swa})
        };
        let swa_idx: Vec<usize> = self.layers.iter().enumerate().filter(|(_, l)| l.swa).map(|(i, _)| i).collect();
        let (f16_full, f16_swa) = self.kv_bytes_per_cell(KvType::F16);
        json!({
            "model": self.name, "file": self.path.display().to_string(), "arch": self.arch, "fingerprint": self.fingerprint,
            "layers": self.n_layer, "embd": self.n_embd, "heads": self.n_head, "kv_heads": self.n_head_kv,
            "head_dim_k": self.head_dim_k, "head_dim_v": self.head_dim_v, "vocab": self.n_vocab, "ctx_train": self.n_ctx_train,
            "sliding_window": self.n_swa,
            "swa_layers": {"count": swa_idx.len(), "full_layers": self.n_full_layers(), "indices": swa_idx, "source": self.swa_source.name()},
            "attention_layers": self.layers.iter().filter(|l| l.n_head_kv > 0).count(),
            "moe": {"experts": self.n_expert, "used": self.n_expert_used, "shared": self.n_expert_shared, "is_moe": self.is_moe()},
            "mla": self.mla, "recurrent": self.recurrent, "hybrid": self.hybrid,
            "weights": {
                "total_bytes": self.weights_total, "dense_bytes": self.weights_dense, "expert_bytes": self.weights_experts,
                "gather_bytes": self.weights_gather, "tied_head": self.tied_head,
                "per_step_bytes_batch1": self.weights_per_step(1), "per_step_bytes_batch8": self.weights_per_step(8),
                "per_step_bytes_batch32": self.weights_per_step(32),
            },
            "kv_bytes_per_token_per_seq": {
                "f16": ty(KvType::F16), "q8_0": ty(KvType::Q8_0), "q4_0": ty(KvType::Q4_0),
                "note": "full_layers grows with the context; swa_layers stops at the sliding window",
            },
            "kv_f16_bytes_per_token": {"full": f16_full, "swa": f16_swa},
            "prefix_sharing_safe": self.prefix_sharing_safe(),
            "prefix_sharing_reason": if self.prefix_sharing_safe() { "plain attention" } else if self.n_swa > 0 { "sliding-window layers" } else { "recurrent or hybrid state" },
            "notes": self.notes,
        })
    }

    /// The one startup line.
    pub fn summary_line(&self) -> String {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let (full, swa) = self.kv_bytes_per_cell(KvType::F16);
        let mut s = format!(
            "profile: {} {}L {}h/{}kv hd {}/{} embd {} vocab {} ctx_train {}",
            self.arch, self.n_layer, self.n_head, self.n_head_kv, self.head_dim_k, self.head_dim_v, self.n_embd, self.n_vocab, self.n_ctx_train
        );
        if self.n_swa > 0 {
            s += &format!(", swa {} on {}/{} layers", self.n_swa, self.n_swa_layers(), self.n_layer);
        }
        if self.is_moe() {
            s += &format!(", moe {}/{} experts", self.n_expert_used, self.n_expert);
        }
        s += &format!(
            ", weights {:.2} GiB ({:.2} GiB/step), kv f16 {} B/token (+{} B windowed), prefix sharing {}",
            gib(self.weights_total),
            gib(self.weights_per_step(1)),
            full,
            swa,
            if self.prefix_sharing_safe() { "safe" } else { "unsafe" }
        );
        s
    }
}

/// Read the profile of `model` and settle what the header cannot say with libllama; the shared first step of `main` and the engine.
pub fn load_profile(model: &Path) -> Result<Profile, String> {
    let mut p = Profile::from_gguf(model)?;
    p.resolve_with_libllama();
    Ok(p)
}

// ---------------------------------------------------------------------------------------------------------------------
// Process-wide startup result (set by main, read by the engine)
// ---------------------------------------------------------------------------------------------------------------------

/// What `main` computed before the engine was built: the engine merges it into `/caps`.
#[derive(Debug, Clone)]
pub struct Startup {
    pub profile: Profile,
    pub machine: Machine,
    /// `calibrate::Calibration` as JSON (null when `--calibrate off`).
    pub calibration: Value,
}

static STARTUP: OnceLock<Startup> = OnceLock::new();

/// What `--auto-mode` chose, for `/caps`.
static AUTO_MODE: std::sync::OnceLock<Value> = std::sync::OnceLock::new();

pub fn set_auto_mode(v: Value) {
    let _ = AUTO_MODE.set(v);
}

pub fn auto_mode() -> Option<&'static Value> {
    AUTO_MODE.get()
}

pub fn set_startup(s: Startup) {
    let _ = STARTUP.set(s);
}

pub fn startup() -> Option<&'static Startup> {
    STARTUP.get()
}

#[cfg(test)]
mod tests;