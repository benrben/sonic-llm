//! `LlamaEngine`: the engine. The math runs through llama.cpp's `libllama` (the vendored b11669), so sllm-server serves ANY
//! GGUF llama.cpp can load, while the server, scheduler, lanes, agents API, helper and compaction stay SLLM's own. (The
//! model-specific native engine that used to sit beside it is preserved at the repository tag `native-engine-final`.)
//!
//! # Shape
//!
//! * **One context, many sequences.** One `llama_context` with `n_seq_max` sequence ids and a unified KV cache (`kv_unified`):
//!   the cells are one pool shared by every sequence, sized to the memory budget (or `--ctx` times the slots, capped).
//!   A `/completion` request gets a sequence id when it is given a slot and frees it when it ends; an agent holds its
//!   sequence for its lifetime, so a turn prefills only the new ids.
//! * **One batch per step.** Every row of the step (prefill chunks and decode tokens of every running request) goes into ONE
//!   `llama_batch`, tagged with its sequence id, logits on for the last token of each sampling row. One `llama_decode`.
//! * **Sampling** is SLLM's phase 1 sampler: greedy (the lowest id on a tie); `ignore_eos` masks every end-of-generation token.
//!   A request's `grammar` (GBNF) is a libllama grammar sampler kept in its [`Job`] ([`Gram`]): the argmax is taken first, and
//!   only when the grammar refuses it the row is masked and the argmax taken again. Every token that reaches a request's
//!   output is checked and then accepted in [`LlamaEngine::on_token`]; speculation is off for the steps that hold such a job.
//! * **The pending token**: a sampled token is fed one step later, so an agent's last sampled
//!   token is kept as `pending` and prefilled in front of the next turn's ids.
//! * **Prefix sharing** (`--prefix-sharing on`, the default): a request whose prompt starts like another sequence's context
//!   copies that prefix with `llama_memory_seq_cp` (unified cells are shared, not copied) and prefills only the rest. Off for
//!   models with sliding-window attention or recurrent state, where a partial copy is not valid. There is no coalesced
//!   prefill (waiting on a writer); [`Engine::prefix_state`] is used by `--kv-read f16` only (below).
//! * **f16 reading** (`--kv-read f16`, with a quantized `--kv-type`): prompts are read in f16 contexts of one sequence (the
//!   readers), in blocks at fixed positions; the reader's state is then rotated and quantized the way the
//!   quantized cache stores it ([`crate::kvconv`]) and set into the request's sequence, which decodes from the quantized
//!   cache. There are two readers: a small one ([`CHAT_READER_CELLS`]) for an interactive prompt that fits it, and a long one
//!   for every other read, whose context is made for the read and freed after a long one (so an idle server holds only the
//!   small reader's f16 cache; a sliding-window model's long reader stays open, see `Reader::ring`; the long reader's context is
//!   made on another thread, so a chat arriving meanwhile is not held up). One request reads in a reader at a time (the others wait as `PrefixState::Waiting`); a sequence
//!   that holds more cached tokens than the reader could run again (`READ_REFILL_MAX`) is read in the main cache. A later read
//!   of the same sequence converts only its new cells. A long read goes to the SSD (`rcache`) as the reader gives its memory
//!   back, so a sequence that returns to it (a warm request, an agent's next turn) reads its new tokens on the same f16 cells
//!   after the state is loaded again. The conversion of
//!   a long read runs off the executor thread, so the other sequences keep decoding; the GPU work of a read does not (two
//!   contexts decoding at once queue behind each other's whole graphs: a chat step waits for a read chunk every time).
//! * **Suspend / wake** copies the sequence out with `llama_state_seq_get_data` into a reused buffer (the only part on the
//!   executor thread), frees its cells, and writes the file on the I/O thread ([`crate::spill_io`]); completion comes back
//!   as a [`LifecycleEvent::Written`]. A wake reads the file on that thread as soon as it is asked for (the executor asks
//!   when a turn for the agent is queued) and `llama_state_seq_set_data` runs when the bytes are there.
//! * **Prefix lookup** is a trie over the token ids of the cached sequences ([`crate::prefix_tree`]), kept in step with the
//!   cache. Models whose cache cannot be copied in part (sliding-window attention, recurrent or hybrid state) share through
//!   **frozen prefix holders**: the second agent created with the same system prompt makes a dedicated holder sequence
//!   that prefills it once and never decodes; new sequences start from a FULL copy of the holder
//!   (`llama_memory_seq_cp(holder, dst, 0, -1)`, valid for every cache type) and prefill only what follows. Holders are
//!   evicted first when memory is needed.
//! * **Warm requests** (when partial copies are valid): a finished `/completion` keeps its cells as a holder, up to
//!   `MAX_HOLDERS`, evicted first when a stream or cells are needed. The next prompt that starts the same way takes the
//!   sequence over (no copy) and prefills only what differs, as llama-server's slots do. `cache_prompt: false` opts out.
//!   Models that cannot drop a tail: a long request (a completion, or an agent's first turn) copies its stream into a frozen
//!   holder before its last chunk, when a stream is free after the requests waiting for one; else it saves that state to the
//!   prompt cache on disk, which a repeat loads like any saved start.
//! * **Memory is granted on demand** ([`crate::kvmem`]): a request is admitted on its prompt plus a small decode headroom,
//!   its sequence grows as tokens are produced, and when the cache is about to be full the engine suspends the
//!   least-recently-used idle agent to the SSD or pauses the lowest-priority running request (tokens kept, cells dropped,
//!   re-prefilled when there is room). A 503 `kv_pool_exhausted_error` only when nothing can be freed.
//! * Not supported: the per-role KV precision policies and the own-q8 layout of the old native engine (one cache type for
//!   every sequence, `--kv-type`), coalesced prefill.
//!
//! All libllama calls come from the executor thread (the engine is built on it).

use crate::engine::*;
use crate::spec::{SpecConfig, SpecMode, SpecState};
use llama_ref::{sys, Batch, Vocab};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, CString};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crate::kvmem::{self, IdleAgent, Pausable, HEADROOM, MARGIN};
use crate::prefix_tree::PrefixTree;
use crate::spill_io::{self, SpillIo};

mod config;
mod decode;
mod engine_impl;
/// libllama functions that llama-ref's bindings do not declare (the signatures are those of the vendored `llama.h`).
pub(crate) mod ffi;
mod lifecycle;
mod load;
mod load_plan;
mod memory;
mod prefix;
mod read;
mod read_plan;
mod requests;
mod reader;
mod sampling;
mod speculation;
mod step;
mod streams;
#[cfg(test)]
mod tests;

pub use config::{EngineConfig, KvType, RowExact};
pub use reader::{read_block_at, reader_cells, reader_kv_bytes, reader_reserve_bytes, CHAT_READER_CELLS};
pub(crate) use sampling::{argmax, argmax_excluding, text_of, Gram, GRAMMAR_DONE};

// what the submodules share (each one sees this module's names through `use super::*`)
use lifecycle::Stall;
use memory::MemStats;
use prefix::{hash_ids, Holder, HolderStats, HOLDER_MIN, WARM_COPY_MIN};
use read::Handover;
use reader::{own_reader, Reader, READ_REFILL_MAX, READ_THREAD_MIN};
use requests::reply_budget;
use sampling::sample_rows;
use speculation::{SpecInfo, SpecStats};
use step::StepWork;
use streams::{CopyQueue, StreamRuns};
// what only the tests use
#[cfg(test)]
use {
    reader::{reader_compute_bytes, READER_COMPUTE_FALLBACK},
    streams::{id_runs, plan_pack, MAX_PACK_MOVES},
};

/// Cells of the unified cache per sequence-id slot: sequence ids beyond the scheduler's lane slots are for agents.
const MIN_SEQ_IDS: usize = 64;
pub const MAX_SEQ_IDS: usize = 256;
const CELL_PAD: usize = 256;

fn pad(n: usize) -> usize {
    n.div_ceil(CELL_PAD) * CELL_PAD
}

/// `general.architecture` of a GGUF (None when the file cannot be read).
pub fn gguf_arch(path: &std::path::Path) -> Option<String> {
    sllm_core::gguf::GgufFile::open(path).ok()?.get_str("general.architecture")
}

struct Seq {
    sid: i32,
    /// The tokens whose K/V are in the cache, in position order.
    tokens: Vec<i32>,
    /// The last sampled token, whose K/V are not in the cache yet.
    pending: Option<i32>,
}

struct AgentSeq {
    #[allow(dead_code)]
    lane: Lane,
    seq: Seq,
    /// Suspended: the sequence's cells are freed, `snapshot` holds its state on the SSD. Its sequence id is kept.
    suspended: Option<Snapshot>,
    /// Last time a turn started or ended on it (or it was created or restored): the LRU order of preemption.
    last_used: Instant,
}

struct Snapshot {
    path: PathBuf,
    bytes: u64,
    /// The tokens in the cache when it was taken (the cache is empty meanwhile).
    tokens: Vec<i32>,
    /// Identifies the snapshot's I/O jobs (completions of a snapshot that is gone are dropped).
    key: u64,
    st: SnapSt,
    /// When the wake was asked for.
    resumed: Option<Instant>,
}

enum SnapSt {
    /// The write is queued or running on the I/O thread.
    Writing,
    Written,
    /// A wake is reading the file on the I/O thread.
    Reading,
    /// The bytes are in memory, waiting for `llama_state_seq_set_data` on the executor thread.
    Ready(Vec<u8>),
}

/// An admitted request, queued or running.
struct Job {
    req: ReqId,
    agent: Option<AgentId>,
    ids: Vec<i32>,
    params: SamplingParams,
    budget: usize,
    ctx_limited: bool,
    lane: Lane,
    /// Admission order (larger = younger).
    order: u64,
    /// Paused for memory: its sequence state is saved in `snap` (like an agent suspend), its cells and stream are free,
    /// and it waits for room. The sampled token stays in `Seq::pending`.
    paused: bool,
    snap: Option<Snapshot>,
    /// A completion's own sequence (assigned when it is given a slot).
    seq: Option<Seq>,
    begun: bool,
    started: Option<Instant>,
    first: Option<Instant>,
    /// What the prefill feeds: the pending token (an agent), then the ids not served from a shared prefix.
    feed: Vec<i32>,
    fed: usize,
    shared_n: usize,
    len_before: usize,
    pending_before: Option<i32>,
    emitted: usize,
    carry: Vec<u8>,
    /// `--kv-read f16`: None until decided (or while it waits for the reader), Some(true) when the reader reads its prompt,
    /// Some(false) when it is read in the main cache.
    read: Option<bool>,
    /// `--kv-read f16`: the reader that reads (or read) its prompt, chosen with `read` in [`LlamaEngine::prefix_state`].
    reader: usize,
    /// `--kv-read f16`: its read is being converted ([`Handover`]); the request has no pending token yet.
    handover: bool,
    /// The request's grammar state (`grammar`). Lives and dies with the job (an agent's next turn builds its own), so a
    /// paused job keeps it and a cancelled one frees it.
    gram: Option<Box<Gram>>,
}

/// One row of a step.
struct Row {
    req: ReqId,
    sid: i32,
    /// Position of the first token.
    pos0: usize,
    toks: Toks,
    samples: bool,
    prefill: bool,
    ignore_eos: bool,
    /// The request has a grammar: its token is picked by [`Gram::pick`] and it never drafts.
    gram: bool,
    /// Draft tokens after the pending one (speculation): the row holds `[pending, draft...]` and every position has logits.
    n_draft: usize,
}

/// The tokens of a row: a decode row holds its one token inline (no allocation per row per step).
enum Toks {
    One(i32),
    Many(Vec<i32>),
}

impl Toks {
    fn as_slice(&self) -> &[i32] {
        match self {
            Toks::One(t) => std::slice::from_ref(t),
            Toks::Many(v) => v,
        }
    }
    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

pub struct LlamaEngine {
    #[allow(dead_code)]
    cfg: EngineConfig,
    model: *mut sys::llama_model,
    ctx: *mut sys::llama_context,
    batch: Batch,
    vocab: Arc<Vocab>,
    n_vocab: usize,
    eog: Arc<Vec<i32>>,
    arch: String,
    info: EngineInfo,
    caps: Value,
    /// Cells one sequence may hold (`--ctx` rounded up to 256).
    capacity: usize,
    /// The sliding-window layers keep a ring (not `swa_full`).
    swa_ring: bool,
    /// Cells of the unified cache.
    total_cells: usize,
    share: bool,
    /// One shared cell pool (true) or one KV stream per sequence id (false).
    unified: bool,
    free_sids: Vec<i32>,
    agents: HashMap<AgentId, AgentSeq>,
    jobs: HashMap<ReqId, Job>,
    counters: EngineCounters,
    prefix: PrefixCounters,
    timings: HashMap<ReqId, (f64, f64)>,
    last_bd: StepBreakdown,
    /// Lifecycle events to report at the next poll.
    events: Vec<LifecycleEvent>,
    spill_dir: PathBuf,
    /// The spill directory is this process's own (the default), so Drop may remove it.
    own_spill_dir: bool,
    ssd_bytes: u64,
    ssd_writes: u64,
    ssd_reads: u64,
    /// Diagnostics (SLLM_LLAMA_PROF=1): per decode with sampling rows: (wait for the GPU in the first get_logits, argmax, rows).
    prof: Option<Vec<(u64, u64, u32)>>,
    /// Snapshot writes and reads run here, off the executor thread.
    io: SpillIo,
    next_key: u64,
    stall: Stall,
    next_order: u64,
    mem: MemStats,
    /// Token trie of the cached sequences (all of them when partial copies are valid, else the frozen holders).
    ptree: std::cell::RefCell<PrefixTree>,
    /// Frozen prefix holders (models where only whole-sequence copies are valid).
    holders: Vec<Holder>,
    holders_on: bool,
    /// Per-sequence streams: libllama builds one micro-batch from consecutive, increasing sequence ids, so a step whose rows
    /// have a gap in their ids runs as several graphs. Counted per step (`stream_runs` in the engine metrics).
    stream_runs: StreamRuns,
    /// Per-sequence streams: running sequences are moved down into free ids so their ids stay consecutive (`SLLM_SEQ_PACK=0`
    /// turns it off).
    seq_pack: bool,
    /// Per-sequence streams, plain attention: a stream copy moves only the rows that hold cells instead of the whole stream
    /// (`SLLM_COPY_PARTIAL=0` turns it off; off too where ggml cannot do it). See [`Self::stream_copy`].
    copy_partial: bool,
    /// What the copies queued since the last decode need; the decode that applies them gets it as the row hint.
    copy_queue: CopyQueue,
    /// A cut that left a hole in a stream's cells was seen: rows are no longer a prefix, every copy is whole from then on.
    copy_holes: std::cell::Cell<bool>,
    /// A finished request's state goes to the prompt cache on disk when the engine is otherwise idle (`SLLM_TURN_SAVE=0`
    /// turns it off).
    turn_save: bool,
    /// `SLLM_KV_READ_PICK=0`: a request always uses its lane's reader (checks only: the negative control of the reader choice).
    read_pick: bool,
    /// How many times a system prompt was seen (hash), to decide when a holder pays.
    seen_prefix: HashMap<u64, u32>,
    /// The starts of recent fresh prompts (holder models): the second prompt that shares one makes a holder for it.
    recent_starts: std::collections::VecDeque<Vec<i32>>,
    holder_stats: HolderStats,
    /// `SLLM_PREFIX_CHECK=1`: every lookup is compared with the linear scan (diagnostics).
    check_prefix: bool,
    /// Paused requests whose state could not be saved or restored: reported as failed at the next step.
    job_failures: Vec<(ReqId, String)>,
    /// The model's sequence state cannot be saved: idle agents are not preemption candidates.
    no_suspend: bool,
    /// Effective speculation settings (Off when the model or the build cannot do it) and the row-exact switch.
    spec: SpecConfig,
    spec_why: String,
    row_exact: bool,
    /// The prompt cache on disk (`EngineConfig::prompt_cache`; off with a unified pool).
    pcache: Option<crate::prompt_cache::DiskCache>,
    /// The readers' long reads on the SSD (`--prompt-cache`): f16 states, one directory beside the prompt cache's.
    rcache: Option<crate::prompt_cache::DiskCache>,
    /// Saves to `rcache` still being written (I/O key, tokens): a read of one waits for it ([`LlamaEngine::stash_wait`]).
    rstash: Vec<(u64, Vec<i32>)>,
    handovers: Vec<Handover>,
    /// Shared starts are whole multiples of this (`n_ubatch` in row-exact mode, else 1): the rest of the prompt is then
    /// prefilled in the same chunks as from scratch, so the same kernels see the same rows and a shared or cached start
    /// changes no bit (a start ending mid-chunk leaves a short remainder that takes the few-row path instead).
    share_align: usize,
    /// Lookup state per sequence id.
    spec_state: HashMap<i32, SpecState>,
    spec_stats: SpecStats,
    /// `--kv-read f16`: the f16 reader of prompts (None: prompts are read in the main cache).
    readers: Vec<Reader>,
}

// SAFETY: the engine is built on the executor thread and used only from it; the raw pointers are owned by it. The trait asks
// for `Send` so that the box can be handed to that thread's loop.
unsafe impl Send for LlamaEngine {}

fn model_err(msg: impl Into<String>) -> EngineError {
    EngineError::new(500, "server_error", msg)
}

impl LlamaEngine {
    pub fn arch(&self) -> &str {
        &self.arch
    }

    fn seq_for<'s>(agents: &'s mut HashMap<AgentId, AgentSeq>, job: &'s mut Job) -> Option<&'s mut Seq> {
        match &job.agent {
            Some(a) => agents.get_mut(a).map(|x| &mut x.seq),
            None => job.seq.as_mut(),
        }
    }
}
