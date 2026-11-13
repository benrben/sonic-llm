//! `--engine next`: sllm-next, SLLM's own engine (engine/sllm-next/proto: Metal kernels of its own, no ggml for the model),
//! behind the same [`Engine`] trait as libllama. Every token in flight is a row of one step; a row's result does not depend
//! on the other rows, so an agent gets the same tokens at any agent count. Qwen3 and Gemma 4 GGUFs, greedy decoding.
//!
//! Sequences live in fixed slots of `n_ctx` cells (`--kv-pages` x 256 / `--ctx` of them). Interactive agents' write rows take
//! the one-row kernels and everyone else's the batch kernels; the kind is fixed per agent (its lane), never by the step.
//!
//! - Grammars: libllama's grammar sampler on the vocabulary, as the libllama engine does: the GPU's argmax is checked first and
//!   only a refused row's scores come back to the CPU. `ignore_eos` the same way (an end-of-generation argmax is replaced by the
//!   best other token).
//! - Prefix sharing: a cell depends only on the tokens up to it, so a new sequence starts on a free slot that still holds a
//!   prefix of its tokens (kept from an ended sequence; `cache_prompt: false` skips these), or with the prefix copied from a
//!   live sequence's slot. Either is what a prefill would write, bit for bit.
//! - Compaction and rollover: [`Engine::agent_cut`] keeps the first cells of the slot.
//! - Suspend / resume: the slot's cells go to an SSD file (written on a thread) and the slot is free; a resume reads them
//!   back into any free slot (or takes the old slot when nothing has used it since).
//! - Writers: a step of write rows only (one interactive writer, or many background ones) has the next one queued behind it
//!   (the GPU never waits for the host; 8 background agents +4-5%; not Qwen3.5, whose dropped queued step would have moved
//!   its states). One interactive writer alone also
//!   drafts by prompt lookup ([`crate::spec`], as the libllama engine's `--spec auto`; SLLM_NEXT_SPEC=off turns it off): up
//!   to 3 tokens checked in one step of one-row kernels, each row the bits of a step of its own. Steps queue unless the
//!   next lookup can match; a recurrent model (Qwen3.5 / 3.6) drafts 2 tokens a step with spare state entries as the way
//!   back (sn_draft_rows), its background writers never. Background writers draft too while a step of theirs
//!   (no chat rows) has rows to spare, exactly (2 / 4 / 8 agents after documents +15 / +8 / +5%), and every draft may
//!   come from the shared memory of all tokens the server has seen ([`crate::shared_draft`]; a repeated request 3.1x).
//!
//! Cells past what a slot holds are always zero (cut cells are cleared), so a reused slot is the same as a fresh one.

use crate::engine::*;
use crate::llama_engine::{argmax_excluding, text_of, Gram, GRAMMAR_DONE};
use crate::shared_draft::SharedDraft;
use crate::spec::{SpecConfig, SpecMode, SpecState};
use llama_ref::Vocab;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, CStr, CString};
use std::path::PathBuf;
use std::sync::{mpsc, Arc};
use std::time::Instant;

mod agents;
mod draft;
mod engine_impl;
mod ffi;
mod latent;
mod life;
mod setup;
mod slot_table;
mod slots;
mod start;
mod step;
mod take;

use ffi::*;
use slot_table::{SlotOwner, SlotTable};


const KERNELS: &str = concat!(include_str!("../../sllm-next/proto/kernels.metal"), "\0");
/// Most interactive write rows in a step (the one-row kernels take up to 4).
const MAX_FAST: usize = 4;
/// The shortest prefix worth taking from another slot (a copy is one command buffer; a few tokens are cheaper to read).
const MIN_SHARE: usize = 32;
/// A recurrent model's checkpoint: this many tokens before a prompt's end (a chat's next turn renders the last turn's
/// generation prompt again as the reply's start, so the prompts part a few tokens before the end).
const CK_LAG: usize = 16;
/// The shortest prompt a checkpoint is kept for: it costs a step of its own (Qwen3.6-35B-A3B: a 69-token read 73 -> 103 ms),
/// and a short prompt read again costs little.
const CK_MIN: usize = 256;
/// A recurrent model's pinned starts: the shortest shared start of prompts (a system prompt) worth a state of its own
/// (SLLM_NEXT_PIN_MIN), how many starts are kept (SLLM_NEXT_PINS, 0: off), and how many recent prompts they are found among.
const PIN_MIN: usize = 12;
const PIN_MAX: usize = 6;
const PIN_RECENT: usize = 8;
/// Coalesced prefill: a new sequence whose first tokens another running request is reading now waits for them (and then
/// copies them) when that saves at least this many tokens of reading (SLLM_NEXT_COALESCE=0: off).
const COALESCE_MIN: usize = 128;
/// The longest a busy engine goes without giving memory back (sn_trim) after a busy moment.
const TRIM_EVERY: std::time::Duration = std::time::Duration::from_secs(2);
/// The pool key of [`Engine::kv_pools`]: pages of [`PAGE`] cells.
const POOL: &str = "page";
/// Cells of one page of the library's cache.
const PAGE: usize = 64;

/// Pages that hold `n` cells.
fn chunks(n: usize) -> usize {
    n.div_ceil(PAGE)
}

#[derive(Debug, Clone)]
pub struct NextConfig {
    pub model: PathBuf,
    /// Cells per sequence (the window of one agent).
    pub n_ctx: usize,
    /// Sequence slots (agents and completions running at once).
    pub slots: usize,
    /// Most rows in a step (prompt tokens plus write rows).
    pub max_rows: usize,
    /// The cache: pages of 64 cells, taken as sequences grow and shared by sequences that start the same (0: every slot
    /// its whole window).
    pub pages: usize,
    /// Where suspended agents' cells go (`--spill-dir`, `--agent-dir`/kv); none: a directory of this process's own under
    /// `SLLM_SPILL_DIR` or the temp directory, removed at exit.
    pub spill_dir: Option<PathBuf>,
}

/// A suspended agent: its tokens and where its cells are.
struct Off {
    tokens: Vec<i32>,
    path: PathBuf,
    /// The host copy until the SSD write has finished (a resume before then loads it from here).
    host: Option<Arc<Vec<u8>>>,
    /// The slot a resume is reading into.
    reading: Option<i32>,
    resumed: Option<Instant>,
    /// Which suspend this is (a write that ends after a resume and a new suspend is not this one's).
    gen: u64,
}

struct Agent {
    /// None: suspended, or its KV dropped (a turn then takes a free slot).
    slot: Option<i32>,
    interactive: bool,
    /// Tokens of its context its cells do not hold yet, read with its next turn: tokens a cut far back took from the
    /// sliding-window layers, then the sampled token not fed yet.
    pending: Vec<i32>,
    off: Option<Off>,
}

/// A sampled token's log-probability and the two best tokens' (id, log-probability, best first), from the GPU's 8-word record of its
/// row (`sn_probs`: chosen id, chosen score, best id and score, second id and score, log-sum-exp, 0; ids as ints, scores as float bits).
struct Rec {
    lp: f32,
    top: [(i32, f32); 2],
}

impl Rec {
    /// `chosen`: the score of the token that was taken (the record's own, or another one's the CPU picked).
    fn new(w: &[u32; 8], chosen: f32) -> Rec {
        let (f, lse) = (|i: usize| f32::from_bits(w[i]), f32::from_bits(w[6]));
        Rec { lp: chosen - lse, top: [(w[2] as i32, f(3) - lse), (w[4] as i32, f(5) - lse)] }
    }
}

struct Job {
    agent: Option<AgentId>,
    slot: i32,
    interactive: bool,
    feed: Vec<i32>,
    fed: usize,
    /// Position of the next token to feed.
    pos: usize,
    budget: usize,
    /// The cells it may reach (its pages are reserved up to there).
    end: usize,
    ctx_limited: bool,
    emitted: usize,
    ignore_eos: bool,
    gram: Option<Box<Gram>>,
    last: Option<i32>,
    carry: Vec<u8>,
    /// The agent's context before this turn (a turn cancelled before its first token leaves it so).
    before: Option<(usize, Vec<i32>)>,
    /// Prompt lookup (an interactive writer alone), and its lookups in a row that found nothing.
    spec: SpecState,
    no_match: u32,
    /// Cell index of its first new token (what the shared draft memory has not seen: the cells before it came from earlier requests).
    seen_from: usize,
    /// Latent steps still to run (a latent turn after its prompt), and the vector its next step is fed.
    lat: usize,
    lvec: Vec<f32>,
    /// `n_probs` of its request (0: none): its tokens come with their log-probabilities (at most 2 best tokens beside).
    n_probs: usize,
    /// Feed index where a recurrent job keeps its state (sn_mark): CK_LAG before its prompt's end, or where a waiting
    /// request's prefix ends (coalesced prefill); usize::MAX: none.
    ck_at: usize,
    /// A recurrent job that reads a start other recent prompts share: the feed index where that start ends; the rows up to
    /// it are a step of their own and their state is then copied to a pinned slot (usize::MAX: none).
    pin_at: usize,
    /// Coalesced prefill: the request reading this one's first `n` tokens now (it waits, then copies them).
    wait: Option<(ReqId, usize)>,
}

/// The rows of one step as built for sn_step: token, slot and position per row, the output rows, and which request each
/// output row is for (`groups`: each write's request, its first output row and its draft count; `out_req`: a prompt read to
/// its end and its output row).
#[derive(Default)]
struct StepRows {
    tok: Vec<i32>,
    seq: Vec<i32>,
    pos: Vec<i32>,
    out_rows: Vec<i32>,
    out_req: Vec<(ReqId, usize)>,
    groups: Vec<(ReqId, usize, usize)>,
}

enum Io {
    Written { agent: AgentId, gen: u64, path: PathBuf, result: Result<(u64, f64), String> },
    Read { agent: AgentId, slot: i32, result: Result<Vec<u8>, String> },
}

#[derive(Default)]
struct LifeCounters {
    ssd_bytes: u64,
    writes: u64,
    reads: u64,
    warm_resumes: u64,
    in_flight: usize,
    suspends: u64,
}

pub struct NextEngine {
    e: *mut SnEngine,
    /// Recurrent states (Qwen3.5): cut back only to 0, the end or the checkpoint ([`CK_LAG`]).
    recurrent: bool,
    /// Wide drafting (on by default; SLLM_NEXT_SPEC_WIDE=0 off): up to 7 drafts, a check of 4 rows or more on the block matmuls (background
    /// rows: rows past the first cost little there, 8 rows 1.56x a one-row mat-vec step, against 1.55x for 4 mat-vec rows);
    /// the rows of such a check are no longer the bits of one-row steps, so text with drafts may part from text without on
    /// a near tie
    wide: bool,
    cfg: NextConfig,
    vocab: Arc<Vocab>,
    n_vocab: usize,
    /// Who owns each slot (the free ones, a job's, an agent's, a resume's read, a pin): see [`SlotTable`].
    owners: SlotTable,
    /// The tokens whose cells each slot holds, in position order (also for free slots: a prefix the next sequence may use).
    cells: Vec<Vec<i32>>,
    agents: HashMap<AgentId, Agent>,
    jobs: HashMap<ReqId, Job>,
    counters: EngineCounters,
    prefix: PrefixCounters,
    gpu_ms: f64,
    run_ms: f64,  // host time inside sn_step (encoding, and waiting on the step before when queued)
    /// The next `run` takes the argmax_p kernel (a row of the step wants token probabilities; the flag is cleared by the run), and
    /// `probs`: the 8-word record of each output row of the last run that did (see `sn_probs`; empty after one that did not).
    want_probs: bool,
    probs: Vec<u32>,
    io_tx: mpsc::Sender<Io>,
    io_rx: mpsc::Receiver<Io>,
    events: Vec<LifecycleEvent>,
    life: LifeCounters,
    spill_dir: PathBuf,
    own_spill_dir: bool,
    /// Each slot's last step (the cells kept in free slots go least recently used first when pages run short).
    last_used: Vec<u64>,
    tick: u64,
    pages_peak: usize,
    /// Queue a step of writers only behind the one before (SLLM_NEXT_CHAIN=0: off).
    chain: bool,
    /// The GPU's argmax skips the end-of-generation tokens of an `ignore_eos` row (8 tokens or fewer).
    gpu_eog: bool,
    /// Prompt lookup for one interactive writer (on by default; SLLM_NEXT_SPEC=off), and what it did.
    spec: SpecConfig,
    /// The makers' guess head (a GGUF with the extra block; SLLM_NEXT_GUESS=0 turns it off): a lone interactive writer's
    /// drafts come from it (`guess_k` of them a step, SLLM_NEXT_GUESS_K, 1..3, default 3) in place of prompt lookup.
    guess: bool,
    guess_k: usize,
    guess_steps: u64,
    guess_drafted: u64,
    guess_kept: u64,
    draft_guess: bool,
    spec_steps: u64,
    spec_drafted: u64,
    spec_kept: u64,
    /// Background writers draft too while the step has rows to spare (on by default; SLLM_NEXT_BG_DRAFT=0: off): their drafts are
    /// extra write rows of their own sequence, on the block matmuls (a tile holds 16 rows) and the per-row attention, so
    /// their tokens are the ones they take without drafts. `bg_rows`: the most rows such a step carries (SLLM_NEXT_BG_ROWS,
    /// 16); `bg_k0`: the draft length a background writer starts at (SLLM_NEXT_BG_K0, 2).
    bg_draft: bool,
    /// Coalesced prefill on (SLLM_NEXT_COALESCE=0: off): see [`COALESCE_MIN`].
    coalesce: bool,
    /// Memory given back after a busy moment (SLLM_NEXT_TRIM=1: on; off by default, see `new`): sn_trim when the last request ends, and at most
    /// every TRIM_EVERY while requests run (the cache packed down, the chunks above unmapped); `trim_at`: the last one
    trim: bool,
    trim_at: std::time::Instant,
    /// An idle engine keeps what free slots hold (prefixes for a next request) for `idle_after` (SLLM_NEXT_IDLE_S, 5 s); then
    /// only up to `idle_cache` bytes of cache (SLLM_NEXT_IDLE_CACHE_MB, 1024; the most recently used slots kept): the rest
    /// is dropped, its pages free for the next run instead of new memory (off by default, SLLM_NEXT_IDLE_DROP=1: on), and
    /// with `trim` its memory goes back to the system. `idle_since`: when the last request ended (None: busy, or done).
    idle_drop: bool,
    idle_since: Option<std::time::Instant>,
    idle_after: std::time::Duration,
    idle_cache: usize,
    /// sn_trim's unmaps still running on the GPU: the idle loop polls until their memory is let go
    trim_pending: bool,
    /// A recurrent model: the most state memory the ended sequences in free slots may keep (SLLM_NEXT_KEPT_STATE_MB, 1024). Each
    /// kept sequence holds a state entry (Qwen3.6-35B-A3B: 66 MB; two with a checkpoint), and a new sequence takes an empty slot
    /// before a kept one, so without a cap all 256 slots fill: 17 GB beyond the weights after ~350 requests on a 48 GB Mac
    /// (2026-11-11: swap, one read took 26 s). Above the cap the least recently used kept sequences are dropped ([`Self::cap_kept_states`]).
    kept_state: usize,
    /// A recurrent model's pinned starts (a system prompt read once): slots that are neither free nor a job's, each holding
    /// exactly a start that recent prompts shared, least recently used first. A new sequence copies its state from one
    /// (`place`) instead of reading the start again: an ended sequence can only be cut back to 0, its end or its checkpoint,
    /// so without a pin a new question after the same system prompt reads every token. The first two prompts of a start read it
    /// plainly; the third reads it as a step of its own and pins it; later ones copy.
    pins: Vec<i32>,
    pin_max: usize,
    pin_min: usize,
    recent: std::collections::VecDeque<Vec<i32>>,
    pin_made: u64,
    pin_used: u64,
    pin_tokens: u64,
    bg_rows: usize,
    /// sn_small_rows: a step of at most this many background writers stays within it with its drafts (their tokens are the
    /// same whenever at most this many write; a draft row past it would move the step to the block matmuls)
    small: usize,
    bg_k0: usize,
    bg_steps: u64,
    bg_drafted: u64,
    bg_kept: u64,
    /// Shared draft memory (on by default, SLLM_NEXT_SHARED_DRAFT=0: off; SLLM_NEXT_SHARED_TOKENS ring size, default 4M; SLLM_NEXT_SHARED_N n-gram, default 4):
    /// all tokens this server has seen, a second source of drafts beside the writer's own context. The drafts it gave, and the
    /// last draft's source.
    shared: Option<SharedDraft>,
    shared_steps: u64,
    shared_drafted: u64,
    shared_kept: u64,
    draft_shared: bool,
    /// Latent turns (`latent_steps`): the token table's mean row norm (0: not measured yet), the residual width, the next
    /// silent cell's marker in `cells` (negative, never a token: no prefix or draft ever matches it), the silent steps run.
    emb_norm: f32,
    n_embd: usize,
    /// LatentMAS's realign matrix W_a (n_embd x n_embd, row-major: aligned = h W_a), from SLLM_LATENT_REALIGN; empty: identity
    realign: Vec<f32>,
    /// soft words (SLLM_LATENT_SOFT="T,K"): a silent step is fed the mix of its K best words' input rows, weighted by
    /// softmax(score / T), instead of its rescaled hidden state
    soft: Option<(f32, usize)>,
    /// the soft mix made on the GPU (SLLM_LATENT_SOFT_GPU=0: on the CPU): a silent row's vector never leaves the GPU, and silent
    /// steps are queued ahead like written ones
    soft_gpu: bool,
    /// SLLM_LATENT_STOP="id,id": a silent turn ends early when its step's best word is one of these (or an end of turn):
    /// the model decides it has thought enough, as when it writes the end of its thinking
    lat_stop: Vec<i32>,
    lat_mark: i32,
    lat_rows: u64,
}

// The executor owns the engine on one thread; the raw pointers (the library, the grammar samplers) are only used there.
unsafe impl Send for NextEngine {}

impl Drop for NextEngine {
    fn drop(&mut self) {
        self.jobs.clear(); // the grammar samplers
        for o in self.agents.values().filter_map(|a| a.off.as_ref()) {
            let _ = std::fs::remove_file(&o.path);
        }
        if self.own_spill_dir {
            let _ = std::fs::remove_dir(&self.spill_dir); // only if empty: a write still running leaves its file
        }
        unsafe { sn_close(self.e) };
    }
}

fn err(status: u16, kind: &'static str, msg: impl Into<String>) -> EngineError {
    EngineError::new(status, kind, msg)
}

fn server_err(m: String) -> EngineError {
    err(500, "server_error", m)
}

fn lcp(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Whether sllm-next runs this GGUF (its header and tensor table only): Ok with the prompt rows a step should take (512 or
/// 1024, sn.h's sn_check), or the reason it does not.
pub fn check(model: &std::path::Path) -> Result<usize, String> {
    let path = CString::new(model.display().to_string()).map_err(|e| e.to_string())?;
    let mut rows: c_int = 1024;
    call(|b, n| unsafe { sn_check(path.as_ptr(), &mut rows, b, n) })?;
    Ok(rows.max(64) as usize)
}

/// The prompt rows a step for `model` while no chat is about (sn.h's sn_quiet_rows), None: its usual rows.
pub fn quiet_rows(model: &std::path::Path) -> Option<usize> {
    let path = CString::new(model.display().to_string()).ok()?;
    let r = unsafe { sn_quiet_rows(path.as_ptr()) };
    (r > 0).then_some(r as usize)
}

