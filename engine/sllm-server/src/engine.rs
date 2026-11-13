//! The `Engine` trait: what the executor thread drives. Task 5a implements it with [`crate::fake::FakeEngine`];
//! [`crate::llama_engine::LlamaEngine`] runs real models through libllama.
//!
//! The executor owns the engine on ONE thread (D029: every Metal call from one thread), so the trait is plain `&mut self`
//! and synchronous. Requests are admitted with [`Engine::start_request`] (validation and the worst-case page
//! reservation; nothing runs yet), started with [`Engine::try_begin`] when the scheduler gives them a lane slot, and
//! progress in [`Engine::step`] calls. The scheduler (phase 2, `crate::sched`) decides what each step carries: a list
//! of [`WorkItem`]s, one per running request, each a prefill chunk or one decode token. Every item that completes a
//! prefill or decodes produces that request's next token in the same step.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Cells of one notional KV page: the unit of `--kv-pages`, of the helper's pool, of the fake engine's page model and of
/// the generic engine's padding.
pub const PAGE_CELLS: usize = 256;

/// Id of one generation request (assigned by the HTTP side).
pub type ReqId = u64;
/// Id of an agent: `ag_<n>`.
pub type AgentId = String;

/// The scheduling lane of an agent or a request (phase 2 task 2). Interactive: the user is waiting, at most 4 running,
/// its rows go first in every step. Background: the self-managing agents, up to the measured compute ceiling. Helper:
/// the reserved helper slot (rule 073; compaction and the like), its own queue and a reserved share of each step's
/// token budget, so it never waits behind the background workers it serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lane {
    Interactive,
    Background,
    Helper,
}

impl Lane {
    pub const ALL: [Lane; 3] = [Lane::Interactive, Lane::Background, Lane::Helper];

    pub fn index(self) -> usize {
        match self {
            Lane::Interactive => 0,
            Lane::Background => 1,
            Lane::Helper => 2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Lane::Interactive => "interactive",
            Lane::Background => "background",
            Lane::Helper => "helper",
        }
    }
}

/// What a step does for one running request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Work {
    /// Feed the next `n` tokens of the request's prefill (its feed, [`Started::feed_n`]). The chunk that ends the
    /// feed also samples the first token.
    Prefill { n: usize },
    /// Feed the last sampled token and sample the next one.
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct WorkItem {
    pub req: ReqId,
    pub work: Work,
}

/// Decoding parameters. Phase 1 is greedy: only `temperature`, `n_predict`, `ignore_eos` and `grammar` have an effect.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    /// Tokens to generate; negative means "until EOS or the context is full".
    pub n_predict: i64,
    pub temperature: f32,
    /// When true the end-of-generation token is never produced, so a request emits exactly `n_predict` tokens.
    pub ignore_eos: bool,
    /// llama-server's `cache_prompt` (default true): false keeps the request off the prompt cache on disk (a cold run
    /// neither loads nor saves a start there; sharing with live sequences is unchanged).
    pub cache_prompt: bool,
    /// GBNF text (llama.cpp's grammar syntax, root rule `root`) that every token of this request must follow; `None`
    /// (also an empty string) is unconstrained. An engine that cannot build it refuses the request with a 400.
    pub grammar: Option<String>,
    /// Latent steps (LatentMAS): after the prompt the turn writes no text; it feeds its own last hidden state back as the
    /// next input this many times, leaving that many silent cells in the agent's context for its next turns, and ends
    /// with no tokens. 0: a normal turn. Engines without it refuse a request that asks for it.
    pub latent_steps: usize,
    /// llama-server's `n_probs` on `/completion` (the next engine only): each generated token comes with its log-probability and
    /// this many of the best tokens' (0: none; 1 or 2: values above 2 are served as 2). Other engines are never given a non-zero value.
    pub n_probs: usize,
}

#[derive(Debug, Clone)]
pub struct StartRequest {
    pub id: ReqId,
    /// Present for an agent turn: `ids` are appended to the agent's context and generation continues from there.
    pub agent: Option<AgentId>,
    /// Token ids to evaluate (the whole prompt, or the turn's new ids).
    pub ids: Vec<i32>,
    pub params: SamplingParams,
    /// The scheduling lane (informational for the engine: page types follow the agent's role, not the request's lane).
    pub lane: Lane,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum StopReason {
    /// `n_predict` tokens were generated (llama-server `stop_type` "limit").
    Limit,
    /// The model produced its end-of-generation token ("eos").
    Eos,
    /// The context window is full (reported as "limit" on the wire, like llama-server).
    Context,
    /// Cancelled by the client, the cancel endpoint, or a delete.
    Cancelled,
    /// The request's `deadline_ms` passed before it finished (phase 2 task 8).
    Deadline,
}

/// What `start_request` reports on success.
#[derive(Debug, Clone, PartialEq)]
pub struct Started {
    /// Tokens evaluated for this request (prompt length, or the turn's new ids).
    pub prompt_n: usize,
    /// Tokens the engine prefills before the first token: `prompt_n`, plus the agent's pending sampled token when it
    /// has one (the real engine). The scheduler cuts this into [`Work::Prefill`] chunks.
    pub feed_n: usize,
}

/// Where a running request's prefill stands with respect to prefix sharing ([`Engine::prefix_state`], phase 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase", tag = "state")]
pub enum PrefixState {
    /// Run the prefill: `feed_n` feed tokens are left (the `shared_n` prompt tokens are already in the cache as shared
    /// pages, so they are not computed again).
    Ready { feed_n: usize, shared_n: usize },
    /// Waiting on pages (coalesced prefill): another request is prefilling the first `tokens` tokens of this prompt
    /// and has committed `written` of them. The request keeps its lane slot but is not planned until it attaches.
    Waiting { tokens: usize, written: usize },
}

/// One token with its log-probability (the log-softmax of the raw scores at temperature 1): `token` is its text piece (for the
/// sampled token the text the request streams for it) and `bytes` that text's bytes.
#[derive(Debug, Clone)]
pub struct TokProb {
    pub id: i32,
    pub token: String,
    pub bytes: Vec<u8>,
    pub logprob: f32,
}

/// What `n_probs` asks for with a token: the sampled token's log-probability, and the best tokens' (`top`, best first, at most
/// `n_probs` of them).
#[derive(Debug, Clone)]
pub struct TokProbs {
    pub chosen: TokProb,
    pub top: Vec<TokProb>,
}

#[derive(Debug, Clone)]
pub enum EngineEvent {
    Token { req: ReqId, id: i32, text: String, probs: Option<Box<TokProbs>> },
    Finished { req: ReqId, reason: StopReason },
    Failed { req: ReqId, message: String },
}

/// Why the engine refused a call. `status` is the HTTP status the API maps it to.
#[derive(Debug, Clone)]
pub struct EngineError {
    pub status: u16,
    pub kind: &'static str,
    pub message: String,
    /// For a refusal on memory grounds: the bytes missing (the typed part of the 503, next to the message).
    pub shortfall_bytes: Option<u64>,
    /// Backpressure (phase 2 task 7): when to try again. The HTTP layer adds a `Retry-After` header (seconds, rounded up).
    pub retry_after_ms: Option<u64>,
    /// The structured deferred-work signal for the agent layer (lane, running, queued, limit), next to the message.
    pub deferred: Option<Value>,
    /// For `exceed_context_size_error`: the request's tokens and the context size, which llama-server adds to the
    /// error object as `n_prompt_tokens` and `n_ctx`.
    pub context: Option<(usize, usize)>,
}

impl EngineError {
    pub fn new(status: u16, kind: &'static str, message: impl Into<String>) -> Self {
        EngineError { status, kind, message: message.into(), shortfall_bytes: None, retry_after_ms: None, deferred: None, context: None }
    }

    pub fn with_shortfall(mut self, bytes: u64) -> Self {
        self.shortfall_bytes = Some(bytes);
        self
    }

    /// An `exceed_context_size_error` of `n_prompt_tokens` against a context of `n_ctx` cells (llama-server's fields).
    pub fn with_context(mut self, n_prompt_tokens: usize, n_ctx: usize) -> Self {
        self.context = Some((n_prompt_tokens, n_ctx));
        self
    }

    pub fn with_retry_after(mut self, ms: u64) -> Self {
        self.retry_after_ms = Some(ms);
        self
    }
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({}): {}", self.kind, self.status, self.message)
    }
}
impl std::error::Error for EngineError {}

/// Static facts about the engine, read once before it moves onto the executor thread.
#[derive(Debug, Clone, Serialize)]
pub struct EngineInfo {
    pub name: String,
    /// What the `model` field of responses says (the GGUF path, or "fake").
    pub model: String,
    pub n_vocab: usize,
    pub n_ctx: usize,
    pub eos_id: i32,
}

/// One sequence's share of the KV cache, for /metrics.
#[derive(Debug, Clone, Serialize)]
pub struct KvUse {
    /// The agent id, or `"completions"` for the lane that serves `/completion` requests.
    pub id: String,
    /// Tokens whose K/V are stored (plus the one sampled token still waiting to be fed).
    pub cells: usize,
    /// Pool pages held; `None` for a cache that is not paged.
    pub pages: Option<usize>,
    /// The type of those pages (f16, q8_0, q4_0: the KV policy's choice for the sequence's role); `None` if not paged.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_type: Option<&'static str>,
}

/// Engine-side counters for /metrics. The page figures are filled by an engine with a page pool (null for the fake engine
/// and the generic engine, whose unified cache has no pages); `scratch_bytes` is reserved for an engine that measures it.
#[derive(Debug, Clone, Default, Serialize)]
pub struct EngineCounters {
    /// Engine steps: one prefill-plus-first-token or one decode token.
    pub steps: u64,
    pub tokens_generated: u64,
    pub prompt_tokens_evaluated: u64,
    /// Pages in use / in the pool (summed over page types).
    pub pages_active: Option<u64>,
    pub pages_reserved: Option<u64>,
    /// The most pages in use at the end of any step so far.
    pub pages_peak: Option<u64>,
    pub scratch_bytes: Option<u64>,
    /// The process footprint when the engine published (/metrics reads it live; see `Shared::metrics_json`).
    pub footprint_bytes: Option<u64>,
    /// The KV cache kind, e.g. `paged-f16`.
    pub kv_mode: Option<String>,
    pub kv: Vec<KvUse>,
    /// Prefix sharing (phase 2): the `/metrics` `prefix` object (null for an engine without it).
    pub prefix: Option<PrefixCounters>,
    /// Speculative decoding (`--spec`): the mode and what it did so far (null for an engine without it).
    pub spec: Option<serde_json::Value>,
    /// `--kv-read f16`: what the f16 reader did so far (null without it).
    pub kv_read: Option<serde_json::Value>,
}

/// Prefix-sharing counters for `/metrics` (summed over the batched executors).
#[derive(Debug, Clone, Default, Serialize)]
pub struct PrefixCounters {
    /// `--prefix-sharing on|off`.
    pub sharing: bool,
    /// Sequences that attached shared pages, and the prompt tokens they did not prefill.
    pub attached_agents: usize,
    pub attached_tokens: usize,
    /// Sequences that waited on another request's prefill (coalesced prefill).
    pub waited_agents: usize,
    /// Copy-on-write page copies.
    pub cow_copies: usize,
    /// Requests waiting on pages right now.
    pub waiting: usize,
    /// Device bytes of the KV pages in use, a shared page counted once.
    pub kv_bytes_used: u64,
    /// What the same sequences would hold without sharing (a shared page counted once per sequence).
    pub kv_bytes_unshared: u64,
    /// Pages in use, a shared page counted once, and page-table entries (a shared page once per sequence).
    pub pages_used: u64,
    pub page_refs: u64,
}

/// Agent lifecycle (phase 3, `crate::lifecycle`): what a suspend freed.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SuspendReport {
    /// Pool pages freed now (the agent's private pages; shared prefix pages stay resident).
    pub pages_freed: usize,
    pub bytes_freed: u64,
    /// Host copy of the private pages, written to the SSD snapshot in the background.
    pub stage_ms: f64,
}

/// A finished background step of the lifecycle ([`Engine::lifecycle_poll`]).
#[derive(Debug, Clone)]
pub enum LifecycleEvent {
    /// A suspended agent's snapshot write finished: `Ok((bytes written, private bytes, write ms, full rewrite))`. On an
    /// error the agent's KV is gone: the engine has made it an empty agent (its history is what is left).
    Written { agent: AgentId, result: Result<(u64, u64, f64, bool), String> },
    /// A restore finished: `Ok(ms from the resume call to the KV being back in the pool)`. On an error (the snapshot is
    /// missing, stale or damaged) the engine has made it an empty agent: re-prefill from its history.
    Restored { agent: AgentId, result: Result<f64, String> },
    /// The engine suspended an idle agent by itself to make room (memory pressure); its snapshot write follows as a
    /// [`LifecycleEvent::Written`]. The executor moves the agent to its suspended state.
    Suspended { agent: AgentId },
    /// The engine dropped an agent's KV (a turn stopped while paused for memory left it incomplete): history only.
    KvLost { agent: AgentId },
}

/// One page pool's room for the lifecycle's memory-pressure policy ([`Engine::kv_pools`]).
#[derive(Debug, Clone, Serialize)]
pub struct KvPoolRoom {
    /// The page type's name: agents with [`Engine::agent_kv`] of the same key use this pool.
    pub key: &'static str,
    pub total_pages: usize,
    /// Pages neither in use nor promised to an admitted request.
    pub free_pages: usize,
    pub page_bytes: u64,
}

fn unsupported(what: &str) -> EngineError {
    EngineError::new(501, "not_supported_error", format!("this engine does not support {what}"))
}

pub trait Engine: Send + 'static {
    fn info(&self) -> EngineInfo;
    /// The `/caps` document: engine name plus the task 1 capability record where the engine has probed it.
    fn caps(&self) -> Value;
    /// Admit a request: validate it and reserve its worst case (pages), so a queued request is counted against the
    /// memory budget before it runs. Errors refuse it before any token is produced (unknown agent, context overflow, a
    /// pool that cannot hold it, ...). Nothing runs until [`Engine::try_begin`].
    fn start_request(&mut self, req: StartRequest) -> Result<Started, EngineError>;
    /// The scheduler gives an admitted request a lane slot. An engine that cannot run it yet (the contiguous cache has
    /// one completions sequence) says false and the request stays queued. Called once per request until it says true.
    fn try_begin(&mut self, _req: ReqId) -> bool {
        true
    }
    /// Prefix sharing (phase 2): asked once a request has begun, and again at every step boundary while it waits on
    /// pages. `Ok(None)`: no sharing, run the feed as admitted (the default). `Ready`: the feed left to prefill (shared
    /// tokens skipped). `Waiting`: hold the slot and do not plan the request; cancellation and deadlines still apply.
    /// An error (the pages it counted on are gone and the pool cannot hold a full prefill) fails the request.
    fn prefix_state(&mut self, _req: ReqId) -> Result<Option<PrefixState>, EngineError> {
        Ok(None)
    }
    /// Stop a request, queued or running, and release what it holds (a completion's pages go back to the pool now). A
    /// turn cancelled before its first token leaves the agent's context as it was before the turn. Its
    /// [`EngineEvent::Finished`] is NOT emitted by the engine: the executor reports the stop itself. Unknown ids are ignored.
    fn cancel(&mut self, req: ReqId);
    /// Run one step over `items` (one per running request, each request at most once). Blocks for as long as it takes.
    fn step(&mut self, items: &[WorkItem]) -> Vec<EngineEvent>;
    fn agent_create(&mut self, id: &str, lane: Lane, system_ids: &[i32]) -> Result<(), EngineError>;
    /// A new agent `id` whose context is a copy of idle agent `from`'s (its cells shared, not read again: silent cells
    /// included). Engines without it refuse.
    fn agent_fork(&mut self, id: &str, lane: Lane, from: &str) -> Result<(), EngineError> {
        let _ = (id, lane, from);
        Err(EngineError::new(501, "not_supported_error", "forking an agent needs --engine next"))
    }
    /// Free an agent and its context. Unknown ids are ignored. Any request on it has been cancelled before this call.
    fn agent_free(&mut self, id: &str);
    /// Tokens in the agent's context (system prompt, turns and generated tokens), if it exists.
    fn agent_context_tokens(&self, id: &str) -> Option<usize>;
    /// Called when no step can run: background work that has ended and that a waiting request may be behind.
    fn idle(&mut self) -> Vec<EngineEvent> {
        Vec::new()
    }
    fn counters(&self) -> EngineCounters;
    /// `(prompt_ms, predicted_ms)` the engine measured for a finished or cancelled request: from the start of its prefill
    /// to its first token, and from there to its end. Taken once. An engine that queues requests returns these so the time
    /// a request waited behind another is not counted as its prefill; the default (`None`) leaves the executor's own clock.
    fn take_timings(&mut self, _req: ReqId) -> Option<(f64, f64)> {
        None
    }
    // ---- agent lifecycle (phase 3; `crate::lifecycle` drives these) --------------------------------------------

    /// Suspend an idle agent (no request on it): its private KV pages go to an SSD snapshot (written in the background)
    /// and are freed now. Its context length is kept; turns are refused until [`Engine::agent_resume`] has finished.
    fn agent_suspend(&mut self, _id: &str) -> Result<SuspendReport, EngineError> {
        Err(unsupported("suspending agents"))
    }
    /// Start restoring a suspended agent (pages are allocated now, the read runs in the background). Its
    /// [`LifecycleEvent::Restored`] comes from [`Engine::lifecycle_poll`].
    fn agent_resume(&mut self, _id: &str) -> Result<(), EngineError> {
        Err(unsupported("resuming agents"))
    }
    /// Drop the agent's KV, resident or suspended (history only): it stays as an empty agent (context 0).
    fn agent_drop_kv(&mut self, _id: &str) -> Result<(), EngineError> {
        Err(unsupported("dropping an agent's KV"))
    }
    /// Background lifecycle work that finished since the last call. Uploads restored pages (at most about
    /// `upload_budget` bytes per call, so a burst of wakes does not stall the step loop).
    fn lifecycle_poll(&mut self, _upload_budget: u64) -> Vec<LifecycleEvent> {
        Vec::new()
    }
    /// Background lifecycle work is in flight (the executor polls instead of blocking on its commands).
    fn lifecycle_busy(&self) -> bool {
        false
    }
    /// An idle engine's next due work (`idle` runs it): how long until then. The executor polls instead of blocking while
    /// this is Some.
    fn idle_wake(&self) -> Option<std::time::Duration> {
        None
    }
    /// The page pools' room, for the memory-pressure policy (empty: no pool).
    fn kv_pools(&self) -> Vec<KvPoolRoom> {
        Vec::new()
    }
    /// The pool key and the private pages (what a suspend would free) of a resident agent.
    fn agent_kv(&self, _id: &str) -> Option<(&'static str, usize)> {
        None
    }
    /// The `/metrics` `lifecycle.engine` object: SSD bytes, writes, reads.
    fn lifecycle_counters(&self) -> Value {
        Value::Null
    }

    /// Where the last [`Engine::step`] spent its time (phase 2 task 12, the loop profile): `None` for an engine that does
    /// not time its steps (the fake engine).
    fn last_step_breakdown(&self) -> Option<StepBreakdown> {
        None
    }
    /// Compaction (phase 3, [`crate::compaction`]): the agent's context as token ids, in position order (what its cache
    /// holds, then the sampled token not fed yet), if the engine keeps them.
    fn agent_tokens(&self, _id: &str) -> Option<Vec<i32>> {
        None
    }
    /// Compaction: cut the agent back to the first `keep` tokens of its context (its prefix, whose pages stay as they
    /// are) and free what held the rest; the pending sampled token goes too. The caller then prefills the summary and
    /// the tail as a request of the agent. No request of the agent may be admitted.
    fn agent_cut(&mut self, _id: &str, _keep: usize) -> Result<CutStats, EngineError> {
        Err(EngineError::new(501, "not_supported_error", "this engine does not support compaction"))
    }
}

/// What a compaction cut did to an agent's pages ([`Engine::agent_cut`]); `None` without a page pool.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct CutStats {
    pub pages_before: Option<usize>,
    pub pages_after: Option<usize>,
    /// Pages back on the free list.
    pub pages_freed: Option<usize>,
    /// Prefix pages kept, and how many of them other agents share.
    pub prefix_pages: Option<usize>,
    pub prefix_shared: Option<usize>,
}

/// One engine step's time split, summed over the `llama_decode` calls the step ran. Host time is everything but `compute_us` (submission to completion).
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct StepBreakdown {
    pub graphs: u32,
    pub reused: u32,
    pub build_us: u64,
    pub alloc_us: u64,
    pub set_inputs_us: u64,
    pub compute_us: u64,
    pub sample_us: u64,
    pub total_us: u64,
    pub n_decode_rows: usize,
    pub n_prefill_tokens: usize,
}
