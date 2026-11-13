//! The scheduler's settings and their defaults ([`super`] re-exports every name here).

use crate::engine::Lane;
use serde::Serialize;

/// The scheduler's settings (sllm-server flags; the task 11 sweeps pick the values).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SchedConfig {
    /// Running requests per lane (interactive, background, helper). The interactive lane holds at most 4.
    pub caps: [usize; 3],
    /// Admitted requests waiting for a slot, per lane. Past it a request is deferred: 503 with `retry_after`.
    pub queue_limits: [usize; 3],
    /// Largest prefill chunk of one request in one step.
    pub chunk: usize,
    /// A bigger chunk and step budget while no interactive request has run for `quiet_after_ms` (None: never). A MoE reads
    /// faster in bigger steps, but a chat that arrives waits for the step under way, so only while no chat is about.
    pub quiet_chunk: Option<usize>,
    pub quiet_after_ms: u64,
    /// Prefill tokens per step over all requests (decode rows come on top).
    pub step_tokens: usize,
    /// Of `step_tokens`, what a prefilling helper request is guaranteed (the reserved helper slot).
    pub helper_tokens: usize,
    /// Background prefill tokens per step while an interactive request decodes.
    pub bg_prefill_interactive: usize,
    /// After this many consecutive deferred steps one full background chunk goes through anyway (0 = never).
    pub bg_prefill_period: usize,
    /// Background decode rows per step while an interactive request decodes (default
    /// [`DEFAULT_BG_ROWS_INTERACTIVE`]; `None` = all of them).
    pub bg_rows_interactive: Option<usize>,
    /// The average gap between an interactive request's tokens that background decode rows must leave it, in ms
    /// (default [`DEFAULT_CHAT_GAP_MS`]; 0 = they ride along in every step).
    pub chat_gap_ms: f64,
    /// Split a lane's prefill budget fairly over its prefilling requests (true), or give it out in arrival order.
    pub prefill_fair: bool,
    /// The smallest fair share of one request in one step (default [`MIN_FAIR_CHUNK`]).
    pub fair_min: usize,
    /// Most prefill chunks a fair step carries next to background decode rows (default [`MAX_MIXED_CHUNKS`]).
    pub max_mixed_chunks: usize,
    /// Burst prefill-first: background decode rows sit out the background prefill steps while at least this many
    /// background requests are prefilling (0 = off, the default).
    pub burst_min: usize,
    /// Burst yield (phase 3 T4): while at least this many background requests are prefilling, the interactive guard is
    /// lifted for prefill: background prefill gets the whole step budget and background decode rows sit out (as with
    /// `burst_min`), until the burst's prefill is done. The interactive row still decodes in every step (0 = off).
    pub burst_yield: usize,
    /// Most background prefill tokens of the chunk forced through while an interactive request decodes (`None` =
    /// `chunk`).
    pub bg_prefill_forced: Option<usize>,
    /// Defer background prefill while an interactive request is prefilling too, not only while it decodes.
    pub front_prefill_solo: bool,
    /// Fair shares in whole multiples of `fair_min` (a share that finishes its prompt excepted); the rest of the budget
    /// goes to the earliest requests, a `fair_min` at a time.
    pub fair_granule: bool,
    /// The share of the time background writers keep while background prompts are read beside them (0 = off: every step
    /// carries the whole read budget and each writer one token, so the writers all but stop while anything is read).
    pub bg_write_share: f64,
    /// The longest read step beside background writers, in ms (0 = no limit): such a step reads only what fits.
    pub bg_gap_ms: f64,
}

/// Interactive slots, default and most: every sequence id the engine holds (Ben 2026-11-09: no own cap; was 4). The
/// engine still holds at most 256 sequences, so a request past that waits in the queue.
pub const MAX_INTERACTIVE: usize = 256;

/// Decode rows of the interactive and background lanes together that the engines are sized for: the 256 sequence ids
/// plus the 4 rows of headroom the old interactive cap of 4 gave (so the memory plan stays what it was).
pub const LANE_ROWS: usize = 260;

/// The smallest fair prefill share of one request in one step: below it the budget goes to the earliest requests.
/// 256 (phase 2 task 11): many small prefill groups run several times slower than a few large ones (dev, Qwen3-0.6B
/// agents-32: a step of 32 chunks of 64 tokens took 1.2 to 1.4 s of compute, one of 8 chunks of 256 about 0.3 s), and
/// shares of 64 tokens flipped a near tie in the coalesced-arrival exactness test (496 of 512 positions; 512 with 256).
pub const MIN_FAIR_CHUNK: usize = 256;

/// Most prefill chunks a step carries next to background decode rows (fair mode): a step with more leaves them out.
pub const MAX_MIXED_CHUNKS: usize = 8;

/// Burst prefill-first: most consecutive steps the background decode rows sit out; then they ride along once.
pub const BURST_MAX_SKIP: usize = 16;

/// The batch-mode `burst_yield` (`--burst-yield 8`; off by default, Ben 2026-10-27): 8 background requests prefilling together. Batch-32 on Qwen3-0.6B (dev): the front agent's
/// 128 decode steps ran alone (2.7 ms, no prefill) while 31 prompts waited, 0.36 s of a 3.4 s pass; with the burst
/// prefilled first the decode window shrank by about 8 to 9% and TTFT p50 fell from 910 to 550 ms, at the price of the
/// front agent's tokens during the burst's prefill steps (each about 35 to 140 ms), so it is off unless asked for.
/// G3 gate, GPU to itself: T4 Qwen 1,292 vs llama 908 tok/s (1.42x; 1.30x without it).
pub const BURST_YIELD_MIN: usize = 8;

/// The default step budget: four 512-token chunks, or 8 fair shares of [`MIN_FAIR_CHUNK`] (phase 2 srvbatch: 512 = one
/// chunk let only one agent prompt into a step, so agents arriving together started decoding one prefill step apart).
/// Task 11: 4,096 was about 7% faster on Qwen3-0.6B agents-32 (dev), but the compute buffer scales with the budget and
/// agents-64 then no longer fit the memory budget; `--step-tokens` overrides it.
pub const DEFAULT_STEP_TOKENS: usize = 2048;

/// The default `bg_prefill_period`: one forced background chunk per 256 deferred steps. A 512-token chunk costs about
/// 30 Qwen3-0.6B or 15 Gemma-3-4B decode steps, so a period of 256 keeps the forced chunks under about 12% of the
/// interactive request's decode time (task 11; 32 cost the front agent about half its speed: the forced chunks, and
/// the background rows that then ride along).
pub const DEFAULT_BG_PREFILL_PERIOD: usize = 256;

/// The default background slots (was 64): 256 agents sent at once all start (the engine holds at most 256 sequence ids, and
/// with fewer streams, say a long `--ctx`, a request waits in the queue for an id instead of failing). Interactive requests
/// keep their own lane, ahead of the background rows in every step.
pub const DEFAULT_BACKGROUND_CAP: usize = 256;

/// The default cap on background decode rows while an interactive request decodes (was all of them). Chat beside writing
/// agents, median of 3: 18 -> 93 tok/s beside 244 on Qwen3-0.6B, 6 -> 25 beside 200 on Qwen3-4B, 7 -> 26 beside 200 on
/// Gemma-3-4B. The agents pay while the chat writes: 15.6 -> 16.7 s on that Gemma run, about 25% if a chat lasts as long
/// as they do. Fewer agents than this are not touched. A value at or above the background slots turns the cap off.
pub const DEFAULT_BG_ROWS_INTERACTIVE: usize = 32;

/// Background writers beside background readers (2026-11-13). Before: 4 writers with a 2,200-token prompt arriving every 3 s
/// wrote 32% slower on the MoE with gaps of 0.9 s, and on Qwen3.8-27B 95% slower (12.3 -> 0.6 tok/s, gaps of 1.7 to 4 s),
/// because a step read up to 2,048 tokens and gave each writer one token. Half of the time is the writers', and a read
/// step lasts about a quarter of a second. This holds beside readers that arrive while the writers write; readers of the
/// writers' own burst keep whole read steps (`Planner::writer_terms`).
pub const DEFAULT_BG_WRITE_SHARE: f64 = 0.5;
pub const DEFAULT_BG_GAP_MS: f64 = 250.0;

impl Default for SchedConfig {
    fn default() -> Self {
        SchedConfig {
            caps: [MAX_INTERACTIVE, DEFAULT_BACKGROUND_CAP, 1],
            queue_limits: [16, 64, 4],
            chunk: 512,
            quiet_chunk: None,
            quiet_after_ms: 20_000,
            step_tokens: DEFAULT_STEP_TOKENS,
            helper_tokens: 128,
            bg_prefill_interactive: 0,
            bg_prefill_period: DEFAULT_BG_PREFILL_PERIOD,
            bg_rows_interactive: Some(DEFAULT_BG_ROWS_INTERACTIVE),
            chat_gap_ms: DEFAULT_CHAT_GAP_MS,
            prefill_fair: true,
            fair_min: MIN_FAIR_CHUNK,
            max_mixed_chunks: MAX_MIXED_CHUNKS,
            burst_min: 0,
            burst_yield: 0,
            bg_prefill_forced: None,
            // phase 2 burst A/B (dev, agents-32, quiet pairs against 71b9686): these two cut the front agent's time to
            // first token 370 -> 224 ms (Qwen3-0.6B) and 1,657 -> 1,027 ms (Gemma-3-4B), TTFT p50 -10%, e2e unchanged;
            // greedy tokens identical on both models. burst_min stays off: +1.4% / +5.4% e2e on agents-32 with it, but
            // its cost to background inter-token gaps under open-loop arrivals is not measured yet
            front_prefill_solo: true,
            fair_granule: true,
            bg_write_share: DEFAULT_BG_WRITE_SHARE,
            bg_gap_ms: DEFAULT_BG_GAP_MS,
        }
    }
}

impl SchedConfig {
    pub fn cap(&self, lane: Lane) -> usize {
        self.caps[lane.index()]
    }

    pub fn queue_limit(&self, lane: Lane) -> usize {
        self.queue_limits[lane.index()]
    }

    /// Most decode rows a step can carry: every slot of every lane.
    pub fn max_rows(&self) -> usize {
        (self.caps[0] + self.caps[1]).min(LANE_ROWS) + self.caps[2]
    }

    pub fn validate(&self) -> Result<(), String> {
        self.validate_lanes()?;
        self.validate_budget()?;
        self.validate_writers()
    }

    /// Every lane has a slot, and the interactive lane at most [`MAX_INTERACTIVE`].
    fn validate_lanes(&self) -> Result<(), String> {
        if self.caps[0] == 0 || self.caps[0] > MAX_INTERACTIVE {
            return Err(format!("the interactive lane holds 1 to {MAX_INTERACTIVE} requests, got {}", self.caps[0]));
        }
        if self.caps[1] == 0 || self.caps[2] == 0 {
            return Err("the background and helper lanes need at least one slot".into());
        }
        Ok(())
    }

    /// The chunk and the step budget are positive, and the helper's reserve fits the budget.
    fn validate_budget(&self) -> Result<(), String> {
        if self.chunk == 0 || self.step_tokens == 0 {
            return Err("--chunk and --step-tokens must be positive".into());
        }
        if self.bg_prefill_forced == Some(0) {
            return Err("--bg-prefill-forced must be positive".into());
        }
        if self.helper_tokens > self.step_tokens {
            return Err(format!("--helper-tokens {} is more than --step-tokens {}", self.helper_tokens, self.step_tokens));
        }
        Ok(())
    }

    /// The background writers' share and gap are in range.
    fn validate_writers(&self) -> Result<(), String> {
        // a share of 1 or more would leave the readers nothing; NaN compares false to everything, so test the valid range
        if !(self.bg_write_share >= 0.0 && self.bg_write_share < 1.0) {
            return Err(format!("--bg-write-share is the writers' part of the time, 0 (off) up to but not 1, got {}", self.bg_write_share));
        }
        if !(self.bg_gap_ms >= 0.0) {
            return Err(format!("--bg-gap-ms must be 0 (no limit) or more, got {}", self.bg_gap_ms));
        }
        Ok(())
    }
}

/// 25 tokens a second: quicker than anyone reads.
pub const DEFAULT_CHAT_GAP_MS: f64 = 40.0;
