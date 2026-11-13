//! The step policy of the continuous-batching scheduler (phase 2 tasks 2, 7, 8 and 9; design: "Scheduler policy").
//!
//! The executor thread ([`crate::executor`]) owns the lanes, the queues and admission; this module is the pure part:
//! given the running requests, which work goes into the next step. It is kept free of I/O so the policy can be tested
//! on its own.
//!
//! A step carries:
//!
//! 1. **Decode rows of every running request that has finished its prefill**: the interactive lane first, then the
//!    helper lane, then the background lane (in arrival order inside a lane). While an interactive request decodes,
//!    `bg_rows_interactive` can cap the background rows (round robin), the knob that trades aggregate throughput for the
//!    front agent's speed; by default [`DEFAULT_BG_ROWS_INTERACTIVE`] background rows ride along.
//!    Rows of long agents make a step slow, and the chat waits for all of it: when a step with background rows takes
//!    longer than `chat_gap_ms`, those rows sit out enough steps that the chat's tokens come that far apart on average
//!    ([`Planner::beat`], worked out from the measured step times). They ride along in full in the others, so agents lose
//!    less than with a cap on the rows, and short agents, whose steps are quick, lose nothing.
//! 2. **Prefill chunks** up to `step_tokens` tokens in all, at most `chunk` tokens and one chunk per request:
//!    interactive prefill first, then the helper, then the background. Inside a lane the lane's share of the budget is
//!    split **fairly** over its prefilling requests (`prefill_fair`, the default; water-filling: a request that needs
//!    less than the even share takes what it needs and the rest is shared again), so prompts that arrive together finish
//!    their prefill in the same step and start decoding together. A share is never cut below `fair_min` tokens:
//!    when the budget cannot give every prefilling request that much, the earliest requests get it and the others
//!    wait for the next step (a wave). With `prefill_fair` off, chunks go to the requests in arrival order (first come,
//!    first served: the earliest prompt finishes first). When a helper request is prefilling, the
//!    interactive lane leaves it `helper_tokens` of the budget (the reserved helper slot, rule 073: a compaction never
//!    waits behind the workers it frees); the helper lane has its own slot and queue, so a full background lane never
//!    blocks it.
//! 3. **Background prefill is deferred while an interactive request decodes** (the user is waiting for its tokens, and a
//!    512-token chunk would make one of its inter-token gaps ten times longer, the G0 stall): it gets at most
//!    `bg_prefill_interactive` tokens per step, except that after `bg_prefill_period` consecutive deferred steps one
//!    full chunk (`chunk` tokens, however large `step_tokens` is) goes through, so background work is never starved by
//!    a long chat. A deferred step is counted. `bg_prefill_forced` caps that forced chunk (phase 2 burst, T6: the one
//!    background prefill that can enter the interactive request's decode). With `front_prefill_solo` the same guard
//!    also holds while an interactive request is **prefilling** (the default), so its prompt steps are not slowed by
//!    background chunks (its time to first token). With `fair_granule` (the default) fair shares are whole multiples of
//!    `fair_min`.
//! 4. **Burst prefill-first** (`burst_min`, phase 2 burst; 0 = off): when at least `burst_min` background requests are
//!    prefilling (a burst arrived together), background decode rows sit out the steps that carry background prefill,
//!    so the burst's prefill finishes first and its decode then runs at full batch width. Interactive and helper rows
//!    stay. At most [`BURST_MAX_SKIP`] steps in a row sit out; then the rows ride along once (no starvation).

mod config;

pub use config::*;

use crate::engine::{Lane, ReqId, Work, WorkItem};

/// The most writing time the readers can owe (ms).
const MAX_WRITE_DEBT_MS: f64 = 1000.0;
/// The writers' share beside readers of their own burst (2026-11-13): none, the writers get the one token of every read
/// step. 8 agents with a document each, sent together, end to end: Mistral 7B 18.0 s with the full share and gap, 12.4 s
/// with a tenth and whole read steps, 10.8 s with none; Llama 3.1 8B 19.0 / 12.2 / 10.7 s; Gemma 4 12B 39.8 / 21.7 / 19.0 s.
const BURST_WRITE_SHARE: f64 = 0.0;

/// A running request as the planner sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Running {
    pub req: ReqId,
    pub lane: Lane,
    /// Feed tokens still to prefill; 0 = decoding.
    pub prefill_left: usize,
    /// Arrival order (smaller = earlier).
    pub order: u64,
    /// The largest chunk at this request's position ([`crate::llama_engine::read_block_at`]).
    pub chunk_cap: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Plan {
    pub items: Vec<WorkItem>,
    pub decode_rows: usize,
    /// Of `decode_rows`, the interactive and helper rows.
    pub front_rows: usize,
    pub prefill_tokens: usize,
    /// An interactive request was decoding in this step.
    pub interactive: bool,
    /// Background prefill wanted more than it got because an interactive request was decoding (or, with
    /// `front_prefill_solo`, prefilling).
    pub bg_deferred: bool,
    /// Background decode rows sat out this step for a burst's prefill (`burst_min`).
    pub burst_skip: bool,
    /// A background request was decoding (whether or not its row is in this step).
    pub bg_writers: bool,
    /// Of `prefill_tokens`, the background lane's.
    pub bg_prefill: usize,
}

/// The most steps in a row a beat gives the interactive rows alone, plus the one with everybody.
const MAX_BEAT: usize = 8;
/// Weight of the newest step in the measured step times.
const STEP_EMA: f64 = 0.2;

/// The planner: the config plus the little state the policy keeps between steps.
#[derive(Debug, Clone)]
pub struct Planner {
    pub cfg: SchedConfig,
    /// When an interactive request last ran (`quiet_chunk`).
    last_interactive: Option<std::time::Instant>,
    /// Consecutive steps in which background prefill was deferred.
    deferred_steps: usize,
    /// Round-robin offset over the background decode rows (only with `bg_rows_interactive`).
    rr: usize,
    /// Consecutive steps in which background decode rows sat out for a burst.
    burst_skips: usize,
    /// A burst yield is under way (set at `burst_yield` prefilling background requests, cleared when none prefill):
    /// its last few prompts must not wait for the guard's forced chunk.
    yield_active: bool,
    /// Measured step times in ms while an interactive request decodes: its rows alone, with background rows.
    solo_ms: f64,
    mixed_ms: f64,
    /// Steps the background rows still sit out.
    beat_left: usize,
    /// Measured ms per prefill token of a step (0 = not seen yet), for `bg_gap_ms`.
    read_ms_per_tok: f64,
    /// Decode-only time the background writers are owed for the read steps they sat through (`bg_write_share`).
    write_debt_ms: f64,
    /// The newest arrival there was when the background writing going on now began (`None`: nobody writes). Readers up to
    /// it came with the writers (one burst of agents: some finished reading first); later ones arrived beside them.
    spell_order: Option<u64>,
    /// The writers' share and the read step's limit in force for the step planned last ([`Planner::writer_terms`]).
    share_now: f64,
    gap_now_ms: f64,
}

impl Planner {
    pub fn new(cfg: SchedConfig) -> Planner {
        Planner { cfg, last_interactive: None, deferred_steps: 0, rr: 0, burst_skips: 0, yield_active: false, solo_ms: 0.0, mixed_ms: 0.0, beat_left: 0, read_ms_per_tok: 0.0, write_debt_ms: 0.0, spell_order: None, share_now: 0.0, gap_now_ms: 0.0 }
    }

    /// How long a step took, for the beat.
    pub fn observe(&mut self, ms: f64, p: &Plan) {
        if p.prefill_tokens > 0 {
            let per = ms / p.prefill_tokens as f64;
            self.read_ms_per_tok = if self.read_ms_per_tok == 0.0 { per } else { self.read_ms_per_tok + STEP_EMA * (per - self.read_ms_per_tok) };
        }
        self.observe_write_debt(ms, p);
        if !p.interactive || p.prefill_tokens > 0 {
            return;
        }
        let e = if p.decode_rows > p.front_rows { &mut self.mixed_ms } else { &mut self.solo_ms };
        *e = if *e == 0.0 { ms } else { *e + STEP_EMA * (ms - *e) };
    }

    /// The writing time the background readers owe the writers (`bg_write_share`): a read step without a chat adds the
    /// writers' share of it, a step of background writing pays it back.
    fn observe_write_debt(&mut self, ms: f64, p: &Plan) {
        let share = self.share_now;
        if !p.bg_writers || share <= 0.0 || share >= 1.0 {
            self.write_debt_ms = 0.0;
        } else if !p.interactive {
            if p.bg_prefill > 0 {
                self.write_debt_ms = (self.write_debt_ms + ms * share / (1.0 - share)).min(MAX_WRITE_DEBT_MS);
            } else if p.decode_rows > p.front_rows {
                self.write_debt_ms = (self.write_debt_ms - ms).max(0.0);
            }
        }
    }

    /// One step with background rows in every this many: `(n - 1) x solo + mixed = n x gap`.
    fn beat(&self) -> usize {
        let gap = self.cfg.chat_gap_ms;
        if gap <= 0.0 || self.mixed_ms <= gap {
            return 1;
        }
        let solo = if self.solo_ms > 0.0 { self.solo_ms } else { self.mixed_ms / 4.0 };
        if solo >= gap {
            return MAX_BEAT;
        }
        (((self.mixed_ms - solo) / (gap - solo)).ceil() as usize).clamp(1, MAX_BEAT)
    }

    pub fn plan(&mut self, running: &[Running]) -> Plan {
        let by_lane = by_lane(running);
        let [inter, back, help] = &by_lane;
        let interactive = inter.iter().any(|r| r.prefill_left == 0);
        let mut p = Plan { interactive, bg_writers: back.iter().any(|r| r.prefill_left == 0), ..Default::default() };

        // 1. decode rows: interactive, helper, background
        let n_front_rows = self.decode_rows(inter, back, help, &mut p);

        // 2. prefill chunks: interactive (leaving the helper its reserve), helper, background
        let (chunk, mut budget) = self.prefill_budget(inter);
        let helper_waiting = help.iter().any(|r| r.prefill_left > 0);
        let fair = self.cfg.prefill_fair;
        let shares = Shares { chunk, fair_min: self.cfg.fair_min.max(1), granule: self.cfg.fair_granule };
        let inter_limit = if helper_waiting { budget.saturating_sub(self.cfg.helper_tokens) } else { budget };
        shares.take(inter, inter_limit, fair, &mut budget, &mut p);
        shares.take(help, usize::MAX, fair, &mut budget, &mut p);
        let bg_waiting = back.iter().any(|r| r.prefill_left > 0);
        // the user is waiting: an interactive request decodes (or, with front_prefill_solo, prefills)
        let bg_prefilling = back.iter().filter(|r| r.prefill_left > 0).count();
        let (yielding, guard) = self.guard(inter, interactive, bg_prefilling);
        let (bg_limit, forced) = self.guarded_bg_limit(budget, chunk, guard, bg_waiting);
        self.writer_terms(back, p.bg_writers);
        let bg_limit = self.writer_share_limit(bg_limit, guard, yielding, bg_waiting, p.bg_writers);
        let before = p.prefill_tokens;
        // the chunk forced through (or the small share) while the user waits goes in arrival order: one prompt's chunk,
        // not a fair slice for every waiting prompt, keeps the interactive row's step a one-group prefill step
        let wanted = shares.take(back, bg_limit, fair && !(guard && bg_waiting), &mut budget, &mut p);
        let got = p.prefill_tokens - before;
        p.bg_prefill = got;
        // deferred: the interactive guard (not the budget) kept background prefill from its chunk this step
        let budget_share = wanted.min(budget + got);
        p.bg_deferred = guard && bg_waiting && !forced && got < budget_share;
        self.sit_out_bg_rows(&mut p, n_front_rows, yielding, bg_prefilling, got, fair);
        if p.bg_deferred {
            self.deferred_steps += 1;
        } else {
            self.deferred_steps = 0;
        }
        p
    }

    /// Step 1: the decode rows of every request done with its prefill, interactive, helper, then background (the background
    /// rows capped by `bg_rows_interactive` and the beat while an interactive request decodes), pushed onto `p.items`.
    /// Returns how many of them are interactive or helper rows.
    fn decode_rows(&mut self, inter: &[Running], back: &[Running], help: &[Running], p: &mut Plan) -> usize {
        let interactive = p.interactive;
        let mut rows = decoding(inter);
        rows.extend(decoding(help));
        let mut bg = decoding(back);
        if let (true, Some(k)) = (interactive, self.cfg.bg_rows_interactive) {
            if k < bg.len() {
                let start = self.rr % bg.len();
                bg.rotate_left(start);
                bg.truncate(k);
                self.rr = self.rr.wrapping_add(k.max(1));
            }
        }
        if interactive && !bg.is_empty() {
            if self.beat_left > 0 {
                self.beat_left -= 1;
                bg.clear();
            } else {
                self.beat_left = self.beat() - 1;
            }
        }
        let n_front_rows = rows.len();
        p.front_rows = n_front_rows;
        rows.extend(bg);
        p.decode_rows = rows.len();
        p.items.extend(rows.into_iter().map(|req| WorkItem { req, work: Work::Decode }));
        n_front_rows
    }

    /// The chunk and the step's prefill budget: the bigger `quiet_chunk` ones while no interactive request has run for
    /// `quiet_after_ms` (this step's interactive requests count as running now).
    fn prefill_budget(&mut self, inter: &[Running]) -> (usize, usize) {
        if !inter.is_empty() {
            self.last_interactive = Some(std::time::Instant::now());
        }
        let quiet = self.cfg.quiet_chunk.filter(|_| self.last_interactive.is_none_or(|t| t.elapsed().as_millis() as u64 >= self.cfg.quiet_after_ms));
        let chunk = quiet.unwrap_or(self.cfg.chunk);
        let budget = quiet.map_or(self.cfg.step_tokens, |q| q.max(self.cfg.step_tokens));
        (chunk, budget)
    }

    /// Whether a burst yield is under way (`burst_yield`; kept in `yield_active`) and whether the interactive guard holds
    /// background prefill back this step: `(yielding, guard)`.
    fn guard(&mut self, inter: &[Running], interactive: bool, bg_prefilling: usize) -> (bool, bool) {
        self.yield_active = self.cfg.burst_yield > 0 && bg_prefilling > 0 && (self.yield_active || bg_prefilling >= self.cfg.burst_yield);
        let yielding = self.yield_active;
        let guard = !yielding && (interactive || (self.cfg.front_prefill_solo && inter.iter().any(|r| r.prefill_left > 0)));
        (yielding, guard)
    }

    /// The background lane's prefill limit under the interactive guard: `bg_prefill_interactive`, or one forced chunk
    /// once `bg_prefill_period` steps were deferred. Returns `(limit, forced)`.
    fn guarded_bg_limit(&self, budget: usize, chunk: usize, guard: bool, bg_waiting: bool) -> (usize, bool) {
        let mut bg_limit = budget;
        let mut forced = false;
        if guard && bg_waiting {
            let due = self.cfg.bg_prefill_period > 0 && self.deferred_steps + 1 >= self.cfg.bg_prefill_period;
            if due {
                forced = true;
                // one chunk, not the whole step budget
                bg_limit = bg_limit.min(self.cfg.bg_prefill_forced.unwrap_or(chunk).min(chunk));
            } else {
                bg_limit = bg_limit.min(self.cfg.bg_prefill_interactive);
            }
        }
        (bg_limit, forced)
    }

    /// The writers' share and the read step's limit for this step (`share_now`, `gap_now_ms`). Readers that arrived
    /// beside writers already writing get `bg_write_share` and `bg_gap_ms`. Readers of the writers' own burst (8 agents
    /// sent together, the short documents done first) get [`BURST_WRITE_SHARE`] and whole read steps: there the
    /// work ends soonest when the reading ends first, and short read steps cost far more than the writers gain.
    fn writer_terms(&mut self, back: &[Running], bg_writers: bool) {
        if !bg_writers {
            self.spell_order = None;
        } else if self.spell_order.is_none() {
            self.spell_order = back.iter().map(|r| r.order).max();
        }
        let late = back.iter().any(|r| r.prefill_left > 0 && self.spell_order.is_some_and(|o| r.order > o));
        let share = self.cfg.bg_write_share;
        (self.share_now, self.gap_now_ms) = if late { (share, self.cfg.bg_gap_ms) } else { (share.min(BURST_WRITE_SHARE), 0.0) };
    }

    /// Background writers beside background readers: the writers keep their share of the time (a step without reading
    /// while the readers owe them), and a read step is no longer than the gap ([`Planner::writer_terms`]).
    fn writer_share_limit(&self, mut bg_limit: usize, guard: bool, yielding: bool, bg_waiting: bool, bg_writers: bool) -> usize {
        if !guard && !yielding && bg_waiting && bg_writers && self.share_now > 0.0 {
            if self.write_debt_ms > 0.0 {
                bg_limit = 0;
            } else if self.gap_now_ms > 0.0 && self.read_ms_per_tok > 0.0 {
                bg_limit = bg_limit.min(((self.gap_now_ms / self.read_ms_per_tok) as usize / 32 * 32).max(32));
            }
        }
        bg_limit
    }

    /// The background decode rows sit out a step of many prefill chunks, or of a burst's prefill (`burst_min`,
    /// `burst_yield`); sets `p.burst_skip` and counts the skips in a row.
    fn sit_out_bg_rows(&mut self, p: &mut Plan, n_front_rows: usize, yielding: bool, bg_prefilling: usize, got: usize, fair: bool) {
        // A step of many prefill chunks runs much slower with any decode row in it (dev, srvbatch: Qwen3-0.6B, 62 chunks
        // of 33 tokens: 0.5 s alone, 2.8 s with one decode row; 4 chunks of 512 show no such cost). Background decode
        // rows sit such a step out; interactive and helper rows stay.
        let n_chunks = p.items.len() - p.decode_rows;
        let has_bg_rows = p.decode_rows > n_front_rows;
        p.burst_skip = self.burst_skip(yielding, bg_prefilling, got, has_bg_rows);
        if has_bg_rows && (p.burst_skip || (fair && n_chunks > self.cfg.max_mixed_chunks)) {
            p.items.drain(n_front_rows..p.decode_rows);
            p.decode_rows = n_front_rows;
        }
        self.burst_skips = if p.burst_skip { self.burst_skips + 1 } else { 0 };
    }

    /// Burst prefill-first: a burst of background prompts prefills before its many-row decode starts (the background rows
    /// sit out this step, at most [`BURST_MAX_SKIP`] steps in a row).
    fn burst_skip(&self, yielding: bool, bg_prefilling: usize, got: usize, has_bg_rows: bool) -> bool {
        (yielding || (self.cfg.burst_min > 0 && bg_prefilling >= self.cfg.burst_min))
            && bg_prefilling >= 1
            && got > 0
            && has_bg_rows
            && self.burst_skips < BURST_MAX_SKIP
    }
}

/// The running requests by lane (interactive, background, helper), each lane in arrival order.
fn by_lane(running: &[Running]) -> [Vec<Running>; 3] {
    let mut by_lane: [Vec<Running>; 3] = Default::default();
    for r in running {
        by_lane[r.lane.index()].push(*r);
    }
    for v in &mut by_lane {
        v.sort_by_key(|r| r.order);
    }
    by_lane
}

/// The requests of `v` that are decoding (done with their prefill).
fn decoding(v: &[Running]) -> Vec<ReqId> {
    v.iter().filter(|r| r.prefill_left == 0).map(|r| r.req).collect::<Vec<_>>()
}

/// How a step's prefill budget is shared out inside a lane: the chunk cap, and the fair-share settings.
#[derive(Clone, Copy)]
struct Shares {
    chunk: usize,
    fair_min: usize,
    granule: bool,
}

impl Shares {
    /// One lane's prefill chunks, at most `limit` (and what is left of `budget`) tokens: pushed onto `p`, taken from
    /// `budget`. Returns the tokens the lane wanted.
    fn take(&self, v: &[Running], limit: usize, fair: bool, budget: &mut usize, p: &mut Plan) -> usize {
        let (chunk, fair_min, granule) = (self.chunk, self.fair_min, self.granule);
        let lim = limit.min(*budget);
        let want: Vec<(ReqId, usize)> = v.iter().filter(|r| r.prefill_left > 0).map(|r| (r.req, r.prefill_left.min(chunk).min(r.chunk_cap))).collect();
        let wanted = want.iter().map(|w| w.1).sum();
        let got = match (fair, granule) {
            (true, true) => granular_shares(&want, lim, fair_min),
            (true, false) => fair_shares(&want, lim, fair_min),
            (false, _) => fifo_shares(&want, lim),
        };
        for (&(req, _), n) in want.iter().zip(got) {
            if n == 0 {
                continue;
            }
            p.items.push(WorkItem { req, work: Work::Prefill { n } });
            p.prefill_tokens += n;
            *budget -= n;
        }
        wanted
    }
}

/// Arrival order: each request (`(req, want)`, earliest first) takes what it wants while the budget lasts.
fn fifo_shares(want: &[(ReqId, usize)], mut lim: usize) -> Vec<usize> {
    want.iter()
        .map(|&(_, w)| {
            let n = w.min(lim);
            lim -= n;
            n
        })
        .collect()
}

/// Fair shares of `lim` over `want` (earliest first): water-filling (a request that wants less than the even share
/// takes what it wants, the rest is shared again among the others), the remainder of the division to the earliest. When
/// `lim` cannot give every request `min` tokens, only the earliest `lim / min` requests share it.
fn fair_shares(want: &[(ReqId, usize)], lim: usize, min: usize) -> Vec<usize> {
    let mut got = vec![0usize; want.len()];
    if want.iter().map(|w| w.1).sum::<usize>() <= lim {
        return want.iter().map(|w| w.1).collect();
    }
    let k = want.len().min((lim / min).max(1));
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_by_key(|&i| (want[i].1, i));
    let mut left = lim;
    for (j, &i) in idx.iter().enumerate() {
        let share = left / (k - j);
        if want[i].1 > share {
            // this one and every later (larger) one is capped at the level `share`
            for &i in &idx[j..] {
                got[i] = share;
            }
            left -= share * (k - j);
            break;
        }
        got[i] = want[i].1;
        left -= want[i].1;
    }
    for i in 0..k {
        let n = (want[i].1 - got[i]).min(left);
        got[i] += n;
        left -= n;
    }
    got
}

/// Fair shares in whole units of `unit` tokens (phase 2 burst): one unit at a time to each request in turn, earliest
/// first, while the budget lasts; a request's last unit may be short (the end of its prompt). What is left under one
/// unit (a budget that is not a multiple of `unit`, such as the helper's reserve taken out) goes to the earliest
/// requests, as in [`fair_shares`]. A step of shares such as 7 x 292 or 7 x 220 tokens ran 2.2 to 2.6x slower on the GPU
/// than one of 8 x 256 (dev, Qwen3-0.6B agents-32).
fn granular_shares(want: &[(ReqId, usize)], lim: usize, unit: usize) -> Vec<usize> {
    if want.iter().map(|w| w.1).sum::<usize>() <= lim {
        return want.iter().map(|w| w.1).collect();
    }
    let k = want.len().min((lim / unit).max(1));
    let mut got = vec![0usize; want.len()];
    let mut left = lim;
    loop {
        let mut more = false;
        for i in 0..k {
            let give = (want[i].1 - got[i]).min(unit);
            if give > 0 && give <= left {
                got[i] += give;
                left -= give;
                more = true;
            }
        }
        if !more {
            break;
        }
    }
    for i in 0..k {
        let n = (want[i].1 - got[i]).min(left);
        got[i] += n;
        left -= n;
    }
    got
}

#[cfg(test)]
mod tests;
