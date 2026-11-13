//! Exact prompt-lookup speculative decoding for the generic engine (any architecture, no per-model code).
//!
//! A sequence's own context (prompt + history + generated tokens) is indexed by n-grams. When the last `n` tokens occurred
//! before, the tokens that followed that earlier occurrence are proposed as a draft. The engine puts `[pending, draft...]`
//! in the step's batch, takes the greedy argmax of every row, and keeps the longest draft prefix that matches those
//! argmaxes plus the model's own next token. With the row-exact Metal switch on (see `engine/vendor/ggml-rowexact-b11669`)
//! a row's logits are bitwise what a one-row step gives, so the emitted tokens are exactly the tokens speculation off
//! emits (with the same switch); the draft only decides how many steps they take.
//!
//! This file holds the index, the proposal, the adaptive draft length and the flags. The batch, the acceptance and the
//! KV rollback are in `llama_engine.rs` (`step`).

use std::collections::HashMap;

/// `--spec off|auto|on`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpecMode {
    /// No speculation (the default): the engine is the plain one.
    Off,
    /// Speculate while the step is small enough for the row-exact path; back off per sequence when drafts miss.
    Auto,
    /// Like auto but every sequence always drafts its full length (no back-off).
    On,
}

variant_names!(SpecMode { Off => "off", Auto => "auto", On => "on" }, "expected off, auto or on, got {:?}");

#[derive(Clone, Debug)]
pub struct SpecConfig {
    pub mode: SpecMode,
    /// Most draft tokens per sequence per step (`--spec-k`).
    pub k: usize,
    /// N-gram length of the lookup (`--spec-n`).
    pub n: usize,
    /// Most rows (tokens) a step may carry with speculation on (`--spec-rows`, default 8, at most 16: the row-exact path
    /// holds 16). Measured on Gemma-3-4B: with 3 sequences a 12-row step lost to a 6-row one, with 8 sequences drafts lose.
    pub max_rows: usize,
    /// Most decoding sequences a step may have and still draft (`--spec-max-seqs`, default 1). One at a time measured
    /// Qwen3-4B 1.27x, Gemma-3 4B 1.09x (spec.py singles); with more the step's weights are already shared by several rows
    /// and drafts did not pay: 8 concurrent 0.82-0.99x, the Gemma-3 4B agent loop 1.11x at 4 agents but 0.93x at 16 with
    /// up to 4 drafting (2026-10-28).
    pub max_seqs: usize,
    /// Auto mode: a sequence whose drafts keep less than this share of their tokens (a running rate, after 3 draft steps)
    /// sits out a growing number of lookups. 0: off (the libllama engine). sllm-next: drafting also takes its queued step.
    pub min_keep: f32,
}

impl Default for SpecConfig {
    fn default() -> SpecConfig {
        SpecConfig { mode: SpecMode::Off, k: 4, n: 3, max_rows: 8, max_seqs: 1, min_keep: 0.0 }
    }
}

/// Per-sequence lookup state: the n-gram index over the sequence's context and the adaptive draft length.
#[derive(Default)]
pub struct SpecState {
    /// hash of n tokens -> index of the LAST token of their most recent occurrence
    idx: HashMap<u64, u32>,
    /// n-grams ending before this position are in `idx`
    upto: usize,
    /// current draft length (0 = not yet set: `k` max)
    k_cur: usize,
    /// consecutive steps whose whole draft was rejected
    miss: u32,
    /// steps to sit out
    cooldown: u32,
    /// draft steps seen, the running share of drafted tokens kept, times the rate has sent the sequence out (`min_keep`)
    steps: u32,
    keep_rate: f32,
    gated: u32,
    /// the last `propose`'s match: the index of the last token of the earlier occurrence it drafted from
    pub last_end: Option<usize>,
}

fn hash_ngram(t: &[i32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &x in t {
        h ^= x as u32 as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    }
    h
}

impl SpecState {
    /// The draft for a context `ctx` (all tokens of the sequence, the pending one last): the up to `k` tokens that followed
    /// the most recent earlier occurrence of its last `n` tokens. Empty when there is none or the sequence sits out.
    pub fn propose(&mut self, ctx: &[i32], cfg: &SpecConfig) -> Vec<i32> {
        let n = cfg.n.max(1);
        let len = ctx.len();
        self.last_end = None;
        if len < n + 1 {
            return Vec::new();
        }
        if self.upto > len {
            // the context shrank (a cut, a restored snapshot): index it again
            self.idx.clear();
            self.upto = 0;
        }
        // n-grams ending at positions upto..len-1 (the suffix itself is looked up first, then added)
        let mut e = self.upto.max(n - 1);
        while e < len - 1 {
            self.idx.insert(hash_ngram(&ctx[e + 1 - n..=e]), e as u32);
            e += 1;
        }
        self.upto = len - 1;
        let suffix = &ctx[len - n..];
        let found = self.idx.get(&hash_ngram(suffix)).copied();
        // the suffix becomes an earlier occurrence for the next lookup
        self.idx.insert(hash_ngram(suffix), (len - 1) as u32);
        self.upto = len;
        if cfg.mode == SpecMode::Auto && self.cooldown > 0 {
            self.cooldown -= 1;
            return Vec::new();
        }
        let Some(end) = found else { return Vec::new() };
        let end = end as usize;
        // verify against the real tokens (a hash collision or a stale entry must not draft)
        if end + 1 < n || end >= len - 1 || ctx[end + 1 - n..=end] != *suffix {
            return Vec::new();
        }
        self.last_end = Some(end);
        let k = self.k_now(cfg);
        let from = end + 1;
        ctx[from..(from + k).min(len)].to_vec()
    }

    /// Whether the lookup after one more token can find anything, for a context of `cells` then `last`: that lookup's
    /// n-gram ends with the new token, so the context's last n - 1 tokens must occur earlier, with a token after them.
    /// False: the next step will not draft.
    pub fn may_match_next(cells: &[i32], last: i32, cfg: &SpecConfig) -> bool {
        let m = cfg.n.max(1) - 1;  // the tail's length
        if m == 0 {
            return true;
        }
        if cells.len() < m {
            return false;
        }
        let head = &cells[cells.len() + 1 - m..];  // the tail but its last token (`last`)
        (m - 1..cells.len()).any(|e| cells[e] == last && cells[e + 1 - m..e] == *head)
    }

    /// The sequence sits out its next lookups (auto mode after drafts that missed).
    pub fn sitting_out(&self) -> bool {
        self.cooldown > 0
    }

    /// Current draft length.
    pub fn k_now(&self, cfg: &SpecConfig) -> usize {
        if cfg.mode == SpecMode::On || self.k_cur == 0 {
            cfg.k
        } else {
            self.k_cur.min(cfg.k)
        }
    }

    /// Start the adaptive draft length at `k` (before the first lookup; afterwards it grows with drafts kept whole).
    pub fn start_at(&mut self, k: usize) {
        if self.k_cur == 0 {
            self.k_cur = k;
        }
    }

    /// The outcome of a step that verified `proposed` real draft tokens and kept `accepted` of them.
    pub fn record(&mut self, proposed: usize, accepted: usize, cfg: &SpecConfig) {
        if proposed == 0 || cfg.mode != SpecMode::Auto {
            return;
        }
        let cur = self.k_now(cfg);
        if cfg.min_keep > 0.0 {
            self.track_keep_rate(proposed, accepted, cfg.min_keep);
        }
        self.adapt_k(cur, proposed, accepted, cfg);
    }

    /// The running share of drafted tokens kept: under `min_keep` the sequence sits out its next lookups.
    fn track_keep_rate(&mut self, proposed: usize, accepted: usize, min_keep: f32) {
        let obs = accepted as f32 / proposed as f32;
        self.keep_rate = if self.steps == 0 { obs } else { 0.75 * self.keep_rate + 0.25 * obs };
        self.steps += 1;
        if self.steps >= 3 && self.keep_rate < min_keep {
            // drafts that pay less than they cost: sit out 16, 32, .. 128 lookups, then try again from the threshold
            self.cooldown = self.cooldown.max(16 << self.gated.min(3));
            self.gated += 1;
            self.keep_rate = min_keep;
        }
    }

    /// The next draft length from the current one `cur`: halved on a miss (and a growing sit-out after misses in a row),
    /// one longer when a draft was kept whole, halved when less than half of it was.
    fn adapt_k(&mut self, cur: usize, proposed: usize, accepted: usize, cfg: &SpecConfig) {
        if accepted == 0 {
            self.miss += 1;
            self.k_cur = (cur / 2).max(1);
            if self.miss >= 3 {
                // sit out a growing number of steps: a context that does not repeat costs nothing but the lookup
                self.cooldown = 1 << (self.miss - 2).min(5);
            }
        } else {
            self.miss = 0;
            self.k_cur = if accepted >= proposed { (cur + 1).min(cfg.k) } else if accepted * 2 < proposed { (cur / 2).max(1) } else { cur };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SpecConfig {
        SpecConfig { mode: SpecMode::Auto, k: 4, n: 3, max_rows: 16, max_seqs: 1, min_keep: 0.0 }
    }

    #[test]
    fn proposes_what_followed_the_last_occurrence() {
        let mut s = SpecState::default();
        let ctx: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 9, 9, 1, 2, 3];
        assert_eq!(s.propose(&ctx, &cfg()), vec![4, 5, 6, 9]);
    }

    #[test]
    fn most_recent_occurrence_wins_and_draft_is_cut_at_the_end() {
        let mut s = SpecState::default();
        let ctx: Vec<i32> = vec![1, 2, 3, 7, 1, 2, 3, 8, 8, 1, 2, 3];
        // earlier occurrences end at 2 and 6; the later one (followed by 8 8 1 2) wins
        assert_eq!(s.propose(&ctx, &cfg()), vec![8, 8, 1, 2]);
        let mut s = SpecState::default();
        let ctx: Vec<i32> = vec![5, 5, 5, 5];
        assert_eq!(s.propose(&ctx, &cfg()), vec![5]); // overlapping repetition: one token follows the earlier occurrence
    }

    #[test]
    fn no_match_no_draft_and_incremental_equals_fresh() {
        let mut s = SpecState::default();
        let ctx: Vec<i32> = (0..40).collect();
        assert!(s.propose(&ctx, &cfg()).is_empty());
        // grow one token at a time like a decode loop; the result matches a fresh index of the same context
        let mut grown: Vec<i32> = vec![1, 2, 3, 4, 1, 2];
        let mut inc = SpecState::default();
        for t in [3, 9, 1, 2, 3] {
            grown.push(t);
            let a = inc.propose(&grown, &cfg());
            let b = SpecState::default().propose(&grown, &cfg());
            assert_eq!(a, b, "context {grown:?}");
        }
    }

    #[test]
    fn adaptive_backs_off_and_recovers() {
        let c = cfg();
        let mut s = SpecState::default();
        assert_eq!(s.k_now(&c), 4);
        s.record(4, 0, &c);
        assert_eq!(s.k_now(&c), 2);
        s.record(2, 0, &c);
        s.record(1, 0, &c);
        assert_eq!(s.k_now(&c), 1);
        assert!(s.cooldown > 0);
        s.cooldown = 0;
        s.record(1, 1, &c);
        assert_eq!(s.k_now(&c), 2);
        let on = SpecConfig { mode: SpecMode::On, ..c.clone() };
        let mut s2 = SpecState::default();
        s2.record(4, 0, &on);
        assert_eq!(s2.k_now(&on), 4);
    }

    #[test]
    fn the_next_lookup_can_match_only_after_a_tail_seen_before() {
        let cfg = SpecConfig { n: 3, ..SpecConfig::default() };
        // context 5 6 7 8 | 6: tail (8, 6) never seen -> no match possible
        assert!(!SpecState::may_match_next(&[5, 6, 7, 8], 6, &cfg));
        // context 5 6 7 5 | 6: tail (5, 6) seen at the start, followed by 7 -> may match
        assert!(SpecState::may_match_next(&[5, 6, 7, 5], 6, &cfg));
        // too short
        assert!(!SpecState::may_match_next(&[], 6, &cfg));
    }

    #[test]
    fn a_low_keep_rate_sits_out() {
        let c = SpecConfig { min_keep: 0.4, ..cfg() };
        let mut s = SpecState::default();
        for _ in 0..3 {
            s.record(2, 1, &c);  // 50%: drafting goes on
        }
        assert!(!s.sitting_out());
        s.record(4, 1, &c);  // 25%: the rate 0.44
        assert!(!s.sitting_out());
        s.record(4, 1, &c);  // 0.39: under 0.4, out for 16 lookups
        assert!(s.sitting_out());
        assert_eq!(s.cooldown, 16);
    }
}
