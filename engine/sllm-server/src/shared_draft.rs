//! Shared draft memory ("suffix decoding"): every token this server has seen (each request's new prompt tokens and its
//! generated tokens, the most recent `cap` of them) sits in one ring. A writer whose own context offers no lookup match
//! (or a shorter one) asks the ring: the last tokens of its context are matched against all earlier text and the tokens
//! that followed the longest, most recent match are proposed. Requests that repeat formats and content (the same task over
//! different documents, tool calls, JSON) then draft from their predecessors' answers.
//!
//! The index is two direct-mapped hash tables of positions (a short n-gram and a long one, newer entries overwrite older
//! ones), so the memory is fixed: 4 bytes a ring token, 4 bytes a table slot. A lookup verifies the real tokens, so a
//! collision or a stale entry cannot draft. On by default, SLLM_NEXT_SHARED_DRAFT=0 turns it off (see `next_engine.rs`).

/// Ring separator between two requests: no real token, so a draft stops at it and no lookup matches across it.
const SEP: i32 = -1;
/// Longest match length looked at (longer matches are not told apart).
const MAX_MATCH: usize = 32;
/// Positions at the ring's oldest end that drafts do not read (they may be overwritten while read).
const MARGIN: u64 = 64;

pub struct SharedDraft {
    ring: Vec<i32>,
    mask: u64,
    /// Tokens ever appended (the next token's absolute position).
    total: u64,
    /// absolute position (low 32 bits) of the last token of the most recent n-gram per hash, u32::MAX empty
    short: Vec<u32>,
    long: Vec<u32>,
    tmask: u64,
    /// the short n-gram's first n - 1 tokens seen (a bitset): can the next lookup match at all
    pre: Vec<u64>,
    pre_set: u64,
    n: usize,
    n_long: usize,
    pub proposals: u64,
}

fn hash(t: impl Iterator<Item = i32>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in t {
        h ^= x as u32 as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    }
    h
}

impl SharedDraft {
    /// A memory of `cap` tokens (rounded up to a power of two), short n-grams of `n` tokens and long ones of `3 n`.
    pub fn new(cap: usize, n: usize) -> SharedDraft {
        let cap = cap.max(1024).next_power_of_two();
        let tbits = cap.trailing_zeros().max(10);
        SharedDraft {
            ring: vec![SEP; cap], mask: cap as u64 - 1, total: 0,
            short: vec![u32::MAX; 1 << tbits], long: vec![u32::MAX; 1 << tbits], tmask: (1u64 << tbits) - 1,
            pre: vec![0; 1 << 18], pre_set: 0, n: n.max(2), n_long: n.max(2) * 3, proposals: 0,
        }
    }

    /// Bytes held.
    pub fn bytes(&self) -> usize {
        (self.ring.len() + self.short.len() + self.long.len()) * 4 + self.pre.len() * 8
    }

    fn at(&self, abs: u64) -> i32 {
        self.ring[(abs & self.mask) as usize]
    }

    fn window_hash(&self, end: u64, n: usize) -> u64 {
        hash((0..n as u64).map(|i| self.at(end + 1 - n as u64 + i)))
    }

    fn pre_bit(h: u64) -> (usize, u64) {
        let b = (h >> 20) & ((1 << 24) - 1);
        ((b >> 6) as usize, 1 << (b & 63))
    }

    fn push(&mut self, t: i32) {
        let a = self.total;
        self.ring[(a & self.mask) as usize] = t;
        self.total += 1;
        if t == SEP {
            return;
        }
        for (n, long) in [(self.n, false), (self.n_long, true)] {
            if a + 1 < n as u64 || (0..n as u64).any(|i| self.at(a - i) == SEP) {
                continue;
            }
            let h = self.window_hash(a, n);
            let slot = (h & self.tmask) as usize;
            if long { self.long[slot] = a as u32 } else { self.short[slot] = a as u32 }
            if !long {
                let (w, m) = Self::pre_bit(hash((0..n as u64 - 1).map(|i| self.at(a + 1 - n as u64 + i))));
                if self.pre[w] & m == 0 {
                    self.pre[w] |= m;
                    self.pre_set += 1;
                }
            }
        }
    }

    /// Remember a request's tokens (its new prompt tokens and its generated ones).
    pub fn add(&mut self, tokens: &[i32]) {
        if tokens.len() < self.n + 1 {
            return;
        }
        self.push(SEP);
        for &t in tokens {
            self.push(t);
        }
        // stale bits (n-grams the ring has dropped) pile up: start the bitset over from the ring now and then
        if self.pre_set > 6 << 20 {
            self.pre.iter_mut().for_each(|w| *w = 0);
            self.pre_set = 0;
            let from = self.total.saturating_sub(self.mask + 1 - MARGIN);
            let n = self.n as u64;
            for a in from + n..self.total {
                if (0..n).all(|i| self.at(a - i) != SEP) {
                    let (w, m) = Self::pre_bit(hash((0..n - 1).map(|i| self.at(a + 1 - n + i))));
                    if self.pre[w] & m == 0 {
                        self.pre[w] |= m;
                        self.pre_set += 1;
                    }
                }
            }
        }
    }

    /// Could the lookup after one more token find a match: for a context of `cells` then `last`, its n-gram ends with the
    /// new token, so the n - 1 tokens before it must have started an n-gram before. (A bitset: false positives only.)
    pub fn may_match_next(&self, cells: &[i32], last: i32) -> bool {
        let m = self.n - 2;
        if cells.len() < m {
            return false;
        }
        let (w, b) = Self::pre_bit(hash(cells[cells.len() - m..].iter().copied().chain(std::iter::once(last))));
        self.pre[w] & b != 0
    }

    /// The longest match of the end of `ctx` in the memory (the most recent among equals) and up to `k` tokens that
    /// followed it: (match length, draft).
    pub fn propose(&mut self, ctx: &[i32], k: usize) -> Option<(usize, Vec<i32>)> {
        if ctx.len() < self.n || k == 0 {
            return None;
        }
        let (l, e) = self.best_match(ctx)?;
        let mut d = Vec::with_capacity(k);
        for i in 1..=k as u64 {
            if e + i >= self.total || self.at(e + i) == SEP {
                break;
            }
            d.push(self.at(e + i));
        }
        if d.is_empty() {
            return None;
        }
        self.proposals += 1;
        Some((l, d))
    }

    /// The longer of the long and the short n-gram table's matches of the end of `ctx` (the long one among equals), at
    /// least `n` tokens: (match length, position of its last token).
    fn best_match(&self, ctx: &[i32]) -> Option<(usize, u64)> {
        let mut best: Option<(usize, u64)> = None;
        for (n, long) in [(self.n_long, true), (self.n, false)] {
            let Some((l, e)) = self.match_in(ctx, n, long) else { continue };
            if l >= self.n && best.map_or(true, |(bl, _)| l > bl) {
                best = Some((l, e));
            }
        }
        best
    }

    /// Where the table of `n`-grams (`long` or short) last saw the end of `ctx`, and how far back the match goes: (match
    /// length, position of its last token). None: no entry, one gone from the ring, or one that nothing follows.
    fn match_in(&self, ctx: &[i32], n: usize, long: bool) -> Option<(usize, u64)> {
        if ctx.len() < n {
            return None;
        }
        let h = hash(ctx[ctx.len() - n..].iter().copied());
        let p = if long { self.long[(h & self.tmask) as usize] } else { self.short[(h & self.tmask) as usize] };
        if p == u32::MAX {
            return None;
        }
        let d = (self.total as u32).wrapping_sub(p) as u64;
        if d < 2 || d + MARGIN >= self.mask + 1 || d > self.total {
            return None;
        }
        let e = self.total - d;
        if self.at(e + 1) == SEP {
            return None; // nothing follows: the request ended there
        }
        Some((self.match_len(ctx, e, d), e))
    }

    /// Tokens the memory before position `e` (`d` back from its end) shares with the end of `ctx`.
    fn match_len(&self, ctx: &[i32], e: u64, d: u64) -> usize {
        let mut l = 0;
        while l < MAX_MATCH.min(ctx.len()) && (l as u64) <= e && d + (l as u64) + MARGIN < self.mask + 1 && self.at(e - l as u64) == ctx[ctx.len() - 1 - l] {
            l += 1;
        }
        l
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_the_longest_and_most_recent_match() {
        let mut s = SharedDraft::new(4096, 3);
        s.add(&[10, 11, 12, 13, 14, 15, 16, 17]);
        s.add(&[20, 11, 12, 13, 99, 98, 97, 96]);
        // 11 12 13 matches both; the most recent is followed by 99
        assert_eq!(s.propose(&[5, 11, 12, 13], 3), Some((3, vec![99, 98, 97])));
        // a long (9-token) match wins over a more recent short one
        let run: Vec<i32> = (100..116).collect();
        s.add(&run);
        s.add(&[50, 109, 110, 111, 77, 78]);
        let q: Vec<i32> = (103..112).collect();
        assert_eq!(s.propose(&q, 2), Some((9, vec![112, 113])));
        // no match, too short
        assert_eq!(s.propose(&[1, 2, 3, 4], 3), None);
        assert_eq!(s.propose(&[12, 13], 3), None);
    }

    #[test]
    fn a_draft_stops_at_the_end_of_a_request() {
        let mut s = SharedDraft::new(4096, 3);
        s.add(&[1, 2, 3, 4, 5]);
        s.add(&[7, 8, 9, 10, 11]);
        assert_eq!(s.propose(&[1, 2, 3, 4, 5], 4), None);
        assert_eq!(s.propose(&[0, 3, 4, 5], 4), None); // 5 is the last token of its request: nothing follows
        assert_eq!(s.propose(&[0, 2, 3, 4], 4), Some((3, vec![5])));
    }

    #[test]
    fn the_oldest_text_goes_and_the_next_lookup_check_matches() {
        let mut s = SharedDraft::new(1024, 3);
        s.add(&[1, 2, 3, 4, 5, 6]);
        assert!(s.may_match_next(&[0, 1], 2)); // (1, 2) began 1 2 3
        assert!(!s.may_match_next(&[0, 4], 2));
        for i in 0..40 {
            let v: Vec<i32> = (0..100).map(|j| 1000 + i * 100 + j).collect();
            s.add(&v);
        }
        assert_eq!(s.propose(&[0, 1, 2, 3], 3), None);
        let last: Vec<i32> = (0..100).map(|j| 1000 + 39 * 100 + j).collect();
        assert_eq!(s.propose(&last[10..20], 2), Some((10, vec![last[20], last[21]])));
    }
}
