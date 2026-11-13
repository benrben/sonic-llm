//! Prefix sharing: the prefix lookup, frozen and warm holders, and the prompt cache on disk.

use super::*;

/// A sequence that only holds a prompt prefix (a system prompt) and is never decoded: sequences that start with these tokens
/// begin from a full copy of it.
pub(super) struct Holder {
    pub(super) sid: i32,
    pub(super) tokens: Vec<i32>,
    pub(super) last_used: Instant,
}

#[derive(Default)]
pub(super) struct HolderStats {
    pub(super) created: u64,
    pub(super) evicted: u64,
    pub(super) attached: u64,
    pub(super) attached_tokens: u64,
    pub(super) warm_kept: u64,
    pub(super) warm_taken: u64,
}

pub(super) const HOLDER_MIN: usize = 64;
/// Holder models: a request that has read this much keeps a copy of it (a shorter prompt is cheaper to read again).
pub(super) const WARM_COPY_MIN: usize = 2048;
const MAX_HOLDERS: usize = 4;
/// Fresh prompts remembered to find a shared start that nobody declared (an agent loop sends its system prompt in the
/// first turn), and the tokens kept of each.
const RECENT_STARTS: usize = 16;
const RECENT_START_TOKENS: usize = 8192;

pub(super) fn hash_ids(ids: &[i32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &t in ids {
        h = (h ^ t as u32 as u64).wrapping_mul(0x100000001b3);
    }
    h ^ ids.len() as u64
}

impl LlamaEngine {
    /// The longest token-exact prefix (at most `ids.len() - 1` tokens) another sequence's cache holds: `(sid, tokens)`.
    /// Partial copies valid: the longest common prefix over all cached sequences (the trie). Otherwise: the longest frozen
    /// holder that is a whole prefix of the prompt.
    pub(super) fn find_prefix(&self, ids: &[i32], skip_sid: Option<i32>) -> Option<(i32, usize)> {
        self.find_prefix_n(ids, skip_sid, true)
    }

    /// [`Self::find_prefix`]; `reserve_last` keeps the prompt's last token out of the match (a request needs it to get logits;
    /// an agent's system prompt does not).
    pub(super) fn find_prefix_n(&self, ids: &[i32], skip_sid: Option<i32>, reserve_last: bool) -> Option<(i32, usize)> {
        if (!self.share && !self.holders_on) || ids.len() < 2 {
            return None;
        }
        let max = if reserve_last { ids.len() - 1 } else { ids.len() };
        // a stream copy moves the rows its source uses (all `capacity` cells where partial copies are off), so a short shared
        // prefix of a long sequence is cheaper to prefill again
        let min_k = if !self.share { 32 } else { self.min_share() };
        let best = if self.share { self.ptree.borrow().longest_prefix(ids, max, skip_sid) } else { self.ptree.borrow().longest_end(ids, max, skip_sid) };
        if self.check_prefix && self.share {
            self.check_against_scan(ids, skip_sid, reserve_last, best);
        }
        let best = if self.share { best.map(|(s, k)| (s, k / self.share_align * self.share_align)) } else { best };
        best.filter(|b| b.1 >= min_k)
    }

    /// `SLLM_PREFIX_CHECK=1`: the trie's match `best` against the linear scan.
    fn check_against_scan(&self, ids: &[i32], skip_sid: Option<i32>, reserve_last: bool, best: Option<(i32, usize)>) {
        let lin = self.find_prefix_linear(ids, skip_sid, 1, reserve_last).map_or(0, |b| b.1);
        let got = best.map_or(0, |b| b.1);
        if lin != got {
            eprintln!("sllm-server: PREFIX CHECK MISMATCH: trie {got}, linear scan {lin} (prompt {} tokens)", ids.len());
        }
    }

    /// The shortest start worth sharing when partial copies are valid.
    pub(super) fn min_share(&self) -> usize {
        if self.unified { 16 } else { (self.capacity / 16).max(16) }
    }

    /// The scan the trie replaced (kept for `SLLM_PREFIX_CHECK`).
    fn find_prefix_linear(&self, ids: &[i32], skip_sid: Option<i32>, min_k: usize, reserve_last: bool) -> Option<(i32, usize)> {
        let max = if reserve_last { ids.len().saturating_sub(1) } else { ids.len() };
        let mut best: Option<(i32, usize)> = None;
        let mut consider = |sid: i32, tokens: &[i32]| {
            if Some(sid) == skip_sid {
                return;
            }
            let k = tokens.iter().zip(ids).take(max).take_while(|(a, b)| a == b).count();
            if k >= min_k && best.is_none_or(|b| k > b.1) {
                best = Some((sid, k));
            }
        };
        for a in self.agents.values().filter(|a| a.suspended.is_none()) {
            consider(a.seq.sid, &a.seq.tokens);
        }
        for j in self.jobs.values() {
            if let Some(s) = &j.seq {
                consider(s.sid, &s.tokens);
            }
        }
        best
    }

    /// Prefill `ids[from..]` into `sid` synchronously, in chunks of the batch size (no logits wanted).
    pub(super) fn prefill_sync(&mut self, sid: i32, ids: &[i32], from: usize) -> Result<(), String> {
        let cap = self.batch.cap;
        let mut pos = from;
        while pos < ids.len() {
            let end = (pos + cap).min(ids.len());
            let row = Row { req: 0, sid, pos0: pos, toks: Toks::Many(ids[pos..end].to_vec()), samples: false, prefill: true, ignore_eos: false, gram: false, n_draft: 0 };
            let mut evs = Vec::new();
            let mut sampled = Vec::new();
            self.decode_rows(&[row], &mut sampled, &mut evs);
            if let Some(EngineEvent::Failed { message, .. }) = evs.into_iter().next() {
                return Err(message);
            }
            self.pt_extend(sid, &ids[pos..end]);
            pos = end;
        }
        Ok(())
    }

    /// A frozen holder for `ids` (a system prompt that more than one agent uses): prefilled once into a sequence of its own.
    /// None when there is no room for it (it never preempts anything: a holder is a cache, not a promise).
    pub(super) fn make_holder(&mut self, ids: &[i32]) -> Option<i32> {
        if !self.readers.is_empty() {
            return None; // --kv-read f16: a holder filled through the main 4-bit cache would not be read in f16
        }
        if self.holders.len() >= MAX_HOLDERS {
            let i = self.holders.iter().enumerate().min_by_key(|(_, h)| h.last_used).map(|(i, _)| i)?;
            self.evict_holder(i);
        }
        if self.unified && self.committed_cells(None) + ids.len() + MARGIN > self.total_cells {
            return None;
        }
        // the highest free id: running sequences are packed into the lowest ones, a holder in the middle would split every step
        let (i, _) = self.free_sids.iter().enumerate().max_by_key(|(_, &s)| s)?;
        let sid = self.free_sids.swap_remove(i);
        let loaded = self.cache_load(sid, ids, ids.len(), 0);
        if loaded < ids.len() {
            if self.prefill_sync(sid, ids, loaded).is_err() {
                self.release_sid(sid);
                return None;
            }
            self.cache_save(sid, ids);
        }
        self.ptree.borrow_mut().extend(sid, ids);
        self.holders.push(Holder { sid, tokens: ids.to_vec(), last_used: Instant::now() });
        self.holder_stats.created += 1;
        Some(sid)
    }

    /// Holder models: a fresh sequence's prompt that starts like a recent fresh prompt (at least `HOLDER_MIN` tokens, not
    /// counting the last token, which needs logits) makes a holder for the shared start, so this and later prompts copy it
    /// instead of prefilling it. It finds a system prompt sent inside a first turn, as `agent_create` finds a declared one.
    /// `covered`: the start the prompt already gets from a live holder or a saved state; a shared start inside a longer one
    /// makes no holder (the prefill would block the executor for a start the request gets anyway). True when a holder was made.
    pub(super) fn holder_for_start(&mut self, ids: &[i32], covered: usize) -> bool {
        if !self.holders_on || ids.len() <= HOLDER_MIN {
            return false;
        }
        let max = ids.len() - 1;
        let shared = self.recent_starts.iter().map(|p| p.iter().zip(&ids[..max]).take_while(|(a, b)| a == b).count()).max().unwrap_or(0);
        self.recent_starts.push_back(ids[..ids.len().min(RECENT_START_TOKENS)].to_vec());
        if self.recent_starts.len() > RECENT_STARTS {
            self.recent_starts.pop_front();
        }
        let shared = shared / self.share_align * self.share_align;
        if shared < HOLDER_MIN || covered > shared {
            return false;
        }
        let held = self.find_prefix_n(&ids[..shared], None, false).is_some_and(|b| b.1 == shared);
        !held && self.make_holder(&ids[..shared]).is_some()
    }

    /// The longest saved start of `ids` (all but the last token) in the prompt cache on disk, which a fresh sequence can
    /// load ([`Self::cache_load`]): its length, 0 when none.
    pub(super) fn saved_start(&self, ids: &[i32]) -> usize {
        let Some(c) = self.pcache.as_ref() else { return 0 };
        let max = ids.len().saturating_sub(1);
        let best = if self.share { c.longest_common(ids, max, self.share_align) } else { c.longest(ids, max) };
        best.map_or(0, |(_, n)| n)
    }

    /// Load the longest saved start of `ids[..max]` into the empty sequence `sid` when it is longer than `live` (the
    /// longest live match); returns its length, 0 when nothing was loaded.
    pub(super) fn cache_load(&mut self, sid: i32, ids: &[i32], max: usize, live: usize) -> usize {
        let Some((n, saved, buf)) = self.read_saved_start(ids, max, live) else { return 0 };
        let t0 = Instant::now();
        for rd in &self.readers {
            rd.cells_changed(sid, 0);
        }
        // SAFETY: `buf` holds a state libllama wrote for this model and context layout (the directory's fingerprint)
        self.flush_copies(sid);
        let got = unsafe { ffi::llama_state_seq_set_data(self.ctx, buf.as_ptr(), buf.len(), sid) };
        self.ssd_reads += 1;
        self.stall.apply(t0.elapsed().as_secs_f64() * 1e3);
        if got == 0 || (n < saved && !self.seq_rm(sid, n as i32, -1)) {
            self.seq_rm(sid, -1, -1);
            return 0;
        }
        self.pt_set(sid, &ids[..n]);
        if let Some(c) = self.pcache.as_mut() {
            c.hit_tokens += n as u64;
        }
        n
    }

    /// The longest saved start of `ids[..max]` in the prompt cache that is longer than `live`, read from the disk: its
    /// length, the length of the saved state it is the start of, and the state's bytes.
    fn read_saved_start(&mut self, ids: &[i32], max: usize, live: usize) -> Option<(usize, usize, Vec<u8>)> {
        let align = self.share_align;
        let c = self.pcache.as_mut()?;
        // a cache that can drop a tail takes the longest common start of any saved state; else an exact saved start
        let best = if self.share { c.longest_common(ids, max, align) } else { c.longest(ids, max) };
        let (i, n) = best.filter(|&(_, n)| n >= crate::prompt_cache::MIN_TOKENS && n > live)?;
        let saved = c.tokens(i).len();
        let buf = c.read(i)?;
        Some((n, saved, buf))
    }

    /// The computed tokens of the live sequence `sid` (an agent's, a request's or a holder's).
    fn tokens_of_sid(&self, sid: i32) -> Option<Vec<i32>> {
        if let Some(a) = self.agents.values().find(|a| a.suspended.is_none() && a.seq.sid == sid) {
            return Some(a.seq.tokens.clone());
        }
        if let Some(s) = self.jobs.values().filter_map(|j| j.seq.as_ref()).find(|s| s.sid == sid) {
            return Some(s.tokens.clone());
        }
        self.holders.iter().find(|h| h.sid == sid).map(|h| h.tokens.clone())
    }

    /// A shared start of `k` tokens was attached from the live sequence `src`: save `src` (its cells are computed; the
    /// copy into the new sequence is only made at its next decode) unless a saved state already covers the start.
    pub(super) fn cache_save_shared(&mut self, src: i32, ids: &[i32], k: usize) {
        let covered = self.pcache.as_ref().is_none_or(|c| c.longest_common(ids, k, self.share_align).is_some_and(|(_, n)| n >= k));
        if covered {
            return;
        }
        if let Some(toks) = self.tokens_of_sid(src) {
            self.cache_save(src, &toks);
        }
    }

    /// Save the state of `sid`, which holds exactly `ids`, to the prompt cache when it is not there yet (the write runs
    /// on the I/O thread).
    fn cache_save(&mut self, sid: i32, ids: &[i32]) -> bool {
        let Some(c) = self.pcache.as_ref() else { return false };
        if ids.len() < crate::prompt_cache::MIN_TOKENS || c.contains(ids) {
            return false;
        }
        let t0 = Instant::now();
        self.flush_copies(sid);
        let size = unsafe { ffi::llama_state_seq_get_size(self.ctx, sid) };
        if size == 0 {
            return false;
        }
        let mut buf = self.io.take(size);
        // SAFETY: `buf` has room for `size` bytes; libllama writes at most `size` and returns how many.
        let got = unsafe { ffi::llama_state_seq_get_data(self.ctx, buf.as_mut_ptr(), size, sid) };
        unsafe { buf.set_len(got.min(buf.capacity())) };
        self.stall.suspend(t0.elapsed().as_secs_f64() * 1e3);
        match self.pcache.as_mut().and_then(|c| if got > 0 { c.insert(ids, got as u64) } else { None }) {
            Some(path) => {
                let key = self.next_key;
                self.next_key += 1;
                self.ssd_writes += 1;
                self.io.write(key, path, buf);
                true
            }
            None => {
                self.io.give(buf);
                false
            }
        }
    }

    /// A request ended with `seq` holding its prompt and reply: save that state to the prompt cache, so the next turn of the
    /// conversation loads it (also after a restart, or once the holders moved on). The copy runs on this thread, so it is
    /// done only when no other request is running; otherwise it is skipped and counted. Not with `--kv-read f16`: the
    /// readers' f16 caches are not what the main cache holds, a state saved from it would not match a cold read.
    pub(super) fn turn_save_at_end(&mut self, seq: &Seq, complete: bool) {
        let enabled = self.turn_save && self.pcache.is_some() && self.readers.is_empty();
        match crate::prompt_cache::turn_save(enabled, complete, seq.tokens.len(), self.jobs.len()) {
            crate::prompt_cache::TurnSave::No => {}
            crate::prompt_cache::TurnSave::Skip => {
                if let Some(c) = self.pcache.as_mut() {
                    c.turn_save_skips += 1;
                }
            }
            crate::prompt_cache::TurnSave::Save => {
                // what speculation may have left past the tokens must not be in the state (a cache that cannot cut: no save)
                if self.spec_state.contains_key(&seq.sid) && !self.seq_rm(seq.sid, seq.tokens.len() as i32, -1) {
                    return;
                }
                if self.cache_save(seq.sid, &seq.tokens) {
                    if let Some(c) = self.pcache.as_mut() {
                        c.turn_saves += 1;
                    }
                }
            }
        }
    }

    pub(super) fn evict_holder(&mut self, i: usize) {
        let h = self.holders.remove(i);
        self.seq_rm(h.sid, -1, -1);
        self.ptree.borrow_mut().remove(h.sid);
        self.release_sid(h.sid);
        self.holder_stats.evicted += 1;
    }

    pub(super) fn holder_cells(&self) -> usize {
        self.holders.iter().map(|h| h.tokens.len()).sum()
    }

    /// A finished request's sequence stays as a holder until its room is needed ([`Self::take_warm`] reuses it).
    pub(super) fn keep_warm(&mut self, seq: Seq) {
        if self.holders.len() >= MAX_HOLDERS {
            if let Some(i) = self.holders.iter().enumerate().min_by_key(|(_, h)| h.last_used).map(|(i, _)| i) {
                self.evict_holder(i);
            }
        }
        self.spec_state.remove(&seq.sid);
        self.seq_rm(seq.sid, seq.tokens.len() as i32, -1); // what speculation may have left past the tokens
        self.holders.push(Holder { sid: seq.sid, tokens: seq.tokens, last_used: Instant::now() });
        self.holder_stats.warm_kept += 1;
    }

    /// Holder models, before a request's last prefill chunk: its stream, which holds exactly `tokens`, is copied to a holder
    /// when a stream is free after the requests still waiting for one (the copy only evicts the oldest holder, never a
    /// running or queued request's room). With no such stream a lone request's state goes to the prompt cache on disk
    /// instead: a repeat loads it ([`Self::cache_load`]) as it does after a restart.
    pub(super) fn warm_copy(&mut self, src: i32, tokens: Vec<i32>) {
        if self.find_prefix_n(&tokens, None, false).is_some_and(|b| b.1 == tokens.len()) {
            return;
        }
        let waiting = self.jobs.values().filter(|j| !j.begun || (j.agent.is_none() && j.seq.as_ref().is_some_and(|s| s.sid < 0))).count();
        let full = self.holders.len() >= MAX_HOLDERS;
        // evicting the oldest holder gives its stream back, which the copy takes
        if self.free_sids.len() + full as usize <= waiting {
            // the state copy runs on this thread: only when nothing else is running to wait for it
            if self.jobs.len() == 1 {
                self.cache_save(src, &tokens);
            }
            return;
        }
        if full {
            if let Some(i) = self.holders.iter().enumerate().min_by_key(|(_, h)| h.last_used).map(|(i, _)| i) {
                self.evict_holder(i);
            }
        }
        // the highest free id: running sequences are packed into the lowest ones, a holder in the middle would split every step
        let Some((i, _)) = self.free_sids.iter().enumerate().max_by_key(|(_, &s)| s) else { return };
        let dst = self.free_sids.swap_remove(i);
        self.stream_copy(src, dst);
        self.ptree.borrow_mut().extend(dst, &tokens);
        self.holders.push(Holder { sid: dst, tokens, last_used: Instant::now() });
        self.holder_stats.warm_kept += 1;
    }

    /// The warm holder `ids` starts like, when the shared start is most of it: its sequence, cut to the shared start, and
    /// the start's length. A smaller share copies from the holder like any live sequence, which keeps the rest.
    pub(super) fn take_warm(&mut self, ids: &[i32]) -> Option<(i32, usize)> {
        if !self.share {
            return None;
        }
        let (sid, k) = self.find_prefix(ids, None)?;
        let i = self.holders.iter().position(|h| h.sid == sid)?;
        if k * 2 < self.holders[i].tokens.len() {
            return None;
        }
        self.holders.remove(i);
        self.seq_rm(sid, k as i32, -1);
        self.holder_stats.warm_taken += 1;
        Some((sid, k))
    }
}
