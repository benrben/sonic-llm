//! Sequence slots and the cache's pages: placing a new sequence, cutting a slot back, room for a request, the pinned starts
//! and the cells kept in free slots.

use super::*;

impl NextEngine {
    /// Keep the first `n` cells of a slot; the rest are zeroed. A model's sliding-window layers keep only the last window
    /// of cells, so a cut far behind the end keeps none, and a recurrent one goes back only to where its last prompt ended:
    /// the tokens [kept, n) it could not keep are returned (to be read again).
    pub(super) fn truncate(&mut self, slot: i32, n: usize) -> Result<Vec<i32>, String> {
        let have = self.cells[slot as usize].len();
        if have <= n {
            return Ok(Vec::new());
        }
        let keep = self.cut_best(slot, n);
        call(|b, l| unsafe { sn_clear(self.e, slot, keep as c_int, have as c_int, b, l) })?;
        let lost = self.cells[slot as usize][keep..n].to_vec();
        self.cells[slot as usize].truncate(keep);
        Ok(lost)
    }

    /// The most of the slot's first `n` cells it can be cut to (or copied as).
    pub(super) fn cut_best(&self, slot: i32, n: usize) -> usize {
        unsafe { sn_cut_best(self.e, slot, n as c_int) }.max(0) as usize
    }

    /// A free slot for a new sequence of `ids`, and how many of them it holds already (at most `max`): a free slot that
    /// still holds a prefix of them (`reuse`), else one with the longest prefix a live slot has copied in.
    pub(super) fn place(&mut self, ids: &[i32], max: usize, reuse: bool, owner: SlotOwner) -> Result<(i32, usize), EngineError> {
        if self.owners.free().is_empty() {
            return Err(self.slots_full());
        }
        let held = |s: i32| self.cut_best(s, lcp(&self.cells[s as usize], ids).min(max));
        let warm = self.owners.free().iter().copied().filter(|_| reuse).map(|s| (held(s), s)).max().filter(|&(n, _)| n >= MIN_SHARE);
        let busy: Vec<i32> = (0..self.cfg.slots as i32).filter(|&s| !self.owners.is_free(s)).collect();
        let live = busy.into_iter().map(|s| (held(s), s)).max().filter(|&(n, s)| n >= if self.pins.contains(&s) { self.pin_min } else { MIN_SHARE });
        let (slot, n) = match (warm, live) {
            (Some((w, s)), l) if l.is_none_or(|(n, _)| w >= n) => {
                self.truncate(s, w).map_err(server_err)?;
                (s, w)
            }
            (_, l) => self.place_empty(ids, l)?,
        };
        self.owners.take(slot, owner).map_err(server_err)?;
        if n > 0 {
            self.prefix.attached_agents += 1;
            self.prefix.attached_tokens += n;
        }
        self.cap_kept_states();  // (slots freed outside `end`, an agent's for one, are counted here at the latest)
        Ok((slot, n))
    }

    /// The free slot that holds the least (kept prefixes last longest), emptied, with the prefix `live` (its length and the
    /// live slot it is in) copied in when there is one.
    fn place_empty(&mut self, ids: &[i32], live: Option<(usize, i32)>) -> Result<(i32, usize), EngineError> {
        let s = *self.owners.free().iter().min_by_key(|&&s| self.cells[s as usize].len()).expect("checked above");
        self.truncate(s, 0).map_err(server_err)?;
        let Some((n, src)) = live else { return Ok((s, 0)) };
        // a copy that cannot be made (a recurrent model with every state entry taken) is no error: it reads them
        if call(|b, len| unsafe { sn_copy(self.e, src, s, n as c_int, b, len) }).is_err() {
            self.reset_slot(s, n);
            return Ok((s, 0));
        }
        if let Some(k) = self.pins.iter().position(|&p| p == src) {
            let p = self.pins.remove(k);
            self.pins.push(p);
            self.pin_used += 1;
            self.pin_tokens += n as u64;
        }
        self.cells[s as usize] = ids[..n].to_vec();
        Ok((s, n))
    }

    /// Every slot is taken: the pool's refusal, short by one slot, so the executor's memory policy can suspend an idle
    /// agent (`--pressure-relief`) and try again, and a resume waits for room instead of dropping the agent's cells.
    pub(super) fn slots_full(&self) -> EngineError {
        let slot_bytes = unsafe { sn_kv_bytes(self.e, self.cfg.n_ctx as c_int) } as u64;
        err(503, "kv_pool_exhausted_error", format!("all {} sequence slots are in use; retry when one ends, free an agent, or start with a larger --kv-pages", self.cfg.slots))
            .with_shortfall(slot_bytes)
            .with_retry_after(200)
    }

    pub(super) fn pages_used(&self) -> usize {
        unsafe { (sn_pages_total(self.e) - sn_pages_free(self.e)) as usize }
    }

    /// Pages a slot holds (`own`: those no other slot shares too).
    pub(super) fn held(&self, slot: i32, own: bool) -> usize {
        let mut o: c_int = 0;
        let n = unsafe { sn_seq_pages(self.e, slot, &mut o) };
        (if own { o } else { n }) as usize
    }

    /// Pages the admitted requests (and the resumes reading in) may still take.
    pub(super) fn reserved(&self) -> usize {
        let jobs: usize = self.jobs.values().map(|j| chunks(j.end).saturating_sub(self.held(j.slot, false))).sum();
        let reads: usize = self.agents.values().filter_map(|a| a.off.as_ref()).filter_map(|o| o.reading.map(|s| chunks(o.tokens.len()).saturating_sub(self.held(s, false)))).sum();
        jobs + reads
    }

    /// Free pages nothing has reserved.
    pub(super) fn room(&self) -> usize {
        (unsafe { sn_pages_free(self.e) } as usize).saturating_sub(self.reserved())
    }

    /// Window pages (a sliding-window model's own pool, capped by the GPU's memory) slot `slot` may still take to reach
    /// `cells` cells.
    pub(super) fn win_need(&self, cells: usize, slot: i32) -> usize {
        unsafe { (sn_win_need(self.e, cells as c_int) - sn_win_pages(self.e, slot)).max(0) as usize }
    }

    /// Window pages nothing has reserved (the admitted requests and the resumes reading in reserve theirs).
    pub(super) fn win_room(&self) -> usize {
        let jobs: usize = self.jobs.values().map(|j| self.win_need(j.end, j.slot)).sum();
        let reads: usize = self.agents.values().filter_map(|a| a.off.as_ref()).filter_map(|o| o.reading.map(|s| self.win_need(o.tokens.len(), s))).sum();
        (unsafe { sn_win_room(self.e) }.max(0) as usize).saturating_sub(jobs + reads)
    }

    /// `need` more pages for slot `keep` (and the window pages to reach `cells` cells): the cells kept in free slots go
    /// first, least recently used first; then the pool's refusal, short by what is missing, for the executor's memory
    /// policy (`--pressure-relief`).
    pub(super) fn make_room(&mut self, need: usize, cells: usize, keep: i32) -> Result<(), EngineError> {
        while self.room() < need || self.win_room() < self.win_need(cells, keep) {
            let victim = self.owners.free().iter().copied().filter(|&s| s != keep && !self.cells[s as usize].is_empty()).min_by_key(|&s| self.last_used[s as usize]);
            let Some(s) = victim else {
                let wshort = self.win_need(cells, keep).saturating_sub(self.win_room());
                let short = need.saturating_sub(self.room()).max(wshort) as u64 * unsafe { sn_page_bytes(self.e) } as u64;
                let msg = if wshort > 0 {
                    format!("the sliding-window layers' cache (what the GPU's memory holds) cannot take {wshort} more pages now; retry when a request ends or free an agent")
                } else {
                    format!("the cache ({} pages of {PAGE} cells) cannot hold {need} more pages now; free an agent or start with a larger --kv-pages", unsafe { sn_pages_total(self.e) })
                };
                return Err(err(503, "kv_pool_exhausted_error", msg).with_shortfall(short).with_retry_after(200));
            };
            self.truncate(s, 0).map_err(server_err)?;
        }
        Ok(())
    }

    /// Where a new prompt's shared start ends: the shortest common start, of at least `pin_min` tokens, it has with a recent
    /// prompt (the shortest: two questions that happen to begin alike must not hide the system prompt they both follow).
    /// The prompt then joins the recent ones.
    pub(super) fn shared_start(&mut self, feed: &[i32]) -> Option<usize> {
        // (two recent prompts must share it: the first two questions after a system prompt may begin alike by chance, and a
        // pin longer than the system prompt serves no other question)
        let ls: Vec<usize> = self.recent.iter().map(|r| lcp(r, feed)).filter(|&n| n >= self.pin_min && n < feed.len()).collect();
        let l = if ls.len() >= 2 { ls.iter().copied().min() } else { None };
        if self.recent.len() >= PIN_RECENT {
            self.recent.pop_front();
        }
        self.recent.push_back(feed[..feed.len().min(1024)].to_vec());
        l
    }

    /// Pin the first `n` cells of `src` (its state is there now) in a slot of their own; the least recently used pin goes
    /// when there are `pin_max`. No free slot or no state entry: no pin, the start is read again next time.
    pub(super) fn pin(&mut self, src: i32, n: usize) {
        if self.pins.iter().any(|&p| self.cells[p as usize].len() == n && self.cells[p as usize][..] == self.cells[src as usize][..n]) {
            return;
        }
        while self.pins.len() >= self.pin_max.max(1) {
            let old = self.pins.remove(0);
            let _ = self.truncate(old, 0);
            self.cells[old as usize].clear();
            self.slot_back(old, SlotOwner::Pinned);
        }
        let Some(&s) = self.owners.free().iter().min_by_key(|&&s| self.cells[s as usize].len()) else { return };
        if self.truncate(s, 0).is_err() {
            return;
        }
        self.cells[s as usize].clear();
        if call(|b, len| unsafe { sn_copy(self.e, src, s, n as c_int, b, len) }).is_ok() {
            self.cells[s as usize] = self.cells[src as usize][..n].to_vec();
            if let Err(m) = self.owners.take(s, SlotOwner::Pinned) {
                eprintln!("next: slot bookkeeping: {m}");
                return;
            }
            self.pins.push(s);
            self.pin_made += 1;
        } else {
            self.reset_slot(s, n);
        }
    }

    /// A recurrent model: the ended sequences kept in free slots hold at most `kept_state` bytes of state entries; above that
    /// the least recently used ones are dropped (their cells and state; the entries go back to the pool, zeroed, for the
    /// next sequence, so the pool stops growing). Sequences of running requests and of agents are never touched.
    pub(super) fn cap_kept_states(&mut self) {
        if !self.recurrent {
            return;
        }
        let e = self.e;
        let per = unsafe { sn_state_bytes(e) };
        let entries = move |s: i32| unsafe { sn_state_entries(e, s) }.max(0) as usize;
        let mut kept: Vec<i32> = self.owners.free().iter().copied().filter(|&s| entries(s) > 0).collect();
        kept.sort_by_key(|&s| std::cmp::Reverse(self.last_used[s as usize]));  // the least recently used last
        let mut held: usize = kept.iter().map(|&s| entries(s) * per).sum();
        while held > self.kept_state {
            let Some(s) = kept.pop() else { break };
            let was = entries(s) * per;
            if self.truncate(s, 0).is_err() {
                break;
            }
            self.cells[s as usize].clear();
            held -= was.min(held);
        }
    }

    /// Idle long enough: free slots' cells dropped, least recently used first, until the cache holds at most `idle_cache`
    /// bytes; then sn_trim gives the memory back.
    pub(super) fn drop_idle_cache(&mut self) {
        let mut k = 0;
        while k < self.owners.free().len() && unsafe { sn_mem_used(self.e) } > self.idle_cache {
            let s = self.owners.free()[k];
            if !self.cells[s as usize].is_empty() && self.truncate(s, 0).is_ok() {
                self.cells[s as usize].clear();
            }
            k += 1;
        }
        self.give_back(0);
        if std::env::var_os("SLLM_LOG_TRIM").is_some() {
            eprintln!("next: idle drop: {k} free slots looked at, cache now {} MB", unsafe { sn_mem_used(self.e) } >> 20);
        }
    }

    /// sn_trim: memory above what the cache holds (plus `spare` chunks) back to the system
    pub(super) fn give_back(&mut self, spare: i32) {
        if self.trim {
            self.trim_at = std::time::Instant::now();
            unsafe { sn_trim(self.e, spare) };
            self.trim_pending = unsafe { sn_trim_pending(self.e) } > 0;
        }
    }

    /// A request refused after `place`: a new sequence gives its slot back (an agent keeps one it already had).
    pub(super) fn unplace(&mut self, req: &StartRequest, slot: i32, base: usize, shared: usize) {
        let Some(id) = req.agent.clone() else { return self.slot_back(slot, SlotOwner::Job(req.id)) };
        if let Some(a) = self.agents.get_mut(&id) {
            if base == 0 && a.slot == Some(slot) && self.cells[slot as usize].len() == shared {
                a.slot = None;
                self.slot_back(slot, SlotOwner::Agent(id));
            }
        }
    }

    /// sn.h's rule for its callers: a slot that an sn_copy, sn_kv_load or step failed on may be half made (pages shared,
    /// a recurrent state moved), so it is cleared from cell 0 up to `to` (past anything the failed call wrote) and holds
    /// nothing.
    pub(super) fn reset_slot(&mut self, s: i32, to: usize) {
        if let Err(m) = call(|b, l| unsafe { sn_clear(self.e, s, 0, to as c_int, b, l) }) {
            eprintln!("next: slot {s}: clearing it after a failed call: {m}");
        }
        self.cells[s as usize].clear();
    }

    /// A recurrent model's step failed: the states of its slots (rows `seq` at `pos`) may have moved past what their cells
    /// say, so each slot is reset; an agent's tokens become its unread ones, read again with its next turn.
    pub(super) fn reset_after_failed_step(&mut self, seq: &[i32], pos: &[i32]) {
        if !self.recurrent {
            return;
        }
        let mut ends: Vec<(i32, usize)> = Vec::new();
        for (&s, &p) in seq.iter().zip(pos) {
            match ends.iter_mut().find(|e| e.0 == s) {
                Some(e) => e.1 = e.1.max(p as usize + 1),
                None => ends.push((s, p as usize + 1)),
            }
        }
        for (s, end) in ends {
            let held = std::mem::take(&mut self.cells[s as usize]);
            self.reset_slot(s, end.max(held.len()));
            if let Some(a) = self.agents.values_mut().find(|a| a.slot == Some(s)) {
                a.pending = held.into_iter().chain(std::mem::take(&mut a.pending)).collect();
            }
        }
    }

    /// Slot `s`, held by `from`, back to the free slots. A refusal (it is not `from`'s: freed already) is logged and changes
    /// nothing, so a running sequence's slot is never handed out twice.
    pub(super) fn slot_back(&mut self, s: i32, from: SlotOwner) {
        if let Err(m) = self.owners.give_back(s, &from) {
            eprintln!("next: slot bookkeeping: {m}");
        }
    }
}
