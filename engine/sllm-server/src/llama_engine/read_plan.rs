//! `--kv-read f16`: which reader reads a request's prompt, and when (`Engine::prefix_state`).

use super::*;

/// Where a request's prompt is read: in `reader`, which keeps `keep` of what it holds; `read` false: in the main cache.
struct ReadPlan {
    reader: usize,
    keep: usize,
    read: bool,
    /// The cells the read needs: the sequence's cached tokens and the feed.
    cells: usize,
    feed_n: usize,
}

/// The choice of a reader for a read of `target` tokens: the lane's own reader, and what each free reader the read fits
/// would keep of it (None: held, or too small).
struct ReaderPick {
    lane: Lane,
    own: usize,
    target: usize,
    keeps: Vec<Option<usize>>,
}

impl ReaderPick {
    /// Reader `j`, which keeps `k`, is taken over the lane's own: it holds much more of the prompt (`READ_REFILL_MAX` more),
    /// and the chat reader takes a background request only when the rest it still has to run is short.
    fn beats_own(&self, j: usize, k: usize) -> bool {
        let own_keep = self.keeps[self.own].unwrap_or(0);
        j != self.own && k >= own_keep + READ_REFILL_MAX && (self.lane == Lane::Interactive || j != 0 || self.target - k <= READ_REFILL_MAX)
    }

    /// The free reader that holds most of the target, the lane's own on a tie (`read_pick` false: the lane's own only).
    fn best(&self, read_pick: bool) -> Option<usize> {
        let mut best = self.keeps[self.own].map(|_| self.own);
        for (j, k) in self.keeps.iter().enumerate() {
            let Some(k) = *k else { continue };
            if read_pick && self.beats_own(j, k) && best.is_none_or(|b| k > self.keeps[b].unwrap_or(0)) {
                best = Some(j);
            }
        }
        best
    }
}

/// The request waits for a reader.
fn waiting() -> Option<PrefixState> {
    Some(PrefixState::Waiting { tokens: 0, written: 0 })
}

impl Job {
    /// The sequence's cached tokens `h` are only a prefix attached from another sequence (a warm holder, a live sequence or
    /// the SSD): the read may drop them and read the whole prompt.
    fn holds_only_attached(&self, h: &[i32]) -> bool {
        self.shared_n > 0 && h.len() == self.shared_n && self.ids.len() >= self.shared_n && self.ids[..self.shared_n] == h[..] && !self.paused
    }
}

impl LlamaEngine {
    /// Readers reserved for a request that is gone are free again.
    pub(super) fn drop_gone_reservations(&mut self) {
        for i in 0..self.readers.len() {
            if self.readers[i].req.is_some_and(|r| !self.jobs.contains_key(&r)) {
                self.reader_unreserve(i); // its request is gone
            }
        }
    }

    /// Reader `i` is reserved for nobody now; the long reader gives its memory back.
    pub(super) fn reader_unreserve(&mut self, i: usize) {
        let rd = &mut self.readers[i];
        rd.req = None;
        rd.started = false;
        rd.queued.clear();
        rd.opening_for = None;
        self.reader_free(i);
    }

    /// `req` reserved reader `i` while its context was being made (on another thread): ready when it is there.
    pub(super) fn reader_opened(&mut self, req: ReqId, i: usize) -> Result<Option<PrefixState>, EngineError> {
        let rd = &mut self.readers[i];
        let (feed_n, shared_n) = rd.opening_for.map_or((0, 0), |(_, f, s)| (f, s));
        match rd.finish_open(false) {
            Ok(false) => return Ok(waiting()),
            Ok(true) => rd.opening_for = None,
            Err(e) => {
                rd.opening_for = None;
                rd.req = None;
                return Err(EngineError::new(503, "kv_pool_exhausted_error", e));
            }
        }
        let Some(job) = self.jobs.get_mut(&req) else { return Ok(None) };
        job.read = Some(true);
        job.reader = i;
        Ok(Some(PrefixState::Ready { feed_n, shared_n }))
    }

    /// A request that has no reader yet reserves one, or waits while the readers it may use are held.
    pub(super) fn reserve_reader(&mut self, req: ReqId, lane: Lane) -> Result<Option<PrefixState>, EngineError> {
        let Some(job) = self.jobs.get(&req) else { return Ok(None) };
        if job.read.is_some() || job.started.is_some() {
            return Ok(None);
        }
        if self.readers.iter().all(|r| r.req.is_some()) {
            return Ok(self.wait_for_reader(self.reader_for(lane, 0)));
        }
        let (h, feed) = self.read_target(req);
        // one long state out of the reader at a time: a read of what the reader already holds takes less than a second and its
        // conversion seconds, so the states of every waiting agent would pile up in memory (6.7 GB each at 40k cells)
        let target = h.len() + feed.len();
        if target >= READ_THREAD_MIN && !self.handovers.is_empty() {
            return Ok(self.wait_for_reader(self.reader_for(lane, target)));
        }
        match self.plan_read(req, lane, h, feed) {
            Ok(plan) => self.start_read_plan(req, lane, plan),
            Err(state) => Ok(state),
        }
    }

    /// The request waits for a reader: counted on `own`, the reader its lane uses.
    fn wait_for_reader(&mut self, own: usize) -> Option<PrefixState> {
        self.readers[own].stats.waited += 1;
        waiting()
    }

    /// What will end up in the reader: the sequence's cached tokens, then the feed.
    fn read_target(&mut self, req: ReqId) -> (Vec<i32>, Vec<i32>) {
        let Some(j) = self.jobs.get_mut(&req) else { return (Vec::new(), Vec::new()) };
        let (shared, ids) = (j.shared_n.min(j.ids.len()), j.ids.clone());
        match Self::seq_for(&mut self.agents, j) {
            Some(s) => (s.tokens.clone(), s.pending.into_iter().chain(ids[shared..].iter().copied()).collect::<Vec<i32>>()),
            None => (Vec::new(), Vec::new()),
        }
    }

    /// The reader and what it keeps for a read of the cached tokens `h` and the `feed`. A read that cannot be exact in the
    /// reader (too many cached tokens to run again) drops cached tokens that are only an attached prefix and reads the whole
    /// prompt; else it is read in the main cache. `Err`: what `prefix_state` answers now (wait, or the request is gone).
    fn plan_read(&mut self, req: ReqId, lane: Lane, mut h: Vec<i32>, mut feed: Vec<i32>) -> Result<ReadPlan, Option<PrefixState>> {
        let droppable = self.jobs.get(&req).is_some_and(|j| j.holds_only_attached(&h));
        let mut dropped = false;
        loop {
            let (reader, keep) = self.pick_reader(lane, &h, &feed).map_err(|own| self.wait_for_reader(own))?;
            let keep = self.restore_for_read(reader, &mut h, &feed, keep);
            let read = h.len().saturating_sub(keep) <= READ_REFILL_MAX;
            if read || !droppable || dropped {
                return Ok(ReadPlan { reader, keep, read, cells: h.len() + feed.len(), feed_n: feed.len() });
            }
            // drop the attached cells: the whole prompt is the feed, read in the reader
            feed = self.drop_attached(req).ok_or(None)?;
            h.clear();
            dropped = true;
        }
    }

    /// The reader to use: the free one that holds most of the target, the lane's own on a tie; another lane's reader
    /// only when it holds much more, and a background request takes the interactive reader for a short rest only. Returns
    /// it and what it keeps; `Err(own)` when none can take the read now.
    fn pick_reader(&self, lane: Lane, h: &[i32], feed: &[i32]) -> Result<(usize, usize), usize> {
        let target = h.len() + feed.len();
        let own = self.reader_for(lane, target);
        let keeps: Vec<Option<usize>> = self.readers.iter().map(|r| (r.req.is_none() && r.max_cells >= target).then(|| Self::read_keep(r, h, feed))).collect();
        let pick = ReaderPick { lane, own, target, keeps };
        let i = pick.best(self.read_pick).ok_or(own)?;
        Ok((i, pick.keeps[i].unwrap_or(0)))
    }

    /// A saved read is cut back to the shared start: a sliding-window model's ring would not be laid out as the cold
    /// read's (see `Reader::ring`), so those models read again (exact) instead of restoring. Returns what reader `i` keeps.
    fn restore_for_read(&mut self, i: usize, h: &mut Vec<i32>, feed: &[i32], keep: usize) -> usize {
        if self.readers[i].ring || h.len() + feed.len() - keep <= READ_REFILL_MAX {
            return keep;
        }
        let mut rd = self.readers.remove(i);
        let n = h.len();
        h.extend_from_slice(feed);
        self.reader_restore(&mut rd, h, keep);
        h.truncate(n);
        let keep = Self::read_keep(&rd, h, feed);
        self.readers.insert(i, rd);
        keep
    }

    /// Drop the request's attached cells: the whole prompt becomes the feed (returned). None when the request is gone.
    fn drop_attached(&mut self, req: ReqId) -> Option<Vec<i32>> {
        let job = self.jobs.get_mut(&req)?;
        job.shared_n = 0;
        let agent = job.agent.is_some();
        let seq = Self::seq_for(&mut self.agents, job)?;
        seq.tokens.clear();
        let sid = seq.sid;
        if agent {
            self.seq_rm(sid, 0, -1);
        } else {
            // a request waits for its reader without a stream (see try_begin)
            self.release_sid(sid);
            if let Some(seq) = self.jobs.get_mut(&req).and_then(|j| j.seq.as_mut()) {
                seq.sid = -1;
            }
        }
        Some(self.jobs.get(&req).map(|j| j.ids.clone()).unwrap_or_default())
    }

    /// The read is planned: the request gets a stream and its reader (or waits for them), and is ready to read.
    fn start_read_plan(&mut self, req: ReqId, lane: Lane, plan: ReadPlan) -> Result<Option<PrefixState>, EngineError> {
        if !self.read_stream(req, lane) {
            return Ok(waiting());
        }
        if !self.take_reader(req, &plan)? {
            return Ok(waiting());
        }
        let Some(job) = self.jobs.get_mut(&req) else { return Ok(None) };
        job.read = Some(plan.read);
        job.reader = plan.reader;
        Ok(Some(PrefixState::Ready { feed_n: plan.feed_n, shared_n: job.shared_n }))
    }

    /// A request that waited for its reader without a stream takes one now, the highest free id (the read moves it to the
    /// lowest when it ends). False when none can be had.
    fn read_stream(&mut self, req: ReqId, lane: Lane) -> bool {
        let no_sid = self.jobs.get_mut(&req).and_then(|j| Self::seq_for(&mut self.agents, j)).is_some_and(|s| s.sid < 0);
        if !no_sid {
            return true;
        }
        let rank = kvmem::rank(lane);
        let Some(sid) = self.alloc_sid_top(rank) else { return false };
        if let Some(seq) = self.jobs.get_mut(&req).and_then(|j| Self::seq_for(&mut self.agents, j)) {
            seq.sid = sid;
        }
        true
    }

    /// The plan's reader is reserved for `req` (a read in the main cache only counts). False while the long reader's context
    /// for it is being made on another thread (a chat keeps running): the request waits for it.
    fn take_reader(&mut self, req: ReqId, plan: &ReadPlan) -> Result<bool, EngineError> {
        let i = plan.reader;
        if !plan.read {
            self.readers[i].stats.in_main += 1;
            return Ok(true);
        }
        if plan.keep + READ_REFILL_MAX < self.readers[i].tokens.len() {
            let mut rd = self.readers.remove(i);
            self.reader_stash(&mut rd);
            self.readers.insert(i, rd);
        }
        // the long reader's memory for this read (sized to it); the context is made on another thread, the request
        // waits for it (a chat keeps running)
        let shared_n = self.jobs.get(&req).map_or(0, |j| j.shared_n);
        let rd = &mut self.readers[i];
        let ready = rd.ensure_soon(plan.cells).map_err(|e| EngineError::new(503, "kv_pool_exhausted_error", e))?;
        rd.req = Some(req);
        rd.started = false;
        if !ready {
            rd.opening_for = Some((req, plan.feed_n, shared_n));
        }
        Ok(ready)
    }
}
