//! Sequence ids and KV streams: allocating and releasing ids, removing cells, stream copies and the copy queue, and
//! packing running sequences into consecutive ids.

use super::*;

/// What the stream copies libllama has queued for the next decode of the main context need of the device copy
/// (`LlamaEngine::stream_copy`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum CopyQueue {
    /// No copy is queued.
    #[default]
    Idle,
    /// The first rows of each stream carry every cell of every queued copy.
    Rows(usize),
    /// At least one queued copy needs a whole stream (or nothing is known about its cells).
    Whole,
}

impl CopyQueue {
    /// The queue after one more copy; `rows` None: that copy needs the whole stream. Only ever grows, until a decode applies it.
    pub(super) fn queue(self, rows: Option<usize>) -> Self {
        match (self, rows) {
            (CopyQueue::Whole, _) | (_, None) | (_, Some(0)) => CopyQueue::Whole,
            (CopyQueue::Rows(a), Some(b)) => CopyQueue::Rows(a.max(b)),
            (CopyQueue::Idle, Some(b)) => CopyQueue::Rows(b),
        }
    }

    /// The row count `ggml_backend_metal_sllm_set_copy_rows` gets for the decode that applies the queue (0: whole streams).
    pub(super) fn hint(self) -> i64 {
        match self {
            CopyQueue::Rows(n) => n as i64,
            CopyQueue::Idle | CopyQueue::Whole => 0,
        }
    }
}

/// Steps that decoded rows in the main cache and the runs of consecutive sequence ids in them (one run = one graph at least).
#[derive(Default)]
pub(super) struct StreamRuns {
    pub(super) steps: u64,
    pub(super) runs: u64,
    pub(super) split_steps: u64,
    pub(super) max_runs: u64,
    /// Steps that held rows of 8 or more sequences, and the runs of those.
    pub(super) wide_steps: u64,
    pub(super) wide_runs: u64,
    /// Sequences moved to a lower free id to close a gap (`SLLM_SEQ_PACK`), and the steps that moved any.
    pub(super) moves: u64,
    pub(super) move_steps: u64,
    move_steps_mark: u64,
    /// Stream copies queued (moves, attaches, warm holders) that carry only the rows a sequence uses, the rows they carry
    /// (`SLLM_COPY_PARTIAL`), and the ones that copy the whole stream.
    pub(super) copies_partial: u64,
    pub(super) copy_rows: u64,
    pub(super) copies_whole: u64,
}

/// Most sequences moved per step. A move is a device copy of a KV stream, deferred to the next decode: of the rows the sequence
/// uses where partial copies are on (plain attention, `SLLM_COPY_PARTIAL`), else of `--ctx` cells per layer. At a long context
/// it is still the cost of a move, so it is bounded: a finishing agent leaves one hole, a few per step keep up with the usual
/// rate, and a burst of finishes is closed over the next steps (the plan always makes progress).
pub(super) const MAX_PACK_MOVES: usize = 4;

/// Runs of consecutive ids in sorted `ids` (one run = one graph at least).
pub(super) fn id_runs(ids: &[i32]) -> u64 {
    if ids.is_empty() { 0 } else { 1 + ids.windows(2).filter(|w| w[1] != w[0] + 1).count() as u64 }
}

/// Moves `(from, to)` that close the gaps between the running sequence ids `rows` (sorted, distinct) with the `free` ids:
/// the highest running id goes to the lowest free id while that is lower. Only when the whole plan would leave fewer runs than
/// now (ids held by idle agents cannot be moved into, so a gap they sit in is left alone); at most `max` moves of it.
pub(super) fn plan_pack(free: &[i32], rows: &[i32], max: usize) -> Vec<(i32, i32)> {
    if id_runs(rows) <= 1 || free.is_empty() || max == 0 {
        return Vec::new();
    }
    let mut free: Vec<i32> = free.to_vec();
    free.sort_unstable();
    let mut cur: Vec<i32> = rows.to_vec();
    let mut plan = Vec::new();
    for &f in &free {
        let Some(&top) = cur.last() else { break };
        if f >= top {
            break;
        }
        cur.pop();
        let at = cur.partition_point(|&r| r < f);
        cur.insert(at, f);
        plan.push((top, f));
    }
    if id_runs(&cur) >= id_runs(rows) {
        return Vec::new();
    }
    plan.truncate(max);
    plan
}

impl LlamaEngine {
    pub(super) fn mem(&self) -> sys::llama_memory_t {
        unsafe { sys::llama_get_memory(self.ctx) }
    }

    pub(super) fn seq_rm(&self, sid: i32, p0: i32, p1: i32) -> bool {
        if sid < 0 {
            return true; // no stream (per-sequence streams: a suspended or empty agent gives its id back)
        }
        if p1 >= 0 {
            // every cut here is a tail (p1 < 0) or a whole sequence; a range with an end could leave a hole, and the rows of a
            // stream are only a prefix without holes: stop trusting them (the copies become whole)
            self.copy_holes.set(true);
        }
        let ok = unsafe { ffi::llama_memory_seq_rm(self.mem(), sid, p0, p1) };
        for rd in &self.readers {
            rd.cells_changed(sid, if p1 < 0 { p0.max(0) as usize } else { 0 });
        }
        if ok && self.share && p1 < 0 {
            // the cached tokens of `sid` end at `p0` now
            if p0 <= 0 {
                self.ptree.borrow_mut().remove(sid);
            } else {
                self.ptree.borrow_mut().truncate(sid, p0 as usize);
            }
        }
        ok
    }

    /// Register tokens the cache of `sid` now holds (when partial copies are valid, the trie holds every sequence).
    pub(super) fn pt_extend(&self, sid: i32, toks: &[i32]) {
        self.update_ptree(sid, |tree| tree.extend(sid, toks));
    }

    pub(super) fn pt_set(&self, sid: i32, toks: &[i32]) {
        self.update_ptree(sid, |tree| tree.set(sid, toks));
    }

    /// `update` on the prefix tree, when partial copies are valid and `sid` is a sequence.
    fn update_ptree(&self, sid: i32, update: impl FnOnce(&mut PrefixTree)) {
        if self.share && sid >= 0 {
            update(&mut self.ptree.borrow_mut());
        }
    }

    /// The lowest free sequence id: libllama batches streams only when their ids are consecutive, so running requests are
    /// packed into the lowest ids.
    fn alloc_sid(&mut self) -> Option<i32> {
        // a warm holder under the free ids and over a running sequence would split every step: its id goes first
        let low = self.free_sids.iter().copied().min()?;
        let run = self.jobs.values().filter_map(|j| j.seq.as_ref()).map(|s| s.sid).filter(|&s| s >= 0).min();
        let warm = self.holders.iter().enumerate().filter(|(_, h)| self.share && h.sid < low && run.is_some_and(|r| r < h.sid)).min_by_key(|(_, h)| h.sid).map(|(i, _)| i);
        if let Some(i) = warm {
            self.evict_holder(i);
        }
        let (i, _) = self.free_sids.iter().enumerate().min_by_key(|(_, &s)| s)?;
        Some(self.free_sids.swap_remove(i))
    }

    /// The rows of the stream of `sid` that can hold cells: `seq_pos_max + 1` as libllama reports it right now, or None (copy the
    /// whole stream) when that cannot be shown to cover them.
    ///
    /// Why it covers every cell: partial copies are on only for plain attention (`share`: no sliding-window ring, no recurrent
    /// state) in per-sequence streams. There a stream holds one sequence, and its cells are the first `c` cells of the stream,
    /// with no hole: `find_slot` gives a new cell the first free one from `head`, `head` stays at or below the first free cell
    /// (`apply_ubatch` sets it behind a placed cell, `seq_rm` moves it back to the first cell it frees), nothing but a cache with
    /// a sliding window lets a token take a used cell, and the one way to free a cell below a used one is a cut with an end
    /// ([`Self::seq_rm`] drops to whole copies if one is ever made). `seq_cp` and `llama_state_seq_set_data` (into a sequence
    /// just emptied) keep that shape: the first `c` cells again. The positions of a sequence are distinct and not negative, so
    /// `c <= pos_max + 1`. A copy applies the cells the stream holds when `seq_cp` is called, and those are what this counts.
    fn copy_rows(&self, sid: i32) -> Option<usize> {
        if !self.copy_partial || self.copy_holes.get() {
            return None;
        }
        let pos_max = unsafe { ffi::llama_memory_seq_pos_max(self.mem(), sid) };
        (pos_max >= 0).then(|| pos_max as usize + 1)
    }

    /// Give the stream of `dst` the cells of sequence `src` (all of them): `llama_memory_seq_cp` across per-sequence streams. libllama
    /// copies the cells' metadata now and queues the data copy, which it runs first thing in the next `llama_decode` of this
    /// context (`decode_rows`), through ggml's tensor copy: a whole stream view per layer, at 40k cells gigabytes on this
    /// thread. The queue takes the most rows any queued copy needs ([`Self::copy_rows`]), and that decode passes them to
    /// ggml-metal (`ffi::CopyHint`), which stops each copy there: the rows past them hold no cell and are masked.
    ///
    /// Every `llama_memory_seq_cp` of the main context between streams goes through here (the test `every_main_stream_copy_is_queued`
    /// checks), because one copy that is missing from the queue would be cut at another copy's row count. A copy that does not
    /// know its rows turns the queue whole, so every failure here is a slower copy, never a short one.
    pub(super) fn stream_copy(&mut self, src: i32, dst: i32) {
        if !self.unified {
            let rows = self.copy_rows(src);
            self.copy_queue = self.copy_queue.queue(rows);
            match rows {
                Some(n) => {
                    self.stream_runs.copies_partial += 1;
                    self.stream_runs.copy_rows += n as u64;
                }
                None => self.stream_runs.copies_whole += 1,
            }
        }
        unsafe { ffi::llama_memory_seq_cp(self.mem(), src, dst, 0, -1) };
    }

    /// Apply the queued stream copies now. libllama runs them at the next decode: a state read before that misses them, and
    /// a state set before that is overwritten by them, so another agent's context ends up under this one. One token in
    /// `sid` (after its cells: their metadata is copied at once) runs them, and goes again.
    pub(super) fn flush_copies(&mut self, sid: i32) {
        if sid < 0 || self.copy_queue == CopyQueue::Idle || std::env::var_os("SLLM_COPY_FLUSH").is_some_and(|v| v == "0") {
            return;
        }
        unsafe {
            let pos = ffi::llama_memory_seq_pos_max(self.mem(), sid) + 1;
            let mut b = Batch::new(1);
            b.set(0, 0, pos, sid, true);
            b.b.n_tokens = 1;
            let _hint = ffi::CopyHint::set(self.copy_queue.hint());
            if sys::llama_decode(self.ctx, b.b) == 0 {
                self.copy_queue = CopyQueue::Idle;
            }
            ffi::llama_memory_seq_rm(self.mem(), sid, pos, -1);
        }
    }

    /// Per-sequence streams, at the sort point of a step (the rows are final, nothing has read or written state of their
    /// sequences since): move the highest running sequence into the lowest free id below it, so the rows keep consecutive
    /// ids and the step stays one graph. Free ids appear under the running ones whenever an agent finishes.
    ///
    /// The move is `seq_cp` then `seq_rm` of the old id. libllama copies across streams at the next `llama_decode` (queued in
    /// `sc_info`, applied in order before the batch is built; llama-kv-cache.cpp `seq_cp`/`update`); the cells' metadata is
    /// copied at once and `seq_rm` only edits the old stream's metadata, so the queued buffer copy still finds the data. Nothing
    /// may touch the new id's state before that decode, which holds here: this runs right before `decode_rows`.
    pub(super) fn pack_rows(&mut self, rows: &mut [Row]) {
        // a holder between running sequences splits every step: its stream goes, and the plan closes the gap
        if let (Some(lo), Some(hi)) = (rows.first().map(|r| r.sid), rows.last().map(|r| r.sid)) {
            while let Some(i) = self.holders.iter().position(|h| lo < h.sid && h.sid < hi) {
                self.evict_holder(i);
            }
        }
        let ids: Vec<i32> = rows.iter().map(|r| r.sid).collect();
        let plan = plan_pack(&self.free_sids, &ids, MAX_PACK_MOVES);
        if plan.is_empty() {
            return;
        }
        for (src, dst) in plan {
            self.pack_move(rows, src, dst);
        }
        rows.sort_by_key(|r| r.sid);
    }

    /// One move of [`Self::pack_rows`]: the running sequence `src` goes to the free id `dst`.
    fn pack_move(&mut self, rows: &mut [Row], src: i32, dst: i32) {
        let Some(i) = self.free_sids.iter().position(|&s| s == dst) else { return };
        let Some(row) = rows.iter_mut().find(|r| r.sid == src) else { return };
        let req = row.req;
        // a sequence with no cells yet takes the lower id without a copy
        let Some(has_cells) = self.jobs.get_mut(&req).and_then(|j| Self::seq_for(&mut self.agents, j)).filter(|s| s.sid == src).map(|s| !s.tokens.is_empty()) else { return };
        self.free_sids.swap_remove(i);
        for rd in &self.readers {
            rd.cells_changed(dst, 0);
        }
        if has_cells {
            self.stream_copy(src, dst);
        }
        if let Some(seq) = self.jobs.get_mut(&req).and_then(|j| Self::seq_for(&mut self.agents, j)) {
            seq.sid = dst;
        }
        row.sid = dst;
        self.sid_moved(src, dst);
        self.release_sid(src); // seq_rm of the old id, back to the free list
        self.stream_runs.moves += 1;
        self.stream_runs.move_steps += (self.stream_runs.move_steps_mark != self.counters.steps) as u64;
        self.stream_runs.move_steps_mark = self.counters.steps;
    }

    /// Everything keyed by a sequence id follows a sequence moved from `src` to `dst`.
    fn sid_moved(&mut self, src: i32, dst: i32) {
        for rd in &self.readers {
            let mut last = rd.last.borrow_mut();
            if let Some(t) = last.remove(&src) {
                last.insert(dst, t);
            }
        }
        self.ptree.borrow_mut().rename(src, dst);
        if let Some(st) = self.spec_state.remove(&src) {
            self.spec_state.insert(dst, st);
        }
    }

    /// The highest free sequence id, for a sequence that only takes cells when its read ends ([`Self::read_in`] moves it
    /// to the lowest then): while it reads, it leaves no hole under the running sequences.
    pub(super) fn alloc_sid_top(&mut self, rank: u8) -> Option<i32> {
        if self.free_sids.is_empty() {
            self.free_once(1, None, rank)?;
        }
        let (i, _) = self.free_sids.iter().enumerate().max_by_key(|(_, &s)| s)?;
        Some(self.free_sids.swap_remove(i))
    }

    /// A sequence id for a sequence of `rank`. Per-sequence streams make ids the scarce resource (each owns a whole stream):
    /// when none is free the least-recently-used idle agent gives its id back (suspended to the SSD), else a request of a
    /// lower rank is paused, as for cells in the unified cache.
    pub(super) fn alloc_sid_for(&mut self, rank: u8, except: Option<&str>) -> Option<i32> {
        if let Some(s) = self.alloc_sid() {
            return Some(s);
        }
        self.free_once(1, except, rank)?;
        self.alloc_sid()
    }

    /// Give `dst` the first `k` tokens of `src`. Unified: the cells are shared. Streams: libllama only copies a whole sequence
    /// ([`Self::stream_copy`]: the data copy comes at the next decode, of the rows `src` uses), so copy it all and cut `dst`
    /// back to `k` tokens.
    pub(super) fn attach_prefix(&mut self, src: i32, dst: i32, k: usize) {
        for rd in &self.readers {
            rd.cells_changed(dst, 0);
        }
        if let Some(h) = self.holders.iter_mut().find(|h| h.sid == src) {
            h.last_used = Instant::now();
            self.holder_stats.attached += 1;
            self.holder_stats.attached_tokens += k as u64;
        }
        if !self.share {
            // a frozen holder: its whole cache is the prefix, so the copy is a full one (valid for every cache type)
            self.stream_copy(src, dst);
        } else if self.unified {
            // one pool: the cells are shared, no data is copied (so nothing for the queue)
            unsafe { ffi::llama_memory_seq_cp(self.mem(), src, dst, 0, k as i32) };
        } else {
            self.stream_copy(src, dst);
            self.seq_rm(dst, k as i32, -1);
        }
    }

    pub(super) fn release_sid(&mut self, sid: i32) {
        if sid < 0 {
            return;
        }
        self.spec_state.remove(&sid);
        self.seq_rm(sid, -1, -1);
        if !self.free_sids.contains(&sid) {
            self.free_sids.push(sid);
        }
    }
}
