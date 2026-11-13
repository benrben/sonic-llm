//! `--kv-read f16` on the engine's side: reading prompt chunks in a reader, saving long reads to the SSD, and
//! handing the converted state over to the main cache.

use super::*;

/// A long read whose conversion runs on its own thread; [`LlamaEngine::poll_handovers`] sets it into the main cache.
pub(super) struct Handover {
    req: ReqId,
    sid: i32,
    tok: i32,
    tokens: Vec<i32>,
    reader: usize,
    t0: Instant,
    /// The converted state, and the f16 state it was made from (kept for `stash`).
    work: std::thread::JoinHandle<(Result<Vec<u8>, String>, Vec<u8>)>,
    /// The read the f16 state is saved under when the conversion is done (the long reader gave its memory back).
    stash: Option<Vec<i32>>,
}

/// A read's state on its way to the main cache: rewritten for the main cache's layout ([`crate::kvconv`]).
struct Conversion {
    /// The reader's f16 state (kept for a save to the SSD).
    state: Vec<u8>,
    layout: crate::kvconv::Layout,
    sid: i32,
    threads: usize,
    /// A splice: the main cache's own state of `sid` and the position its cells are kept below.
    keep: Option<(Vec<u8>, i32)>,
}

impl Conversion {
    /// The converted state, and the f16 state it was made from.
    fn run(self) -> (Result<Vec<u8>, String>, Vec<u8>) {
        (crate::kvconv::convert(&self.state, &self.layout, self.sid, self.threads, self.keep.as_ref().map(|(m, p)| (&m[..], *p))), self.state)
    }
}

/// The state of sequence `seq` of the context `ctx` (`llama_state_seq_get_data`).
fn state_of(ctx: *mut sys::llama_context, seq: i32) -> Result<Vec<u8>, String> {
    let n = unsafe { ffi::llama_state_seq_get_size(ctx, seq) };
    let mut st = vec![0u8; n];
    let got = unsafe { ffi::llama_state_seq_get_data(ctx, st.as_mut_ptr(), n, seq) };
    if got == 0 {
        return Err("--kv-read: llama_state_seq_get_data failed".into());
    }
    st.truncate(got);
    Ok(st)
}

impl Seq {
    /// A sequence without cells yet takes the lowest free id below its own (its own id goes back in its place).
    fn take_lower_free_id(&mut self, free_sids: &mut [i32]) {
        if !self.tokens.is_empty() {
            return;
        }
        if let Some(low) = free_sids.iter_mut().filter(|s| **s < self.sid).min() {
            std::mem::swap(low, &mut self.sid);
        }
    }
}

impl LlamaEngine {
    /// `--kv-read f16`: a prefill chunk of the request the reader is reserved for runs in the reader. The last chunk samples
    /// the first token and moves the state to the request's sequence in the main cache. A failure fails the request and
    /// empties the reader.
    pub(super) fn read_chunk(&mut self, req: ReqId, toks: Vec<i32>, samples: bool, sampled: &mut Vec<(ReqId, i32)>, events: &mut Vec<EngineEvent>) {
        let Some(i) = self.jobs.get(&req).map(|j| j.reader.min(self.readers.len().saturating_sub(1))) else { return };
        let mut rd = self.readers.remove(i);
        if let Err(e) = self.read_in(&mut rd, i, req, toks, samples, sampled) {
            rd.stats.failures += 1;
            rd.reset();
            events.push(EngineEvent::Failed { req, message: e });
        }
        self.readers.insert(i, rd);
        self.reader_release();
    }

    /// A request has not got a reader yet.
    fn read_waiting(&self) -> bool {
        self.jobs.values().any(|j| j.read.is_none() && j.started.is_none() && !j.paused)
    }

    /// Nobody waits to read: a long reader left open for a waiter gives its memory back.
    pub(super) fn reader_release(&mut self) {
        if self.read_waiting() {
            return;
        }
        for i in 0..self.readers.len() {
            if self.readers[i].req.is_none() && self.readers[i].big() {
                self.reader_free(i);
            }
        }
    }

    /// The reader a request tries first for a read of `target` tokens: an interactive prompt that fits the chat reader goes
    /// there, every other read to the long reader. (A chat that has outgrown the chat reader waits for the long reader, and is
    /// read there in the same blocks as it would have been anywhere: the same cells.)
    pub(super) fn reader_for(&self, lane: Lane, target: usize) -> usize {
        own_reader(lane, target, self.readers[0].max_cells, self.readers.len())
    }

    /// A read of `tokens` is worth saving to the SSD (`rcache`): too long to run again in a reader, and not saved yet.
    fn stash_wanted(&self, tokens: &[i32]) -> bool {
        tokens.len() > READ_REFILL_MAX && self.rcache.as_ref().is_some_and(|c| !c.contains(tokens))
    }

    /// Save `tokens`' f16 state `buf` to the SSD (the write runs on the I/O thread, which keeps `buf` until it is done).
    /// False when it was not saved (no room, or already there).
    fn stash_bytes(&mut self, tokens: &[i32], buf: Vec<u8>) -> bool {
        let Some(path) = self.rcache.as_mut().and_then(|c| c.insert(tokens, buf.len() as u64)) else {
            self.io.give(buf);
            return false;
        };
        let key = self.next_key;
        self.next_key += 1;
        self.ssd_writes += 1;
        self.rstash.push((key, tokens.to_vec()));
        self.io.write(key, path, buf);
        true
    }

    /// `rd`'s read is about to leave the reader: save it (the write runs on the I/O thread) when it is too long to run again.
    pub(super) fn reader_stash(&mut self, rd: &mut Reader) {
        if rd.ctx.is_null() || !self.stash_wanted(&rd.tokens) {
            return;
        }
        let size = unsafe { ffi::llama_state_seq_get_size(rd.ctx, 0) };
        let mut buf = self.io.take(size);
        // SAFETY: `buf` has room for `size` bytes; libllama writes at most `size` and returns how many.
        let got = unsafe { ffi::llama_state_seq_get_data(rd.ctx, buf.as_mut_ptr(), size, 0) };
        unsafe { buf.set_len(got.min(buf.capacity())) };
        if got > 0 && self.stash_bytes(&rd.tokens, buf) {
            rd.stats.stashed += 1;
        }
    }

    /// Wait for the saves to the SSD that start like `ids` (at least `align` tokens alike): a read of one that is still being
    /// written would find the file short and drop the entry.
    /// Returns the milliseconds it waited.
    fn stash_wait(&mut self, ids: &[i32], align: usize) -> f64 {
        let t0 = Instant::now();
        let alike = |t: &[i32]| t.iter().zip(ids).take_while(|(a, b)| a == b).count() >= align;
        while self.rstash.iter().any(|(_, t)| alike(t)) {
            self.poll_io();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        t0.elapsed().as_secs_f64() * 1e3
    }

    /// `rd` has nothing to read for anyone: the long reader gives its memory back, its read saved first (a read that is
    /// already saved, or too short, is not).
    pub(super) fn reader_free(&mut self, i: usize) {
        let mut rd = self.readers.remove(i);
        let _ = rd.finish_open(true); // a context nobody waits for any more is still freed below
        if rd.big() && self.rcache.is_some() {
            self.reader_stash(&mut rd);
            rd.close();
        }
        self.readers.insert(i, rd);
    }

    /// How many tokens `rd` keeps for a request: its tokens that match the sequence's cached tokens `h` and then the feed
    /// (what will end up in the reader), back to a block boundary, with at least one token left to run. A context too small
    /// for them is made again, so it keeps nothing.
    pub(super) fn read_keep(rd: &Reader, h: &[i32], feed: &[i32]) -> usize {
        if rd.cells < h.len() + feed.len() {
            return 0;
        }
        let common = rd.tokens.iter().zip(h.iter().chain(feed)).take_while(|(a, b)| a == b).count();
        common.min((h.len() + feed.len()).saturating_sub(1)) / rd.block * rd.block
    }

    /// Load into `rd` the saved read that shares the longest start with `h`, when that holds more of `h` than `rd` does.
    /// The file is read on another thread while the long reader's context is made, and a save still being written is waited
    /// for only after that.
    pub(super) fn reader_restore(&mut self, rd: &mut Reader, h: &[i32], keep: usize) {
        let Some(c) = self.rcache.as_ref() else { return };
        let Some((i, n)) = c.longest_common(h, h.len(), rd.block) else { return };
        if n <= keep {
            return;
        }
        let saved = c.tokens(i).to_vec();
        self.reader_stash(rd);
        // the stash may have dropped entries: look the read up again
        let Some(i) = self.rcache.as_ref().and_then(|c| c.longest_common(h, h.len(), rd.block).filter(|b| b.1 == n).map(|b| b.0)) else { return };
        let (path, bytes) = self.rcache.as_ref().expect("looked up above").file(i);
        let pending = self.rstash.iter().any(|(_, t)| *t == saved);
        let file = (!pending).then(|| {
            let path = path.clone();
            std::thread::spawn(move || std::fs::read(path).ok().filter(|b| b.len() as u64 == bytes))
        });
        let made = rd.ensure(h.len().max(saved.len()));
        if pending {
            rd.stats.stash_wait_ms += self.stash_wait(&saved, saved.len());
        }
        let tr = Instant::now();
        let buf = match file {
            Some(t) => t.join().ok().flatten(),
            None => std::fs::read(&path).ok().filter(|b| b.len() as u64 == bytes),
        };
        rd.stats.restore_read_ms += tr.elapsed().as_secs_f64() * 1e3;
        if let Some(c) = self.rcache.as_mut() {
            c.loaded(i, buf.is_some());
        }
        let Some(buf) = buf.filter(|_| made.is_ok()) else { return };
        self.ssd_reads += 1;
        let ts = Instant::now();
        // SAFETY: `buf` holds a state libllama wrote for a reader of this model and layout (the directory's fingerprint)
        let got = unsafe { ffi::llama_state_seq_set_data(rd.ctx, buf.as_ptr(), buf.len(), 0) };
        rd.stats.restore_set_ms += ts.elapsed().as_secs_f64() * 1e3;
        rd.tokens = if got == 0 { Vec::new() } else { saved };
        rd.stats.restored += (got != 0) as u64;
    }

    fn read_in(&mut self, rd: &mut Reader, reader: usize, req: ReqId, toks: Vec<i32>, samples: bool, sampled: &mut Vec<(ReqId, i32)>) -> Result<(), String> {
        if rd.req != Some(req) || rd.ctx.is_null() {
            return Err("--kv-read: the reader is not reserved for this request".into());
        }
        let Some((first, sid)) = self.read_run(rd, req, toks, samples)? else { return Ok(()) };
        if !samples {
            return Ok(());
        }
        self.read_out(rd, reader, req, sid, first, sampled)
    }

    /// Run a chunk of `req`'s read in `rd` (a read's first chunk sets the reader up for it). Returns the first token when
    /// the chunk `samples` (the read's last), and the sequence id the state goes to; None when the request is gone.
    fn read_run(&mut self, rd: &mut Reader, req: ReqId, toks: Vec<i32>, samples: bool) -> Result<Option<(Vec<(ReqId, i32)>, i32)>, String> {
        let Some(job) = self.jobs.get_mut(&req) else { return Ok(None) };
        let ignore_eos = job.params.ignore_eos;
        let gram: *mut Gram = job.gram.as_deref_mut().map_or(std::ptr::null_mut(), |g| g);
        job.fed += toks.len();
        let feed = if rd.started { Vec::new() } else { job.feed.clone() };
        let Some(seq) = Self::seq_for(&mut self.agents, job) else { return Err("the request's sequence is gone".into()) };
        if !rd.started {
            rd.start_read(&seq.tokens, &feed, seq.sid);
        }
        let n = rd.queue_feed(&toks);
        self.counters.prompt_tokens_evaluated += n as u64;
        self.last_bd.n_prefill_tokens += n;
        let t0 = Instant::now();
        let mut first = Vec::new();
        if let Some(row) = rd.run(samples)? {
            if gram.is_null() {
                sample_rows(&[(req, ignore_eos, row)], &self.eog, &mut first);
            } else {
                // SAFETY: the job owns the boxed Gram and nothing since the pointer was taken has removed the job
                first.push((req, unsafe { &mut *gram }.pick(row, ignore_eos, &self.eog)));
            }
        }
        let us = t0.elapsed().as_micros() as u64;
        self.last_bd.graphs += 1;
        self.last_bd.compute_us += us;
        self.last_bd.total_us += us;
        if samples {
            // the state goes to the main cache; a sequence without cells yet takes the lowest free id (see step)
            seq.take_lower_free_id(&mut self.free_sids);
        }
        Ok(Some((first, seq.sid)))
    }

    /// The read in `rd` is complete: its state leaves the reader for the main cache's sequence `sid`, converted on its own
    /// thread when it is long ([`Self::poll_handovers`] sets it), else here; `first` is the request's first token.
    fn read_out(&mut self, rd: &mut Reader, reader: usize, req: ReqId, sid: i32, first: Vec<(ReqId, i32)>, sampled: &mut Vec<(ReqId, i32)>) -> Result<(), String> {
        let tc = Instant::now();
        let (state, keep) = self.read_state(rd, sid)?;
        rd.req = None;
        rd.started = false;
        let tokens = rd.tokens.clone();
        let stash = self.reader_after_read(rd, &tokens, keep.is_some()).then(|| tokens.clone());
        let conv = Conversion { state, layout: rd.layout.clone(), sid, threads: rd.threads, keep };
        if tokens.len() >= READ_THREAD_MIN && rd.thread_on {
            let Some(&(_, tok)) = first.first() else { return Err("--kv-read: the read sampled nothing".into()) };
            self.hand_over(rd, Handover { req, sid, tok, tokens, reader, t0: tc, work: std::thread::spawn(move || conv.run()), stash });
            return Ok(());
        }
        let (conv, state) = conv.run();
        if let Some(t) = stash {
            rd.stats.stashed += self.stash_bytes(&t, state) as u64;
        }
        self.read_set(rd, req, sid, &conv?, tokens, tc)?;
        sampled.extend(first);
        Ok(())
    }

    /// The read's state out of `rd`: with a splice only the reader's cells from `splice` on, and the main cache's own state
    /// of `sid` to keep below them; else the reader's whole sequence.
    fn read_state(&self, rd: &mut Reader, sid: i32) -> Result<(Vec<u8>, Option<(Vec<u8>, i32)>), String> {
        if rd.splice == 0 {
            return Ok((state_of(rd.ctx, 0)?, None));
        }
        // only the reader's cells from `splice` on (tagged as its sequence 1), the main cache's own below
        let rmem = unsafe { sys::llama_get_memory(rd.ctx) };
        unsafe { ffi::llama_memory_seq_cp(rmem, 0, 1, rd.splice as i32, -1) };
        let part = state_of(rd.ctx, 1);
        unsafe { ffi::llama_memory_seq_rm(rmem, 1, -1, -1) };
        let main = state_of(self.ctx, sid)?;
        rd.stats.spliced_cells += rd.splice as u64;
        Ok((part?, Some((main, rd.splice as i32))))
    }

    /// A long read gives the reader's memory back as soon as its state is out: it is saved to the SSD (from the state in
    /// hand when that is the whole of it) and read again from there when the prompt returns. With another read waiting the
    /// reader stays open (`read_waiting`): agents on one long document would each load it from the SSD otherwise. True when
    /// the read is to be saved from the state in hand (`spliced`: only the new cells are in hand, it is saved from `rd` here).
    fn reader_after_read(&mut self, rd: &mut Reader, tokens: &[i32], spliced: bool) -> bool {
        let stash = self.rcache.is_some() && rd.big() && self.stash_wanted(tokens);
        if stash && spliced {
            self.reader_stash(rd); // only the new cells are in hand
        }
        if self.rcache.is_some() && rd.big() && !self.read_waiting() {
            rd.close();
        }
        stash && !spliced
    }

    /// A long read's conversion runs on its own thread: its request waits for it ([`Self::poll_handovers`]).
    fn hand_over(&mut self, rd: &mut Reader, h: Handover) {
        let req = h.req;
        self.handovers.push(h);
        rd.stats.states_out_max = rd.stats.states_out_max.max(self.handovers.len() as u64);
        if let Some(job) = self.jobs.get_mut(&req) {
            job.handover = true;
        }
    }

    /// A read's converted state goes into the main cache: `req`'s sequence holds `tokens` now.
    fn read_set(&mut self, rd: &mut Reader, req: ReqId, sid: i32, conv: &[u8], tokens: Vec<i32>, tc: Instant) -> Result<(), String> {
        self.flush_copies(sid);
        let Some(seq) = self.jobs.get_mut(&req).and_then(|job| Self::seq_for(&mut self.agents, job)) else { return Ok(()) };
        let ts = Instant::now();
        let set = unsafe { ffi::llama_state_seq_set_data(self.ctx, conv.as_ptr(), conv.len(), sid) };
        rd.stats.convert_ms += ts.duration_since(tc).as_secs_f64() * 1e3;
        rd.stats.set_ms += ts.elapsed().as_secs_f64() * 1e3;
        rd.stats.state_bytes += conv.len() as u64;
        let us = ts.elapsed().as_micros() as u64;
        self.last_bd.compute_us += us;
        self.last_bd.total_us += us;
        if set == 0 {
            // the set emptied the sequence first: its history is gone
            seq.tokens.clear();
            return Err("--kv-read: llama_state_seq_set_data into the main cache failed".into());
        }
        rd.stats.reads += 1;
        // the other readers' record of this sequence is stale now: its cells are this read's
        for o in &self.readers {
            o.last.borrow_mut().remove(&sid);
        }
        rd.last.borrow_mut().insert(sid, tokens.clone());
        seq.tokens = tokens;
        if self.share {
            let seq_tokens = seq.tokens.clone();
            self.pt_set(sid, &seq_tokens);
        }
        Ok(())
    }

    /// Conversions that ended go into the main cache and their requests get their first token.
    pub(super) fn poll_handovers(&mut self, events: &mut Vec<EngineEvent>) {
        let mut i = 0;
        while i < self.handovers.len() {
            if !self.handovers[i].work.is_finished() {
                i += 1;
                continue;
            }
            let h = self.handovers.swap_remove(i);
            let (conv, state) = h.work.join().unwrap_or_else(|_| (Err("--kv-read: the conversion panicked".into()), Vec::new()));
            let stashed = match h.stash {
                Some(t) => self.stash_bytes(&t, state),
                None => false,
            };
            self.readers[h.reader].stats.stashed += stashed as u64;
            if self.jobs.get_mut(&h.req).map(|j| j.handover = false).is_none() {
                continue; // cancelled meanwhile
            }
            let mut rd = self.readers.remove(h.reader);
            let done = conv.and_then(|c| self.read_set(&mut rd, h.req, h.sid, &c, h.tokens, h.t0));
            if done.is_err() {
                rd.stats.failures += 1;
            }
            self.readers.insert(h.reader, rd);
            match done {
                Ok(()) => {
                    self.on_token(h.req, h.tok, events);
                }
                Err(message) => events.push(EngineEvent::Failed { req: h.req, message }),
            }
        }
    }
}
