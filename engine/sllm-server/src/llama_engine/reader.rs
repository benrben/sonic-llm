//! The f16 readers of `--kv-read f16` ([`Reader`]) and their sizes.

use super::*;

/// Most cached tokens a read may have to run again in the reader (an agent's history it does not hold): above it the
/// turn is read in the main cache.
pub(super) const READ_REFILL_MAX: usize = 4096;
/// A read this long is converted on its own thread (a shorter one takes less than a step).
pub(super) const READ_THREAD_MIN: usize = 2048;

/// Cells of the chat reader (`--kv-read f16`): an interactive prompt that fits it is read there, so a chat never waits behind a
/// long background read. Its context stays allocated.
pub const CHAT_READER_CELLS: usize = 4096;
/// The long reader's context is made for a read, in steps of this many cells; between reads it keeps no more than this (a
/// longer read's state is saved to the SSD and the memory goes back).
const READER_STEP: usize = 4096;
/// What a reader's compute buffer is taken to cost when libllama cannot be asked.
pub(super) const READER_COMPUTE_FALLBACK: u64 = 512 << 20;

/// From this position on a read runs in blocks of 256: a block's cost grows with the position, and it holds the GPU, so
/// a chat waits half as long behind one. Below it short prompts and bursts keep their larger blocks.
const LONG_READ_FROM: usize = 8000;

/// The block of a read at `pos`. A fixed function of the position: the same read gets the same blocks.
pub fn read_block_at(pos: usize, short: usize) -> usize {
    static FROM: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let from = *FROM.get_or_init(|| std::env::var("SLLM_LONG_READ_FROM").ok().and_then(|v| v.parse().ok()).unwrap_or(LONG_READ_FROM));
    if from > 0 && pos >= from { 256.min(short) } else { short }
}

/// Cells of the chat reader and of the long reader for sequences of `capacity` cells.
pub fn reader_cells(capacity: usize) -> [usize; 2] {
    [CHAT_READER_CELLS.min(capacity), capacity]
}

/// Bytes of a reader's compute buffer at `cells` cells, from libllama (a `no_alloc` context of the reader's shape, as the main
/// context's is read); asked once per shape.
pub(super) fn reader_compute_bytes(p: &crate::profile::Profile, cells: usize, n_ubatch: usize, swa_full: bool) -> u64 {
    static SEEN: std::sync::Mutex<Vec<((std::path::PathBuf, usize, usize, bool), u64)>> = std::sync::Mutex::new(Vec::new());
    let key = (p.path.clone(), cells, n_ubatch, swa_full);
    if let Some(&(_, b)) = SEEN.lock().expect("reader sizes").iter().find(|(k, _)| *k == key) {
        return b;
    }
    let probe = crate::profile::ProbeParams { cells: cells as u32, n_seq: 2, n_ubatch: n_ubatch as u32, n_batch: n_ubatch as u32, kv_type: crate::profile::KvType::F16, kv_unified: true, swa_full, ..Default::default() };
    let b = crate::profile::probe_memory(&p.path, &probe).ok().map(|pr| pr.compute_bytes).filter(|&b| b > 0).unwrap_or(READER_COMPUTE_FALLBACK);
    SEEN.lock().expect("reader sizes").push((key, b));
    b
}

/// Bytes of the f16 caches of the two readers when both hold a full read: each as libllama allocates it (a sliding-window layer
/// keeps its ring, not the context, unless `swa_full`; a reader has two sequences, the second for a splice).
pub fn reader_kv_bytes(p: &crate::profile::Profile, capacity: usize, n_ubatch: usize, swa_full: bool) -> u64 {
    reader_cells(capacity).iter().map(|&c| p.kv_alloc_bytes(crate::profile::KvType::F16, c as u64, 2, n_ubatch as u64, swa_full)).sum()
}

/// What `--kv-read f16` takes from the memory budget: the readers' caches ([`reader_kv_bytes`]) and their compute buffers.
pub fn reader_reserve_bytes(p: &crate::profile::Profile, capacity: usize, n_ubatch: usize, swa_full: bool) -> u64 {
    let compute: u64 = reader_cells(capacity).iter().map(|&c| reader_compute_bytes(p, c, n_ubatch, swa_full)).sum();
    reader_kv_bytes(p, capacity, n_ubatch, swa_full) + compute
}

/// The reader (an index into the chat reader, long reader pair) a read of `target` tokens tries first: an interactive prompt
/// that fits the chat reader's `chat_cells` goes there, every other read to the last (long) reader.
pub(super) fn own_reader(lane: Lane, target: usize, chat_cells: usize, readers: usize) -> usize {
    if lane == Lane::Interactive && target <= chat_cells { 0 } else { readers - 1 }
}

/// `--kv-read f16` (kvconv.rs): an f16 context of one sequence on the same model. A prompt is read there, then its state
/// is rewritten for the main (quantized) cache and set into the request's sequence; decode runs in the main cache.
/// There are two. The chat reader ([`CHAT_READER_CELLS`]) takes an interactive prompt that fits it, so a chat never waits behind
/// a long background read. The long reader reads everything else and has the context's full size at most: its context is made
/// for a read (sized to it) and freed after a long one, so an idle server holds no f16 cache beyond the chat reader's.
pub(super) struct Reader {
    /// Null while the long reader holds no memory.
    pub(super) ctx: *mut sys::llama_context,
    model: *mut sys::llama_model,
    /// Cells of `ctx` (0 without one) and the most it may be given.
    pub(super) cells: usize,
    pub(super) max_cells: usize,
    /// The long reader: made on demand ([`Reader::ensure`]) and freed ([`Reader::close`]) after a long read.
    lazy: bool,
    /// A sliding-window model: its reader keeps its context (and what it read) between reads, and a saved read is not loaded
    /// again. A ring restored from a state is not laid out cell for cell like the ring that wrote it, so a repeat would not
    /// be bit-for-bit the cold read; a reader that stays open cuts its own ring back as it always did.
    pub(super) ring: bool,
    /// The context is being made on another thread (so a chat keeps running meanwhile): its cells, and the request that waits
    /// for it with the `(feed, shared)` token counts to report when it is ready ([`Reader::begin_open`]).
    opening: Option<(usize, std::thread::JoinHandle<(usize, f64)>)>,
    pub(super) opening_for: Option<(ReqId, usize, usize)>,
    n_ubatch: usize,
    swa_full: bool,
    cpu_threads: usize,
    batch: Batch,
    pub(super) layout: crate::kvconv::Layout,
    /// Tokens run in blocks at fixed positions (multiples of `block`): a read gives the same cache whatever chunks the
    /// scheduler cut and whatever the reader held before.
    pub(super) block: usize,
    /// The tokens whose f16 K/V are in the reader's sequence 0, in position order.
    pub(super) tokens: Vec<i32>,
    /// The request the reader is reserved for, and whether its read has started (the refill is queued).
    pub(super) req: Option<ReqId>,
    pub(super) started: bool,
    /// Tokens handed over and not run yet (part of a block, or the refill of the sequence's cached tokens).
    pub(super) queued: Vec<i32>,
    /// The leading tokens of the request's feed that the reader already holds (kept past the cached tokens): dropped.
    pub(super) skip: usize,
    pub(super) threads: usize,
    n_vocab: usize,
    /// The main cache's sequence the last read went to and its tokens then: while that sequence's cells below a position
    /// are still the ones the read wrote (`cells_changed` narrows it), the next read of it converts only the cells after.
    pub(super) last: std::cell::RefCell<HashMap<i32, Vec<i32>>>,
    /// `last` of an agent suspended to the SSD (its wake puts the same cells back, under another sequence id).
    parked: std::cell::RefCell<Option<(String, Vec<i32>)>>,
    /// The current read's kept cells (below this position the main cache's own cells stay); `SLLM_KV_READ_SPLICE=0` turns
    /// the splice off (every read converts the whole sequence: the same bytes, for checks).
    pub(super) splice: usize,
    pub(super) splice_on: bool,
    /// `SLLM_KV_READ_THREAD=0` converts every read on the executor thread.
    pub(super) thread_on: bool,
    pub(super) stats: ReadStats,
}

/// The reader's counters; they are the `kv_read` report's keys, in this order.
#[derive(Default, serde::Serialize)]
pub(super) struct ReadStats {
    pub(super) reads: u64,
    tokens: u64,
    refill_tokens: u64,
    reused_tokens: u64,
    pub(super) waited: u64,
    pub(super) in_main: u64,
    pub(super) failures: u64,
    pub(super) spliced_cells: u64,
    pub(super) convert_ms: f64,
    pub(super) set_ms: f64,
    pub(super) state_bytes: u64,
    pub(super) stashed: u64,
    pub(super) restored: u64,
    /// Most long states being converted at once (one: a read waits for the conversion before it).
    pub(super) states_out_max: u64,
    /// Contexts made and freed, and the time making and freeing them took (on the executor thread).
    created: u64,
    released: u64,
    create_ms: f64,
    close_ms: f64,
    /// Saved reads loaded again: the wait for a save still being written, the file read that had to finish after the
    /// context was made, and the state set into the context.
    pub(super) stash_wait_ms: f64,
    pub(super) restore_read_ms: f64,
    pub(super) restore_set_ms: f64,
}

impl Reader {
    /// The target layout of a main context whose sequence 0 is free: a one-cell state of it (removed again).
    pub(super) unsafe fn probe(main: *mut sys::llama_context) -> Result<crate::kvconv::Layout, String> {
        unsafe {
            let mut b = Batch::new(1);
            b.set(0, 0, 0, 0, true);
            b.b.n_tokens = 1;
            if sys::llama_decode(main, b.b) != 0 {
                return Err("--kv-read: the layout probe's decode failed".into());
            }
            let n = ffi::llama_state_seq_get_size(main, 0);
            let mut st = vec![0u8; n];
            let got = ffi::llama_state_seq_get_data(main, st.as_mut_ptr(), n, 0);
            ffi::llama_memory_seq_rm(sys::llama_get_memory(main), 0, -1, -1);
            crate::kvconv::Layout::probe(&st[..got]).map_err(|e| format!("--kv-read: {e}"))
        }
    }

    /// A reader of `model` for the target `layout` with up to `max_cells` cells. The chat reader has its context at once; a
    /// `lazy` one (the long reader) makes it for a read ([`Reader::ensure`]).
    pub(super) fn new(model: *mut sys::llama_model, layout: crate::kvconv::Layout, max_cells: usize, lazy: bool, n_ubatch: usize, swa_full: bool, cpu_threads: usize, n_vocab: usize) -> Result<Reader, String> {
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
        let mut rd = Reader { ctx: std::ptr::null_mut(), model, cells: 0, max_cells, lazy, ring: false, opening: None, opening_for: None, n_ubatch, swa_full, cpu_threads, batch: Batch::new(n_ubatch), layout, block: n_ubatch, tokens: Vec::new(), req: None, started: false, queued: Vec::new(), skip: 0, threads, n_vocab, last: Default::default(), parked: Default::default(), splice: 0,
            splice_on: std::env::var_os("SLLM_KV_READ_SPLICE").is_none_or(|v| v != "0"), thread_on: std::env::var_os("SLLM_KV_READ_THREAD").is_none_or(|v| v != "0"), stats: ReadStats::default() };
        if !lazy {
            rd.open(max_cells)?;
        }
        Ok(rd)
    }

    /// The parameters of the reader's context with `cells` cells.
    fn params(&self, cells: usize) -> sys::llama_context_params {
        // SAFETY: plain libllama call returning a struct of defaults.
        let mut cp = unsafe { sys::llama_context_default_params() };
        cp.n_ctx = cells as u32;
        cp.n_seq_max = 2; // 0: the read; 1: the cells a splice exports
        cp.n_ubatch = self.n_ubatch as u32;
        cp.n_batch = self.n_ubatch as u32;
        cp.flash_attn_type = sys::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.kv_unified = true;
        cp.swa_full = self.swa_full;
        cp.type_k = KvType::F16.ggml();
        cp.type_v = KvType::F16.ggml();
        if self.cpu_threads > 0 {
            cp.n_threads = self.cpu_threads as i32;
            cp.n_threads_batch = self.cpu_threads as i32;
        }
        cp
    }

    /// Make the context with `cells` cells.
    fn open(&mut self, cells: usize) -> Result<(), String> {
        let t0 = Instant::now();
        // SAFETY: plain libllama call on the model the reader was made for; the context is checked before use and freed in `close`/Drop.
        let ctx = unsafe { sys::llama_init_from_model(self.model, self.params(cells)) };
        self.opened(ctx, cells, t0.elapsed().as_secs_f64() * 1e3)
    }

    fn opened(&mut self, ctx: *mut sys::llama_context, cells: usize, ms: f64) -> Result<(), String> {
        if ctx.is_null() {
            return Err(format!("--kv-read: the f16 reader context ({cells} cells) could not be created"));
        }
        self.ctx = ctx;
        self.cells = cells;
        self.stats.created += 1;
        self.stats.create_ms += ms;
        Ok(())
    }

    /// Start making the context with `cells` cells on another thread: a context of a long read takes tens of milliseconds
    /// to make (its cache is cleared), which would hold up every chat if the executor made it. (Contexts of one model are
    /// independent; the executor keeps decoding in the others meanwhile.)
    fn begin_open(&mut self, cells: usize) {
        struct Work(*mut sys::llama_model, sys::llama_context_params);
        // SAFETY: the model outlives the reader's contexts and is only read; the parameters are plain values.
        unsafe impl Send for Work {}
        impl Work {
            fn run(self) -> (usize, f64) {
                let t0 = Instant::now();
                let ctx = unsafe { sys::llama_init_from_model(self.0, self.1) };
                (ctx as usize, t0.elapsed().as_secs_f64() * 1e3)
            }
        }
        let work = Work(self.model, self.params(cells));
        self.opening = Some((cells, std::thread::spawn(move || work.run())));
    }

    /// Take the context being made: false while it is not finished (`wait` waits for it).
    pub(super) fn finish_open(&mut self, wait: bool) -> Result<bool, String> {
        let Some((cells, h)) = self.opening.take() else { return Ok(true) };
        if !wait && !h.is_finished() {
            self.opening = Some((cells, h));
            return Ok(false);
        }
        let (ctx, ms) = h.join().map_err(|_| "--kv-read: making the reader's context panicked".to_string())?;
        self.opened(ctx as *mut sys::llama_context, cells, ms).map(|()| true)
    }

    /// Like [`Reader::ensure`], but a context that has to be made is made on another thread: true when it is there, false
    /// while it is being made (poll with [`Reader::finish_open`]).
    pub(super) fn ensure_soon(&mut self, need: usize) -> Result<bool, String> {
        match self.cells_to_open(need)? {
            None => Ok(true),
            Some(cells) => {
                self.begin_open(cells);
                Ok(false)
            }
        }
    }

    /// Get the reader ready to hold `need` cells: `None` when its context already does, else the context is closed and the
    /// cells of the one to make (sized to the read, in steps of [`READER_STEP`]) are returned.
    fn cells_to_open(&mut self, need: usize) -> Result<Option<usize>, String> {
        self.finish_open(true)?;
        if !self.ctx.is_null() && self.cells >= need {
            return Ok(None);
        }
        if need > self.max_cells {
            return Err(format!("--kv-read: a read of {need} tokens does not fit the reader ({} cells)", self.max_cells));
        }
        self.close();
        Ok(Some(need.div_ceil(READER_STEP).saturating_mul(READER_STEP).min(self.max_cells)))
    }

    /// Make sure the context holds `need` cells: the long reader's is made (sized to the read, in steps of [`READER_STEP`]) or,
    /// when it is too small, made again; what it held is gone then ([`LlamaEngine::read_keep`] counts nothing for it).
    pub(super) fn ensure(&mut self, need: usize) -> Result<(), String> {
        match self.cells_to_open(need)? {
            None => Ok(()),
            Some(cells) => self.open(cells),
        }
    }

    /// Free the context (and with it what the reader held). `last` and `parked` describe the main cache and stay.
    pub(super) fn close(&mut self) {
        let _ = self.finish_open(true);
        self.opening_for = None;
        if !self.ctx.is_null() {
            let t0 = Instant::now();
            // SAFETY: the context is this reader's own, and no pointer into it outlives this call.
            unsafe { sys::llama_free(self.ctx) };
            self.ctx = std::ptr::null_mut();
            self.stats.released += 1;
            self.stats.close_ms += t0.elapsed().as_secs_f64() * 1e3;
        }
        self.cells = 0;
        self.tokens.clear();
        self.queued.clear();
        self.started = false;
    }

    /// The long reader holds more memory than a chat reader does: it is freed after a long read.
    pub(super) fn big(&self) -> bool {
        self.lazy && !self.ring && self.cells > READER_STEP
    }

    /// Run queued tokens in whole blocks; `all` runs the last partial block too. Returns the logits of the last token run
    /// when `all` (the read's first sample).
    pub(super) fn run(&mut self, all: bool) -> Result<Option<&[f32]>, String> {
        let mut last = None;
        while !self.queued.is_empty() {
            let pos0 = self.tokens.len();
            let block = read_block_at(pos0, self.block);
            let room = block - pos0 % block;
            if self.queued.len() < room && !all {
                break;
            }
            let n = room.min(self.queued.len());
            let toks: Vec<i32> = self.queued.drain(..n).collect();
            let end = self.queued.is_empty() && all;
            for (i, &t) in toks.iter().enumerate() {
                self.batch.set(i, t, (pos0 + i) as i32, 0, i + 1 == n);
            }
            self.batch.b.n_tokens = n as i32;
            let rc = unsafe { sys::llama_decode(self.ctx, self.batch.b) };
            if rc != 0 {
                return Err(format!("--kv-read: the reader's llama_decode returned {rc}"));
            }
            self.tokens.extend_from_slice(&toks);
            if end {
                last = Some(n as i32 - 1);
            }
        }
        Ok(match last {
            Some(i) => {
                let p = unsafe { sys::llama_get_logits_ith(self.ctx, i) };
                if p.is_null() {
                    return Err("--kv-read: the reader has no logits".into());
                }
                Some(unsafe { std::slice::from_raw_parts(p, self.n_vocab) })
            }
            None => None,
        })
    }

    /// The main cache's cells of `sid` from position `from` on were removed or replaced.
    pub(super) fn cells_changed(&self, sid: i32, from: usize) {
        if let Some(t) = self.last.borrow_mut().get_mut(&sid) {
            t.truncate(from);
        }
    }

    /// Agent `id` is suspended from `sid`: keep `last` for its wake.
    pub(super) fn park(&self, sid: i32, id: &str) {
        if let Some(t) = self.last.borrow_mut().remove(&sid) {
            *self.parked.borrow_mut() = Some((id.to_string(), t));
        }
    }

    /// Agent `id` woke into `sid` (-1: its snapshot is lost).
    pub(super) fn unpark(&self, id: &str, sid: i32) {
        let mut p = self.parked.borrow_mut();
        if p.as_ref().is_some_and(|(a, _)| a == id) {
            let (_, t) = p.take().expect("checked");
            if sid >= 0 {
                self.last.borrow_mut().insert(sid, t);
            }
        }
    }

    /// Empty the reader (after a failure: what it holds is not known); the long reader gives its memory back.
    pub(super) fn reset(&mut self) {
        if self.big() {
            self.close();
        } else if !self.ctx.is_null() {
            unsafe { ffi::llama_memory_seq_rm(sys::llama_get_memory(self.ctx), 0, -1, -1) };
        }
        self.tokens.clear();
        self.queued.clear();
        self.req = None;
        self.started = false;
        self.last.borrow_mut().clear();
        *self.parked.borrow_mut() = None;
    }

    pub(super) fn json(&self) -> Value {
        let mut report = json!(self.stats);
        report["open_cells"] = json!(self.cells);
        report
    }
}

impl Reader {
    /// A read starts: what the reader holds of the sequence's cached tokens `h` and the `feed`, back to a block boundary,
    /// stays; the rest runs (a fresh sequence has no cached tokens: a repeat of the reader's prompt runs only its tail). The
    /// main cache's cells of the last read of the sequence `sid` that are still there stay too (the splice).
    pub(super) fn start_read(&mut self, h: &[i32], feed: &[i32], sid: i32) {
        let keep = LlamaEngine::read_keep(self, h, feed);
        unsafe { ffi::llama_memory_seq_rm(sys::llama_get_memory(self.ctx), 0, keep as i32, -1) };
        self.tokens.truncate(keep);
        self.queued = h[keep.min(h.len())..].to_vec();
        self.skip = keep.saturating_sub(h.len());
        self.splice = self.last.borrow().get(&sid).filter(|_| self.splice_on).map_or(0, |t| t.iter().zip(h).take_while(|(a, b)| a == b).count().min(keep));
        self.stats.reused_tokens += keep as u64;
        self.stats.refill_tokens += h.len().saturating_sub(keep) as u64;
        self.started = true;
    }

    /// Queue a chunk of the request's feed, without the leading tokens the reader already holds; returns how many it queued.
    pub(super) fn queue_feed(&mut self, toks: &[i32]) -> usize {
        let drop = self.skip.min(toks.len());
        self.skip -= drop;
        let toks = &toks[drop..];
        self.queued.extend_from_slice(toks);
        self.stats.tokens += toks.len() as u64;
        toks.len()
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        self.close();
    }
}
