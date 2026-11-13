//! The dedicated I/O thread for KV snapshots (suspend writes, resume reads) and the reused byte buffers they travel in.
//!
//! The executor thread must never wait on the SSD: it copies a sequence's state into a buffer taken from [`SpillIo::take`]
//! (capacity only, never zero-filled), hands the buffer to the thread with [`SpillIo::write`], and frees the cells at once.
//! A restore is a [`SpillIo::read`] that the thread runs in the background; the executor applies the bytes when
//! [`SpillIo::poll`] reports them. A read runs before the writes queued ahead of it (a waking agent waits for its bytes; a
//! suspend does not wait for its write), except behind a write or remove of the same file, which it must see; everything
//! else runs in submission order.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::JoinHandle;
use std::time::Instant;

enum Job {
    Write { key: u64, path: PathBuf, buf: Vec<u8> },
    Read { key: u64, path: PathBuf, buf: Vec<u8> },
    Remove { path: PathBuf },
}

/// A finished job. The buffer always comes back (also on an error) so it can be reused.
pub enum Done {
    Written { key: u64, buf: Vec<u8>, result: Result<f64, String> },
    Read { key: u64, result: Result<Vec<u8>, (String, Vec<u8>)>, ms: f64 },
}

/// Buffers kept for reuse are capped by their total capacity: a few snapshots of a large model must not stay pinned in RAM.
pub const POOL_MAX_BYTES: usize = 256 << 20;
// `SLLM_SPILL_POOL_MB` overrides it (diagnostics).

/// The queue index to run next: the oldest read that no earlier write or remove of the same file is queued ahead of,
/// else the oldest job.
fn next_job(queue: &VecDeque<Job>) -> usize {
    let path = |j: &Job| -> PathBuf {
        match j {
            Job::Write { path, .. } | Job::Read { path, .. } | Job::Remove { path } => path.clone(),
        }
    };
    queue
        .iter()
        .enumerate()
        .find(|(i, j)| {
            matches!(j, Job::Read { .. }) && !queue.iter().take(*i).any(|e| !matches!(e, Job::Read { .. }) && path(e) == path(j))
        })
        .map_or(0, |(i, _)| i)
}

fn run_job(job: Job, done_tx: &Sender<Done>) {
    match job {
        Job::Write { key, path, buf } => {
            let t0 = Instant::now();
            let result = (|| {
                if let Some(dir) = path.parent() {
                    std::fs::create_dir_all(dir).map_err(|e| format!("spill dir: {e}"))?;
                }
                std::fs::write(&path, &buf).map_err(|e| format!("snapshot write: {e}"))
            })()
            .map(|()| t0.elapsed().as_secs_f64() * 1e3);
            let _ = done_tx.send(Done::Written { key, buf, result });
        }
        Job::Read { key, path, mut buf } => {
            let t0 = Instant::now();
            buf.clear();
            let r = (|| {
                use std::io::Read;
                let mut f = std::fs::File::open(&path)?;
                if let Ok(m) = f.metadata() {
                    buf.reserve(m.len() as usize);
                }
                f.read_to_end(&mut buf).map(|_| ())
            })();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let result = match r {
                Ok(()) => Ok(buf),
                Err(e) => Err((format!("snapshot read: {e}"), buf)),
            };
            let _ = done_tx.send(Done::Read { key, result, ms });
        }
        Job::Remove { path } => {
            let _ = std::fs::remove_file(&path);
        }
    }
}

pub struct SpillIo {
    tx: Option<Sender<Job>>,
    rx: Receiver<Done>,
    thread: Option<JoinHandle<()>>,
    pool: Vec<Vec<u8>>,
    pool_max: usize,
    inflight: usize,
}

impl SpillIo {
    pub fn new() -> SpillIo {
        let (tx, jobs) = channel::<Job>();
        let (done_tx, rx) = channel::<Done>();
        let thread = std::thread::Builder::new()
            .name("sllm-spill-io".into())
            .spawn(move || {
                let mut queue: VecDeque<Job> = VecDeque::new();
                loop {
                    if queue.is_empty() {
                        match jobs.recv() {
                            Ok(j) => queue.push_back(j),
                            Err(_) => return,
                        }
                    }
                    while let Ok(j) = jobs.try_recv() {
                        queue.push_back(j);
                    }
                    let job = queue.remove(next_job(&queue)).expect("index from next_job");
                    run_job(job, &done_tx);
                }
            })
            .expect("spawn the spill I/O thread");
        SpillIo { tx: Some(tx), rx, thread: Some(thread), pool: Vec::new(), pool_max: std::env::var("SLLM_SPILL_POOL_MB").ok().and_then(|v| v.parse::<usize>().ok()).map_or(POOL_MAX_BYTES, |mb| mb << 20), inflight: 0 }
    }

    /// An empty buffer with room for `size` bytes (not zero-filled: the caller fills it with `set_len`).
    pub fn take(&mut self, size: usize) -> Vec<u8> {
        // the smallest pooled buffer that is large enough, else grow the largest one
        let pick = self.pool.iter().enumerate().filter(|(_, b)| b.capacity() >= size).min_by_key(|(_, b)| b.capacity()).map(|(i, _)| i);
        let mut buf = match pick {
            Some(i) => self.pool.swap_remove(i),
            None => self.pool.pop().unwrap_or_default(),
        };
        buf.clear();
        buf.reserve(size);
        buf
    }

    pub fn give(&mut self, mut buf: Vec<u8>) {
        buf.clear();
        if buf.capacity() > self.pool_max {
            return; // too large to keep: freed
        }
        self.pool.push(buf);
        // over the cap: drop the largest until the pool fits
        while self.pool.iter().map(|b| b.capacity()).sum::<usize>() > self.pool_max {
            let Some(i) = self.pool.iter().enumerate().max_by_key(|(_, b)| b.capacity()).map(|(i, _)| i) else { break };
            self.pool.swap_remove(i);
        }
    }

    /// Bytes held by the pooled buffers.
    pub fn pooled_bytes(&self) -> usize {
        self.pool.iter().map(|b| b.capacity()).sum()
    }

    fn send(&mut self, job: Job) {
        self.inflight += 1;
        if let Some(tx) = &self.tx {
            let _ = tx.send(job);
        }
    }

    pub fn write(&mut self, key: u64, path: PathBuf, buf: Vec<u8>) {
        self.send(Job::Write { key, path, buf });
    }

    pub fn read(&mut self, key: u64, path: PathBuf, size_hint: usize) {
        let buf = self.take(size_hint);
        self.send(Job::Read { key, path, buf });
    }

    /// Delete a file (after any write or read queued before it).
    pub fn remove(&mut self, path: PathBuf) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Job::Remove { path });
        }
    }

    /// Jobs submitted and not yet reported by [`SpillIo::poll`].
    /// Jobs submitted and not yet reported by [`SpillIo::poll`] (lifecycle counters: `io_in_flight`).
    pub fn in_flight(&self) -> usize {
        self.inflight
    }

    pub fn busy(&self) -> bool {
        self.inflight > 0
    }

    /// Finished jobs, without waiting.
    pub fn poll(&mut self) -> Vec<Done> {
        let mut out = Vec::new();
        while let Ok(d) = self.rx.try_recv() {
            self.inflight = self.inflight.saturating_sub(1);
            out.push(d);
        }
        out
    }
}

impl Default for SpillIo {
    fn default() -> Self {
        Self::new()
    }
}

impl SpillIo {
    /// Let the thread finish what is queued (pending removals included) and stop it. Idempotent.
    pub fn finish(&mut self) {
        self.tx = None;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for SpillIo {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wait(io: &mut SpillIo) -> Vec<Done> {
        for _ in 0..2000 {
            let d = io.poll();
            if !d.is_empty() {
                return d;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        panic!("io timeout");
    }

    #[test]
    fn write_then_read_back_through_the_thread_and_reuse_buffers() {
        let dir = std::env::temp_dir().join(format!("sllm-spill-io-test-{}", std::process::id()));
        let path = dir.join("a.seq");
        let mut io = SpillIo::new();
        let mut buf = io.take(1000);
        let cap = buf.capacity();
        assert!(cap >= 1000 && buf.is_empty());
        buf.extend((0..1000u32).map(|i| (i % 251) as u8));
        io.write(7, path.clone(), buf);
        io.read(8, path.clone(), 1000); // queued behind the write: sees the file
        assert!(io.busy());
        let mut got_read = None;
        let mut done = 0;
        while done < 2 {
            for d in wait(&mut io) {
                done += 1;
                match d {
                    Done::Written { key, buf, result } => {
                        assert_eq!(key, 7);
                        assert!(result.is_ok());
                        io.give(buf);
                    }
                    Done::Read { key, result, .. } => {
                        assert_eq!(key, 8);
                        got_read = Some(result.unwrap());
                    }
                }
            }
        }
        let data = got_read.unwrap();
        assert_eq!(data.len(), 1000);
        assert!(data.iter().enumerate().all(|(i, &b)| b == (i % 251) as u8));
        assert!(!io.busy());
        io.give(data);
        let again = io.take(500);
        assert!(again.capacity() >= 1000, "a pooled buffer is reused");
        io.remove(path.clone());
        drop(io); // joins: the removal ran
        assert!(!path.exists());
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn a_read_runs_before_the_writes_ahead_of_it_unless_one_is_its_own_file() {
        let w = |p: &str| Job::Write { key: 0, path: PathBuf::from(p), buf: Vec::new() };
        let r = |p: &str| Job::Read { key: 0, path: PathBuf::from(p), buf: Vec::new() };
        let q = |v: Vec<Job>| v.into_iter().collect::<VecDeque<Job>>();
        assert_eq!(next_job(&q(vec![w("a"), w("b"), r("c")])), 2);
        assert_eq!(next_job(&q(vec![w("a"), w("c"), r("c")])), 0);
        assert_eq!(next_job(&q(vec![w("c"), r("c"), r("d")])), 2);
        assert_eq!(next_job(&q(vec![Job::Remove { path: PathBuf::from("d") }, r("d"), w("e")])), 0);
        assert_eq!(next_job(&q(vec![w("a"), w("b")])), 0);
    }

    #[test]
    fn the_pool_is_capped_by_bytes() {
        let mut io = SpillIo::new();
        io.pool_max = 1000;
        io.give(Vec::with_capacity(5000)); // above the cap: freed
        assert_eq!(io.pooled_bytes(), 0);
        io.give(Vec::with_capacity(400));
        io.give(Vec::with_capacity(500));
        assert_eq!(io.pooled_bytes(), 900);
        io.give(Vec::with_capacity(300)); // 1200 > 1000: the largest goes
        assert!(io.pooled_bytes() <= 1000 && io.pooled_bytes() >= 700);
    }

    #[test]
    fn a_missing_file_is_an_error_and_the_buffer_comes_back() {
        let mut io = SpillIo::new();
        io.read(1, std::env::temp_dir().join("sllm-spill-io-missing.seq"), 10);
        match wait(&mut io).pop().unwrap() {
            Done::Read { result: Err((msg, buf)), .. } => {
                assert!(msg.contains("snapshot read"));
                io.give(buf);
            }
            _ => panic!("expected a read error"),
        }
    }
}
