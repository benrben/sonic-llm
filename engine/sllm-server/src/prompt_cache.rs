//! The prompt cache on disk: the sequence state of shared prompt starts, kept across restarts.
//!
//! When a prompt start is shared (a prefix attach of at least [`MIN_TOKENS`] tokens), the engine saves the state of exactly
//! that start here; a later fresh sequence whose prompt begins with a saved start (also after a restart) loads it instead
//! of prefilling it, and a holder is made from it instead of a prefill. One directory per model and context layout (the
//! state only loads into the same layout): `<root>/<fingerprint>/`, with `<hash>.tok` (the tokens, i32 little endian),
//! `<hash>.seq` (the state) and `index.json`. The least recently used starts go when the total passes the byte budget.
//! A `.seq` whose size is not the indexed one (a write cut short) is dropped when it is read.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Shorter shared starts are cheaper to prefill than to read.
pub const MIN_TOKENS: usize = 64;

/// A finished request is saved at its end ([`turn_save`]) only from this many tokens: a conversation worth resuming, not a
/// short one-shot (the 64-token floor of a shared start would fill the disk with one-shots).
pub const TURN_SAVE_MIN_TOKENS: usize = 512;

/// What to do with a finished request's state (`SLLM_TURN_SAVE`).
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum TurnSave {
    /// Not a candidate (off, unfinished, too short): nothing counted.
    No,
    /// A candidate, but other work is running and the state copy would stall it: skipped, counted.
    Skip,
    Save,
}

/// `enabled`: the switch and a cache on disk; `complete`: the request finished with a cacheable prompt; `tokens`: the
/// sequence's computed tokens; `others`: requests still running (the copy runs on the executor thread, so it needs none).
pub fn turn_save(enabled: bool, complete: bool, tokens: usize, others: usize) -> TurnSave {
    if !enabled || !complete || tokens < TURN_SAVE_MIN_TOKENS {
        TurnSave::No
    } else if others > 0 {
        TurnSave::Skip
    } else {
        TurnSave::Save
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexEntry {
    hash: u64,
    n: usize,
    bytes: u64,
    last_used: u64,
}

pub struct DiskCache {
    dir: PathBuf,
    budget: u64,
    entries: Vec<(IndexEntry, Vec<i32>)>,
    pub hits: u64,
    /// Tokens of the starts actually used (a dropped tail not counted); the engine adds them.
    pub hit_tokens: u64,
    pub saves: u64,
    pub evictions: u64,
    /// End-of-turn saves written, and those skipped because other work was running.
    pub turn_saves: u64,
    pub turn_save_skips: u64,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

pub fn hash_tokens(ids: &[i32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &t in ids {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

impl DiskCache {
    /// Open (or create) the cache directory and read its index; entries whose token file is missing are dropped.
    pub fn open(dir: PathBuf, budget: u64) -> std::io::Result<DiskCache> {
        std::fs::create_dir_all(&dir)?;
        let index: Vec<IndexEntry> = std::fs::read(dir.join("index.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        let mut entries = Vec::new();
        for e in index {
            if let Ok(b) = std::fs::read(dir.join(format!("{:016x}.tok", e.hash))) {
                let toks: Vec<i32> = b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                if toks.len() == e.n && hash_tokens(&toks) == e.hash {
                    entries.push((e, toks));
                }
            }
        }
        Ok(DiskCache { dir, budget, entries, hits: 0, hit_tokens: 0, saves: 0, evictions: 0, turn_saves: 0, turn_save_skips: 0 })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn bytes(&self) -> u64 {
        self.entries.iter().map(|(e, _)| e.bytes).sum()
    }

    pub fn contains(&self, ids: &[i32]) -> bool {
        let h = hash_tokens(ids);
        self.entries.iter().any(|(e, t)| e.hash == h && t == ids)
    }

    /// The longest saved start that is a prefix of `ids[..max]`: (entry index, its length).
    pub fn longest(&self, ids: &[i32], max: usize) -> Option<(usize, usize)> {
        let max = max.min(ids.len());
        self.entries.iter().enumerate().filter(|(_, (e, t))| e.n <= max && t[..] == ids[..e.n]).max_by_key(|(_, (e, _))| e.n).map(|(i, (e, _))| (i, e.n))
    }

    /// The saved state sharing the longest start with `ids[..max]`, that start cut to a multiple of `align`: (entry
    /// index, start length). For caches that can drop a tail (the start is kept and the rest of the state cut away).
    pub fn longest_common(&self, ids: &[i32], max: usize, align: usize) -> Option<(usize, usize)> {
        let max = max.min(ids.len());
        self.entries
            .iter()
            .enumerate()
            .map(|(i, (_, t))| (i, t.iter().zip(&ids[..max]).take_while(|(a, b)| a == b).count() / align.max(1) * align.max(1)))
            .filter(|&(_, n)| n > 0)
            .max_by_key(|&(_, n)| n)
    }

    pub fn tokens(&self, i: usize) -> &[i32] {
        &self.entries[i].1
    }

    /// The saved state of entry `i` (and it becomes the most recently used); None (and the entry goes) when its file is
    /// missing or short.
    pub fn read(&mut self, i: usize) -> Option<Vec<u8>> {
        let (path, bytes) = self.file(i);
        let b = std::fs::read(&path).ok().filter(|b| b.len() as u64 == bytes);
        self.loaded(i, b.is_some());
        b
    }

    /// Entry `i`'s state file and the bytes it must hold, for a caller that reads it elsewhere (another thread, while it does
    /// other work); it then reports with [`DiskCache::loaded`].
    pub fn file(&self, i: usize) -> (PathBuf, u64) {
        (self.dir.join(format!("{:016x}.seq", self.entries[i].0.hash)), self.entries[i].0.bytes)
    }

    /// Entry `i` was read (`ok`: it becomes the most recently used) or its file was missing or short (the entry goes).
    pub fn loaded(&mut self, i: usize, ok: bool) {
        if ok {
            self.entries[i].0.last_used = now();
            self.hits += 1;
            self.save_index();
        } else {
            self.remove(i);
        }
    }

    /// Record a start whose state `bytes` long is being written to the returned path (the caller writes it, off its
    /// thread); the least recently used starts go first while the budget is passed. None when it is already saved or
    /// cannot fit at all.
    pub fn insert(&mut self, ids: &[i32], bytes: u64) -> Option<PathBuf> {
        if self.contains(ids) || bytes > self.budget {
            return None;
        }
        while self.bytes() + bytes > self.budget {
            let Some(i) = self.entries.iter().enumerate().min_by_key(|(_, (e, _))| e.last_used).map(|(i, _)| i) else { break };
            self.remove(i);
            self.evictions += 1;
        }
        let hash = hash_tokens(ids);
        let tok: Vec<u8> = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
        std::fs::write(self.dir.join(format!("{hash:016x}.tok")), tok).ok()?;
        self.entries.push((IndexEntry { hash, n: ids.len(), bytes, last_used: now() }, ids.to_vec()));
        self.saves += 1;
        self.save_index();
        Some(self.dir.join(format!("{hash:016x}.seq")))
    }

    fn remove(&mut self, i: usize) {
        let (e, _) = self.entries.remove(i);
        let _ = std::fs::remove_file(self.dir.join(format!("{:016x}.seq", e.hash)));
        let _ = std::fs::remove_file(self.dir.join(format!("{:016x}.tok", e.hash)));
        self.save_index();
    }

    fn save_index(&self) {
        let index: Vec<&IndexEntry> = self.entries.iter().map(|(e, _)| e).collect();
        if let Ok(b) = serde_json::to_vec(&index) {
            let tmp = self.dir.join("index.json.tmp");
            if std::fs::write(&tmp, b).is_ok() {
                let _ = std::fs::rename(tmp, self.dir.join("index.json"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sllm-pcache-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn starts_survive_a_reopen_and_the_longest_prefix_wins() {
        let d = dir("reopen");
        let mut c = DiskCache::open(d.clone(), 1 << 20).unwrap();
        let a: Vec<i32> = (0..100).collect();
        let b: Vec<i32> = (0..150).collect();
        for ids in [&a, &b] {
            let p = c.insert(ids, 10).unwrap();
            std::fs::write(p, vec![7u8; 10]).unwrap();
        }
        assert!(c.insert(&a, 10).is_none(), "saved once");
        let mut c = DiskCache::open(d.clone(), 1 << 20).unwrap();
        let q: Vec<i32> = (0..200).collect();
        assert_eq!(c.longest(&q, 199).map(|x| x.1), Some(150));
        assert_eq!(c.longest(&q, 120).map(|x| x.1), Some(100));
        let (i, _) = c.longest(&q, 199).unwrap();
        assert_eq!(c.read(i).unwrap(), vec![7u8; 10]);
        let other: Vec<i32> = (1..200).collect();
        assert!(c.longest(&other, 199).is_none());
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn the_budget_drops_the_least_recently_used_and_a_short_file_is_dropped() {
        let d = dir("budget");
        let mut c = DiskCache::open(d.clone(), 25).unwrap();
        let s = |k: i32| -> Vec<i32> { (0..80).map(|i| i * 1000 + k).collect() };
        for k in 0..2 {
            let p = c.insert(&s(k), 10).unwrap();
            std::fs::write(p, vec![1u8; 10]).unwrap();
        }
        c.entries[1].0.last_used = 0; // the second is the oldest now
        let p = c.insert(&s(2), 10).unwrap();
        std::fs::write(p, vec![1u8; 3]).unwrap(); // cut short
        assert_eq!(c.len(), 2);
        assert!(c.contains(&s(0)) && !c.contains(&s(1)) && c.contains(&s(2)));
        let (i, _) = c.longest(&s(2), 80).unwrap();
        assert!(c.read(i).is_none());
        assert!(!c.contains(&s(2)));
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn a_turn_is_saved_only_when_finished_big_enough_and_the_engine_is_otherwise_idle() {
        use TurnSave::*;
        let big = TURN_SAVE_MIN_TOKENS;
        assert_eq!(turn_save(true, true, big, 0), Save);
        assert_eq!(turn_save(true, true, big, 1), Skip, "other work is running: skipped (and counted)");
        assert_eq!(turn_save(true, true, big - 1, 0), No, "a short one-shot is not worth a state");
        assert_eq!(turn_save(true, true, big - 1, 5), No, "short ones are not counted as skips");
        assert_eq!(turn_save(true, false, big, 0), No, "a paused or failed request, or one that did not cache its prompt");
        assert_eq!(turn_save(false, true, big, 0), No, "SLLM_TURN_SAVE=0 or --prompt-cache off");
        assert!(TURN_SAVE_MIN_TOKENS > MIN_TOKENS);
    }

    #[test]
    fn turn_counters_start_at_zero_and_a_saved_turn_dedupes() {
        let d = dir("turn");
        let mut c = DiskCache::open(d.clone(), 1 << 20).unwrap();
        assert_eq!((c.turn_saves, c.turn_save_skips), (0, 0));
        let t: Vec<i32> = (0..600).collect();
        assert!(c.insert(&t, 10).is_some());
        assert!(c.contains(&t) && c.insert(&t, 10).is_none(), "the same turn twice writes once");
        let _ = std::fs::remove_dir_all(d);
    }
}
