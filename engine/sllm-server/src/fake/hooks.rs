//! Test hooks of the fake engine: events the real engines produce that the fake does not by itself (no room for a request
//! at a chosen moment, a snapshot write that fails, a write still running when a wake starts), and the fake's lifecycle
//! poll that reports them. Every hook is off by default, so the fake behaves as without them.

use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};

/// Shared with the test that holds a clone of the config: it arms a hook at the moment it wants the event.
#[derive(Debug, Clone, Default)]
pub struct FakeHooks {
    /// The next this many `start_request` calls are refused with `kv_pool_exhausted_error` (no room now).
    pub refuse_starts: Arc<AtomicUsize>,
    /// The next this many snapshot writes fail. A failed write loses the snapshot when a restore waits behind it (libllama:
    /// the read finds no snapshot and no restore event comes); otherwise the agent stays suspended (the next engine keeps
    /// its host copy) until `agent_drop_kv`.
    pub fail_writes: Arc<AtomicUsize>,
    /// While set, snapshot writes stay in flight (not reported), and a restore waits behind its agent's write.
    pub hold_writes: Arc<AtomicBool>,
    /// While set, plain completions are admitted but not given a slot (`try_begin` says no): they stay queued, so the
    /// executor has a request that holds room and ends only when the test lets it.
    pub hold_completions: Arc<AtomicBool>,
}

/// One from a countdown: true when it was above 0.
fn take_one(n: &AtomicUsize) -> bool {
    n.try_update(SeqCst, SeqCst, |v| v.checked_sub(1)).is_ok()
}

impl FakeHooks {
    pub(super) fn refuse_start(&self) -> Result<(), EngineError> {
        if take_one(&self.refuse_starts) {
            return Err(EngineError::new(503, "kv_pool_exhausted_error", "fake: no room now (test hook)").with_retry_after(10));
        }
        Ok(())
    }

    /// Whether a request of `agent` (None: a plain completion) may not begin now.
    pub(super) fn holds(&self, agent: Option<&str>) -> bool {
        agent.is_none() && self.hold_completions.load(SeqCst)
    }
}

impl FakeEngine {
    /// The lifecycle's background results: snapshot writes, then restores that are due (agents in id order).
    pub(super) fn poll_lifecycle(&mut self) -> Vec<LifecycleEvent> {
        let mut ids: Vec<String> = self.agents.keys().cloned().collect();
        ids.sort();
        let mut out = Vec::new();
        for id in ids {
            out.extend(self.poll_write(&id));
            out.extend(self.poll_restore(&id));
        }
        out
    }

    /// The agent's snapshot write, once it has ended (a held write has not).
    fn poll_write(&mut self, id: &str) -> Option<LifecycleEvent> {
        let (held, fails, seed) = (self.cfg.hooks.hold_writes.load(SeqCst), &self.cfg.hooks.fail_writes, self.cfg.seed);
        let a = self.agents.get_mut(id)?;
        if !a.written || held {
            return None;
        }
        a.written = false;
        if !take_one(fails) {
            let b = pages(a.ctx) as u64 * FAKE_PAGE as u64;
            return Some(LifecycleEvent::Written { agent: id.to_string(), result: Ok((b, b, 0.0, true)) });
        }
        if a.restoring.take().is_some() {
            // libllama: the read queued behind the write finds no snapshot: an empty agent, and no restore event
            (a.state, a.ctx, a.suspended) = (seed, 0, false);
            a.tokens.clear();
        }
        Some(LifecycleEvent::Written { agent: id.to_string(), result: Err("fake: snapshot write failed (test hook)".into()) })
    }

    /// The agent's restore, once `restore_ms` has passed since it was asked for (and its write has ended).
    fn poll_restore(&mut self, id: &str) -> Option<LifecycleEvent> {
        let (ms, fails, seed) = (self.cfg.restore_ms, self.cfg.restore_fails, self.cfg.seed);
        let a = self.agents.get_mut(id)?;
        let t = a.restoring.filter(|t| !a.written && t.elapsed().as_secs_f64() * 1e3 >= ms)?;
        a.restoring = None;
        a.suspended = false;
        if fails {
            (a.state, a.ctx) = (seed, 0);
            a.tokens.clear();
            return Some(LifecycleEvent::Restored { agent: id.to_string(), result: Err("fake: snapshot lost".into()) });
        }
        Some(LifecycleEvent::Restored { agent: id.to_string(), result: Ok(t.elapsed().as_secs_f64() * 1e3) })
    }
}
