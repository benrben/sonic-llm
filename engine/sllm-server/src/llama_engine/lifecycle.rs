//! Suspend and wake: an agent's sequence state to the SSD and back, through the I/O thread ([`crate::spill_io`]).

use super::*;

/// What the executor thread spent inside the engine's suspend and wake work (the stall the step loop sees).
#[derive(Default)]
pub(super) struct Stall {
    pub(super) suspends: u64,
    pub(super) suspend_ms_sum: f64,
    pub(super) suspend_ms_max: f64,
    pub(super) resumes: u64,
    pub(super) resume_ms_sum: f64,
    pub(super) resume_ms_max: f64,
    pub(super) applies: u64,
    pub(super) apply_ms_sum: f64,
    pub(super) apply_ms_max: f64,
}

impl Stall {
    pub(super) fn suspend(&mut self, ms: f64) {
        self.suspends += 1;
        self.suspend_ms_sum += ms;
        self.suspend_ms_max = self.suspend_ms_max.max(ms);
    }
    pub(super) fn resume(&mut self, ms: f64) {
        self.resumes += 1;
        self.resume_ms_sum += ms;
        self.resume_ms_max = self.resume_ms_max.max(ms);
    }
    pub(super) fn apply(&mut self, ms: f64) {
        self.applies += 1;
        self.apply_ms_sum += ms;
        self.apply_ms_max = self.apply_ms_max.max(ms);
    }
}

fn ag_touch(agents: &mut HashMap<AgentId, AgentSeq>, id: &str) {
    if let Some(a) = agents.get_mut(id) {
        a.last_used = Instant::now();
    }
}

impl LlamaEngine {
    /// Copy an idle resident agent's state out into a reused buffer, free its cells, and hand the write to the I/O thread.
    /// The executor thread pays for the copy only.
    pub(super) fn suspend_agent(&mut self, id: &str) -> Result<SuspendReport, EngineError> {
        let ag = self.agents.get(id).ok_or_else(|| EngineError::new(404, "not_found_error", format!("no agent {id}")))?;
        let (sid, tokens) = (ag.seq.sid, ag.seq.tokens.clone());
        let t0 = Instant::now();
        let Some(buf) = self.state_out(sid) else {
            return Err(EngineError::new(501, "not_supported_error", "this model's sequence state cannot be saved"));
        };
        for rd in &self.readers {
            rd.park(sid, id); // the wake puts the same cells back
        }
        // a stream per sequence id: the suspended agent gives its id (its stream) back; the wake takes one again
        self.drop_stream(id, sid);
        let bytes = buf.len() as u64;
        let (key, path) = (self.next_key, self.spill_dir.join(format!("{id}.seq")));
        self.next_key += 1;
        self.io.write(key, path.clone(), buf);
        self.ssd_bytes += bytes;
        self.ssd_writes += 1;
        let stage_ms = t0.elapsed().as_secs_f64() * 1e3;
        self.stall.suspend(stage_ms);
        let cells = tokens.len();
        let ag = self.agents.get_mut(id).expect("present above");
        ag.suspended = Some(Snapshot { path, bytes, tokens, key, st: SnapSt::Writing, resumed: None });
        Ok(SuspendReport { pages_freed: cells.div_ceil(CELL_PAD), bytes_freed: bytes, stage_ms })
    }

    /// Sequence `sid`'s state copied into a buffer from the I/O pool (the executor thread pays for the copy only); `None`
    /// when the model's state cannot be saved.
    pub(super) fn state_out(&mut self, sid: i32) -> Option<Vec<u8>> {
        self.flush_copies(sid);
        let size = unsafe { ffi::llama_state_seq_get_size(self.ctx, sid) };
        let mut buf = self.io.take(size);
        // SAFETY: `buf` has room for `size` bytes; libllama writes at most `size` and returns how many.
        let got = if size > 0 { unsafe { ffi::llama_state_seq_get_data(self.ctx, buf.as_mut_ptr(), size, sid) } } else { 0 };
        if size > 0 && got == 0 {
            self.io.give(buf);
            return None;
        }
        unsafe { buf.set_len(got.min(buf.capacity())) };
        Some(buf)
    }

    /// `llama_state_seq_set_data` of a snapshot's bytes into sequence `sid` (the buffer goes back to the pool): how many bytes
    /// libllama took (0: it failed) and the milliseconds the executor thread spent.
    pub(super) fn state_in(&mut self, sid: i32, buf: Vec<u8>) -> (usize, f64) {
        let t0 = Instant::now();
        self.flush_copies(sid);
        let n = if buf.is_empty() { 1 } else { unsafe { ffi::llama_state_seq_set_data(self.ctx, buf.as_ptr(), buf.len(), sid) } };
        let apply_ms = t0.elapsed().as_secs_f64() * 1e3;
        self.stall.apply(apply_ms);
        self.io.give(buf);
        (n, apply_ms)
    }

    /// A snapshot nobody needs any more: its file goes (after any queued job on it), its bytes leave the SSD tally.
    pub(super) fn discard_snapshot(&mut self, s: Snapshot) {
        self.ssd_bytes = self.ssd_bytes.saturating_sub(s.bytes);
        self.io.remove(s.path);
        if let SnapSt::Ready(buf) = s.st {
            self.io.give(buf);
        }
    }

    /// The agent whose snapshot has this key.
    fn agent_of_key(&self, key: u64) -> Option<AgentId> {
        self.agents.iter().find(|(_, a)| a.suspended.as_ref().is_some_and(|s| s.key == key)).map(|(id, _)| id.clone())
    }

    /// An agent whose snapshot is unusable: an empty agent (its history is what is left), cells freed.
    fn lose_snapshot(&mut self, id: &str) {
        for rd in &self.readers {
            rd.unpark(id, -1);
        }
        let Some(ag) = self.agents.get_mut(id) else { return };
        let snap = ag.suspended.take();
        let sid = ag.seq.sid;
        ag.seq.tokens.clear();
        ag.seq.pending = None;
        if let Some(s) = snap {
            self.discard_snapshot(s);
        }
        self.drop_stream(id, sid);
    }

    /// Per-sequence streams: an agent without cache gives its sequence id back (unified: it keeps it).
    pub(super) fn drop_stream(&mut self, id: &str, sid: i32) {
        self.seq_rm(sid, -1, -1);
        if !self.unified {
            self.release_sid(sid);
            if let Some(a) = self.agents.get_mut(id) {
                a.seq.sid = -1;
            }
        }
    }

    /// Completions from the I/O thread; then restores whose bytes have arrived are applied.
    pub(super) fn poll_io(&mut self) {
        for d in self.io.poll() {
            match d {
                spill_io::Done::Written { key, buf, result } => self.on_written(key, buf, result),
                spill_io::Done::Read { key, result, .. } => self.on_read(key, result),
            }
        }
        let ready: Vec<AgentId> = self.agents.iter().filter(|(_, a)| matches!(a.suspended.as_ref().map(|s| &s.st), Some(SnapSt::Ready(_)))).map(|(id, _)| id.clone()).collect();
        for id in ready {
            self.apply_restore(&id);
        }
        let ready_jobs: Vec<ReqId> = self.jobs.values().filter(|j| matches!(j.snap.as_ref().map(|s| &s.st), Some(SnapSt::Ready(_)))).map(|j| j.req).collect();
        for req in ready_jobs {
            self.apply_job_restore(req);
        }
    }

    /// A write of the I/O thread ended: an agent's snapshot is on the SSD now (or lost), as is a paused request's (or the
    /// request fails); a reader's save is no longer waited for.
    fn on_written(&mut self, key: u64, buf: Vec<u8>, result: Result<f64, String>) {
        self.io.give(buf);
        self.rstash.retain(|(k, _)| *k != key);
        let Some(id) = self.agent_of_key(key) else {
            // a paused request's snapshot
            if let Some(req) = self.job_of_key(key) {
                self.job_written(req, result);
            }
            return;
        };
        match result {
            Ok(ms) => {
                let ag = self.agents.get_mut(&id).expect("found above");
                let bytes = ag.suspended.as_ref().map_or(0, |s| s.bytes);
                if let Some(s) = ag.suspended.as_mut().filter(|s| matches!(s.st, SnapSt::Writing)) {
                    s.st = SnapSt::Written;
                }
                self.events.push(LifecycleEvent::Written { agent: id, result: Ok((bytes, bytes, ms, true)) });
            }
            Err(e) => {
                self.lose_snapshot(&id);
                self.events.push(LifecycleEvent::Written { agent: id, result: Err(e) });
            }
        }
    }

    /// The write of the paused request `req`'s snapshot ended.
    fn job_written(&mut self, req: ReqId, result: Result<f64, String>) {
        match result {
            Ok(_) => {
                if let Some(s) = self.jobs.get_mut(&req).and_then(|j| j.snap.as_mut()).filter(|s| matches!(s.st, SnapSt::Writing)) {
                    s.st = SnapSt::Written;
                }
            }
            Err(e) => self.lose_job_snapshot(req, e),
        }
    }

    /// A read of the I/O thread ended: the bytes of a wake wait for [`Self::apply_restore`] (a paused request's for
    /// [`Self::apply_job_restore`]); a failed read loses the snapshot. A snapshot nobody has any more gives its bytes back.
    fn on_read(&mut self, key: u64, result: Result<Vec<u8>, (String, Vec<u8>)>) {
        let owner = match self.agent_of_key(key) {
            Some(id) => Some(SnapOwner::Agent(id)),
            None => self.job_of_key(key).map(SnapOwner::Job),
        };
        match (owner, result) {
            (Some(owner), Ok(buf)) => {
                let snap = match &owner {
                    SnapOwner::Agent(id) => self.agents.get_mut(id).and_then(|a| a.suspended.as_mut()),
                    SnapOwner::Job(req) => self.jobs.get_mut(req).and_then(|j| j.snap.as_mut()),
                };
                if let Some(s) = snap {
                    s.st = SnapSt::Ready(buf);
                }
            }
            (Some(SnapOwner::Agent(id)), Err((e, buf))) => {
                self.io.give(buf);
                self.lose_snapshot(&id);
                self.events.push(LifecycleEvent::Restored { agent: id, result: Err(e) });
            }
            (Some(SnapOwner::Job(req)), Err((e, buf))) => {
                self.io.give(buf);
                self.lose_job_snapshot(req, e);
            }
            (None, Ok(buf)) | (None, Err((_, buf))) => self.io.give(buf),
        }
    }

    /// A paused request's state is gone: the request fails (its agent loses its cache with it, see `cancel`).
    fn lose_job_snapshot(&mut self, req: ReqId, e: String) {
        if let Some(snap) = self.jobs.get_mut(&req).and_then(|j| j.snap.take()) {
            self.discard_snapshot(snap);
        }
        self.job_failures.push((req, e));
    }

    /// `llama_state_seq_set_data` of the bytes a wake read: the part of a restore that must run on the executor thread.
    fn apply_restore(&mut self, id: &str) {
        if !self.room_to_restore(id) {
            return;
        }
        let Some(ag) = self.agents.get_mut(id) else { return };
        let Some((snap, buf)) = take_ready(&mut ag.suspended) else { return };
        let sid = ag.seq.sid;
        let (n, apply_ms) = self.state_in(sid, buf);
        let wake_ms = snap.resumed.map_or(apply_ms, |t| t.elapsed().as_secs_f64() * 1e3);
        ag_touch(&mut self.agents, id);
        self.ssd_bytes = self.ssd_bytes.saturating_sub(snap.bytes);
        self.io.remove(snap.path.clone());
        if n == 0 {
            let ag = self.agents.get_mut(id).expect("present above");
            ag.seq.tokens.clear();
            ag.seq.pending = None;
            self.drop_stream(id, sid);
            self.events.push(LifecycleEvent::Restored { agent: id.to_string(), result: Err("llama_state_seq_set_data failed".to_string()) });
        } else {
            self.restored(id, sid, snap.tokens, wake_ms);
        }
    }

    /// The cells come first: other requests may have taken them since the wake was asked for. Idle agents are suspended
    /// (and requests of a lower rank than the agent's lane paused) for it; false when that is not enough yet (the bytes wait).
    fn room_to_restore(&mut self, id: &str) -> bool {
        let Some(ag) = self.agents.get(id) else { return true };
        let (lane, cells) = (ag.lane, ag.suspended.as_ref().map_or(0, |s| s.tokens.len()));
        if self.unified {
            let short = (self.resident_cells() + cells + MARGIN).saturating_sub(self.total_cells);
            return short == 0 || self.free_once(short, Some(id), kvmem::rank(lane)).is_some();
        }
        if ag.seq.sid < 0 {
            // streams: the agent needs an id again before its state can go back
            let Some(sid) = self.alloc_sid_for(kvmem::rank(lane), Some(id)) else { return false };
            if let Some(a) = self.agents.get_mut(id) {
                a.seq.sid = sid;
            }
        }
        true
    }

    /// Agent `id`'s state is back in `sid`: it holds `tokens` again.
    fn restored(&mut self, id: &str, sid: i32, tokens: Vec<i32>, wake_ms: f64) {
        if self.share {
            self.ptree.borrow_mut().set(sid, &tokens);
        }
        let ag = self.agents.get_mut(id).expect("present above");
        ag.seq.tokens = tokens;
        for rd in &self.readers {
            rd.unpark(id, sid);
        }
        self.ssd_reads += 1;
        self.events.push(LifecycleEvent::Restored { agent: id.to_string(), result: Ok(wake_ms) });
    }
}

/// Whose snapshot a finished read belongs to.
enum SnapOwner {
    Agent(AgentId),
    Job(ReqId),
}

/// The snapshot in `slot` whose bytes a read has brought (its state is now `Written`), with those bytes; otherwise the snapshot
/// stays in the slot.
pub(super) fn take_ready(slot: &mut Option<Snapshot>) -> Option<(Snapshot, Vec<u8>)> {
    let mut snap = slot.take()?;
    match std::mem::replace(&mut snap.st, SnapSt::Written) {
        SnapSt::Ready(buf) => Some((snap, buf)),
        _ => {
            *slot = Some(snap);
            None
        }
    }
}
