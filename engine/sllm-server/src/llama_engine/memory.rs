//! Memory granted on demand ([`crate::kvmem`]): cells held and promised, freeing room by suspending idle agents or
//! pausing requests, and bringing paused requests back.

use super::*;

/// What the on-demand memory policy did, for `/metrics` `lifecycle.engine.mem`.
#[derive(Default)]
pub(super) struct MemStats {
    pub(super) preempt_suspends: u64,
    pub(super) preempt_pauses: u64,
    pub(super) resumed: u64,
    pub(super) refused: u64,
    pub(super) peak_committed: usize,
    pub(super) max_inflight_jobs: usize,
}

/// Whether a paused request can come back now ([`LlamaEngine::resume_paused`]).
enum Resume {
    /// Its room is reserved: its state is read now.
    Go,
    /// It is gone, or has no sequence: the next one.
    Skip,
    /// No room: the rest wait too.
    Stop,
}

impl LlamaEngine {
    fn seq_cells(s: &Seq) -> usize {
        s.tokens.len() + s.pending.is_some() as usize
    }

    /// Cells of a job's sequence right now.
    fn job_cells(&self, j: &Job) -> usize {
        match &j.agent {
            Some(a) => self.agents.get(a).map_or(0, |x| Self::seq_cells(&x.seq)),
            None => j.seq.as_ref().map_or(0, Self::seq_cells),
        }
    }

    /// Cells a job holds and has promised: what its sequence has, what it still has to prefill, and a small decode headroom.
    /// A paused job holds nothing.
    fn job_commit(&self, j: &Job) -> usize {
        if j.paused {
            // a restore under way has its room reserved: the sequence, the sampled token and the headroom
            return match &j.snap {
                Some(s) if matches!(s.st, SnapSt::Reading | SnapSt::Ready(_)) => s.tokens.len() + 1 + HEADROOM.min(j.budget.saturating_sub(j.emitted)),
                _ => 0,
            };
        }
        let left = if j.started.is_some() { j.feed.len().saturating_sub(j.fed) } else { j.ids.len().saturating_sub(j.shared_n) };
        self.job_cells(j) + left + HEADROOM.min(j.budget.saturating_sub(j.emitted))
    }

    /// Cells promised: idle resident agents at their context, running and queued requests at what they need now
    /// ([`Self::job_commit`]), not at their worst case. `except`: an agent whose idle cells are replaced by its request.
    pub(super) fn committed_cells(&self, except: Option<&str>) -> usize {
        let busy: HashSet<&str> = self.jobs.values().filter_map(|j| j.agent.as_deref()).collect();
        let idle: usize = self
            .agents
            .iter()
            .filter(|(id, a)| a.suspended.is_none() && !busy.contains(id.as_str()) && Some(id.as_str()) != except)
            .map(|(_, a)| Self::seq_cells(&a.seq))
            .sum();
        idle + self.holder_cells() + self.jobs.values().map(|j| self.job_commit(j)).sum::<usize>()
    }

    /// Cells in the cache right now (the sampled token not fed yet counts, a slight overestimate).
    pub(super) fn resident_cells(&self) -> usize {
        let agents: usize = self.agents.values().filter(|a| a.suspended.is_none()).map(|a| Self::seq_cells(&a.seq)).sum();
        agents + self.holder_cells() + self.jobs.values().filter(|j| j.agent.is_none()).filter_map(|j| j.seq.as_ref()).map(Self::seq_cells).sum::<usize>()
    }

    /// Free cells for a shortfall: suspend LRU idle agents, then pause running requests of rank below `below`. `except`:
    /// the agent the room is for. `None` when what could be freed does not cover it (nothing is touched); otherwise the
    /// requests paused.
    pub(super) fn free_once(&mut self, short: usize, except: Option<&str>, below: u8) -> Option<Vec<ReqId>> {
        let idle = self.idle_to_free(except);
        let pausable: Vec<Pausable> = self
            .jobs
            .values()
            .filter(|j| self.pausable(j))
            .map(|j| Pausable { req: j.req, rank: kvmem::rank(j.lane), order: j.order, cells: if self.unified { self.job_commit(j) } else { 1 } })
            .collect();
        let plan = kvmem::plan_free(short, &idle, &pausable, below)?;
        let mut freed_agents = 0usize;
        for id in plan.suspend {
            freed_agents += self.free_idle(id) as usize;
        }
        let mut paused = Vec::new();
        for &r in &plan.pause {
            if self.pause_job(r) {
                self.mem.preempt_pauses += 1;
                paused.push(r);
            }
        }
        // nothing was freed (the model's state cannot be saved): no plan
        if paused.is_empty() && freed_agents == 0 {
            return None;
        }
        Some(paused)
    }

    /// What [`kvmem::plan_free`] may suspend: the idle resident agents (not `except`; none when the model's state cannot be
    /// saved), and the frozen holders.
    fn idle_to_free(&self, except: Option<&str>) -> Vec<IdleAgent> {
        let busy: HashSet<&str> = self.jobs.values().filter_map(|j| j.agent.as_deref()).collect();
        let idle: Vec<IdleAgent> = if self.no_suspend {
            Vec::new()
        } else {
            self.agents
                .iter()
                .filter(|(id, a)| a.suspended.is_none() && !busy.contains(id.as_str()) && Some(id.as_str()) != except)
                .filter(|(_, a)| self.unified || a.seq.sid >= 0)
                .map(|(id, a)| IdleAgent { id: id.clone(), last_used: a.last_used, cells: if self.unified { Self::seq_cells(&a.seq) } else { 1 } })
                .collect()
        };
        let mut idle = idle;
        // frozen holders go first: they are a cache, losing one costs a prefill later, not a restore
        for h in &self.holders {
            idle.push(IdleAgent {
                id: format!("\u{1}holder{}", h.sid),
                last_used: h.last_used.checked_sub(std::time::Duration::from_secs(3600)).unwrap_or(h.last_used),
                cells: if self.unified { h.tokens.len() } else { 1 },
            });
        }
        idle
    }

    /// Free what [`kvmem::plan_free`] chose of the idle (`id`, from [`Self::idle_to_free`]): a holder is evicted, an empty
    /// agent gives its stream back, any other agent is suspended to the SSD. True when it freed something.
    fn free_idle(&mut self, id: AgentId) -> bool {
        if let Some(sid) = id.strip_prefix("\u{1}holder").and_then(|x| x.parse::<i32>().ok()) {
            let Some(i) = self.holders.iter().position(|h| h.sid == sid) else { return false };
            self.evict_holder(i);
            return true;
        }
        if !self.unified && self.agents.get(&id).is_some_and(|a| a.seq.tokens.is_empty() && a.seq.pending.is_none()) {
            // an empty agent holds a stream and nothing else: just give the id back
            let Some(a) = self.agents.get_mut(&id) else { return false };
            let sid = std::mem::replace(&mut a.seq.sid, -1);
            self.release_sid(sid);
            return true;
        }
        match self.suspend_agent(&id) {
            Ok(_) => {
                self.mem.preempt_suspends += 1;
                self.events.push(LifecycleEvent::Suspended { agent: id });
                true
            }
            Err(e) => {
                if e.status == 501 {
                    self.no_suspend = true;
                }
                false
            }
        }
    }

    /// A request that can be paused: decoding (its prompt is in the cache, a sampled token waits).
    fn pausable(&self, j: &Job) -> bool {
        !self.no_suspend && !j.paused && j.started.is_some() && j.fed >= j.feed.len() && j.first.is_some() && j.begun && self.job_has_pending(j)
    }

    fn job_has_pending(&self, j: &Job) -> bool {
        match &j.agent {
            Some(a) => self.agents.get(a).is_some_and(|x| x.seq.pending.is_some()),
            None => j.seq.as_ref().is_some_and(|s| s.pending.is_some()),
        }
    }

    /// Pause a running request: its sequence state is copied out exactly like an agent suspend (the I/O thread writes it),
    /// its cells and stream are freed, and the step skips it. The resume sets the state back, so its later tokens are
    /// those of an unpaused run. False when the model's state cannot be saved.
    fn pause_job(&mut self, req: ReqId) -> bool {
        let unified = self.unified;
        let Some(job) = self.jobs.get(&req) else { return false };
        let sid = match &job.agent {
            Some(a) => self.agents.get(a).map(|x| x.seq.sid),
            None => job.seq.as_ref().map(|x| x.sid),
        };
        let Some(sid) = sid.filter(|&s| s >= 0) else { return false };
        let t0 = Instant::now();
        let Some(buf) = self.state_out(sid) else {
            self.no_suspend = true;
            return false;
        };
        let bytes = buf.len() as u64;
        let (key, path) = (self.next_key, self.spill_dir.join(format!("req{req}.seq")));
        self.next_key += 1;
        let job = self.jobs.get_mut(&req).expect("present above");
        let Some(seq) = Self::seq_for(&mut self.agents, job) else { return false };
        let tokens = std::mem::take(&mut seq.tokens);
        if !unified {
            seq.sid = -1; // the stream goes back to the pool with the cells
        }
        job.snap = Some(Snapshot { path: path.clone(), bytes, tokens, key, st: SnapSt::Writing, resumed: None });
        job.paused = true;
        self.io.write(key, path, buf);
        self.ssd_bytes += bytes;
        self.ssd_writes += 1;
        self.release_sid_keep(sid, unified);
        self.stall.suspend(t0.elapsed().as_secs_f64() * 1e3);
        true
    }

    /// Free a sequence's cells; per-sequence streams also give the id back.
    fn release_sid_keep(&mut self, sid: i32, unified: bool) {
        if unified {
            self.seq_rm(sid, -1, -1);
        } else {
            self.release_sid(sid);
        }
    }

    /// At the start of a step: paused requests start coming back, highest rank and oldest first, when there is room for
    /// their sequence (idle agents are suspended for it when that helps). The room is reserved and the state is read on the
    /// I/O thread; [`Self::apply_job_restore`] sets it when the bytes are there.
    pub(super) fn resume_paused(&mut self) {
        let mut order: Vec<(u8, u64, ReqId)> = self
            .jobs
            .values()
            .filter(|j| j.paused && j.snap.as_ref().is_some_and(|s| matches!(s.st, SnapSt::Writing | SnapSt::Written)))
            .map(|j| (kvmem::rank(j.lane), j.order, j.req))
            .collect();
        if order.is_empty() {
            return;
        }
        order.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        for (_, _, req) in order {
            match self.room_to_resume(req) {
                Resume::Stop => break,
                Resume::Skip => continue,
                Resume::Go => self.read_paused(req),
            }
        }
    }

    /// Room for the paused request `req` to come back: cells (unified) or its own stream again.
    fn room_to_resume(&mut self, req: ReqId) -> Resume {
        let Some(j) = self.jobs.get(&req) else { return Resume::Skip };
        let Some(snap) = &j.snap else { return Resume::Skip };
        let need = snap.tokens.len() + 1 + HEADROOM.min(j.budget.saturating_sub(j.emitted)) + MARGIN;
        let except = j.agent.clone();
        if self.unified {
            let short = (self.committed_cells(None) + need).saturating_sub(self.total_cells);
            if short > 0 && self.free_once(short, except.as_deref(), 0).is_none() {
                return Resume::Stop;
            }
            return Resume::Go;
        }
        // a stream: its own id again
        let Some(sid) = self.alloc_sid_for(kvmem::rank(self.jobs[&req].lane), except.as_deref()) else { return Resume::Stop };
        let Some(j) = self.jobs.get_mut(&req) else { return Resume::Skip };
        match Self::seq_for(&mut self.agents, j) {
            Some(seq) => {
                seq.sid = sid;
                Resume::Go
            }
            None => Resume::Skip,
        }
    }

    /// The room for the paused request `req` is reserved: its state is read on the I/O thread.
    fn read_paused(&mut self, req: ReqId) {
        let Some(s) = self.jobs.get_mut(&req).and_then(|j| j.snap.as_mut()) else { return };
        s.st = SnapSt::Reading;
        s.resumed = Some(Instant::now());
        let (key, path, bytes) = (s.key, s.path.clone(), s.bytes as usize);
        self.io.read(key, path, bytes);
    }

    /// The paused request whose snapshot has this key.
    pub(super) fn job_of_key(&self, key: u64) -> Option<ReqId> {
        self.jobs.iter().find(|(_, j)| j.snap.as_ref().is_some_and(|s| s.key == key)).map(|(r, _)| *r)
    }

    /// `llama_state_seq_set_data` for a paused request whose bytes have been read: it runs again from the next step.
    pub(super) fn apply_job_restore(&mut self, req: ReqId) {
        let Some(job) = self.jobs.get_mut(&req) else { return };
        let Some((snap, buf)) = lifecycle::take_ready(&mut job.snap) else { return };
        let Some(seq) = Self::seq_for(&mut self.agents, job) else { return };
        let sid = seq.sid;
        let (n, _) = self.state_in(sid, buf);
        self.ssd_bytes = self.ssd_bytes.saturating_sub(snap.bytes);
        self.io.remove(snap.path.clone());
        let job = self.jobs.get_mut(&req).expect("present above");
        if n == 0 {
            self.job_failures.push((req, "llama_state_seq_set_data failed while resuming a paused request".into()));
            return; // stays paused until the executor cancels it
        }
        if let Some(seq) = Self::seq_for(&mut self.agents, job) {
            if self.share {
                self.ptree.borrow_mut().set(seq.sid, &snap.tokens);
            }
            seq.tokens = snap.tokens;
        }
        job.paused = false;
        self.mem.resumed += 1;
    }

    /// Before a step's decode: make room for the cells its rows add. Suspends LRU idle agents, then pauses running requests
    /// (lowest rank first, those in this step included, their rows dropped). Only when nothing can be freed does a row fail.
    pub(super) fn make_step_room(&mut self, rows: &mut Vec<Row>, events: &mut Vec<EngineEvent>) {
        if !self.unified {
            return; // every sequence has its own stream of `capacity` cells: it cannot outgrow it
        }
        for _ in 0..16 {
            let need: usize = rows.iter().map(|r| r.toks.len() + (r.prefill && r.samples) as usize).sum();
            let short = (self.resident_cells() + need + MARGIN).saturating_sub(self.total_cells);
            if short == 0 {
                return;
            }
            match self.free_once(short, None, u8::MAX) {
                Some(paused) => rows.retain(|r| !paused.contains(&r.req)),
                None => {
                    // nothing left to free: the lowest-rank, youngest row fails (the executor cancels it)
                    let victim = rows.iter().enumerate().min_by_key(|(_, r)| self.jobs.get(&r.req).map_or((0, 0), |j| (kvmem::rank(j.lane), u64::MAX - j.order))).map(|(i, _)| i);
                    let Some(i) = victim else { return };
                    let r = rows.remove(i);
                    events.push(EngineEvent::Failed { req: r.req, message: "the KV cache is full and nothing can be freed".into() });
                }
            }
        }
    }

    /// Make room for `need` more promised cells for a new request or agent: false when nothing can be freed.
    pub(super) fn admit_room(&mut self, need: usize, except: Option<&str>, rank: u8) -> bool {
        if !self.unified {
            return true; // streams: nothing to count at admission, a sequence id is taken (and freed for) when it begins
        }
        for _ in 0..4 {
            let short = (self.committed_cells(except) + need).saturating_sub(self.total_cells);
            if short == 0 {
                return true;
            }
            if self.free_once(short, except, rank).is_none() {
                return false;
            }
        }
        (self.committed_cells(except) + need) <= self.total_cells
    }

    pub(super) fn refusal(&mut self, need: usize, what: &str) -> EngineError {
        self.mem.refused += 1;
        let short = (self.committed_cells(None) + need).saturating_sub(self.total_cells);
        EngineError::new(
            503,
            "kv_pool_exhausted_error",
            format!(
                "the KV cache ({} cells) cannot hold {what}: it needs {need} cells now next to what admitted requests hold and nothing can be freed (idle agents suspended, running requests paused); short by {short} cells; raise --kv-pages or --ctx",
                self.total_cells
            ),
        )
        .with_shortfall(short as u64)
    }
}
