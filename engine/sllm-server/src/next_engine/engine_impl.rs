//! The [`Engine`] trait for [`NextEngine`].

use super::*;

impl Engine for NextEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "next".into(),
            model: self.cfg.model.display().to_string(),
            n_vocab: self.n_vocab,
            n_ctx: self.cfg.n_ctx,
            eos_id: self.vocab.eog_ids().first().copied().unwrap_or(-1),
        }
    }

    fn caps(&self) -> Value {
        // the cache's bytes per token in the libllama engine's /caps shape (profile.kv_bytes_per_token_per_seq.<kv.type>)
        // (a page's bytes over its cells: sn_kv_bytes(1) also counts a recurrent model's per-sequence states, 52 MB on
        // Qwen3.5-4B, which made the playground think one token took 52 MB)
        let per_token = unsafe { sn_page_bytes(self.e) } / PAGE;
        let state = unsafe { sn_kv_bytes(self.e, 1) }.saturating_sub(per_token);
        json!({"engine": "next", "slots": self.cfg.slots, "n_ctx": self.cfg.n_ctx, "max_rows": self.cfg.max_rows,
               "kv": {"type": "f16"}, "profile": {"kv_bytes_per_token_per_seq": {"f16": {"total_once": per_token}}, "state_bytes_per_seq": state},
               "note": "sllm-next: SLLM's own Metal kernels; exact at any agent count; greedy only",
               "grammar": "GBNF through libllama's grammar sampler; the GPU argmax is checked first and only a refused row's scores are read back",
               "prefix_sharing": "a free slot that holds the prompt's prefix, or the prefix copied from a live slot (bit-exact)",
               "lifecycle": format!("suspend / resume to the SSD ({})", self.spill_dir.display())})
    }

    fn start_request(&mut self, req: StartRequest) -> Result<Started, EngineError> {
        self.start(req)
    }

    fn prefix_state(&mut self, req: ReqId) -> Result<Option<PrefixState>, EngineError> {
        self.wait_state(req)
    }

    fn cancel(&mut self, req: ReqId) {
        if let Some(job) = self.jobs.get(&req) {
            if let (Some(id), Some((n, pending)), 0) = (job.agent.clone(), job.before.clone(), job.emitted) {
                let slot = job.slot;
                let lost = self.truncate(slot, n).unwrap_or_default();
                if let Some(a) = self.agents.get_mut(&id) {
                    a.pending = lost.into_iter().chain(pending).collect();
                }
            }
        }
        self.end(req);
    }

    fn step(&mut self, items: &[WorkItem]) -> Vec<EngineEvent> {
        self.run_step(items)
    }

    fn agent_create(&mut self, id: &str, lane: Lane, system_ids: &[i32]) -> Result<(), EngineError> {
        self.check_ids(system_ids)?;
        if system_ids.len() >= self.cfg.n_ctx {
            return Err(err(400, "exceed_context_size_error", "the system prompt does not fit the context"));
        }
        if system_ids.is_empty() || self.owners.free().is_empty() {
            // no slot yet: the agent takes one with its first turn, which reads the system prompt then (the playground's
            // turns made 256 agents beside 40 slots, and the 41st create was refused with a 503)
            self.agents.insert(id.to_string(), Agent { slot: None, interactive: lane == Lane::Interactive, pending: system_ids.to_vec(), off: None });
            return Ok(());
        }
        let owner = SlotOwner::Agent(id.to_string());
        let (slot, shared) = self.place(system_ids, system_ids.len(), true, owner.clone())?;
        if let Err(e) = self.make_room(chunks(system_ids.len()).saturating_sub(self.held(slot, false)), system_ids.len(), slot) {
            self.slot_back(slot, owner);
            return Err(e);
        }
        // the rest of the system prompt is read now, in steps of at most max_rows rows, nothing sampled
        let rest = &system_ids[shared..];
        for (c, chunk) in rest.chunks(self.cfg.max_rows).enumerate() {
            let pos: Vec<i32> = (0..chunk.len()).map(|k| (shared + c * self.cfg.max_rows + k) as i32).collect();
            let seq = vec![slot; chunk.len()];
            if let Err(m) = self.run(chunk, &seq, &pos, 0, 0, &[], false) {
                self.reset_after_failed_step(&seq, &pos);
                self.slot_back(slot, owner);
                return Err(server_err(m));
            }
            self.counters.prompt_tokens_evaluated += chunk.len() as u64;
        }
        self.agents.insert(id.to_string(), Agent { slot: Some(slot), interactive: lane == Lane::Interactive, pending: Vec::new(), off: None });
        Ok(())
    }

    fn agent_fork(&mut self, id: &str, lane: Lane, from: &str) -> Result<(), EngineError> {
        let a = self.agent_at_rest(from, "running", || err(409, "agent_suspended", format!("agent {from} is suspended: wake it before forking it")))?;
        let (src, pending) = (a.slot, a.pending.clone());
        let n = src.map_or(0, |s| self.cells[s as usize].len());
        if n > 0 && self.cut_best(src.expect("cells in a slot"), n) < n {
            return Err(err(409, "conflict_error", format!("agent {from}'s cells cannot be copied whole (sliding-window layers past their window)")));
        }
        let owner = SlotOwner::Agent(id.to_string());
        let (slot, _) = self.place(&[], 0, false, owner.clone())?;
        // the copy shares the full pages; the last, partly filled one is copied
        if let Err(e) = self.make_room(1, n, slot) {
            self.slot_back(slot, owner);
            return Err(e);
        }
        if let Some(s) = src.filter(|_| n > 0) {
            if let Err(m) = call(|b, l| unsafe { sn_copy(self.e, s, slot, n as c_int, b, l) }) {
                self.reset_slot(slot, n);
                self.slot_back(slot, owner);
                return Err(server_err(m));
            }
            self.cells[slot as usize] = self.cells[s as usize].clone();
            self.prefix.attached_agents += 1;
            self.prefix.attached_tokens += n;
        }
        self.agents.insert(id.to_string(), Agent { slot: Some(slot), interactive: lane == Lane::Interactive, pending, off: None });
        Ok(())
    }

    fn agent_free(&mut self, id: &str) {
        let reqs: Vec<ReqId> = self.jobs.iter().filter(|(_, j)| j.agent.as_deref() == Some(id)).map(|(&r, _)| r).collect();
        for r in reqs {
            self.jobs.remove(&r);
        }
        if let Some(a) = self.agents.remove(id) {
            // (a slot a resume reads into stays the read's: it is free when the read ends, see `read_back`)
            if let Some(o) = a.off {
                let _ = std::fs::remove_file(&o.path);
            }
            if let Some(s) = a.slot {
                self.slot_back(s, SlotOwner::Agent(id.to_string()));
            }
        }
    }

    fn agent_context_tokens(&self, id: &str) -> Option<usize> {
        self.agents.get(id).map(|a| self.context_of(a).len())
    }

    fn agent_tokens(&self, id: &str) -> Option<Vec<i32>> {
        // silent (latent) cells have no tokens: such a context cannot be rendered or read again
        self.agents.get(id).map(|a| self.context_of(a)).filter(|c| c.iter().all(|&t| t >= 0))
    }

    fn agent_cut(&mut self, id: &str, keep: usize) -> Result<CutStats, EngineError> {
        let a = self.agent_at_rest(id, "admitted", || err(409, "agent_suspended", format!("agent {id} is suspended: resume it first")))?;
        let held = a.slot.map_or(0, |s| self.cells[s as usize].len());
        let ctx = held + a.pending.len();
        if keep > ctx {
            return Err(err(422, "invalid_request_error", format!("the cut keeps {keep} of {ctx} tokens")));
        }
        // a cut inside the unread tokens keeps the cells; the tokens it keeps are read with the next request
        let slot = a.slot;
        let lost = match slot.filter(|_| keep <= held) {
            Some(s) => self.truncate(s, keep).map_err(server_err)?,
            None => Vec::new(),
        };
        let a = self.agents.get_mut(id).expect("checked above");
        if keep <= held {
            a.pending = lost;
        } else {
            a.pending.truncate(keep - held);
        }
        Ok(CutStats::default())
    }

    // ---- lifecycle: a slot's cells to an SSD file and back (life.rs) ------------------------------------------------

    fn agent_suspend(&mut self, id: &str) -> Result<SuspendReport, EngineError> {
        self.suspend(id)
    }

    fn agent_resume(&mut self, id: &str) -> Result<(), EngineError> {
        self.resume(id)
    }

    fn agent_drop_kv(&mut self, id: &str) -> Result<(), EngineError> {
        self.drop_kv(id)
    }

    fn lifecycle_poll(&mut self, _upload_budget: u64) -> Vec<LifecycleEvent> {
        self.poll_io()
    }

    fn lifecycle_busy(&self) -> bool {
        self.life.in_flight > 0 || !self.events.is_empty()
    }

    fn idle_wake(&self) -> Option<std::time::Duration> {
        if self.trim_pending {
            return Some(std::time::Duration::from_millis(20));
        }
        let t = self.idle_since.filter(|_| self.idle_drop && self.jobs.is_empty())?;
        Some(self.idle_after.saturating_sub(t.elapsed()))
    }

    fn idle(&mut self) -> Vec<EngineEvent> {
        if std::env::var_os("SLLM_LOG_TRIM").is_some() {
            eprintln!("next: idle jobs {} idle_since {:?} free {} mem {} MB", self.jobs.len(), self.idle_since.map(|t| t.elapsed()), self.owners.free().len(), unsafe { sn_mem_used(self.e) } >> 20);
        }
        if self.trim_pending {
            self.trim_pending = unsafe { sn_trim_pending(self.e) } > 0;
        }
        if !self.trim_pending && self.idle_wake() == Some(std::time::Duration::ZERO) {
            self.idle_since = None;
            self.drop_idle_cache();
        }
        Vec::new()
    }

    fn kv_pools(&self) -> Vec<KvPoolRoom> {
        vec![KvPoolRoom { key: POOL, total_pages: unsafe { sn_pages_total(self.e) } as usize, free_pages: self.room(), page_bytes: unsafe { sn_page_bytes(self.e) } as u64 }]
    }

    fn agent_kv(&self, id: &str) -> Option<(&'static str, usize)> {
        self.agents.get(id).filter(|a| a.off.is_none()).map(|a| (POOL, a.slot.map_or(0, |s| self.held(s, true))))
    }

    fn lifecycle_counters(&self) -> Value {
        let l = &self.life;
        json!({"ssd_bytes": l.ssd_bytes, "writes": l.writes, "reads": l.reads, "warm_resumes": l.warm_resumes, "io_in_flight": l.in_flight})
    }

    fn counters(&self) -> EngineCounters {
        let mut c = self.counters.clone();
        c.kv_mode = Some(format!("sllm-next pages ({} x {PAGE} cells, {} sequences of up to {} cells)", unsafe { sn_pages_total(self.e) }, self.cfg.slots, self.cfg.n_ctx));
        let comp: usize = self.jobs.values().filter(|j| j.agent.is_none()).map(|j| self.cells[j.slot as usize].len() + j.last.is_some() as usize).sum();
        c.kv.push(KvUse { id: "completions".into(), cells: comp, pages: None, page_type: None });
        for (id, a) in &self.agents {
            c.kv.push(KvUse { id: id.clone(), cells: self.context_of(a).len(), pages: None, page_type: None });
        }
        c.prefix = Some(PrefixCounters { waiting: self.jobs.values().filter(|j| j.wait.is_some()).count(), ..self.prefix.clone() });
        c.pages_reserved = Some(unsafe { sn_pages_total(self.e) } as u64);
        c.pages_active = Some(self.pages_used() as u64);
        c.pages_peak = Some(self.pages_peak as u64);
        c.spec = Some(json!({"mode": self.spec.mode.name(), "k": self.spec.k, "n": self.spec.n, "steps": self.spec_steps,
                             "lib_steps": self.counters.steps, "gpu_ms": self.gpu_ms, "run_ms": self.run_ms,
                             "drafted": self.spec_drafted, "kept": self.spec_kept,
                             "guess": {"on": self.guess, "k": self.guess_k, "steps": self.guess_steps, "drafted": self.guess_drafted, "kept": self.guess_kept},
                             "bg_draft": self.bg_draft, "bg_steps": self.bg_steps,
                             "pins": {"held": self.pins.len(), "made": self.pin_made, "used": self.pin_used, "tokens": self.pin_tokens},
                             "bg_drafted": self.bg_drafted, "bg_kept": self.bg_kept, "latent_rows": self.lat_rows,
                             "shared": self.shared.as_ref().map(|s| json!({"bytes": s.bytes(), "steps": self.shared_steps, "drafted": self.shared_drafted, "kept": self.shared_kept}))}));
        c
    }
}
