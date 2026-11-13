//! The [`Engine`] trait on [`LlamaEngine`].

use super::*;

impl Engine for LlamaEngine {
    fn info(&self) -> EngineInfo {
        self.info.clone()
    }

    fn caps(&self) -> Value {
        let mut c = self.caps.clone();
        c["row_exact"] = json!(self.row_exact);
        c["spec"] = self.spec_json();
        c["kv_read"] = self.readers.first().map_or(json!({"mode": "kv"}), |r| {
            json!({"mode": "f16", "readers": self.readers.len(), "reader_cells": self.readers.iter().map(|r| r.max_cells).collect::<Vec<_>>(), "block": r.block, "refill_max": READ_REFILL_MAX, "target_row_types": r.layout.row_types()})
        });
        c
    }

    fn start_request(&mut self, req: StartRequest) -> Result<Started, EngineError> {
        if req.params.latent_steps > 0 {
            return Err(EngineError::new(501, "not_supported_error", "latent_steps needs --engine next"));
        }
        self.check_ids(&req.ids)?;
        // the grammar is built first: a request libllama cannot build one for is refused before anything is reserved
        let gram = self.request_grammar(&req.params)?;
        let (base, pending) = self.agent_context(req.agent.as_deref())?;
        let total = base + req.ids.len();
        let feed_n = req.ids.len() + pending as usize;
        if total == 0 || feed_n == 0 {
            return Err(EngineError::new(422, "invalid_request_error", "empty prompt: the turn has no ids to evaluate"));
        }
        if total >= self.capacity {
            return Err(EngineError::new(
                400,
                "exceed_context_size_error",
                format!("the request exceeds the available context size ({total} >= n_ctx {})", self.capacity),
            )
            .with_context(total, self.capacity));
        }
        let (budget, ctx_limited) = reply_budget(req.params.n_predict, self.capacity - total);
        // Admission on what the request needs now: its context, its prompt and a small decode headroom, next to what is
        // already promised (queued requests included). Its sequence grows as tokens are produced; when the cache is about
        // to be full the engine frees cells then (idle agents to the SSD, lowest-priority requests paused). Short on
        // cells, it frees the same way first, and refuses only when nothing can be freed.
        let need = total + HEADROOM.min(budget);
        if !self.admit_room(need, req.agent.as_deref(), kvmem::rank(req.lane)) {
            return Err(self.refusal(need, "this request"));
        }
        if let Some(a) = req.agent.as_deref().and_then(|a| self.agents.get_mut(a)) {
            a.last_used = Instant::now();
        }
        let prompt_n = req.ids.len();
        let order = self.next_order;
        self.next_order += 1;
        self.jobs.insert(
            req.id,
            Job {
                req: req.id,
                agent: req.agent,
                ids: req.ids,
                params: req.params,
                budget,
                ctx_limited,
                lane: req.lane,
                order,
                paused: false,
                snap: None,
                seq: None,
                begun: false,
                started: None,
                first: None,
                feed: Vec::new(),
                fed: 0,
                shared_n: 0,
                len_before: 0,
                pending_before: None,
                emitted: 0,
                carry: Vec::new(),
                read: None,
                reader: 0,
                handover: false,
                gram,
            },
        );
        self.mem.max_inflight_jobs = self.mem.max_inflight_jobs.max(self.jobs.len());
        self.mem.peak_committed = self.mem.peak_committed.max(self.committed_cells(None));
        Ok(Started { prompt_n, feed_n })
    }

    fn try_begin(&mut self, req: ReqId) -> bool {
        let Some(job) = self.jobs.get(&req) else { return false };
        if job.begun {
            return true;
        }
        // the ids are only read by the prefix search: no copy when prefix sharing is off
        let (agent, ids) = (job.agent.clone(), if self.share || self.holders_on { job.ids.clone() } else { Vec::new() });
        let (rank, use_cache) = (kvmem::rank(job.lane), job.params.cache_prompt);
        match agent {
            None => self.begin_completion(req, &ids, rank, use_cache),
            Some(aid) => self.begin_agent_turn(req, &aid, &ids, rank, use_cache),
        }
    }

    /// `--kv-read f16`: a request reserves a reader when it starts, or waits (`Waiting`) while the readers it may use are held.
    /// It reads in the free reader that holds most of its prompt (`read_keep`), the one `reader_for` names on a tie; only a
    /// reader the whole prompt fits can read it. The other reader is taken only when it holds much more of the prompt
    /// (`READ_REFILL_MAX` more), and the chat reader for a background request only when the rest it still has to run is short:
    /// a chat that fits the chat reader never waits behind a long background read. A chat that has outgrown it is read in the
    /// long reader like any long prompt: in blocks at fixed positions, so the cells are the same whichever reader read them.
    /// A request whose sequence holds more cached tokens than the reader could run again (`READ_REFILL_MAX`) cannot be read
    /// exactly there. A request (or an agent's first turn) whose cached tokens are only a prefix attached from another sequence
    /// drops them and reads the whole prompt in the reader: slow, but the same tokens as a cold read. Anything else (an agent
    /// with a long history of its own: a full refill is not possible cheaply) is read in the main cache, where `--kv-read f16`
    /// is not honoured (`in_main` on /metrics counts these).
    fn prefix_state(&mut self, req: ReqId) -> Result<Option<PrefixState>, EngineError> {
        if self.readers.is_empty() {
            return Ok(self.jobs.get(&req).filter(|j| j.shared_n > 0 && j.started.is_none()).map(|j| PrefixState::Ready { feed_n: j.ids.len() - j.shared_n, shared_n: j.shared_n }));
        }
        self.drop_gone_reservations();
        let Some(lane) = self.jobs.get(&req).map(|j| j.lane) else { return Ok(None) };
        // its reader's context was being made (on another thread) when it reserved the reader
        if let Some(i) = self.readers.iter().position(|r| r.opening_for.is_some_and(|(q, _, _)| q == req)) {
            return self.reader_opened(req, i);
        }
        self.reserve_reader(req, lane)
    }

    fn cancel(&mut self, req: ReqId) {
        for i in 0..self.readers.len() {
            if self.readers[i].req == Some(req) {
                self.reader_unreserve(i);
            }
        }
        let Some(mut job) = self.jobs.remove(&req) else { return };
        self.undo_cancelled_turn(&mut job);
        self.end(job);
        self.reader_release();
    }

    fn idle(&mut self) -> Vec<EngineEvent> {
        let mut events = Vec::new();
        self.poll_handovers(&mut events);
        events
    }

    fn step(&mut self, items: &[WorkItem]) -> Vec<EngineEvent> {
        let mut events = Vec::new();
        if items.is_empty() {
            return events;
        }
        self.step_start(items, &mut events);
        // resolve the items into rows (a chunk of the prompt the reader reads goes there instead)
        let StepWork { mut rows, reads, warm } = self.resolve_items(items, &mut events);
        for (sid, tokens) in warm {
            self.warm_copy(sid, tokens);
        }
        let mut read_sampled: Vec<(ReqId, i32)> = Vec::new();
        for (req, toks, samples) in reads {
            self.read_chunk(req, toks, samples, &mut read_sampled, &mut events);
        }
        self.wait_if_all_paused(&rows);
        self.make_step_room(&mut rows, &mut events);
        self.order_rows(&mut rows);
        let mut spec_info: HashMap<ReqId, SpecInfo> = HashMap::new();
        if self.spec.mode != SpecMode::Off {
            self.plan_spec(&mut rows, &mut spec_info);
        }
        // sequences must hold their positions in order: one row per sequence per decode (the scheduler gives one per request)
        let sampled = self.decode_groups(&rows, &mut events);
        for (req, t) in read_sampled {
            self.on_token(req, t, &mut events);
        }
        if spec_info.is_empty() {
            for (req, t) in sampled {
                self.on_token(req, t, &mut events);
            }
        } else {
            self.finish_spec_rows(sampled, spec_info, &mut events);
        }
        events
    }

    fn agent_create(&mut self, id: &str, lane: Lane, system_ids: &[i32]) -> Result<(), EngineError> {
        self.check_system_ids(system_ids)?;
        if !self.admit_room(system_ids.len() + MARGIN, None, 0) {
            return Err(self.refusal(system_ids.len(), "the system prompt"));
        }
        if system_ids.is_empty() && !self.unified {
            // nothing to cache yet: no stream until its first turn
            self.agents.insert(id.to_string(), AgentSeq { lane, seq: Seq { sid: -1, tokens: Vec::new(), pending: None }, suspended: None, last_used: Instant::now() });
            return Ok(());
        }
        let sid = self.alloc_sid_for(kvmem::rank(lane), None).ok_or_else(|| self.no_free_sid())?;
        self.holder_for_system_prompt(sid, system_ids);
        let skip = self.attach_system_prompt(sid, system_ids);
        // The prefill is synchronous, in chunks of the context's batch size. The last token gets logits (libllama wants an
        // output) which are ignored.
        if let Err(message) = self.prefill_sync(sid, system_ids, skip) {
            self.release_sid(sid);
            return Err(model_err(message));
        }
        let seq = Seq { sid, tokens: system_ids.to_vec(), pending: None };
        self.agents.insert(id.to_string(), AgentSeq { lane, seq, suspended: None, last_used: Instant::now() });
        Ok(())
    }

    fn agent_free(&mut self, id: &str) {
        let reqs: Vec<ReqId> = self.agent_jobs(id).map(|j| j.req).collect();
        for r in reqs {
            if let Some(job) = self.jobs.remove(&r) {
                self.end(job);
            }
        }
        if let Some(a) = self.agents.remove(id) {
            if let Some(s) = a.suspended {
                self.discard_snapshot(s);
            }
            self.release_sid(a.seq.sid);
        }
    }

    fn agent_context_tokens(&self, id: &str) -> Option<usize> {
        self.agents.get(id).map(|a| match &a.suspended {
            Some(s) => s.tokens.len() + a.seq.pending.is_some() as usize,
            None => a.seq.tokens.len() + a.seq.pending.is_some() as usize,
        })
    }

    fn counters(&self) -> EngineCounters {
        let mut c = self.counters.clone();
        c.kv_mode = Some(if self.unified { "unified (libllama)" } else { "per-sequence streams (libllama)" }.into());
        let comp: Vec<&Seq> = self.jobs.values().filter_map(|j| j.seq.as_ref()).collect();
        c.kv.push(KvUse { id: "completions".into(), cells: comp.iter().map(|s| s.tokens.len() + s.pending.is_some() as usize).sum(), pages: None, page_type: None });
        let mut ids: Vec<&AgentId> = self.agents.keys().collect();
        ids.sort_by_key(|s| s.trim_start_matches("ag_").parse::<u64>().unwrap_or(0));
        for id in ids {
            let cells = self.agent_context_tokens(id).unwrap_or(0);
            c.kv.push(KvUse { id: id.clone(), cells, pages: None, page_type: None });
        }
        c.prefix = Some(self.prefix.clone());
        c.spec = Some(self.spec_json());
        c.kv_read = (!self.readers.is_empty()).then(|| {
            let mut sum = json!({});
            for r in &self.readers {
                for (k, v) in r.json().as_object().expect("an object") {
                    let a = sum.get(k).and_then(Value::as_f64).unwrap_or(0.0) + v.as_f64().unwrap_or(0.0);
                    sum[k] = if v.is_u64() { json!(a as u64) } else { json!(a) };
                }
            }
            sum
        });
        c
    }

    fn take_timings(&mut self, req: ReqId) -> Option<(f64, f64)> {
        self.timings.remove(&req)
    }

    fn last_step_breakdown(&self) -> Option<StepBreakdown> {
        (self.last_bd.graphs > 0).then_some(self.last_bd)
    }

    // ---- lifecycle: suspend to the SSD with llama_state_seq_get_data, restore with llama_state_seq_set_data ----

    fn agent_suspend(&mut self, id: &str) -> Result<SuspendReport, EngineError> {
        self.resident_agent(id, "queued or running", || EngineError::new(409, "conflict_error", format!("agent {id} is already suspended")))?;
        self.suspend_agent(id)
    }

    fn agent_resume(&mut self, id: &str) -> Result<(), EngineError> {
        let t0 = Instant::now();
        let ag = self.agents.get_mut(id).ok_or_else(|| EngineError::new(404, "not_found_error", format!("no agent {id}")))?;
        let Some(snap) = &mut ag.suspended else {
            return Err(EngineError::new(409, "conflict_error", format!("agent {id} is not suspended")));
        };
        if matches!(snap.st, SnapSt::Reading | SnapSt::Ready(_)) {
            return Ok(()); // the read is on its way already
        }
        // The read starts now, on the I/O thread (behind the write of the same file when that is still running).
        snap.st = SnapSt::Reading;
        snap.resumed = Some(t0);
        let (key, path, bytes) = (snap.key, snap.path.clone(), snap.bytes as usize);
        self.io.read(key, path, bytes);
        self.stall.resume(t0.elapsed().as_secs_f64() * 1e3);
        Ok(())
    }

    fn agent_drop_kv(&mut self, id: &str) -> Result<(), EngineError> {
        let ag = self.agents.get_mut(id).ok_or_else(|| EngineError::new(404, "not_found_error", format!("no agent {id}")))?;
        let sid = ag.seq.sid;
        ag.seq.tokens.clear();
        ag.seq.pending = None;
        let snap = ag.suspended.take();
        if let Some(s) = snap {
            self.discard_snapshot(s);
        }
        self.drop_stream(id, sid);
        Ok(())
    }

    fn lifecycle_poll(&mut self, _upload_budget: u64) -> Vec<LifecycleEvent> {
        self.poll_io();
        std::mem::take(&mut self.events)
    }

    fn lifecycle_busy(&self) -> bool {
        !self.events.is_empty() || self.io.busy() || self.jobs.values().any(|j| matches!(j.snap.as_ref().map(|s| &s.st), Some(SnapSt::Ready(_)))) || self.agents.values().any(|a| matches!(a.suspended.as_ref().map(|s| &s.st), Some(SnapSt::Ready(_))))
    }

    fn lifecycle_counters(&self) -> Value {
        let st = &self.stall;
        json!({
            "ssd_bytes": self.ssd_bytes, "writes": self.ssd_writes, "reads": self.ssd_reads, "io_in_flight": self.io.in_flight(),
            // time the executor thread spent inside the engine's suspend (state copy), wake call and restore (set_data)
            "mem": {
                "total_cells": self.total_cells, "resident_cells": self.resident_cells(), "committed_cells": self.committed_cells(None),
                "headroom": HEADROOM, "margin": MARGIN, "preempt_suspends": self.mem.preempt_suspends,
                "preempt_pauses": self.mem.preempt_pauses, "resumed": self.mem.resumed, "refused": self.mem.refused,
                "paused_now": self.jobs.values().filter(|j| j.paused).count(), "peak_committed": self.mem.peak_committed,
                "max_inflight_jobs": self.mem.max_inflight_jobs,
            },
            "prefix_holders": {
                "on": self.holders_on, "n": self.holders.len(), "cells": self.holder_cells(), "created": self.holder_stats.created,
                "evicted": self.holder_stats.evicted, "attached": self.holder_stats.attached, "attached_tokens": self.holder_stats.attached_tokens,
                "warm_kept": self.holder_stats.warm_kept, "warm_taken": self.holder_stats.warm_taken,
            },
            "stream_runs": {
                "steps": self.stream_runs.steps, "runs": self.stream_runs.runs, "split_steps": self.stream_runs.split_steps,
                "max_runs": self.stream_runs.max_runs, "wide_steps": self.stream_runs.wide_steps, "wide_runs": self.stream_runs.wide_runs,
                "moves": self.stream_runs.moves, "move_steps": self.stream_runs.move_steps, "pack": self.seq_pack,
                "copy_partial": self.copy_partial, "copies_partial": self.stream_runs.copies_partial,
                "copy_rows": self.stream_runs.copy_rows, "copies_whole": self.stream_runs.copies_whole,
                "copies_cut": ffi::copy_stats().0, "copy_bytes_saved": ffi::copy_stats().1,
            },
            "prompt_cache": self.pcache.as_ref().map(|c| json!({
                "dir": c.dir().display().to_string(), "entries": c.len(), "bytes": c.bytes(), "hits": c.hits,
                "hit_tokens": c.hit_tokens, "saves": c.saves, "evictions": c.evictions,
                "turn_saves": c.turn_saves, "turn_save_skips": c.turn_save_skips, "turn_save": self.turn_save && self.readers.is_empty(),
            })),
            "stall_ms": {
                "suspend": {"n": st.suspends, "sum": st.suspend_ms_sum, "max": st.suspend_ms_max},
                "wake_call": {"n": st.resumes, "sum": st.resume_ms_sum, "max": st.resume_ms_max},
                "restore_apply": {"n": st.applies, "sum": st.apply_ms_sum, "max": st.apply_ms_max},
            },
        })
    }

    fn agent_tokens(&self, id: &str) -> Option<Vec<i32>> {
        let a = self.agents.get(id)?;
        let mut v = match &a.suspended {
            Some(s) => s.tokens.clone(),
            None => a.seq.tokens.clone(),
        };
        v.extend(a.seq.pending);
        Some(v)
    }

    fn agent_cut(&mut self, id: &str, keep: usize) -> Result<CutStats, EngineError> {
        let a = self.resident_agent(id, "admitted", || EngineError::new(409, "agent_suspended", format!("agent {id} is suspended: resume it first")))?;
        if keep > a.seq.tokens.len() {
            return Err(EngineError::new(422, "invalid_request_error", format!("the cut keeps {keep} of {} cached tokens", a.seq.tokens.len())));
        }
        let sid = a.seq.sid;
        // a sliding-window ring has overwritten what it held longest ago: it can only start again
        if (keep > 0 && self.swa_ring) || !self.seq_rm(sid, keep as i32, -1) {
            return Err(EngineError::new(501, "not_supported_error", "this model's cache cannot drop a token range (compaction needs it)"));
        }
        let a = self.agents.get_mut(id).expect("present above");
        a.seq.tokens.truncate(keep);
        a.seq.pending = None;
        Ok(CutStats::default())
    }
}

impl LlamaEngine {
    /// A system prompt must be valid token ids and leave room in the context.
    fn check_system_ids(&self, system_ids: &[i32]) -> Result<(), EngineError> {
        self.check_ids(system_ids)?;
        if system_ids.len() >= self.capacity {
            return Err(EngineError::new(400, "exceed_context_size_error", "the system prompt does not fit the context"));
        }
        Ok(())
    }

    /// The requests agent `id` has queued, admitted or running.
    fn agent_jobs<'a>(&'a self, id: &'a str) -> impl Iterator<Item = &'a Job> {
        self.jobs.values().filter(move |j| j.agent.as_deref() == Some(id))
    }

    /// The agent `id` for an operation that needs it in memory and without a request: refused when it is unknown, has a request
    /// in `state` ("queued or running", "admitted") or is suspended (`suspended` is that refusal).
    fn resident_agent(&self, id: &str, state: &str, suspended: impl FnOnce() -> EngineError) -> Result<&AgentSeq, EngineError> {
        if self.agent_jobs(id).next().is_some() {
            return Err(EngineError::new(409, "conflict_error", format!("agent {id} has a request {state}")));
        }
        let ag = self.agents.get(id).ok_or_else(|| EngineError::new(404, "not_found_error", format!("no agent {id}")))?;
        if ag.suspended.is_some() {
            return Err(suspended());
        }
        Ok(ag)
    }
}
