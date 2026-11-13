//! Requests and agents coming and going: admission checks, a request's start (its sequence and the shared start it gets),
//! a cancel, and a new agent's system prompt.

use super::*;

impl LlamaEngine {
    pub(super) fn check_ids(&self, ids: &[i32]) -> Result<(), EngineError> {
        let n = self.n_vocab;
        if ids.iter().any(|&i| i < 0 || i as usize >= n) {
            return Err(EngineError::new(422, "invalid_request_error", format!("token id outside 0..{n}")));
        }
        Ok(())
    }

    /// The request's grammar (`grammar`), None without one; an error when libllama cannot build it.
    pub(super) fn request_grammar(&self, params: &SamplingParams) -> Result<Option<Box<Gram>>, EngineError> {
        let Some(g) = params.grammar.as_deref().filter(|g| !g.is_empty()) else { return Ok(None) };
        let vocab = unsafe { sys::llama_model_get_vocab(self.model) };
        let bad = || EngineError::new(400, "invalid_request_error", "invalid grammar: libllama could not build it (GBNF syntax error, no `root` rule, an undefined rule or left recursion; the reason is on the server's stderr)");
        Ok(Some(Box::new(Gram::new(vocab, g).ok_or_else(bad)?)))
    }

    /// The context a turn of `agent` starts from: its cells (the pending token included), and whether a token is pending.
    /// (0, false) for a completion; an error when the agent is unknown or suspended.
    pub(super) fn agent_context(&self, agent: Option<&str>) -> Result<(usize, bool), EngineError> {
        let Some(a) = agent else { return Ok((0usize, false)) };
        let ag = self.agents.get(a).ok_or_else(|| EngineError::new(404, "not_found_error", format!("no agent {a}")))?;
        if ag.suspended.is_some() {
            return Err(EngineError::new(409, "agent_suspended", format!("agent {a} is suspended: resume it first")));
        }
        Ok((ag.seq.tokens.len() + ag.seq.pending.is_some() as usize, ag.seq.pending.is_some()))
    }

    /// A completion takes a sequence: the warm holder it starts like, else a free id and the longest start it can get
    /// without prefilling. False while no sequence id can be had.
    pub(super) fn begin_completion(&mut self, req: ReqId, ids: &[i32], rank: u8, use_cache: bool) -> bool {
        if let Some((sid, k)) = use_cache.then(|| self.take_warm(ids)).flatten() {
            self.prefix.attached_agents += 1;
            self.prefix.attached_tokens += k;
            let job = self.jobs.get_mut(&req).expect("present above");
            job.seq = Some(Seq { sid, tokens: ids[..k].to_vec(), pending: None });
            job.shared_n = k;
            job.begun = true;
            return true;
        }
        let Some(sid) = self.alloc_sid_for(rank, None) else { return false };
        let shared = self.attach_start(sid, ids, use_cache);
        let mut seq = Seq { sid, tokens: ids[..shared].to_vec(), pending: None };
        if shared == 0 && !self.readers.is_empty() {
            // --kv-read: it waits for a reader without a stream, an id it held would be a hole (see step)
            self.release_sid(sid);
            seq.sid = -1;
        }
        let job = self.jobs.get_mut(&req).expect("present above");
        job.seq = Some(seq);
        job.shared_n = shared;
        job.begun = true;
        true
    }

    /// An agent's turn begins: an agent without a stream takes one, and the first turn of an empty agent may start from a
    /// shared start. False while no sequence id can be had.
    pub(super) fn begin_agent_turn(&mut self, req: ReqId, aid: &str, ids: &[i32], rank: u8, use_cache: bool) -> bool {
        // an agent without a stream (per-sequence streams: it was empty, or its cache was dropped) takes one
        if self.agents.get(aid).is_some_and(|a| a.seq.sid < 0) {
            let Some(sid) = self.alloc_sid_for(rank, Some(aid)) else { return false };
            if let Some(a) = self.agents.get_mut(aid) {
                a.seq.sid = sid;
            }
        }
        // the first turn of an agent with an empty context may share a prefix
        let empty = self.agents.get(aid).is_some_and(|a| a.seq.tokens.is_empty() && a.seq.pending.is_none());
        let mut shared = 0;
        if empty {
            let sid = self.agents[aid].seq.sid;
            shared = self.attach_start(sid, ids, use_cache);
            if let Some(a) = self.agents.get_mut(aid).filter(|_| shared > 0) {
                a.seq.tokens = ids[..shared].to_vec();
            }
        }
        let job = self.jobs.get_mut(&req).expect("present above");
        job.shared_n = shared;
        job.begun = true;
        true
    }

    /// The fresh sequence `sid` starts from the longest start of `ids` it can get without prefilling: a saved state on the
    /// disk (`use_cache`), else a live sequence's cells (holder models make a holder for a shared start first). Returns
    /// how many tokens it starts with (0: none).
    fn attach_start(&mut self, sid: i32, ids: &[i32], use_cache: bool) -> usize {
        let mut live = self.find_prefix(ids, Some(sid));
        let covered = live.map_or(0, |b| b.1).max(if use_cache { self.saved_start(ids) } else { 0 });
        if self.holder_for_start(ids, covered) {
            live = self.find_prefix(ids, Some(sid));
        }
        let disk = if use_cache { self.cache_load(sid, ids, ids.len().saturating_sub(1), live.map_or(0, |b| b.1)) } else { 0 };
        let k = if disk > 0 {
            disk
        } else if let Some((src, k)) = live {
            self.attach_prefix(src, sid, k);
            self.pt_set(sid, &ids[..k]);
            if use_cache {
                self.cache_save_shared(src, ids, k);
            }
            k
        } else {
            0
        };
        if k > 0 {
            self.prefix.attached_agents += 1;
            self.prefix.attached_tokens += k;
        }
        k
    }

    /// What a cancelled agent turn did to the agent's context is undone, as far as it can be.
    pub(super) fn undo_cancelled_turn(&mut self, job: &mut Job) {
        if job.agent.is_none() {
            return;
        }
        if job.paused {
            // stopped while paused: the agent's cache is incomplete, its history is what is left
            let aid = job.agent.clone().expect("checked");
            let sid = Self::seq_for(&mut self.agents, job).map_or(-1, |seq| {
                seq.tokens.clear();
                seq.pending = None;
                seq.sid
            });
            self.drop_stream(&aid, sid);
            self.events.push(LifecycleEvent::KvLost { agent: aid });
        } else if job.first.is_none() && job.started.is_some() {
            // a turn stopped before its first token did not happen: the agent's context goes back to before it
            let (len_before, pending_before) = (job.len_before, job.pending_before);
            if let Some(seq) = Self::seq_for(&mut self.agents, job) {
                let sid = seq.sid;
                seq.tokens.truncate(len_before);
                seq.pending = pending_before;
                self.seq_rm(sid, len_before as i32, -1);
            }
        } else if job.started.is_none() && job.shared_n > 0 {
            // attached a shared prefix but never prefilled: empty again
            if let Some(seq) = Self::seq_for(&mut self.agents, job) {
                let sid = seq.sid;
                seq.tokens.clear();
                self.seq_rm(sid, 0, -1);
            }
        }
    }

    /// No sequence id for a new agent.
    pub(super) fn no_free_sid(&self) -> EngineError {
        let held = if self.unified { "the sequence ids are all taken".to_string() } else { format!("the {} KV streams are all held by running requests", self.total_cells / self.capacity.max(1)) };
        EngineError::new(
            503,
            "unavailable_error",
            format!("no free sequence id: {held}; retry when one ends, or start with a larger --kv-pages or --auto-mode --agents N (without them the cache holds the default job of {} agents and a chat)", crate::profile::DEFAULT_AGENTS),
        )
    }

    /// The second agent with this system prompt makes a frozen holder for it (models that cannot copy a cache in part).
    pub(super) fn holder_for_system_prompt(&mut self, sid: i32, system_ids: &[i32]) {
        if !self.holders_on || system_ids.len() < HOLDER_MIN {
            return;
        }
        let seen = self.seen_prefix.entry(hash_ids(system_ids)).or_insert(0);
        *seen += 1;
        let seen = *seen;
        if self.seen_prefix.len() > 256 {
            self.seen_prefix.clear();
        }
        let n = system_ids.len() / self.share_align * self.share_align;
        let held = self.find_prefix_n(&system_ids[..n], Some(sid), false).is_some_and(|b| b.1 == n);
        if seen >= 2 && !held && n >= HOLDER_MIN {
            self.make_holder(&system_ids[..n]);
        }
    }

    /// A new agent's sequence `sid` starts from the longest start of its system prompt it can get without prefilling: a
    /// saved state on the disk, else a live sequence's cells. Returns how many tokens it starts with (0: none).
    pub(super) fn attach_system_prompt(&mut self, sid: i32, system_ids: &[i32]) -> usize {
        let live = self.find_prefix_n(system_ids, Some(sid), false);
        let disk = self.cache_load(sid, system_ids, system_ids.len(), live.map_or(0, |b| b.1));
        if disk > 0 {
            self.prefix.attached_agents += 1;
            self.prefix.attached_tokens += disk;
            return disk;
        }
        let Some((src, k)) = live else { return 0 };
        self.attach_prefix(src, sid, k);
        self.pt_set(sid, &system_ids[..k]);
        self.prefix.attached_agents += 1;
        self.prefix.attached_tokens += k;
        self.cache_save_shared(src, system_ids, k);
        k
    }
}

/// The tokens a request may produce in the `room` left in its context, and whether the context (not `n_predict`) sets them.
pub(super) fn reply_budget(n_predict: i64, room: usize) -> (usize, bool) {
    if n_predict < 0 || n_predict as usize > room { (room, true) } else { (n_predict as usize, false) }
}
