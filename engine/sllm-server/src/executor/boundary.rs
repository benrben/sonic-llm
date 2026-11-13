//! The executor's work at a step boundary: sweep, fill, prefix sharing, plan, and the step's events fanned out.

use super::*;

impl Executor {
    /// At the step boundary: cancel requests whose client is gone, stop those past their deadline.
    pub(super) fn sweep(&mut self) {
        let now = Instant::now();
        let mut gone = Vec::new();
        let mut expired = Vec::new();
        for (&id, s) in &self.slots {
            if s.events.is_closed() {
                gone.push(id);
            } else if s.deadline.is_some_and(|d| now >= d) {
                expired.push(id);
            }
        }
        for id in gone {
            self.stop_request(id, StopReason::Cancelled);
        }
        for id in expired {
            self.stop_request(id, StopReason::Deadline);
        }
    }

    /// Queued requests take free slots, in arrival order per lane.
    pub(super) fn fill(&mut self) {
        for lane in Lane::ALL {
            let l = lane.index();
            let cap = self.cfg().caps[l];
            let mut i = 0;
            while i < self.queues[l].len() {
                let id = self.queues[l][i];
                // the executor's own compactions run side by side: agents that reach their limit together batch, they do
                // not wait in line for the one helper slot
                if self.running[l].len() >= cap && !is_compaction(id) {
                    i += 1;
                    continue;
                }
                if self.engine.try_begin(id) {
                    self.queues[l].remove(i);
                    self.running[l].push(id);
                    if let Some(s) = self.slots.get_mut(&id) {
                        s.began = Some(Instant::now());
                    }
                    self.refresh_prefix(id);
                } else {
                    i += 1;
                }
            }
        }
    }

    /// Running requests of `lane` that wait on pages.
    pub(super) fn waiting_in(&self, lane: Lane) -> usize {
        self.running[lane.index()].iter().filter(|id| self.slots.get(id).is_some_and(|s| s.waiting)).count()
    }

    /// Ask the engine how a request's prompt is served (prefix sharing): when it is given a slot, and at every step
    /// boundary while it waits on pages.
    pub(super) fn refresh_prefix(&mut self, id: ReqId) {
        match self.engine.prefix_state(id) {
            Ok(None) => {}
            Ok(Some(PrefixState::Ready { feed_n, shared_n })) => {
                if let Some(s) = self.slots.get_mut(&id) {
                    if let Some(t) = s.wait_since.take() {
                        s.wait_ms += t.elapsed().as_secs_f64() * 1e3;
                    }
                    s.waiting = false;
                    s.prefill_left = feed_n;
                    s.shared_n = shared_n;
                    self.stats.prefix_shared_tokens += shared_n as u64;
                }
            }
            Ok(Some(PrefixState::Waiting { .. })) => {
                if let Some(s) = self.slots.get_mut(&id) {
                    if !s.waiting {
                        s.waiting = true;
                        s.wait_since = Some(Instant::now());
                        self.stats.prefix_waited += 1;
                    }
                }
            }
            Err(e) => self.fail(id, e),
        }
    }

    /// At the step boundary: requests waiting on pages attach once the pages are committed.
    pub(super) fn poll_waiting(&mut self) {
        let waiting: Vec<ReqId> = self.running.iter().flatten().copied().filter(|id| self.slots.get(id).is_some_and(|s| s.waiting)).collect();
        for id in waiting {
            self.refresh_prefix(id);
        }
    }

    pub(super) fn plan(&mut self) -> Plan {
        let running: Vec<Running> = self
            .running
            .iter()
            .flatten()
            .filter_map(|id| self.slots.get(id).filter(|s| !s.waiting).map(|s| Running { req: *id, lane: s.lane, prefill_left: s.prefill_left, order: s.order, chunk_cap: crate::llama_engine::read_block_at(s.cached_n + s.prompt_n.saturating_sub(s.prefill_left), usize::MAX) }))
            .filter(|r| !(r.lane == Lane::Background && self.compact_yield && !self.summarizing.is_empty()))
            .collect();
        self.planner.plan(&running)
    }

    pub(super) fn after_step(&mut self, plan: &Plan, events: Vec<EngineEvent>, us: u64) {
        let ms = us as f64 / 1e3;
        let st = &mut self.stats;
        st.step_ms_ema = if st.steps == 0 { ms } else { 0.9 * st.step_ms_ema + 0.1 * ms };
        st.steps += 1;
        st.steps_with_interactive += plan.interactive as u64;
        st.bg_prefill_deferred_steps += plan.bg_deferred as u64;
        st.burst_skip_steps += plan.burst_skip as u64;
        // Prefill progress first: a request that finishes in this step is removed by its events below.
        for w in &plan.items {
            if let (Work::Prefill { n }, Some(s)) = (w.work, self.slots.get_mut(&w.req)) {
                s.prefill_left = s.prefill_left.saturating_sub(n);
            }
        }
        self.planner.observe(ms, plan);
        self.dispatch(events);
        self.stats.last = plan.clone();
    }

    pub(super) fn dispatch(&mut self, events: Vec<EngineEvent>) {
        for ev in events {
            match ev {
                EngineEvent::Token { req, id, text, probs } => self.dispatch_token(req, id, text, probs),
                EngineEvent::Finished { req, reason } => self.finish(req, reason),
                EngineEvent::Failed { req, message } => self.fail(req, EngineError::new(500, "server_error", message)),
            }
        }
    }

    /// A token to its request's channel (and its agent's history); a client that has stopped reading is cancelled.
    fn dispatch_token(&mut self, req: ReqId, id: i32, text: String, probs: Option<Box<TokProbs>>) {
        let Some(slot) = self.slots.get_mut(&req) else { return };
        slot.first.get_or_insert_with(Instant::now);
        slot.predicted_n += 1;
        // the agent's token history (phase 3 lifecycle)
        if let Some(a) = slot.agent.as_deref().and_then(|aid| self.agents.get_mut(aid)) {
            a.life.history.push(id);
        }
        // One slot stays free for the final event: a client that has stopped reading is cancelled.
        let stalled = slot.events.capacity() <= 1 && !slot.events.is_closed();
        if slot.events.is_closed() || stalled || slot.events.try_send(ReqEvent::Token { id, text, probs }).is_err() {
            if stalled {
                self.shared.requests_stalled.fetch_add(1, Relaxed);
            }
            self.stop_request(req, StopReason::Cancelled);
        }
    }
}
