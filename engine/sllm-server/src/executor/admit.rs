//! Admission: a request goes to a lane (or waits behind its agent, a compaction or a rollover), is deferred when the
//! lane is full, or waits for room in the engine while others run.

use super::*;

impl Executor {
    /// Steps until the soonest running request of `lane` frees its slot, times the recent step time, times the waves of
    /// queued requests ahead: when a retry has a fair chance.
    pub(super) fn retry_after_ms(&self, lane: Lane) -> u64 {
        let l = lane.index();
        let chunk = self.cfg().chunk.max(1);
        let soonest = self.running[l]
            .iter()
            .filter_map(|id| self.slots.get(id))
            .map(|s| {
                let left = if s.n_predict >= 0 { (s.n_predict as usize).saturating_sub(s.predicted_n) } else { 256 };
                s.prefill_left.div_ceil(chunk) + left.max(1)
            })
            .min()
            .unwrap_or(1);
        let waves = self.queues[l].len() / self.cfg().caps[l].max(1) + 1;
        let step = self.stats.step_ms_ema.max(1.0);
        ((step * soonest as f64 * waves as f64).ceil() as u64).clamp(10, 60_000)
    }

    pub(super) fn start(&mut self, s: StartCmd) {
        self.start_inner(s, false)
    }

    /// `from_queue`: the turn comes out of its agent's input queue (it is the queue's head, it does not queue again).
    pub(super) fn start_inner(&mut self, s: StartCmd, from_queue: bool) {
        // an agent turn as it arrived: what waits for room, so that a retry runs every admission step below once again
        let arrived = s.agent.is_some().then(|| s.clone());
        let Some(s) = self.hold_behind_compaction(s) else { return };
        // an agent turn waits in the agent's queue while it is busy or its KV is on the SSD (phase 3 lifecycle)
        let Some(s) = self.life_admit(s, from_queue) else { return };
        let Some(s) = self.rollover_admit(s) else { return };
        self.admit(s, arrived);
    }

    /// A turn on an agent whose compaction is under way (its summary being written, its window being read back) waits for
    /// the compaction (one turn; a second is refused); during a rollover it queues behind the rolled turn. `None`: held,
    /// queued or refused.
    fn hold_behind_compaction(&mut self, s: StartCmd) -> Option<StartCmd> {
        let Some(a) = s.agent.as_ref().and_then(|aid| self.agents.get(aid)) else { return Some(s) };
        if is_compaction(s.id) || a.comp.phase.is_idle() {
            return Some(s);
        }
        let aid = s.agent.clone().unwrap();
        if self.held.get(&aid).is_some_and(|h| self.rolled.contains_key(&h.id)) || self.writing.iter().any(|w| w.agent == aid) {
            self.life_queue_turn(&aid, s, false);
            return None;
        }
        if self.held.contains_key(&aid) {
            self.shared.requests_started.fetch_add(1, Relaxed);
            self.shared.requests_failed.fetch_add(1, Relaxed);
            let _ = s.events.try_send(ReqEvent::Error(compaction::compacting(&aid)));
        } else {
            self.held.insert(aid, s);
        }
        None
    }

    /// The request goes to its lane: deferred when the lane's slots and queue are full, else admitted by the engine (or
    /// waiting for room while others run, as it `arrived`, or refused).
    fn admit(&mut self, s: StartCmd, arrived: Option<StartCmd>) {
        let StartCmd { id, agent, ids, params, lane, deadline_ms, events } = s;
        self.shared.requests_started.fetch_add(1, Relaxed);
        let (cached_n, agent_lane) = match self.turn_of(agent.as_deref()) {
            Ok(t) => t,
            Err(e) => return self.refuse(id, &events, e),
        };
        let lane = lane.unwrap_or(agent_lane);
        let l = lane.index();
        let (cap, limit) = (self.cfg().caps[l], self.cfg().queue_limits[l]);
        let (running, queued) = (self.running[l].len(), self.queues[l].len());
        // the executor's own compactions are never deferred: they free pages and a held turn waits on them
        if running + queued >= cap + limit && !is_compaction(id) {
            let e = self.deferral(lane, running, cap, queued, limit);
            self.rolled.remove(&id); // a clip noted for it goes with it
            let _ = events.try_send(ReqEvent::Error(e));
            return;
        }
        let n_predict = params.n_predict;
        let turn_ids = agent.as_ref().map(|_| ids.clone());
        let req = StartRequest { id, agent: agent.clone(), ids, params, lane };
        let res = self.engine_start(&req, agent.as_deref());
        if self.waits_for_room(&res, &events) {
            // no room now, but running requests will free some: wait for one to end (see `room_retry`)
            self.shared.requests_started.fetch_sub(1, Relaxed);
            let cmd = arrived.unwrap_or(StartCmd { id, agent, ids: req.ids, params: req.params, lane: Some(lane), deadline_ms, events });
            self.room_wait_push(cmd);
            return;
        }
        match res {
            Ok(started) => {
                if let Some(aid) = &agent {
                    self.agents.get_mut(aid).unwrap().busy = Some(id);
                    self.life_turn_started(aid, turn_ids.as_deref().unwrap_or(&[]), id);
                }
                let _ = events.try_send(ReqEvent::Started { prompt_n: started.prompt_n, lane });
                let now = Instant::now();
                let order = self.next_order;
                self.next_order += 1;
                self.slots.insert(
                    id,
                    Slot {
                        events,
                        lane,
                        arrived: now,
                        began: None,
                        first: None,
                        deadline: deadline_ms.map(|ms| now + Duration::from_millis(ms)),
                        order,
                        prompt_n: started.prompt_n,
                        cached_n,
                        predicted_n: 0,
                        prefill_left: started.feed_n,
                        n_predict,
                        agent,
                        waiting: false,
                        wait_since: None,
                        wait_ms: 0.0,
                        shared_n: 0,
                    },
                );
                self.queued(id, l, cap, running);
            }
            Err(e) => self.refuse(id, &events, e),
        }
    }

    /// A refused request: counted and told why (a clip noted for it goes with it).
    fn refuse(&mut self, id: ReqId, events: &mpsc::Sender<ReqEvent>, e: EngineError) {
        self.rolled.remove(&id);
        if std::env::var_os("SLLM_LOG_FAIL").is_some() {
            eprintln!("sllm-server: refused request {id:?}: {} {} {}", e.status, e.kind, e.message);
        }
        self.shared.requests_failed.fetch_add(1, Relaxed);
        let _ = events.try_send(ReqEvent::Error(e));
    }

    /// An agent turn's context so far and its agent's lane (a plain completion: none, background); an unknown or busy
    /// agent refuses it.
    fn turn_of(&self, agent: Option<&str>) -> Result<(usize, Lane), EngineError> {
        let Some(aid) = agent else { return Ok((0, Lane::Background)) };
        match self.agents.get(aid) {
            None => Err(Self::no_agent(aid)),
            Some(a) if a.busy.is_some() => Err(EngineError::new(409, "conflict_error", format!("agent {aid} is busy with a turn"))),
            Some(a) => Ok((self.engine.agent_context_tokens(aid).unwrap_or(0), a.lane)),
        }
    }

    /// The lane is full: a 503 with when to retry and the `deferred` object the agent layer acts on.
    fn deferral(&mut self, lane: Lane, running: usize, cap: usize, queued: usize, limit: usize) -> EngineError {
        let l = lane.index();
        let ms = self.retry_after_ms(lane);
        self.stats.lanes[l].deferred += 1;
        self.shared.requests_deferred.fetch_add(1, Relaxed);
        let mut e = EngineError::new(
            503,
            "deferred",
            format!(
                "the {} lane is full: {running} running (cap {cap}), {queued} queued (limit {limit}); retry after {ms} ms",
                lane.name()
            ),
        )
        .with_retry_after(ms);
        e.deferred = Some(json!({
            "lane": lane.name(), "reason": "lane_queue_full", "running": running, "cap": cap, "queued": queued,
            "queue_limit": limit, "retry_after_ms": ms,
        }));
        e
    }

    /// The engine admits the request; under memory pressure idle agents are suspended to make room and it is asked once more.
    fn engine_start(&mut self, req: &StartRequest, agent: Option<&str>) -> Result<Started, EngineError> {
        let mut res = self.engine.start_request(req.clone());
        if self.life_cfg.relieve_on_refusal && matches!(&res, Err(e) if e.kind == "kv_pool_exhausted_error") {
            // memory pressure: suspend idle agents to make room, then try once more
            let need = res.as_ref().err().and_then(|e| e.shortfall_bytes).unwrap_or(1);
            if self.life_relieve(need, agent, None, crate::lifecycle::Why::Pressure) > 0 {
                res = self.engine.start_request(req.clone());
            }
        }
        res
    }

    /// No room in the engine now, but running requests will free some: the request waits (unless its client left).
    fn waits_for_room(&self, res: &Result<Started, EngineError>, events: &mpsc::Sender<ReqEvent>) -> bool {
        matches!(res, Err(e) if e.kind == "kv_pool_exhausted_error") && !self.slots.is_empty() && !events.is_closed()
    }

    /// An admitted request joins its lane's queue.
    fn queued(&mut self, id: ReqId, l: usize, cap: usize, running: usize) {
        self.queues[l].push_back(id);
        self.stats.lanes[l].admitted += 1;
        let q = self.queues[l].len().saturating_sub(cap.saturating_sub(running));
        self.stats.lanes[l].max_queued = self.stats.lanes[l].max_queued.max(q);
    }
}
