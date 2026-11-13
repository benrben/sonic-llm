//! The executor's commands from the HTTP side ([`Command`]), handled between two steps.

use super::*;

impl Executor {
    pub(super) fn no_agent(id: &str) -> EngineError {
        EngineError::new(404, "not_found_error", format!("no agent {id}"))
    }

    pub(super) fn handle(&mut self, c: Command) {
        self.life_events(); // the engine may have suspended agents by itself (memory pressure) since the last command
        match c {
            Command::Start(s) => self.start(*s),
            Command::Cancel(id) => self.cancel_cmd(id),
            Command::CancelReq { id, reply } => {
                let r = self.room_cancel(id) || self.stop_request(id, StopReason::Cancelled) || self.rollover_cancel(id) || self.life_cancel_queued(id);
                let _ = reply.send(r);
            }
            Command::Shutdown => {}
            agent_cmd => self.handle_agent(agent_cmd),
        }
        self.publish_counters();
    }

    /// The client went away: its request is dropped wherever it is (waiting for room, running, held behind a rollover, or
    /// queued behind its agent).
    fn cancel_cmd(&mut self, id: ReqId) {
        if !self.room_cancel(id) && !self.stop_request(id, StopReason::Cancelled) && !self.rollover_cancel(id) {
            self.life_cancel_queued(id);
        }
    }

    /// The agents' commands: create, read, list, delete, cancel ([`Executor::handle_agent_state`] the rest).
    fn handle_agent(&mut self, c: Command) {
        match c {
            Command::AgentCreate { lane, system_ids, from, reply } => {
                let _ = reply.send(self.agent_create(lane, system_ids, from));
            }
            Command::AgentGet { id, reply } => {
                let _ = reply.send(self.agent_info(&id).ok_or_else(|| Self::no_agent(&id)));
            }
            Command::AgentList { reply } => {
                let _ = reply.send(self.agent_list());
            }
            Command::AgentDelete { id, reply } => {
                let _ = reply.send(self.agent_delete(&id));
            }
            Command::AgentCancel { id, reply } => {
                let _ = reply.send(self.agent_cancel(&id));
            }
            state_cmd => self.handle_agent_state(state_cmd),
        }
    }

    /// The commands on an agent's state: suspend, wake, compact, its compaction policy and its extracted fields.
    fn handle_agent_state(&mut self, c: Command) {
        match c {
            Command::AgentSuspend { id, tier, reply } => {
                let _ = reply.send(self.life_suspend_cmd(&id, tier));
            }
            Command::AgentWake { id, reply } => {
                let _ = reply.send(self.life_wake_cmd(&id));
            }
            Command::AgentCompact { id, spec, reply } => {
                if let Err((e, Some(r))) = self.begin_compaction(&id, spec, Some(reply)) {
                    let _ = r.send(Err(e));
                }
            }
            Command::AgentPolicy { id, policy, reply } => {
                let r = self.agent_policy(&id, policy);
                let _ = reply.send(r.map(|()| self.agent_info(&id).unwrap()));
            }
            Command::AgentFields { id, op, reply } => {
                let _ = reply.send(self.agent_fields(&id, op));
            }
            _ => unreachable!("request commands and the other agent commands are handled by Executor::handle"),
        }
    }

    /// `POST /v1/agents`: a new agent, or a fork of `from`; under memory pressure idle agents are suspended to make room
    /// and the engine is asked once more.
    fn agent_create(&mut self, lane: Lane, system_ids: Vec<i32>, from: Option<String>) -> Result<AgentInfo, EngineError> {
        let id = format!("ag_{}", self.next_agent);
        // a fork: the source must be here and idle; it starts with the source's history (what a rebuild reads)
        let src = self.fork_source(from.as_deref());
        let src_hist = src.as_ref().ok().cloned().flatten();
        let mut r0 = src.and_then(|_| self.engine_agent_new(&id, lane, &system_ids, from.as_deref()));
        if let Err(e) = &r0 {
            if e.kind == "kv_pool_exhausted_error" && self.life_cfg.relieve_on_refusal {
                // memory pressure (phase 3): suspend idle agents to make room, then try once more
                let need = e.shortfall_bytes.unwrap_or(1);
                if self.life_relieve(need, from.as_deref(), None, crate::lifecycle::Why::Pressure) > 0 {
                    r0 = self.engine_agent_new(&id, lane, &system_ids, from.as_deref());
                }
            }
        }
        r0?;
        Ok(self.register_agent(id, lane, src_hist.unwrap_or_else(|| system_ids.clone())))
    }

    /// The history a fork of `from` starts with (None: not a fork); the source must be here and idle.
    fn fork_source(&self, from: Option<&str>) -> Result<Option<Vec<i32>>, EngineError> {
        match from {
            None => Ok(None),
            Some(f) => match self.agents.get(f) {
                None => Err(Self::no_agent(f)),
                Some(a) if a.busy.is_some() || !a.inbox.is_empty() => {
                    Err(EngineError::new(409, "conflict_error", format!("agent {f} has a turn running or queued")))
                }
                Some(a) => Ok(Some(a.life.history.clone())),
            },
        }
    }

    /// The engine's side of a new agent: a fork of `from`, or an agent reading `system_ids`.
    fn engine_agent_new(&mut self, id: &str, lane: Lane, system_ids: &[i32], from: Option<&str>) -> Result<(), EngineError> {
        match from {
            Some(f) => self.engine.agent_fork(id, lane, f),
            None => self.engine.agent_create(id, lane, system_ids),
        }
    }

    /// The executor's record of a new agent whose history is `hist` (the default compaction policy when it fits).
    fn register_agent(&mut self, id: String, lane: Lane, hist: Vec<i32>) -> AgentInfo {
        self.next_agent += 1;
        let life = AgentLife::new(hist.clone());
        let policy = self.comp_cfg.default_policy.filter(|p| p.validate(hist.len()).is_ok());
        let comp = compaction::AgentState::new(hist.len(), policy);
        self.agents.insert(id.clone(), AgentRec { lane, turns: 0, busy: None, created: now_unix(), comp, life, inbox: VecDeque::new() });
        self.shared.agents_created.fetch_add(1, Relaxed);
        self.life_save(&id);
        self.agent_info(&id).unwrap()
    }

    /// `GET /v1/agents`: every agent, oldest first.
    fn agent_list(&self) -> Vec<AgentInfo> {
        let mut ids: Vec<&String> = self.agents.keys().collect();
        ids.sort_by_key(|s| s.trim_start_matches("ag_").parse::<u64>().unwrap_or(0));
        ids.into_iter().filter_map(|i| self.agent_info(i)).collect()
    }

    /// `DELETE /v1/agents/{id}`: its turn is stopped, its queued and held turns refused, its cells freed. Returns its last
    /// record and the context tokens freed.
    fn agent_delete(&mut self, id: &str) -> Result<(AgentInfo, usize), EngineError> {
        self.room_gen += 1; // its pages and slot free up
        let Some(info) = self.agent_info(id) else { return Err(Self::no_agent(id)) };
        if let Some(req) = self.agents[id].busy {
            self.stop_request(req, StopReason::Cancelled);
        }
        self.life_forget(id);
        self.rollover_forget(id);
        if let Some(h) = self.held.remove(id) {
            self.rolled.remove(&h.id);
            let _ = h.events.try_send(ReqEvent::Error(Self::no_agent(id)));
        }
        self.engine.agent_free(id);
        self.agents.remove(id);
        let freed = info.context_tokens;
        Ok((info, freed))
    }

    /// `POST /v1/agents/{id}/cancel`: stop its running turn, or the turn held behind its compaction. True if one was.
    fn agent_cancel(&mut self, id: &str) -> Result<bool, EngineError> {
        match self.agents.get(id) {
            None => Err(Self::no_agent(id)),
            // a compaction runs to its end (it is short, and the context is cut already); a turn held behind it (or behind
            // its rollover, whether its notes run or wait for room) is what the agent's caller is waiting on
            Some(_) if self.held.get(id).is_some_and(|h| self.rolled.contains_key(&h.id)) => {
                let req = self.held[id].id;
                Ok(self.rollover_cancel(req))
            }
            Some(a) if a.busy.is_some_and(is_compaction) || self.held.contains_key(id) => match self.held.remove(id) {
                Some(h) => {
                    self.shared.requests_started.fetch_add(1, Relaxed); // counted as rollover_cancel counts a held turn
                    self.shared.requests_cancelled.fetch_add(1, Relaxed);
                    let _ = h.events.try_send(ReqEvent::Error(EngineError::new(499, "cancelled", "cancelled while its agent was compacting")));
                    Ok(true)
                }
                None => Ok(false),
            },
            Some(a) => match a.busy {
                Some(req) => Ok(self.stop_request(req, StopReason::Cancelled)),
                None => Ok(false),
            },
        }
    }

    /// Read (`None`) or set the agent's compaction policy (`Some(None)` clears it).
    fn agent_policy(&mut self, id: &str, policy: Option<Option<compaction::Policy>>) -> Result<(), EngineError> {
        let Some(a) = self.agents.get_mut(id) else { return Err(Self::no_agent(id)) };
        match policy {
            None => Ok(()),
            Some(None) => {
                a.comp.policy = None;
                Ok(())
            }
            Some(Some(p)) => match p.validate(a.comp.prefix_tokens) {
                Ok(()) => {
                    a.comp.policy = Some(p);
                    Ok(())
                }
                Err(m) => Err(EngineError::new(422, "invalid_request_error", m)),
            },
        }
    }

    /// Read or change the agent's extracted-fields record; the record.
    fn agent_fields(&mut self, id: &str, op: compaction::FieldsOp) -> Result<Value, EngineError> {
        let Some(a) = self.agents.get_mut(id) else { return Err(Self::no_agent(id)) };
        match a.comp.fields.apply(op) {
            Ok(found) => Ok(json!({"id": id, "object": "agent.fields", "fields": a.comp.fields.to_json(), "found": found})),
            Err(m) => Err(EngineError::new(422, "invalid_request_error", m)),
        }
    }
}
