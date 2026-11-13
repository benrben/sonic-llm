//! The executor's side of the agent lifecycle (phase 3, [`crate::lifecycle`]): the per-agent input queue, the wake
//! queue and its budget, the idle and memory-pressure policy, the token history and the durable registry. Every
//! function here runs on the executor thread, at a step boundary.

use super::*;
use crate::lifecycle::*;

/// A turn waiting in an agent's input queue.
pub(super) struct Queued {
    pub cmd: StartCmd,
    pub arrived: Instant,
}

/// The reply of a suspend that has nothing to do: the agent holds no KV in RAM already, or is on (or coming back from) the
/// SSD when the SSD is asked for.
fn already_suspended(aid: &str, tier: Tier, state: AgentState) -> Option<Value> {
    match (tier, state) {
        (_, AgentState::History) => Some(json!({"id": aid, "state": "history", "suspended": false, "already": true})),
        (Tier::Ssd, AgentState::Suspended | AgentState::Restoring) => {
            Some(json!({"id": aid, "state": state.name(), "suspended": state == AgentState::Suspended, "already": true}))
        }
        _ => None,
    }
}

/// What one queued restore did.
enum Resume {
    /// The engine is reading the agent back.
    Started,
    /// No room, and none could be made: the restores wait for the next boundary.
    NoRoom,
    /// Room was made for a later try, or there was nothing to restore (the agent re-prefills from its history).
    Later,
}

impl Executor {
    pub(super) fn life_mut(&mut self, id: &str) -> Option<&mut AgentLife> {
        self.agents.get_mut(id).map(|a| &mut a.life)
    }

    /// Persist an agent's record (best effort: a failed write is counted, the agent goes on).
    pub(super) fn life_save(&mut self, id: &str) {
        let Some(reg) = &self.registry else { return };
        let Some(a) = self.agents.get(id) else { return };
        let rec = AgentRecord { id: id.to_string(), lane: a.lane, created: a.created, turns: a.turns, history: a.life.history.clone(), prefix_tokens: a.comp.prefix_tokens };
        match reg.save(&rec) {
            Ok(()) => self.life_stats.registry_writes += 1,
            Err(e) => {
                self.life_stats.registry_errors += 1;
                eprintln!("sllm-server: registry write of {id}: {e}");
            }
        }
    }

    /// Load the durable registry: every agent comes back as a History agent (its KV is rebuilt on its next turn).
    pub(super) fn life_load(&mut self) {
        let Some(reg) = self.registry.clone() else { return };
        let (recs, bad) = reg.load();
        for b in bad {
            eprintln!("sllm-server: registry: skipped unreadable {}", b);
        }
        for r in recs {
            if let Err(e) = self.engine.agent_create(&r.id, r.lane, &[]) {
                eprintln!("sllm-server: registry: agent {} not restored: {e}", r.id);
                continue;
            }
            let n = r.id.trim_start_matches("ag_").parse::<u64>().unwrap_or(0);
            self.next_agent = self.next_agent.max(n + 1);
            let mut life = AgentLife::new(r.history);
            if !life.history.is_empty() {
                life.state = AgentState::History;
            }
            let prefix = r.prefix_tokens.min(life.history.len());
            let policy = self.comp_cfg.default_policy.filter(|p| p.validate(prefix).is_ok());
            let comp = compaction::AgentState::new(prefix, policy);
            self.agents.insert(r.id.clone(), AgentRec { lane: r.lane, turns: r.turns, busy: None, created: r.created, comp, life, inbox: VecDeque::new() });
            self.life_stats.loaded_from_registry += 1;
        }
    }

    /// An agent turn arrives: run it now (returns it, rewritten to re-prefill the history for a History agent), or
    /// queue it behind the running turn or the agent's restore (returns `None`).
    pub(super) fn life_admit(&mut self, s: StartCmd, from_queue: bool) -> Option<StartCmd> {
        let Some(aid) = s.agent.clone() else { return Some(s) };
        if crate::compaction::is_compaction(s.id) {
            // the executor's own compaction request: the agent is resident and idle (begin_compaction checked)
            return Some(s);
        }
        let Some(a) = self.agents.get(&aid) else { return Some(s) }; // start() refuses the unknown agent
        // (a turn of the agent waiting for room is ahead of this one)
        let behind = (!from_queue && !a.inbox.is_empty()) || self.turn_waits_for_room(&aid);
        let must_wait = a.busy.is_some() || behind || matches!(a.life.state, AgentState::Suspended | AgentState::Restoring);
        if must_wait {
            self.life_queue_turn(&aid, s, from_queue);
            return None;
        }
        Some(self.life_ready_turn(&aid, s))
    }

    /// A turn that must wait for its agent: into the agent's input queue (refused when that is full), and a suspended
    /// agent starts waking.
    pub(super) fn life_queue_turn(&mut self, aid: &str, s: StartCmd, from_queue: bool) {
        let cap = self.life_cfg.inbox_cap;
        let a = self.agents.get_mut(aid).expect("life_admit found the agent");
        if a.inbox.len() >= cap && !from_queue {
            self.life_stats.inbox_full += 1;
            self.shared.requests_failed.fetch_add(1, Relaxed);
            let e = EngineError::new(409, "conflict_error", format!("agent {aid} has {cap} turns queued (its input queue is full)"));
            let _ = s.events.try_send(ReqEvent::Error(e));
            return;
        }
        if from_queue {
            // back to the head: it keeps its place
            a.inbox.push_front(Queued { cmd: s, arrived: Instant::now() });
            return;
        }
        a.inbox.push_back(Queued { cmd: s, arrived: Instant::now() });
        self.life_stats.turns_queued += 1;
        if a.life.state == AgentState::Suspended {
            self.life_wake(aid);
        }
    }

    /// A turn that runs now: a History agent's turn is rewritten to re-prefill the history first.
    fn life_ready_turn(&mut self, aid: &str, mut s: StartCmd) -> StartCmd {
        let a = self.agents.get_mut(aid).expect("life_admit found the agent");
        if a.life.state == AgentState::History && !a.life.history.is_empty() {
            // the KV is gone: re-prefill the history in front of the turn (the scheduler cuts it into chunks)
            let n = a.life.history.len();
            let mut ids = a.life.history.clone();
            ids.extend_from_slice(&s.ids);
            s.ids = ids;
            a.life.rebuild = n;
            a.life.wake = Some(WakeInfo { via: "reprefill", restore_ms: 0.0, reprefill_tokens: n });
        } else {
            a.life.rebuild = 0;
            if a.life.wake.is_none() {
                a.life.wake = Some(WakeInfo { via: "ram", restore_ms: 0.0, reprefill_tokens: 0 });
            }
        }
        s
    }

    /// The engine admitted the turn: the history takes its ids.
    pub(super) fn life_turn_started(&mut self, aid: &str, ids: &[i32], req: ReqId) {
        let Some(a) = self.agents.get_mut(aid) else { return };
        let l = &mut a.life;
        if crate::compaction::is_compaction(req) {
            // a compaction rewrites the window: the history is read back from the engine when it ends
            l.compacting = true;
            l.rebuild = 0;
            let _ = l.apply(Ev::TurnStarted);
            return;
        }
        let skip = l.rebuild.min(ids.len());
        l.hist_before = l.history.len();
        l.history.extend_from_slice(&ids[skip..]);
        if l.rebuild > 0 {
            self.life_stats.reprefills += 1;
            self.life_stats.reprefill_tokens += l.rebuild as u64;
        }
        if let Err(e) = l.apply(Ev::TurnStarted) {
            eprintln!("sllm-server: {aid}: {e}");
        }
    }


    /// The turn ended. `rolled_back`: it stopped before its first token, so the engine put the context back.
    pub(super) fn life_turn_ended(&mut self, aid: &str, rolled_back: bool) {
        let Some(a) = self.agents.get_mut(aid) else { return };
        let l = &mut a.life;
        if std::mem::take(&mut l.compacting) {
            if l.state == AgentState::Active {
                let _ = l.apply(Ev::TurnEnded { kv_kept: true });
            }
            return;
        }
        if rolled_back {
            l.history.truncate(l.hist_before);
        }
        // a rebuild that never ran leaves the agent empty: still History
        let kv_kept = !(rolled_back && l.rebuild > 0);
        l.rebuild = 0;
        l.wake = None;
        if l.state == AgentState::Active {
            let _ = l.apply(Ev::TurnEnded { kv_kept });
        }
        self.life_save(aid);
    }

    /// A compaction ended (or failed after its cut): the agent's history becomes its window as the engine holds it, so a
    /// snapshot and a re-prefill both reflect the compacted window.
    pub(super) fn life_sync_history(&mut self, aid: &str) {
        let Some(toks) = self.engine.agent_tokens(aid) else { return };
        if let Some(l) = self.life_mut(aid) {
            l.history = toks;
        }
        self.life_save(aid);
    }

    /// A compaction needs the agent's KV in RAM: refused while it is suspended, restoring or history only.
    pub(super) fn life_compaction_guard(&self, aid: &str) -> Option<EngineError> {
        let a = self.agents.get(aid)?;
        match a.life.state {
            AgentState::Suspended | AgentState::Restoring | AgentState::History => Some(EngineError::new(
                409,
                "agent_suspended",
                format!("agent {aid} is {} (its KV is not in RAM): wake it or give it a turn before compacting", a.life.state.name()),
            )),
            _ => None,
        }
    }

    /// Start (or queue) the restore of a suspended agent.
    pub(super) fn life_wake(&mut self, aid: &str) {
        let Some(l) = self.life_mut(aid) else { return };
        if l.state != AgentState::Suspended {
            return;
        }
        if l.wake_t0.is_none() {
            l.wake_t0 = Some(Instant::now());
        }
        if !self.wake_queue.iter().any(|x| x == aid) {
            self.wake_queue.push_back(aid.to_string());
        }
    }

    /// Suspend an idle agent now (to the SSD, or drop its KV to history).
    pub(super) fn life_suspend_now(&mut self, aid: &str, tier: Tier, why: Why) -> Result<Value, EngineError> {
        let Some(a) = self.agents.get(aid) else { return Err(Self::no_agent(aid)) };
        let state = a.life.state;
        if let Some(already) = already_suspended(aid, tier, state) {
            return Ok(already);
        }
        if tier == Tier::History {
            return self.life_drop_now(aid, state);
        }
        let rep = match self.engine.agent_suspend(aid) {
            Ok(r) => r,
            Err(e) => {
                self.life_stats.suspend_errors += 1;
                return Err(e);
            }
        };
        self.room_gen += 1; // its pages are free: requests waiting for room are asked again
        let l = self.life_mut(aid).unwrap();
        l.apply(Ev::Suspended).map_err(|e| EngineError::new(500, "server_error", e))?;
        match why {
            Why::Api => self.life_stats.suspends_api += 1,
            Why::Idle => self.life_stats.suspends_idle += 1,
            Why::Pressure | Why::Budget => self.life_stats.suspends_pressure += 1,
        }
        Ok(json!({"id": aid, "state": "suspended", "suspended": true, "pages_freed": rep.pages_freed, "bytes_freed": rep.bytes_freed, "stage_ms": rep.stage_ms}))
    }

    /// Suspend to history: the agent's KV is dropped (a restore on its way is called off).
    fn life_drop_now(&mut self, aid: &str, state: AgentState) -> Result<Value, EngineError> {
        if state == AgentState::Restoring {
            self.wake_queue.retain(|x| x != aid);
        }
        self.engine.agent_drop_kv(aid)?;
        self.room_gen += 1;
        let l = self.life_mut(aid).unwrap();
        l.apply(Ev::KvDropped).map_err(|e| EngineError::new(500, "server_error", e))?;
        l.wake_t0 = None;
        self.life_stats.history_drops += 1;
        Ok(json!({"id": aid, "state": "history", "suspended": true}))
    }

    /// `POST /v1/agents/{id}/suspend`: now when idle, at the end of the turn when busy.
    pub(super) fn life_suspend_cmd(&mut self, aid: &str, tier: Tier) -> Result<Value, EngineError> {
        let empty = self.engine.agent_context_tokens(aid) == Some(0);
        let Some(a) = self.agents.get_mut(aid) else { return Err(Self::no_agent(aid)) };
        if a.busy.is_some() || !a.inbox.is_empty() {
            a.life.suspend_after = Some(tier);
            return Ok(json!({"id": aid, "state": a.life.state.name(), "suspended": false, "deferred": true}));
        }
        if empty {
            return Ok(json!({"id": aid, "state": a.life.state.name(), "suspended": false})); // it holds nothing to save
        }
        self.life_suspend_now(aid, tier, Why::Api)
    }

    /// `POST /v1/agents/{id}/wake`: start restoring now (a tool result is about to arrive).
    pub(super) fn life_wake_cmd(&mut self, aid: &str) -> Result<Value, EngineError> {
        let Some(a) = self.agents.get_mut(aid) else { return Err(Self::no_agent(aid)) };
        a.life.suspend_after = None;
        let state = a.life.state;
        if state == AgentState::Suspended {
            self.life_wake(aid);
            self.life_start_restores();
        }
        let state = self.agents[aid].life.state;
        Ok(json!({"id": aid, "state": state.name(), "kv": state.tier()}))
    }

    /// Requests waiting in input queues: cancel one by id (true if found).
    pub(super) fn life_cancel_queued(&mut self, id: ReqId) -> bool {
        for a in self.agents.values_mut() {
            if let Some(i) = a.inbox.iter().position(|q| q.cmd.id == id) {
                a.inbox.remove(i);
                self.shared.requests_cancelled.fetch_add(1, Relaxed);
                return true;
            }
        }
        false
    }

    /// An agent is deleted: its queued turns are refused.
    pub(super) fn life_forget(&mut self, aid: &str) {
        self.wake_queue.retain(|x| x != aid);
        if let Some(a) = self.agents.get_mut(aid) {
            for q in a.inbox.drain(..) {
                let _ = q.cmd.events.try_send(ReqEvent::Error(EngineError::new(404, "not_found_error", format!("agent {aid} was deleted"))));
            }
        }
        if let Some(reg) = &self.registry {
            let _ = reg.remove(aid);
        }
    }

    /// Make room in the page pools by suspending idle agents, coldest first, until `need_bytes` are freed (or no idle
    /// agent is left). `except`: the agent the room is for. Returns the bytes freed.
    pub(super) fn life_relieve(&mut self, need_bytes: u64, except: Option<&str>, key: Option<&'static str>, why: Why) -> u64 {
        let mut cands = self.relief_candidates(except, key);
        cands.sort();
        let mut freed = 0u64;
        for (_, id, _) in cands {
            if freed >= need_bytes.max(1) {
                break;
            }
            if let Ok(v) = self.life_suspend_now(&id, Tier::Ssd, why) {
                freed += v["bytes_freed"].as_u64().unwrap_or(0);
            }
        }
        if freed > 0 {
            self.life_stats.pressure_relief += 1;
        }
        freed
    }

    /// Idle agents (not `except`) holding pages (of pool `key`, if given): (last active, id, bytes their pages hold).
    fn relief_candidates(&self, except: Option<&str>, key: Option<&'static str>) -> Vec<(Instant, AgentId, u64)> {
        let mut cands: Vec<(Instant, AgentId, u64)> = Vec::new();
        let pools = self.engine.kv_pools();
        for (id, a) in &self.agents {
            if Some(id.as_str()) == except || a.busy.is_some() || !a.inbox.is_empty() || a.life.state != AgentState::Idle || !a.comp.phase.is_idle() {
                continue;
            }
            let Some((k, pages)) = self.engine.agent_kv(id) else { continue };
            if pages == 0 || key.is_some_and(|w| w != k) {
                continue;
            }
            let pb = pools.iter().find(|p| p.key == k).map_or(0, |p| p.page_bytes);
            cands.push((a.life.last_active, id.clone(), pages as u64 * pb));
        }
        cands
    }

    /// Start queued restores up to the in-flight budget, interactive agents first.
    pub(super) fn life_start_restores(&mut self) {
        let mut restoring = self.agents.values().filter(|a| a.life.state == AgentState::Restoring).count();
        let mut order: Vec<AgentId> = self.wake_queue.iter().cloned().collect();
        order.sort_by_key(|id| self.agents.get(id).map_or(2, |a| if a.lane == Lane::Interactive { 0 } else { 1 }));
        for id in order {
            if restoring >= self.life_cfg.max_restoring {
                break;
            }
            if self.agents.get(&id).map(|a| a.life.state) != Some(AgentState::Suspended) {
                self.wake_queue.retain(|x| *x != id);
                continue;
            }
            match self.life_resume(&id) {
                Resume::Started => restoring += 1,
                Resume::NoRoom => break,
                Resume::Later => {}
            }
        }
    }

    /// Ask the engine to read a suspended agent back.
    fn life_resume(&mut self, id: &str) -> Resume {
        match self.engine.agent_resume(id) {
            Ok(()) => {
                self.wake_queue.retain(|x| *x != id);
                self.life_apply(id, Ev::WakeStarted);
                Resume::Started
            }
            Err(e) if e.kind == "kv_pool_exhausted_error" => {
                // no room yet: make some from idle agents, try again at the next boundary
                let need = e.shortfall_bytes.unwrap_or(1);
                if self.life_relieve(need, Some(id), None, Why::Pressure) == 0 {
                    return Resume::NoRoom;
                }
                Resume::Later
            }
            Err(_) => {
                // nothing to restore (the snapshot was never written): re-prefill from the history
                self.life_to_history(id);
                Resume::Later
            }
        }
    }

    /// The agent's KV is gone (dropped, lost, or a snapshot that was never written or read): whatever the engine still keeps
    /// of it is dropped, the agent keeps its history (its next turn re-prefills it) and its wake is called off.
    pub(super) fn life_to_history(&mut self, id: &str) {
        self.wake_queue.retain(|x| x != id);
        if let Err(e) = self.engine.agent_drop_kv(id) {
            eprintln!("sllm-server: {id}: dropping its KV: {e}");
        }
        self.room_gen += 1;
        self.life_apply(id, Ev::KvDropped);
        if let Some(l) = self.life_mut(id) {
            l.wake_t0 = None;
        }
    }

    /// Move the agent's lifecycle state; a move the state machine refuses is logged (a bug shows, the state stays). False
    /// when refused or the agent is unknown.
    pub(super) fn life_apply(&mut self, id: &str, ev: Ev) -> bool {
        let Some(l) = self.life_mut(id) else { return false };
        match l.apply(ev) {
            Ok(_) => true,
            Err(e) => {
                eprintln!("sllm-server: {id}: {e}");
                false
            }
        }
    }

    pub(super) fn life_publish(&self) {
        let mut per = [0usize; 5];
        for a in self.agents.values() {
            per[AgentState::ALL.iter().position(|s| *s == a.life.state).unwrap()] += 1;
        }
        let per_state: Vec<(AgentState, usize)> = AgentState::ALL.iter().copied().zip(per).collect();
        let v = self.life_stats.json(&per_state, self.engine.lifecycle_counters(), &self.life_cfg);
        *self.shared.lifecycle.lock().unwrap() = v;
    }
}
