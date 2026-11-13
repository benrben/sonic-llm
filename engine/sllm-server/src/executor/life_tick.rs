//! The agent lifecycle at a step boundary ([`crate::lifecycle`]): the engine's lifecycle events, queued turns that can run,
//! the idle limit, memory pressure and the SSD budget, and how often the idle loop must poll for them.

use super::*;
use crate::lifecycle::*;

impl Executor {
    /// Background results and the engine's own suspends (memory pressure): applied as they arrive, at every command too, so a
    /// turn or a suspend request never meets an agent whose state the engine has already changed. This is the one place where
    /// what the engine did to an agent's KV becomes the agent's residency.
    pub(super) fn life_events(&mut self) {
        for ev in self.engine.lifecycle_poll(self.life_cfg.upload_budget_bytes) {
            match ev {
                LifecycleEvent::Suspended { agent } => {
                    if self.life_apply(&agent, Ev::Suspended) {
                        self.life_stats.suspends_pressure += 1;
                    }
                }
                LifecycleEvent::KvLost { agent } => self.life_to_history(&agent),
                LifecycleEvent::Written { agent, result } => self.life_written(&agent, result),
                LifecycleEvent::Restored { agent, result } => self.life_restored(&agent, result),
            }
        }
    }

    /// A snapshot write ended: counted. A failed one leaves an agent whose KV was only in it (suspended, or being read back
    /// from it) with its history: the engine drops what it kept and the next turn re-prefills. An agent back in RAM already
    /// does not need the snapshot.
    fn life_written(&mut self, agent: &str, result: Result<(u64, u64, f64, bool), String>) {
        match result {
            Ok((bytes, private, _, _)) => self.life_stats.push_write(bytes, private),
            Err(e) => {
                self.life_stats.write_failures += 1;
                eprintln!("sllm-server: snapshot of {agent}: {e}");
                if self.agents.get(agent).is_some_and(|a| matches!(a.life.state, AgentState::Suspended | AgentState::Restoring)) {
                    self.life_to_history(agent);
                }
            }
        }
    }

    /// A restore ended: the agent is back in RAM, or (it failed, or the agent was not waiting for it) re-prefills from its
    /// history.
    fn life_restored(&mut self, agent: &str, result: Result<f64, String>) {
        let Some(a) = self.agents.get_mut(agent) else { return };
        let wake_ms = a.life.wake_t0.take().map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e3);
        if let Err(e) = result {
            eprintln!("sllm-server: restore of {agent}: {e}; re-prefill from the history");
            self.life_stats.restore_failures += 1;
            return self.life_to_history(agent);
        }
        if !self.life_apply(agent, Ev::Restored) {
            return self.life_to_history(agent); // the engine's KV is not what the agent expects: it rebuilds
        }
        let ctx = self.agents[agent].life.history.len();
        self.life_mut(agent).expect("checked above").wake = Some(WakeInfo { via: "ssd", restore_ms: wake_ms, reprefill_tokens: 0 });
        self.life_stats.restores += 1;
        self.life_stats.push_restore(wake_ms);
        self.life_stats.reprefills_avoided += 1;
        self.life_stats.reprefill_tokens_avoided += ctx as u64;
    }

    /// The lifecycle at a step boundary: background results, restores, queued turns, the idle and pressure policy.
    pub(super) fn life_tick(&mut self) {
        let now = Instant::now();
        self.life_events();
        self.life_start_restores();
        self.life_run_ready_turns();
        self.life_wake_stalled();
        self.life_drop_gone_turns();
        self.life_deferred_suspends();
        self.life_idle_limit(now);
        self.life_pressure();
        self.life_ssd_budget();
    }

    /// Queued turns of agents that can run now (restored, or idle again after their turn).
    fn life_run_ready_turns(&mut self) {
        let ready: Vec<AgentId> = self
            .agents
            .iter()
            .filter(|(id, a)| a.busy.is_none() && !a.inbox.is_empty() && !self.held.contains_key(*id) && matches!(a.life.state, AgentState::Idle | AgentState::History))
            .filter(|(id, _)| !self.turn_waits_for_room(id))
            .map(|(id, _)| id.clone())
            .collect();
        for id in ready {
            let Some(q) = self.agents.get_mut(&id).and_then(|a| a.inbox.pop_front()) else { continue };
            let mut cmd = q.cmd;
            if let Some(ms) = cmd.deadline_ms {
                cmd.deadline_ms = Some(ms.saturating_sub(q.arrived.elapsed().as_millis() as u64));
            }
            self.start_inner(cmd, true);
        }
    }

    /// Turns queued behind a suspended agent that is not waking (a wake that could not start): queue its wake.
    fn life_wake_stalled(&mut self) {
        let stalled: Vec<AgentId> = self
            .agents
            .iter()
            .filter(|(id, a)| a.life.state == AgentState::Suspended && !a.inbox.is_empty() && !self.wake_queue.contains(id))
            .map(|(id, _)| id.clone())
            .collect();
        for id in stalled {
            self.life_wake(&id);
        }
    }

    /// Queued turns whose client left or whose deadline passed.
    fn life_drop_gone_turns(&mut self) {
        let mut gone: Vec<(AgentId, usize, bool)> = Vec::new();
        for (id, a) in &self.agents {
            for (i, q) in a.inbox.iter().enumerate() {
                let late = q.cmd.deadline_ms.is_some_and(|ms| q.arrived.elapsed() >= Duration::from_millis(ms));
                if q.cmd.events.is_closed() || late {
                    gone.push((id.clone(), i, late));
                }
            }
        }
        for (id, i, late) in gone.into_iter().rev() {
            let Some(q) = self.agents.get_mut(&id).and_then(|a| a.inbox.remove(i)) else { continue };
            if late {
                self.shared.requests_expired.fetch_add(1, Relaxed);
                let _ = q.cmd.events.try_send(ReqEvent::Error(EngineError::new(504, "deadline_exceeded", "the turn's deadline passed while it waited for its agent")));
            } else {
                self.shared.requests_cancelled.fetch_add(1, Relaxed);
            }
        }
    }

    /// Suspends asked for during a turn.
    fn life_deferred_suspends(&mut self) {
        let deferred: Vec<(AgentId, Tier)> = self
            .agents
            .iter()
            .filter(|(_, a)| a.busy.is_none() && a.inbox.is_empty() && a.life.state == AgentState::Idle && a.comp.phase.is_idle())
            .filter_map(|(id, a)| a.life.suspend_after.map(|t| (id.clone(), t)))
            .collect();
        for (id, tier) in deferred {
            if let Some(l) = self.life_mut(&id) {
                l.suspend_after = None;
            }
            let _ = self.life_suspend_now(&id, tier, Why::Api);
        }
    }

    /// The idle limit: agents idle that long (at `now`, the start of the tick) go to the SSD.
    fn life_idle_limit(&mut self, now: Instant) {
        let Some(ms) = self.life_cfg.idle_suspend_ms else { return };
        let limit = Duration::from_millis(ms);
        let idle: Vec<AgentId> = self
            .agents
            .iter()
            .filter(|(_, a)| a.busy.is_none() && a.inbox.is_empty() && a.life.state == AgentState::Idle && a.comp.phase.is_idle() && a.life.idle_for(now) >= limit)
            .filter(|(id, _)| self.engine.agent_kv(id).is_some_and(|k| k.1 > 0))
            .map(|(id, _)| id.clone())
            .collect();
        for id in idle {
            let _ = self.life_suspend_now(&id, Tier::Ssd, Why::Idle);
        }
    }

    /// Memory pressure: free share under the low mark suspends the coldest idle agents up to the high mark.
    fn life_pressure(&mut self) {
        if self.life_cfg.pressure_low > 0.0 {
            for p in self.engine.kv_pools() {
                if p.total_pages == 0 || (p.free_pages as f64) >= self.life_cfg.pressure_low * p.total_pages as f64 {
                    continue;
                }
                let want = (self.life_cfg.pressure_high * p.total_pages as f64).ceil() as usize;
                let need = want.saturating_sub(p.free_pages) as u64 * p.page_bytes;
                self.life_relieve(need, None, Some(p.key), Why::Pressure);
            }
        }
    }

    /// The SSD budget: the longest-suspended agents drop to history.
    fn life_ssd_budget(&mut self) {
        let Some(budget) = self.life_cfg.ssd_budget_bytes else { return };
        let mut ssd = self.engine.lifecycle_counters()["ssd_bytes"].as_u64().unwrap_or(0);
        if ssd <= budget {
            return;
        }
        let mut old: Vec<(Instant, AgentId)> = self
            .agents
            .iter()
            .filter(|(_, a)| a.life.state == AgentState::Suspended && a.inbox.is_empty())
            .filter_map(|(id, a)| a.life.suspended_at.map(|t| (t, id.clone())))
            .collect();
        old.sort();
        for (_, id) in old {
            if ssd <= budget {
                break;
            }
            self.life_to_history(&id);
            self.life_stats.history_drops += 1;
            ssd = self.engine.lifecycle_counters()["ssd_bytes"].as_u64().unwrap_or(0);
        }
    }

    /// Whether the loop must keep polling while no request is admitted, and how long it may sleep (the engine's own idle
    /// work counts too: at most 50 ms a sleep, so a command never waits long).
    pub(super) fn life_poll_every(&self) -> Option<Duration> {
        let eng = self.engine.idle_wake().map(|d| d.clamp(Duration::from_millis(1), Duration::from_millis(50)));
        match (self.life_poll_every_agents(), eng) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    fn life_poll_every_agents(&self) -> Option<Duration> {
        // an asynchronous summary is only noticed by polling it (compaction_tick): never block for a command while one waits
        if !self.summarizing.is_empty() {
            return Some(Duration::from_millis(1));
        }
        if self.engine.lifecycle_busy() || !self.wake_queue.is_empty() || self.agents.values().any(|a| !a.inbox.is_empty()) {
            return Some(Duration::from_millis(1));
        }
        if self.agents.values().any(|a| a.life.state == AgentState::Idle && a.life.suspend_after.is_some()) {
            return Some(Duration::from_millis(1));
        }
        let ms = self.life_cfg.idle_suspend_ms?;
        let now = Instant::now();
        let limit = Duration::from_millis(ms);
        let next = self
            .agents
            .values()
            .filter(|a| a.life.state == AgentState::Idle && a.busy.is_none())
            .map(|a| limit.saturating_sub(a.life.idle_for(now)))
            .min()?;
        Some(next.clamp(Duration::from_millis(1), Duration::from_millis(50)))
    }
}
