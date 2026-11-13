//! Agents: lookups and the context an agent holds.

use super::*;

impl NextEngine {
    pub(super) fn agent(&self, id: &str) -> Result<&Agent, EngineError> {
        self.agents.get(id).ok_or_else(|| err(404, "not_found_error", format!("no agent {id}")))
    }

    /// The agent, at rest: refused with a 409 (the request being `doing`) while it has a request, and with `off`'s error while
    /// it is suspended.
    pub(super) fn agent_at_rest(&self, id: &str, doing: &str, off: impl FnOnce() -> EngineError) -> Result<&Agent, EngineError> {
        if self.agent_busy(id) {
            return Err(err(409, "conflict_error", format!("agent {id} has a request {doing}")));
        }
        let a = self.agent(id)?;
        if a.off.is_some() {
            return Err(off());
        }
        Ok(a)
    }

    pub(super) fn agent_busy(&self, id: &str) -> bool {
        self.jobs.values().any(|j| j.agent.as_deref() == Some(id))
    }

    /// The agent's context: what its slot holds, or its suspended tokens.
    pub(super) fn context_of(&self, a: &Agent) -> Vec<i32> {
        let held = match (&a.off, a.slot) {
            (Some(o), _) => o.tokens.clone(),
            (None, Some(s)) => self.cells[s as usize].clone(),
            (None, None) => Vec::new(),
        };
        held.into_iter().chain(a.pending.iter().copied()).collect()
    }
}
