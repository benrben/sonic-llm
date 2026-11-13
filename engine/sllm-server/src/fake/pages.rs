//! The page pool model of the fake engine (`pool_pages`): what admitted requests and idle sequences hold and reserve.

use super::*;

/// The page check of the fake engine's pool model (the rule the native engine's ledger used): an admitted request reserves its
/// worst case in pages INSIDE the pool, and a request is refused when its worst case does not fit next to what the admitted
/// requests may still grow into. Pure arithmetic over pages.
///
/// Every sequence with an admitted request counts at least its worst case, so `agents` holds `(held, target)` of each; the
/// completions lane's requests run one at a time and the lane is emptied between them, so its demand is the largest target.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PageReservation {
    /// `(held, target)` of each sequence that has an admitted request; `target` is 0 for a sequence without one.
    pub agents: Vec<(usize, usize)>,
    /// Pages held by sequences without a request.
    pub idle_held: usize,
    /// The largest worst case among admitted completions requests (0 if none).
    pub completion_target: usize,
}

impl PageReservation {
    /// Pages the pool must be able to provide for every admitted request's worst case plus what idle sequences hold.
    pub fn demand(&self) -> usize {
        self.idle_held + self.completion_target + self.agents.iter().map(|&(held, target)| held.max(target)).sum::<usize>()
    }

    /// Pages still to be taken from the pool by admitted requests (their reservation).
    pub fn outstanding(&self, pages_in_use: usize) -> usize {
        self.demand().saturating_sub(pages_in_use)
    }

    /// Would one more request with worst case `target` on a sequence holding `held` fit in `pool_pages`?
    /// Returns the pages missing when it would not.
    pub fn shortfall_with(&self, held: usize, target: usize, is_completion: bool, pool_pages: usize) -> Option<usize> {
        let mut after = self.clone();
        if is_completion {
            after.completion_target = after.completion_target.max(target);
        } else {
            after.agents.push((held, target));
            after.idle_held = after.idle_held.saturating_sub(held);
        }
        let need = after.demand();
        need.checked_sub(pool_pages).filter(|&s| s > 0)
    }
}

impl FakeEngine {
    /// Context tokens a sequence holds now: an agent its context, a completion what it has fed and generated.
    fn held_pages(&self) -> (usize, HashMap<&str, usize>) {
        let mut by_agent = HashMap::new();
        let mut completions = 0;
        for a in &self.active {
            if a.agent.is_none() {
                completions += pages(a.fed + a.emitted);
            }
        }
        for (id, a) in &self.agents {
            by_agent.insert(id.as_str(), if a.suspended { 0 } else { pages(a.ctx) });
        }
        (completions, by_agent)
    }

    pub(super) fn reservation(&self) -> PageReservation {
        let (_, held) = self.held_pages();
        let mut r = PageReservation::default();
        let mut busy = std::collections::HashSet::new();
        for a in &self.active {
            match &a.agent {
                None => r.agents.push((pages(a.fed + a.emitted), a.target_pages)),
                Some(id) => {
                    busy.insert(id.as_str());
                    r.agents.push((held.get(id.as_str()).copied().unwrap_or(0), a.target_pages));
                }
            }
        }
        r.idle_held = held.iter().filter(|(id, _)| !busy.contains(*id)).map(|(_, p)| p).sum();
        r
    }

    pub(super) fn pages_in_use(&self) -> usize {
        let (c, a) = self.held_pages();
        c + a.values().sum::<usize>()
    }
}
