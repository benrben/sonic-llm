//! KV-cache memory policy of the generic engine, kept free of libllama so it can be tested alone.
//!
//! A request is admitted on the cells it needs now (its prompt plus a small decode headroom), not on its worst case; its
//! sequence grows one cell per token. When the cache is about to be full the engine frees cells in this order
//! ([`plan_free`]):
//!
//! 1. suspend the least-recently-used idle agent (its state goes to the SSD, the cells are free now);
//! 2. pause a running request, the lowest priority first (the youngest of equal priority): its cells are dropped, its
//!    tokens are kept, and it re-prefills them when the cache has room again.
//!
//! The plan is all or nothing: when what could be freed does not cover the shortfall, nothing is touched and the caller
//! refuses with the 503 `kv_pool_exhausted_error`.

use crate::engine::Lane;
use std::time::Instant;

/// Decode cells promised to a request beyond its prompt when it is admitted.
pub const HEADROOM: usize = 64;
/// Cells kept unpromised in a step, so a decode never meets a full cache.
pub const MARGIN: usize = 32;

/// Scheduling priority of a lane for memory decisions: a request may only pause requests of a strictly lower rank at
/// admission.
pub fn rank(lane: Lane) -> u8 {
    match lane {
        Lane::Background => 0,
        Lane::Interactive => 1,
        Lane::Helper => 2,
    }
}

#[derive(Debug, Clone)]
pub struct IdleAgent {
    pub id: String,
    pub last_used: Instant,
    pub cells: usize,
}

#[derive(Debug, Clone)]
pub struct Pausable {
    pub req: u64,
    pub rank: u8,
    /// Admission order (larger = younger).
    pub order: u64,
    /// Cells pausing it frees (what it holds and has promised).
    pub cells: usize,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// Idle agents to suspend, in order.
    pub suspend: Vec<String>,
    /// Requests to pause, in order.
    pub pause: Vec<u64>,
}

/// Take the candidates in order, each worth `cells`, until `got` reaches `short`; `take` records each one taken.
fn take_until<T>(candidates: &[&T], short: usize, got: &mut usize, cells: impl Fn(&T) -> usize, mut take: impl FnMut(&T)) {
    for &c in candidates {
        if *got >= short {
            break;
        }
        *got += cells(c);
        take(c);
    }
}

/// Choose what to free to cover `short` cells; `None` when everything allowed together falls short. Only requests with
/// `rank < below` are candidates for pausing (`below` 0: none).
pub fn plan_free(short: usize, idle: &[IdleAgent], pausable: &[Pausable], below: u8) -> Option<Plan> {
    let mut plan = Plan::default();
    if short == 0 {
        return Some(plan);
    }
    let mut got = 0usize;
    let mut idle: Vec<&IdleAgent> = idle.iter().filter(|a| a.cells > 0).collect();
    idle.sort_by(|a, b| a.last_used.cmp(&b.last_used).then(b.cells.cmp(&a.cells)).then(a.id.cmp(&b.id)));
    take_until(&idle, short, &mut got, |a| a.cells, |a| plan.suspend.push(a.id.clone()));
    if got < short {
        let mut p: Vec<&Pausable> = pausable.iter().filter(|p| p.rank < below && p.cells > 0).collect();
        p.sort_by(|a, b| a.rank.cmp(&b.rank).then(b.order.cmp(&a.order)).then(a.req.cmp(&b.req)));
        take_until(&p, short, &mut got, |v| v.cells, |v| plan.pause.push(v.req));
    }
    (got >= short).then_some(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn idle(id: &str, age_ms: u64, cells: usize, base: Instant) -> IdleAgent {
        IdleAgent { id: id.into(), last_used: base - Duration::from_millis(age_ms), cells }
    }

    #[test]
    fn suspends_the_least_recently_used_idle_agents_first() {
        let now = Instant::now();
        let a = [idle("new", 10, 500, now), idle("old", 900, 300, now), idle("mid", 400, 300, now)];
        let p = plan_free(500, &a, &[], 0).unwrap();
        assert_eq!(p.suspend, vec!["old", "mid"]);
        assert!(p.pause.is_empty());
        assert_eq!(plan_free(0, &a, &[], 0).unwrap(), Plan::default());
    }

    #[test]
    fn pauses_the_lowest_rank_youngest_when_no_idle_agent_is_left() {
        let now = Instant::now();
        let a = [idle("a", 100, 100, now)];
        let v = [
            Pausable { req: 1, rank: 1, order: 1, cells: 400 },
            Pausable { req: 2, rank: 0, order: 2, cells: 400 },
            Pausable { req: 3, rank: 0, order: 3, cells: 400 },
        ];
        let p = plan_free(450, &a, &v, 2).unwrap();
        assert_eq!(p.suspend, vec!["a"]);
        assert_eq!(p.pause, vec![3]); // background first, the younger one
        let p = plan_free(1000, &a, &v, 2).unwrap();
        assert_eq!(p.pause, vec![3, 2, 1]);
    }

    #[test]
    fn a_request_pauses_only_lower_ranks_and_the_plan_is_all_or_nothing() {
        let now = Instant::now();
        let v = [Pausable { req: 1, rank: 0, order: 1, cells: 400 }, Pausable { req: 2, rank: 1, order: 2, cells: 400 }];
        assert_eq!(plan_free(300, &[], &v, 0), None); // background may not pause anything
        assert_eq!(plan_free(300, &[], &v, 1).unwrap().pause, vec![1]); // interactive pauses background only
        assert_eq!(plan_free(500, &[], &v, 1), None); // not enough below its rank: nothing is touched
        let a = [idle("x", 1, 50, now)];
        assert_eq!(plan_free(60, &a, &[], 2), None);
    }
}
