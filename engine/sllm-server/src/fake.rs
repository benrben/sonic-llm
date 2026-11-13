//! `FakeEngine`: deterministic token ids and configurable latency, for the HTTP layer's tests and for the playground's
//! third column before the real forward pass exists (task 5b).
//!
//! Tokens are a pure function of the seed and every token the request has seen: the state starts at the seed, folds in the
//! prompt ids, and folds in each generated token, so the same prompt always gives the same ids and an agent turn gives
//! exactly what a plain `/completion` over the concatenated context would. The end-of-generation token is never emitted as
//! a token. Where the fake model "wants to stop" (a deterministic length of 8..64 tokens after the prompt, or
//! `natural_len`), a request with `ignore_eos` false finishes with [`StopReason::Eos`]; with `ignore_eos` true it
//! continues to `n_predict`. Latency: each [`Engine::step`] sleeps `prefill_ms` once per request whose prefill starts in
//! it plus `prefill_ms_per_token` per prefilled token, then `token_ms` once for the whole batch.
//!
//! Phase 2: the engine follows the scheduler's work items (a prefill chunk or one decode token per request); the first
//! token comes with the chunk that ends the prompt. A turn changes the agent's state only once its prefill is done, so
//! a turn cancelled before its first token leaves the agent as it was. With `pool_pages` the engine also models a page
//! pool (256-token pages, the real engine's worst-case reservation per admitted request), so admission can be tested
//! without a GPU. `trace` records every step's items.
//!
//! With `prefix_page` the engine also models prefix sharing for `/completion` requests (the real engine's rules, with a
//! page of `prefix_page` tokens): a request given a slot shares the longest page-aligned prefix of its prompt that a
//! running completion has already fed (it prefills only the rest), or waits on the pages of one that is still
//! prefilling them (coalesced prefill), and attaches once they are fed. At least one prompt token is always left to
//! prefill. A waiter whose writer leaves takes whatever is there then. The page pool model does not count shared pages.

use crate::engine::*;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Every step's work items, in order (tests).
pub type StepTrace = Arc<Mutex<Vec<Vec<WorkItem>>>>;

const FAKE_PAGE: usize = crate::engine::PAGE_CELLS;

mod config;
mod hooks;
mod pages;
pub use config::FakeConfig;
pub use hooks::FakeHooks;
pub use pages::PageReservation;

/// splitmix64-style mixing of one value into a state.
pub fn mix(h: u64, x: u64) -> u64 {
    let mut z = h.wrapping_add(x).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fold(mut h: u64, ids: &[i32]) -> u64 {
    for &i in ids {
        h = mix(h, i as u32 as u64);
    }
    h
}

struct Agent {
    #[allow(dead_code)]
    lane: Lane,
    state: u64,
    ctx: usize,
    /// Suspended (phase 3): its pages are not held; `restoring` is when its restore was asked for.
    suspended: bool,
    restoring: Option<std::time::Instant>,
    /// A snapshot write to report at the next poll.
    written: bool,
    /// The context's ids (compaction): the state is `fold(seed, tokens)`.
    tokens: Vec<i32>,
}

struct Active {
    id: ReqId,
    agent: Option<AgentId>,
    params: SamplingParams,
    /// The state and context once the prompt is folded in (applied to the agent when the prefill completes).
    state: u64,
    ctx: usize,
    emitted: usize,
    budget: usize,
    ctx_limited: bool,
    natural_len: usize,
    feed_n: usize,
    fed: usize,
    /// Worst-case pages of the sequence after this request (the pool model).
    target_pages: usize,
    /// The prompt (prefix sharing model).
    ids: Vec<i32>,
    begun: bool,
    /// Waiting on request `.0` to feed the first `.1` prompt tokens.
    waiting_on: Option<(ReqId, usize)>,
    /// Prompt tokens served from shared pages.
    shared: usize,
}

pub struct FakeEngine {
    cfg: FakeConfig,
    agents: HashMap<AgentId, Agent>,
    active: Vec<Active>,
    counters: EngineCounters,
    prefix: PrefixCounters,
}

fn pages(tokens: usize) -> usize {
    tokens.div_ceil(FAKE_PAGE)
}

/// Whether a shared prefix of `k` tokens (`ready`: fed already) is better than the best one found so far: longer, or as
/// long and ready where that one is not.
fn beats(k: usize, ready: bool, best: Option<(ReqId, usize, bool)>) -> bool {
    k > 0 && best.is_none_or(|b| k > b.1 || (k == b.1 && ready && !b.2))
}

impl Active {
    /// A begun, non-waiting completion: its prompt's pages may be shared.
    fn may_lend_prefix(&self) -> bool {
        self.agent.is_none() && self.begun && self.waiting_on.is_none()
    }

    /// Why the request is finished now, if it is: its budget, or where the fake model wants to stop.
    fn stop_reason(&self) -> Option<StopReason> {
        if self.emitted >= self.budget {
            return Some(if self.ctx_limited { StopReason::Context } else { StopReason::Limit });
        }
        if !self.params.ignore_eos && self.emitted >= self.natural_len {
            return Some(StopReason::Eos);
        }
        None
    }
}

impl FakeEngine {
    pub fn new(cfg: FakeConfig) -> Self {
        FakeEngine { cfg, agents: HashMap::new(), active: Vec::new(), counters: EngineCounters::default(), prefix: PrefixCounters::default() }
    }

    /// The prefix the completion `self.active[i]` can share now: `(holder, tokens, ready)`, the longest page-aligned
    /// common prefix with a begun, non-waiting completion (at least one prompt token left to prefill).
    fn resolve_prefix(&self, i: usize) -> Option<(ReqId, usize, bool)> {
        let pg = self.cfg.prefix_page?;
        let a = &self.active[i];
        if a.agent.is_some() || a.ids.len() < 2 {
            return None;
        }
        let mut best: Option<(ReqId, usize, bool)> = None;
        for (j, h) in self.active.iter().enumerate() {
            if j == i || !h.may_lend_prefix() {
                continue;
            }
            let common = a.ids.iter().zip(&h.ids).take_while(|(x, y)| x == y).count().min(a.ids.len() - 1);
            let k = common / pg * pg;
            if beats(k, h.fed >= k, best) {
                best = Some((h.id, k, h.fed >= k));
            }
        }
        best
    }

    /// Attach a ready prefix, or wait for one being fed; `Fresh` otherwise.
    fn plan_prefix(&mut self, i: usize) {
        match self.resolve_prefix(i) {
            Some((_, k, true)) => self.attach(i, k),
            Some((h, k, false)) => {
                if self.active[i].waiting_on.is_none() {
                    self.prefix.waited_agents += 1;
                }
                self.active[i].waiting_on = Some((h, k));
            }
            None => self.active[i].waiting_on = None,
        }
    }

    /// `self.active[i]` takes the `k` prompt tokens a running completion has fed.
    fn attach(&mut self, i: usize, k: usize) {
        let a = &mut self.active[i];
        (a.fed, a.shared, a.waiting_on) = (k, k, None);
        self.prefix.attached_agents += 1;
        self.prefix.attached_tokens += k;
    }

    /// An agent without a request (`doing`: what the 409 says it has), for a call that changes its cells.
    fn idle_agent(&mut self, id: &str, doing: &str) -> Result<&mut Agent, EngineError> {
        if self.active.iter().any(|a| a.agent.as_deref() == Some(id)) {
            return Err(EngineError::new(409, "conflict_error", format!("agent {id} has a request {doing}")));
        }
        self.agents.get_mut(id).ok_or_else(|| EngineError::new(404, "not_found_error", format!("no agent {id}")))
    }

    /// The token the fake model emits next from `state` (never the EOS id).
    fn next_token(&self, state: u64) -> i32 {
        let mut t = (mix(state, 0x70C3) % self.cfg.n_vocab as u64) as i32;
        if t == self.cfg.eos_id {
            t = (t + 1) % self.cfg.n_vocab as i32;
        }
        t
    }

    fn sleep_ms(ms: f64) {
        if ms > 0.0 {
            std::thread::sleep(Duration::from_secs_f64(ms / 1000.0));
        }
    }
}

impl Engine for FakeEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "fake".into(),
            model: self.cfg.model.clone(),
            n_vocab: self.cfg.n_vocab,
            n_ctx: self.cfg.n_ctx,
            eos_id: self.cfg.eos_id,
        }
    }

    fn caps(&self) -> Value {
        json!({
            "engine": "fake",
            "prefix_sharing": {"mode": if self.cfg.prefix_page.is_some() { "on (modelled)" } else { "off" }},
            "capabilities": self.cfg.capabilities,
            "fake": {
                "seed": self.cfg.seed, "prefill_ms": self.cfg.prefill_ms,
                "prefill_ms_per_token": self.cfg.prefill_ms_per_token, "token_ms": self.cfg.token_ms,
                "natural_len": self.cfg.natural_len,
                "prefix_page": self.cfg.prefix_page,
                "note": "tokens are a hash of the seed and the context, not model output; capabilities is null unless --probe-caps",
            },
        })
    }

    fn start_request(&mut self, req: StartRequest) -> Result<Started, EngineError> {
        self.check_request(&req)?;
        self.cfg.hooks.refuse_start()?;
        let (mut state, mut ctx) = (self.cfg.seed, 0usize);
        if let Some(aid) = &req.agent {
            let a = self.agents.get(aid).ok_or_else(|| EngineError::new(404, "not_found_error", format!("no agent {aid}")))?;
            if a.suspended {
                return Err(EngineError::new(409, "agent_suspended", format!("agent {aid} is suspended: resume it first")));
            }
            (state, ctx) = (a.state, a.ctx);
        }
        state = fold(state, &req.ids);
        ctx += req.ids.len();
        if ctx >= self.cfg.n_ctx {
            return Err(EngineError::new(
                400,
                "exceed_context_size_error",
                format!("the request exceeds the available context size ({ctx} >= n_ctx {})", self.cfg.n_ctx),
            )
            .with_context(ctx, self.cfg.n_ctx));
        }
        let room = self.cfg.n_ctx - ctx;
        let (budget, ctx_limited) = if req.params.n_predict < 0 || req.params.n_predict as usize > room {
            (room, true)
        } else {
            (req.params.n_predict as usize, false)
        };
        let natural_len = self.cfg.natural_len.or(self.cfg.script.as_ref().map(Vec::len)).unwrap_or_else(|| 8 + (mix(state, 0xE05) % 56) as usize);
        let prompt_n = req.ids.len();
        let target_pages = pages(ctx + budget);
        self.check_pool(req.agent.as_deref(), target_pages)?;
        self.active.push(Active {
            id: req.id, agent: req.agent, params: req.params, state, ctx, emitted: 0, budget, ctx_limited, natural_len,
            feed_n: prompt_n, fed: 0, target_pages, ids: req.ids, begun: false, waiting_on: None, shared: 0,
        });
        Ok(Started { prompt_n, feed_n: prompt_n })
    }

    fn try_begin(&mut self, req: ReqId) -> bool {
        let Some(i) = self.active.iter().position(|a| a.id == req) else { return false };
        if self.cfg.hooks.holds(self.active[i].agent.as_deref()) {
            return false;
        }
        if !self.active[i].begun {
            self.active[i].begun = true;
            self.plan_prefix(i);
        }
        true
    }

    fn prefix_state(&mut self, req: ReqId) -> Result<Option<PrefixState>, EngineError> {
        if self.cfg.prefix_page.is_none() {
            return Ok(None);
        }
        let Some(i) = self.active.iter().position(|a| a.id == req) else { return Ok(None) };
        if let Some((h, k)) = self.active[i].waiting_on {
            match self.active.iter().find(|x| x.id == h) {
                Some(w) if w.fed < k => return Ok(Some(PrefixState::Waiting { tokens: k, written: w.fed })),
                Some(_) => self.attach(i, k),
                None => {
                    // the writer left: whatever is there now
                    self.active[i].waiting_on = None;
                    self.plan_prefix(i);
                    if let Some((h, k)) = self.active[i].waiting_on {
                        let written = self.active.iter().find(|x| x.id == h).map_or(0, |w| w.fed);
                        return Ok(Some(PrefixState::Waiting { tokens: k, written }));
                    }
                }
            }
        }
        let a = &self.active[i];
        Ok(Some(PrefixState::Ready { feed_n: a.feed_n - a.shared, shared_n: a.shared }))
    }

    fn cancel(&mut self, req: ReqId) {
        self.active.retain(|a| a.id != req);
    }

    fn step(&mut self, items: &[WorkItem]) -> Vec<EngineEvent> {
        if let Some(t) = &self.cfg.trace {
            t.lock().unwrap().push(items.to_vec());
        }
        if items.is_empty() {
            return Vec::new();
        }
        let prefill = self.feed_prefill(items);
        Self::sleep_ms(prefill);
        Self::sleep_ms(self.cfg.token_ms);
        self.counters.steps += 1;
        let mut events = Vec::new();
        let mut done = Vec::new();
        for w in items {
            let Some(i) = self.active.iter().position(|a| a.id == w.req) else { continue };
            let a = &self.active[i];
            if a.fed < a.feed_n {
                continue; // an intermediate prefill chunk
            }
            if a.emitted == 0 {
                self.prompt_in(i);
            }
            if let Some(r) = self.step_one(i, &mut events) {
                events.push(EngineEvent::Finished { req: w.req, reason: r });
                done.push(w.req);
            }
        }
        self.active.retain(|a| !done.contains(&a.id));
        if self.cfg.pool_pages.is_some() {
            let used = self.pages_in_use() as u64;
            self.counters.pages_peak = Some(self.counters.pages_peak.unwrap_or(0).max(used));
        }
        events
    }

    fn agent_create(&mut self, id: &str, lane: Lane, system_ids: &[i32]) -> Result<(), EngineError> {
        if system_ids.iter().any(|&i| i < 0 || i as usize >= self.cfg.n_vocab) {
            return Err(EngineError::new(422, "invalid_request_error", format!("token id outside 0..{}", self.cfg.n_vocab)));
        }
        if system_ids.len() >= self.cfg.n_ctx {
            return Err(EngineError::new(400, "exceed_context_size_error", "the system prompt does not fit the context"));
        }
        let state = fold(self.cfg.seed, system_ids);
        self.counters.prompt_tokens_evaluated += system_ids.len() as u64;
        self.agents.insert(id.to_string(), Agent { lane, state, ctx: system_ids.len(), tokens: system_ids.to_vec(), suspended: false, restoring: None, written: false });
        Ok(())
    }

    fn agent_free(&mut self, id: &str) {
        self.active.retain(|a| a.agent.as_deref() != Some(id));
        self.agents.remove(id);
    }

    fn agent_context_tokens(&self, id: &str) -> Option<usize> {
        self.agents.get(id).map(|a| a.ctx)
    }

    fn agent_suspend(&mut self, id: &str) -> Result<SuspendReport, EngineError> {
        let a = self.idle_agent(id, "queued or running")?;
        if a.suspended {
            return Err(EngineError::new(409, "conflict_error", format!("agent {id} is already suspended")));
        }
        a.suspended = true;
        a.written = true;
        Ok(SuspendReport { pages_freed: pages(a.ctx), bytes_freed: pages(a.ctx) as u64 * FAKE_PAGE as u64, stage_ms: 0.0 })
    }

    fn agent_resume(&mut self, id: &str) -> Result<(), EngineError> {
        let a = self.agents.get_mut(id).ok_or_else(|| EngineError::new(404, "not_found_error", format!("no agent {id}")))?;
        if !a.suspended {
            return Err(EngineError::new(409, "conflict_error", format!("agent {id} is not suspended")));
        }
        a.restoring.get_or_insert_with(std::time::Instant::now);
        Ok(())
    }

    fn agent_drop_kv(&mut self, id: &str) -> Result<(), EngineError> {
        let seed = self.cfg.seed;
        let a = self.agents.get_mut(id).ok_or_else(|| EngineError::new(404, "not_found_error", format!("no agent {id}")))?;
        (a.state, a.ctx, a.suspended, a.restoring) = (seed, 0, false, None);
        a.tokens.clear();
        Ok(())
    }

    fn lifecycle_poll(&mut self, _upload_budget: u64) -> Vec<LifecycleEvent> {
        self.poll_lifecycle()
    }

    fn lifecycle_busy(&self) -> bool {
        self.agents.values().any(|a| a.restoring.is_some() || a.written)
    }

    fn kv_pools(&self) -> Vec<KvPoolRoom> {
        match self.cfg.pool_pages {
            Some(pool) => {
                let demand = self.reservation().demand();
                vec![KvPoolRoom { key: "fake", total_pages: pool, free_pages: pool.saturating_sub(demand), page_bytes: FAKE_PAGE as u64 }]
            }
            None => Vec::new(),
        }
    }

    fn agent_kv(&self, id: &str) -> Option<(&'static str, usize)> {
        let a = self.agents.get(id)?;
        (!a.suspended).then_some(("fake", pages(a.ctx)))
    }

    fn lifecycle_counters(&self) -> Value {
        let ssd: usize = self.agents.values().filter(|a| a.suspended).map(|a| pages(a.ctx) * FAKE_PAGE).sum();
        json!({"ssd_bytes": ssd})
    }

    fn agent_tokens(&self, id: &str) -> Option<Vec<i32>> {
        self.agents.get(id).map(|a| a.tokens.clone())
    }

    fn agent_cut(&mut self, id: &str, keep: usize) -> Result<CutStats, EngineError> {
        let seed = self.cfg.seed;
        let pooled = self.cfg.pool_pages.is_some();
        let a = self.idle_agent(id, "admitted")?;
        if keep > a.tokens.len() {
            return Err(EngineError::new(422, "invalid_request_error", format!("the cut keeps {keep} of {} tokens", a.tokens.len())));
        }
        let before = pages(a.ctx);
        a.tokens.truncate(keep);
        a.state = fold(seed, &a.tokens);
        a.ctx = keep;
        let after = pages(keep);
        let p = |n: usize| pooled.then_some(n);
        Ok(CutStats { pages_before: p(before), pages_after: p(after), pages_freed: p(before - after), prefix_pages: p(after), prefix_shared: p(0) })
    }

    fn counters(&self) -> EngineCounters {
        let mut c = self.counters.clone();
        if let Some(pool) = self.cfg.pool_pages {
            let used = self.pages_in_use();
            c.pages_reserved = Some(pool as u64);
            c.pages_active = Some(used as u64);
            c.pages_peak = Some(c.pages_peak.unwrap_or(0).max(used as u64));
            // the worst case admitted requests may still take: the fake's `kv_pages_request_reserved`
            c.kv.push(KvUse { id: "reserved".into(), cells: 0, pages: Some(self.reservation().outstanding(used)), page_type: None });
        }
        if self.cfg.prefix_page.is_some() {
            c.prefix = Some(PrefixCounters {
                sharing: true,
                waiting: self.active.iter().filter(|a| a.waiting_on.is_some()).count(),
                ..self.prefix.clone()
            });
        }
        c
    }
}

impl FakeEngine {
    /// One token (or the end) for `self.active[i]`. Returns the stop reason when the request is finished.
    fn step_one(&mut self, i: usize, events: &mut Vec<EngineEvent>) -> Option<StopReason> {
        let (state, emitted, id) = {
            let a = &self.active[i];
            (a.state, a.emitted, a.id)
        };
        if let Some(r) = self.active[i].stop_reason() {
            return Some(r);
        }
        let tok = self.next_token(state);
        let a = &mut self.active[i];
        a.state = mix(state, tok as u32 as u64);
        a.ctx += 1;
        a.emitted += 1;
        if let Some(aid) = &a.agent {
            if let Some(ag) = self.agents.get_mut(aid) {
                ag.state = a.state;
                ag.ctx = a.ctx;
                ag.tokens.push(tok);
            }
        }
        self.counters.tokens_generated += 1;
        let text = self.cfg.script.as_ref().and_then(|s| s.get(emitted)).cloned().unwrap_or_else(|| format!(" t{tok}"));
        let probs = self.token_probs(i, tok, &text);
        events.push(EngineEvent::Token { req: id, id: tok, text, probs });
        self.active[i].stop_reason()
    }

    /// The token probabilities of `self.active[i]`'s token `tok` (`text`), when it asked for them and the engine serves them.
    fn token_probs(&self, i: usize, tok: i32, text: &str) -> Option<Box<TokProbs>> {
        match self.cfg.token_probs {
            Some((best, second)) if self.active[i].params.n_probs > 0 => {
                let one = |id: i32, token: String, p: f64| TokProb { id, bytes: token.as_bytes().to_vec(), token, logprob: p.ln() as f32 };
                let top = vec![one(tok, text.to_string(), best), one(tok + 1, "?".into(), second)];
                Some(Box::new(TokProbs { chosen: one(tok, text.to_string(), best), top }))
            }
            _ => None,
        }
    }

    /// The work items' prefill chunks are fed; returns how long they take (ms).
    fn feed_prefill(&mut self, items: &[WorkItem]) -> f64 {
        let mut prefill = 0.0;
        for w in items {
            if let (Work::Prefill { n }, Some(a)) = (w.work, self.active.iter_mut().find(|a| a.id == w.req)) {
                if a.fed == 0 {
                    prefill += self.cfg.prefill_ms;
                }
                prefill += self.cfg.prefill_ms_per_token * n as f64;
                a.fed = (a.fed + n).min(a.feed_n);
                self.counters.prompt_tokens_evaluated += n as u64;
            }
        }
        prefill
    }

    /// The prompt of `self.active[i]` is in: the turn now counts in the agent's context.
    fn prompt_in(&mut self, i: usize) {
        let a = &self.active[i];
        if let Some(aid) = a.agent.clone() {
            let (st, cx) = (a.state, a.ctx);
            if let Some(ag) = self.agents.get_mut(&aid) {
                (ag.state, ag.ctx) = (st, cx);
                ag.tokens.extend_from_slice(&a.ids);
            }
        }
    }

    /// What `start_request` refuses before anything else: latent steps, token ids outside the vocabulary, a grammar
    /// without a `root` rule.
    fn check_request(&self, req: &StartRequest) -> Result<(), EngineError> {
        if req.params.latent_steps > 0 {
            return Err(EngineError::new(501, "not_supported_error", "latent_steps needs --engine next"));
        }
        if req.ids.iter().any(|&i| i < 0 || i as usize >= self.cfg.n_vocab) {
            return Err(EngineError::new(422, "invalid_request_error", format!("token id outside 0..{}", self.cfg.n_vocab)));
        }
        // the real engine builds the grammar here; the fake only knows a grammar needs a `root` rule (the output is not constrained)
        if req.params.grammar.as_deref().is_some_and(|g| !g.lines().any(|l| l.starts_with("root ::="))) {
            return Err(EngineError::new(400, "invalid_request_error", "invalid grammar: no `root ::=` rule"));
        }
        Ok(())
    }

    /// The pool model's admission: a request of worst case `target_pages` (for `agent`, or a completion) is refused when
    /// it does not fit next to what admitted requests may still take.
    fn check_pool(&self, agent: Option<&str>, target_pages: usize) -> Result<(), EngineError> {
        let Some(pool) = self.cfg.pool_pages else { return Ok(()) };
        let held = agent.and_then(|a| self.agents.get(a)).map_or(0, |a| pages(a.ctx));
        if let Some(short) = self.reservation().shortfall_with(held, target_pages, false, pool) {
            return Err(EngineError::new(
                503,
                "kv_pool_exhausted_error",
                format!("the fake page pool ({pool} pages) cannot hold this request next to what admitted requests may still take; short by {short} pages"),
            ));
        }
        Ok(())
    }
}
