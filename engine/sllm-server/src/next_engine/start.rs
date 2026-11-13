//! A request admitted: its slot, its budget and the pages it may reach, a waiting request's copied prefix.

use super::*;

/// Where a turn starts: the agent's slot (None: a new sequence takes one), its lane, the cells it holds, the tokens it
/// reads (the agent's unread ones; the request's are added next), and the agent's context before the turn.
struct Turn {
    slot: Option<i32>,
    interactive: bool,
    base: usize,
    feed: Vec<i32>,
    before: Option<(usize, Vec<i32>)>,
}

/// A turn admitted: its slot and the prefix of `feed` the slot holds already (`shared`), its budget, the cells it may reach
/// (`end`, reserved), and the rest of its [`Turn`].
struct Admitted {
    slot: i32,
    interactive: bool,
    base: usize,
    feed: Vec<i32>,
    before: Option<(usize, Vec<i32>)>,
    shared: usize,
    budget: usize,
    ctx_limited: bool,
    end: usize,
}

impl NextEngine {
    pub(super) fn check_ids(&self, ids: &[i32]) -> Result<(), EngineError> {
        if ids.iter().any(|&i| i < 0 || i as usize >= self.n_vocab) {
            return Err(err(422, "invalid_request_error", format!("token id outside 0..{}", self.n_vocab)));
        }
        Ok(())
    }

    /// [`Engine::start_request`]: the request checked, its slot and pages taken, and its job made.
    pub(super) fn start(&mut self, req: StartRequest) -> Result<Started, EngineError> {
        self.idle_since = None;
        self.check_ids(&req.ids)?;
        let lat = req.params.latent_steps;
        if lat > 0 {
            self.latent_setup(&req)?;
        }
        // the grammar is built first: a request libllama cannot build one for is refused before anything is reserved
        let gram = self.grammar(&req)?;
        let admitted = self.admit_turn(&req, lat)?;
        Ok(self.begin_job(req, admitted, gram, lat))
    }

    /// The request's grammar (none without one, or with an empty one), or its refusal.
    fn grammar(&self, req: &StartRequest) -> Result<Option<Box<Gram>>, EngineError> {
        let Some(g) = req.params.grammar.as_deref().filter(|g| !g.is_empty()) else { return Ok(None) };
        Ok(Some(Box::new(Gram::new(self.vocab.as_ptr(), g).ok_or_else(|| {
            err(400, "invalid_request_error", "invalid grammar: libllama could not build it (GBNF syntax error, no `root` rule, an undefined rule or left recursion; the reason is on the server's stderr)")
        })?)))
    }

    /// The turn checked against the context, placed in a slot, its budget set and the pages it can reach reserved (a
    /// refusal there gives a new sequence's slot back).
    fn admit_turn(&mut self, req: &StartRequest, lat: usize) -> Result<Admitted, EngineError> {
        let Turn { slot, interactive, base, mut feed, before } = self.turn_of(req)?;
        feed.extend_from_slice(&req.ids);
        let used = self.check_feed(&feed, base)?;
        // everything that can refuse the turn without its slot comes before the slot is taken
        let (budget, ctx_limited) = self.turn_budget(req, lat, used)?;
        let (slot, shared) = self.slot_for(req, slot, &feed)?;
        // the pages it can reach are reserved now, so a step never runs short
        let end = (used + budget).min(self.cfg.n_ctx);
        if let Err(e) = self.reserve(slot, end) {
            self.unplace(req, slot, base, shared);
            return Err(e);
        }
        Ok(Admitted { slot, interactive, base, feed, before, shared, budget, ctx_limited, end })
    }

    /// Where the request's turn starts: an agent's (refused while it is suspended or busy), or a new sequence's (refused
    /// when no slot is free).
    fn turn_of(&self, req: &StartRequest) -> Result<Turn, EngineError> {
        match &req.agent {
            Some(id) => {
                let a = self.agent(id)?;
                if a.off.is_some() {
                    return Err(err(409, "agent_suspended", format!("agent {id} is suspended: resume it first")));
                }
                if self.agent_busy(id) {
                    return Err(err(409, "conflict_error", format!("agent {id} has a request running")));
                }
                let held = a.slot.map_or(0, |s| self.cells[s as usize].len());
                Ok(Turn { slot: a.slot, interactive: a.interactive, base: held, feed: a.pending.clone(), before: Some((held, a.pending.clone())) })
            }
            None => {
                if self.owners.free().is_empty() {
                    return Err(self.slots_full());
                }
                Ok(Turn { slot: None, interactive: req.lane == Lane::Interactive, base: 0, feed: Vec::new(), before: None })
            }
        }
    }

    /// The cells the turn uses (`base` held and `feed` to read): refused when it has silent cells to read again or does not
    /// fit the context.
    fn check_feed(&self, feed: &[i32], base: usize) -> Result<usize, EngineError> {
        if feed.iter().any(|&t| t < 0) {
            return Err(err(409, "conflict_error", "the agent's context lost silent (latent) cells that cannot be read again from tokens"));
        }
        let used = base + feed.len();
        if feed.is_empty() || used >= self.cfg.n_ctx {
            return Err(err(400, "exceed_context_size_error", format!("the request exceeds the available context size ({used} >= n_ctx {})", self.cfg.n_ctx))
                .with_context(used, self.cfg.n_ctx));
        }
        Ok(used)
    }

    /// The turn's slot and how many of `feed`'s tokens it holds already: an agent's own slot, or for a new sequence (a
    /// completion, or an agent without cells) a slot where its prefix already is; the last token is always read (its
    /// scores give the first token).
    fn slot_for(&mut self, req: &StartRequest, slot: Option<i32>, feed: &[i32]) -> Result<(i32, usize), EngineError> {
        if let Some(s) = slot {
            return Ok((s, 0));
        }
        let owner = req.agent.clone().map_or(SlotOwner::Job(req.id), SlotOwner::Agent);
        let (s, n) = self.place(feed, feed.len() - 1, req.params.cache_prompt, owner)?;
        if let Some(a) = req.agent.as_ref().and_then(|id| self.agents.get_mut(id)) {
            a.slot = Some(s);
        }
        Ok((s, n))
    }

    /// The turn's budget and whether the context (not `n_predict`) set it: `lat` latent steps, else `n_predict` within the
    /// cells left after `used`.
    fn turn_budget(&self, req: &StartRequest, lat: usize, used: usize) -> Result<(usize, bool), EngineError> {
        let room = self.cfg.n_ctx - used;
        if lat > 0 {
            if lat > room {
                return Err(err(400, "exceed_context_size_error", format!("{lat} latent steps do not fit the {room} cells left")).with_context(used + lat, self.cfg.n_ctx));
            }
            return Ok((lat, false));
        }
        Ok(if req.params.n_predict < 0 || req.params.n_predict as usize > room { (room, true) } else { (req.params.n_predict as usize, false) })
    }

    /// The pages slot `slot` can reach up to cell `end`, and a recurrent model's state entry, taken now.
    fn reserve(&mut self, slot: i32, end: usize) -> Result<(), EngineError> {
        // a recurrent model's state entry is taken now: a step that found none would fail every request in it
        if self.recurrent && unsafe { sn_state_take(self.e, slot) } != 0 {
            Err(err(503, "kv_pool_exhausted_error", "no recurrent state is free; retry when a sequence ends".to_string()).with_retry_after(200))
        } else {
            self.make_room(chunks(end).saturating_sub(self.held(slot, false)), end, slot)
        }
    }

    /// The admitted turn's job: a new sequence may wait for a running request reading the same start (coalesced prefill),
    /// a recurrent one keeps a checkpoint near its prompt's end and may pin a start recent prompts share.
    fn begin_job(&mut self, req: StartRequest, ad: Admitted, gram: Option<Box<Gram>>, lat: usize) -> Started {
        let Admitted { slot, interactive, base, feed, before, shared, budget, ctx_limited, end } = ad;
        // a new sequence that reads its prompt and keeps it in the prompt cache
        let fresh = base == 0 && lat == 0 && req.params.cache_prompt;
        // coalesced prefill: a new sequence whose first tokens a running request (also new, still reading) reads now waits
        // for them and copies them, instead of reading them again beside it (238 agents asking about one document read it once)
        let mut wait = None;
        if self.coalesce && fresh {
            wait = self.coalesce_lead(&feed, shared);
        }
        let feed_n = feed.len() - shared;
        unsafe { sn_no_end(self.e, slot as c_int, req.params.ignore_eos as c_int) };
        let ck_at = checkpoint_at(&feed, req.params.cache_prompt);
        // a start that recent prompts share and no slot gave: read as a step of its own and pinned
        let pin_at = if fresh && wait.is_none() { self.pin_point(&feed, shared) } else { usize::MAX };
        self.jobs.insert(req.id, Job {
            agent: req.agent.clone(), slot, interactive, feed, fed: shared, pos: base + shared, budget, end, ctx_limited, emitted: 0,
            ignore_eos: req.params.ignore_eos, gram, last: None, carry: Vec::new(), before, spec: SpecState::default(), no_match: 0, seen_from: base + shared,
            lat, lvec: Vec::new(), ck_at, pin_at, wait, n_probs: req.params.n_probs.min(2),
        });
        if let Some(a) = req.agent.as_ref().and_then(|id| self.agents.get_mut(id)) {
            a.pending.clear();
        }
        Started { prompt_n: req.ids.len(), feed_n }
    }

    /// A recurrent model's pinned start for a new prompt `feed` (holding `shared` of its tokens): where the start it shares
    /// with recent prompts ends, when that is past what it holds (usize::MAX: none).
    fn pin_point(&mut self, feed: &[i32], shared: usize) -> usize {
        if !(self.recurrent && self.pin_max > 0) {
            return usize::MAX;
        }
        match self.shared_start(feed) {
            Some(l) if l > shared => l,
            _ => usize::MAX,
        }
    }

    /// Coalesced prefill: the running request (also new, still reading) whose reading of `feed`'s first tokens this new
    /// sequence (holding `shared` of them) waits for, and how many tokens it then copies; a recurrent lead keeps its state there.
    pub(super) fn coalesce_lead(&mut self, feed: &[i32], shared: usize) -> Option<(ReqId, usize)> {
        let lead = self.jobs.iter()
            .filter(|(_, j)| j.reads_new_prompt())
            .map(|(&r, j)| (self.lead_tokens(r, j, feed), r))
            .filter(|&(n, r)| n >= shared + COALESCE_MIN && self.jobs[&r].fed < n)
            .max_by_key(|&(n, r)| (n, self.jobs[&r].fed));
        let (n, r) = lead?;
        if self.recurrent {
            let j = self.jobs.get_mut(&r).expect("found above");
            j.ck_at = n;
        }
        self.prefix.waited_agents += 1;
        Some((r, n))
    }

    /// How many of `feed`'s first tokens the running request `r` (job `j`) can hand over: those they share (the new prompt's
    /// last token excepted).
    fn lead_tokens(&self, r: ReqId, j: &Job, feed: &[i32]) -> usize {
        let n = lcp(&j.feed, feed).min(feed.len() - 1);
        if self.recurrent && j.ck_at != usize::MAX && j.ck_at > j.fed && self.jobs.values().any(|w| w.wait.is_some_and(|(l, _)| l == r)) {
            // its one state copy is promised to another waiting request: only there
            return if j.ck_at <= n { j.ck_at } else { 0 };
        }
        n
    }

    /// [`Engine::prefix_state`]: a request waiting for its lead's reading (coalesced prefill) copies the tokens once they are
    /// read (or reads them all when the lead ended first).
    pub(super) fn wait_state(&mut self, req: ReqId) -> Result<Option<PrefixState>, EngineError> {
        let Some(j) = self.jobs.get(&req) else { return Ok(None) };
        let Some((lr, n)) = j.wait else { return Ok(None) };
        let to = j.slot;
        let from = match self.jobs.get(&lr) {
            Some(l) if l.fed < n && l.wait.is_none() => return Ok(Some(PrefixState::Waiting { tokens: n, written: l.fed })),
            Some(l) => Some(l.slot),
            None => None,  // it ended (or failed) first: read it all
        };
        let got = self.copy_lead(req, from, to, n)?;
        let j = self.jobs.get_mut(&req).expect("checked above");
        j.wait = None;
        if got > 0 {
            (j.fed, j.pos, j.seen_from) = (got, got, got);
            self.cells[to as usize] = j.feed[..got].to_vec();
            self.prefix.attached_agents += 1;
            self.prefix.attached_tokens += got;
        }
        let j = &self.jobs[&req];
        Ok(Some(PrefixState::Ready { feed_n: j.feed.len() - j.fed, shared_n: got }))
    }

    /// The `n` tokens the lead read copied from its slot `from` into the waiting request's slot `to`: how many it got (none
    /// when the lead is gone or its slot cannot be copied that far).
    fn copy_lead(&mut self, req: ReqId, from: Option<i32>, to: i32, n: usize) -> Result<usize, EngineError> {
        let Some(from) = from.filter(|&f| self.cut_best(f, n) == n) else { return Ok(0) };
        match call(|b, len| unsafe { sn_copy(self.e, from, to, n as c_int, b, len) }) {
            Ok(()) => Ok(n),
            Err(_) => {
                // a copy cut short leaves the slot half made: start it empty
                self.reset_slot(to, n);
                let j = self.jobs.get_mut(&req).expect("checked above");
                (j.fed, j.pos, j.seen_from) = (0, 0, 0);
                Ok(0)
            }
        }
    }
}

/// Feed index where a recurrent job keeps its state (CK_LAG before its prompt's end; usize::MAX: none).
fn checkpoint_at(feed: &[i32], cache_prompt: bool) -> usize {
    // (a request that opts out of the prompt cache keeps no checkpoint: no later turn reads from it, and the extra step
    // costs a 256+ token prompt on Qwen3.6-35B-A3B about 30 ms)
    if feed.len() >= CK_MIN && cache_prompt { feed.len() - CK_LAG } else { usize::MAX }
}

impl Job {
    /// A new request still reading its prompt (not waiting for another, not a latent turn): one a new sequence can wait for.
    fn reads_new_prompt(&self) -> bool {
        self.wait.is_none() && self.lat == 0 && self.fed < self.feed.len() && self.pos == self.fed
    }
}
