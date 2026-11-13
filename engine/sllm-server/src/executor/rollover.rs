//! Rollover ([`crate::compaction`]): the end of the window does not end a turn. [`Executor::rollover_admit`] clips a turn
//! so that it stops just short of the end, with room left for the notes. When such a leg stops, the agent writes its notes
//! in its own stream (an executor-owned request on the helper lane), the window is compacted to them (the same steps as any
//! compaction, the summary is the notes) and the turn, held behind that, starts its next leg under the same request id. The
//! client sees one stream with a pause in it. A turn that fits is not touched.

use super::*;
use crate::compaction::{ask_with_request, facts_cap, notes_text, request_of, turn_start, Spec, Trigger, COMPACT_REQ_BASE, NOTES_REQUEST_TOKENS, RECALL_QUERY_TOKENS};

/// Room kept at the end of the window beside the ask and the notes.
const SPARE: usize = 8;
/// The fewest prompt tokens worth reading before a rollover.
const MIN_SLICE: usize = 64;
/// The room a rollover must leave for the work, or it is not worth its notes: below it the plain limits answer.
const MIN_PROGRESS: usize = 128;

/// A turn that was clipped: what it has left, and what its finished legs did.
#[derive(Default)]
pub(super) struct Turn {
    /// Its request, with the `n_predict` of the whole turn.
    params: Option<SamplingParams>,
    /// What the turn's prompt asks (the first leg's).
    request: Vec<i32>,
    /// Prompt tokens not read yet.
    rest: Vec<i32>,
    /// Tokens still to write (-1: until the end of generation).
    left: i64,
    rolls: u32,
    /// Summed over the finished legs.
    prompt_n: usize,
    predicted_n: usize,
    prompt_ms: f64,
    predicted_ms: f64,
    paused_ms: f64,
    first: Option<(usize, f64)>,
}

impl Turn {
    /// Prompt left to read: the turn goes on whatever the last leg stopped with, because what the model would say next in
    /// the middle of a prompt means nothing, and an end of generation there is not the end of the turn. Otherwise a leg
    /// that was clipped goes on, and one that the model ended, or that someone stopped, does not.
    fn goes_on(&self, reason: StopReason) -> bool {
        match reason {
            StopReason::Cancelled | StopReason::Deadline => false,
            StopReason::Eos => !self.rest.is_empty(),
            StopReason::Limit | StopReason::Context => self.left != 0 || !self.rest.is_empty(),
        }
    }
}

/// An agent writing its notes.
pub(super) struct Writing {
    pub(super) agent: AgentId,
    rx: mpsc::Receiver<ReqEvent>,
    /// The window to compact (the notes are not part of it).
    ctx: Vec<i32>,
    /// A leg that stopped in the middle of its prompt, not of a reply.
    reading: bool,
    /// What the turn asks, when it stopped in the middle of its reply.
    request: Vec<i32>,
    tail_tokens: usize,
    notes: String,
    t0: Instant,
}

/// What a turn too long for the window does ([`Executor::rollover_long`]).
enum Long {
    /// Held behind a rollover.
    Held,
    /// Runs now as it is (a first slice of its prompt, or its last leg).
    Run(StartCmd),
    /// Its prompt fits after all, without the recalled lines.
    Fits(StartCmd),
}

struct Shape {
    prefix: usize,
    /// The turn stops here.
    limit: usize,
    summary: usize,
    tail: usize,
    /// How much of what a turn asks stays beside the notes.
    request: usize,
}

impl Executor {
    fn roll_shape(&self, aid: &str) -> Option<Shape> {
        let text = self.comp_cfg.text.as_ref()?;
        let n_ctx = self.shared.info.get()?.n_ctx;
        let a = self.agents.get(aid)?;
        let window = a.comp.policy.and_then(|p| p.window_tokens).map_or(n_ctx, |w| w.min(n_ctx));
        let prefix = a.comp.prefix_tokens;
        let (summary, tail) = self.comp_cfg.rollover.sizes(window.checked_sub(prefix)?)?;
        let request = NOTES_REQUEST_TOKENS.min(window / 16);
        let ask = text.ask_writing.len().max(text.ask_reading.len() + text.request.len() + request);
        let limit = window.checked_sub(ask + summary + SPARE)?;
        let kept = prefix + text.working_on.len() + request + text.your_notes.len() + summary + facts_cap(window) + text.foot_writing.len() + tail;
        (kept + MIN_PROGRESS <= limit).then_some(Shape { prefix, limit, summary, tail, request })
    }

    /// A turn about to start: clip it to the window, or roll over first. `None`: it is held behind a rollover.
    pub(super) fn rollover_admit(&mut self, mut s: StartCmd) -> Option<StartCmd> {
        let Some(aid) = s.agent.clone() else { return Some(s) };
        if !self.rollover_applies(&aid, &s) {
            return Some(s);
        }
        let Some(sh) = self.rollover_shape_for(&aid, &s) else { return Some(self.rollover_last_leg(s)) };
        let len = self.engine.agent_context_tokens(&aid).unwrap_or(0);
        let mut with_lines = self.rollover_recall(&aid, &s, &sh);
        let mut n = with_lines.as_ref().map_or(s.ids.len(), Vec::len);
        if len + n >= sh.limit {
            match self.rollover_long(s, &sh, len, n) {
                Long::Held => return None,
                Long::Run(s) => return Some(s),
                Long::Fits(f) => {
                    s = f;
                    with_lines = None; // the prompt fits, its lines do not, and there is nothing to roll over
                    n = s.ids.len();
                }
            }
        }
        if let Some(ids) = with_lines {
            self.rollover_use_lines(&aid, &mut s, ids);
        }
        self.rollover_clip(&mut s, (sh.limit - len - n) as i64, sh.request);
        Some(s)
    }

    /// Whether the rollover looks at this turn at all: not the executor's own compactions, not with rollover off, not for an
    /// unknown agent, and not for a latent turn or an agent whose context holds silent cells (no tokens to write notes
    /// from): no clip, no rollover.
    fn rollover_applies(&self, aid: &str, s: &StartCmd) -> bool {
        if is_compaction(s.id) || !self.comp_cfg.rollover.on || !self.agents.contains_key(aid) {
            return false;
        }
        !(s.params.latent_steps > 0 || (self.engine.agent_context_tokens(aid).is_some() && self.engine.agent_tokens(aid).is_none()))
    }

    /// The window's shape for this turn, None when it runs as its last leg: a busy agent, a re-prefill, the turn's rollovers
    /// used up, or a window with no room for one.
    fn rollover_shape_for(&self, aid: &str, s: &StartCmd) -> Option<Shape> {
        let a = &self.agents[aid];
        let rolls = self.rolled.get(&s.id).map_or(0, |t| t.rolls);
        if a.busy.is_some() || a.life.rebuild > 0 || rolls >= self.comp_cfg.rollover.max_per_turn {
            return None;
        }
        self.roll_shape(aid)
    }

    /// A turn of `n` tokens that does not fit after the `len` the agent holds: roll over first, read what fits (then roll
    /// over and read on), or run it to the plain limits.
    fn rollover_long(&mut self, mut s: StartCmd, sh: &Shape, len: usize, n: usize) -> Long {
        let can_roll = len > sh.prefix + sh.tail + sh.summary;
        let left_after = sh.prefix + sh.tail + 2 * sh.summary + crate::compaction::facts_cap(sh.limit);
        if can_roll && left_after + n < sh.limit {
            self.rollover_begin(s, false);
            return Long::Held;
        }
        // longer than a window: read what fits, roll over, read on
        let take = sh.limit.saturating_sub(len);
        if take >= MIN_SLICE && take < s.ids.len() {
            let t = self.rolled.entry(s.id).or_default();
            t.rest = s.ids.split_off(take);
            t.left = s.params.n_predict;
            t.params = Some(s.params.clone());
            s.params.n_predict = 0;
            return Long::Run(s);
        }
        if can_roll {
            self.rollover_begin(s, false);
            return Long::Held;
        }
        if len + s.ids.len() >= sh.limit {
            return Long::Run(self.rollover_last_leg(s)); // nothing to replace: the plain limits answer
        }
        Long::Fits(s)
    }

    /// The turn's prompt with the recalled lines it is about.
    fn rollover_use_lines(&mut self, aid: &str, s: &mut StartCmd, ids: Vec<i32>) {
        s.ids = ids;
        if let Some(a) = self.agents.get_mut(aid) {
            a.comp.recalled += 1;
        }
    }

    /// Clip the turn's reply to the `room` left before the window's limit (what is left over is written after a rollover);
    /// a reply that fits is the turn's last leg.
    fn rollover_clip(&mut self, s: &mut StartCmd, room: i64, request_tokens: usize) {
        let want = s.params.n_predict;
        if want < 0 || want > room {
            let request = self.comp_cfg.text.as_ref().map_or(Vec::new(), |x| request_of(x, &s.ids, request_tokens).to_vec());
            let t = self.rolled.entry(s.id).or_default();
            if t.request.is_empty() && s.ids.len() > 1 {
                t.request = request;
            }
            t.left = if want < 0 { -1 } else { want - room };
            t.params = Some(s.params.clone());
            s.params.n_predict = room;
        } else if let Some(t) = self.rolled.get_mut(&s.id) {
            t.left = 0;
        }
    }

    /// This leg runs to the plain limits, and the turn ends with it: a leg that is not clipped must not count as one that
    /// has more to do, or the turn would go round for ever.
    fn rollover_last_leg(&mut self, s: StartCmd) -> StartCmd {
        if let Some(t) = self.rolled.get_mut(&s.id) {
            t.left = 0;
            t.rest.clear();
        }
        s
    }

    /// A new turn of an agent that has lost lines: its prompt with the ones it is about. Only used if the turn then fits; if
    /// it does not, this is asked again after the rollover. Not a leg that goes on from its last token.
    fn rollover_recall(&mut self, aid: &str, s: &StartCmd, sh: &Shape) -> Option<Vec<i32>> {
        // an agent that has not rolled over pays nothing here: most turns of most agents
        if self.agents.get(aid)?.comp.archive.is_empty() || s.ids.len() < 2 || std::env::var_os("SLLM_RECALL").is_some_and(|v| v == "0") {
            return None;
        }
        let text = self.comp_cfg.text.clone()?;
        let ctx = self.engine.agent_tokens(aid)?;
        let a = self.agents.get_mut(aid)?;
        a.comp.archive.add(&(text.decode)(&ctx[a.comp.archived_to.min(ctx.len())..]));
        a.comp.archived_to = ctx.len();
        let lines = a.comp.archive.recall(&(text.decode)(request_of(&text, &s.ids, RECALL_QUERY_TOKENS)))?;
        if std::env::var_os("SLLM_LOG_FACTS").is_some() {
            eprintln!("sllm-server: recall for {aid}: {lines}");
        }
        Some(text.with_recall(&s.ids, &lines, sh.limit / 16))
    }

    /// Does this finished leg go on?
    pub(super) fn rollover_wanted(&self, id: ReqId, reason: StopReason) -> bool {
        self.rolled.get(&id).is_some_and(|t| t.goes_on(reason))
    }

    /// A leg stopped short of the end of the window: note what it did, write the notes, hold the next leg.
    pub(super) fn rollover_continue(&mut self, id: ReqId, slot: Slot, d: &Done) {
        let Some(aid) = slot.agent.clone() else { return };
        let deadline_ms = slot.deadline.map(|t| t.saturating_duration_since(Instant::now()).as_millis() as u64);
        self.release_agent(&slot, true);
        self.shared.requests_started.fetch_sub(1, Relaxed); // the turn goes on: it is not started a second time
        let t = self.rolled.get_mut(&id).expect("rollover_wanted checked");
        t.first.get_or_insert((d.cached_n, d.queued_ms));
        t.prompt_n += d.prompt_n;
        t.predicted_n += d.predicted_n;
        t.prompt_ms += d.prompt_ms;
        t.predicted_ms += d.predicted_ms;
        let reading = !t.rest.is_empty();
        let ids = std::mem::take(&mut t.rest);
        let params = SamplingParams { n_predict: t.left, ..t.params.clone().expect("set with the clip") };
        self.rollover_begin(StartCmd { id, agent: Some(aid), ids, params, lane: Some(slot.lane), deadline_ms, events: slot.events }, reading);
    }

    /// Hold `turn` and have the agent write its notes.
    fn rollover_begin(&mut self, mut turn: StartCmd, reading: bool) {
        let t0 = Instant::now();
        let aid = turn.agent.clone().expect("a rollover is an agent's");
        let Some(mut ctx) = self.engine.agent_tokens(&aid) else {
            return self.rollover_refused(turn, EngineError::new(501, "not_supported_error", "this engine does not keep the agent's tokens"));
        };
        if turn.ids.is_empty() {
            // the turn goes on from its last token: the next leg reads it
            turn.ids.extend(ctx.pop());
        }
        let Some(sh) = self.roll_shape(&aid) else {
            return self.rollover_refused(turn, EngineError::new(500, "server_error", "the window has no room for a rollover"));
        };
        let text = self.comp_cfg.text.as_ref().expect("roll_shape checked");
        let ask = if reading { ask_with_request(text, &turn.ids, sh.request) } else { text.ask_writing.clone() };
        let n_ctx = self.shared.info.get().map_or(0, |i| i.n_ctx);
        let room = n_ctx.saturating_sub(ctx.len() + 1 + ask.len() + SPARE).min(sh.summary);
        // a turn that stopped in its reply: its next leg has no prompt, and the request has to stay in the window
        let request = if reading || turn.ids.len() > 1 { Vec::new() } else { self.rolled.get(&turn.id).map_or(Vec::new(), |t| t.request.clone()) };
        self.rolled.entry(turn.id).or_default();
        self.held.insert(aid.clone(), turn);
        let (tx, rx) = mpsc::channel(EVENT_CAP);
        let w = Writing { agent: aid.clone(), rx, ctx, reading, request, tail_tokens: sh.tail, notes: String::new(), t0 };
        if room == 0 {
            return self.rollover_cut(w); // a window filled before this server: the facts record alone
        }
        let id = COMPACT_REQ_BASE + self.next_compact;
        self.next_compact += 1;
        let params = SamplingParams { n_predict: room as i64, temperature: 0.0, ignore_eos: false, cache_prompt: true, grammar: None, latent_steps: 0, n_probs: 0 };
        self.comp_phase(&aid, crate::compaction::PhaseEv::NotesStarted);
        self.start(StartCmd { id, agent: Some(aid), ids: ask, params, lane: Some(Lane::Helper), deadline_ms: None, events: tx });
        self.writing.push(w);
    }

    /// At the step boundary: notes that are written go into the window.
    pub(super) fn rollover_tick(&mut self) {
        let mut i = 0;
        while i < self.writing.len() {
            let w = &mut self.writing[i];
            let mut end = false;
            while let Ok(ev) = w.rx.try_recv() {
                match ev {
                    ReqEvent::Token { text, .. } => w.notes.push_str(&text),
                    ReqEvent::Done(_) => end = true,
                    ReqEvent::Error(e) => {
                        // no notes: the facts record and the tail still carry the turn on
                        eprintln!("sllm-server: the notes of {} failed: {e}", w.agent);
                        end = true;
                    }
                    ReqEvent::Started { .. } => {}
                }
            }
            if end {
                let w = self.writing.swap_remove(i);
                self.rollover_cut(w);
            } else {
                i += 1;
            }
        }
    }

    fn rollover_cut(&mut self, w: Writing) {
        let Writing { agent, ctx, reading, request, tail_tokens, notes, t0, .. } = w;
        let text = self.comp_cfg.text.as_ref().expect("roll_shape checked");
        // real turns in the tail: it starts with one, and the notes are an exchange of their own
        let from = ctx.len().saturating_sub(tail_tokens);
        let turn = turn_start(&ctx, from, &text.user_open).filter(|_| !reading);
        let foot = match (reading, turn) {
            (true, _) => text.foot_reading.clone(),
            (false, Some(_)) => text.foot_turns.clone(),
            (false, None) => text.foot_writing.clone(),
        };
        if std::env::var_os("SLLM_LOG_FACTS").is_some() {
            eprintln!("sllm-server: notes of {agent}: {notes}");
        }
        let summary = (text.encode)(notes_text(&notes));
        let head = if request.is_empty() { text.head.clone() } else { text.working_on.iter().chain(&request).chain(&text.your_notes).copied().collect() };
        let spec = Spec {
            summary: Some(summary),
            tail_tokens: Some(tail_tokens),
            tail_from: turn,
            prefix_tokens: self.agents.get(&agent).map(|a| a.comp.prefix_tokens.max(if !text.bos.is_empty() && ctx.starts_with(&text.bos) { text.bos.len() } else { 0 })),
            summary_tokens: None,
            trigger: Trigger::Limit,
            summarized_by: Some(("brain".to_string(), t0.elapsed().as_secs_f64() * 1e3, t0)),
            context: Some(ctx),
            frame: Some((head, foot)),
        };
        if let Err((e, _)) = self.begin_compaction(&agent, spec, None) {
            if let Some(h) = self.held.remove(&agent) {
                self.rollover_refused(h, e);
            }
            self.compaction_over(&agent);
        }
    }

    /// The rollover cannot go on: the turn ends with the reason.
    fn rollover_refused(&mut self, turn: StartCmd, e: EngineError) {
        self.rolled.remove(&turn.id);
        self.shared.requests_failed.fetch_add(1, Relaxed);
        let _ = turn.events.try_send(ReqEvent::Error(e));
    }

    /// A rollover's compaction ended.
    pub(super) fn rollover_ended(&mut self, agent: &str, ms: f64) {
        let Some(id) = self.held.get(agent).map(|h| h.id) else { return };
        if let Some(t) = self.rolled.get_mut(&id) {
            t.rolls += 1;
            t.paused_ms += ms;
        }
    }

    /// A rollover's compaction failed after its cut: the window holds its prefix only, so the turn cannot go on.
    pub(super) fn rollover_failed(&mut self, agent: &str, e: EngineError) {
        if let Some(h) = self.held.remove(agent) {
            self.rollover_refused(h, e);
        }
    }

    /// The last leg's report, with what the legs before it did.
    pub(super) fn rollover_totals(&mut self, id: ReqId, d: &mut Done) {
        let Some(t) = self.rolled.remove(&id) else { return };
        if let Some((cached, queued)) = t.first {
            d.cached_n = cached;
            d.queued_ms = queued;
        }
        d.prompt_n += t.prompt_n;
        d.predicted_n += t.predicted_n;
        d.prompt_ms += t.prompt_ms;
        d.predicted_ms += t.predicted_ms;
        d.rollovers = t.rolls;
        d.rollover_ms = t.paused_ms;
    }

    /// Cancel a turn held behind its rollover (or its compaction). False for any other id. Notes still being written for
    /// it are called off: the window stays as it is.
    pub(super) fn rollover_cancel(&mut self, id: ReqId) -> bool {
        let Some(aid) = self.held.iter().find(|(_, h)| h.id == id).map(|(a, _)| a.clone()) else { return false };
        let h = self.held.remove(&aid).expect("found above");
        if self.writing.iter().any(|w| w.agent == aid) {
            self.rollover_forget(&aid);
            self.compaction_over(&aid);
        }
        let context_tokens = self.engine.agent_context_tokens(&aid);
        let mut d = Done {
            stop: StopReason::Cancelled,
            prompt_n: 0,
            predicted_n: 0,
            cached_n: 0,
            prompt_ms: 0.0,
            predicted_ms: 0.0,
            queued_ms: 0.0,
            context_tokens,
            shared_n: 0,
            wait_ms: 0.0,
            wake: None,
            rollovers: 0,
            rollover_ms: 0.0,
        };
        self.rollover_totals(id, &mut d);
        self.shared.requests_started.fetch_add(1, Relaxed);
        self.shared.requests_cancelled.fetch_add(1, Relaxed);
        let _ = h.events.try_send(ReqEvent::Done(d));
        true
    }

    /// The agent is going away.
    pub(super) fn rollover_forget(&mut self, agent: &str) {
        self.writing.retain(|w| w.agent != agent);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prompt_left_to_read_goes_on_whatever_the_model_said() {
        let reading = Turn { rest: vec![1, 2, 3], left: 40, ..Default::default() };
        assert!(reading.goes_on(StopReason::Eos), "an end of generation in the middle of a prompt");
        assert!(reading.goes_on(StopReason::Limit));
        assert!(!reading.goes_on(StopReason::Cancelled));
        assert!(!reading.goes_on(StopReason::Deadline));
        let writing = Turn { left: 40, ..Default::default() };
        assert!(writing.goes_on(StopReason::Limit), "clipped: more to write");
        assert!(!writing.goes_on(StopReason::Eos), "the model is done");
        let done = Turn::default();
        assert!(!done.goes_on(StopReason::Limit), "the last leg");
    }
}
