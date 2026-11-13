//! The executor's compaction steps ([`crate::compaction`]): a compaction is the engine's cut
//! ([`Engine::agent_cut`]) followed by an executor-owned request of the agent on the helper lane that prefills the
//! summary and the tail (`n_predict` 0). Its events come back on a channel the executor holds, collected at every step
//! boundary ([`Executor::compaction_tick`]); the requester's reply is sent then.

use super::*;
use crate::compaction::{cut_points, feed, CompactionPhase, PhaseEv, Report, Spec, Trigger, COMPACT_REQ_BASE};
use std::hash::{Hash, Hasher};

type Reply = oneshot::Sender<Result<Report, EngineError>>;

/// A compaction waiting for an asynchronous summary: the cut points are fixed when it starts.
pub(super) struct Summarizing {
    agent: AgentId,
    spec: Spec,
    reply: Option<Reply>,
    pending: Box<dyn crate::compaction::PendingSummary>,
    source: String,
    t0: Instant,
    /// The window up to the tail as it was when the summary began ([`window_print`]): the summary replaces that span only.
    window: u64,
}

/// A fingerprint of a window's tokens.
fn window_print(ids: &[i32]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ids.hash(&mut h);
    h.finish()
}

/// Where a compaction's summary comes from.
enum SummaryOf {
    /// Written: its ids, its source, and how long an asynchronous summarizer took (None: timed here).
    Ready { ids: Vec<i32>, source: String, ms: Option<f64> },
    /// An asynchronous summarizer has started on it.
    Started { pending: Box<dyn crate::compaction::PendingSummary>, source: String },
}

fn invalid(m: impl Into<String>) -> EngineError {
    EngineError::new(422, "invalid_request_error", m)
}

impl Executor {
    /// Start a compaction of `aid`. On a refusal before the cut the agent is unchanged and the reply comes back with the
    /// error, for the caller to send.
    pub(super) fn begin_compaction(&mut self, aid: &str, mut spec: Spec, reply: Option<Reply>) -> Result<(), (EngineError, Option<Reply>)> {
        let t0 = Instant::now();
        if let Some(e) = self.compaction_refused(aid, &spec) {
            return Err((e, reply));
        }
        let Some(ctx) = spec.context.take().or_else(|| self.engine.agent_tokens(aid)) else {
            return Err((EngineError::new(501, "not_supported_error", "this engine does not keep the agent's tokens (compaction needs the paged cache)"), reply));
        };
        let prefix = spec.prefix_tokens.unwrap_or(self.agents[aid].comp.prefix_tokens);
        let (prefix, tail_start) = match cut_points(ctx.len(), prefix, spec.tail_tokens, spec.tail_from) {
            Ok(c) => c,
            Err(m) => return Err((invalid(m), reply)),
        };
        let replaced = &ctx[prefix..tail_start];
        let t_sum = Instant::now();
        let mut t0 = t0;
        let (summary, source, summarize_ms_async) = match self.compaction_summary(aid, &mut spec, replaced, &mut t0) {
            Ok(SummaryOf::Ready { ids, source, ms }) => (ids, source, ms),
            Ok(SummaryOf::Started { pending, source }) => {
                // asynchronous: the agent stays as it is until the summary is written; the tail is kept from here on
                let spec = Spec { prefix_tokens: Some(prefix), tail_tokens: None, tail_from: Some(tail_start), ..spec };
                let window = window_print(&ctx[..tail_start]);
                self.summarizing.push(Summarizing { agent: aid.to_string(), spec, reply, pending, source, t0, window });
                self.comp_phase(aid, PhaseEv::SummaryStarted);
                return Ok(());
            }
            Err(e) => return Err((e, reply)),
        };
        let summarize_ms = summarize_ms_async.unwrap_or_else(|| t_sum.elapsed().as_secs_f64() * 1e3);
        let (facts, facts_lines) = self.compaction_facts(&source, replaced, &ctx[tail_start..]);
        let (head, foot) = spec.frame.take().unwrap_or_default();
        let with_facts: Vec<i32> = head.iter().chain(&summary).chain(&facts).chain(&foot).copied().collect();
        let ids = feed(&ctx, tail_start, &with_facts);
        self.log_window(aid, prefix, &ids);
        // everything that start_request would refuse is checked before the cut, so a refusal leaves the agent as it was
        if let Some(e) = self.compacted_window_refused(&summary, prefix, ids.len()) {
            return Err((e, reply));
        }
        let t_cut = Instant::now();
        let (cut, ids) = match self.compaction_cut(aid, &ctx, prefix, ids) {
            Ok(c) => c,
            Err(e) => return Err((e, reply)),
        };
        let cut_ms = t_cut.elapsed().as_secs_f64() * 1e3;
        self.archive_replaced(aid, &ctx, prefix, with_facts.len());
        let report = Report {
            agent_id: aid.to_string(),
            trigger: Some(spec.trigger),
            summary_source: source,
            prefix_tokens: prefix,
            replaced_tokens: tail_start - prefix,
            summary_tokens: summary.len(),
            facts_tokens: facts.len(),
            facts_lines,
            tail_tokens: ctx.len() - tail_start,
            context_before: ctx.len(),
            context_after: prefix,
            pages_before: cut.pages_before,
            pages_after: cut.pages_after,
            pages_freed: cut.pages_freed,
            prefix_pages: cut.prefix_pages,
            prefix_shared_pages: cut.prefix_shared,
            summarize_ms,
            cut_ms,
            reprefill_tokens: ids.len(),
            ..Default::default()
        };
        let f = InFlight { agent: aid.to_string(), rx: mpsc::channel(1).1, reply, report, t0 };
        self.prefill_compacted(aid, f, ids);
        Ok(())
    }

    /// Why a compaction of `aid` cannot start now: no such agent, one already running or waiting for its summary, a turn
    /// running, or the KV not in RAM (phase 3 lifecycle).
    fn compaction_refused(&self, aid: &str, spec: &Spec) -> Option<EngineError> {
        let Some(a) = self.agents.get(aid) else { return Some(Self::no_agent(aid)) };
        // a compaction under way goes on only with its own next step (its summary written, or a rollover's notes)
        let own_next = spec.summarized_by.is_some() && matches!(a.comp.phase, CompactionPhase::Summarizing | CompactionPhase::WritingNotes);
        if !a.comp.phase.is_idle() && !own_next {
            return Some(compaction::compacting(aid));
        }
        if a.busy.is_some() {
            return Some(EngineError::new(409, "conflict_error", format!("agent {aid} is busy with a turn")));
        }
        self.life_compaction_guard(aid) // phase 3 lifecycle: the KV must be in RAM
    }

    /// The summary of `replaced`: the client's, an asynchronous summarizer's that is written now (then `t0` is when the
    /// compaction began), or the summarizer's; or an asynchronous summarizer has just started on it.
    fn compaction_summary(&self, aid: &str, spec: &mut Spec, replaced: &[i32], t0: &mut Instant) -> Result<SummaryOf, EngineError> {
        if let Some(s) = spec.summary.take() {
            return Ok(match &spec.summarized_by {
                Some((name, ms, started)) => {
                    *t0 = *started;
                    SummaryOf::Ready { ids: s, source: name.clone(), ms: Some(*ms) }
                }
                None => SummaryOf::Ready { ids: s, source: "client".to_string(), ms: None },
            });
        }
        let target = spec.summary_tokens.unwrap_or(compaction::DEFAULT_SUMMARY_TOKENS);
        let s = self.comp_cfg.summarizer.0.clone();
        if let Some(pending) = s.start(aid, replaced, target) {
            return Ok(SummaryOf::Started { pending, source: s.name().to_string() });
        }
        match s.summarize(aid, replaced, target) {
            Ok(v) => Ok(SummaryOf::Ready { ids: v, source: s.name().to_string(), ms: None }),
            Err(m) => Err(EngineError::new(500, "server_error", format!("summarizer {}: {m}", s.name()))),
        }
    }

    /// The facts record (lines of the replaced text word for word) that goes right after the summary, and its line count; a
    /// client's own summary is placed as it is.
    fn compaction_facts(&self, source: &str, replaced: &[i32], tail: &[i32]) -> (Vec<i32>, usize) {
        match (source, self.comp_cfg.text.as_ref().and_then(|t| t.facts.as_ref())) {
            ("client", _) => None,
            ("brain", Some(f)) => f(replaced, tail, crate::compaction::facts_cap(self.shared.info.get().map_or(0, |i| i.n_ctx))),
            _ => self.comp_cfg.summarizer.0.facts(replaced, tail),
        }
        .unwrap_or_default()
    }

    /// `SLLM_LOG_FACTS`: the window as it will be after the cut.
    fn log_window(&self, aid: &str, prefix: usize, ids: &[i32]) {
        if std::env::var_os("SLLM_LOG_FACTS").is_none() {
            return;
        }
        if let Some(t) = &self.comp_cfg.text {
            eprintln!("sllm-server: window of {aid} after the cut ({} tokens):\n{}\n[end of window]", prefix + ids.len(), (t.decode)(ids));
        }
    }

    /// What start_request would refuse in the compacted window: a summary token outside the vocabulary, or a window of
    /// `prefix` plus `n_ids` that does not fit the context.
    fn compacted_window_refused(&self, summary: &[i32], prefix: usize, n_ids: usize) -> Option<EngineError> {
        let info = self.shared.info.get()?;
        if summary.iter().any(|&t| t < 0 || t as usize >= info.n_vocab) {
            return Some(invalid(format!("summary token id outside 0..{}", info.n_vocab)));
        }
        if prefix + n_ids >= info.n_ctx {
            let e = EngineError::new(400, "exceed_context_size_error", format!("the compacted window ({} tokens) exceeds the context size {}", prefix + n_ids, info.n_ctx));
            return Some(e.with_context(prefix + n_ids, info.n_ctx));
        }
        None
    }

    /// The engine's cut to `prefix`, and the ids to prefill after it.
    fn compaction_cut(&mut self, aid: &str, ctx: &[i32], prefix: usize, ids: Vec<i32>) -> Result<(CutStats, Vec<i32>), EngineError> {
        match self.engine.agent_cut(aid, prefix) {
            Ok(c) => Ok((c, ids)),
            // a cache that cannot drop a range (a recurrent state): the whole window is read again
            Err(e) if e.status == 501 && prefix > 0 => {
                let c = self.engine.agent_cut(aid, 0)?;
                Ok((c, ctx[..prefix].iter().chain(&ids).copied().collect()))
            }
            Err(e) => Err(e),
        }
    }

    /// The replaced text (since the last notes) goes to the agent's archive; the new notes end after the framed summary.
    fn archive_replaced(&mut self, aid: &str, ctx: &[i32], prefix: usize, framed: usize) {
        if let (Some(t), Some(a)) = (&self.comp_cfg.text, self.agents.get_mut(aid)) {
            a.comp.archive.add(&(t.decode)(&ctx[a.comp.notes_end.clamp(prefix, ctx.len())..]));
            a.comp.notes_end = prefix + framed;
        }
    }

    /// After the cut: the summary and the tail are prefilled by an executor-owned request of the agent on the helper lane.
    fn prefill_compacted(&mut self, aid: &str, mut f: InFlight, ids: Vec<i32>) {
        if ids.is_empty() {
            // no summary and no tail: the cut is the whole compaction
            self.end_compaction(f, None);
            return;
        }
        let req = COMPACT_REQ_BASE + self.next_compact;
        self.next_compact += 1;
        let (tx, rx) = mpsc::channel(EVENT_CAP);
        f.rx = rx;
        let params = SamplingParams { n_predict: 0, temperature: 0.0, ignore_eos: false, cache_prompt: true, grammar: None, latent_steps: 0, n_probs: 0 };
        self.start(StartCmd { id: req, agent: Some(aid.to_string()), ids, params, lane: Some(Lane::Helper), deadline_ms: None, events: tx });
        match f.rx.try_recv() {
            Ok(ReqEvent::Error(e)) => {
                // refused after the cut: the agent holds its prefix only
                self.end_compaction(f, Some(Err(e)));
            }
            // admitted, or no room yet (it starts when a request ends): its end comes on `f.rx`
            Ok(ReqEvent::Started { .. }) => self.prefill_started(aid, req, f),
            Err(_) if self.room_wait.iter().any(|w| w.cmd.id == req) => self.prefill_started(aid, req, f),
            _ => {
                self.end_compaction(f, Some(Err(EngineError::new(500, "server_error", "the compaction request was not admitted"))));
            }
        }
    }

    fn prefill_started(&mut self, aid: &str, req: ReqId, f: InFlight) {
        self.comp_phase(aid, PhaseEv::PrefillStarted(req));
        self.inflight.push(f);
    }

    /// Move the agent's compaction phase; a move [`CompactionPhase::next`] refuses is logged and changes nothing.
    pub(super) fn comp_phase(&mut self, aid: &str, ev: PhaseEv) {
        let Some(a) = self.agents.get_mut(aid) else { return };
        match a.comp.phase.next(ev) {
            Some(p) => a.comp.phase = p,
            None => eprintln!("sllm-server: {aid}: compaction {ev:?} refused in {:?}", a.comp.phase),
        }
    }

    /// The agent's compaction is over (ended, failed or called off): its phase is Idle, and the turn held behind it starts
    /// (the next leg of a rolled turn is older than the turns queued meanwhile: it goes first).
    pub(super) fn compaction_over(&mut self, aid: &str) {
        self.comp_phase(aid, PhaseEv::Ended);
        if let Some(h) = self.held.remove(aid) {
            let leg = self.rolled.contains_key(&h.id);
            self.start_inner(h, leg);
        }
    }

    /// At the step boundary: finished compactions reply and release the turn held behind them; agents whose turn just
    /// ended are checked against their policy.
    pub(super) fn compaction_tick(&mut self) {
        self.rollover_tick();
        self.summaries_written();
        self.compactions_ended();
        for aid in std::mem::take(&mut self.due) {
            self.policy_check(&aid);
        }
    }

    /// Asynchronous summaries that are written: their compaction cuts now (or at a later boundary, if a turn of the agent
    /// is running).
    fn summaries_written(&mut self) {
        let mut i = 0;
        while i < self.summarizing.len() {
            let busy = self.agents.get(&self.summarizing[i].agent).is_some_and(|a| a.busy.is_some());
            let ready = if busy { None } else { self.summarizing[i].pending.poll() };
            match ready {
                None => i += 1,
                Some(r) => {
                    let s = self.summarizing.swap_remove(i);
                    self.compact_with_summary(s, r);
                }
            }
        }
    }

    /// An asynchronous summary `r` is written: the compaction goes on with it (or replies with the failure).
    fn compact_with_summary(&mut self, s: Summarizing, r: Result<Vec<i32>, String>) {
        let Summarizing { agent, spec, reply, source, t0, window, .. } = s;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let res = match r {
            Ok(_) if !self.window_unchanged(&agent, spec.tail_from.unwrap_or(0), window) => {
                Err((EngineError::new(409, "conflict_error", format!("the window of agent {agent} changed while its summary was written")), reply))
            }
            Ok(ids) => {
                let spec = Spec { summary: Some(ids), summarized_by: Some((source, ms, t0)), ..spec };
                self.begin_compaction(&agent, spec, reply)
            }
            Err(m) => Err((EngineError::new(500, "server_error", format!("summarizer {source}: {m}")), reply)),
        };
        if let Err((e, reply)) = res {
            self.compaction_over(&agent);
            match reply {
                Some(r) => {
                    let _ = r.send(Err(e));
                }
                None => eprintln!("sllm-server: compaction of {agent} failed: {e}"),
            }
        }
    }

    /// Whether the agent's window still holds, up to `tail_from`, what it held when its summary began (`window`).
    fn window_unchanged(&self, aid: &str, tail_from: usize, window: u64) -> bool {
        self.engine.agent_tokens(aid).is_some_and(|c| c.len() >= tail_from && window_print(&c[..tail_from]) == window)
    }

    /// Compactions whose prefill has ended (or failed).
    fn compactions_ended(&mut self) {
        let mut i = 0;
        while i < self.inflight.len() {
            match self.inflight[i].end() {
                None => i += 1,
                Some(r) => {
                    let f = self.inflight.swap_remove(i);
                    self.end_compaction(f, Some(r));
                }
            }
        }
    }

    fn end_compaction(&mut self, f: InFlight, end: Option<Result<Done, EngineError>>) {
        let InFlight { agent, reply, mut report, t0, .. } = f;
        let r = match end {
            None => Ok(()),
            Some(Ok(d)) if matches!(d.stop, StopReason::Cancelled | StopReason::Deadline) => {
                Err(EngineError::new(499, "cancelled", "the compaction was stopped before its prefill ended"))
            }
            Some(Ok(d)) => {
                report.reprefill_ms = d.prompt_ms;
                report.queued_ms = d.queued_ms;
                Ok(())
            }
            Some(Err(e)) => Err(e),
        };
        report.total_ms = t0.elapsed().as_secs_f64() * 1e3;
        let limit = report.trigger == Some(Trigger::Limit);
        self.record_compaction(&agent, &mut report, r.is_ok(), limit);
        if limit {
            match &r {
                Ok(()) => self.rollover_ended(&agent, report.total_ms),
                Err(e) => self.rollover_failed(&agent, e.clone()),
            }
        }
        if let Some(reply) = reply {
            let _ = reply.send(r.map(|()| report));
        } else if let Err(e) = r {
            eprintln!("sllm-server: compaction of {agent} failed: {e}");
        }
        self.life_sync_history(&agent); // phase 3 lifecycle: the history is the compacted window
        self.compaction_over(&agent);
    }

    /// The agent's compaction state after one ended (`ok`: it did; `limit`: a rollover's), and the report's context after.
    fn record_compaction(&mut self, agent: &str, report: &mut Report, ok: bool, limit: bool) {
        let ctx = self.engine.agent_context_tokens(agent);
        if let Some(c) = ctx {
            report.context_after = c;
        }
        if let Some(ag) = self.agents.get_mut(agent) {
            ag.comp.context_at_last = ctx.unwrap_or(ag.comp.context_at_last);
            ag.comp.archived_to = ag.comp.context_at_last;
            if ok {
                ag.comp.compactions += 1;
                ag.comp.rollovers += limit as u64;
                ag.comp.last = Some(report.clone());
            }
        }
    }

    /// After a turn of `aid` ended: compact if its policy says so.
    fn policy_check(&mut self, aid: &str) {
        let Some(a) = self.agents.get(aid) else { return };
        let Some(p) = a.comp.policy else { return };
        if a.busy.is_some() || !a.comp.phase.is_idle() {
            return;
        }
        let ctx = self.engine.agent_context_tokens(aid).unwrap_or(0);
        let since = ctx.saturating_sub(a.comp.context_at_last);
        let Some(trigger) = p.due(ctx, since) else { return };
        match self.begin_compaction(aid, Spec::from_policy(&p, trigger), None) {
            Ok(()) => {}
            Err((e, _)) => {
                let a = self.agents.get_mut(aid).expect("checked above");
                a.comp.skipped += 1;
                if trigger == Trigger::Every {
                    a.comp.context_at_last = ctx; // count again from here, not after every turn
                }
                if e.status != 422 {
                    eprintln!("sllm-server: compaction of {aid} ({trigger:?}) refused: {e}");
                }
            }
        }
    }
}

impl InFlight {
    /// The compaction request's events so far: its end (done or failed), if it has come.
    fn end(&mut self) -> Option<Result<Done, EngineError>> {
        let mut end = None;
        while let Ok(ev) = self.rx.try_recv() {
            match ev {
                ReqEvent::Done(d) => end = Some(Ok(d)),
                ReqEvent::Error(e) => end = Some(Err(e)),
                _ => {}
            }
        }
        end
    }
}
