//! A step's output rows taken: each token checked against the request (grammar, `ignore_eos`), streamed or the request ended.

use super::*;

/// What a step that ran hands back: each output row's token, the output rows' hidden states (when the CPU makes the
/// latent vectors; else empty), and the soft words made for the step's silent rows (by output row).
pub(super) struct StepOut<'a> {
    pub(super) out: &'a [i32],
    pub(super) hid: &'a [f32],
    pub(super) pre: HashMap<usize, Vec<f32>>,
}

impl NextEngine {
    /// The token of output row `k` (the GPU's argmax `t`), checked against the request's grammar and `ignore_eos`: only a
    /// refused argmax brings the row's scores to the CPU.
    ///
    /// A request with `n_probs` also gets the token's [`Rec`] from the GPU's record of the row (`self.probs`, made by the step that
    /// asked for it); a token the CPU picked instead takes its score from the fetched row and the record's log-sum-exp.
    pub(super) fn sample(&mut self, req: ReqId, k: usize, t: i32) -> Result<(i32, Option<Rec>), String> {
        let Some(job) = self.jobs.get_mut(&req) else { return Ok((t, None)) };
        let w: Option<[u32; 8]> = if job.n_probs > 0 {
            self.probs.get(k * 8..k * 8 + 8).map(|s| std::array::from_fn(|i| s[i]))
        } else {
            None
        };
        let eos_banned = job.ignore_eos && self.vocab.is_eog(t);
        let ok = !eos_banned && job.gram.as_mut().is_none_or(|g| g.allows(t));
        if ok {
            return Ok((t, w.map(|w| Rec::new(&w, f32::from_bits(w[1])))));
        }
        let mut row = vec![0f32; self.n_vocab];
        call(|b, n| unsafe { sn_logits(self.e, k as c_int, row.as_mut_ptr(), b, n) })?;
        let eog = self.vocab.eog_ids();
        let t = match job.gram.as_mut() {
            Some(g) => g.pick(&row, job.ignore_eos, eog),
            None => argmax_excluding(&row, eog),
        };
        let rec = w.and_then(|w| row.get(usize::try_from(t).ok()?).map(|&x| Rec::new(&w, x)));
        Ok((t, rec))
    }

    /// A sampled token for `req`: stream it or end the request. True when the request ended.
    pub(super) fn on_token(&mut self, req: ReqId, t: i32, rec: Option<Rec>, events: &mut Vec<EngineEvent>) -> bool {
        let Some(job) = self.jobs.get_mut(&req) else { return true };
        let eos = job.ends_at(t, &self.vocab);
        if !job.grammar_takes(t, eos) {
            events.push(EngineEvent::Failed { req, message: format!("the grammar refused token {t} (a sampler bug)") });
            self.end(req);
            return true;
        }
        let pend = if eos || job.budget == 0 { None } else { Some(t) };
        if let Some(a) = job.agent.as_ref().and_then(|id| self.agents.get_mut(id)) {
            a.pending = pend.into_iter().collect();
        }
        let finished = if eos {
            events.push(EngineEvent::Finished { req, reason: StopReason::Eos });
            true
        } else if job.budget == 0 {
            events.push(EngineEvent::Finished { req, reason: StopReason::Limit });
            true
        } else {
            self.counters.tokens_generated += 1;
            let text = job.count_token(t, &self.vocab);
            let probs = rec.map(|r| r.probs(t, &text, job.n_probs, &self.vocab, self.n_vocab));
            events.push(EngineEvent::Token { req, id: t, text, probs });
            job.budget_spent(req, events)
        };
        if finished {
            self.end(req);
        }
        finished
    }

    pub(super) fn end(&mut self, req: ReqId) {
        if let Some(job) = self.jobs.remove(&req) {
            if self.recurrent {
                unsafe { sn_draft_rows(self.e, job.slot, 0) };  // its spare state entries (drafts) go back to the pool
            }
            self.share_seen(job.slot, job.seen_from);
            if job.agent.is_none() {
                self.slot_back(job.slot, SlotOwner::Job(req)); // its cells stay: a prefix for the next sequence
                self.last_used[job.slot as usize] = self.tick;
                self.cap_kept_states();
            }
        }
        if std::env::var_os("SLLM_LOG_TRIM").is_some() {
            eprintln!("next: end, {} jobs left", self.jobs.len());
        }
        if self.jobs.is_empty() {
            self.give_back(0);
            self.idle_since = Some(std::time::Instant::now());
        }
    }

    /// The tokens a sequence holds from cell `from` on (what the shared draft memory has not seen yet) go into that memory.
    fn share_seen(&mut self, slot: i32, from: usize) {
        let Some(sh) = self.shared.as_mut() else { return };
        if let Some(new) = self.cells[slot as usize].get(from..) {
            for run in new.split(|&t| t < 0).filter(|r| !r.is_empty()) {  // silent cells are no tokens
                sh.add(run);
            }
        }
    }

    /// The writers' output rows of a step that ran: a silent row moves its latent turn on; a writer's rows are sampled in
    /// order, its drafts kept while they are the tokens it takes (the cells of the rest cleared).
    pub(super) fn take_writes(&mut self, groups: &[(ReqId, usize, usize)], drafts: &[Vec<i32>], step: &mut StepOut, events: &mut Vec<EngineEvent>) {
        for (&(r, k0, k), d) in groups.iter().zip(drafts) {
            let Some(p0) = self.jobs.get(&r).map(|j| j.pos) else { continue };
            let slot = self.jobs[&r].slot;
            if self.jobs[&r].lat > 0 {
                self.take_silent(r, k0, step, events);
                continue;
            }
            let fed = self.take_writer_rows(r, k0, k, d, step.out, events);
            if k > 0 {
                self.bg_drafted += k as u64;
                self.bg_kept += (fed - 1) as u64;
                let _ = self.truncate(slot, p0 + fed);
            }
            if let Some(j) = self.jobs.get_mut(&r) {
                j.pos = p0 + fed;
                if k > 0 {
                    j.spec.record(k, fed - 1, &self.spec);
                }
            }
        }
    }

    /// A silent step of a latent turn (output row `k0`): its cell gets a marker of its own, its hidden state is the next
    /// step's vector; the turn ends when its steps are done, or at a best word that stops it (SLLM_LATENT_STOP).
    fn take_silent(&mut self, r: ReqId, k0: usize, step: &mut StepOut, events: &mut Vec<EngineEvent>) {
        let (slot, p0) = (self.jobs[&r].slot, self.jobs[&r].pos);
        // a silent step: its cell gets a marker of its own, its hidden state is the next step's vector
        self.cells[slot as usize][p0] = self.lat_mark;
        self.lat_mark = self.lat_mark.checked_sub(1).unwrap_or(-1);
        self.lat_rows += 1;
        let v = self.silent_input(k0, step);
        let best = step.out[k0];
        let stop = !self.lat_stop.is_empty() && (self.lat_stop.contains(&best) || self.vocab.is_eog(best));
        let j = self.jobs.get_mut(&r).expect("checked above");
        j.pos = p0 + 1;
        j.lat = if stop { 0 } else { j.lat - 1 };
        j.lvec = v;
        if j.lat == 0 {
            events.push(EngineEvent::Finished { req: r, reason: StopReason::Limit });
            self.end(r);
        }
    }

    /// The next silent input of output row `k`: none when soft words are made on the GPU (the row reads it there), else the
    /// soft words made for the whole step, or one made from this row alone.
    fn silent_input(&self, k: usize, step: &mut StepOut) -> Vec<f32> {
        let e_n = self.n_embd;
        if self.soft_gpu { Vec::new() } else { step.pre.remove(&k).unwrap_or_else(|| self.latent_input(k, &step.hid[k * e_n..(k + 1) * e_n])) }
    }

    /// One writer's rows `k0..=k0 + k` in order: each kept while its draft `d` is the token the writer takes. Returns the rows
    /// that are the writer's own tokens.
    pub(super) fn take_writer_rows(&mut self, r: ReqId, k0: usize, k: usize, d: &[i32], out: &[i32], events: &mut Vec<EngineEvent>) -> usize {
        // the writer's rows in order: each kept while its draft is the token the writer takes
        let mut fed = 0;
        for i in 0..=k {
            fed = i + 1;
            let (t, rec) = match self.sample(r, k0 + i, out[k0 + i]) {
                Ok(s) => s,
                Err(message) => {
                    events.push(EngineEvent::Failed { req: r, message });
                    self.end(r);
                    break;
                }
            };
            if self.on_token(r, t, rec, events) || i == k || d[i] != t {
                break;
            }
        }
        fed
    }

    /// The output rows of prompts read to their end in a step that ran: a latent turn's first vector, or the first token.
    pub(super) fn take_prompts(&mut self, out_req: &[(ReqId, usize)], step: &mut StepOut, events: &mut Vec<EngineEvent>) {
        for &(r, k) in out_req {
            if self.jobs.get(&r).is_some_and(|j| j.lat > 0) {
                // the prompt is read: no token, the first silent step is fed this row's hidden state
                let v = self.silent_input(k, step);
                let j = self.jobs.get_mut(&r).expect("checked above");
                j.lvec = v;
                j.last = Some(0);
                continue;
            }
            match self.sample(r, k, step.out[k]) {
                Ok((t, rec)) => {
                    self.on_token(r, t, rec, events);
                }
                Err(message) => {
                    events.push(EngineEvent::Failed { req: r, message });
                    self.end(r);
                }
            }
        }
    }
}

impl Job {
    /// Whether token `t` ends the generation. GRAMMAR_DONE: the grammar is complete and `ignore_eos` forbids the
    /// end-of-generation token that would end it.
    fn ends_at(&self, t: i32, vocab: &Vocab) -> bool {
        let done = t == GRAMMAR_DONE;
        done || (!self.ignore_eos && vocab.is_eog(t))
    }

    /// Token `t` through the request's grammar (none, or `eos`: nothing to check). False when the grammar refuses it.
    fn grammar_takes(&mut self, t: i32, eos: bool) -> bool {
        let Some(g) = self.gram.as_mut().filter(|_| !eos) else { return true };
        if !g.allows(t) {
            return false;
        }
        g.accept(t);
        true
    }

    /// A written token counted: it is the job's last, and its text (a piece cut inside a character joins the next).
    fn count_token(&mut self, t: i32, vocab: &Vocab) -> String {
        self.emitted += 1;
        self.last = Some(t);
        text_of(vocab, t, &mut self.carry)
    }

    /// Whether the job has written all its budget allows (then its end is in `events`).
    fn budget_spent(&self, req: ReqId, events: &mut Vec<EngineEvent>) -> bool {
        if self.emitted < self.budget {
            return false;
        }
        events.push(EngineEvent::Finished { req, reason: if self.ctx_limited { StopReason::Context } else { StopReason::Limit } });
        true
    }
}

impl Rec {
    /// The token probabilities a request with `n_probs` streams with token `t` (`text`: what it streams for it), at most
    /// `n_probs` of the best tokens beside.
    fn probs(&self, t: i32, text: &str, n_probs: usize, vocab: &Vocab, n_vocab: usize) -> Box<TokProbs> {
        let piece = |id: i32, logprob: f32| {
            let bytes = vocab.piece(id);
            TokProb { id, token: String::from_utf8_lossy(&bytes).into_owned(), bytes, logprob }
        };
        let top = self.top.iter().take(n_probs).filter(|&&(id, _)| id >= 0 && (id as usize) < n_vocab).map(|&(id, lp)| piece(id, lp)).collect();
        // the sampled token's own entry carries the text the request streams for it (a piece cut inside a character joins the next)
        Box::new(TokProbs { chosen: TokProb { id: t, token: text.to_owned(), bytes: text.as_bytes().to_vec(), logprob: self.lp }, top })
    }
}
