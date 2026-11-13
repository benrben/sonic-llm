//! Drafts: prompt lookup and the guess head for a lone interactive writer, prompt lookup for background writers, and the
//! step that checks a writer's drafts.

use super::*;

impl NextEngine {
    /// Could this writer's next step draft? Then queueing it might waste the queued step: not while it sits out, nor when
    /// the next lookup cannot match (the context's last n - 1 tokens, the token fed now among them, occur nowhere before).
    pub(super) fn may_draft(&self, j: &Job, rows: usize) -> bool {
        self.spec.mode != SpecMode::Off
            && (j.interactive || (self.bg_draft && rows < self.row_cap(rows)))  // a background writer drafts only while the step has rows to spare
            && !j.spec.sitting_out()
            && j.last.is_some_and(|t| {
                SpecState::may_match_next(&self.cells[j.slot as usize], t, &self.spec)
                    || self.shared.as_ref().is_some_and(|s| s.may_match_next(&self.cells[j.slot as usize], t))
            })
    }

    /// The prompt-lookup draft of a lone interactive writer for this step (None: a plain step), at most what its budget
    /// and the window leave.
    pub(super) fn draft(&mut self, r: ReqId) -> Option<Vec<i32>> {
        let job = self.jobs.get(&r)?;
        if !job.interactive || job.gram.is_some() || job.lat > 0 {
            return None;
        }
        if self.guess {
            // the guess head: the token after `last`, then the tokens after its own guesses, in one call
            let (slot, pos, last) = (job.slot, job.pos, job.last?);
            let left = job.budget.saturating_sub(job.emitted + 1);
            return self.guess_draft(slot, pos, last, left);
        }
        self.lookup_draft(r)
    }

    /// The guess head's draft of the writer in `slot` (its token `last` at `pos`): `guess_k` tokens, at most `left` (what its
    /// budget leaves) and what the window leaves.
    fn guess_draft(&mut self, slot: i32, pos: usize, last: i32, left: usize) -> Option<Vec<i32>> {
        let n_ctx = self.cfg.n_ctx;
        let mut k = self.guess_k.min(left).min(n_ctx.saturating_sub(pos + 2));
        if self.recurrent && k > 0 {
            k = k.min(unsafe { sn_draft_rows(self.e, slot, k as c_int) }.max(0) as usize);
        }
        let mut d = vec![0i32; k];
        let e = self.e;
        if k == 0 || call(|b, n| unsafe { sn_guess(e, slot, last, pos as c_int, k as c_int, d.as_mut_ptr(), b, n) }).is_err() {
            return None;
        }
        self.gpu_ms += unsafe { sn_last_gpu_ms(self.e) };
        let vocab = &self.vocab;
        d.truncate(d.iter().position(|&g| vocab.is_eog(g)).unwrap_or(k));  // nothing is written after an end token: no row for it
        self.draft_guess = !d.is_empty();
        (!d.is_empty()).then_some(d)
    }

    /// Prompt lookup's draft of the writer of `r`: from its own context, or from the shared memory when that match is longer.
    fn lookup_draft(&mut self, r: ReqId) -> Option<Vec<i32>> {
        let cfg = self.spec.clone();
        let n_ctx = self.cfg.n_ctx;
        let job = self.jobs.get_mut(&r)?;
        let mut ctx = self.cells[job.slot as usize].clone();
        ctx.push(job.last?);
        if self.wide {
            job.spec.start_at(2);  // the cheap one-row-kernel check first; longer drafts once drafts are kept whole
        }
        let out = job.spec.sitting_out();
        let mut d = job.spec.propose(&ctx, &cfg);
        self.draft_shared = false;
        if let Some(sh) = self.shared.as_mut().filter(|_| !out) {
            // the shared memory's match counts when it is longer than the context's own
            let own = job.spec.last_end.map_or(0, |e| (0..32.min(e + 1)).take_while(|&l| ctx[e - l] == ctx[ctx.len() - 1 - l]).count());
            if let Some((_, ds)) = sh.propose(&ctx, job.spec.k_now(&cfg)).filter(|(l, _)| *l > own) {
                d = ds;
                self.draft_shared = true;
            }
        }
        d.truncate(d.iter().position(|&t| t < 0).unwrap_or(d.len()));  // a silent cell's marker is no token
        job.no_match = if d.is_empty() { job.no_match + 1 } else { 0 };
        d.truncate(cfg.k.min(job.budget.saturating_sub(job.emitted + 1)).min(n_ctx.saturating_sub(job.pos + 2)));
        if self.recurrent && !d.is_empty() {
            // a recurrent model: as many drafts as the sequence holds spare state entries for (the way back from a refused one)
            d.truncate(unsafe { sn_draft_rows(self.e, job.slot, d.len() as c_int) }.max(0) as usize);
        }
        (!d.is_empty()).then_some(d)
    }

    /// The prompt-lookup drafts of the background writers `writes[from..]` of a step of only write rows (one per writer, `rows`
    /// of them), as many rows as the step has to spare (`bg_rows` in all): one list per write, empty for the interactive ones
    /// and for a writer that sits out, has no match or may not draft.
    /// The most rows a step of `writers` background writers carries with their drafts: a small step stays small.
    pub(super) fn row_cap(&self, writers: usize) -> usize {
        if self.small > 0 && writers <= self.small { self.small } else { self.bg_rows }
    }

    pub(super) fn bg_drafts(&mut self, writes: &[(ReqId, bool)], from: usize) -> Vec<Vec<i32>> {
        let mut out = vec![Vec::new(); writes.len()];
        let spare = self.row_cap(writes.len()).saturating_sub(writes.len()).min(self.cfg.max_rows.saturating_sub(writes.len()));
        if spare == 0 {
            return out;
        }
        let cfg = self.spec.clone();
        let n_ctx = self.cfg.n_ctx;
        for (i, &(r, _)) in writes.iter().enumerate().skip(from) {
            let Some(job) = self.jobs.get_mut(&r) else { continue };
            if job.gram.is_some() || job.lat > 0 {
                continue;
            }
            let Some(last) = job.last else { continue };
            let mut ctx = self.cells[job.slot as usize].clone();
            ctx.push(last);
            job.spec.start_at(self.bg_k0);
            let sits = job.spec.sitting_out();
            let mut d = job.spec.propose(&ctx, &cfg);
            if let Some(sh) = self.shared.as_mut().filter(|_| !sits) {  // the shared memory's match when longer, as in draft()
                let own = job.spec.last_end.map_or(0, |e| (0..32.min(e + 1)).take_while(|&l| ctx[e - l] == ctx[ctx.len() - 1 - l]).count());
                if let Some((_, ds)) = sh.propose(&ctx, job.spec.k_now(&cfg)).filter(|(l, _)| *l > own) {
                    d = ds;
                }
            }
            d.truncate(d.iter().position(|&t| t < 0).unwrap_or(d.len()));  // a silent cell's marker is no token
            d.truncate(cfg.k.min(job.budget.saturating_sub(job.emitted + 1)).min(n_ctx.saturating_sub(job.pos + 2)));
            out[i] = d;
        }
        // too many for the rows left: the longest lists give up a token at a time
        let mut total: usize = out.iter().map(Vec::len).sum();
        while total > spare {
            let i = (0..out.len()).max_by_key(|&i| (out[i].len(), std::cmp::Reverse(i))).expect("some draft");
            out[i].pop();
            total -= 1;
        }
        out
    }

    /// One step of the writer's pending token and its drafts: each row is checked in order, kept while the draft is the
    /// token the writer takes; the cells of the rest are cleared.
    pub(super) fn spec_step(&mut self, r: ReqId, drafts: &[i32], mut events: Vec<EngineEvent>) -> Vec<EngineEvent> {
        let j = &self.jobs[&r];
        let (slot, p0) = (j.slot, j.pos);
        let k = drafts.len();
        let tok: Vec<i32> = std::iter::once(j.last.unwrap()).chain(drafts.iter().copied()).collect();
        let seq = vec![slot; k + 1];
        let pos: Vec<i32> = (0..=k).map(|i| (p0 + i) as i32).collect();
        let rows: Vec<i32> = (0..=k as i32).collect();
        let fast = if self.wide && k + 1 >= 4 { -(k as i32 + 1) } else { k as i32 + 1 };  // a wide check: the block matmuls (sn_step n_fast < 0)
        self.want_probs = j.n_probs > 0;
        let out = match self.run(&tok, &seq, &pos, k + 1, fast, &rows, false) {
            Ok(o) => o,
            Err(message) => {
                self.reset_after_failed_step(&seq, &pos);
                events.push(EngineEvent::Failed { req: r, message });
                self.end(r);
                return events;
            }
        };
        self.spec_steps += 1;
        self.spec_drafted += k as u64;
        let fed = self.take_writer_rows(r, 0, k, drafts, &out, &mut events);  // rows that are the writer's own tokens
        let kept = fed - 1;
        self.spec_kept += kept as u64;
        let guessed = std::mem::take(&mut self.draft_guess);
        if guessed {
            self.guess_steps += 1;
            self.guess_drafted += k as u64;
            self.guess_kept += kept as u64;
        }
        if std::mem::take(&mut self.draft_shared) {
            self.shared_steps += 1;
            self.shared_drafted += k as u64;
            self.shared_kept += kept as u64;
        }
        let _ = self.truncate(slot, p0 + fed);
        if let Some(j) = self.jobs.get_mut(&r) {
            j.pos = p0 + fed;
            if !guessed {
                j.spec.record(k, kept, &self.spec);  // (prompt lookup's own pacing: not for the guess head's drafts)
            }
        }
        events
    }
}
