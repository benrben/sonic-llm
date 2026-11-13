//! One step: the rows the jobs can take (write rows, then prompt rows), run as one `sn_step`.

use super::take::StepOut;
use super::*;

/// The latent turns' rows of a step (see [`NextEngine::latent_rows`]).
#[derive(Default)]
struct LatentRows {
    lrows: Vec<i32>,
    lvecs: Vec<f32>,
    srows: Vec<i32>,
    souts: Vec<i32>,
}

impl NextEngine {
    /// [`Engine::step`]: one step of the work the scheduler gave.
    pub(super) fn run_step(&mut self, items: &[WorkItem]) -> Vec<EngineEvent> {
        let mut events = Vec::new();
        // write rows (interactive first), then prompt rows
        let (writes, mut prompts) = self.step_work(items);
        // the first MAX_FAST interactive rows take the one-row kernels; more interactive rows ride along as ordinary rows
        // (Ben 2026-11-09: no cap of 4 chats at once)
        let n_fast = writes.iter().take_while(|w| w.1).count().min(MAX_FAST);
        // a recurrent model's prompt that reaches CK_LAG tokens from its end in this step: its rows up to there as a step of
        // their own, then its state kept there (sn_mark); the next turn's prompt (the reply rendered again) usually parts from
        // this one only in those last tokens. The rest of its grant goes on in this step as usual.
        if self.recurrent {
            self.cut_steps(&mut prompts, &mut events);
        }
        let drafts = self.step_drafts(&writes, &prompts, n_fast);
        let mut rows = StepRows::default();
        self.write_rows(&writes, &drafts, &mut rows);
        let n_write = rows.tok.len();
        let drafted: usize = drafts.iter().map(Vec::len).sum();
        if let Some((r, drafts)) = self.lone_writer_draft(&writes, &prompts, n_fast) {
            return self.spec_step(r, &drafts, events);
        }
        self.prompt_rows(&prompts, &mut rows);
        if rows.tok.is_empty() {
            return events;
        }
        if rows.tok.len() > self.cfg.max_rows {
            self.fail_oversized(&writes, &prompts, rows.tok.len(), &mut events);
            return events;
        }
        let chain = self.may_chain(&writes, &prompts, drafted, n_fast);
        let (res, hid) = self.run_rows(&rows, n_write, n_fast, chain);
        if drafted > 0 {
            self.bg_steps += 1;
        }
        self.advance_prompts(&prompts, res.is_ok());
        self.take_step(res, &rows, &drafts, &hid, &mut events);
        if !self.jobs.is_empty() && self.trim_at.elapsed() >= TRIM_EVERY {
            self.give_back(2); // a busy moment passed while others still run (two chunks spare: no map / unmap churn)
        }
        events
    }

    /// The background writers' drafts of this step (SLLM_NEXT_BG_DRAFT=0: none): only in a step of write rows only, none of
    /// them a chat's (its rows would pay for theirs: chat beside 8 background agents 42 -> 39 tok/s); each writer's rows then
    /// follow each other. One list per write.
    fn step_drafts(&mut self, writes: &[(ReqId, bool)], prompts: &[(ReqId, usize)], n_fast: usize) -> Vec<Vec<i32>> {
        if self.bg_draft && self.spec.mode != SpecMode::Off && prompts.is_empty() && n_fast == 0 && !writes.is_empty() {
            self.bg_drafts(writes, n_fast)
        } else {
            vec![Vec::new(); writes.len()]
        }
    }

    /// One interactive writer alone: its prompt-lookup (or guess head) draft, when it has one.
    fn lone_writer_draft(&mut self, writes: &[(ReqId, bool)], prompts: &[(ReqId, usize)], n_fast: usize) -> Option<(ReqId, Vec<i32>)> {
        if self.spec.mode != SpecMode::Off && prompts.is_empty() && writes.len() == 1 && n_fast == 1 {
            let r = writes[0].0;
            return self.draft(r).map(|d| (r, d));
        }
        None
    }

    /// A step of more rows than the library takes: every request in it fails.
    fn fail_oversized(&mut self, writes: &[(ReqId, bool)], prompts: &[(ReqId, usize)], n_rows: usize, events: &mut Vec<EngineEvent>) {
        for r in writes.iter().map(|w| w.0).chain(prompts.iter().map(|p| p.0)) {
            events.push(EngineEvent::Failed { req: r, message: format!("a step of {} rows is above --next-rows {}", n_rows, self.cfg.max_rows) });
            self.end(r);
        }
    }

    /// The step's output rows taken (writers, then prompts read to their end), or, when it failed, every request in it ended.
    fn take_step(&mut self, res: Result<Vec<i32>, String>, rows: &StepRows, drafts: &[Vec<i32>], hid: &[f32], events: &mut Vec<EngineEvent>) {
        match res {
            Ok(out) => {
                // soft words for every silent row of this step at once
                let mut step = StepOut { out: &out, hid, pre: self.soft_pre(rows) };
                self.take_writes(&rows.groups, drafts, &mut step, events);
                self.take_prompts(&rows.out_req, &mut step, events);
            }
            Err(message) => {
                self.reset_after_failed_step(&rows.seq, &rows.pos);
                for r in rows.groups.iter().map(|g| g.0).chain(rows.out_req.iter().map(|o| o.0)) {
                    events.push(EngineEvent::Failed { req: r, message: message.clone() });
                    self.end(r);
                }
            }
        }
    }

    /// One step. `chain`: the same rows are expected next, fed these tokens (the library queues that step behind this one).
    pub(super) fn run(&mut self, tok: &[i32], seq: &[i32], pos: &[i32], n_write: usize, n_fast: i32, out_rows: &[i32], chain: bool) -> Result<Vec<i32>, String> {
        let mut out = vec![0i32; out_rows.len()];
        let want = std::mem::take(&mut self.want_probs) && !out_rows.is_empty();
        unsafe { sn_want_probs(self.e, want as c_int) };
        let t0 = std::time::Instant::now();
        call(|b, n| unsafe {
            sn_step(self.e, tok.len() as c_int, tok.as_ptr(), seq.as_ptr(), pos.as_ptr(), n_write as c_int, n_fast as c_int, out_rows.len() as c_int,
                    out_rows.as_ptr(), out.as_mut_ptr(), chain as c_int, b, n)
        })?;
        self.run_ms += t0.elapsed().as_secs_f64() * 1e3;
        self.probs.clear();
        if want {
            self.probs.resize(out_rows.len() * 8, 0);
            call(|b, n| unsafe { sn_probs(self.e, out_rows.len() as c_int, self.probs.as_mut_ptr().cast::<f32>(), b, n) })?;
        }
        self.gpu_ms += unsafe { sn_last_gpu_ms(self.e) };
        self.counters.steps += 1;
        self.tick += 1;
        for &q in seq {
            self.last_used[q as usize] = self.tick;
        }
        let used = self.pages_used();
        self.pages_peak = self.pages_peak.max(used);
        for k in 0..tok.len() {
            let c = &mut self.cells[seq[k] as usize];
            c.truncate(pos[k] as usize);
            c.push(tok[k]);
        }
        Ok(out)
    }

    /// The work of a step that the jobs can take: write rows (interactive first) and prompt grants.
    pub(super) fn step_work(&self, items: &[WorkItem]) -> (Vec<(ReqId, bool)>, Vec<(ReqId, usize)>) {
        let mut writes: Vec<(ReqId, bool)> = Vec::new();
        let mut prompts: Vec<(ReqId, usize)> = Vec::new();
        for w in items {
            let Some(job) = self.jobs.get(&w.req) else { continue };
            match w.work {
                Work::Decode if job.last.is_some() => writes.push((w.req, job.interactive)),
                Work::Prefill { n } if n > 0 && job.fed < job.feed.len() => prompts.push((w.req, n)),
                _ => {}
            }
        }
        writes.sort_by_key(|&(_, i)| !i);
        (writes, prompts)
    }

    /// A recurrent model's prompts that cross a pinned start or the checkpoint in this step: the rows up to there run as a
    /// step of their own (then the start is pinned, or the state marked), and the grant shrinks by them. A prompt whose step
    /// fails ends; prompts with nothing left drop out.
    pub(super) fn cut_steps(&mut self, prompts: &mut Vec<(ReqId, usize)>, events: &mut Vec<EngineEvent>) {
        for (r, n) in prompts.iter_mut() {
            // (a pinned start ends before the checkpoint: at most two cuts a prompt)
            for which in 0..2 {
                if !self.cut_step(*r, n, which == 0, events) {
                    break;
                }
            }
        }
        prompts.retain(|&(r, n)| n > 0 && self.jobs.contains_key(&r));
    }

    /// One cut of prompt `r` when its grant `n` crosses it (`pin`: its pinned start, else its checkpoint): the rows up to
    /// there run as a step of their own, then the start is pinned or the state marked, and the grant shrinks by them.
    /// False when the prompt is gone (it ended before, or its step failed and ended it).
    fn cut_step(&mut self, r: ReqId, n: &mut usize, pin: bool, events: &mut Vec<EngineEvent>) -> bool {
        let Some(j) = self.jobs.get(&r) else { return false };
        let cut = if pin { j.pin_at } else { j.ck_at };
        if cut == usize::MAX || j.fed >= cut || j.fed + *n <= cut {
            return true;
        }
        let (slot, from, at) = (j.slot, j.fed, j.pos);
        let toks: Vec<i32> = j.feed[from..cut].to_vec();
        let seqs = vec![slot; toks.len()];
        let poss: Vec<i32> = (0..toks.len()).map(|k| (at + k) as i32).collect();
        if let Err(e) = self.run(&toks, &seqs, &poss, 0, 0, &[], false) {
            self.reset_after_failed_step(&seqs, &poss);
            events.push(EngineEvent::Failed { req: r, message: e });
            self.end(r);
            *n = 0;
            return false;
        }
        self.counters.prompt_tokens_evaluated += toks.len() as u64;
        if pin {
            self.pin(slot, at + toks.len());
        } else {
            unsafe { sn_mark(self.e, slot) };
        }
        let j = self.jobs.get_mut(&r).expect("checked above");
        j.fed = cut;
        j.pos += toks.len();
        *n -= toks.len();
        true
    }

    /// The write rows of a step: each writer's pending token, then its drafts; each its own output row.
    pub(super) fn write_rows(&self, writes: &[(ReqId, bool)], drafts: &[Vec<i32>], rows: &mut StepRows) {
        let StepRows { tok, seq, pos, out_rows, groups, .. } = rows;
        for (&(r, _), d) in writes.iter().zip(drafts) {
            let j = &self.jobs[&r];
            groups.push((r, out_rows.len(), d.len()));
            for (i, &t) in std::iter::once(&j.last.unwrap()).chain(d).enumerate() {
                out_rows.push(tok.len() as i32);
                tok.push(t);
                seq.push(j.slot);
                pos.push((j.pos + i) as i32);
            }
        }
    }

    /// The prompt rows of a step (each grant, at most what is left of its prompt); a prompt read to its end in this step
    /// has its last row as an output row.
    pub(super) fn prompt_rows(&mut self, prompts: &[(ReqId, usize)], rows: &mut StepRows) {
        let StepRows { tok, seq, pos, out_rows, out_req, .. } = rows;
        for &(r, n) in prompts {
            let j = &self.jobs[&r];
            let take = n.min(j.feed.len() - j.fed);
            for k in 0..take {
                tok.push(j.feed[j.fed + k]);
                seq.push(j.slot);
                pos.push((j.pos + k) as i32);
            }
            if j.fed + take == j.feed.len() {
                out_req.push((r, out_rows.len()));
                out_rows.push(tok.len() as i32 - 1);
            }
            self.counters.prompt_tokens_evaluated += take as u64;
        }
    }

    /// Writers only, none whose token may not be the argmax (a grammar, an end token the GPU does not skip: the CPU would
    /// read scores that a queued step of more than 4 rows writes over), each going on after this token unless it ends:
    /// the next step is these rows again, queued now.
    pub(super) fn may_chain(&self, writes: &[(ReqId, bool)], prompts: &[(ReqId, usize)], drafted: usize, n_fast: usize) -> bool {
        // (a background writer beside a chat does not draft: no rows to spare then)
        let rows = if n_fast > 0 { usize::MAX } else { writes.len() };
        self.chain
            && prompts.is_empty()
            && drafted == 0
            && writes.iter().all(|(r, _)| self.jobs.get(r).is_some_and(|j| self.goes_on(j, rows)))
    }

    /// Whether a writer of a step of `rows` write rows goes on after this step exactly as queued: its token is the GPU's
    /// argmax, it does not end on its budget, and its next step does not draft.
    fn goes_on(&self, j: &Job, rows: usize) -> bool {
        // (a soft silent row on the GPU goes on silent while it has steps left after this one: no CPU input, no token, no draft;
        // when its best word ends the turn early, the step queued behind is dropped and its cell cut, as a written end token's)
        if j.lat > 0 {
            self.soft_gpu && j.lat >= 2
        } else {
            j.gram.is_none() && (self.gpu_eog || !j.ignore_eos) && j.emitted + 1 < j.budget && !self.may_draft(j, rows)
        }
    }

    /// Run the step's rows: latent turns' write rows are fed their vectors (or, soft words on the GPU, read them there), and
    /// the step hands back the hidden states of its output rows when the CPU makes the next vectors. Returns the output
    /// rows' tokens (or the reason the step failed) and those hidden states (empty when not wanted).
    pub(super) fn run_rows(&mut self, rows: &StepRows, n_write: usize, n_fast: usize, chain: bool) -> (Result<Vec<i32>, String>, Vec<f32>) {
        let StepRows { tok, seq, pos, out_rows, out_req, groups } = rows;
        // latent turns: their write rows are fed their vectors, and the step hands back the hidden states of its output rows
        // (theirs, and a latent turn's last prompt row: its first vector)
        let e_n = self.n_embd;
        let LatentRows { lrows, lvecs, srows, souts } = self.latent_rows(rows);
        let soft_set = if self.soft_gpu {
            call(|b, n| unsafe { sn_soft_rows(self.e, srows.len() as c_int, srows.as_ptr(), souts.len() as c_int, souts.as_ptr(), b, n) })
        } else {
            Ok(())
        };
        let want_hid = !self.soft_gpu && (!lrows.is_empty() || out_req.iter().any(|(r, _)| self.jobs[r].lat > 0));
        let mut hid = if want_hid { vec![0f32; out_rows.len() * e_n] } else { Vec::new() };
        // token probabilities: this step takes the argmax_p kernel when a row of it wants them
        self.want_probs = groups.iter().map(|g| g.0).chain(out_req.iter().map(|o| o.0)).any(|r| self.jobs.get(&r).is_some_and(|j| j.n_probs > 0));
        let res = if let Err(m) = soft_set {
            Err(m)
        } else if want_hid {
            call(|b, n| unsafe { sn_latent(self.e, lrows.len() as c_int, lrows.as_ptr(), lvecs.as_ptr(), hid.as_mut_ptr(), b, n) })
                .and_then(|_| self.run(tok, seq, pos, n_write, n_fast as i32, out_rows, false))
        } else {
            self.run(tok, seq, pos, n_write, n_fast as i32, out_rows, chain)
        };
        self.want_probs = false;  // (run takes it; a step that failed before it must not leave it for another)
        (res, hid)
    }

    /// The latent turns' rows of a step: write rows fed their vectors (`lrows`, the vectors in `lvecs`), or with soft words on
    /// the GPU, the silent write rows that read their vector there (`srows`) and the output rows that make the next (`souts`:
    /// theirs and the last row of a latent prompt).
    fn latent_rows(&self, rows: &StepRows) -> LatentRows {
        let mut l = LatentRows::default();
        for &(r, k0, _) in &rows.groups {
            let j = &self.jobs[&r];
            if j.lat > 0 {
                if self.soft_gpu {
                    l.srows.push(rows.out_rows[k0]);
                    l.souts.push(k0 as i32);
                } else {
                    l.lrows.push(rows.out_rows[k0]);
                    l.lvecs.extend_from_slice(&j.lvec);
                }
            }
        }
        if self.soft_gpu {
            l.souts.extend(rows.out_req.iter().filter(|(r, _)| self.jobs[r].lat > 0).map(|&(_, k)| k as i32));
        }
        l
    }

    /// The prompts move on by what this step read of them (whether or not it ran); a recurrent one whose step ended right at
    /// its checkpoint or its shared start has its state marked or the start pinned.
    pub(super) fn advance_prompts(&mut self, prompts: &[(ReqId, usize)], ran: bool) {
        let mut pin_now = Vec::new();
        for &(r, n) in prompts {
            if let Some(j) = self.jobs.get_mut(&r) {
                let take = n.min(j.feed.len() - j.fed);
                j.fed += take;
                j.pos += take;
                if self.recurrent && ran && j.fed == j.ck_at {
                    unsafe { sn_mark(self.e, j.slot) };  // this step ended right at the checkpoint
                }
                if self.recurrent && ran && j.fed == j.pin_at {
                    pin_now.push((j.slot, j.pos));  // this step ended right at a shared start
                }
            }
        }
        for (slot, n) in pin_now {
            self.pin(slot, n);
        }
    }
}
