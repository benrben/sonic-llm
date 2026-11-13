//! The parts of a step (`Engine::step`): its start, the work items resolved into rows, the order of the rows, and the
//! decode of the rows in groups that fit the batch.

use super::*;

/// A step's work items resolved ([`LlamaEngine::resolve_items`]).
pub(super) struct StepWork {
    /// The rows of the main context's decode.
    pub(super) rows: Vec<Row>,
    /// Prompt chunks a reader reads: (request, tokens, the chunk samples).
    pub(super) reads: Vec<(ReqId, Vec<i32>, bool)>,
    /// Streams copied to a warm holder before the decode: (sequence, the tokens it holds).
    pub(super) warm: Vec<(i32, Vec<i32>)>,
}

/// What a work item's job says about it, read before its sequence is looked at.
struct ItemJob {
    req: ReqId,
    work: Work,
    ignore_eos: bool,
    gram: bool,
    /// Its prompt is read in a reader.
    to_reader: bool,
    /// Holder models: what it reads is a prompt others may repeat.
    copy: bool,
    /// The prefill chunk ([`Job::next_chunk`]).
    chunk: Option<(Vec<i32>, bool)>,
}

impl ItemJob {
    /// The item for the sequence `seq`: a row of the decode, or a chunk for its reader; holder models copy a long prompt's
    /// stream to a warm holder before its last chunk.
    fn route(self, seq: &Seq, share_align: usize, work: &mut StepWork, events: &mut Vec<EngineEvent>) {
        let (sid, pos0) = (seq.sid, seq.tokens.len());
        let (req, ignore_eos, gram) = (self.req, self.ignore_eos, self.gram);
        match self.work {
            Work::Prefill { .. } => {
                let Some((toks, samples)) = self.chunk else { return };
                if self.to_reader {
                    work.reads.push((req, toks, samples));
                    return;
                }
                if samples && self.copy && pos0 >= WARM_COPY_MIN && pos0 % share_align == 0 {
                    work.warm.push((sid, seq.tokens.clone()));
                }
                work.rows.push(Row { req, sid, pos0, toks: Toks::Many(toks), samples, prefill: true, ignore_eos, gram, n_draft: 0 });
            }
            Work::Decode => {
                let Some(tok) = seq.pending else {
                    events.push(EngineEvent::Failed { req, message: "decode without a pending token".into() });
                    return;
                };
                work.rows.push(Row { req, sid, pos0, toks: Toks::One(tok), samples: true, prefill: false, ignore_eos, gram, n_draft: 0 });
            }
        }
    }
}

impl Job {
    /// The next prefill chunk: at most `n` tokens of the feed (and the batch's capacity), and whether it ends the feed (it
    /// samples the first token). None when nothing is left to feed.
    fn next_chunk(&self, n: usize, batch_cap: usize) -> Option<(Vec<i32>, bool)> {
        let (fed, feed_len) = (self.fed, self.feed.len());
        let n = n.min(feed_len - fed.min(feed_len)).min(batch_cap);
        (n > 0).then(|| (self.feed[fed..fed + n].to_vec(), fed + n == feed_len))
    }
}

/// The end of the decode group that starts at row `start`: the rows whose tokens fit the batch's `cap` (one row at least).
fn group_end(rows: &[Row], start: usize, cap: usize) -> usize {
    let (mut end, mut n) = (start, 0usize);
    while end < rows.len() && (n + rows[end].toks.len() <= cap || end == start) {
        n += rows[end].toks.len();
        end += 1;
    }
    end
}

impl LlamaEngine {
    /// The start of a step: the I/O thread's completions, paused requests coming back, failures to report, conversions that
    /// ended, and the feed of every request whose prefill starts.
    pub(super) fn step_start(&mut self, items: &[WorkItem], events: &mut Vec<EngineEvent>) {
        self.counters.steps += 1;
        self.last_bd = StepBreakdown::default();
        self.poll_io();
        self.resume_paused();
        for (req, message) in std::mem::take(&mut self.job_failures) {
            events.push(EngineEvent::Failed { req, message });
        }
        self.poll_handovers(events);
        for w in items {
            if matches!(w.work, Work::Prefill { .. }) {
                self.begin_prefill(w.req);
            }
        }
    }

    /// The step's work items as rows of the decode, chunks for the readers and warm copies.
    pub(super) fn resolve_items(&mut self, items: &[WorkItem], events: &mut Vec<EngineEvent>) -> StepWork {
        let mut work = StepWork { rows: Vec::with_capacity(items.len()), reads: Vec::new(), warm: Vec::new() };
        let batch_cap = self.batch.cap;
        for w in items {
            self.resolve_item(w, batch_cap, &mut work, events);
        }
        work
    }

    fn resolve_item(&mut self, w: &WorkItem, batch_cap: usize, work: &mut StepWork, events: &mut Vec<EngineEvent>) {
        let Some(job) = self.jobs.get_mut(&w.req) else { return };
        if job.paused || job.handover {
            return; // waits for room, or for its conversion: no row, no event
        }
        let item = ItemJob {
            req: w.req,
            work: w.work,
            ignore_eos: job.params.ignore_eos,
            gram: job.gram.is_some(),
            to_reader: job.read == Some(true) && !self.readers.is_empty(),
            // a completion, or an agent's first turn (its context was empty): what it reads is a prompt others may repeat
            copy: self.holders_on && job.params.cache_prompt && (job.agent.is_none() || job.len_before == 0),
            chunk: match w.work {
                Work::Prefill { n } => job.next_chunk(n, batch_cap),
                Work::Decode => None,
            },
        };
        let Some(seq) = Self::seq_for(&mut self.agents, job) else {
            events.push(EngineEvent::Failed { req: w.req, message: "the request's sequence is gone".into() });
            return;
        };
        item.route(seq, self.share_align, work, events);
    }

    /// Everything runnable is paused (no rows): wait for the I/O or a conversion.
    pub(super) fn wait_if_all_paused(&self, rows: &[Row]) {
        if rows.is_empty() && (self.jobs.values().any(|j| j.paused) || !self.handovers.is_empty()) {
            std::thread::sleep(std::time::Duration::from_millis(1)); // everything runnable is paused: wait for the I/O or a conversion
        }
    }

    /// Per-sequence streams: libllama builds one micro-batch from consecutive, increasing sequence ids; any other order
    /// splits the step into several graphs. The rows go in id order (packed into the lowest ids), and the runs are counted.
    pub(super) fn order_rows(&mut self, rows: &mut [Row]) {
        if self.unified {
            return;
        }
        rows.sort_by_key(|r| r.sid);
        if self.seq_pack {
            self.pack_rows(rows);
        }
        if !rows.is_empty() {
            self.count_runs(rows);
        }
    }

    /// The runs of consecutive sequence ids in the step's sorted rows (`stream_runs` in the engine metrics).
    fn count_runs(&mut self, rows: &[Row]) {
        let runs = 1 + rows.windows(2).filter(|w| w[1].sid != w[0].sid + 1).count() as u64;
        let sr = &mut self.stream_runs;
        sr.steps += 1;
        sr.runs += runs;
        sr.split_steps += (runs > 1) as u64;
        sr.max_runs = sr.max_runs.max(runs);
        if rows.len() >= 8 {
            sr.wide_steps += 1;
            sr.wide_runs += runs;
        }
    }

    /// Decode the rows in groups that fit the batch; returns the sampled tokens in row order.
    pub(super) fn decode_groups(&mut self, rows: &[Row], events: &mut Vec<EngineEvent>) -> Vec<(ReqId, i32)> {
        let mut sampled: Vec<(ReqId, i32)> = Vec::new();
        let cap = self.batch.cap;
        let mut start = 0;
        while start < rows.len() {
            let end = group_end(rows, start, cap);
            let group = &rows[start..end];
            let mut group_sampled = Vec::new();
            let before = events.len();
            self.decode_rows(group, &mut group_sampled, events);
            let failed: HashSet<ReqId> = events[before..].iter().filter_map(|e| if let EngineEvent::Failed { req, .. } = e { Some(*req) } else { None }).collect();
            // the rows that ran: their tokens are in the cache now
            for r in group.iter().filter(|r| !failed.contains(&r.req)) {
                self.book_row(r);
            }
            sampled.extend(group_sampled);
            start = end;
        }
        sampled
    }

    /// A decoded row's tokens are in its sequence's cache now.
    fn book_row(&mut self, r: &Row) {
        if r.n_draft > 0 {
            return; // finish_spec books the tokens that stay
        }
        let Some(job) = self.jobs.get_mut(&r.req) else { return };
        if r.prefill {
            job.fed += r.toks.len();
        }
        let Some(seq) = Self::seq_for(&mut self.agents, job) else { return };
        seq.tokens.extend_from_slice(r.toks.as_slice());
        if self.share {
            self.ptree.borrow_mut().extend(seq.sid, r.toks.as_slice());
        }
        if !r.prefill {
            seq.pending = None; // the fed token left `pending`; on_token sets the next one
        }
    }
}
