//! The thread's loop: calls queued, admitted as jobs, polled, and their requests batched into steps.

use super::jobs::job;
use super::prepare::{prepare, Prepared};
use super::*;

pub(super) struct Queued {
    call: Box<Call>,
    prep: Prepared,
    need: usize,
    prio: bool,
    deadline: Option<Instant>,
}

pub(super) struct Active<'c> {
    fut: Pin<Box<dyn Future<Output = Res<Value>> + 'c>>,
    reply: oneshot::Sender<Result<Value, EngineError>>,
    need: usize,
    prio: bool,
    op: usize,
    arrived: Instant,
    began: Instant,
    deadline: Option<Instant>,
}

pub(super) struct Loop<'c> {
    pub(super) ctx: &'c Ctx,
    pub(super) rx: mpsc::Receiver<Msg>,
    pub(super) shared: Arc<HelperShared>,
    pub(super) queue: VecDeque<Queued>,
    pub(super) active: Vec<Active<'c>>,
    pub(super) inflight: Vec<Rc<Req>>,
    pub(super) next: u64,
}

/// What a step picked (request index, tokens) and the requests refused (index, reason).
#[derive(Default)]
struct Picks {
    picks: Vec<(usize, usize)>,
    refused: Vec<(usize, String)>,
}

impl Picks {
    /// Request `i` takes `n` tokens of the step when its sequence has the room; else it is refused. True when taken.
    fn take(&mut self, i: usize, n: usize, room: Result<usize, String>) -> bool {
        match room {
            Ok(room) if room >= n => {
                self.picks.push((i, n));
                true
            }
            Ok(_) => {
                self.refused.push((i, "helper sequence full".into()));
                false
            }
            Err(e) => {
                self.refused.push((i, e));
                false
            }
        }
    }
}

/// The prefill room left in a step: tokens, the ordinary calls' share, and the output rows it carries.
struct StepBudget {
    tokens: usize,
    ord_left: usize,
    out_rows: usize,
}

impl StepBudget {
    /// The tokens of request `r` this step reads, None when it reads none of them now.
    fn chunk(&mut self, r: &Req) -> Option<usize> {
        let avail = if r.prio { self.tokens } else { self.tokens.min(self.ord_left) };
        if avail == 0 {
            return None;
        }
        let rem = r.tokens.len() - r.done.get();
        if r.out == OutputPolicy::All {
            self.whole(rem, avail)
        } else {
            self.part(r, rem, avail)
        }
    }

    /// The whole item in one step, and the step's output rows bounded (the context's logits buffer).
    fn whole(&mut self, rem: usize, avail: usize) -> Option<usize> {
        if rem > avail || (self.out_rows > 0 && self.out_rows + rem > MAX_STEP_OUTPUTS) {
            return None;
        }
        self.out_rows += rem;
        Some(rem)
    }

    fn part(&mut self, r: &Req, rem: usize, avail: usize) -> Option<usize> {
        let n = rem.min(avail);
        if n == rem && r.out == OutputPolicy::Last {
            // the item's final chunk has one output row
            if self.out_rows + 1 > MAX_STEP_OUTPUTS {
                return None;
            }
            self.out_rows += 1;
        }
        Some(n)
    }
}

impl<'c> Loop<'c> {
    pub(super) fn run(&mut self) {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        loop {
            if self.active.is_empty() && self.queue.is_empty() {
                match self.rx.blocking_recv() {
                    None | Some(Msg::Shutdown) => break,
                    Some(Msg::Call(c)) => self.intake(c),
                }
            }
            if !self.take_messages() {
                break;
            }
            self.sweep();
            self.admit();
            self.poll(&mut cx);
            self.inflight.extend(self.ctx.submitted.borrow_mut().drain(..));
            // a request whose job is gone (only the loop still holds it) is dropped
            self.inflight.retain(|r| Rc::strong_count(r) > 1);
            if self.inflight.is_empty() {
                self.publish();
                self.idle_pause();
                continue;
            }
            self.step();
            self.publish();
        }
        for q in self.queue.drain(..) {
            let _ = q.call.reply.send(Err(EngineError::new(503, "unavailable_error", "the helper is shutting down")));
        }
        for a in self.active.drain(..) {
            let _ = a.reply.send(Err(EngineError::new(503, "unavailable_error", "the helper is shutting down")));
        }
        self.inflight.clear();
    }

    /// Takes every message waiting; false when the loop ends (a shutdown, or the senders gone with nothing left to do).
    fn take_messages(&mut self) -> bool {
        loop {
            match self.rx.try_recv() {
                Ok(Msg::Shutdown) => return false,
                Ok(Msg::Call(c)) => self.intake(c),
                Err(mpsc::error::TryRecvError::Empty) => return true,
                Err(mpsc::error::TryRecvError::Disconnected) => return !(self.active.is_empty() && self.queue.is_empty()),
            }
        }
    }

    /// No request to run: every live job waits on a request, so nothing to run means a job is stuck: do not spin.
    fn idle_pause(&self) {
        if !self.active.is_empty() {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn intake(&mut self, c: Box<Call>) {
        let prio = c.op.priority() == Priority::Compaction;
        if self.queue.len() >= self.ctx.cfg.queue_cap {
            self.ctx.stats.borrow_mut().deferred += 1;
            let _ = c.reply.send(Err(EngineError::new(503, "unavailable_error", "the helper queue is full").with_retry_after(100)));
            return;
        }
        match prepare(self.ctx, &c.op) {
            Err(e) => {
                self.ctx.stats.borrow_mut().failed[c.op.index()] += 1;
                let _ = c.reply.send(Err(e));
            }
            Ok((_, need)) if need > self.ctx.cfg.pool_pages => {
                self.ctx.stats.borrow_mut().failed[c.op.index()] += 1;
                let _ = c.reply.send(Err(EngineError::new(
                    400,
                    "exceed_context_size_error",
                    format!("the call needs {need} pages, more than the helper pool's {} (--helper-kv-pages)", self.ctx.cfg.pool_pages),
                )));
            }
            Ok((prep, need)) => {
                let deadline = c.op.deadline_ms().map(|ms| c.arrived + Duration::from_millis(ms));
                self.enqueue(Queued { call: c, prep, need, prio, deadline });
            }
        }
    }

    fn enqueue(&mut self, q: Queued) {
        if q.prio {
            // compaction calls go ahead of every ordinary call, in their own arrival order
            let at = self.queue.iter().position(|x| !x.prio).unwrap_or(self.queue.len());
            self.queue.insert(at, q);
        } else {
            self.queue.push_back(q);
        }
        let n = self.queue.len();
        let mut st = self.ctx.stats.borrow_mut();
        st.max_queued = st.max_queued.max(n);
    }

    fn sweep(&mut self) {
        let now = Instant::now();
        let mut st = self.ctx.stats.borrow_mut();
        self.queue.retain(|q| {
            if q.call.reply.is_closed() {
                st.cancelled += 1;
                return false;
            }
            true
        });
        let mut i = 0;
        while i < self.queue.len() {
            if self.queue[i].deadline.is_some_and(|d| now >= d) {
                let q = self.queue.remove(i).unwrap();
                st.expired += 1;
                let _ = q.call.reply.send(Err(deadline_err()));
            } else {
                i += 1;
            }
        }
        drop(st);
        let mut i = 0;
        while i < self.active.len() {
            let a = &self.active[i];
            if a.reply.is_closed() {
                self.ctx.stats.borrow_mut().cancelled += 1;
                drop(self.active.swap_remove(i)); // drops its sequences: their pages are free now
            } else if a.deadline.is_some_and(|d| now >= d) {
                let a = self.active.swap_remove(i);
                self.ctx.stats.borrow_mut().expired += 1;
                let Active { fut, reply, .. } = a;
                drop(fut);
                let _ = reply.send(Err(deadline_err()));
            } else {
                i += 1;
            }
        }
    }

    fn reserved(&self) -> usize {
        self.active.iter().map(|a| a.need).sum()
    }

    /// Whether a queued call needing `need` pages may start now.
    fn fits(&self, need: usize, prio: bool, normal_blocked: bool) -> bool {
        let cfg = &self.ctx.cfg;
        let slots = cfg.slots.max(1);
        let reserved = self.reserved();
        let n_normal = self.active.iter().filter(|a| !a.prio).count();
        if prio {
            return self.active.len() < slots && reserved + need <= cfg.pool_pages;
        }
        // ordinary calls leave one slot and the reserved pages to compaction (rule 073); one call alone may use
        // the whole pool, or a call bigger than the rest could never run
        !normal_blocked
            && self.active.len() < slots
            && n_normal < slots.saturating_sub(1).max(1)
            && (reserved + need + cfg.reserve() <= cfg.pool_pages || (self.active.is_empty() && need <= cfg.pool_pages))
    }

    fn admit(&mut self) {
        let mut normal_blocked = false;
        let mut i = 0;
        while i < self.queue.len() {
            let (need, prio) = (self.queue[i].need, self.queue[i].prio);
            if !self.fits(need, prio, normal_blocked) {
                if !prio {
                    normal_blocked = true; // first come, first served among ordinary calls
                }
                i += 1;
                continue;
            }
            let q = self.queue.remove(i).unwrap();
            let now = Instant::now();
            let op = q.call.op.index();
            self.ctx.stats.borrow_mut().queued_ms[op] += now.duration_since(q.call.arrived).as_secs_f64() * 1e3;
            let fut: Pin<Box<dyn Future<Output = Res<Value>> + 'c>> = Box::pin(job(self.ctx, q.prep, q.prio));
            let Call { arrived, reply, .. } = *q.call;
            self.active.push(Active { fut, reply, need: q.need, prio: q.prio, op, arrived, began: now, deadline: q.deadline });
            self.next += 1;
        }
    }

    fn poll(&mut self, cx: &mut Context<'_>) {
        let mut i = 0;
        while i < self.active.len() {
            match self.active[i].fut.as_mut().poll(cx) {
                Poll::Pending => i += 1,
                Poll::Ready(r) => {
                    let a = self.active.swap_remove(i);
                    self.finish(a, r);
                }
            }
        }
    }

    /// A job is done: its sequences go, its result (with timings) goes to its caller.
    fn finish(&self, a: Active<'c>, r: Res<Value>) {
        let Active { fut, reply, op, arrived, began, .. } = a;
        drop(fut); // frees the job's sequences before the next admission
        let now = Instant::now();
        let (queued_ms, run_ms) = (began.duration_since(arrived).as_secs_f64() * 1e3, now.duration_since(began).as_secs_f64() * 1e3);
        let mut st = self.ctx.stats.borrow_mut();
        let r = r.map(|mut v| {
            v["timing"] = json!({"queued_ms": queued_ms, "run_ms": run_ms, "total_ms": queued_ms + run_ms});
            v["op"] = json!(OPS[op]);
            v
        });
        if r.is_ok() {
            st.completed[op] += 1;
            st.run_ms[op] += run_ms;
        } else {
            st.failed[op] += 1;
        }
        let _ = reply.send(r);
    }

    /// One batched step over the requests in flight: decode rows first, then prefill chunks up to the step budget,
    /// compaction calls first and the rest in submission order.
    fn step(&mut self) {
        let ctx = self.ctx;
        self.inflight.sort_by_key(|r| (!r.prio, r.order));
        // while a compaction call has requests in flight, only those run: ordinary sequences pause (KV kept) and resume after
        let only_prio = self.inflight.iter().any(|r| r.prio);
        let Picks { picks, refused } = self.pick(only_prio);
        for (i, m) in refused {
            *self.inflight[i].result.borrow_mut() = Some(Err(m));
        }
        if picks.is_empty() {
            self.inflight.retain(|r| r.result.borrow().is_none());
            return;
        }
        let items: Vec<StepItem> = picks
            .iter()
            .map(|&(i, n)| {
                let r = &self.inflight[i];
                if r.decode {
                    StepItem::decode(r.agent, &r.tokens[..1], r.sampler)
                } else {
                    let d = r.done.get();
                    let out = if d + n == r.tokens.len() { r.out } else { OutputPolicy::None };
                    StepItem::prefill(r.agent, &r.tokens[d..d + n], out, r.sampler)
                }
            })
            .collect();
        let t = Instant::now();
        let res = ctx.bf.borrow_mut().step(&items);
        let us = t.elapsed().as_micros() as u64;
        drop(items);
        self.record(&picks, res, us);
        self.inflight.retain(|r| r.result.borrow().is_none());
    }

    /// The step's work: decode rows (at most MAX_ROWS), then prefill chunks within the step's token budget.
    fn pick(&self, only_prio: bool) -> Picks {
        let bf = self.ctx.bf.borrow();
        let mut p = Picks::default();
        let rows = self.pick_decode_rows(&bf, only_prio, &mut p);
        self.pick_prefill_chunks(&bf, only_prio, rows, &mut p);
        p
    }

    /// The decode rows of a step; returns how many were taken.
    fn pick_decode_rows(&self, bf: &Exec, only_prio: bool, p: &mut Picks) -> usize {
        let mut rows = 0usize;
        for (i, r) in self.inflight.iter().enumerate() {
            if r.decode && rows < MAX_ROWS && (r.prio || !only_prio) && p.take(i, 1, bf.agent_room(r.agent)) {
                rows += 1;
            }
        }
        rows
    }

    /// The prefill chunks of a step after its `rows` decode rows.
    fn pick_prefill_chunks(&self, bf: &Exec, only_prio: bool, rows: usize, p: &mut Picks) {
        let budget = self.ctx.cfg.step_tokens.max(1);
        // ordinary prefill chunks yield: a compaction call that arrives during a step waits for it, so the ordinary
        // part of a step is short (G3 helper2: 2k-token classify steps made a compaction call wait 93 ms to be admitted)
        let mut b = StepBudget { tokens: budget, ord_left: ORDINARY_STEP_TOKENS.min(budget), out_rows: rows };
        for (i, r) in self.inflight.iter().enumerate() {
            if r.decode || b.tokens == 0 || (only_prio && !r.prio) {
                continue;
            }
            let Some(n) = b.chunk(r) else { continue };
            if !r.prio {
                b.ord_left -= n.min(b.ord_left);
            }
            if p.take(i, n, bf.agent_room(r.agent)) {
                b.tokens -= n;
            }
        }
    }

    /// The step's results to its requests (a request whose last tokens were read gets its output), and the stats.
    fn record(&self, picks: &[(usize, usize)], res: Result<Vec<ItemOut>, String>, us: u64) {
        let mut st = self.ctx.stats.borrow_mut();
        st.steps += 1;
        st.step_us += us;
        let mut r = match res {
            Ok(r) => r,
            Err(e) => {
                st.failed_steps += 1;
                let m = format!("helper step failed: {e}");
                for &(i, _) in picks {
                    *self.inflight[i].result.borrow_mut() = Some(Err(m.clone()));
                }
                return;
            }
        };
        for (k, &(i, n)) in picks.iter().enumerate() {
            let req = &self.inflight[i];
            if req.decode {
                st.decode_rows += 1;
            } else {
                st.prefill_tokens += n as u64;
            }
            req.done.set(req.done.get() + n);
            if req.done.get() == req.tokens.len() {
                *req.result.borrow_mut() = Some(Ok(std::mem::take(&mut r[k])));
            }
        }
    }

    pub(super) fn publish(&self) {
        let st = self.ctx.stats.borrow();
        let bf = self.ctx.bf.borrow();
        let per_op = |f: &dyn Fn(usize) -> Value| -> Value { OPS.iter().enumerate().map(|(i, o)| (o.to_string(), f(i))).collect::<serde_json::Map<_, _>>().into() };
        let v = json!({
            "ready": true,
            "queued": self.queue.len(),
            "queued_compaction": self.queue.iter().filter(|q| q.prio).count(),
            "active": self.active.len(),
            "active_compaction": self.active.iter().filter(|a| a.prio).count(),
            "max_queued": st.max_queued,
            "completed": per_op(&|i| json!(st.completed[i])),
            "failed": per_op(&|i| json!(st.failed[i])),
            "mean_run_ms": per_op(&|i| json!(if st.completed[i] > 0 { st.run_ms[i] / st.completed[i] as f64 } else { 0.0 })),
            "cancelled": st.cancelled, "expired": st.expired, "deferred": st.deferred,
            "steps": st.steps, "failed_steps": st.failed_steps, "step_ms_total": st.step_us as f64 / 1e3,
            "prefill_tokens": st.prefill_tokens, "decode_rows": st.decode_rows,
            "pages": {"total": self.ctx.cfg.pool_pages, "used": bf.used_pages(), "reserved": self.reserved(),
                      "held_for_compaction": self.ctx.cfg.reserve(), "sequences": bf.n_agents()},
        });
        *self.shared.metrics.lock().unwrap() = v;
    }
}
