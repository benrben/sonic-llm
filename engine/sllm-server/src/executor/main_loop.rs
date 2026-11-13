//! The step loop ([`run_with`]): commands, the step boundary, the step, and the loop profile.

use super::*;

/// One step-loop iteration of the loop profile (task 12). `*_ns` are the executor's non-step phases; `bd` is the
/// engine's own split of `step_us`. `intake_us` is in nanoseconds too (the name is kept short for the JSON).
#[derive(Default, Serialize)]
pub(super) struct LoopRec {
    t_us: u64,
    iter_us: u64,
    /// Blocked waiting for a command (only when nothing was admitted).
    wait_us: u64,
    n_cmds: u32,
    #[serde(rename = "intake_ns")]
    intake_us: u64,
    sweep_ns: u64,
    fill_ns: u64,
    plan_ns: u64,
    step_us: u64,
    /// `after_step`: prefill bookkeeping plus the event fan-out (one `try_send` per token).
    dispatch_ns: u64,
    publish_ns: u64,
    n_items: usize,
    n_events: usize,
    bd: Option<StepBreakdown>,
}

/// Whether the step loop goes on, or shuts down (a shutdown command, or the command channel closed).
#[derive(PartialEq)]
enum Flow {
    Go,
    Stop,
}

impl Executor {
    /// The step loop, until a shutdown: then every request is stopped and the loop profile written.
    pub(super) fn run_loop(&mut self, rx: &mut mpsc::Receiver<Command>) {
        let epoch = Instant::now();
        // a one-thread timer runtime, only to wait on the command channel with a timeout
        let wait_rt = tokio::runtime::Builder::new_current_thread().enable_time().build().ok();
        loop {
            self.shared.loop_iterations.fetch_add(1, Relaxed);
            // The loop profile (task 12): a handful of `Instant::now()` per iteration (~20 ns each on macOS), kept only when
            // SLLM_LOOP_TRACE names a file.
            let t0 = Instant::now();
            let mut rec = LoopRec { t_us: t0.duration_since(epoch).as_micros() as u64, ..Default::default() };
            if self.wait_for_command(rx, &mut rec, t0) == Flow::Stop {
                break;
            }
            let t_intake = Instant::now();
            if self.take_commands(rx, &mut rec) == Flow::Stop {
                break;
            }
            let (plan, t) = self.boundary(t_intake, &mut rec);
            if plan.items.is_empty() {
                if self.idle_step(rx, wait_rt.as_ref()) == Flow::Stop {
                    break;
                }
                continue;
            }
            self.step_once(&plan, t, t0, rec);
        }
        self.shutdown();
    }

    /// Nothing admitted: requests waiting for room start now (or are refused, never wait on an idle loop); if still nothing
    /// runs, the loop waits for a command.
    fn wait_for_command(&mut self, rx: &mut mpsc::Receiver<Command>, rec: &mut LoopRec, t0: Instant) -> Flow {
        if self.slots.is_empty() && !self.room_wait.is_empty() {
            self.room_retry(); // nothing runs: the waiting requests start now or are refused, never wait on an idle loop
            self.publish_dirty.set(true); // and /metrics shows where they went
        }
        if !self.slots.is_empty() {
            return Flow::Go;
        }
        if self.publish_dirty.get() {
            self.publish_counters(); // going idle: the metrics must show the final state
        }
        match self.life_poll_every() {
            // lifecycle work is pending (I/O, queued turns, idle timers): poll instead of blocking
            Some(every) => match rx.try_recv() {
                Ok(Command::Shutdown) => Flow::Stop,
                Ok(c) => {
                    rec.n_cmds += 1;
                    self.handle(c);
                    Flow::Go
                }
                Err(mpsc::error::TryRecvError::Disconnected) => Flow::Stop,
                Err(mpsc::error::TryRecvError::Empty) => {
                    std::thread::sleep(every);
                    Flow::Go
                }
            },
            None => self.block_for_command(rx, rec, t0),
        }
    }

    /// Block until a command arrives.
    fn block_for_command(&mut self, rx: &mut mpsc::Receiver<Command>, rec: &mut LoopRec, t0: Instant) -> Flow {
        match rx.blocking_recv() {
            Some(Command::Shutdown) | None => Flow::Stop,
            Some(c) => {
                rec.wait_us = t0.elapsed().as_micros() as u64;
                rec.n_cmds += 1;
                self.handle(c);
                Flow::Go
            }
        }
    }

    /// Every command waiting now. A closed channel stops the loop once nothing is admitted.
    fn take_commands(&mut self, rx: &mut mpsc::Receiver<Command>, rec: &mut LoopRec) -> Flow {
        loop {
            match rx.try_recv() {
                Ok(Command::Shutdown) => return Flow::Stop,
                Ok(c) => {
                    rec.n_cmds += 1;
                    self.handle(c)
                }
                Err(mpsc::error::TryRecvError::Empty) => return Flow::Go,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    return if self.slots.is_empty() { Flow::Stop } else { Flow::Go };
                }
            }
        }
    }

    /// The step boundary: sweep, the lifecycle, room, fill, waiting prefixes, then the plan (and when it was made).
    fn boundary(&mut self, t_intake: Instant, rec: &mut LoopRec) -> (Plan, Instant) {
        let t_sweep = Instant::now();
        rec.intake_us = t_sweep.duration_since(t_intake).as_nanos() as u64;
        self.sweep();
        self.life_tick();
        self.room_retry();
        let t_fill = Instant::now();
        rec.sweep_ns = t_fill.duration_since(t_sweep).as_nanos() as u64;
        self.fill();
        self.poll_waiting();
        let t_plan = Instant::now();
        rec.fill_ns = t_plan.duration_since(t_fill).as_nanos() as u64;
        let plan = self.plan();
        let t = Instant::now();
        rec.plan_ns = t.duration_since(t_plan).as_nanos() as u64;
        (plan, t)
    }

    /// Nothing to step: the engine's idle work, then (requests admitted that cannot run) a short wait for a command.
    fn idle_step(&mut self, rx: &mut mpsc::Receiver<Command>, wait_rt: Option<&tokio::runtime::Runtime>) -> Flow {
        let events = self.engine.idle();
        self.dispatch(events);
        if !self.slots.is_empty() && self.wait_a_moment(rx, wait_rt) == Flow::Stop {
            return Flow::Stop;
        }
        self.compaction_tick();
        self.publish_due();
        Flow::Go
    }

    /// Admitted requests but nothing can run (the engine is not ready for any queued one): sleep until a command arrives (a
    /// cancel, a new request) or the lifecycle/deadline poll interval is up.
    fn wait_a_moment(&mut self, rx: &mut mpsc::Receiver<Command>, wait_rt: Option<&tokio::runtime::Runtime>) -> Flow {
        let every = self.life_poll_every().unwrap_or(IDLE_WAIT).min(IDLE_WAIT);
        match wait_rt {
            // the timer is made inside the runtime (a Sleep made outside it panics: no reactor)
            Some(rt) => match rt.block_on(async { tokio::time::timeout(every, rx.recv()).await }) {
                Ok(Some(Command::Shutdown)) | Ok(None) => return Flow::Stop,
                Ok(Some(c)) => self.handle(c),
                Err(_) => {}
            },
            None => std::thread::sleep(Duration::from_millis(1)),
        }
        Flow::Go
    }

    /// One engine step of `plan` (planned at `t`), its events, and the iteration's loop-profile record (begun at `t0`).
    fn step_once(&mut self, plan: &Plan, t: Instant, t0: Instant, mut rec: LoopRec) {
        let events = self.engine.step(&plan.items);
        let t_after = Instant::now();
        let us = t_after.duration_since(t).as_micros() as u64;
        self.shared.last_step_us.store(us, Relaxed);
        self.shared.max_step_us.fetch_max(us, Relaxed);
        rec.step_us = us;
        rec.n_items = plan.items.len();
        rec.n_events = events.len();
        rec.bd = self.engine.last_step_breakdown();
        self.after_step(plan, events, us);
        self.compaction_tick();
        let t_pub = Instant::now();
        rec.dispatch_ns = t_pub.duration_since(t_after).as_nanos() as u64;
        self.publish_due();
        let t_end = Instant::now();
        rec.publish_ns = t_end.duration_since(t_pub).as_nanos() as u64;
        rec.iter_us = t_end.duration_since(t0).as_micros() as u64;
        if let Some(tr) = self.trace.as_mut() {
            tr.push(rec);
        }
    }
}
