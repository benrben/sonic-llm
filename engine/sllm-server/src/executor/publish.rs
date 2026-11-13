//! What the executor publishes for the HTTP side: the `/metrics` document, the counters behind it (at most every
//! [`PUBLISH_EVERY`] while steps run, at once when the request set changes) and an agent's `/v1/agents` record.

use super::*;

impl Shared {
    pub fn metrics_json(&self, queue_depth: usize, queue_cap: usize, http_threads: usize, uptime_s: f64) -> Value {
        let e = self.engine.lock().unwrap().clone();
        // The process footprint is read now, not when the engine last published: it moves while the executor is idle (and a
        // scrape must not wait for it).
        let os = sllm_core::mem::snapshot();
        json!({
            "uptime_s": uptime_s,
            "http_threads": http_threads,
            "queue_depth": queue_depth,
            "queue_capacity": queue_cap,
            "active_requests": self.active_requests.load(Relaxed),
            "requests": {
                "started": self.requests_started.load(Relaxed),
                "completed": self.requests_completed.load(Relaxed),
                "cancelled": self.requests_cancelled.load(Relaxed),
                "failed": self.requests_failed.load(Relaxed),
                "expired": self.requests_expired.load(Relaxed),
                "deferred": self.requests_deferred.load(Relaxed),
                "stalled_clients": self.requests_stalled.load(Relaxed),
            },
            "agents": {"active": self.agents_active.load(Relaxed), "created": self.agents_created.load(Relaxed)},
            "steps": e.steps,
            "tokens_generated": e.tokens_generated,
            "prompt_tokens_evaluated": e.prompt_tokens_evaluated,
            "last_step_us": self.last_step_us.load(Relaxed),
            "max_step_us": self.max_step_us.load(Relaxed),
            "executor_loop_iterations": self.loop_iterations.load(Relaxed),
            "scheduler": self.sched.lock().unwrap().clone(),
            // Pages come from the engine's page pool (null for the fake engine and the contiguous cache).
            "pages_active": e.pages_active,
            "pages_reserved": e.pages_reserved,
            "pages_peak": e.pages_peak,
            // The process footprint (any engine). `scratch_bytes` and `memory` stay in the document, null: the generic engine
            // does not measure its own scratch (the memory ledger went with the native engine).
            "scratch_bytes": Value::Null,
            "footprint_bytes": os.phys_footprint_bytes,
            "memory": Value::Null,
            "kv": {"mode": e.kv_mode, "sequences": e.kv},
            // Prefix sharing (phase 2): shared pages counted once (`kv_bytes_used`) next to what the same sequences would
            // hold without sharing (`kv_bytes_unshared`); null for an engine without it.
            "prefix": e.prefix,
            "spec": e.spec,
            "kv_read": e.kv_read,
            // the agent lifecycle (phase 3): agents per state, SSD bytes, restores, re-prefills avoided
            "lifecycle": self.lifecycle.lock().unwrap().clone(),
            // the patched libggml-metal's live kernel counters (qmm, cascade: `dispatches` = attention or matmul ops it
            // took so far; they stay 0 under the fake engine)
            "ggml_metal": sllm_core::caps::ggml_metal_status(),
            "placeholders_note": "pages_*, scratch_bytes and memory are null (no page pool or memory ledger in the fake and generic \
                                  engines). footprint_bytes is the live proc_pid_rusage footprint of this process",
        })
    }
}

impl Executor {
    /// Publish the counters when the last publish is `PUBLISH_EVERY` old; otherwise note that one is due. Building the engine
    /// counters and the scheduler JSON costs more than a decode step of a small model, so the step loop does not do it every
    /// step; the loop publishes before it blocks, and every command publishes.
    pub(super) fn publish_due(&self) {
        // a request admitted, queued, waiting or finished, or an agent created or freed, shows in /metrics at once; only the
        // per-step counters (tokens, steps, timings) wait for the interval
        if self.state_sig() != self.last_sig.get() || self.last_publish.get().elapsed() >= PUBLISH_EVERY {
            self.publish_counters();
        } else {
            self.publish_dirty.set(true);
        }
    }

    /// Cheap shape of the request set: slots, queued, running, waiting on a prefix, agents.
    pub(super) fn state_sig(&self) -> [usize; 5] {
        let queued = self.queues.iter().map(|q| q.len()).sum();
        let running = self.running.iter().map(|q| q.len()).sum();
        let waiting = self.slots.values().filter(|s| s.waiting).count();
        [self.slots.len(), queued, running, waiting, self.agents.len()]
    }

    pub(super) fn publish_counters(&self) {
        self.last_publish.set(Instant::now());
        self.publish_dirty.set(false);
        self.last_sig.set(self.state_sig());
        *self.shared.engine.lock().unwrap() = self.engine.counters();
        self.shared.active_requests.store(self.slots.len() as u64, Relaxed);
        self.shared.agents_active.store(self.agents.len() as u64, Relaxed);
        let cfg = self.cfg();
        let mut lanes = serde_json::Map::new();
        for lane in Lane::ALL {
            let l = lane.index();
            let c = self.stats.lanes[l];
            lanes.insert(
                lane.name().into(),
                json!({
                    "cap": cfg.caps[l], "running": self.running[l].len(), "queued": self.queues[l].len(),
                    "queue_limit": cfg.queue_limits[l], "max_queued": c.max_queued, "admitted": c.admitted,
                    "waiting": self.waiting_in(lane),
                    "deferred": c.deferred, "completed": c.completed, "cancelled": c.cancelled, "expired": c.expired,
                }),
            );
        }
        let s = &self.stats;
        self.life_publish();
        *self.shared.sched.lock().unwrap() = json!({
            "lanes": lanes,
            "queued": self.queues.iter().map(|q| q.len()).sum::<usize>(),
            "running": self.running.iter().map(|q| q.len()).sum::<usize>(),
            "waiting_on_pages": Lane::ALL.iter().map(|&l| self.waiting_in(l)).sum::<usize>(),
            "prefix_waited": s.prefix_waited,
            "prefix_shared_tokens": s.prefix_shared_tokens,
            // requests waiting for room in the engine, and the turns held, clipped or writing notes (compaction, rollover)
            "room_waiting": self.room_wait.len(),
            "held_turns": self.held.len(),
            "rollover_turns": self.rolled.len(),
            "notes_writing": self.writing.len(),
            "summaries_pending": self.summarizing.len(),
            "deferred": s.lanes.iter().map(|c| c.deferred).sum::<u64>(),
            "expired": s.lanes.iter().map(|c| c.expired).sum::<u64>(),
            "steps": s.steps,
            "steps_with_interactive": s.steps_with_interactive,
            "bg_prefill_deferred_steps": s.bg_prefill_deferred_steps,
            "burst_skip_steps": s.burst_skip_steps,
            "step_ms_ema": s.step_ms_ema,
            "last_step": {"items": s.last.items.len(), "decode_rows": s.last.decode_rows, "prefill_tokens": s.last.prefill_tokens,
                          "interactive": s.last.interactive, "bg_deferred": s.last.bg_deferred},
        });
    }

    pub(super) fn agent_info(&self, id: &str) -> Option<AgentInfo> {
        let a = self.agents.get(id)?;
        Some(AgentInfo {
            id: id.to_string(),
            object: "agent",
            lane: a.lane,
            context_tokens: self.engine.agent_context_tokens(id).unwrap_or(0),
            turns: a.turns,
            busy: a.busy.is_some(),
            created: a.created,
            state: a.life.state.name(),
            kv: a.life.state.tier(),
            queued_turns: a.inbox.len(),
            history_tokens: a.life.history.len(),
            fields: a.comp.fields.to_json(),
            compaction: a.comp.to_json(),
        })
    }

}
