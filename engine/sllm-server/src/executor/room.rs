//! Requests waiting for room in the engine (cache pages, sequence slots, recurrent states) while others run: they are
//! asked again when a request ends. What waits is the request as it arrived, before any admission step rewrote it (a
//! History agent's rebuild, a rollover's clip), so every retry runs admission from the start and each step runs once per
//! admission; and a turn waiting here holds its agent's place (the agent's later turns queue behind it).

use super::*;

/// A request waiting for room.
pub(super) struct RoomWait {
    /// The request as it arrived.
    pub(super) cmd: StartCmd,
    /// When it began to wait (its deadline counts the wait).
    since: Instant,
}

impl Executor {
    /// `cmd` (as it arrived) waits for room.
    pub(super) fn room_wait_push(&mut self, cmd: StartCmd) {
        self.room_wait.push_back(RoomWait { cmd, since: Instant::now() });
    }

    /// Whether a turn of `aid` waits for room (its agent's later turns wait behind it).
    pub(super) fn turn_waits_for_room(&self, aid: &str) -> bool {
        self.room_wait.iter().any(|w| w.cmd.agent.as_deref() == Some(aid))
    }

    /// A request that leaves the wait without being admitted: what its admission noted (a rollover clip) goes with it.
    fn room_forget(&mut self, id: ReqId) {
        self.rolled.remove(&id);
    }

    /// Cancel a request waiting for room (its client went away). True if it was waiting.
    pub(super) fn room_cancel(&mut self, id: ReqId) -> bool {
        let Some(i) = self.room_wait.iter().position(|w| w.cmd.id == id) else { return false };
        self.room_wait.remove(i);
        self.room_forget(id);
        self.shared.requests_cancelled.fetch_add(1, Relaxed);
        true
    }

    /// Ask the engine again for the requests waiting on room, once something has ended since the last try (or nothing runs
    /// any more: then a request that still finds no room is refused, as it would have been on arrival).
    pub(super) fn room_retry(&mut self) {
        if self.room_wait.is_empty() || (self.room_gen == self.room_tried && !self.slots.is_empty()) {
            return;
        }
        self.room_tried = self.room_gen;
        let waiting: Vec<RoomWait> = self.room_wait.drain(..).collect();
        for w in waiting {
            if let Some(cmd) = self.room_still_wanted(w) {
                // the head of its agent's turns: it does not queue behind the turns that arrived while it waited
                self.start_inner(cmd, true);
            }
        }
    }

    /// The waiting request, its deadline less the wait; None when its client left or its deadline passed (then it is told).
    fn room_still_wanted(&mut self, w: RoomWait) -> Option<StartCmd> {
        let RoomWait { mut cmd, since } = w;
        if cmd.events.is_closed() {
            self.room_forget(cmd.id);
            self.shared.requests_cancelled.fetch_add(1, Relaxed);
            return None;
        }
        let waited = since.elapsed().as_millis() as u64;
        match cmd.deadline_ms {
            Some(ms) if waited >= ms => {
                self.room_forget(cmd.id);
                self.shared.requests_expired.fetch_add(1, Relaxed);
                let e = EngineError::new(504, "deadline_exceeded", "the request's deadline passed while it waited for room");
                let _ = cmd.events.try_send(ReqEvent::Error(e));
                None
            }
            Some(ms) => {
                cmd.deadline_ms = Some(ms - waited);
                Some(cmd)
            }
            None => Some(cmd),
        }
    }
}
