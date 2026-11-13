"""Events of the live playground (live.py): the bus behind the page's event stream (live.BUS is its one instance) and the
per-second rate of an engine's running counter."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque

from .config import RATE_MOVES, RATE_WINDOW_S, REPLAY


# -- events ---------------------------------------------------------------------------------------------------------
STREAM_LIFETIME_S = 20.0
QUEUE_MAX = 2000  # events waiting for one subscriber; past that it has fallen behind and is told to resync


def epoch_start() -> int:
    """The first event id of a server: microseconds since the epoch, so the ids keep growing across a restart and a
    browser's Last-Event-ID from before the restart is older than every event of the new server."""
    return time.time_ns() // 1000


def _frame(seq: int | None, ev: dict) -> str:
    """One server-sent event; without an id (resync, snapshot) the browser's Last-Event-ID stays the last real event."""
    head = f"id: {seq}\n" if seq is not None else ""
    return f"{head}data: {json.dumps(ev, default=str)}\n\n"


class Bus:
    def __init__(self, start: int = 0):
        self.subs: set[asyncio.Queue] = set()
        self.lagged: set[asyncio.Queue] = set()  # subscribers that missed events: they get a resync, then events again
        self.seq = start
        self.recent: deque[tuple[int, dict]] = deque(maxlen=REPLAY)

    def publish(self, ev: dict) -> None:
        ev.setdefault("t", time.time())
        self.seq += 1
        self.recent.append((self.seq, ev))
        for q in list(self.subs):
            if q in self.lagged:
                continue
            if q.qsize() < QUEUE_MAX:
                q.put_nowait((self.seq, ev))
            else:
                self.lagged.add(q)

    def _replay(self, q: asyncio.Queue, last_id: int) -> None:
        """The events after `last_id` into `q`. A gap (they are no longer all in the buffer) starts with a resync; an id
        ahead of this server (one from before a restart that counted from 0) replays nothing: events from now."""
        missed = [r for r in self.recent if r[0] > last_id]
        if missed and missed[0][0] > last_id + 1:
            q.put_nowait((None, {"type": "resync", "t": time.time()}))
        for item in missed:
            q.put_nowait(item)

    async def stream(self, lifetime_s: float = STREAM_LIFETIME_S, last_id: int | None = None, first: dict | None = None):
        """Events as server-sent events, after `first` (the snapshot of the state). The stream ends after `lifetime_s` and
        the browser reconnects at once: an open-ended stream would hold uvicorn's reload (it waits for every connection)
        for as long as a tab is open. A reconnect sends Last-Event-ID, and the events published in between are replayed,
        so no tick is lost. A subscriber that fell behind gets {"type": "resync"} once it has read what it holds."""
        q: asyncio.Queue = asyncio.Queue()
        self.subs.add(q)
        if last_id is not None:
            self._replay(q, last_id)
        end = time.monotonic() + lifetime_s
        try:
            yield "retry: 100\n\n"
            if first is not None:
                yield _frame(None, first)
            while (left := end - time.monotonic()) > 0:
                yield await self._next_frame(q, left)
        finally:
            self.subs.discard(q)
            self.lagged.discard(q)

    async def _next_frame(self, q: asyncio.Queue, left: float) -> str:
        """The next frame for subscriber `q`: a resync when it fell behind and has read what it holds, else its next
        event, or a keepalive after 15 s without one."""
        if q in self.lagged and q.empty():
            self.lagged.discard(q)
            return _frame(None, {"type": "resync", "t": time.time()})
        try:
            seq, ev = await asyncio.wait_for(q.get(), min(15, left))
        except asyncio.TimeoutError:
            return ": keepalive\n\n"
        return _frame(seq, ev)


def _rate(points: deque, now: float, count: int) -> float | None:
    """Per-second rate of a running counter. `points` holds (time, count) at the moments the counter moved.

    The counter moves once per engine step, and a step of a big model takes longer than a tick (Qwen3.8-27B reads 512 tokens
    in about 1.4 s). A fixed 2 s window then holds one step or two and the graph jumps between half and full speed while the
    engine reads evenly. So the rate is measured from one move of the counter to a later one: over at least RATE_WINDOW_S and
    at least RATE_MOVES moves. No move for RATE_WINDOW_S and for 1.5 times the last gap between moves: the rate is 0."""
    if points and count < points[-1][1]:  # the counter started again (a new engine)
        points.clear()
    if not points:
        points.append((now, count))
        return None
    if count != points[-1][1]:
        points.append((now, count))
    while len(points) > RATE_MOVES + 1 and points[-1][0] - points[1][0] >= RATE_WINDOW_S:
        points.popleft()
    if len(points) < 2:
        return 0.0
    (t0, c0), (t1, c1) = points[0], points[-1]
    if now - t1 >= max(RATE_WINDOW_S, 1.5 * (t1 - points[-2][0])):  # it stopped: the next move is measured from now
        points.clear()
        points.append((now, c1))
        return 0.0
    return (c1 - c0) / (t1 - t0)
