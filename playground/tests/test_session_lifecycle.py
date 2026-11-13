"""The live session's lifecycle (live.py): start, stop and replace are one at a time, a failed start clears only its own
session, a stop ends the run first, the idle stop waits for an accepted run, an engine that dies fails the session.

No GPU, no server: a fake session stands in for Session (its engines)."""

import asyncio
import time
from pathlib import Path

import pytest

from sonic_playground import live, models
from sonic_playground.live_parts import session as session_mod
from sonic_playground.live_parts.config import IDLE_STOP_S
from sonic_playground.live_parts.engine import Engine
from sonic_playground.live_parts.session import Session, SessionState

ORDER: list[tuple[str, str]] = []


class FakeSession:
    """Session's lifecycle without engines: start waits for `gate`, stop for `stop_gate` (when set)."""

    def __init__(self, model, *_):
        self.model, self.state, self.error = model, "starting", None
        self.busy, self.run_task, self.restarting = asyncio.Lock(), None, False
        self.gate, self.stop_gate = asyncio.Event(), None

    def info(self):
        return {"model": self.model["id"], "state": str(self.state)}

    async def start(self):
        ORDER.append(("start begins", self.model["id"]))
        try:
            await self.gate.wait()
            if self.state == "stopped":  # the real one: its engine's process is gone (proc None) -> it fails
                raise RuntimeError("its engine was stopped while it started")
            self.state = "ready"
        except BaseException as ex:
            self.state, self.error = "failed", str(ex)
            raise

    async def stop(self):
        ORDER.append(("stop begins", self.model["id"]))
        if self.stop_gate is not None:
            await self.stop_gate.wait()
        if self.state != "failed":
            self.state = "stopped"
        ORDER.append(("stop ends", self.model["id"]))


@pytest.fixture
def fake_models(monkeypatch):
    ORDER.clear()
    made: list[FakeSession] = []

    rows = [{"id": i, "file": f"{i}.gguf", "error": None, "size_bytes": 1} for i in ("a", "b")]
    def make(*a):
        made.append(FakeSession(*a))
        return made[-1]

    monkeypatch.setattr(models, "scan", lambda folder: rows)
    monkeypatch.setattr(live, "template_for", lambda file, *a: object())
    monkeypatch.setattr(live, "stop_orphans", lambda: [])
    monkeypatch.setattr(live, "Session", make)
    monkeypatch.setattr(live, "SESSION", None)
    events: list[dict] = []
    monkeypatch.setattr(live.BUS, "publish", events.append)
    return made, events


async def until(cond, s=2.0):
    end = time.monotonic() + s
    while not cond():
        assert time.monotonic() < end, "timed out"
        await asyncio.sleep(0.005)


# -- 1a: a second load while the first is starting --------------------------------------------------------------------
async def test_a_second_load_while_the_first_starts_keeps_the_second_session(fake_models):
    made, _ = fake_models
    first = asyncio.create_task(live.post_session(live.SessionReq(model_id="a")))
    await until(lambda: len(made) == 1)
    second = asyncio.create_task(live.post_session(live.SessionReq(model_id="b")))
    await asyncio.sleep(0.02)
    made[0].gate.set()  # the first start ends (it fails if the second stopped it under its feet)
    await until(lambda: len(made) == 2)
    made[1].gate.set()
    await asyncio.gather(first, second, return_exceptions=True)
    assert live.SESSION is made[1] and made[1].state == "ready"  # the page's model is the one it loaded last
    assert made[0].state == "stopped"  # the first was started fully, then stopped: no half-started engine


# -- 1b: a stop ends before the next session starts (it holds the GPU lock until then) ---------------------------------
async def test_a_load_during_a_stop_starts_only_after_the_stop_ended(fake_models):
    made, _ = fake_models
    first = asyncio.create_task(live.post_session(live.SessionReq(model_id="a")))
    await until(lambda: len(made) == 1)
    made[0].gate.set()
    await first
    made[0].stop_gate = asyncio.Event()
    stop = asyncio.create_task(live.delete_session())
    await until(lambda: ("stop begins", "a") in ORDER)
    load = asyncio.create_task(live.post_session(live.SessionReq(model_id="b")))
    await asyncio.sleep(0.02)
    made[0].stop_gate.set()
    await until(lambda: len(made) == 2)
    made[1].gate.set()
    await asyncio.gather(stop, load)
    assert ORDER.index(("stop ends", "a")) < ORDER.index(("start begins", "b"))
    assert live.SESSION is made[1]


# -- 1c: DELETE /session during a run ends the run first (no engine restarted after the stop) ----------------------------
async def test_stopping_the_session_cancels_the_run_after_its_restart_and_waits_for_it(fake_models):
    made, events = fake_models
    s = FakeSession({"id": "a"})
    s.state, s.restarting = "ready", True
    ended = []

    async def run():
        try:
            await asyncio.sleep(60)
        finally:
            ended.append(s.state)  # the session must not be stopped under the run

    s.run_task = asyncio.create_task(run())
    live.SESSION = s
    stop = asyncio.create_task(live.delete_session())
    await asyncio.sleep(0.12)
    assert not s.run_task.done()  # an engine is restarting: the stop waits for that
    s.restarting = False
    assert await stop == {"stopped": True}
    assert s.run_task.done() and ended == ["ready"] and s.state == "stopped"
    assert live.SESSION is None


# -- the idle stop: compare-and-stop, and an accepted run counts as use -------------------------------------------------------
def sampling_session(monkeypatch):
    s = Session.__new__(Session)
    s.engines, s.busy, s.run_task, s.state = {}, asyncio.Lock(), None, "ready"
    s.last_used = time.monotonic() - IDLE_STOP_S - 1

    async def nothing(*a):
        await asyncio.sleep(60)

    async def tick(*a):
        pass

    s._gpu_util_loop, s._tick = nothing, tick
    monkeypatch.setattr(session_mod, "TICK_S", 0.01)
    stops = []

    async def stop_session(*a):
        stops.append(a)

    monkeypatch.setattr(live, "stop_session", stop_session)
    monkeypatch.setattr(live, "log", lambda m: None)
    return s, stops


async def test_the_idle_stop_does_not_fire_while_a_run_is_accepted_but_not_yet_running(monkeypatch):
    s, stops = sampling_session(monkeypatch)
    gate = asyncio.Event()
    s.run_task = asyncio.create_task(gate.wait())  # accepted: it waits for the read ahead, busy is not held yet
    sampler = asyncio.create_task(s._sample())
    await asyncio.sleep(0.1)
    assert stops == []
    sampler.cancel()
    gate.set()
    await asyncio.gather(sampler, s.run_task, return_exceptions=True)


async def test_the_idle_stop_names_the_session_it_stops(monkeypatch):
    s, stops = sampling_session(monkeypatch)
    await asyncio.wait_for(s._sample(), 1)
    await asyncio.sleep(0.01)
    assert stops == [(s,)]  # a session loaded meanwhile is not the one the idle stop meant


async def test_stop_session_with_another_session_named_leaves_the_current_one(fake_models):
    current = FakeSession({"id": "b"})
    current.state = "ready"
    live.SESSION = current
    await live.stop_session(FakeSession({"id": "a"}))
    assert live.SESSION is current and current.state == "ready"


# -- 1e: an engine that died while the session was ready ------------------------------------------------------------------
class DeadProc:
    pid, returncode = 4242, -9

    def poll(self):
        return self.returncode


async def test_an_engine_that_died_fails_the_session_and_the_page_hears_it(monkeypatch):
    s, stops = sampling_session(monkeypatch)
    s.last_used = time.monotonic()
    e = Engine("sllm", Path("m.gguf"), True)
    e.up, e.proc = True, DeadProc()
    s.engines = {"sllm": e}
    s.error = None
    events = []
    monkeypatch.setattr(live.BUS, "publish", events.append)
    monkeypatch.setattr(Session, "info", lambda self: {"state": str(self.state)})
    sampler = asyncio.create_task(s._sample())
    await asyncio.sleep(0.1)
    sampler.cancel()
    await asyncio.gather(sampler, return_exceptions=True)
    assert s.state == "failed" and "exited" in s.error
    assert any(ev.get("type") == "session" and ev["session"]["state"] == "failed" for ev in events)
    assert not e.up


def test_session_states_stay_the_same_strings_on_the_wire():
    import json

    assert [str(x) for x in SessionState] == ["starting", "ready", "stopped", "failed"]
    assert json.dumps({"state": SessionState.READY}) == '{"state": "ready"}'
    s = Session.__new__(Session)
    s.state = SessionState.STOPPED
    with pytest.raises(RuntimeError):  # a stopped session never comes back
        s.move(SessionState.READY)
