"""A run's lifecycle (live.py, live_parts/agents.py, runners.py): accepted at once, stopped from the moment it is accepted,
always one run_done, agents that fail or are stopped take their siblings with them, the history write cannot turn a
finished run into an error; the read ahead and the follow-up readers belong to their session.

No GPU, no server: fake engines and a session made without __init__."""

import asyncio

import pytest

from sonic_playground.live_parts.agents import _agents
from sonic_playground.live_parts.runners import _run_burst, _run_load
from sonic_playground import live
from sonic_playground.live_parts import runners


@pytest.fixture
def events(monkeypatch):
    seen: list[dict] = []
    monkeypatch.setattr(live.BUS, "publish", seen.append)
    return seen


@pytest.fixture
def logs(monkeypatch):
    out: list[str] = []
    monkeypatch.setattr(live, "log", out.append)
    return out


def ready_session(monkeypatch):
    s = live.Session.__new__(live.Session)
    s.state, s.busy, s.engines, s.restarting, s.last_used, s.run_task = "ready", asyncio.Lock(), {}, False, 0.0, None
    s.use_thinking = lambda level: None
    s.readers = None
    monkeypatch.setattr(live, "SESSION", s)
    return s


def read_ahead_in_flight(monkeypatch, session=None):
    """A read ahead whose request is still with the engine: a run waits for it (preread_settle)."""
    gate = asyncio.Event()
    monkeypatch.setattr(live, "_PRE", {"task": asyncio.create_task(gate.wait()), "next": None, "reads": 0, "tokens": 0,
                                       "session": session})  # fmt: skip
    return gate


async def forever(*_):
    await asyncio.sleep(60)


# -- acceptance ------------------------------------------------------------------------------------------------------------
async def test_two_runs_posted_while_the_read_ahead_settles_one_is_refused(monkeypatch, events):
    s = ready_session(monkeypatch)
    gate = read_ahead_in_flight(monkeypatch, s)
    monkeypatch.setattr(live, "run_scenario", forever)
    posts = [asyncio.create_task(live.post_run(live.RunReq())) for _ in range(2)]
    await asyncio.sleep(0.05)
    gate.set()
    got = await asyncio.gather(*posts, return_exceptions=True)
    refused = [g for g in got if isinstance(g, live.HTTPException)]
    assert len(refused) == 1 and refused[0].status_code == 409
    s.run_task.cancel()
    await asyncio.gather(s.run_task, return_exceptions=True)


async def test_stop_while_the_read_ahead_settles_stops_the_run_and_the_page_hears_run_done(monkeypatch, events):
    s = ready_session(monkeypatch)
    read_ahead_in_flight(monkeypatch, s)
    monkeypatch.setattr(live, "run_scenario", forever)
    post = asyncio.create_task(live.post_run(live.RunReq()))
    await asyncio.sleep(0.02)
    assert await live.post_run_stop() == {"stopped": True}
    await asyncio.gather(post, return_exceptions=True)
    await asyncio.gather(s.run_task, return_exceptions=True)
    done = [e for e in events if e["type"] == "run_done"]
    assert len(done) == 1 and done[0]["stopped"] is True
    assert not s.busy.locked() and s.run_task.done()


async def test_a_failed_history_write_still_ends_the_run_with_its_record(monkeypatch, events, logs, cfg):
    s = ready_session(monkeypatch)
    monkeypatch.setattr(live, "_PRE", {"task": None, "next": None, "reads": 0, "tokens": 0, "session": None})
    cfg.runs_file.mkdir(parents=True)  # a directory: the append fails

    async def finished(s, req, run_id):
        return {"id": run_id, "metrics": {}}

    monkeypatch.setattr(live, "run_scenario", finished)
    await live.post_run(live.RunReq())
    await s.run_task
    done = events[-1]
    assert done["type"] == "run_done" and "error" not in done and done["record"]["id"] == done["run"]
    assert any("writing run" in x and "history" in x for x in logs)


# -- agents that fail or are stopped ---------------------------------------------------------------------------------------
class OneFails:
    """Agent prompt "bad" fails at once; the others write until cancelled."""

    def __init__(self):
        self.ctx, self.cells, self.cancelled = 4096, 4096, 0

    async def tokenize(self, text):
        return text.split()

    async def generate(self, ids, n, lane=None, on_token=None, ignore_eos=False):
        if ids == ["bad"]:
            raise RuntimeError("engine said no")
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


def agent_session(e):
    s = live.Session.__new__(live.Session)
    s.engines, s.lim = {"sllm": e}, {"answer": None, "agent": None, "doc": None}
    s.template = type("T", (), {"eog": ()})
    return s


async def test_one_failing_agent_cancels_its_siblings(events):
    e = OneFails()
    with pytest.raises(RuntimeError):
        await _agents(agent_session(e), "sllm", "r", [("a", "good"), ("b", "bad"), ("c", "good")])
    assert e.cancelled == 2  # nobody writes on after the run ended


async def test_one_failing_burst_agent_cancels_its_siblings(monkeypatch, events):
    e = OneFails()
    monkeypatch.setattr(runners, "burst_prompts", lambda s, n: [("a", "good"), ("b", "bad"), ("c", "good")])
    with pytest.raises(RuntimeError):
        await _run_burst(agent_session(e), live.RunReq(scenario="burst", agents=3), "r", "sllm")
    assert e.cancelled == 2


async def test_chat_under_load_takes_its_swarm_down_when_it_ends_early(monkeypatch, events):
    asked = []
    swarm_cancelled = []

    async def chat_answer(s, rt, run_id, system, q, remember):
        asked.append(q)
        if len(asked) == 2:
            raise RuntimeError("the loaded answer failed")
        return {"ids": [1], "decode_tok_s": 10.0}

    async def swarm(*a, **k):
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            swarm_cancelled.append(True)
            raise

    monkeypatch.setattr(runners, "_chat_answer", chat_answer)
    monkeypatch.setattr(runners, "_agents", swarm)
    monkeypatch.setattr(runners, "_docs", lambda files, n, chars: [("a.md", "text")])
    monkeypatch.setattr(runners.asyncio, "sleep", _fast_sleep)
    s = agent_session(OneFails())
    s.one_shot = lambda system, message: message
    with pytest.raises(RuntimeError):
        await _run_load(s, live.RunReq(scenario="load", agents=2), "r", "sllm")
    assert swarm_cancelled == [True]


_real_sleep = asyncio.sleep


async def _fast_sleep(t, *a):
    await _real_sleep(min(t, 0.01))


# -- the read ahead belongs to its session and never runs beside an accepted run -------------------------------------------
async def test_a_read_ahead_posted_after_a_run_was_accepted_is_refused(monkeypatch, events):
    s = ready_session(monkeypatch)
    read_ahead_in_flight(monkeypatch, s)
    monkeypatch.setattr(live, "run_scenario", forever)
    post = asyncio.create_task(live.post_run(live.RunReq()))
    await asyncio.sleep(0.02)
    assert (await live.post_preread(live.PrereadReq(message="next question")))["ok"] is False
    s.run_task.cancel()
    await asyncio.gather(post, s.run_task, return_exceptions=True)


def reads_ahead(monkeypatch) -> list:
    read = []

    async def preread_once(sess, req):
        read.append(sess)
        return 3

    monkeypatch.setattr(live, "preread_once", preread_once)
    return read


async def test_typing_counts_as_use_of_the_session(monkeypatch, events):
    s = ready_session(monkeypatch)
    monkeypatch.setattr(live, "_PRE", {"task": None, "next": None, "reads": 0, "tokens": 0, "session": None})
    reads_ahead(monkeypatch)
    assert (await live.post_preread(live.PrereadReq(message="hi")))["ok"] is True
    assert s.last_used > 0  # the idle stop does not stop a session someone is typing into
    await live._PRE["task"]


async def test_a_new_session_gets_its_own_read_ahead_while_the_old_one_is_in_flight(monkeypatch, events):
    old = ready_session(monkeypatch)
    gate = read_ahead_in_flight(monkeypatch, old)  # the old session's read ahead is still in flight
    s = ready_session(monkeypatch)
    read = reads_ahead(monkeypatch)
    assert (await live.post_preread(live.PrereadReq(message="hi")))["ok"] is True
    await asyncio.sleep(0.02)
    assert read == [s]
    gate.set()


# -- the follow-up's readers -------------------------------------------------------------------------------------------------
async def test_reset_forgets_the_follow_up_readers(monkeypatch):
    s = ready_session(monkeypatch)
    s.engines = {"sllm": None}
    s.history = {"sllm": [("q", "a")]}
    s.readers = {"kind": "swarm", "docs": [("a.md", "text")]}
    await live.post_reset()
    assert s.history == {"sllm": []} and s.readers is None
