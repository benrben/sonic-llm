"""live.py's error handlers that used to be silent: a failure now shows as one log line, and nothing new is raised.

No GPU, no server: httpx.MockTransport stands in for sllm-server."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException

from types import SimpleNamespace

from sonic_playground.live_parts.engine import Engine
from sonic_playground.live_parts.scenarios import RunReq
from sonic_playground.live_parts.session import Session
from sonic_playground.live_parts.run import _note_spec, _scenario_metrics
from sonic_playground.live_parts.turns import _turns
from sonic_playground import live
from sonic_playground.chat_template import GEMMA3


@pytest.fixture
def logs(monkeypatch):
    out: list[str] = []
    monkeypatch.setattr(live, "log", out.append)
    return out


def engine(handler, up=True) -> Engine:
    e = Engine("sllm", Path("m.gguf"), True)
    e.port, e.up = 1, up
    e.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return e


def refused(request):
    raise httpx.ConnectError("connection refused", request=request)


# -- polled reads: one line when they start failing, not one per tick --------------------------------------------------
async def test_a_failing_metrics_poll_logs_once_and_again_only_after_it_worked(logs):
    state = {"fail": True}

    def handler(request):
        if state["fail"]:
            refused(request)
        return httpx.Response(200, json={"prompt_tokens_evaluated": 42})

    e = engine(handler)
    assert [await e.prompt_tokens_read() for _ in range(5)] == [None] * 5
    assert len(logs) == 1 and "/metrics" in logs[0] and "ConnectError" in logs[0]
    state["fail"] = False
    assert await e.prompt_tokens_read() == 42 and len(logs) == 1  # working again: no line
    state["fail"] = True
    assert await e.prompt_tokens_read() is None and await e.prompt_tokens_read() is None
    assert len(logs) == 2  # it broke again: one more line


async def test_an_engine_that_is_not_up_or_stopping_does_not_log_its_failing_reads(logs):
    e = engine(refused, up=False)  # starting: health has not answered yet
    assert await e.prompt_tokens_read() is None and await e.real_kv_bytes_per_token() is None
    e = engine(refused)
    await e.stop()  # no process: stop only closes the client and marks the engine down
    assert await e.prompt_tokens_read() is None
    assert logs == []


async def test_a_failing_caps_read_logs_once_and_returns_none(logs):
    e = engine(lambda request: httpx.Response(200, json={"engine": {}}))
    assert await e.real_kv_bytes_per_token() is None and await e.real_kv_bytes_per_token() is None
    assert len(logs) == 1 and "KV bytes per token" in logs[0] and "KeyError" in logs[0]


async def test_the_spec_rate_read_after_a_run_logs_and_leaves_the_metrics_alone(logs):
    e = engine(refused)
    m = {"a": 1}
    await _note_spec(e, m)
    assert m == {"a": 1} and len(logs) == 1 and "speculation" in logs[0]
    ok = engine(lambda request: httpx.Response(200, json={"engine": {"spec": {"accept_rate": 0.51234}}}))
    await _note_spec(ok, m)
    assert m["spec_accept_rate_total"] == 0.512 and len(logs) == 1


# -- the session -------------------------------------------------------------------------------------------------------------
class BadProc:
    pid = 4242


async def test_session_stop_stops_every_engine_even_when_one_fails_to_stop():
    stopped = []

    class E:
        def __init__(self, name):
            self.name = name

        async def stop(self):
            stopped.append(self.name)

    s = object.__new__(Session)
    s.engines, s.sampler, s.state = {"sllm": E("a"), "llama": E("b")}, None, "ready"
    await s.stop()
    assert stopped == ["a", "b"] and s.state == "stopped"


def test_the_exit_hook_kills_the_engines_process_groups(monkeypatch):
    killed = []
    monkeypatch.setattr(live.os, "killpg", lambda pid, sig: killed.append(pid))
    s = object.__new__(Session)
    s.engines = {"sllm": SimpleNamespace(proc=BadProc()), "llama": SimpleNamespace(proc=None)}
    monkeypatch.setattr(live, "SESSION", s)
    live._kill_engines_at_exit()
    assert killed == [4242]


# -- read ahead ---------------------------------------------------------------------------------------------------------
async def test_a_failing_read_ahead_is_logged_and_the_loop_ends_as_before(logs, monkeypatch):
    s = object.__new__(Session)
    s.state, s.busy = "ready", asyncio.Lock()
    monkeypatch.setattr(live, "SESSION", s)
    monkeypatch.setattr(live, "_PRE", {"task": None, "next": live.PrereadReq(message="hi"), "reads": 0, "tokens": 0})

    async def boom(s, req):
        raise httpx.ReadTimeout("engine too slow")

    monkeypatch.setattr(live, "preread_once", boom)
    await live._preread_loop(s)
    assert live._PRE["next"] is None and live._PRE["reads"] == 0
    assert len(logs) == 1 and "read ahead failed" in logs[0] and "ReadTimeout" in logs[0]


# -- the turns: a failing DELETE leaks an agent; it is logged, the run goes on --------------------------------------------
class AgentsWithoutDelete:
    """sllm-server's /v1/agents, except that DELETE fails."""

    def __init__(self):
        self.made, self.deletes = 0, []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content or b"null")
        if path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(len(body["content"]) // 8 + 1))})
        if path == "/metrics":
            return httpx.Response(200, json={"lifecycle": {}})
        if request.method == "DELETE":
            self.deletes.append(path.rsplit("/", 1)[-1])
            return httpx.Response(500, json={"error": "boom"})
        if path == "/v1/agents":
            self.made += 1
            return httpx.Response(200, json={"id": f"a{self.made}"})
        if path.endswith("/suspend"):
            return httpx.Response(200, json={"suspended": True})
        assert path.endswith("/turn")
        sse = "".join(f"data: {json.dumps({'tokens': [t], 'content': 'x'})}\n\n" for t in range(3)) + 'data: {"stop": true}\n\n'
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})


async def test_a_failing_agent_delete_is_logged_with_the_agent_id_and_the_run_finishes(logs, monkeypatch):
    monkeypatch.setattr(live.BUS, "publish", lambda ev: None)
    fake = AgentsWithoutDelete()
    s = object.__new__(Session)
    s.template, s.kvb = GEMMA3, None
    e = engine(fake.handler)
    e.ctx, e.cells = 1280, 2 * 1280
    s.engines = {"sllm": e}
    m = await _turns(s, RunReq(scenario="turns", agents=2, parallel=2), "r1", "sllm")
    assert m["same_as_ram"] is True and m["agents"] == 2
    assert sorted(fake.deletes) == ["a1", "a2", "a3"]  # the two agents and the same-in-RAM check, each asked once
    lines = [x for x in logs if "deleting agent" in x]
    assert len(lines) == 3 and all("HTTPStatusError" in x for x in lines) and any("a3" in x for x in lines)


# -- the dispatcher keeps the old answer to an unknown scenario ---------------------------------------------------------
async def test_an_unknown_scenario_is_still_a_422():
    with pytest.raises(HTTPException) as ex:
        await _scenario_metrics(object.__new__(Session), RunReq(scenario="nope"), "r1", "sllm", 0.0)
    assert ex.value.status_code == 422 and "nope" in ex.value.detail
