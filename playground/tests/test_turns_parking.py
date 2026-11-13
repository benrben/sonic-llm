"""Agents taking turns (live_parts/turns.py): an agent whose park on the SSD failed is not shown as parked.

No GPU, no server: httpx.MockTransport acts as sllm-server's /v1/agents API."""

import json
from pathlib import Path

import httpx

from sonic_playground.live_parts.engine import Engine
from sonic_playground.live_parts.scenarios import TURNS_ROUNDS
from sonic_playground.live_parts.turns import _turns
from sonic_playground import live
from sonic_playground.chat_template import GEMMA3


class AgentsThatCannotPark:
    """sllm-server's /v1/agents, except that suspend fails."""

    def __init__(self):
        self.made = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content or b"null")
        if path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(len(body["content"]) // 8 + 1))})
        if path == "/metrics":
            return httpx.Response(200, json={"lifecycle": {}})
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        if path == "/v1/agents":
            self.made += 1
            return httpx.Response(200, json={"id": f"a{self.made}"})
        if path.endswith("/suspend"):
            return httpx.Response(500, json={"error": "no room on the SSD"})
        sse = "".join(f"data: {json.dumps({'tokens': [t], 'content': 'x'})}\n\n" for t in range(3)) + 'data: {"stop": true}\n\n'
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})


async def test_an_agent_whose_park_failed_is_not_shown_as_parked(monkeypatch):
    seen, logs = [], []
    monkeypatch.setattr(live.BUS, "publish", seen.append)
    monkeypatch.setattr(live, "log", logs.append)
    s = object.__new__(live.Session)
    s.template, s.kvb = GEMMA3, None
    e = Engine("sllm", Path("m.gguf"), True)
    e.port, e.up = 1, True
    e.client = httpx.AsyncClient(transport=httpx.MockTransport(AgentsThatCannotPark().handler))
    e.ctx, e.cells = 1280, 2 * 1280
    s.engines = {"sllm": e}
    m = await _turns(s, live.RunReq(scenario="turns", agents=2, parallel=2), "r1", "sllm")
    states = [ev["state"] for ev in seen if ev.get("type") == "agent"]
    assert "parked" not in states  # it stayed in RAM: the row says it waits for its next turn
    assert m["suspend_errors"] == 2 * (TURNS_ROUNDS - 1)
    assert sum("suspend failed" in x for x in logs) == m["suspend_errors"]
