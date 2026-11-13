"""The first token of an agent turn (live_parts/turns.py `_TurnReader`): a stream whose first chunk carries text but no token
ids still has its first token there, so ttft and `on_first` belong to that chunk and happen once.

No GPU, no server: a canned stream stands in for sllm-server's /v1/agents/{id}/turn."""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx

from sonic_playground.live_parts.engine import Engine
from sonic_playground.live_parts import turns


def engine_streaming(chunks: list[dict]) -> Engine:
    sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + 'data: {"stop": true}\n\n'
    e = Engine("sllm", Path("m.gguf"), True)
    e.port, e.up = 1, True
    e.client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})))
    return e


async def test_on_first_fires_once_when_the_first_chunk_has_text_but_no_token_ids():
    e = engine_streaming([{"content": "He"}, {"tokens": [7], "content": "llo"}, {"tokens": [8], "content": "!"}])
    firsts, pieces = [], []
    r = await turns.agent_turn(e, "a1", [1, 2], 10, on_first=lambda: firsts.append(1), on_token=pieces.append)
    assert len(firsts) == 1
    assert pieces == ["He", "llo", "!"] and r["text"] == "Hello!" and r["ids"] == [7, 8]


def test_ttft_is_the_time_of_the_first_chunk_with_text_even_without_token_ids(monkeypatch):
    clock = iter([1.0, 1.0, 5.0, 5.0, 9.0, 9.0])  # two reads per first token: ttft_ms and t_first
    monkeypatch.setattr(turns, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    reader = turns._TurnReader(SimpleNamespace(tokens=0), 0.0)
    reader.take({"content": "He"})  # text, no ids: the first token
    reader.take({"tokens": [7], "content": "llo"})
    assert reader.out["ttft_ms"] == 1000.0 and reader.out["t_first"] == 1.0
