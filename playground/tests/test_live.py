"""The pure pieces and the small engine parts of the live playground that need no server: how a session sizes its memory, how
replies are read and counted, the event stream, an engine's completion and its peaks."""

import json
import os
import subprocess
from pathlib import Path

import httpx
import pytest

from sonic_playground import footprint, live
from sonic_playground.live_parts.agents import _agents, _read_only_tok_s, _write_tok_s, fit_prompt, prompt_room
from sonic_playground.live_parts.config import ANSWER_ROOM
from sonic_playground.live_parts.engine import Engine, _engine_pids, _find_key, _remember_engine, deferred_wait_s, stop_orphans
from sonic_playground.live_parts.events import Bus, _rate
from sonic_playground.live_parts.scenarios import (DOC_OUT_MIN, DOC_TASKS, DOC_THINK_ROOM, SCENARIOS, RunReq, doc_room,
                                                   doc_verdict, json_of, parse_report, pick_label, questions_of)
from sonic_playground.live_parts.run import run_scenario
from sonic_playground.live_parts.session import Session, exact_status, kv_plan


@pytest.fixture
def lib(cfg):
    """A library of two documents."""
    cfg.samples_dir.mkdir()
    (cfg.samples_dir / "a.md").write_text("# A\n" + "word " * 400)
    (cfg.samples_dir / "b.txt").write_text("short")


def idle_client():
    """An HTTP client whose server answers every request with an empty JSON object."""
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))


def test_team_questions_and_reports():
    assert questions_of("What was decided?\n- What is open?\n\n") == ["What was decided?", "What is open?"]
    assert questions_of("Who? When? Why?") == ["Who?", "When?", "Why?"]
    rep = "Q1: The team chose generic.\nQ2: NOT FOUND\n**Q3.** 291 GB/s\nQ9: out of range"
    assert parse_report(rep, 3) == {1: "The team chose generic.", 3: "291 GB/s"}


def test_find_key_reads_nested_counters():
    assert _find_key({"requests": {"x": 1, "prompt_tokens_evaluated": 512}}, "prompt_tokens_evaluated") == 512
    assert _find_key([{"a": {}}, {"b": {"prompt_tokens_evaluated": 3.0}}], "prompt_tokens_evaluated") == 3
    assert _find_key({"a": 1}, "prompt_tokens_evaluated") is None


def test_sllm_ctx_follows_the_scenario():
    s = Session.__new__(Session)
    e = Engine("sllm", Path("m.gguf"), False)
    s.engines, s.train, s.cells_max = {"sllm": e}, 32768, 65536
    assert s.ctx_for("burst") == 1024 and s.cells_max // s.ctx_for("burst") == 64
    assert s.ctx_for("team", 16) == 4096  # 16 agents x 4096 cells fill the cache
    assert s.ctx_for("team", 57) == 1024  # all 57 at once
    assert s.ctx_for("team", 500) == 1024  # more than fit even at 1024: the rest wait
    assert s.ctx_for("team", 2) == 32768  # few agents: the longest context the model was trained for
    assert s.ctx_for("team", 2, cap=8192) == 8192  # an optional limit wins
    assert s.ctx_for("long") == 32768  # capped by the trained context
    for task in DOC_TASKS:
        assert s.ctx_for(task, 16) == 4096 and s.streams_for(task, 16) == 16  # all at once, like the swarm
        assert s.ctx_for(task, 16, need=1500) == 1536  # a job's window holds the job and no more: agents x window is the RAM
        assert s.ctx_for(task, 16, need=1 << 30) == 4096  # never larger than all agents at once allow
        assert s.ctx_for(task, 2, need=1 << 30) == 32768  # nor than the trained context


def test_doc_room_ends_a_reply_that_would_not_stop():
    room = doc_room
    extract, classify = DOC_TASKS["extract"], DOC_TASKS["classify"]
    assert room(extract, "off", 106) == DOC_OUT_MIN  # a tiny document still has room for its answer
    assert room(extract, "off", 9000) == 9000  # never longer than what was read
    assert room(classify, "off", 9000) == 24  # a job with its own length keeps it
    assert room(extract, "high", 106) == DOC_OUT_MIN + DOC_THINK_ROOM  # thinking is not cut
    assert room(extract, "max", 106) is None  # "max" promises no limit


class FakeAgentEngine:
    """Answers at once; `tokens` is how many tokens a reply streams."""
    tokens = 0

    def __init__(self, tokens=0):
        self.asked, self.n, self.ctx, self.cells = [], tokens, 32768, 32768

    async def tokenize(self, text):
        return text.split()

    async def generate(self, ids, n, lane=None, on_token=None):
        self.asked.append(n)
        for _ in range(self.n):
            on_token("x")
        return {"prompt_tokens": len(ids), "text": "the reply", "ids": [], "n": self.n, "stop": "eos"}


def agent_session(e):
    s = Session.__new__(Session)
    s.engines, s.lim = {"sllm": e}, {"answer": None, "agent": None, "doc": None}
    s.template = type("T", (), {"eog": ()})
    return s


def test_agents_take_the_reply_limit_of_their_prompt_and_a_run_limit_wins():
    import asyncio

    e = FakeAgentEngine()
    s = agent_session(e)
    run = lambda: asyncio.run(_agents(s, "sllm", "r", [("a", "w " * 106)], room=lambda n: n + 1))
    run()
    s.lim["agent"] = 64
    run()
    assert e.asked == [107, 64]


def test_an_agent_tells_the_page_it_is_writing_and_ends_with_its_whole_reply(monkeypatch):
    import asyncio

    seen = []
    monkeypatch.setattr(live.BUS, "publish", seen.append)
    s = agent_session(FakeAgentEngine(tokens=50))
    asyncio.run(_agents(s, "sllm", "r", [("a.md", "w")], extra=lambda r: {"ok": True}))
    assert [x["state"] for x in seen] == ["reading", "writing", "done"]  # a count at most once per tick, not one event per token
    assert seen[1]["tokens"] == 1
    done = seen[-1]
    assert (done["reply"], done["tokens"], done["stop"], done["ok"]) == ("the reply", 50, "eos", True)


def test_doc_verdict_says_how_each_job_came_out():
    r = {"n": 12, "prompt_tokens": 120, "stop": "eos"}
    assert doc_verdict("extract", "a, b", '{"a": 1}', r) == (True, "valid JSON")
    assert doc_verdict("extract", "a, b", "nope", r) == (False, "not valid JSON")
    assert doc_verdict("extract", "a, b", '{"a"', {**r, "stop": "limit"}) == (False, "cut at the length limit")
    assert doc_verdict("classify", "plan, other", "plan", r) == (True, "plan")
    assert doc_verdict("classify", "plan, other", "maybe", r) == (False, "not one of the labels")
    assert doc_verdict("summarize", "in 3 sentences", "text", r) == (True, "10% of what was read")


def test_stop_ends_the_run_after_a_restart_finishes_and_keeps_no_record(monkeypatch):
    import asyncio

    seen = []
    monkeypatch.setattr(live.BUS, "publish", seen.append)

    async def main():
        s = Session.__new__(Session)
        s.state, s.busy, s.engines, s.restarting, s.last_used = "ready", asyncio.Lock(), {}, True, 0.0
        s.use_thinking = lambda level: None

        async def forever(*_):
            await asyncio.sleep(60)

        monkeypatch.setattr(live, "SESSION", s)
        monkeypatch.setattr(live, "run_scenario", forever)
        await live.post_run(RunReq())
        await asyncio.sleep(0)
        stop = asyncio.create_task(live.post_run_stop())
        await asyncio.sleep(0.12)
        assert not stop.done() and not s.run_task.cancelled()  # the engine is still restarting
        s.restarting = False
        assert await stop == {"stopped": True}
        await s.run_task
        assert not s.busy.locked()
        import pytest

        with pytest.raises(live.HTTPException):  # nothing left to stop
            await live.post_run_stop()

    asyncio.run(main())
    assert seen[-1]["type"] == "run_done" and seen[-1]["stopped"] is True


def test_a_limit_past_the_context_does_not_cut_the_document_away():
    import asyncio

    assert prompt_room(16384, None) == 16384 - ANSWER_ROOM - 8
    assert prompt_room(16384, 64) == 16384 - 64 - 8
    assert prompt_room(16384, 20000) == 16384 // 2 - 8  # a negative room used to cut the prompt to its question
    e = FakeAgentEngine()
    e.ctx, e.cells = 16384, 16384
    s = agent_session(e)
    s.lim["agent"] = 20000
    seen = []
    live.BUS.publish, keep = seen.append, live.BUS.publish
    try:
        asyncio.run(_agents(s, "sllm", "r", [("a", "sys <document a>\n" + "w " * 100 + "\n</document>\n\nQuestion: what?")]))
    finally:
        live.BUS.publish = keep
    assert seen[-1]["prompt_tokens"] > 100  # the whole document went in


def test_every_scenario_is_in_a_group():
    assert {x["group"] for x in SCENARIOS} == {"Talk", "Documents", "Agents", "Engine tests"}


def test_fit_prompt_cuts_the_document_not_the_question():
    import asyncio

    class E:
        async def tokenize(self, text):
            return text.split()

    p = "sys <document a.md>\n" + "w " * 500 + "\n</document>\n\nQuestion: what?"
    ids = asyncio.run(fit_prompt(E(), p, 100))
    assert len(ids) <= 100 and ids[-2:] == ["Question:", "what?"]




def test_stream_replays_missed_events():
    import asyncio

    bus = Bus()
    bus.publish({"type": "tick", "n": 1})
    bus.publish({"type": "tick", "n": 2})

    async def first_two():
        out = []
        async for chunk in bus.stream(lifetime_s=0.2, last_id=1):
            out.append(chunk)
            if len(out) == 2:
                break
        return out

    retry, ev = asyncio.run(first_two())
    assert retry.startswith("retry:")
    assert ev.startswith("id: 2\n") and '"n": 2' in ev


def test_rate_averages_over_the_window():
    from collections import deque

    pts = deque([(0.0, 0)])
    assert _rate(pts, 0.5, 0) == 0.0  # the counter did not move this tick ...
    assert _rate(pts, 1.0, 100) == 100.0  # ... then jumped: 100 tokens over the 1 s so far
    assert _rate(pts, 3.0, 100) == 0.0  # 2 s later nothing new: the window has slid past the jump






def test_write_rate_is_over_the_writing_time_only():
    rs = [{"n": 100, "first_s": 10.0, "last_s": 12.0}, {"n": 100, "first_s": 11.0, "last_s": 14.0}]
    assert _write_tok_s(rs) == 50.0  # 200 tokens from the first first-token (10) to the last last-token (14)
    assert _write_tok_s([{"n": 5}]) is None  # no timings: no rate


def test_read_only_rate_counts_evaluated_tokens_until_the_last_first_token():
    rs = [{"prompt_tokens": 1000, "cached_tokens": 200, "first_s": 5.0}, {"prompt_tokens": 1000, "cached_tokens": None, "first_s": 9.0}]
    assert _read_only_tok_s(rs, 1.0) == 225.0  # (800 + 1000) tokens in 8 s
    assert _read_only_tok_s([{"prompt_tokens": 1}], 1.0) is None


def test_deferred_503_is_retried_after_retry_after():
    body = json.dumps({"error": {"type": "deferred", "retry_after_ms": 250}}).encode()
    assert deferred_wait_s(503, {}, body) == 0.25
    assert deferred_wait_s(503, {"retry-after": "2"}, json.dumps({"error": {"type": "deferred"}}).encode()) == 2.0
    assert deferred_wait_s(503, {}, b'{"error": {"type": "overloaded"}}') is None
    assert deferred_wait_s(500, {}, body) is None


def test_generate_waits_out_a_deferral_and_counts_it_in_ttft(monkeypatch):
    import asyncio

    import httpx

    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": {"type": "deferred", "retry_after_ms": 200}})
        return httpx.Response(200, content=b'data: {"content": "hi", "tokens": [7]}\n\ndata: {"stop": true, "stop_type": "eos"}\n\n')

    e = Engine("sllm", Path("m.gguf"), False)
    e.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    r = asyncio.run(e.generate([1, 2, 3], 8, lane="background"))
    assert len(calls) == 2 and r["deferred"] == 1 and r["ids"] == [7]
    assert r["ttft_ms"] >= 190  # the wait is in time to first token


def test_engine_client_has_no_connection_cap():
    import inspect

    assert "max_connections=None" in inspect.getsource(Engine.start)  # the default 100 would queue agents in the client


def test_run_peaks_reset_and_sample_metrics():
    e = Engine("sllm", Path("m.gguf"), False)
    e.peak_mb = e.session_peak_mb = 9000.0
    e.max_running, e.kv_used_peak_cells = 3, 5
    e.begin_run()  # no process: the footprint is unknown, the run peaks restart
    assert (e.peak_mb, e.max_running, e.kv_used_peak_cells) == (0.0, 0, 0) and e.session_peak_mb == 9000.0
    e.note_metrics({"scheduler": {"running": 7}, "kv": {"sequences": [{"id": "a", "cells": 100}, {"id": "b", "cells": 50}]}})
    e.note_metrics({"scheduler": {"running": 4}, "kv": {"sequences": [{"id": "a", "cells": 10}]}})
    assert e.max_running == 7 and e.kv_used_peak_cells == 150






def test_burst_record_has_honest_numbers_and_per_run_memory(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    class FakeEngine(Engine):
        async def tokenize(self, text):
            return [1, 2, 3]

        async def generate(self, ids, n, *, lane, on_token=None, ignore_eos=False):
            i = self.active = self.active + 1
            return {"text": "x", "ids": [1], "prompt_tokens": 3, "n": 100, "ttft_ms": 5.0, "decode_tok_s": 50.0,
                    "start_s": 0.0, "first_s": 10.0 + i, "last_s": 12.0 + i}

    class FakeSession:
        def __init__(self):
            e = FakeEngine("sllm", Path("m.gguf"), False)
            e.cells, e.ctx, e.session_peak_mb, e.peak_mb = 65536, 1024, 9000.0, 9000.0
            e.proc, e.client = SimpleNamespace(pid=1), idle_client()
            self.engines = {"sllm": e}
            self.model, self.compare, self.kvb, self.weight_bytes = {"id": "m"}, False, 32256, 4 * 2**30
            self.template, self.lim, self.gpu_util, self.train = SimpleNamespace(eog=[], name="t"), {}, None, None

        async def ensure_ctx(self, *a, **k):
            pass

        def use_thinking(self, level):
            pass

        level, thinking, exact = "off", "off", None

        def apply_level(self, req):
            return None

        def one_shot(self, system, message):
            return message

    monkeypatch.setattr(live, "BUS", Bus())
    monkeypatch.setattr(footprint, "footprint_bytes", lambda pid: 500 * 1048576)
    s = FakeSession()
    rec = asyncio.run(run_scenario(s, RunReq(scenario="burst", agents=3), "r1"))
    m = rec["metrics"]["sllm"]
    assert m["agents"] == 3 and m["agents_asked"] == 3
    assert m["write_tok_s"] == 75.0  # 300 tokens from first token 11 to last token 15 = 4 s (not the whole run)
    assert "agg_tok_s" in m and "bandwidth_pct" not in m and "max_parallel" not in m
    assert m["peak_mem_mb"] == 500.0 and m["session_peak_mem_mb"] == 9000.0  # the per-run peak restarted; the session's is kept
    assert m["kv_reserved_mb"] == round(65536 * 32256 / 2**20, 1) and m["weights_mb"] == 4096.0


def test_exact_status_follows_the_engine_caps():
    caps = lambda **e: {"engine": e}
    assert exact_status(caps(row_exact=True, kv_read={"mode": "f16"}, kv={"type": "q4_0"}))["level"] == "exact"
    assert exact_status(caps(row_exact=True, kv_read={"mode": "kv"}, kv={"type": "f16"}))["level"] == "writing"
    assert exact_status(caps(row_exact=True, kv_read={"mode": "kv"}, kv={"type": "q4_0"}))["level"] == "no"
    assert exact_status(caps(row_exact=False, kv_read={"mode": "f16"}, kv={"type": "q4_0"}))["level"] == "no"
    assert exact_status({})["level"] == "no"


def test_a_label_counts_only_when_the_reply_is_one():
    labels = ["bug report", "plan", "other"]
    assert pick_label("plan", labels) == "plan"
    assert pick_label(" **Plan**. ", labels) == "plan"
    assert pick_label("The label is: bug report", labels) == "bug report"
    assert pick_label("plan or other", labels) is None  # two labels: not an answer
    assert pick_label("planning", labels) is None  # part of a word
    assert pick_label("", labels) is None


def test_a_reply_is_json_only_when_it_parses_to_an_object():
    assert json_of('{"a": 1}') == {"a": 1}
    assert json_of('```json\n{"a": null}\n```') == {"a": None}
    assert json_of("[1, 2]") is None
    assert json_of('Here it is: {"a": 1}') is None
    assert json_of('{"a": 1') is None  # cut off by the token limit


def test_document_tasks_report_what_they_did(lib, monkeypatch):
    import asyncio
    from types import SimpleNamespace

    replies = iter(["plan", "no idea", '{"a": 1}', "not json", "short", "shorter"])

    class FakeEngine(Engine):
        async def tokenize(self, text):
            return [1, 2, 3]

        async def generate(self, ids, n, *, lane, on_token=None, ignore_eos=False):
            seen.append(n)
            return {"text": next(replies), "ids": [1], "prompt_tokens": 100, "n": 25, "ttft_ms": 5.0, "decode_tok_s": 50.0,
                    "start_s": 0.0, "first_s": 1.0, "last_s": 2.0}

    class FakeSession:
        def __init__(self):
            e = FakeEngine("sllm", Path("m.gguf"), False)
            e.cells, e.ctx, e.session_peak_mb, e.peak_mb = 65536, 4096, 900.0, 900.0
            e.proc, e.client = SimpleNamespace(pid=1), idle_client()
            self.engines = {"sllm": e}
            self.model, self.compare, self.kvb, self.weight_bytes = {"id": "m"}, False, 32256, 2**30
            self.template, self.lim, self.gpu_util, self.train = SimpleNamespace(eog=[], name="t"), {}, None, None
            self.asked = []

        async def ensure_ctx(self, scenario, agents, *a):
            self.asked.append(agents)

        def use_thinking(self, level):
            pass

        level, thinking, exact = "off", "off", None

        def apply_level(self, req):
            return None

        def one_shot(self, system, message):
            return message

    seen: list[int] = []
    monkeypatch.setattr(live, "BUS", Bus())
    monkeypatch.setattr(footprint, "footprint_bytes", lambda pid: 500 * 1048576)
    s = FakeSession()
    run = lambda task, msg="": asyncio.run(run_scenario(s, RunReq(scenario=task, message=msg, agents=256), "r1"))["metrics"]["sllm"]
    m = run("classify", "plan, other")
    assert m["agents"] == 2 and m["agents_asked"] == 256 and s.asked == [2]  # one per file of the library, not 256
    assert m["labeled"] == 1 and m["labels"] == {"plan": 1}  # "no idea" is not a label, and it counts as none
    assert seen == [24, 24]  # a label needs few tokens
    m = run("extract")  # the first suggestion is the default
    assert m["json_ok"] == 1
    assert seen[2:] != [24, 24]  # a task with no cap takes the room the context has
    m = run("compress")
    assert m["kept_pct"] == 25.0  # 50 tokens written of 200 read


def test_rate_of_a_slow_model_with_long_steps_does_not_jump():
    """Qwen3.8-27B reads 512 tokens per engine step of about 1.4 s; the page samples every 0.5 s. The read is even (366
    tok/s), so the graph must be: a fixed 2 s window drew 256 / 512 / 256."""
    from collections import deque

    pts, seen = deque(), []
    for k in range(1, 61):  # 30 s
        now = 0.5 * k
        r = _rate(pts, now, 512 * int(now / 1.4))
        if now >= 8.0:
            seen.append(r)
    assert min(seen) > 0.85 * 366 and max(seen) < 1.15 * 366, (min(seen), max(seen))
    # the read ends: the rate falls to 0 within a few seconds and stays there
    end = 512 * int(30.0 / 1.4)
    tail = [_rate(pts, 30.0 + 0.5 * k, end) for k in range(1, 12)]
    assert tail[-1] == 0.0 and tail[6] == 0.0
    # a fast model (the counter moves every tick) still follows a change of speed within about 2 s
    pts = deque()
    c = 0
    for k in range(1, 21):
        c += 1000 if k <= 10 else 100
        r = _rate(pts, 0.5 * k, c)
    assert 190 < r < 230, r


def test_kv_plan_reads_the_engines_own_plan(monkeypatch):
    seen = {}

    class P:
        stdout = json.dumps({"budget": {"kv_alloc_bytes": 32256 * 4, "kv_cells": {"allocated": 4, "max_fitting": 1000}}})

    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: seen.update(argv=argv) or P())
    assert kv_plan(Path("m.gguf")) == (32256, 1000)
    assert seen["argv"][1:3] == ["--model", "m.gguf"] and seen["argv"][-1] == "--print-profile"


def test_kv_plan_that_cannot_be_read_says_so_and_gives_no_plan(monkeypatch):
    lines = []
    monkeypatch.setattr(live, "log", lines.append)
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: type("P", (), {"stdout": "no json"})())
    assert kv_plan(Path("m.gguf")) == (None, None)
    assert "memory plan" in lines[0]


def test_stop_orphans_keeps_live_children_and_forgets_dead_pids():
    _remember_engine(999999)  # no such process
    _remember_engine(os.getpid())  # alive, but not an engine binary
    assert stop_orphans() == []
    assert _engine_pids() == []
