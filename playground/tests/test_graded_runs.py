"""Graded runs (live_parts/graded.py, runners.py): which questions the agents get from the answer key, what a run records of
its score, and when a run is not graded. A stub engine answers from the key (right for some agents, wrong or silent for
others); no server, no GPU."""

import asyncio
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from sonic_playground import documents, footprint, live
from sonic_playground.app import app
from sonic_playground.live_parts import graded
from sonic_playground.live_parts.engine import Engine
from sonic_playground.live_parts.events import Bus
from sonic_playground.live_parts.run import run_scenario
from sonic_playground.live_parts.scenarios import RunReq
from sonic_playground.settings import PLAYGROUND_DIR
from test_live import idle_client

sys.path.insert(0, str(PLAYGROUND_DIR / "tools"))
import make_samples  # noqa: E402

DOC = re.compile(r"<document (\S+)>")


@pytest.fixture
def samples(cfg):
    cfg.samples_dir.mkdir()
    make_samples.write(cfg.samples_dir)
    return sorted(f"samples/{p.name}" for p in cfg.samples_dir.glob("*.md"))


def make_session(answer):
    """A session whose engine answers every prompt with answer(prompt, n_asked_so_far)."""
    prompts, seen = [], []

    class StubEngine(Engine):
        async def tokenize(self, text):
            prompts.append(text)
            return [len(prompts) - 1]

        async def generate(self, ids, n, *, lane, on_token=None, ignore_eos=False):
            seen.append(n)
            return {"text": answer(prompts[ids[0]], len(seen) - 1), "ids": [1], "prompt_tokens": 100, "n": 25, "ttft_ms": 5.0,
                    "decode_tok_s": 50.0, "start_s": 0.0, "first_s": 1.0, "last_s": 2.0}

    class StubSession:
        level, thinking, exact = "off", "off", None

        def __init__(self):
            e = StubEngine("sllm", Path("m.gguf"), False)
            e.cells, e.ctx, e.session_peak_mb, e.peak_mb = 65536, 4096, 900.0, 900.0
            e.proc, e.client = SimpleNamespace(pid=1), idle_client()
            self.engines = {"sllm": e}
            self.model, self.compare, self.kvb, self.weight_bytes = {"id": "m"}, False, 32256, 2**30
            self.template, self.lim, self.gpu_util, self.train = SimpleNamespace(eog=[], name="t"), {}, None, None
            self.readers = None

        async def ensure_ctx(self, *a):
            pass

        def use_thinking(self, level):
            pass

        def apply_level(self, req):
            return None

        def one_shot(self, system, message):
            return message

    return StubSession(), prompts


def question_of(prompt):
    return re.findall(r"(?:Question|Q1): (.+)", prompt)[-1].strip()


def right_answer(prompt):
    """What the key accepts for the question asked about the document in `prompt`."""
    key = documents.key_of(DOC.search(prompt).group(1))
    return next(q["accept"][0] for q in key["questions"] if q["q"] == question_of(prompt))


def run(s, scenario, files, message="", agents=8):
    return asyncio.run(run_scenario(s, RunReq(scenario=scenario, message=message, files=files, agents=agents), "r1"))["metrics"]["sllm"]


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(live, "BUS", Bus())
    monkeypatch.setattr(footprint, "footprint_bytes", lambda pid: 500 * 1048576)


def test_a_swarm_over_keyed_samples_asks_each_agent_a_keyed_question_and_counts_right_wrong_and_empty(samples, monkeypatch):
    events = []
    monkeypatch.setattr(live.BUS, "publish", events.append)

    def answer(prompt, i):
        if "Summarise what these notes say" in prompt:
            return "A merged answer."
        return {0: "", 1: "I have no idea."}.get(i % 4, right_answer(prompt))  # agents 0, 4: silent; 1, 5: wrong; the rest right

    s, _ = make_session(answer)
    m = run(s, "swarm", samples[:8])
    assert (m["graded"], m["correct"], m["wrong"], m["empty"]) == (True, 4, 2, 2)
    rows = m["graded_agents"]
    assert [r["verdict"] for r in rows] == ["no answer", "wrong", "correct", "correct", "no answer", "wrong", "correct", "correct"]
    assert all(r["document"] == samples[i] and r["expected"] and r["question"].endswith("?") for i, r in enumerate(rows))
    assert len({r["question"] for r in rows}) > 3  # the questions are spread over the agents, not one for all
    done = [e for e in events if e.get("state") == "done"]
    assert len(done) == 8 and {e["verdict"] for e in done} == {"correct", "wrong", "no answer"}
    assert done[2]["question"] == rows[2]["question"] and done[2]["expected"] == rows[2]["expected"] and done[2]["document"] == samples[2]


def test_the_second_question_of_a_followup_is_not_the_first(samples):
    s, prompts = make_session(lambda p, i: right_answer(p) if "Question:" in p else "ok")
    first = run(s, "swarm", samples[:12], agents=12)
    follow = run(s, "followup", [])
    assert follow["graded"] and follow["correct"] == 12
    assert all(a["question"] != b["question"] and a["document"] == b["document"] for a, b in zip(first["graded_agents"], follow["graded_agents"]))


def test_many_questions_on_one_document_cycle_through_the_keyed_questions_and_say_so(samples):
    s, _ = make_session(lambda p, i: right_answer(p))
    m = run(s, "shared", [samples[0]], agents=7)
    qs = [r["question"] for r in m["graded_agents"]]
    assert m["correct"] == 7 and len(set(qs)) == 5 and qs[5] == qs[0]
    assert m["graded_note"] == "2 of 7 agents repeat a question: this document has 5."


def test_a_team_gives_each_agent_one_keyed_question_as_its_Q1(samples):
    s, prompts = make_session(lambda p, i: f"Q1: {right_answer(p)}" if "Questions:" in p and "<document findings>" not in p else "merged")
    m = run(s, "team", samples[:6])
    assert (m["graded"], m["correct"], m["wrong"]) == (True, 6, 0)
    assert all(p.count("\nQ1: ") == 1 and "Q2:" not in p for p in prompts if "Questions:\nQ1:" in p)


def test_classify_uses_the_document_kinds_and_extract_the_keyed_fields(samples):
    kinds = {n: documents.key_of(n)["kind"] for n in samples[:10]}
    s, prompts = make_session(lambda p, i: kinds[DOC.search(p).group(1)] if i % 5 else "recipe or invoice")
    m = run(s, "classify", samples[:10], agents=10)
    assert (m["correct"], m["wrong"]) == (8, 2)  # two replies name two labels: hedging
    assert "meeting notes, product manual" in prompts[0] and m["graded_agents"][0]["expected"] == "meeting notes"

    def fields(p, i):
        k = documents.key_of(DOC.search(p).group(1))["fields"]
        return json.dumps(k if i % 2 else {**k, next(iter(k)): "nobody"})

    s, prompts = make_session(fields)
    m = run(s, "extract", samples[:4])
    assert (m["correct"], m["wrong"]) == (2, 2) and "Give these fields: chair, meeting_date" in prompts[0]
    assert json.loads(m["graded_agents"][0]["expected"])["chair"]


def test_a_run_with_your_own_question_or_an_unkeyed_document_is_not_graded(samples, cfg):
    s, _ = make_session(lambda p, i: "fine")
    m = run(s, "swarm", samples[:3], message="What is this?")
    assert m["graded"] is False and "own question" in m["graded_note"] and "correct" not in m
    documents.save_upload("mine.md", b"my own text")
    m = run(s, "swarm", [samples[0], "uploads/mine.md"])
    assert m["graded"] is False and "no answer key" in m["graded_note"]
    m = run(s, "classify", samples[:2], message="recipe, other")
    assert m["graded"] is False


def test_tests_without_a_right_answer_say_they_are_not_graded(samples):
    s, _ = make_session(lambda p, i: "text")
    for scenario in ("summarize", "compress", "burst"):
        m = run(s, scenario, samples[:2], message="in 3 sentences")
        assert m["graded"] is False and m["graded_note"] == "Not graded: this test has no single right answer."


def test_plan_spreads_questions_deterministically(samples):
    items = graded.plan("swarm", "", samples[:40])
    assert [i.question for i in items] == [i.question for i in graded.plan("swarm", "", samples[:40])]
    assert graded.plan("swarm", "", samples[:2] + ["uploads/x.md"]) is None and graded.plan("chat", "", samples[:2]) is None


def test_the_room_route_says_how_many_agents_fit(samples, monkeypatch):
    s = live.Session.__new__(live.Session)
    s.engines, s.train, s.cells_max, s.state, s.readers = {"sllm": Engine("sllm", Path("m.gguf"), False)}, 32768, 262144, "ready", None
    s.sampler = s.run_task = None
    with TestClient(app) as c:
        monkeypatch.setattr(live, "SESSION", None)
        assert c.post("/api/live/agent-room", json={"scenario": "swarm"}).status_code == 409
        monkeypatch.setattr(live, "SESSION", s)
        r = c.post("/api/live/agent-room", json={"scenario": "swarm", "files": samples[:4]}).json()
        assert r["max"] == 64 and 2000 < r["tokens_per_agent"] < 4096  # 262144 cells, a context of 4096 per agent
        assert c.post("/api/live/agent-room", json={"scenario": "classify", "files": samples[:4]}).json()["max"] == 64  # the engine's contexts double: 102 agents would get 2048 tokens each, less than a document needs
        assert c.post("/api/live/agent-room", json={"scenario": "chat"}).json() == {"max": 1, "tokens_per_agent": None}
        assert c.post("/api/live/agent-room", json={"scenario": "turns"}).json()["max"] == 256
        assert c.post("/api/live/agent-room", json={"scenario": "nope"}).status_code == 422
        monkeypatch.setattr(live, "SESSION", None)  # not stopped by the app's shutdown
