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

DOC = re.compile(r"<(?:document|file) (\S+)>")


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
            return f"{system}\n{message}"

    return StubSession(), prompts


def question_of(prompt):
    return re.findall(r"(?:Question|Q1): (?:About \S+: )?(.+)", prompt)[-1].strip()


def key_question(prompt):
    """The key's question asked about the document in `prompt`."""
    key = documents.key_of(DOC.search(prompt).group(1))
    return next(q for q in key["questions"] if q["q"] == question_of(prompt))


def right_answer(prompt):
    """What the key accepts for the question asked about the document in `prompt`."""
    return key_question(prompt)["accept"][0]


def run(s, scenario, files, message="", agents=8, difficulty="mixed"):
    req = RunReq(scenario=scenario, message=message, files=files, agents=agents, difficulty=difficulty)
    return asyncio.run(run_scenario(s, req, "r1"))["metrics"]["sllm"]


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
    assert all(r["document"] == samples[i] and r["expected"] and r["question"] for i, r in enumerate(rows))
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
    m = run(s, "shared", [samples[0]], agents=11)
    qs = [r["question"] for r in m["graded_agents"]]
    assert m["correct"] == 11 and len(set(qs)) == 8 and qs[8] == qs[0]
    assert m["graded_note"] == "3 of 11 agents repeat a question: this document has 8."
    hard = run(s, "shared", [samples[0]], agents=6, difficulty="hard")
    assert {r["level"] for r in hard["graded_agents"]} == {"hard"} and hard["correct"] == 6


def test_a_team_gives_each_agent_one_keyed_question_as_its_Q1(samples):
    s, prompts = make_session(lambda p, i: f"Q1: {right_answer(p)}" if "Questions:" in p and "<document findings>" not in p else "merged")
    m = run(s, "team", samples[:6])
    assert (m["graded"], m["correct"], m["wrong"]) == (True, 6, 0)
    assert all(p.count("\nQ1: ") == 1 and "Q2:" not in p for p in prompts if "Questions:\nQ1:" in p)


def test_classify_uses_the_eight_kinds_and_on_hard_the_rating_inside_the_kind_of_each_document(samples):
    kinds = {n: documents.key_of(n)["kind"] for n in samples[:10]}
    s, prompts = make_session(lambda p, i: kinds[DOC.search(p).group(1)] if i % 5 else "invoice or incident report")
    m = run(s, "classify", samples[:10], agents=10)
    assert (m["correct"], m["wrong"]) == (8, 2)  # two replies name two labels: hedging
    assert "project plan, meeting minutes, invoice" in prompts[0] and m["graded_agents"][0]["expected"] == "project plan"
    assert {r["level"] for r in m["graded_agents"]} == {"medium"}

    fine = {n: documents.key_of(n)["fine"] for n in samples[:10]}
    s, prompts = make_session(lambda p, i: fine[DOC.search(p).group(1)]["label"])
    m = run(s, "classify", samples[:10], agents=10, difficulty="hard")
    assert m["correct"] == 10 and {r["level"] for r in m["graded_agents"]} == {"hard"}
    invoice = next(i for i, n in enumerate(samples[:10]) if documents.key_of(n)["kind"] == "invoice")
    assert "Labels: paid, overdue, disputed" in prompts[invoice] and "Labels: small, medium, large" in prompts[0]  # each document with its own labels
    assert m["graded_agents"][invoice]["expected"] in ("paid", "overdue", "disputed")


def test_extract_is_all_or_nothing_per_document_and_also_reports_the_accuracy_per_field(samples):
    def fields(p, i):
        k = documents.key_of(DOC.search(p).group(1))["fields"]
        return json.dumps(k if i % 2 else {**k, next(iter(k)): "nobody"})

    s, prompts = make_session(fields)
    m = run(s, "extract", samples[:4])
    assert (m["correct"], m["wrong"]) == (2, 2) and "Give these fields: project, budget_eur, item_count" in prompts[0]
    assert json.loads(m["graded_agents"][0]["expected"])["project"]
    n = sum(len(documents.key_of(d)["fields"]) for d in samples[:4])
    assert (m["fields_right"], m["fields_asked"]) == (n - 2, n)  # two documents lost one field each
    assert m["graded_agents"][0]["reason"] == "1 of 6 fields wrong" and m["graded_agents"][1]["reason"] == ""


def test_the_difficulty_picks_the_level_of_the_questions_and_the_score_is_broken_down_by_level_and_skill(samples):
    s, _ = make_session(lambda p, i: ("merged" if "<document notes>" in p else right_answer(p) if i % 4 else "no idea"))
    hard = run(s, "swarm", samples[:16], agents=16, difficulty="hard")
    assert {r["level"] for r in hard["graded_agents"]} == {"hard"} and hard["by_level"] == {"hard": [12, 16]}
    medium = run(s, "swarm", samples[:16], agents=16, difficulty="medium")
    assert set(medium["by_level"]) == {"medium"} and medium["by_level"]["medium"][1] == 16
    mixed = run(s, "swarm", samples[:16], agents=16)
    assert mixed["by_level"].keys() == {"medium", "hard"} and mixed["by_level"]["medium"][1] == mixed["by_level"]["hard"][1] == 8
    assert sum(a for a, _ in mixed["by_skill"].values()) == mixed["correct"] and sum(b for _, b in mixed["by_skill"].values()) == 16


def test_a_followup_asks_a_harder_question_than_the_first_and_never_the_same(samples):
    s, _ = make_session(lambda p, i: right_answer(p) if "Question:" in p else "ok")
    first = run(s, "swarm", samples[:12], agents=12, difficulty="medium")
    follow = run(s, "followup", [])
    assert {r["level"] for r in first["graded_agents"]} == {"medium"} and {r["level"] for r in follow["graded_agents"]} == {"hard"}
    assert follow["correct"] == 12


CASE_PICK = 4  # a case of 8 documents


def case_answers(prompt):
    """A team agent's report, and the manager's answer, both from the keys."""
    case = documents.answer_key()["cases"][CASE_PICK]
    qs = {q["q"]: q for q in case["questions"]}
    if "<document findings>" in prompt:  # the manager: one right line per case question, the last one wrong
        asked = re.findall(r"^Q(\d+): (.+)$", prompt, re.M)
        return "\n".join(f"Q{n}: " + (qs[q]["accept"][0] if n != str(len(asked)) else "nobody") for n, q in asked)
    return f"Q1: {right_answer(prompt)}\n" + "\n".join(f"Q{n}: NOT FOUND" for n in range(2, 2 + len(case["questions"])))


def test_a_team_over_a_whole_case_asks_the_case_questions_to_the_manager_and_grades_the_merged_answer(samples):
    case = documents.cases()[CASE_PICK]
    s, prompts = make_session(lambda p, i: case_answers(p))
    m = run(s, "team", case["documents"], agents=8)
    n, case_n = len(case["documents"]), len(documents.answer_key()["cases"][CASE_PICK]["questions"])
    assert m["case_asked"] == case_n and m["case_correct"] == case_n - 1
    assert (m["correct"], m["wrong"]) == (n + case_n - 1, 1) and len(m["graded_agents"]) == n + case_n
    last = m["graded_agents"][-1]
    assert last["document"] == case["id"] and last["skill"] == "cross-document" and last["verdict"] == "wrong"
    agent_prompt = next(p for p in prompts if "Questions:\nQ1:" in p and "<document findings>" not in p)
    assert "\nQ2: " in agent_prompt and "NOT STATED" in agent_prompt  # the case questions and the format rule come along
    assert m["by_skill"]["cross-document"] == [case_n - 1, case_n]


def test_a_team_over_documents_that_are_not_a_whole_case_has_no_case_questions(samples):
    case = documents.cases()[CASE_PICK]
    s, _ = make_session(lambda p, i: f"Q1: {right_answer(p)}" if "Questions:" in p and "<document findings>" not in p else "merged")
    m = run(s, "team", case["documents"][:-1], agents=8)
    assert m["case_asked"] == 0 and len(m["graded_agents"]) == len(case["documents"]) - 1 and m["correct"] == len(case["documents"]) - 1


def test_a_long_document_is_graded_with_an_update_a_negative_and_more_asked_from_memory(samples, monkeypatch):
    from sonic_playground.live_parts import runners
    monkeypatch.setattr(runners, "train_ctx", lambda gguf: 131072)
    long_docs = [n for n in samples if documents.key_of(n)["size"] == "long"][:2]
    s, prompts = make_session(lambda p, i: "x" if "What is it?" in p else right_answer(p))
    s.engines["sllm"].ctx = 131072
    m = run(s, "long", long_docs)
    assert m["graded"] is True and len(m["graded_agents"]) == 6 and m["correct"] == 6 and m["files"] == 2
    assert [r["skill"] for r in m["graded_agents"]][:2] == ["update", "negative"]
    assert all(r["question"].startswith("About ") for r in m["graded_agents"]) and m["by_level"].keys() == {"medium", "hard"}
    assert len([p for p in prompts if "Question:" in p]) == 6  # the first question (asked twice on the same tokens), five more
    ungraded = run(s, "long", long_docs, message="What is it?")
    assert ungraded["graded"] is False and "own question" in ungraded["graded_note"]


def test_every_reason_a_wrong_answer_can_have_is_said_in_a_graded_run_and_shows_in_the_events(samples, monkeypatch):
    """A stub answer source that gives, for each reason, one answer of that kind (to the first question it fits)."""
    wanted = ["gave the old value", "gave the amount in the other currency", "gave the look-alike", "wrong arithmetic", "wrong date math",
              "invented an answer", "did not say NOT STATED", "gave two answers", "wrong answer", "no answer"]
    todo, events = list(wanted), []
    monkeypatch.setattr(live.BUS, "publish", events.append)
    trap = {"gave the old value": "old", "gave the amount in the other currency": "currency", "gave the look-alike": "similar"}

    def wrong_for(q, reason):
        if reason in trap:
            return q["reject"][0] if q.get("trap") == trap[reason] else None
        kinds = {"wrong arithmetic": ("arithmetic", "number", "9"), "wrong date math": ("date-math", "date", "1999-01-01"),
                 "invented an answer": ("negative", "nostated", "Ilse Brandt"), "did not say NOT STATED": ("negative", "nostated", "It does not mention this."),
                 "gave two answers": ("lookup", "number", f"{q['accept'][0]} or 12,345"), "wrong answer": ("lookup", "text", "Zed Zed"), "no answer": ("", None, "")}
        skill, kind, text = kinds[reason]
        return text if (q["skill"] == skill or not skill) and (q["type"] == kind or kind is None) else None

    def answer(prompt, i):
        if "Summarise what these notes" in prompt:
            return "merged"
        q = key_question(prompt)
        for reason in todo:
            text = wrong_for(q, reason)
            if text is not None:
                todo.remove(reason)
                return text
        return q["accept"][0]

    s, _ = make_session(answer)
    m = run(s, "swarm", samples, agents=256)
    assert todo == [] and set(wanted) <= set(m["reasons"]) and m["wrong"] + m["empty"] == len(wanted)
    done = [e for e in events if e.get("state") == "done"]
    assert {e["reason"] for e in done if e["verdict"] != "correct"} == set(wanted)
    assert all(e["reason"] == "" for e in done if e["verdict"] == "correct")
    assert [r["reason"] for r in m["graded_agents"] if r["verdict"] == "no answer"] == ["no answer"]


def test_the_run_request_takes_the_three_difficulties_and_nothing_else():
    assert RunReq().difficulty == "mixed" and RunReq(difficulty="hard").difficulty == "hard"
    with TestClient(app) as c:
        assert c.post("/api/live/run", json={"scenario": "swarm", "difficulty": "impossible"}).status_code == 422
        assert c.post("/api/live/run", json={"scenario": "swarm", "difficulty": "easy"}).status_code == 422


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
        by_size = {k: [n for n in samples if documents.key_of(n)["size"] == k] for k in ("short", "medium", "long")}
        room = {k: c.post("/api/live/agent-room", json={"scenario": "swarm", "files": v}).json() for k, v in by_size.items()}
        assert room["short"]["max"] > room["medium"]["max"] > room["long"]["max"] >= 1  # the size of the documents picked decides
        assert room["short"]["tokens_per_agent"] < room["medium"]["tokens_per_agent"] < room["long"]["tokens_per_agent"]
        assert room["long"]["max"] <= 16 and room["short"]["max"] >= 64  # 262144 cells: a long document needs 10k+ tokens, a short one 2k
        nothing = c.post("/api/live/agent-room", json={"scenario": "swarm", "files": []}).json()
        assert nothing == room["long"]  # no pick: the biggest documents of the library
        mixed = c.post("/api/live/agent-room", json={"scenario": "swarm", "files": by_size["short"][:3] + by_size["long"][:1]}).json()
        assert mixed["max"] == room["long"]["max"] or mixed["tokens_per_agent"] <= room["long"]["tokens_per_agent"]
        assert c.post("/api/live/agent-room", json={"scenario": "classify", "files": by_size["short"]}).json()["max"] == room["short"]["max"]
        assert c.post("/api/live/agent-room", json={"scenario": "shared", "files": by_size["short"][:1] + by_size["long"][:1]}).json()["max"] == room["short"]["max"]
        assert c.post("/api/live/agent-room", json={"scenario": "chat"}).json() == {"max": 1, "tokens_per_agent": None}
        assert c.post("/api/live/agent-room", json={"scenario": "turns"}).json()["max"] == 256
        assert c.post("/api/live/agent-room", json={"scenario": "nope"}).status_code == 422
        monkeypatch.setattr(live, "SESSION", None)  # not stopped by the app's shutdown
