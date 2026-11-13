"""The live playground's KV block sizing and the "Agents taking turns" scenario, with mocks (no GPU, no server).

The turns run against an httpx.MockTransport that acts like sllm-server's /v1/agents API: it refuses a turn when more
than K agents would sit in RAM, so a run that parks agents too late fails here."""

import asyncio
import json
import shutil
from pathlib import Path

import httpx
import pytest

from sonic_playground.live_parts.config import LONG_STREAMS
from sonic_playground.live_parts.engine import Engine
from sonic_playground.live_parts.scenarios import TURNS_DOC_CHARS, TURNS_NEED, TURNS_OUT
from sonic_playground.live_parts.turns import _turns, turns_docs
from sonic_playground import live
from sonic_playground.live_parts.run import run_scenario
from sonic_playground.live_parts.scenarios import SCENARIOS
from sonic_playground.live_parts.session import Session
from sonic_playground.chat_template import GEMMA3

KVB = 100_000  # KV bytes of one token in the fake model


def session(cells_max=262144, train=40960, compare=False, kvb=KVB):
    s = object.__new__(Session)
    s.engines = {rt: Engine(rt, Path("m.gguf"), True) for rt in (["sllm", "llama"] if compare else ["sllm"])}
    s.cells_max, s.train, s.kvb, s.compare = cells_max, train, kvb, compare
    s.template, s.lim, s.model = GEMMA3, {"answer": None, "agent": None, "doc": None}, {"id": "fake"}
    s.state, s.error, s.weight_bytes, s.busy = "ready", None, None, asyncio.Lock()
    for e in s.engines.values():
        e.ctx = s.ctx_for("chat")
        e.cells = s.block_cells(LONG_STREAMS, e.ctx)
    return s


@pytest.fixture
def logs(monkeypatch):
    out = []
    monkeypatch.setattr(live, "log", out.append)
    return out


# -- task A: the block follows the job ------------------------------------------------------------------------------
def test_chat_block_is_one_long_context_plus_a_few():
    s = session()
    assert s.engines["sllm"].ctx == 40960  # as before: the trained context
    assert s.engines["sllm"].cells == 4 * 40960 // 256 * 256 and s.engines["sllm"].cells < s.cells_max


def test_block_never_exceeds_what_fits_and_is_whole_pages():
    s = session(cells_max=65536, train=1 << 20)
    assert s.engines["sllm"].ctx == 65536 and s.engines["sllm"].cells == 65536  # one context the size of the memory
    assert s.block_cells(256, 2048) == 65536
    assert s.block_cells(3, 1000) == 3072 and s.block_cells(3, 1000) % 256 == 0


@pytest.mark.parametrize("agents", [1, 8, 64, 256])
def test_agent_scenarios_keep_the_context_they_get_today(agents):
    s = session()
    ctx = s.ctx_for("swarm", agents)
    ref = 1024  # the rule before the block followed the job
    while ref * 2 <= 40960 and 262144 // (ref * 2) >= agents:
        ref *= 2
    assert ctx == ref
    for sc in ("swarm", "team", "shared", "exact"):
        assert s.ctx_for(sc, agents) == ctx
        assert s.block_cells(s.streams_for(sc, agents), ctx) == min(262144, agents * ctx)  # every agent at once
    assert s.streams_for("load", agents) == agents + 1  # the chat question runs next to the swarm


def test_the_context_per_agent_limit_still_wins():
    s = session()
    assert s.ctx_for("swarm", 8, cap=2048) == 2048
    assert s.ctx_for("burst", 8, cap=256, need=300) == 256


def test_burst_context_is_the_smallest_multiple_of_256_holding_prompt_and_answer():
    s = session()
    assert s.ctx_for("burst", 256, need=256 + 70) == 512
    assert s.ctx_for("burst", 256, need=512) == 512 and s.ctx_for("burst", 256, need=513) == 768
    ctx = s.ctx_for("burst", 256, need=256 + 70)
    cells = s.block_cells(256, ctx)
    assert cells == 256 * 512 and cells * KVB / 2**30 == pytest.approx(12.5, abs=0.5) or cells < s.cells_max


async def test_ensure_ctx_restarts_both_engines_with_the_same_block(monkeypatch, logs):
    s = session(compare=True)
    started = []

    async def stop(self):
        started.append(("stop", self.rt))

    async def start(self):
        started.append(("start", self.rt, self.ctx, self.cells, tuple(self.extra)))

    monkeypatch.setattr(Engine, "stop", stop)
    monkeypatch.setattr(Engine, "start", start)
    monkeypatch.setattr(Session, "_read_sllm_caps", lambda self, e: asyncio.sleep(0))
    monkeypatch.setattr(live.BUS, "publish", lambda ev: None)
    await s.ensure_ctx("burst", 256, None, 256 + 70)
    cells = {x[3] for x in started if x[0] == "start"}
    assert cells == {256 * 512} and s.engines["sllm"].cells == s.engines["llama"].cells
    assert s.engines["sllm"].ctx == 512
    assert any(m.startswith("KV block for this run: ") and m.endswith("(256 agents x 512 ctx)") for m in logs)
    started.clear()
    await s.ensure_ctx("burst", 256, None, 256 + 70)  # same block: no restart
    assert started == []
    assert logs[-1] == f"KV block for this run: {256 * 512 * KVB / 2**30:.2f} GiB (256 agents x 512 ctx)"


async def test_turns_block_holds_k_agents_and_llama_is_left_alone(monkeypatch, logs):
    s = session(compare=True)
    llama_before = s.engines["llama"].cells
    started = []

    async def stop(self):
        pass

    async def start(self):
        started.append((self.rt, self.ctx, self.cells, list(self.extra)))

    monkeypatch.setattr(Engine, "stop", stop)
    monkeypatch.setattr(Engine, "start", start)
    monkeypatch.setattr(Session, "_read_sllm_caps", lambda self, e: asyncio.sleep(0))
    monkeypatch.setattr(live.BUS, "publish", lambda ev: None)
    await s.ensure_ctx("turns", 256, None, TURNS_NEED, None)
    assert started == [("sllm", 1280, 8 * 1280, ["--pressure-relief", "on"])]
    assert s.engines["llama"].cells == llama_before
    assert any(m.endswith("(8 agents x 1280 ctx)") for m in logs)
    argv = s.engines["sllm"].argv()
    assert argv[argv.index("--ctx") + 1] == "1280" and argv[argv.index("--kv-pages") + 1] == str(8 * 1280 // 256)
    assert argv[-2:] == ["--pressure-relief", "on"]
    await s.ensure_ctx("swarm", 4)  # the flag is for the turns only
    assert s.engines["sllm"].extra == [] and "--pressure-relief" not in s.engines["sllm"].argv()


def test_scenarios_list_has_the_turns_and_the_other_scenarios_are_unchanged():
    ids = [x["id"] for x in SCENARIOS]
    assert ids == ["chat", "swarm", "shared", "load", "exact", "burst", "team", "followup", "relay", "long", "turns", "classify", "extract", "compress", "summarize"]
    t = next(x for x in SCENARIOS if x["id"] == "turns")
    assert (t["name"], t["icon"]) == ("Agents taking turns", "🔁") and "disk" in t["what"]


# -- task B: agents taking turns ------------------------------------------------------------------------------------
class FakeSllm:
    """sllm-server's agent API in memory. A turn on a parked agent restores it first. Refuses (503) when more than `cap`
    agents would be in RAM; `mutate` makes a restored agent answer differently (a broken SSD round trip)."""

    def __init__(self, cap, mutate=False, fail_suspend=False):
        self.cap, self.mutate, self.fail_suspend = cap, mutate, fail_suspend
        self.agents: dict[str, dict] = {}
        self.n = 0
        self.suspends = self.restores = 0
        self.max_ram = 0
        self.calls: list[tuple[str, str]] = []
        self.refused = 0

    def ram(self):
        return sum(1 for a in self.agents.values() if a["ram"] and a["hist"])

    async def handler(self, req: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.002)  # turns of different agents overlap, as on a real server
        path, m = req.url.path, req.method
        if path == "/caps":
            return httpx.Response(200, json={})
        body = json.loads(req.content) if req.content else {}
        self.calls.append((m, path))
        if path == "/tokenize":
            text = body["content"]
            return httpx.Response(200, json={"tokens": [ord(c) % 90 + 3 for c in text[::4]] or [1]})
        if path == "/metrics":
            return httpx.Response(200, json={"lifecycle": {
                "suspends": {"api": self.suspends, "idle": 0, "pressure": 0, "errors": 0}, "restores": self.restores,
                "restore_ms_p50": 4.0 if self.restores else None, "restore_ms_p95": 9.0 if self.restores else None,
                "ssd_bytes": 1_000_000 * sum(1 for a in self.agents.values() if not a["ram"]),
                "agents": {"suspended": sum(1 for a in self.agents.values() if not a["ram"])}}})
        if path == "/v1/agents" and m == "POST":
            self.n += 1
            aid = f"ag_{self.n}"
            self.agents[aid] = {"ram": True, "hist": [], "turns": 0}
            return httpx.Response(201, json={"id": aid})
        aid = path.split("/")[3] if path.startswith("/v1/agents/") else None
        if m == "DELETE":
            self.agents.pop(aid)
            return httpx.Response(200, json={"deleted": True})
        a = self.agents[aid]
        if path.endswith("/suspend"):
            assert body == {"tier": "ssd"}
            if self.fail_suspend:
                return httpx.Response(409, json={"error": "no"})
            a["ram"] = False
            self.suspends += 1
            return httpx.Response(200, json={"suspended": True})
        assert path.endswith("/turn") and body["temperature"] == 0 and body["stream"] is True
        restored = not a["ram"]
        if restored:
            if self.ram() >= self.cap:
                self.refused += 1
                return httpx.Response(503, json={"error": "kv_pool_exhausted_error"})
            a["ram"] = True
            self.restores += 1
        elif not a["hist"] and self.ram() >= self.cap:
            self.refused += 1
            return httpx.Response(503, json={"error": "kv_pool_exhausted_error"})
        a["hist"] += body["prompt"]
        self.max_ram = max(self.max_ram, self.ram())
        seed = sum(a["hist"]) + len(a["hist"]) + (1 if (restored and self.mutate) else 0)
        toks = [(seed * 31 + j * 7) % 997 for j in range(body["n_predict"])]
        a["hist"] += toks
        a["turns"] += 1
        sse = "".join(f"data: {json.dumps({'tokens': [t], 'content': 'x'})}\n\n" for t in toks) + 'data: {"stop": true}\n\n'
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})


def turns_setup(fake, agents=20, k=4, compare=False, files=()):
    s = session(compare=compare)
    e = s.engines["sllm"]
    e.port, e.ctx, e.cells = 1, 1280, k * 1280
    e.client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    req = live.RunReq(scenario="turns", agents=agents, parallel=k, files=list(files))
    return s, e, req


@pytest.fixture
def events(monkeypatch):
    out = []
    monkeypatch.setattr(live.BUS, "publish", out.append)
    monkeypatch.setattr(live, "log", lambda m: out.append({"type": "log", "msg": m}))
    return out


@pytest.fixture
def library(cfg):
    files = cfg.samples_dir
    files.mkdir()
    for i in range(3):
        (files / f"doc{i}.md").write_text(f"# Doc {i}\n" + "".join(f"Line {j} of document {i} says something specific.\n" for j in range(300)))
    return files


async def test_turns_keep_only_k_agents_in_ram_and_record_the_metrics(events, library):
    fake = FakeSllm(cap=4)
    s, e, req = turns_setup(fake, agents=20, k=4)
    m = await _turns(s, req, "run1", "sllm")
    assert fake.max_ram <= 4 and fake.refused == 0, "more than K agents were in RAM at once"
    assert (m["agents"], m["rounds"], m["at_once"]) == (20, 3, 4)
    assert m["tokens_written"] == 20 * 3 * TURNS_OUT
    assert m["suspends"] == 20 * 2 and m["restores"] == 20 * 2  # parked after rounds 1 and 2, woken for rounds 2 and 3
    assert m["restore_ms_p50"] == 4.0 and m["restore_ms_p95"] == 9.0
    assert m["ssd_bytes"] > 0 and m["ssd_mb"] > 0 and m["parked_peak"] > 0
    assert m["kv_reserved_mb"] == round(4 * 1280 * KVB / 2**20, 1)
    assert m["same_as_ram"] is True
    assert m["notes_tokens"] > m["block_cells"] // 2 and m["write_tok_s"] > 0
    assert fake.agents == {}, "every agent is deleted at the end"
    assert e.tokens == (20 * 3 + 3) * TURNS_OUT  # includes the check run, so the sampler sees it


async def test_turns_agents_show_in_the_grid_as_reading_writing_parked_done(events, library):
    fake = FakeSllm(cap=2)
    s, e, req = turns_setup(fake, agents=6, k=2)
    await _turns(s, req, "run1", "sllm")
    ag = [x for x in events if x.get("type") == "agent"]
    states = {(x["i"], x["state"]) for x in ag}
    for i in range(6):
        assert {(i, "waiting"), (i, "reading"), (i, "writing"), (i, "parked"), (i, "done")} <= states
    assert [x["state"] for x in ag if x["i"] == 0] == ["waiting", "reading", "writing", "parked", "reading", "writing", "parked", "reading", "writing", "done"]
    assert any(x.get("type") == "token" and "same tokens after its SSD round trips" in x["text"] for x in events)


async def test_turns_detect_a_changed_answer_after_the_ssd_round_trip(events, library):
    fake = FakeSllm(cap=2, mutate=True)
    s, e, req = turns_setup(fake, agents=4, k=2)
    m = await _turns(s, req, "run1", "sllm")
    assert m["same_as_ram"] is False  # negative control: the check is not vacuous
    assert fake.agents == {}


async def test_turns_delete_every_agent_when_a_turn_fails(events, library):
    fake = FakeSllm(cap=2)
    s, e, req = turns_setup(fake, agents=6, k=2)
    real = fake.handler

    async def boom(r):
        if r.url.path.endswith("/turn") and fake.agents.get("ag_3", {}).get("turns") == 1:
            return httpx.Response(500, json={"error": "boom"})
        return await real(r)

    e.client = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        await _turns(s, req, "run1", "sllm")
    assert fake.agents == {}


async def test_turns_count_failed_suspends_and_still_finish(events, library):
    fake = FakeSllm(cap=100, fail_suspend=True)
    s, e, req = turns_setup(fake, agents=3, k=2)
    m = await _turns(s, req, "run1", "sllm")
    assert m["suspend_errors"] == 3 * 2 and m["suspends"] == 0


async def test_turns_retry_a_turn_the_pool_refused(events, library):
    fake = FakeSllm(cap=1)  # k=2 write at once but the pool holds one: the second is refused (503) until the first is parked
    s, e, req = turns_setup(fake, agents=4, k=2)
    m = await _turns(s, req, "run1", "sllm")
    assert fake.refused > 0 and m["tokens_written"] == 4 * 3 * TURNS_OUT


def test_turns_documents_come_from_the_files_or_builtin_topics(library):
    docs = turns_docs([], 7)
    assert len(docs) == 7 and docs[0][0] == "samples/doc0.md" and docs[3][0] == "samples/doc0.md (part 2)"
    assert docs[0][1] != docs[3][1] and all(len(t) <= TURNS_DOC_CHARS for _, t in docs)
    assert [d[0] for d in turns_docs(["samples/doc2.md"], 2)] == ["samples/doc2.md", "samples/doc2.md (part 2)"]
    shutil.rmtree(library)
    built = turns_docs([], 3)
    assert len({t for _, t in built}) == 3 and all(300 < len(t) < TURNS_DOC_CHARS * 2 for _, t in built)


async def test_turns_on_a_compare_session_run_on_sllm_only(events, library, monkeypatch):
    fake = FakeSllm(cap=2)
    s, e, req = turns_setup(fake, agents=4, k=2, compare=True)
    s.engines["llama"].client = None  # llama-server must not be called
    s.weight_bytes, s.gpu_util, s.exact = None, None, None

    async def noop(*a, **k):
        pass

    monkeypatch.setattr(s, "ensure_ctx", noop, raising=False)
    monkeypatch.setattr(s, "use_thinking", lambda level: None, raising=False)
    rec = await run_scenario(s, req, "run1")
    assert list(rec["metrics"]) == ["sllm"] and "Sonic LLM only" in rec["note"]
    assert rec["metrics"]["sllm"]["same_as_ram"] is True and rec["metrics"]["sllm"]["tokens_generated"] > 0
    assert any(x.get("rt") == "llama" and "no agents" in x.get("msg", "") for x in events)

