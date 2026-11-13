"""An engine's process lifecycle (live_parts/engine.py) and its restart for a run (Session.ensure_ctx): a failed restart
keeps the old block and is started again by the next run; a startup that times out stops its process; a stop returns only
when the process group is gone; the exact status and KV bytes are read again after a restart; info() says "restarting".

No GPU, no server: fake processes, a fake killpg, httpx.MockTransport for /caps."""

import asyncio
from pathlib import Path

import httpx
import pytest

from sonic_playground import live, settings
from sonic_playground.chat_template import GEMMA3
from sonic_playground.live_parts import engine as engine_mod
from sonic_playground.live_parts.config import LONG_STREAMS
from sonic_playground.live_parts.engine import Engine, _engine_pids
from sonic_playground.live_parts.run import _run_block
from sonic_playground.live_parts.session import Session, exact_status


@pytest.fixture
def logs(monkeypatch):
    out: list[str] = []
    monkeypatch.setattr(live, "log", out.append)
    return out


@pytest.fixture
def events(monkeypatch):
    seen: list[dict] = []
    monkeypatch.setattr(live.BUS, "publish", seen.append)
    return seen


def session(compare=False):
    s = object.__new__(Session)
    s.engines = {rt: Engine(rt, Path("m.gguf"), True) for rt in (["sllm", "llama"] if compare else ["sllm"])}
    s.cells_max, s.train, s.kvb, s.compare, s.exact = 262144, 40960, 100_000, compare, None
    s.template, s.lim, s.model = GEMMA3, {"answer": None, "agent": None, "doc": None}, {"id": "fake"}
    s.state, s.error, s.weight_bytes, s.busy = "ready", None, None, asyncio.Lock()
    s.restarting = False
    for e in s.engines.values():
        e.ctx = s.ctx_for("chat")
        e.cells = s.block_cells(LONG_STREAMS, e.ctx)
    return s


# -- 2a: a failed restart -------------------------------------------------------------------------------------------------
async def test_a_failed_restart_keeps_the_old_block_and_the_next_run_starts_the_engine_again(monkeypatch, logs, events):
    s = session()
    e = s.engines["sllm"]
    before = (e.cells, e.ctx, list(e.extra))
    starts = []

    async def stop(self):
        pass

    async def start(self):
        starts.append((self.ctx, self.cells))
        if len(starts) == 1:
            raise RuntimeError("Sonic LLM exited during startup (code 1)")

    monkeypatch.setattr(Engine, "stop", stop)
    monkeypatch.setattr(Engine, "start", start)
    monkeypatch.setattr(Session, "_read_sllm_caps", lambda self, e: asyncio.sleep(0))
    with pytest.raises(RuntimeError):
        await s.ensure_ctx("burst", 256, None, 256 + 70)
    assert (e.cells, e.ctx, list(e.extra)) == before  # the block it never got is not claimed
    assert any("restarting Sonic LLM failed" in x for x in logs) and not s.restarting
    assert [ev["session"]["restarting"] for ev in events if ev["type"] == "session"] == [True, False]  # the page hears the end
    await s.ensure_ctx("burst", 256, None, 256 + 70)  # the next run: started again
    assert len(starts) == 2 and (e.ctx, e.cells) == starts[1]
    starts.clear()
    await s.ensure_ctx("burst", 256, None, 256 + 70)  # now it runs with that block: no restart
    assert starts == []


async def test_a_failed_restart_before_a_chat_run_is_started_again_by_the_chat(monkeypatch, logs, events):
    s = session()
    e = s.engines["sllm"]
    e.ctx = s.ctx_for("chat", 1, None)
    e.cells = s.block_cells(1, e.ctx)  # the chat's block exactly
    asked = []

    async def ensure_ctx(*a, **k):
        asked.append(a)

    monkeypatch.setattr(s, "ensure_ctx", ensure_ctx, raising=False)
    await _run_block(s, live.RunReq(scenario="chat"), "chat", 1, None, None)
    assert asked == []  # a running engine with this block is kept
    e.failed = "RuntimeError: SLLM exited during startup (code 1)"  # what a failed restart leaves
    await _run_block(s, live.RunReq(scenario="chat"), "chat", 1, None, None)
    assert len(asked) == 1  # the block would fit as it is, but the engine is not running


async def test_info_says_when_an_engine_restarts(monkeypatch, logs, events):
    s = session()
    seen = []

    async def stop(self):
        pass

    async def start(self):
        seen.append(s.info()["restarting"])

    monkeypatch.setattr(Engine, "stop", stop)
    monkeypatch.setattr(Engine, "start", start)
    monkeypatch.setattr(Session, "_read_sllm_caps", lambda self, e: asyncio.sleep(0))
    assert s.info()["restarting"] is False
    await s.ensure_ctx("burst", 256, None, 256 + 70)
    assert seen == [True] and s.info()["restarting"] is False
    flags = [ev["session"]["restarting"] for ev in events if ev["type"] == "session"]
    assert flags == [True, False]  # the page hears the restart begin and end


# -- 10: the exact status and the KV bytes after a restart --------------------------------------------------------------------
def caps(row_exact: bool, kvb: int) -> dict:
    return {"engine": {"engine": "next", "row_exact": row_exact, "kv": {"type": "f16"}, "kv_read": {"mode": "f16" if row_exact else "q"},
                       "auto_mode": {"line": "auto"}, "profile": {"kv_bytes_per_token_per_seq": {"f16": {"total_once": kvb}}}}}


async def test_the_exact_status_and_kv_bytes_are_read_again_after_a_restart(monkeypatch, logs, events):
    s = session()
    e = s.engines["sllm"]
    now = {"caps": caps(True, 100_000)}
    s.exact = exact_status(now["caps"])

    async def stop(self):
        pass

    async def start(self):  # the restarted engine says it is not exact any more, with other KV bytes
        now["caps"] = caps(False, 90_000)
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=now["caps"])))
        self.up = True

    monkeypatch.setattr(Engine, "stop", stop)
    monkeypatch.setattr(Engine, "start", start)
    await s.ensure_ctx("swarm", 64)
    assert s.exact["level"] == "no" and s.kvb == 90_000
    assert [ev["session"]["exact"]["level"] for ev in events if ev["type"] == "session"][-1] == "no"
    assert e.up


# -- 2b, 2c, 2f: the process --------------------------------------------------------------------------------------------------
class FakeGroup:
    """A process group: the leader exits at `exits_on` (a signal); the group is gone `lingers` probes after SIGKILL."""

    def __init__(self, exits_on=None, lingers=0, code=None):
        self.signals, self.exits_on, self.lingers, self.killed = [], exits_on, lingers, False
        self.code = code

    def killpg(self, pid, sig):
        if sig == 0:
            if self.killed and self.lingers <= 0:
                raise ProcessLookupError(pid)
            self.lingers -= 1 if self.killed else 0
            return
        self.signals.append(sig)
        if sig == self.exits_on or sig == 9:
            self.code = -sig
        self.killed = self.killed or sig == 9


class FakeProc:
    def __init__(self, pid, group):
        self.pid, self.group = pid, group

    def poll(self):
        return self.group.code

    @property
    def returncode(self):
        return self.group.code


@pytest.fixture
def procs(monkeypatch):
    """Popen makes FakeProcs; killpg goes to their groups; nothing real is started or signalled."""
    made: list[FakeProc] = []
    groups: list[FakeGroup] = []

    def popen(argv, **kw):
        g = groups.pop(0) if groups else FakeGroup()
        made.append(FakeProc(1000 + len(made), g))
        return made[-1]

    def killpg(pid, sig):
        next(p for p in made if p.pid == pid).group.killpg(pid, sig)

    monkeypatch.setattr(engine_mod.subprocess, "Popen", popen)
    monkeypatch.setattr(engine_mod.os, "killpg", killpg)
    monkeypatch.setattr(engine_mod, "free_port", lambda: 1)
    return made, groups


async def test_a_startup_that_times_out_stops_its_process(monkeypatch, procs, logs):
    made, groups = procs
    groups.append(FakeGroup(exits_on=15))
    monkeypatch.setattr(engine_mod, "START_TIMEOUT_S", 0.3, raising=False)

    async def never(self):
        return False

    monkeypatch.setattr(Engine, "_healthy", never)
    e = Engine("sllm", Path("m.gguf"), True)
    with pytest.raises(TimeoutError):
        await e.start()
    assert made[0].group.signals[:1] == [15] and e.proc is None
    assert _engine_pids() == []


async def test_stop_returns_only_when_the_process_group_is_gone(procs, logs):
    made, groups = procs
    groups.append(FakeGroup(exits_on=15, lingers=3))  # the leader exits on SIGTERM, a child lingers a moment after SIGKILL
    e = Engine("sllm", Path("m.gguf"), True)
    e.proc = engine_mod.subprocess.Popen([])
    g = made[0].group
    await e.stop()
    assert g.signals == [15, 9] and g.lingers <= 0  # gone when stop returns, so the next engine starts after it


async def test_a_group_that_will_not_go_is_logged(monkeypatch, procs, logs):
    made, groups = procs
    groups.append(FakeGroup(exits_on=15, lingers=10**6))
    monkeypatch.setattr(engine_mod, "STOP_GONE_S", 0.2, raising=False)
    e = Engine("sllm", Path("m.gguf"), True)
    e.proc = engine_mod.subprocess.Popen([])
    await e.stop()
    assert any("still running" in x for x in logs)


async def test_a_process_that_exits_while_starting_fails_the_start_names_its_log_and_is_forgotten(procs, logs):
    made, groups = procs
    groups.append(FakeGroup(code=1))  # a file the engine cannot run: it exits at once
    e = Engine("sllm", Path("m.gguf"), True)
    with pytest.raises(RuntimeError, match="exited during startup .code 1.; see .*sllm-.*log"):
        await e.start()
    assert e.up is False
    assert _engine_pids() == [made[0].pid]  # still remembered: the caller stops the process group, which forgets it
    await e.stop()
    assert _engine_pids() == []


def test_the_engine_command_lines_carry_the_settings(cfg):
    cfg_args = settings.Settings(**{**cfg.__dict__, "engine_args": ("--threads", "4"), "llama_bin": cfg.server_bin.parent / "llama-server"})
    settings.configure(cfg_args)
    sllm, llama = Engine("sllm", Path("m.gguf"), True), Engine("llama", Path("m.gguf"), True)
    sllm.port, sllm.ctx, sllm.cells, sllm.extra = 8101, 4096, 8192, ["--many-agents", "on"]
    argv = sllm.argv()
    assert argv[0] == str(cfg.server_bin) and "--engine" not in argv  # the default engine, not asked for by name
    assert argv[1:] == ["--model", "m.gguf", "--host", "127.0.0.1", "--port", "8101", "--auto-mode", "--ctx", "4096", "--kv-pages", "32",
                        "--threads", "4", "--many-agents", "on"]
    llama.port, llama.cells = 8102, 8192
    assert llama.argv()[0] == str(cfg.server_bin.parent / "llama-server") and "-kvu" in llama.argv() and "--threads" not in llama.argv()
