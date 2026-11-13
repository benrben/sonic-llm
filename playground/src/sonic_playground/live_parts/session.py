"""The session of the live playground (live.py): the loaded model, its engines and their KV block, the sampler that ticks
the page, the answer level and the chat prompts; and sllm-server's own plan of the KV memory."""

from __future__ import annotations

import asyncio
import enum
import json
import subprocess
import time
from collections import deque
from pathlib import Path

import httpx
from fastapi import HTTPException

from .. import gpu_busy, levels, live, settings  # live: its log and BUS are looked up there when used (tests patch them)
from ..chat_template import Template, template_for
from ..gguf_meta import train_ctx
from .config import BLOCK_PAGE, CTX, IDLE_STOP_S, LONG_STREAMS, MANY_AGENTS, MAX_AGENTS, SERIES, TICK_S
from .engine import Engine
from .events import _rate
from .scenarios import DOC_TASKS, LEVEL_FREE, TURNS_AT_ONCE, TURNS_FLAGS, RunReq


# -- the session ------------------------------------------------------------------------------------------------------
def _all_at_once_ctx(cells: int, top: int, agents: int) -> int:
    """Every agent at once: the longest context that still fits them all in `cells`, doubled from 1024 up to `top`; at
    least 1024 (then the rest wait)."""
    ctx = min(1024, top)
    while ctx * 2 <= top and cells // (ctx * 2) >= agents:
        ctx *= 2
    return ctx


class SessionState(enum.StrEnum):
    """A session's state; the strings are what /state and the session events carry."""
    STARTING = "starting"
    READY = "ready"
    STOPPED = "stopped"
    FAILED = "failed"


# the only moves a session makes (Session.move): a stopped or failed session never comes back; a new one is loaded instead
_MOVES = {SessionState.STARTING: {SessionState.READY, SessionState.FAILED, SessionState.STOPPED},
          SessionState.READY: {SessionState.FAILED, SessionState.STOPPED}, SessionState.STOPPED: set(), SessionState.FAILED: set()}


def run_going(s) -> bool:
    """Whether session `s` has a run: one holding `busy`, or one accepted and not ended yet (it waits for the read ahead)."""
    t = s.run_task
    return s.busy.locked() or (t is not None and not t.done())


class Session:
    level = "off"  # this run's answer level (levels.py); set by apply_level
    run_task: asyncio.Task | None = None  # the run going now
    restarting = False  # an engine is restarting for a run: a stop waits for that, not to leave it half started
    exact: dict | None = None  # the engine's exact status (exact_status), once it is up

    def __init__(self, model: dict, template: Template, compare: bool):
        self.model, self.template, self.compare = model, template, compare
        self.templates, self.thinking = {"off": template}, "off"
        self.weight_bytes = model["size_bytes"]  # the model file's size: what the weights take of memory
        gguf = Path(model["file"])
        self.engines = {rt: Engine(rt, gguf, template.bos) for rt in (["sllm", "llama"] if compare else ["sllm"])}
        self.history: dict[str, list[tuple[str, str]]] = {rt: [] for rt in self.engines}
        # the last Agent swarm / Research team run's readers: a follow-up asks them again (their documents still in the
        # engine's memory, so only the new question is read) instead of answering from their written notes
        self.readers: dict | None = None
        self.busy = asyncio.Lock()
        self.last_used = time.monotonic()
        self.sampler: asyncio.Task | None = None
        self.state = SessionState.STARTING
        self.error: str | None = None
        self.gpu_util: float | None = None
        self.kvb: int | None = None
        self.lim = {"answer": None, "agent": None, "doc": None}  # this run's optional limits (None: no limit)
        self.kv_gib: float | None = None  # optional cap on each engine's KV memory
        self.cells_max = 65536  # the most KV cells that fit for each engine (the memory plan and the KV memory limit)
        self.train: int | None = None

    def info(self) -> dict:
        return {"model": self.model["id"], "compare": self.compare, "engine": self.engines["sllm"].kind, "state": self.state, "error": self.error,
                "busy": self.busy.locked(), "restarting": self.restarting, "template": self.template.name, "weight_bytes": self.weight_bytes,
                "kv_bytes_per_token": self.kvb, "train_ctx": self.train, "exact": self.exact,
                "engines": {rt: {"label": e.label, "ready_s": e.ready_s, "port": e.port, "cells": e.cells,
                                 "ctx": self.engines["sllm"].ctx, "max_parallel": e.max_parallel(self.engines["sllm"].ctx),
                                 "pid": e.proc.pid if e.proc else None} for rt, e in self.engines.items()}}  # fmt: skip

    async def start(self) -> None:
        try:
            await self._plan_kv()
            for rt, e in self.engines.items():
                live.log(f"starting {SERIES[rt]} on {self.model['id']}")
                await e.start()
                live.log(f"{SERIES[rt]} ready in {e.ready_s:.1f} s")
                if rt == "sllm":
                    await self._read_sllm_caps(e)
            self.move(SessionState.READY)
            self.sampler = asyncio.create_task(self._sample())
        except BaseException as ex:
            self.error = str(ex)
            if self.can(SessionState.FAILED):
                self.move(SessionState.FAILED)
            await self.stop()
            raise

    def can(self, to: SessionState) -> bool:
        return to in _MOVES[SessionState(self.state)]

    def move(self, to: SessionState) -> None:
        """The one place the state changes: a move the table (_MOVES) does not have raises."""
        if not self.can(to):
            raise RuntimeError(f"a {self.state} session cannot become {to}")
        self.state = to

    async def _plan_kv(self) -> None:
        """The KV memory of the session from sllm-server's plan: the cells that fit for each engine (cells_max), the
        bytes per token of the session's KV type (kvb), the trained context, and each engine's first block (one chat)."""
        kvb, fitting = await asyncio.to_thread(kv_plan, Path(self.model["file"]))
        cells = (fitting or 65536) // len(self.engines) // 256 * 256  # the most that fits, shared by the engines
        if self.kv_gib and kvb:
            cells = min(cells, int(self.kv_gib * 2**30) // kvb // 256 * 256)
        cells = max(8192, cells)
        self.train = await asyncio.to_thread(train_ctx, Path(self.model["file"]))
        self.cells_max = cells
        for e in self.engines.values():
            e.ctx = self.ctx_for("chat")
            e.cells = self.block_cells(1, e.ctx)  # one chat; a run that needs more restarts with its own block
        self.kvb = kvb  # bytes per token (the cells above are sized as f16)
        live.log(f"KV that fits for each engine: {cells:,} cells ({cells * (self.kvb or 0) / 2**30:.1f} GiB)")
        self.log_block(1, self.engines["sllm"].ctx, self.engines["sllm"].cells)

    async def _read_sllm_caps(self, e: Engine) -> None:
        """What the running sllm-server says of itself (/caps): which engine runs the model, the exact status, the auto-mode
        line, its real KV bytes per token."""
        try:
            caps = (await e.client.get(f"http://127.0.0.1:{e.port}/caps", timeout=3)).json()
            e.kind = caps["engine"]["engine"]
            self.exact = exact_status(caps)
            live.log(self.exact["text"])
            live.log(f"auto-mode: {caps['engine']['auto_mode']['line']}")
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as ex:  # the session runs without them
            live.log(f"reading the engine's /caps failed ({type(ex).__name__}: {ex}); no exact status or auto-mode line")
        real = await e.real_kv_bytes_per_token()  # what the running engine reports for its own KV type
        if real and real != self.kvb:
            live.log(f"KV bytes per token: {real:,} (the running engine's, plan said {self.kvb or 0:,})")
            self.kvb = real

    async def stop(self) -> None:
        if self.sampler:
            self.sampler.cancel()
        for e in self.engines.values():
            await e.stop()
        if self.can(SessionState.STOPPED):
            self.move(SessionState.STOPPED)

    def ctx_for(self, scenario: str, agents: int = 1, cap: int | None = None, need: int | None = None) -> int:
        """sllm-server --ctx for a scenario: the longest context that still lets every asked-for agent run at once
        (agent scenarios), the trained context for one long request, and for the throughput burst, the turns and the document
        jobs the smallest multiple of 256 that holds `need` tokens (prompt plus answer): agents x context is the memory, and a
        window the job does not fill only costs RAM."""
        cells, top = self.cells_max, min(self.train or CTX, CTX, cap or CTX)
        if scenario in ("burst", "turns"):
            return min(-(-(need or 1024) // BLOCK_PAGE) * BLOCK_PAGE, top)
        if scenario in ("swarm", "team", "load", "exact", "shared", "relay", *DOC_TASKS):
            ctx = _all_at_once_ctx(cells, top, agents)
            return min(ctx, -(-need // BLOCK_PAGE) * BLOCK_PAGE) if need and scenario in DOC_TASKS else ctx  # only ever smaller
        if scenario == "chat":  # `agents` requests at once (1 for every answer level): each gets its share
            return max(1024, min(cells // max(1, agents) // BLOCK_PAGE * BLOCK_PAGE, top))
        return max(1024, min(cells, top))

    def streams_for(self, scenario: str, agents: int = 1, at_once: int | None = None) -> int:
        """Sequences (agents) the KV block of a scenario holds at once: all asked-for agents, plus the chat question that
        runs next to them in `load`; K of the turns; the few requests of chat and the long document."""
        if scenario == "turns":
            return max(1, min(at_once or TURNS_AT_ONCE, agents))
        if scenario == "load":
            return agents + 1
        if scenario in ("swarm", "team", "shared", "exact", "burst", "relay", *DOC_TASKS):
            return max(1, agents)
        if scenario == "chat":  # the requests a chat answer runs at once, no more: 4 streams of the trained context cost Qwen3-4B
            return max(1, agents)  # 22.5 GB for one chat (2026-11-09), one stream 6 GB
        return LONG_STREAMS

    def max_agents(self, scenario: str, need: int | None) -> int:
        """The most agents this test can run at once on this memory: the largest count whose context still holds `need` tokens
        (one agent's prompt and answer) and whose block fits in the KV cells. At least 1, at most the engine's own cap."""
        for n in range(MAX_AGENTS, 1, -1):
            ctx = self.ctx_for(scenario, n, None, need)
            if ctx >= (need or 0) and self.streams_for(scenario, n) * ctx <= self.cells_max:
                return n
        return 1

    def block_cells(self, streams: int, ctx: int) -> int:
        """KV cells for `streams` sequences of `ctx` cells, in whole pages, never more than what fits."""
        return min(self.cells_max, -(-streams * ctx // BLOCK_PAGE) * BLOCK_PAGE)

    def log_block(self, streams: int, ctx: int, cells: int) -> None:
        gib = f"{cells * self.kvb / 2**30:.2f}" if self.kvb else "?"
        live.log(f"KV block for this run: {gib} GiB ({max(1, min(streams, cells // ctx))} agents x {ctx} ctx)")

    async def ensure_ctx(self, scenario: str, agents: int = 1, cap: int | None = None, need: int | None = None,
                         at_once: int | None = None) -> None:
        """Size the KV block for this scenario and restart an engine whose --ctx, block or flags are not those (the
        context fixes how many agents run at once). Both engines get the same block, so a comparison stays fair; the
        turns run on Sonic LLM only, llama.cpp is left alone."""
        ctx = self.ctx_for(scenario, agents, cap, need)
        streams = self.streams_for(scenario, agents, at_once)
        cells = self.block_cells(streams, ctx)
        extra = TURNS_FLAGS if scenario == "turns" else ["--many-agents", "on"] if agents > MANY_AGENTS else []
        self.log_block(streams, ctx, cells)
        for rt, e in self.engines.items():
            if scenario == "turns" and rt != "sllm":
                continue
            if not _needs_restart(rt, e, cells, ctx, extra):
                continue
            live.log(f"restarting {SERIES[rt]} for {scenario}: {cells // ctx} agents of {ctx} ctx at once")
            await self._restart(rt, e, cells, ctx, extra)

    async def _restart(self, rt: str, e: Engine, cells: int, ctx: int, extra: list[str]) -> None:
        """Restart engine `e` with a block of `cells` (sllm-server also with --ctx `ctx` and the flags `extra`). The page
        hears the restart begin (info "restarting") and end, also when it failed."""
        old = (e.cells, e.ctx, e.extra)
        self.restarting = True
        live.BUS.publish({"type": "session", "session": self.info()})
        try:
            await e.stop()
            e.cells = cells
            if rt == "sllm":
                e.ctx, e.extra = ctx, extra
            await self._start_again(rt, e, old)
        finally:
            self.restarting = False
            live.BUS.publish({"type": "session", "session": self.info()})

    async def _start_again(self, rt: str, e: Engine, old: tuple) -> None:
        """Start the stopped engine `e` with its new block. It is the engine's only once it started: a failed start stops
        what it started, gives back the `old` (cells, ctx, extra) and marks the engine failed, so the next run starts it
        again. The engine's /caps are read again: the exact status and KV bytes are those of the running setup."""
        try:
            await e.start()
        except BaseException as ex:
            await e.stop()
            e.cells, e.ctx, e.extra = old
            e.failed = f"{type(ex).__name__}: {ex}"
            live.log(f"restarting {SERIES[rt]} failed ({e.failed}); the next run starts it again")
            raise
        e.failed = None
        if rt == "sllm":
            await self._read_sllm_caps(e)

    async def _gpu_util_loop(self) -> None:
        while True:
            self.gpu_util = await gpu_busy.gpu_utilisation()
            await asyncio.sleep(1.0)

    async def _sample(self) -> None:
        util_task = asyncio.create_task(self._gpu_util_loop())
        now = time.monotonic()
        wrote = {rt: deque([(now, e.tokens)]) for rt, e in self.engines.items()}
        read: dict[str, deque] = {rt: deque() for rt in self.engines}
        try:
            while True:
                await asyncio.sleep(TICK_S)
                await self._tick(wrote, read)
                self._check_engines()
                if time.monotonic() - self.last_used > IDLE_STOP_S and not run_going(self):
                    live.log("idle for 15 min: stopping the engines and releasing the GPU")
                    asyncio.create_task(live.stop_session(self))  # this session only: another may be loaded by then
                    return
        finally:
            util_task.cancel()

    def _check_engines(self) -> None:
        """An engine whose process ended while it was up fails the session, and the page hears it (a restart or a stop
        takes the engine down first, so they do not count)."""
        for e in self.engines.values():
            code = e.proc.poll() if e.up and e.proc is not None else None
            if code is None:
                continue
            e.up = False
            self.error = f"{e.label} exited (code {code}); see {e.log_path}"
            live.log(self.error)
            if self.can(SessionState.FAILED):
                self.move(SessionState.FAILED)
            live.BUS.publish({"type": "session", "session": self.info()})

    async def _tick(self, wrote: dict[str, deque], read: dict[str, deque]) -> None:
        """One tick to the page: each engine's write and read rates (from the counter points in `wrote` and `read`),
        requests in flight and footprint; GPU busy."""
        eng = {}
        reads = await asyncio.gather(*(e.prompt_tokens_read() for e in self.engines.values()))
        now = time.monotonic()
        for (rt, e), rd in zip(self.engines.items(), reads):
            tok_s = _rate(wrote[rt], now, e.tokens)
            read_s = None if rd is None else _rate(read[rt], now, rd)
            eng[rt] = {"tok_s": None if tok_s is None else round(tok_s, 1), "read_tok_s": None if read_s is None else round(read_s),
                       "active": e.active, "mem_mb": e.footprint_mb()}  # fmt: skip
        live.BUS.publish({"type": "tick", "gpu_util": self.gpu_util, "engines": eng})

    def use_thinking(self, level: str) -> None:
        """Frame the next prompts with thinking at `level` (the framing comes from the model's GGUF chat template)."""
        if level not in self.templates:
            try:
                self.templates[level] = template_for(self.model["file"], level)
            except (KeyError, ValueError) as e:
                raise HTTPException(422, f"thinking {level} is not possible with this model's chat template: {e}") from e
        self.template = self.templates[level]
        self.thinking = level

    def apply_level(self, req: "RunReq") -> str | None:
        """Set this run's answer level and the chat-template framing it needs. Returns a note for the page when the run
        cannot be exactly what was asked (None otherwise).

        Every answer level frames with thinking off; level off keeps `req.thinking` (gpt-oss low / medium / high / max). The
        level only changes the chat answer (agents._chat_answer); on an engine other than Sonic LLM's own the levels run the same way,
        unmeasured, and the note says so. A request with no level gets level off and `req.thinking`."""
        level = levels.norm_level(req.level)
        scen_ok = req.scenario not in LEVEL_FREE  # the engine tests measure the raw engine
        self.use_thinking(levels.framing_of(level, req.thinking))
        self.level = level if scen_ok else "off"
        note = None
        if level != "off" and not scen_ok:
            note = f"answer level {level} does not apply to this test: it runs plain"
        elif level != "off" and not levels.supported(self.engines["sllm"]):
            note = "the answer levels were measured on Sonic LLM's own engine; on this engine they run the same way, unmeasured"
        return note

    # -- prompts --
    def chat_prompt(self, rt: str, system: str, message: str) -> str:
        t = self.template
        text = t.prefix_text(system)
        hist = self.history[rt]
        if not hist:
            return text + t.task_text(message)
        text += t.task_text(hist[0][0]) + hist[0][1]
        for u, a in hist[1:]:
            text += t.turn_text(u) + a
        return text + t.turn_text(message)

    def one_shot(self, system: str, message: str) -> str:
        return self.template.prefix_text(system) + self.template.task_text(message)


def _needs_restart(rt: str, e: Engine, cells: int, ctx: int, extra: list[str]) -> bool:
    """Whether engine `e` must restart for a block of `cells` (sllm-server also --ctx `ctx` and the flags `extra`): another
    block, or its last restart failed (then it is not running)."""
    return bool(e.failed) or cells != e.cells or (rt == "sllm" and (ctx != e.ctx or extra != e.extra))


def exact_status(caps: dict) -> dict:
    """Whether this setup gives the answer an agent gets alone, from the engine's own /caps. Reading is the weak point: the
    pieces a prompt is read in change the last bits, and only reading in fixed blocks (kv_read f16) is exact."""
    e = caps.get("engine") or {}
    if not e.get("row_exact"):
        return {"level": "no", "text": "Not exact: these weights have no exact kernels, or the setup turns them off. Answers can change when other agents run."}
    if (e.get("kv_read") or {}).get("mode") == "f16":
        return {"level": "exact", "text": "Exact with up to 16 agents writing at once. Above 16 the engine uses stock kernels, and answers can differ from a solo run."}
    if (e.get("kv") or {}).get("type") == "f16":
        return {"level": "writing", "text": "Writing is exact, reading is not: how a prompt is cut can change a rare token."}
    return {"level": "no", "text": "Not exact: 4-bit reading changes under load."}


def kv_plan(gguf: Path) -> tuple[int | None, int | None]:
    """(KV bytes of one token, KV cells that fit in memory) for this model, from sllm-server's own plan (--print-profile; no
    GPU work). (None, None) when the plan cannot be had: the session then takes a default size."""
    cfg = settings.get()
    argv = [str(cfg.server_bin), "--model", str(gguf), "--ctx", "1024", "--kv-pages", "4", *cfg.engine_args, "--print-profile"]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=60).stdout
        budget = json.loads(out)["budget"]
        return int(budget["kv_alloc_bytes"] // budget["kv_cells"]["allocated"]), int(budget["kv_cells"]["max_fitting"])
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError, ZeroDivisionError) as ex:
        live.log(f"the engine's memory plan for {gguf.name} could not be read ({type(ex).__name__}: {ex}); using a default KV size")
        return None, None
