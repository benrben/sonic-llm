"""One run of the live playground (live.py): the KV block sized for it, the test on each engine of the session, and the
numbers and the history record every run keeps."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import httpx
from fastapi import HTTPException

from .. import live  # its log and BUS are looked up there when used (tests patch them)
from ..documents import list_files
from .agents import doc_need
from .graded import not_graded
from .config import CTX, SERIES
from .engine import Engine
from .runners import _RUNNERS, _TIMED_RUNNERS, _run_doc_task
from .scenarios import BURST_OUT, DOC_TASKS, TURNS_NEED, RunReq, burst_prompts
from .session import Session


async def run_scenario(s: Session, req: RunReq, run_id: str) -> dict:
    """Run `req` on every engine of the session, one after the other. Returns the history record."""
    per: dict[str, dict] = {}
    sizing, agents = await _run_sizing(s, req)
    cap = req.ctx
    if req.parallel:  # agents at once: the context per agent is what the KV cache holds divided by that
        cap = min(cap or CTX, max(512, s.cells_max // req.parallel // 256 * 256))
    note = s.apply_level(req)
    if note:
        live.BUS.publish({"type": "phase", "run": run_id, "rt": next(iter(s.engines)), "msg": note})
    need = await _run_need(s, req, agents)
    await _run_block(s, req, sizing, agents, cap, need)
    s.lim = {"answer": req.answer_tokens, "agent": req.agent_tokens, "doc": req.doc_chars}
    for rt in s.engines:
        if req.scenario == "turns" and rt != "sllm":  # llama-server has no agents
            live.BUS.publish({"type": "phase", "run": run_id, "rt": rt, "msg": f"{SERIES[rt]}: has no agents, skipped", "metrics": {}})
            continue
        per[rt] = await _run_on_engine(s, req, run_id, rt)
    return _run_record(s, req, run_id, per)


async def _run_sizing(s: Session, req: RunReq) -> tuple[str, int]:
    """The scenario the KV block is sized for, and the agents it holds: a follow-up is sized like the run it follows, tests
    with one agent per document have as many agents as files (picked, else in the library)."""
    agents = req.agents
    sizing = req.scenario
    if req.scenario == "followup":  # sized like the run it follows, so the engine (and the readers' memory) stays
        if not s.readers:
            raise HTTPException(422, "run Agent swarm or Research team first: a follow-up asks their readers again")
        sizing, agents = s.readers["kind"], len(s.readers["docs"])
    elif req.scenario in ("swarm", "team", "relay", *DOC_TASKS):
        agents = min(agents, len(req.files) if req.files else len(list_files()))
    return sizing, agents


async def _run_need(s: Session, req: RunReq, agents: int) -> int | None:
    """Tokens one agent's prompt and answer take, for the scenarios whose KV block is sized by the job (None: the others)."""
    need = None
    if req.scenario == "burst":  # the block holds each agent's prompt and its answer, no more
        need = BURST_OUT + max(len(ids) for ids in await asyncio.gather(
            *(s.engines["sllm"].tokenize(p) for _, p in burst_prompts(s, req.agents))))  # fmt: skip
    elif req.scenario == "turns":
        need = TURNS_NEED
    elif req.scenario in DOC_TASKS:
        need = doc_need(req.files, agents, req.doc_chars)
    return need


async def _run_block(s: Session, req: RunReq, sizing: str, agents: int, cap: int | None, need: int | None) -> None:
    """Size the engines' KV block for this run (a restart when it changes)."""
    # requests at once per answer: one for every answer level (a second answer starts after the first has ended)
    each = 1
    if req.scenario == "chat":
        e0 = s.engines.get("sllm")
        cur, want = (e0.ctx if e0 else None), s.ctx_for("chat", each, cap)
        # a smaller context already running is kept when it still holds a long chat and its block is sized for this level
        # (no restart on every level switch, and no block bigger than the level needs)
        if not (cur and not e0.failed and min(want, 16384) <= cur <= want and e0.cells == s.block_cells(each, cur)):
            await s.ensure_ctx("chat", each, cap, need)
    else:
        await s.ensure_ctx(sizing, (req.parallel or agents) * each, cap, need, req.parallel)


async def _run_on_engine(s: Session, req: RunReq, run_id: str, rt: str) -> dict:
    """The scenario on one engine, and the numbers every run records (time, memory, energy). Returns its metrics."""
    live.BUS.publish({"type": "phase", "run": run_id, "rt": rt, "msg": f"{SERIES[rt]}: running"})
    w0 = time.monotonic()
    tok0 = s.engines[rt].tokens
    s.engines[rt].begin_run()
    m = await _scenario_metrics(s, req, run_id, rt, w0)
    if not m.get("graded"):  # a graded run says so in its own numbers; the others say why they are not
        m = {**not_graded(req.scenario, req.message), **m}
    if rt == "sllm":
        await _note_spec(s.engines[rt], m)
    wall = time.monotonic() - w0
    e = s.engines[rt]
    e.footprint_mb()  # the peak also covers a run shorter than one sampler tick
    if rt == "sllm":
        try:
            e.note_metrics((await e.client.get(f"http://127.0.0.1:{e.port}/metrics", timeout=1)).json())
        except (httpx.HTTPError, ValueError) as ex:
            e._read_failed("/metrics", ex)  # the same state as the sampler's reads: logged once
        else:
            e._read_worked("/metrics")
    m.update(_engine_numbers(s, req, rt, e, wall, tok0))
    # no bandwidth_pct any more: tok/s x file size / bandwidth is not a bandwidth for a MoE model (only its active
    # experts are read per token), nor with speculation (several tokens per weight read), nor for a batch
    if rt != "sllm":
        m["max_parallel"] = e.max_parallel(s.engines["sllm"].ctx)  # llama-server's slots; sllm reports max_running
    if rt == "sllm":
        m["sllm_ctx"] = s.engines[rt].ctx
    done = {k: v for k, v in m.items() if v is not None}
    live.BUS.publish({"type": "phase", "run": run_id, "rt": rt, "msg": f"{SERIES[rt]}: done in {wall:.1f} s", "metrics": done})
    return done


async def _note_spec(e: Engine, m: dict) -> None:
    """The engine's speculation accept rate since it started (/caps engine.spec), into `m` when it reports one."""
    try:
        caps = (await e.client.get(f"http://127.0.0.1:{e.port}/caps", timeout=3)).json()
        sp = (caps.get("engine") or {}).get("spec") or {}
        if isinstance(sp.get("accept_rate"), (int, float)):
            m["spec_accept_rate_total"] = round(sp["accept_rate"], 3)
    except (httpx.HTTPError, ValueError, AttributeError) as ex:  # the run's record goes without it
        live.log(f"reading the engine's /caps for the speculation accept rate failed ({type(ex).__name__}: {ex})")


def _engine_numbers(s: Session, req: RunReq, rt: str, e: Engine, wall: float, tok0: int) -> dict:
    """What every run records of one engine: time, tokens, memory."""
    toks = e.tokens - tok0
    bpt = s.kvb  # real bytes per KV cell for the session's KV type
    # the turns' sequences list holds the agents parked on the SSD too: their notes are in notes_tokens
    kv_used = rt == "sllm" and e.kv_used_peak_cells and req.scenario != "turns"
    return {"wall_s": round(wall, 2), "tokens_generated": toks,
            # process footprint of this run (the mapped model file is not in it); the session peak is kept apart
            "peak_mem_mb": round(e.peak_mb, 1), "session_peak_mem_mb": round(e.session_peak_mb, 1),
            "weights_mb": round(s.weight_bytes / 1048576, 1) if s.weight_bytes else None,
            "kv_reserved_mb": _cells_mib(e.cells, bpt), "kv_used_mb": _cells_mib(e.kv_used_peak_cells, bpt) if kv_used else None,
            "max_running": e.max_running if rt == "sllm" and e.max_running else None,
            "gpu_util_pct": s.gpu_util}  # fmt: skip


def _cells_mib(cells: int, bpt: int | None) -> float | None:
    """`cells` KV cells of `bpt` bytes in MiB, to 0.1; None when the bytes per cell are unknown."""
    return round(cells * bpt / 1048576, 1) if bpt else None


def _run_record(s: Session, req: RunReq, run_id: str, per: dict[str, dict]) -> dict:
    """The history record of a run."""
    return {"id": run_id, "t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": s.model["id"],
            "compare": s.compare, "scenario": req.scenario, "message": req.message, "files": req.files, "agents": req.agents, "thinking": req.thinking,
            "level": s.level,
            "engines": {rt: {"label": SERIES[rt], "ready_s": e.ready_s} for rt, e in s.engines.items()}, "metrics": per,
            "exact_setup": (s.exact or {}).get("level"),
            "note": "live view: one repetition, engines one after the other, not a measurement"
            + (" · turns ran on Sonic LLM only: llama.cpp has no agents" if req.scenario == "turns" and s.compare else "")}  # fmt: skip


async def _scenario_metrics(s: Session, req: RunReq, run_id: str, rt: str, w0: float) -> dict:
    """Run the scenario of `req` on engine `rt` (started at `w0`, time.monotonic()). Returns its own metrics."""
    if req.scenario in DOC_TASKS:
        return await _run_doc_task(s, req, run_id, rt, w0)
    timed, plain = _TIMED_RUNNERS.get(req.scenario), _RUNNERS.get(req.scenario)
    if timed is not None:
        return await timed(s, req, run_id, rt, w0)
    if plain is not None:
        return await plain(s, req, run_id, rt)
    raise HTTPException(422, f"unknown scenario {req.scenario!r}")
