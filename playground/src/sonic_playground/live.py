"""The live playground: pick a model, chat with it (Sonic LLM alone or next to llama.cpp), run agent tests over a library of
documents, and watch the metrics move. Every test run is kept in a history.

Not a benchmark: both servers of a compare session stay loaded, the tests run one engine after the other, there is one
repetition and no rotation. The numbers are a live view.

Routes (all under /api/live):
  GET  /state                 session, engines, document library summary, the tests
  POST /session               {model_id, compare, kv_gib}: start sllm-server (and llama-server)
  DELETE /session             stop the engines (also after IDLE_STOP_S without use)
  GET  /files                 the document library (path, bytes, estimated tokens)
  PUT  /files/{name}          add a text document of your own (the request body is the file)
  POST /run                   {scenario, message, files, agents}: run one test; progress on /events
  POST /run/stop              stop the run: its requests end at the engines' next step, and nothing goes in the history
  POST /reset                 forget the chat history
  POST /preread               the chat text typed so far: the engine reads it ahead
  GET  /events                server-sent events: tick (1 per 0.5 s), token, agent, run_start, run_done, log, session
  GET  /scenarios             the tests with their suggested messages
  GET  /history, /history/{id}

The live playground is split by job into the modules of live_parts/ (config, events, engine, session, scenarios, agents,
turns, run, runners). They look up `live.log`, `live.BUS` and `live.stop_session` here when they use them, so a test that
patches one of them reaches every module.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import os
import signal
import time
import uuid
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import documents, levels, models, settings
from .chat_template import template_for
from .live_parts.agents import fit_prompt, prompt_room
from .live_parts.engine import stop_orphans
from .live_parts.events import Bus, epoch_start
from .live_parts.run import run_scenario
from .live_parts.scenarios import SCENARIOS, SYSTEM_CHAT, RunReq
from .live_parts.session import Session, run_going

router = APIRouter(prefix="/api/live")


BUS = Bus(start=epoch_start())  # the page's events (GET /events): every part publishes on live.BUS


def log(msg: str) -> None:
    """A line in the page's event log."""
    BUS.publish({"type": "log", "msg": msg})


# -- the session: start, stop and replace take one lock, so they happen one at a time ------------------------------------
SESSION: Session | None = None
_LOCK: list = [None, None]  # [event loop, its lock]: an asyncio lock serves one loop, and tests run one loop per test


def _session_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    if _LOCK[0] is not loop:
        _LOCK[:] = [loop, asyncio.Lock()]
    return _LOCK[1]


async def stop_session(only: Session | None = None) -> None:
    """Stop the session, its run first; with `only`, only when that is still the session (the idle stop names its own)."""
    async with _session_lock():
        if only is None or SESSION is only:
            await _stop_current()


async def _stop_current() -> None:
    """Stop the session (the caller holds the session lock): it is no longer the session at once, so no run starts on
    it; its run is cancelled and has ended before its engines stop."""
    global SESSION
    s, SESSION = SESSION, None
    if s is None:
        return
    t = await _cancel_run(s)
    if t is not None:
        await asyncio.gather(t, return_exceptions=True)
    await s.stop()
    BUS.publish({"type": "session", "session": None})


async def _cancel_run(s) -> asyncio.Task | None:
    """Cancel the session's run once no engine restarts for it (a restart cut in half would leave an engine half started).
    Returns the cancelled task; None when no run is going."""
    t = s.run_task
    if t is None or t.done():
        return None
    while s.restarting:
        await asyncio.sleep(0.05)
    t.cancel()
    return t


# -- routes -----------------------------------------------------------------------------------------------------------
class SessionReq(BaseModel):
    model_id: str
    compare: bool = False
    kv_gib: float | None = Field(None, gt=0, description="optional cap on each engine's KV memory (default: as much as fits)")


@router.get("/state")
async def get_state() -> dict:
    fs = await asyncio.to_thread(documents.list_files)
    return {"session": SESSION.info() if SESSION else None,
            "files": {"n": len(fs), "bytes": sum(f["bytes"] for f in fs)}, "scenarios": SCENARIOS}  # fmt: skip


@router.get("/scenarios")
async def get_scenarios() -> dict:
    return {"scenarios": SCENARIOS}


@router.post("/session")
async def post_session(req: SessionReq) -> dict:
    """Load a model: a load waits for a start or stop in progress, then stops the session it replaces (whole) first."""
    async with _session_lock():
        return await _replace_session(req)


async def _replace_session(req: SessionReq) -> dict:
    global SESSION
    gone = await asyncio.to_thread(stop_orphans)
    if gone:
        log(f"stopped engines left by an earlier playground run: pids {gone}")
    if SESSION is not None:
        if run_going(SESSION):
            raise HTTPException(409, "a scenario is running; wait for it or stop the session")
        await _stop_current()
    s = await _new_session(req)
    SESSION = s
    BUS.publish({"type": "session", "session": s.info()})
    try:
        await s.start()
    except BaseException:
        if SESSION is s:  # compare-and-clear: only this session's own failure takes it away
            SESSION = None
        BUS.publish({"type": "session", "session": None, "error": s.error})
        raise
    BUS.publish({"type": "session", "session": s.info()})
    return s.info()


async def _new_session(req: SessionReq) -> Session:
    """The session `req` asks for, not started. Refused (409) when the engine, or llama.cpp for a comparison, is missing."""
    cfg = settings.get()
    if not cfg.engine_found():
        raise HTTPException(409, f"the Sonic LLM engine is not built: {cfg.server_bin} is missing "
                                 f"(build it with `{settings.BUILD_COMMAND}` in the engine folder, or set SONIC_SERVER_BIN)")
    if req.compare and cfg.llama_bin is None:
        raise HTTPException(409, "llama-server was not found: install llama.cpp or set LLAMA_SERVER_BIN to compare")
    model = next((m for m in await asyncio.to_thread(models.scan, cfg.models_dir) if m["id"] == req.model_id and not m["error"]), None)
    if model is None:
        raise HTTPException(404, f"unknown model {req.model_id!r}")
    try:
        template = await asyncio.to_thread(template_for, model["file"])
    except (KeyError, ValueError) as e:
        raise HTTPException(422, f"no chat template for {req.model_id}: {e}") from e
    s = Session(model, template, req.compare)
    s.kv_gib = req.kv_gib
    return s


@router.delete("/session")
async def delete_session() -> dict:
    await stop_session()
    return {"stopped": True}


@router.post("/reset")
async def post_reset() -> dict:
    if SESSION:
        SESSION.history = {rt: [] for rt in SESSION.engines}
        SESSION.readers = None  # a follow-up after a reset does not ask the readers of a run the page no longer shows
    return {"ok": True}


@router.get("/files")
async def get_files() -> dict:
    return {"files": await asyncio.to_thread(documents.list_files)}


@router.put("/files/{name}")
async def put_file(name: str, request: Request) -> dict:
    """Add a text document of the user's own to the library: the request body is the file."""
    if int(request.headers.get("content-length") or 0) > documents.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"the file is larger than {documents.MAX_UPLOAD_BYTES // 1_000_000} MB")
    try:
        path = await asyncio.to_thread(documents.save_upload, name, await request.body())
    except documents.DocumentError as e:
        raise HTTPException(422, str(e)) from e
    log(f"added {path} to the documents")
    return {"path": path}


@router.post("/run")
async def post_run(req: RunReq) -> dict:
    """Accept a run: between the checks and its task there is no await, so a second POST sees it (409) and Stop reaches
    it from here on, also while it waits for the read ahead."""
    s = SESSION
    if s is None or s.state != "ready":
        raise HTTPException(409, "no model is loaded: pick one at the top")
    if run_going(s):
        raise HTTPException(409, "a scenario is already running")
    if req.scenario not in {x["id"] for x in SCENARIOS}:
        raise HTTPException(422, f"unknown scenario {req.scenario!r}")
    s.apply_level(req)  # a framing the model's template cannot give fails here, before the run starts
    run_id = f"live-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:4]}"
    s.last_used = time.monotonic()
    s.run_task = asyncio.create_task(_run(s, req, run_id))
    return {"run": run_id}


async def _run(s: Session, req: RunReq, run_id: str) -> None:
    """One accepted run, from the read ahead's settle to its one run_done: the record, stopped (also when it is stopped
    before it started), or the error."""
    try:
        await preread_settle()
        async with s.busy:
            s.last_used = time.monotonic()
            BUS.publish({"type": "run_start", "run": run_id, "scenario": req.scenario, "message": req.message,
                         "files": req.files, "engines": list(s.engines)})  # fmt: skip
            rec = await run_scenario(s, req, run_id)
    except asyncio.CancelledError:
        BUS.publish({"type": "run_done", "run": run_id, "stopped": True})
    except Exception as e:
        BUS.publish({"type": "run_done", "run": run_id, "error": f"{type(e).__name__}: {e}"})
    else:
        _save_run(rec)
        BUS.publish({"type": "run_done", "run": run_id, "record": rec})
    finally:
        s.last_used = time.monotonic()


def _save_run(rec: dict) -> None:
    """The run's record into the history. A failed write is logged; the run still ends with its record on the page."""
    path = settings.get().runs_file
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except OSError as ex:
        log(f"writing run {rec.get('id')} to the history ({path}) failed ({type(ex).__name__}: {ex}); the page has its record")


# -- read ahead: the chat prompt is read while the user still types -------------------------------------------------------
# A read costs the model a fixed part (Qwen3.6-35B-A3B: about 35 ms for any prompt) plus its tokens; read before the user
# sends, only the last token is left for the run (2026-11-12: 100 to 174 token prompts, 64 to 98 ms -> about 21 ms after
# send, same answers). The page posts the text as it changes; the newest text wins; a run waits for a read in flight.
_PRE: dict = {"task": None, "next": None, "reads": 0, "tokens": 0, "session": None}  # session: the one the task reads for


class PrereadReq(BaseModel):
    message: str = ""
    thinking: Literal["off", "low", "medium", "high", "max"] = "off"
    level: Literal["off", "sonic", "fast", "normal", "deep"] | None = None
    answer_tokens: int | None = Field(None, ge=16)


def preread_systems(level: str) -> list[str]:
    """The system prompts of the stages a chat answer at `level` starts with (normal and deep start with two: ONE, TIGHT)."""
    if level == "off":
        return [SYSTEM_CHAT]
    names = levels.STAGES[level][:2] if level in ("normal", "deep") else levels.STAGES[level][:1]
    return [levels.stage_system(levels.PROMPTS[n], SYSTEM_CHAT, SYSTEM_CHAT) for n in names]


async def preread_once(s: Session, req: PrereadReq) -> int:
    """Read the chat prompt `req` would send, all but its last token, and keep it in the engine (nothing is written). The
    prompts are built exactly as `_chat_answer` builds them. Returns the tokens read ahead (0: nothing to do)."""
    e = s.engines.get("sllm")
    if e is None or not levels.supported(e) or not req.message.strip():
        return 0
    s.apply_level(RunReq(scenario="chat", message=req.message, thinking=req.thinking, level=req.level))
    lv, cap, n = s.level, req.answer_tokens, 0
    for system in preread_systems(lv):
        room = prompt_room(e.ctx, cap) if lv == "off" else prompt_room(e.ctx, min(max(levels.level_caps(lv)), cap) if cap else max(levels.level_caps(lv)))
        ids = await fit_prompt(e, s.chat_prompt("sllm", system, req.message), room)
        if len(ids) < 2 or s.busy.locked():
            break
        r = await e.client.post(f"http://127.0.0.1:{e.port}/completion", json={
            "prompt": ids[:-1], "n_predict": 0, "temperature": 0, "stream": False, "cache_prompt": True, "lane": "interactive"})
        if r.status_code != 200:
            break
        n += len(ids) - 1
    return n


async def _preread_loop(s: Session) -> None:
    while _PRE["next"] is not None and SESSION is s and s.state == "ready" and not run_going(s):
        req, _PRE["next"] = _PRE["next"], None
        try:
            got = await preread_once(s, req)
            if got:
                _PRE["reads"] += 1
                _PRE["tokens"] += got
        except Exception as ex:  # a failed read ahead only costs the run its head start; the next text is read as usual
            log(f"read ahead failed ({type(ex).__name__}: {ex})")


async def preread_settle() -> None:
    """Before a run: drop what waits and let a read in flight finish (its prompt is then in the engine for the run)."""
    _PRE["next"] = None
    t = _PRE["task"]
    if t is not None and not t.done():
        try:
            await asyncio.wait_for(asyncio.shield(t), 5)
        except Exception as ex:  # the run starts all the same; it reads what the read ahead did not
            log(f"waiting for the read ahead failed ({type(ex).__name__}: {ex}); the run starts without it")


@router.post("/preread")
async def post_preread(req: PrereadReq) -> dict:
    s = SESSION
    if s is None or s.state != "ready" or run_going(s):
        return {"ok": False}
    s.last_used = time.monotonic()  # someone types: the session is in use
    _PRE["next"] = req
    if _PRE["task"] is None or _PRE["task"].done() or _PRE.get("session") is not s:  # a task of an old session ends by itself
        _PRE["task"], _PRE["session"] = asyncio.create_task(_preread_loop(s)), s
    return {"ok": True, "reads": _PRE["reads"], "tokens": _PRE["tokens"]}


@router.post("/run/stop")
async def post_run_stop() -> dict:
    s = SESSION
    if s is None or await _cancel_run(s) is None:
        raise HTTPException(409, "no run is going")
    return {"stopped": True}


@router.get("/events")
async def get_events(last_event_id: str | None = Header(None)) -> StreamingResponse:
    last = int(last_event_id) if last_event_id and last_event_id.isdigit() else None
    snapshot = {"type": "snapshot", "state": await get_state(), "t": time.time()}  # the page starts from it on every connect
    return StreamingResponse(BUS.stream(last_id=last, first=snapshot), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})  # fmt: skip


def _runs() -> list[dict]:
    path = settings.get().runs_file
    if not path.is_file():
        return []
    out = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        try:
            out.append(json.loads(line))
        except ValueError as ex:
            log(f"line {n} of the run history {path} is not JSON ({ex}); it is skipped")
    return out


@router.get("/history")
async def get_history() -> dict:
    return {"runs": list(reversed(_runs()))}


@router.get("/history/{run_id}")
async def get_history_run(run_id: str) -> dict:
    r = next((x for x in _runs() if x["id"] == run_id), None)
    if r is None:
        raise HTTPException(404, f"no run {run_id!r}")
    return r


def _kill_engines_at_exit() -> None:
    """The server reloads (uvicorn --reload) or exits: kill our engine process groups."""
    s = SESSION
    if s is None:
        return
    for e in s.engines.values():
        if e.proc is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(e.proc.pid, signal.SIGKILL)


atexit.register(_kill_engines_at_exit)
