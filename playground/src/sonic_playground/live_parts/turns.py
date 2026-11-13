"""Agents taking turns (the live playground's turns test, live.py): agents on sllm-server's /v1/agents API, parked on the
disk between their turns, and the check that a parked agent writes what one kept in memory writes."""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
from fastapi import HTTPException

from .. import live  # its log and BUS are looked up there when used (tests patch them)
from ..documents import list_files, read_file
from .agents import _clean, _gather_or_cancel, fit_prompt
from .engine import Engine, _find_val, _sse_json
from .scenarios import (BURST_TOPICS, RunReq, SYSTEM_TURNS, TURNS_AT_ONCE, TURNS_DOC_CHARS, TURNS_FOLLOW, TURNS_OUT,
                        TURNS_QUESTIONS, TURNS_ROUNDS, TURNS_WATCH_S)
from .session import Session


def turns_docs(files: list[str], n: int, chars: int = TURNS_DOC_CHARS) -> list[tuple[str, str]]:
    """(label, text) for each of `n` agents: a piece of `chars` characters of the picked files, or of the library when none
    are picked, each agent another piece; a library with no files gives built-in notes on a topic."""
    names = files or [f["path"] for f in list_files()]
    if not names:
        return [(f"notes on {BURST_TOPICS[i % len(BURST_TOPICS)]} #{i + 1}",
                 "\n".join(f"Note {k}: on day {k} station {i + 1} logged {(i + 1) * 7 + k * 13} readings about "
                           f"{BURST_TOPICS[i % len(BURST_TOPICS)]}, and the crew {'agreed' if k % 3 else 'argued'} about what they meant."
                           for k in range(1, 21))) for i in range(n)]  # fmt: skip
    texts: dict[str, str] = {}
    out = []
    for i in range(n):
        f = names[i % len(names)]
        t = texts.setdefault(f, read_file(f))
        start = (i // len(names) * chars) % max(1, len(t))
        piece = t[start:start + chars]
        out.append((f if i < len(names) else f"{f} (part {i // len(names) + 1})", piece if len(piece) >= 200 else t[:chars]))
    return out


async def _tokenize_plain(e: Engine, text: str) -> list[int]:
    """Tokens of text that continues a conversation (no beginning-of-text token)."""
    r = await e.client.post(f"http://127.0.0.1:{e.port}/tokenize", json={"content": text, "add_special": False, "parse_special": True})
    r.raise_for_status()
    return list(r.json()["tokens"])


async def _lifecycle(e: Engine) -> dict:
    """The agent lifecycle counters of sllm-server's /metrics ({} when unreadable)."""
    try:
        r = await e.client.get(f"http://127.0.0.1:{e.port}/metrics", timeout=3)
        v = _find_val(r.json(), "lifecycle")
        return v if isinstance(v, dict) else {}
    except (httpx.HTTPError, ValueError) as ex:
        live.log(f"reading the agent lifecycle counters failed ({type(ex).__name__}: {ex}); the run shows none")
        return {}


def _suspends(lc: dict) -> int:
    d = lc.get("suspends")
    return sum(v for k, v in d.items() if k != "errors" and isinstance(v, (int, float))) if isinstance(d, dict) else 0


async def agent_turn(e: Engine, aid: str, ids: list[int], n: int, on_first=None, tries: int = 6, on_token=None, on_ids=None) -> dict:
    """One streamed turn of agent `aid` (the engine wakes a parked agent by itself). Returns ids, text, ttft_ms, tok_s, stop
    (the server's stop_type: eos / limit / cancelled ...). `on_token(text)` gets each piece of text, `on_ids(k)` the number of
    tokens that came with it. A 503 (the pool is still full of an agent that has not been parked yet) is tried again a moment later."""
    body = {"prompt": ids, "n_predict": n, "temperature": 0, "stream": True, "return_tokens": True}
    for attempt in range(tries):
        turn = _TurnReader(e, time.monotonic(), on_first, on_token, on_ids)
        e.active += 1
        try:
            async with e.client.stream("POST", f"http://127.0.0.1:{e.port}/v1/agents/{aid}/turn", json=body) as resp:
                if resp.status_code == 503 and attempt < tries - 1:
                    await resp.aread()
                    await asyncio.sleep(0.25 * (attempt + 1))
                    continue
                if resp.status_code != 200:
                    raise RuntimeError(f"Sonic LLM HTTP {resp.status_code} on agent {aid}: {(await resp.aread())[:200]!r}")
                await turn.read(resp)
        finally:
            e.active -= 1
        return turn.finish()
    raise RuntimeError("unreachable")


@dataclass
class _TurnReader:
    """The stream of one agent turn read into `out` (ids, text, ttft_ms, stop), with the caller's hooks: `on_first()` at
    the first token, `on_token(text)` for each piece of text, `on_ids(k)` for the tokens that came with it."""
    e: Engine
    t0: float  # time.monotonic() when the turn was sent
    on_first: Callable[[], object] | None = None
    on_token: Callable[[str], object] | None = None
    on_ids: Callable[[int], object] | None = None
    out: dict = field(default_factory=lambda: {"ids": [], "text": ""})
    seen_first: bool = False  # a chunk with text but no token ids is also the first token: `out["ids"]` stays empty after it

    async def read(self, resp) -> None:
        async for line in resp.aiter_lines():
            d = _sse_json(line)
            if d is None:
                continue
            self.take(d)
            if d.get("stop"):
                self.out["stop"] = d.get("stop_type")
                break

    def take(self, d: dict) -> None:
        out, toks = self.out, d.get("tokens") or []
        if not (toks or d.get("content")):
            return
        if not self.seen_first:
            self.seen_first = True
            out["ttft_ms"] = (time.monotonic() - self.t0) * 1000
            out["t_first"] = time.monotonic()
            if self.on_first is not None:
                self.on_first()
        out["ids"] += toks
        out["text"] += d.get("content", "")
        self.e.tokens += len(toks)
        if self.on_token is not None:
            self.on_token(d.get("content", ""))
        if self.on_ids is not None:
            self.on_ids(max(1, len(toks)))

    def finish(self) -> dict:
        """The turn's output with its writing speed (tok_s: None for a single token)."""
        out = self.out
        n_out = len(out["ids"])
        span = time.monotonic() - out.get("t_first", self.t0)
        out["tok_s"] = (n_out - 1) / span if n_out > 1 and span > 0 else None
        return out


async def _agent_call(e: Engine, method: str, path: str, body: dict | None = None) -> dict:
    r = await e.client.request(method, f"http://127.0.0.1:{e.port}{path}", json=body, timeout=120)
    r.raise_for_status()
    return r.json() if r.content else {}


async def _turns(s: Session, req: RunReq, run_id: str, rt: str) -> dict:
    """The turns scenario on one engine. Returns the metrics."""
    e = s.engines[rt]
    n, k = req.agents, max(1, min(req.parallel or TURNS_AT_ONCE, req.agents))
    first_limit = e.ctx - TURNS_ROUNDS * TURNS_OUT - (TURNS_ROUNDS - 1) * TURNS_FOLLOW - 16
    if first_limit < 200:
        raise HTTPException(422, f"a context of {e.ctx} tokens per agent is too small for {TURNS_ROUNDS} rounds of notes (at least 512)")
    first, follow = await _turns_prompts(s, e, req, n, first_limit)
    lc0 = await _lifecycle(e)
    run = _TurnsRun(s, rt, run_id, first, follow, asyncio.Semaphore(k))
    watcher = asyncio.create_task(run.watch())
    try:
        run.ids_of = list(await _gather_or_cancel(run.make(i) for i in range(n)))
        t0 = time.monotonic()
        for r in range(TURNS_ROUNDS):
            live.BUS.publish({"type": "phase", "run": run_id, "rt": rt, "msg": f"Sonic LLM: round {r + 1} of {TURNS_ROUNDS}, {k} write at once"})
            await _gather_or_cancel(run.one(i, r) for i in range(n))
            await run.sample()  # the SSD is fullest when a round ends and every agent of it is parked
        rounds_wall = time.monotonic() - t0
        lc1 = await _lifecycle(e)
        same = await run.same_in_ram()
    finally:
        watcher.cancel()
        await asyncio.gather(*(run.delete(a) for a in list(run.live)), return_exceptions=True)
    live.BUS.publish({"type": "token", "run": run_id, "rt": rt, "text": run.report(n, k, rounds_wall, same)})
    return run.metrics(n, k, rounds_wall, same, _lifecycle_delta(lc0, lc1))


async def _turns_prompts(s: Session, e: Engine, req: RunReq, n: int, first_limit: int) -> tuple[list[list[int]], list[list[int]]]:
    """The tokens of each agent's first prompt (its piece of document and the first question, cut to `first_limit`) and of
    the follow-up question of each later round."""
    docs = turns_docs(req.files, n, req.doc_chars or TURNS_DOC_CHARS)
    first = await asyncio.gather(*(fit_prompt(e, s.one_shot(SYSTEM_TURNS, f"<document {lbl}>\n{text}\n</document>\n\nQuestion: {TURNS_QUESTIONS[0]}"),
                                              first_limit) for lbl, text in docs))  # fmt: skip
    follow = [await _tokenize_plain(e, s.template.turn_text(q)) for q in TURNS_QUESTIONS[1:TURNS_ROUNDS]]
    return first, follow


def _lifecycle_delta(lc0: dict, lc1: dict) -> dict:
    """Parks and wakes of a run from the engine's lifecycle counters before (`lc0`) and after it (`lc1`); the wake times are
    the engine's since it started."""
    return {"suspends": _suspends(lc1) - _suspends(lc0), "restores": int(lc1.get("restores") or 0) - int(lc0.get("restores") or 0),
            "restore_ms_p50": lc1.get("restore_ms_p50"), "restore_ms_p95": lc1.get("restore_ms_p95"),
            "restore_ms_includes_earlier_runs": bool(lc0.get("restores")) or None}  # fmt: skip


class _TurnsRun:
    """The agents of one turns run on one engine: N agents take TURNS_ROUNDS turns, at most K (`gate`) at once, and each is
    parked on the SSD between its turns. Keeps their server ids, agent 1's tokens and text of each round, and the counts
    the metrics take."""

    def __init__(self, s: Session, rt: str, run_id: str, first: list[list[int]], follow: list[list[int]], gate: asyncio.Semaphore):
        self.s, self.e, self.rt, self.run_id = s, s.engines[rt], rt, run_id
        self.first, self.follow = first, follow  # each agent's first prompt; the follow-up question of each later round
        self.gate = gate
        self.live: set[str] = set()
        self.ids_of: list[str] = []
        self.first_agent: list[list[int]] = []  # agent 0's tokens of each round
        self.first_text: list[str] = []
        self.ttfts: list[float] = []
        self.peak = {"ssd": 0, "parked": 0}
        self.stats = {"suspend_errors": 0, "written": 0, "context": sum(len(x) for x in first)}

    def ev(self, i, state, **kw):
        live.BUS.publish({"type": "agent", "run": self.run_id, "rt": self.rt, "i": i, "label": f"agent {i + 1}", "state": state, **kw})

    def prompt(self, i: int, r: int) -> list[int]:
        return self.first[i] if r == 0 else self.follow[r - 1]

    async def sample(self):
        lc = await _lifecycle(self.e)
        self.peak["ssd"] = max(self.peak["ssd"], int(lc.get("ssd_bytes") or 0))
        ag = lc.get("agents") if isinstance(lc.get("agents"), dict) else {}
        self.peak["parked"] = max(self.peak["parked"], int(ag.get("suspended") or 0) + int(ag.get("restoring") or 0))

    async def watch(self):
        while True:
            await self.sample()
            await asyncio.sleep(TURNS_WATCH_S)

    async def make(self, i):
        aid = (await _agent_call(self.e, "POST", "/v1/agents", {"lane": "background"}))["id"]
        self.live.add(aid)
        self.ev(i, "waiting")
        return aid

    async def delete(self, aid):
        if aid in self.live:
            self.live.discard(aid)
            try:
                await _agent_call(self.e, "DELETE", f"/v1/agents/{aid}")
            except Exception as ex:  # the run goes on, but a leaked agent keeps its memory in the server
                live.log(f"deleting agent {aid} failed ({type(ex).__name__}: {ex}); it may still hold server memory")

    async def one(self, i, r):
        """Agent `i`'s turn of round `r`; then it is parked on the SSD, or deleted after its last round."""
        async with self.gate:  # at most K agents hold KV in RAM: the next one starts when this one is parked
            self.ev(i, "reading")
            res = await agent_turn(self.e, self.ids_of[i], self.prompt(i, r), TURNS_OUT, on_first=lambda: self.ev(i, "writing"))
            self.note(i, r, res)
            if r < TURNS_ROUNDS - 1:
                await self.park(i)
            else:
                await self.delete(self.ids_of[i])
                self.ev(i, "done", ttft_ms=res.get("ttft_ms"), tok_s=res.get("tok_s"))

    def note(self, i, r, res):
        """The counts of one turn: tokens written, the notes (context) it adds, its TTFT; agent 1's tokens and text."""
        self.stats["written"] += len(res["ids"])
        self.stats["context"] += len(res["ids"]) + (len(self.follow[r - 1]) if r else 0)
        if res.get("ttft_ms") is not None:
            self.ttfts.append(res["ttft_ms"])
        if i == 0:
            self.first_agent.append(res["ids"])
            self.first_text.append(_clean(res["text"], self.s))

    async def park(self, i):
        try:
            await _agent_call(self.e, "POST", f"/v1/agents/{self.ids_of[i]}/suspend", {"tier": "ssd"})
        except Exception as ex:  # it stays in RAM: its row says it waits for its next turn, not that it is parked
            self.stats["suspend_errors"] += 1
            live.log(f"agent {i + 1}: suspend failed ({ex})")
            self.ev(i, "waiting")
            return
        self.ev(i, "parked")

    async def same_in_ram(self) -> bool | None:
        """Exactness after the SSD round trips: agent 0's conversation again, in RAM the whole time. Whether its tokens are
        the same; None when the check could not run (logged)."""
        ram: list[list[int]] = []
        aid = None
        try:
            aid = (await _agent_call(self.e, "POST", "/v1/agents", {"lane": "background"}))["id"]
            self.live.add(aid)
            for r in range(TURNS_ROUNDS):
                ram.append((await agent_turn(self.e, aid, self.prompt(0, r), TURNS_OUT))["ids"])
            return ram == self.first_agent
        except Exception as ex:
            live.log(f"the same-in-RAM check failed to run: {ex}")
            return None
        finally:
            if aid:
                await self.delete(aid)

    def report(self, n: int, k: int, rounds_wall: float, same: bool | None) -> str:
        """What the page shows of the run: the rounds, the RAM check's verdict, agent 1's answers."""
        return (f"{n} agents took {TURNS_ROUNDS} rounds, {k} writing at once; "
                f"{self.stats['written']:,} tokens written in {rounds_wall:.1f} s.\n"
                + ("Agent 1 wrote the same words after being parked on disk as it did staying in memory\n" if same else
                   "Agent 1's words differ from the run that stayed in memory\n" if same is False else "")
                + "".join(f"\nAgent 1, round {i + 1}: {t[:200]}" for i, t in enumerate(self.first_text)))  # fmt: skip

    def metrics(self, n: int, k: int, rounds_wall: float, same: bool | None, lifecycle: dict) -> dict:
        e, kvb, stats, peak = self.e, self.s.kvb, self.stats, self.peak
        return {"agents": n, "rounds": TURNS_ROUNDS, "at_once": k, "rounds_wall_s": round(rounds_wall, 2), "tokens_written": stats["written"],
                "write_tok_s": round(stats["written"] / rounds_wall, 1) if rounds_wall > 0 else None, **lifecycle,
                "suspend_errors": stats["suspend_errors"] or None, "ssd_bytes": peak["ssd"], "ssd_mb": round(peak["ssd"] / 1e6, 1),
                "parked_peak": peak["parked"], "block_cells": e.cells, "notes_tokens": stats["context"],
                "kv_reserved_mb": round(e.cells * kvb / 2**20, 1) if kvb else None, "same_as_ram": same,
                "ttft_ms_p50": round(statistics.median(self.ttfts), 1) if self.ttfts else None}  # fmt: skip
