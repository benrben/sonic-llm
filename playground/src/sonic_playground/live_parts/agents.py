"""What the tests of the live playground (live.py) are made of: the chat answer (with its answer level), agents on the
background lane, prompts cut to fit the context, and the numbers a run's replies sum to."""

from __future__ import annotations

import asyncio
import contextlib
import re
import statistics
import time

from .. import levels, live  # live: its log and BUS are looked up there when used (tests patch them)
from ..chat_template import split_thinking
from ..documents import list_files, read_file
from .config import AGENT_REPLY_CHARS, AGENT_TICK_S, ANSWER_ROOM, CTX, DOC_CHARS, SERIES
from .engine import Engine
from .scenarios import SYSTEM_CHAT
from .session import Session


# -- documents --------------------------------------------------------------------------------------------------------
def _docs(files: list[str], n: int, chars: int | None = DOC_CHARS) -> list[tuple[str, str]]:
    names = files or [f["path"] for f in sorted(list_files(), key=lambda f: -f["bytes"])][:n]
    return [(f, read_file(f, chars)) for f in names[:n]]


def doc_need(files: list[str], n: int, chars: int | None) -> int | None:
    """Tokens a document job's longest prompt and answer can take, from the file sizes: at most one token per 3 bytes of text, an
    answer of half the prompt (a longer one ends at the window's edge), and the framing. None without files: the old sizing."""
    sizes = {f["path"]: f["bytes"] for f in list_files()}
    names = files or sorted(sizes, key=lambda k: -sizes[k])[:n]
    if not names:
        return None
    longest = max(min(sizes.get(f, 0), chars or CTX * 4) for f in names[:n])
    return longest // 3 * 3 // 2 + ANSWER_ROOM + 256


def _summ(rs: list[dict]) -> dict:
    def med(k):
        xs = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
        return round(statistics.median(xs), 2) if xs else None
    return {"n": len(rs), "ttft_ms_p50": med("ttft_ms"), "decode_tok_s_p50": med("decode_tok_s"),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rs), "out_tokens": sum(r["n"] for r in rs),
            "cached_tokens": sum(r.get("cached_tokens") or 0 for r in rs),
            "deferred_retries": sum(r.get("deferred") or 0 for r in rs)}  # fmt: skip


def _write_tok_s(rs: list[dict]) -> float | None:
    """Output tokens per second over the writing time only: from the first token of the first agent to the last token of
    the last agent (reading is not in it)."""
    firsts, lasts = [r["first_s"] for r in rs if "first_s" in r], [r["last_s"] for r in rs if "last_s" in r]
    span = max(lasts) - min(firsts) if firsts and lasts else 0
    return round(sum(r["n"] for r in rs) / span, 1) if span > 0 else None


def _read_only_tok_s(rs: list[dict], start: float) -> float | None:
    """Prompt tokens the engine really evaluated (cached ones excluded) per second over the reading time only: from
    `start` (time.monotonic()) until the last agent's first token."""
    firsts = [r["first_s"] for r in rs if "first_s" in r]
    span = max(firsts) - start if firsts else 0
    return round(sum(r["prompt_tokens"] - (r.get("cached_tokens") or 0) for r in rs) / span, 1) if span > 0 else None


def prompt_room(ctx: int, cap: int | None) -> int:
    """Tokens a prompt may take with `cap` tokens kept for the answer. A limit near or past the context keeps at most half
    of it: the prompt is mostly the document, and cutting that away leaves the model nothing to read."""
    return ctx - min(cap or ANSWER_ROOM, ctx // 2) - 8


def answer_room(ctx: int, prompt_tokens: int, cap: int | None) -> int:
    """Tokens an answer may take: what is left of the context after the prompt, or the optional cap when lower."""
    room = max(1, ctx - prompt_tokens - 8)
    return min(cap, room) if cap else room


async def _chat_answer(s: Session, rt: str, run_id: str, system: str, message: str, *, remember: bool, lane="interactive"):
    e = s.engines[rt]
    ctx = e.ctx if rt == "sllm" else min(s.cells_max, s.train or CTX)
    cap = s.lim["answer"]
    lv = levels.norm_level(s.level)

    def on_token(t):
        live.BUS.publish({"type": "token", "run": run_id, "rt": rt, "text": t})

    def build_for(sys_prompt: str, lcap: int | None):
        text = s.chat_prompt(rt, sys_prompt, message) if remember else s.one_shot(sys_prompt, message)
        return fit_prompt(e, text, prompt_room(ctx, min(lcap, cap) if lcap and cap else lcap or cap))

    if lv == "off":
        ids = await build_for(system, None)
        n = answer_room(ctx, len(ids), cap)  # no cap: until the model stops or the context is full
        r = await e.generate(ids, n, lane=lane, on_token=on_token)
    else:
        top = max(levels.level_caps(lv))

        async def build(stage_prompt: str) -> list[int]:
            return await build_for(levels.stage_system(stage_prompt, system, SYSTEM_CHAT), top)

        def phase(_row, _label, state, **_kw):  # the level tells what it does; the tokens go to the page as they are sent
            live.BUS.publish({"type": "phase", "run": run_id, "rt": rt, "msg": f"{SERIES[rt]}: {state}"})

        r = await levels.answer_generate(e, build, lv, lane=lane, on_token=on_token, progress=phase,
                                         cap=lambda plen, lcap: answer_room(ctx, plen, min(lcap, cap) if cap else lcap))
    if remember:
        s.history[rt].append((message, _clean(r["text"], s)))
    return r


async def _agents(s: Session, rt: str, run_id: str, prompts: list[tuple[str, str]], n_out: int | None = None,
                  extra=None, room=None) -> list[dict]:
    """Every prompt as one agent on the background lane, all at once. `extra(result)` adds fields to the agent's done event;
    `room(prompt_tokens)` limits a reply by what its agent read (a limit of the run wins)."""
    e = s.engines[rt]
    cap = n_out or s.lim["agent"]
    # Both engines give an agent the same context (sllm-server's --ctx) and run as many at once as their KV holds of
    # that: llama-server's shared pool fails a request with "Context size has been exceeded" when it overfills.
    ctx = s.engines["sllm"].ctx
    limit = prompt_room(ctx, cap)
    gate = asyncio.Semaphore(max(1, e.cells // ctx)) if rt == "llama" else contextlib.nullcontext()

    async def one(i, label, prompt):
        live.BUS.publish({"type": "agent", "run": run_id, "rt": rt, "i": i, "label": label, "state": "reading"})
        ids = await fit_prompt(e, prompt, limit)
        wrote = {"n": 0, "at": 0.0}

        def writing(_text):
            wrote["n"] += 1
            now = time.monotonic()
            if wrote["n"] == 1 or now - wrote["at"] >= AGENT_TICK_S:  # the first token, then a count now and then
                wrote["at"] = now
                live.BUS.publish({"type": "agent", "run": run_id, "rt": rt, "i": i, "label": label, "state": "writing", "tokens": wrote["n"]})

        async with gate:
            r = await e.generate(ids, answer_room(ctx, len(ids), cap or (room(len(ids)) if room else None)), lane="background",
                                 on_token=writing)
        r["label"] = label
        live.BUS.publish({"type": "agent", "run": run_id, "rt": rt, "i": i, "label": label, "state": "done",
                     "ttft_ms": r.get("ttft_ms"), "tok_s": r.get("decode_tok_s"), "cached": r.get("cached_tokens"),
                     "prompt_tokens": r["prompt_tokens"], "tokens": r["n"], "stop": r.get("stop"),
                     "reply": _clean(r["text"], s)[:AGENT_REPLY_CHARS], **(extra(r) if extra else {})})  # fmt: skip
        return r

    return await _gather_or_cancel(one(i, lbl, p) for i, (lbl, p) in enumerate(prompts))


async def _gather_or_cancel(coros) -> list:
    """gather that, when one fails, cancels the others and waits for them before it raises."""
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


_DOC = re.compile(r"(<document [^>]*>\n)(.*?)(\n</document>)", re.S)


async def fit_prompt(e: Engine, prompt: str, limit: int) -> list[int]:
    """Tokenize `prompt`; when it is longer than `limit` tokens, shorten the text inside its <document> tags (the question
    after them stays) and try again. The engine refuses a prompt past its context, so an agent reads less instead."""
    ids = await e.tokenize(prompt)
    for _ in range(4):
        m = _DOC.search(prompt)
        if len(ids) <= limit or m is None:
            break
        doc = m.group(2)
        doc_tokens = max(1, len(ids) * len(doc) // max(1, len(prompt)))  # the document's share of the tokens
        keep = max(0, int(len(doc) * (doc_tokens - (len(ids) - limit)) / doc_tokens * 0.9))
        prompt = prompt[:m.start(2)] + doc[:keep] + "\n[…cut to fit the context]" + prompt[m.end(2):]
        ids = await e.tokenize(prompt)
    return ids


def _clean(text: str, s: Session) -> str:
    """The answer of a reply: end-of-generation tokens and the model's thinking cut off (history keeps answers only, as
    the models' own chat templates do)."""
    for eog in s.template.eog:
        text = text.split(eog)[0]
    return split_thinking(text)[1].strip()
