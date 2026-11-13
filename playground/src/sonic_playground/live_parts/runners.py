"""The tests of the live playground (live.py), each run on one engine: chat, swarm, shared document, document jobs, chat
under load, exactness, throughput, relay, follow-up, research team, long document (turns: turns.py). Each returns its own
metrics."""

from __future__ import annotations

import asyncio
import time

from fastapi import HTTPException

from .. import live  # its log and BUS are looked up there when used (tests patch them)
from ..documents import list_files, read_file
from ..gguf_meta import train_ctx
from .agents import _agents, _chat_answer, _clean, _docs, _gather_or_cancel, _read_only_tok_s, _summ, _write_tok_s
from .config import CTX, SERIES
from .scenarios import (BURST_OUT, DOC_TASKS, RELAY_TRIES, RunReq, SHARED_QUESTIONS, SYSTEM_AGENT, SYSTEM_CHAT,
                        SYSTEM_MANAGER, SYSTEM_RELAY, SYSTEM_TEAM, _suggest, burst_prompts, doc_room, doc_task_report,
                        doc_verdict, parse_report, questions_of)
from .session import Session
from .turns import _turns


async def _run_chat(s: Session, req: RunReq, run_id: str, rt: str) -> dict:
    """Chat: one answer on the interactive lane, the attached files in the message."""
    msg = req.message
    if req.files:
        msg += "\n\n" + "\n\n".join(f"<file {f}>\n{read_file(f, req.doc_chars)}\n</file>" for f in req.files)
    r = await _chat_answer(s, rt, run_id, SYSTEM_CHAT, msg, remember=True)
    return {"ttft_ms": r.get("ttft_ms"), "decode_tok_s": r.get("decode_tok_s"), "itl_p95_ms": r.get("itl_p95_ms"),
            "prompt_tokens": r["prompt_tokens"], "out_tokens": r["n"], "cached_tokens": r.get("cached_tokens")}  # fmt: skip


async def _run_swarm(s: Session, req: RunReq, run_id: str, rt: str, w0: float) -> dict:
    """Agent swarm: one agent per document, then the chat merges their notes."""
    docs = _docs(req.files, req.agents, req.doc_chars)
    q = req.message or _suggest("swarm")
    rs = await _agents(s, rt, run_id, [(f, s.one_shot(SYSTEM_AGENT, f"<document {f}>\n{t}\n</document>\n\nQuestion: {q}"))
                                       for f, t in docs])  # fmt: skip
    swarm_wall = time.monotonic() - w0
    s.readers = {"kind": "swarm", "prefix": "Question: ", "system": SYSTEM_AGENT, "docs": docs}
    notes = "\n\n".join(f"Notes on {r['label']}:\n{_clean(r['text'], s)}" for r in rs)
    fin = await _chat_answer(s, rt, run_id, SYSTEM_CHAT,
                             f"{len(rs)} agents read one document each and wrote these notes.\n\n<document notes>\n{notes}\n</document>\n\n"
                             f"Using only these notes, answer: {q}", remember=False)  # fmt: skip
    return {**_summ(rs), "agents": len(rs), "agents_asked": req.agents, "agents_note": "one per document",
            "swarm_wall_s": round(swarm_wall, 2),
            "swarm_tok_s": round(sum(r["n"] for r in rs) / swarm_wall, 1), "write_tok_s": _write_tok_s(rs),
            "read_wall_tok_s": round(sum(r["prompt_tokens"] for r in rs) / swarm_wall, 1),  # over the whole swarm, writing included
            "read_only_tok_s": _read_only_tok_s(rs, w0),
            "final_ttft_ms": fin.get("ttft_ms"), "final_decode_tok_s": fin.get("decode_tok_s")}  # fmt: skip


async def _run_shared(s: Session, req: RunReq, run_id: str, rt: str, w0: float) -> dict:
    """Many questions, one document: the shared prefix is read once."""
    f = req.files[0] if req.files else max(list_files(), key=lambda x: x["bytes"])["path"]
    doc = read_file(f, req.doc_chars)
    qs = [SHARED_QUESTIONS[i % len(SHARED_QUESTIONS)] for i in range(req.agents)]
    rs = await _agents(s, rt, run_id, [(qq, s.one_shot(SYSTEM_AGENT, f"<document {f}>\n{doc}\n</document>\n\nQuestion: {qq}"))
                                       for qq in qs])  # fmt: skip
    wall = time.monotonic() - w0
    return {**_summ(rs), "agents": len(rs), "agents_asked": req.agents, "wall_s": round(wall, 2),
            "agg_tok_s": round(sum(r["n"] for r in rs) / wall, 1),  # whole run, reading included
            "write_tok_s": _write_tok_s(rs),
            "cache_hit_pct": round(100 * _summ(rs)["cached_tokens"] / max(1, _summ(rs)["prompt_tokens"]), 1)}  # fmt: skip


async def _run_doc_task(s: Session, req: RunReq, run_id: str, rt: str, w0: float) -> dict:
    """Classify / extract / compress / summarize: one agent per document does the job, the run checks the replies."""
    task = DOC_TASKS[req.scenario]
    docs = _docs(req.files, req.agents, req.doc_chars)
    msg = req.message or _suggest(req.scenario)
    ask = task["ask"].format(m=msg)

    def verdict(r):
        ok, said = doc_verdict(req.scenario, msg, _clean(r["text"], s), r)
        return {"ok": ok, "verdict": said}

    rs = await _agents(s, rt, run_id, [(f, s.one_shot(task["system"], f"<document {f}>\n{t}\n</document>\n\n{ask}")) for f, t in docs],
                       extra=verdict, room=lambda n: doc_room(task, s.thinking, n))  # fmt: skip
    wall = time.monotonic() - w0
    lines, did = doc_task_report(req.scenario, msg, rs, [_clean(r["text"], s) for r in rs])
    did["cut_off"] = sum(r.get("stop") == "limit" for r in rs)
    if did["cut_off"]:
        lines.append(f"{did['cut_off']} of {len(rs)} hit the length limit: the model did not end by itself, so these replies are cut.")
    live.BUS.publish({"type": "token", "run": run_id, "rt": rt, "text": "\n".join(lines)})
    return {**_summ(rs), "agents": len(rs), "agents_asked": req.agents, "agents_note": "one per document", "wall_s": round(wall, 2),
            "agg_tok_s": round(sum(r["n"] for r in rs) / wall, 1), "write_tok_s": _write_tok_s(rs),
            "read_only_tok_s": _read_only_tok_s(rs, w0), **did}  # fmt: skip


async def _run_load(s: Session, req: RunReq, run_id: str, rt: str) -> dict:
    """Chat under load: the question alone, then again while a swarm reads."""
    q = req.message or _suggest("load")
    alone = await _chat_answer(s, rt, run_id, SYSTEM_CHAT, q, remember=False)
    live.BUS.publish({"type": "token", "run": run_id, "rt": rt, "text": "\n\n— now again, with the swarm running —\n\n"})
    docs = _docs(req.files, req.agents, req.doc_chars)
    swarm = asyncio.create_task(_agents(s, rt, run_id, [(f, s.one_shot(SYSTEM_AGENT, f"<document {f}>\n{t}\n</document>\n\nSummarise it."))
                                                         for f, t in docs]))  # fmt: skip
    try:
        await asyncio.sleep(1.0)  # the swarm is reading
        loaded = await _chat_answer(s, rt, run_id, SYSTEM_CHAT, q, remember=False)
        rs = await swarm
    finally:  # a stop or a failed answer: the swarm ends with the run, not after it
        swarm.cancel()
        await asyncio.gather(swarm, return_exceptions=True)
    return {"ttft_alone_ms": alone.get("ttft_ms"), "ttft_loaded_ms": loaded.get("ttft_ms"),
            "decode_alone_tok_s": alone.get("decode_tok_s"), "decode_loaded_tok_s": loaded.get("decode_tok_s"),
            "itl_p95_alone_ms": alone.get("itl_p95_ms"), "itl_p95_loaded_ms": loaded.get("itl_p95_ms"),
            "chat_slowdown": round(alone["decode_tok_s"] / loaded["decode_tok_s"], 2)
            if alone.get("decode_tok_s") and loaded.get("decode_tok_s") else None,
            "agents": len(rs), "same_answer_under_load": alone["ids"] == loaded["ids"]}  # fmt: skip


async def _run_exact(s: Session, req: RunReq, run_id: str, rt: str) -> dict:
    """Exactness: the question alone, then as one agent among others; are the tokens the same?"""
    q = req.message or _suggest("exact")
    alone = await _chat_answer(s, rt, run_id, SYSTEM_CHAT, q, remember=False)
    others = _docs(req.files, req.agents - 1, req.doc_chars)
    prompts = [("the question", s.one_shot(SYSTEM_CHAT, q))] + [
        (f, s.one_shot(SYSTEM_AGENT, f"<document {f}>\n{t}\n</document>\n\nSummarise it.")) for f, t in others]
    rs = await _agents(s, rt, run_id, prompts)
    mixed = rs[0]
    n = min(len(alone["ids"]), len(mixed["ids"]))
    diff = next((i for i in range(n) if alone["ids"][i] != mixed["ids"][i]), None)
    same = alone["ids"] == mixed["ids"]
    live.BUS.publish({"type": "token", "run": run_id, "rt": rt,
                 "text": "\n\n" + ("✅ bit-identical next to %d others" % (len(rs) - 1) if same else
                                   f"❌ differs at token {diff if diff is not None else n}")})  # fmt: skip
    return {"exact": same, "first_diff": diff, "others": len(rs) - 1, "tokens_compared": max(len(alone["ids"]), len(mixed["ids"]))}


async def _run_burst(s: Session, req: RunReq, run_id: str, rt: str) -> dict:
    """Throughput: N agents each write BURST_OUT tokens at once, end-of-text ignored."""
    e = s.engines[rt]
    prompts = burst_prompts(s, req.agents)
    ids_all = await asyncio.gather(*(e.tokenize(p) for _, p in prompts))

    async def one(i, ids):
        live.BUS.publish({"type": "agent", "run": run_id, "rt": rt, "i": i, "label": f"agent {i + 1}", "state": "reading"})
        r = await e.generate(ids, BURST_OUT, lane="background", ignore_eos=True)
        live.BUS.publish({"type": "agent", "run": run_id, "rt": rt, "i": i, "label": f"agent {i + 1}", "state": "done",
                     "ttft_ms": r.get("ttft_ms"), "tok_s": r.get("decode_tok_s")})  # fmt: skip
        return r

    t0b = time.monotonic()
    rs = await _gather_or_cancel(one(i, ids) for i, ids in enumerate(ids_all))
    wall_b = time.monotonic() - t0b
    live.BUS.publish({"type": "token", "run": run_id, "rt": rt, "text": f"{len(rs)} agents wrote {sum(r['n'] for r in rs):,} tokens "
                 f"in {wall_b:.1f} s.\n\nAgent 1: " + _clean(rs[0]["text"], s)[:300] + "…"})  # fmt: skip
    return {**_summ(rs), "agents": len(rs), "agents_asked": req.agents,
            "agg_tok_s": round(sum(r["n"] for r in rs) / wall_b, 1), "write_tok_s": _write_tok_s(rs),  # whole run / writing only
            "per_agent_tok_s": _summ(rs)["decode_tok_s_p50"]}  # fmt: skip


async def _run_relay(s: Session, req: RunReq, run_id: str, rt: str, w0: float) -> dict:
    """Relay of fresh agents: laps of workers over a plan of stories, a check on each output, notes between laps."""
    goal = req.message or _suggest("relay")
    docs = _docs(req.files, len(req.files) if req.files else min(12, len(list_files())), req.doc_chars)
    workers = max(1, min(req.agents, len(docs)))
    todo = list(range(len(docs)))
    done: dict[int, list[str]] = {}
    tries = {i: 0 for i in todo}
    notes: list[str] = []
    laps, retries, rs_all = 0, 0, []

    def plan() -> str:
        return "\n".join(f"[{'x' if i in done else ' '}] story {i + 1}: {docs[i][0]}" for i in range(len(docs)))

    while todo:
        wave, todo = todo[:workers], todo[workers:]
        laps += 1
        live.BUS.publish({"type": "phase", "run": run_id, "rt": rt, "msg": f"{SERIES[rt]}: lap {laps}, {len(wave)} fresh agents"})
        # what does not change first (instructions, goal, the story's document): a retry finds it in memory and
        # reads only the plan and the notes, which every lap changes
        prompts = [(docs[i][0], s.one_shot(SYSTEM_RELAY, f"Goal: {goal}\n\nYour story: story {i + 1}, from this document.\n"
                                          f"<document {docs[i][0]}>\n{docs[i][1]}\n</document>\n\nPlan:\n{plan()}\n\n"
                                          "Notes from earlier workers:\n" + ("\n".join(notes[-30:]) or "(none yet)")
                                          + "\n\nWrite 2 to 4 lines that start with FACT:, then one line that starts with LEARNED: "
                                          "(one tip for the next worker). Nothing else."))
                   for i in wave]  # fmt: skip
        rs = await _agents(s, rt, run_id, prompts)
        rs_all += rs
        retries += _relay_check(s, wave, rs, done, tries, notes, todo)
    wall = time.monotonic() - w0
    report = "\n".join(f"{docs[i][0]}:\n" + "\n".join(done[i]) for i in sorted(done))
    live.BUS.publish({"type": "token", "run": run_id, "rt": rt, "text": f"Plan after {laps} laps:\n{plan()}\n\nNotes:\n"
                 + ("\n".join(notes) or "(none)") + "\n\nFacts:\n" + (report or "(none)") + "\n"})  # fmt: skip
    sm = _summ(rs_all)
    return {**sm, "agents": len(rs_all), "agents_asked": req.agents, "agents_note": f"{workers} workers a lap, a fresh agent each",
            "laps": laps, "stories": len(docs), "stories_done": len(done), "retries": retries, "wall_s": round(wall, 2),
            "write_tok_s": _write_tok_s(rs_all),
            "cache_hit_pct": round(100 * sm["cached_tokens"] / max(1, sm["prompt_tokens"]), 1)}  # fmt: skip


def _relay_check(s: Session, wave: list[int], rs: list[dict], done: dict[int, list[str]], tries: dict[int, int],
                 notes: list[str], todo: list[int]) -> int:
    """One lap's outputs through the check: a story with facts in the asked form is done (its tip goes in the notes), one
    without goes back on `todo` while it has tries left. Updates done, tries, notes and todo; returns the retries."""
    retries = 0
    for i, r in zip(wave, rs):
        text = _clean(r["text"], s)
        facts = [ln.strip() for ln in text.splitlines() if ln.strip().upper().startswith("FACT:")]
        tip = next((ln.strip()[8:].strip() for ln in text.splitlines() if ln.strip().upper().startswith("LEARNED:")), "")
        tries[i] += 1
        if facts:  # the check: facts in the asked form
            done[i] = facts
            if tip:
                notes.append(f"- story {i + 1}: {tip}")
        elif tries[i] < RELAY_TRIES:
            retries += 1
            todo.append(i)
            notes.append(f"- story {i + 1}: a worker gave no FACT: lines; write them exactly so")
    return retries


async def _run_followup(s: Session, req: RunReq, run_id: str, rt: str, w0: float) -> dict:
    """Ask the readers again: the last swarm / team's agents answer from their documents still in memory."""
    rd, q = s.readers, req.message or _suggest("followup")
    # the same prompt up to the question: the engine finds each document's cells and reads only the question
    rs = await _agents(s, rt, run_id, [(f, s.one_shot(rd["system"], f"<document {f}>\n{t}\n</document>\n\n{rd['prefix']}{q}"))
                                       for f, t in rd["docs"]])  # fmt: skip
    ask_wall = time.monotonic() - w0
    answers = "\n\n".join(f"Answer from {r['label']}:\n{_clean(r['text'], s)}" for r in rs)
    fin = await _chat_answer(s, rt, run_id, SYSTEM_CHAT,
                             f"{len(rs)} agents each read one document and answered this question from it.\n\n"
                             f"<document answers>\n{answers}\n</document>\n\nUsing only these answers, answer: {q}", remember=False)  # fmt: skip
    sm = _summ(rs)
    return {**sm, "agents": len(rs), "agents_note": f"the readers of the last {rd['kind']} run",
            "swarm_wall_s": round(ask_wall, 2), "write_tok_s": _write_tok_s(rs),
            "reused_pct": round(100 * sm["cached_tokens"] / max(1, sm["prompt_tokens"]), 1),  # read from memory, not again
            "final_ttft_ms": fin.get("ttft_ms"), "final_decode_tok_s": fin.get("decode_tok_s")}  # fmt: skip


def _team_found(s: Session, rs: list[dict], qs: list[str]) -> dict[int, list[tuple[str, str]]]:
    """{question number: [(document name, answer)]} from the agents' reports."""
    found = {i: [] for i in range(1, len(qs) + 1)}
    for r in rs:
        for qi, a in parse_report(_clean(r["text"], s), len(qs)).items():
            found[qi].append((r["label"].split("/")[-1], a))
    return found


async def _team_agents(s: Session, rt: str, run_id: str, docs: list[tuple[str, str]], qs: list[str]) -> list[dict]:
    """One agent per document answers every question from it or says NOT FOUND; its row tells how many it found."""
    numbered = "\n".join(f"Q{i}: {q}" for i, q in enumerate(qs, 1))
    n_out = None  # agents answer until they stop (or an optional limit)
    return await _agents(s, rt, run_id, [(f, s.one_shot(SYSTEM_TEAM, f"<document {f}>\n{t}\n</document>\n\nQuestions:\n{numbered}"))
                                         for f, t in docs], n_out=n_out,
                         extra=lambda r: {"found": len(parse_report(_clean(r["text"], s), len(qs))), "of": len(qs)})  # fmt: skip


async def _team_manager(s: Session, rt: str, run_id: str, qs: list[str], found: dict[int, list[tuple[str, str]]], n_docs: int) -> dict:
    """The manager merges the agents' findings into the final answer to each question (shown with what the agents found)."""
    live.BUS.publish({"type": "token", "run": run_id, "rt": rt, "text": "Agents found: " + " · ".join(
        f"Q{i} in {len(found[i])}/{n_docs} docs" for i in found) + "\n\nManager:\n"})  # fmt: skip
    live.BUS.publish({"type": "phase", "run": run_id, "rt": rt, "msg": f"{SERIES[rt]}: manager writing"})
    reports = "\n\n".join(f"Q{i}: {qs[i - 1]}\n" + ("\n".join(f"- [{d}] {a}" for d, a in found[i]) or "- no agent found it")
                           for i in found)  # fmt: skip
    return await _chat_answer(s, rt, run_id, SYSTEM_MANAGER, f"Questions and the agents' findings:\n\n<document findings>\n{reports}\n</document>\n\n"
                                                             f"Write the final answer to each question.", remember=False)  # fmt: skip


async def _run_team(s: Session, req: RunReq, run_id: str, rt: str, w0: float) -> dict:
    """Research team: one agent per document answers every question or says NOT FOUND; a manager merges the reports."""
    qs = questions_of(req.message or _suggest("team"))
    if not qs:
        raise HTTPException(422, "write at least one question")
    docs = _docs(req.files, req.agents, req.doc_chars)
    rs = await _team_agents(s, rt, run_id, docs, qs)
    team_wall = time.monotonic() - w0
    s.readers = {"kind": "team", "prefix": "Questions:\nQ1: ", "system": SYSTEM_TEAM, "docs": docs}
    found = _team_found(s, rs, qs)
    mgr = await _team_manager(s, rt, run_id, qs, found, len(rs))
    n_found = sum(len(v) for v in found.values())
    return {**_summ(rs), "agents": len(rs), "agents_asked": req.agents, "agents_note": "one per document",
            "questions": len(qs), "swarm_wall_s": round(team_wall, 2),
            "read_tokens": sum(r["prompt_tokens"] for r in rs),
            "read_wall_tok_s": round(sum(r["prompt_tokens"] for r in rs) / team_wall, 1),  # over the whole swarm, writing included
            "read_only_tok_s": _read_only_tok_s(rs, w0),
            "swarm_tok_s": round(sum(r["n"] for r in rs) / team_wall, 1), "write_tok_s": _write_tok_s(rs),
            "found_pct": round(100 * n_found / max(1, len(rs) * len(qs)), 1),
            "manager_prompt_tokens": mgr["prompt_tokens"], "final_ttft_ms": mgr.get("ttft_ms"),
            "final_decode_tok_s": mgr.get("decode_tok_s"),
            "docs_from": "picked files" if req.files else "the biggest files"}  # fmt: skip


async def _run_long(s: Session, req: RunReq, run_id: str, rt: str) -> dict:
    """Long document: the files joined into one prompt up to the context, asked twice (prefill, then the cached ask)."""
    files = req.files or [f["path"] for f in list_files()]
    parts, total = [], 0
    e = s.engines[rt]
    # reading a GGUF's header takes over a second, and the server would stand still for it: no Stop, no events
    tc = min(await asyncio.to_thread(train_ctx, e.gguf) or CTX, e.ctx if rt == "sllm" else s.cells_max)
    for f in files:
        t = read_file(f)
        if total + len(t) > 4 * tc:
            break
        parts.append(f"<file {f}>\n{t}\n</file>")
        total += len(t)
    q = req.message or _suggest("long")
    prompt = s.one_shot(SYSTEM_CHAT, "\n\n".join(parts) + f"\n\nQuestion: {q}")
    ids = await e.tokenize(prompt)
    if len(ids) + 264 > tc:  # past the trained context llama-server refuses the prompt
        ids = ids[: tc - 264]
    first = await e.generate(ids, 200, lane="interactive",
                             on_token=lambda t: live.BUS.publish({"type": "token", "run": run_id, "rt": rt, "text": t}))  # fmt: skip
    again = await e.generate(ids, 200, lane="interactive")  # same length as the first ask, to compare the two
    return {"prompt_tokens": len(ids), "files": len(parts), "ttft_ms": first.get("ttft_ms"),
            "prefill_tok_s": round(len(ids) / (first["ttft_ms"] / 1000), 1) if first.get("ttft_ms") else None,
            "decode_tok_s": first.get("decode_tok_s"), "cached_ttft_ms": again.get("ttft_ms"),
            "cached_tokens": again.get("cached_tokens"), "same_answer_twice": first["ids"] == again["ids"]}  # fmt: skip


# the scenarios by id (the document jobs, DOC_TASKS, all run as _run_doc_task): those that take the run's start time w0
_TIMED_RUNNERS = {"swarm": _run_swarm, "shared": _run_shared, "relay": _run_relay, "followup": _run_followup, "team": _run_team}
_RUNNERS = {"chat": _run_chat, "load": _run_load, "exact": _run_exact, "burst": _run_burst,
            "long": _run_long, "turns": _turns}
