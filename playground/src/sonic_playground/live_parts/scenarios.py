"""The tests of the live playground (live.py; the code calls them scenarios): their system prompts, built-in questions and
limits, the list the page shows (SCENARIOS), the run request (RunReq), and how the replies of a team and of the document jobs
are read."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from .config import MAX_AGENTS

if TYPE_CHECKING:
    from .session import Session


# -- scenarios --------------------------------------------------------------------------------------------------------
SYSTEM_CHAT = "You are a helpful assistant. Answer clearly and briefly."
SYSTEM_AGENT = "You are one agent of a team. Read the document you are given and answer only from it, in at most 5 short bullet points."
# relay ("Ralph" loop): every lap a fresh agent; the plan and the notes file are all it knows of the laps before
SYSTEM_RELAY = ("You are one worker in a relay. You start with an empty memory: everything you know about the earlier work is in "
                "the plan and the notes. Do only the one story you are given, from its document, then stop.")
RELAY_TRIES = 2  # a story whose output fails the check goes to a fresh agent once more
SYSTEM_TEAM = ("You are a research agent on a team. Read the document and answer each numbered question only from it. "
               "Reply with exactly one line per question: 'Q<n>: <one-sentence answer>' or 'Q<n>: NOT FOUND'.")
SYSTEM_MANAGER = ("You are the manager of a research team. Agents each read one document and reported what it says. Answer each "
                  "question in turn using only their reports; put the document names you used in [brackets]. If no agent "
                  "found an answer, say so.")
_QLINE = re.compile(r"^\W*Q\s*(\d+)\s*[:.)\-]\s*(.+)$", re.I)


def questions_of(message: str) -> list[str]:
    """The questions of a team run: one per line, or split after each '?' when the message is one line (8 at most)."""
    lines = [ln.strip(" -•\t") for ln in message.splitlines() if ln.strip(" -•\t")]
    if len(lines) == 1 and lines[0].count("?") > 1:
        lines = [q.strip() + "?" for q in lines[0].split("?") if q.strip()]
    return lines[:8]


def parse_report(text: str, n: int) -> dict[int, str]:
    """{question number: answer} from one agent's report; NOT FOUND answers and lines outside 1..n are left out."""
    out = {}
    for ln in text.splitlines():
        m = _QLINE.match(ln.strip())
        if m and 1 <= int(m.group(1)) <= n and "not found" not in m.group(2).lower():
            out.setdefault(int(m.group(1)), m.group(2).strip(" *_:"))
    return out


BURST_TOPICS = [
    "the ocean", "a lighthouse", "Mars", "a city at night", "a robot", "the first computer", "a forest", "a library",
    "a dragon", "the moon", "an old train", "a storm", "a chess game", "a bakery", "a volcano", "a spaceship",
]  # fmt: skip
BURST_OUT = 256


def burst_prompts(s: "Session", n: int) -> list[tuple[str, str]]:
    return [(f"agent {i + 1}", s.one_shot(SYSTEM_CHAT, f"Write a long, detailed story about {BURST_TOPICS[i % len(BURST_TOPICS)]} "
                                                       f"(version {i // len(BURST_TOPICS) + 1})."))
            for i in range(n)]  # fmt: skip


SHARED_QUESTIONS = [
    "List the decisions this document records.", "What numbers or measurements does it give?", "What is still open or blocked?",
    "Summarise it in three sentences.", "Which files, flags or commands does it name?", "What risks does it mention?",
    "Who decided what, and when?", "What would you do next, based on it?",
]  # fmt: skip

# Agents taking turns: N agents each talk about their own document over a few rounds, but only K write at once. The KV
# block holds K agents; after its turn an agent is parked on the SSD (sllm-server suspend) and its next turn restores it.
TURNS_ROUNDS = 3
TURNS_AT_ONCE = 8  # K: agents that write at once, so the KV block holds K contexts (Limits: "Agents at once")
TURNS_OUT = 60  # tokens an answer takes (about)
TURNS_FIRST = 900  # tokens at most: system prompt, document (about 400-800) and the first question
TURNS_FOLLOW = 48  # tokens a follow-up question takes, with the chat template's turn framing
TURNS_NEED = TURNS_FIRST + TURNS_ROUNDS * TURNS_OUT + (TURNS_ROUNDS - 1) * TURNS_FOLLOW + 16  # tokens one agent ends with
TURNS_DOC_CHARS = 2400  # about 600 tokens
TURNS_WATCH_S = 0.25  # how often the SSD use is read while the rounds run
TURNS_FLAGS = ["--pressure-relief", "on"]  # if a restore finds no room, the engine parks the coldest idle agent instead of refusing
SYSTEM_TURNS = "You keep notes on one document. Answer from the document in at most two short sentences."
TURNS_QUESTIONS = ["What is this document about?", "Which single number or name in it matters most?",
                   "Write one question a careful reader could still ask about it."]  # fmt: skip

# One agent per document does one job on it. The message says what the job needs (labels, fields, a length); a reply is
# data: a label is checked against the list, JSON is parsed, and nothing in a reply changes the run.
DOC_OUT_MIN = 256  # a reply is at most as long as what its agent read, and a tiny document still gets this much
DOC_THINK_ROOM = 4096  # more when the model thinks first: a short limit would cut the thinking
DOC_TASKS = {
    "classify": {"system": "You sort documents into labels. Reply with exactly one label from the list and nothing else.",
                 "ask": "Labels: {m}\n\nWhich one label fits this document best?", "out": 24},
    "extract": {"system": "You pull facts out of documents. Reply with one JSON object and nothing else. Use only what the "
                          "document says; a field it does not give is null.",
                "ask": "Give these fields: {m}"},
    "compress": {"system": "You shorten documents. Keep every name, number, date, flag and decision, drop everything else, and add nothing.",
                 "ask": "Rewrite this document {m}."},
    "summarize": {"system": "You summarize documents. Use only what the document says.", "ask": "Summarize this document {m}."},
}  # fmt: skip

LEVEL_FREE = ("exact", "burst", "turns", "long")  # tests of the raw engine: no answer levels

SCENARIOS = [
    {"id": "chat", "group": "Talk", "name": "Chat",
     "what": "One question, one answer: see how soon the first word comes and how fast the rest is written.",
     "suggest": ["Why is the sky blue? Answer in three sentences.", "Summarise the attached file in 5 bullets."]},
    {"id": "long", "group": "Talk", "name": "Long document",
     "what": "Reads a long text, answers a question about it, then answers again from memory without reading it twice.",
     "suggest": ["What are the three most important things in these documents?"]},
    {"id": "swarm", "group": "Many agents", "name": "Agent swarm",
     "what": "One agent per document reads it at the same time, then one answer merges what they found.",
     "suggest": ["What decisions are recorded, and why?", "Which numbers do the documents give?"], "agents": MAX_AGENTS},
    {"id": "team", "group": "Many agents", "name": "Research team",
     "what": "Ask several questions. An agent per document answers from its own text, then a manager merges the reports.",
     "suggest": ["What was decided, and when?\nWhich amounts of money are named?\nWhat is still open?",
                 "Which people are named?\nWhich dates are named?\nWhat should happen next?"], "agents": MAX_AGENTS},
    {"id": "followup", "group": "Many agents", "name": "Follow-up question",
     "what": "Ask the agents of your last swarm or team a new question: they still remember their documents.",
     "suggest": ["What numbers do the documents give?", "What is still open or unclear?"], "agents": MAX_AGENTS},
    {"id": "relay", "group": "Many agents", "name": "Relay",
     "what": "Fresh agents with an empty memory work through a plan, one small story each, lap after lap.",
     "suggest": ["Collect the key facts and numbers each document gives.", "Find what each document says is still open."], "agents": 4},
    {"id": "turns", "group": "Many agents", "name": "Agents taking turns",
     "what": "Many agents take turns, a few at a time. The waiting ones are parked on disk and woken for their turn. "
             "Sonic LLM only.",
     "suggest": ["(runs by itself)"], "agents": 16},
    {"id": "shared", "group": "Documents", "name": "Many questions, one document",
     "what": "Many agents ask different questions about the same long document, which is read only once.",
     "suggest": ["(questions are built in)"], "agents": 16},
    {"id": "classify", "group": "Documents", "name": "Classify",
     "what": "One agent per document sorts it under one of your labels.",
     "ask": "Labels, separated by commas…",
     "suggest": ["recipe, plan, guide, minutes, history, manual, other", "food, travel, money, home, community"], "agents": MAX_AGENTS},
    {"id": "extract", "group": "Documents", "name": "Extract",
     "what": "One agent per document pulls the fields you name into a JSON object.",
     "ask": "Fields to pull out, separated by commas…",
     "suggest": ["title, date, people, amounts, open items", "topic, steps, tools, warnings"], "agents": MAX_AGENTS},
    {"id": "summarize", "group": "Documents", "name": "Summarize",
     "what": "One agent per document writes its own short summary.",
     "ask": "How, for example: in 3 sentences…",
     "suggest": ["in 3 sentences", "in 5 bullet points"], "agents": MAX_AGENTS},
    {"id": "compress", "group": "Documents", "name": "Compress",
     "what": "One agent per document rewrites it shorter and keeps every name, number and date.",
     "ask": "How short, for example: to about a quarter of its length…",
     "suggest": ["to about a quarter of its length", "to at most 5 lines"], "agents": MAX_AGENTS},
    {"id": "load", "group": "Checks", "name": "Chat under load",
     "what": "A chat question alone, then again while many agents work. Does the chat stay fast?",
     "suggest": ["Explain how a bicycle stays upright, in three sentences.", "Write a haiku about the sea."], "agents": 8},
    {"id": "burst", "group": "Checks", "name": "Throughput burst",
     "what": "Many agents write a story at the same time: how many words per second does the machine write?",
     "suggest": ["(prompts are built in)"], "agents": 32},
    {"id": "exact", "group": "Checks", "name": "Exactness",
     "what": "The same question alone and next to other agents. Are the words identical?",
     "suggest": ["Explain how bread rises, in two sentences."], "agents": 8},
]  # fmt: skip


def doc_room(task: dict, thinking: str, prompt_tokens: int) -> int | None:
    """Longest reply of a document job. A model that never stops would otherwise write until its context is full: minutes,
    on one line of text. Thinking "max" stays without a limit, as it promises."""
    n = task.get("out") or max(DOC_OUT_MIN, prompt_tokens)
    return None if thinking == "max" else n if thinking == "off" else n + DOC_THINK_ROOM


def _suggest(scenario: str) -> str:
    """The first suggested message of a scenario (the default when the message is empty)."""
    return next(x for x in SCENARIOS if x["id"] == scenario)["suggest"][0]


class RunReq(BaseModel):
    scenario: str = "chat"
    message: str = ""
    files: list[str] = Field(default_factory=list)
    agents: int = Field(MAX_AGENTS, ge=1, le=MAX_AGENTS)
    # optional limits; None (the default) = the most the model and the memory allow
    ctx: int | None = Field(None, ge=512, description="sllm-server context of one agent or request")
    parallel: int | None = Field(None, ge=1, le=MAX_AGENTS, description="agents at once (sets the context per agent to cells / parallel)")
    answer_tokens: int | None = Field(None, ge=16, description="longest chat / manager answer")
    agent_tokens: int | None = Field(None, ge=16, description="longest answer of one agent")
    doc_chars: int | None = Field(None, ge=200, description="characters an agent reads of its document")
    thinking: Literal["off", "low", "medium", "high", "max"] = Field(
        "off", description="the model's thinking, through its own chat template: Qwen3 / Gemma 4 know on and off, gpt-oss "
        "low / medium / high; max = the highest level the model knows, answers uncapped. Used as it is with level off or no level.")
    level: Literal["off", "sonic", "fast", "normal", "deep"] | None = Field(
        None, description="answer level (levels.py), chat answer only, thinking off in all: off = as it is; sonic = one short line; "
        "fast = a regular chat answer (system prompt \"You are a helpful assistant.\", up to 4096 tokens); normal = two short "
        "answers, a step-by-step one when they do not agree; deep = for now the same as normal (a long-work level is being built). "
        "None = level off, `thinking` as sent.")


def pick_label(reply: str, labels: list[str]) -> str | None:
    """The label a reply names: all of it, or the one label in it. None when it is neither."""
    low = reply.strip().strip("*`'\".").lower()
    named = [x for x in labels if re.search(rf"(?<!\w){re.escape(x.lower())}(?!\w)", low)]
    return next((x for x in labels if x.lower() == low), named[0] if len(named) == 1 else None)


def json_of(reply: str) -> dict | None:
    """The JSON object a reply is, in a code fence or not. None when it is not one."""
    try:
        v = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip()))
    except ValueError:  # not JSON: the run reports the reply as "not valid JSON" (doc_verdict, doc_task_report)
        v = None
    return v if isinstance(v, dict) else None


def labels_of(message: str) -> list[str]:
    return [x.strip() for x in message.split(",") if x.strip()]


def doc_verdict(task: str, message: str, reply: str, r: dict) -> tuple[bool, str]:
    """One agent's job in a few words, and whether it came out as asked."""
    if task == "classify":
        label = pick_label(reply, labels_of(message))
        return label is not None, label or "not one of the labels"
    if r.get("stop") == "limit":
        return False, "cut at the length limit"
    if task == "extract":
        return (True, "valid JSON") if json_of(reply) is not None else (False, "not valid JSON")
    return True, f"{round(100 * r['n'] / max(1, r['prompt_tokens']))}% of what was read"


def doc_task_report(task: str, message: str, rs: list[dict], replies: list[str]) -> tuple[list[str], dict]:
    """What a document task did: the lines the page shows above the agents' replies, and the numbers of the run."""
    if task == "classify":
        return _classify_report(labels_of(message), replies)
    if task == "extract":
        ok = sum(json_of(t) is not None for t in replies)
        return [f"{ok} of {len(rs)} replies are valid JSON."], {"json_ok": ok}
    kept = round(100 * sum(r["n"] for r in rs) / max(1, sum(r["prompt_tokens"] for r in rs)), 1)
    return [f"The replies are {kept}% as long as what was read."], {"kept_pct": kept}


def _classify_report(labels: list[str], replies: list[str]) -> tuple[list[str], dict]:
    """A classify run: the files under each label, and how many replies named one."""
    picked = [pick_label(t, labels) for t in replies]
    counts = {x: picked.count(x) for x in labels if x in picked}
    return ["Files per label: " + (", ".join(f"{k} {v}" for k, v in counts.items()) or "none")], \
        {"labeled": sum(p is not None for p in picked), "labels": counts}  # fmt: skip
