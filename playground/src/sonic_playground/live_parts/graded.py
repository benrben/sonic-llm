"""The graded versions of the tests: which questions each agent gets from the answer key of the sample documents, and how its
reply is graded (grading.py). A run is graded when its test has a right answer, nobody typed their own question or labels, and
every document it reads has a key; otherwise it runs as an ordinary run and says so."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from .. import grading
from ..documents import answer_key, key_of

GRADED = ("swarm", "team", "followup", "shared", "classify", "extract")
SYSTEM_FACT = ("You are one agent of a team. Read the document and answer only from it. Reply with just the fact asked for, "
               "in one short line, and nothing else.")


@dataclass(frozen=True)
class Item:
    """What one agent is asked about its document and what counts as right. `kind` is qa, label or fields."""
    doc: str
    question: str
    kind: str
    expected: object  # qa: the accepted forms, label: the label, fields: {name: true value}

    def shown(self) -> str:
        """The expected answer as a person reads it."""
        if self.kind == "qa":
            return self.expected[0]
        return self.expected if self.kind == "label" else json.dumps(self.expected, ensure_ascii=False)


def question_index(i: int, n: int, shift: int = 0) -> int:
    """The question of agent `i` among `n`: spread over agents, and not the same one for a kind that comes every tenth document."""
    return (i + i // 10 + shift) % n


def _qa(doc: str, key: dict, qi: int) -> Item:
    q = key["questions"][qi]
    return Item(doc, q["q"], "qa", q["accept"])


def plan(scenario: str, message: str, docs: list[str], shift: int = 0, agents: int | None = None) -> list[Item] | None:
    """One Item per agent, or None when the run is not graded. `docs` are the documents the agents read, in agent order
    (shared: the one document, asked `agents` times); `shift` picks the next question (a follow-up)."""
    keys = [key_of(d) for d in docs]
    if scenario not in GRADED or (message.strip() and scenario != "shared") or not docs or None in keys:
        return None
    if scenario == "shared":
        return [_qa(docs[0], keys[0], i % len(keys[0]["questions"])) for i in range(agents or 1)]
    if scenario == "classify":
        return [Item(d, "Which label fits this document best?", "label", k["kind"]) for d, k in zip(docs, keys)]
    if scenario == "extract":
        return [Item(d, "Give these fields: " + ", ".join(k["fields"]), "fields", k["fields"]) for d, k in zip(docs, keys)]
    return [_qa(d, k, question_index(i, len(k["questions"]), shift)) for i, (d, k) in enumerate(zip(docs, keys))]


def labels() -> list[str]:
    return answer_key().get("labels", [])


def grade(item: Item, reply: str) -> tuple[str, dict[str, bool] | None]:
    """(verdict, the fields that are right (extract only)) of one reply."""
    if item.kind == "label":
        return grading.grade_label(reply, labels(), item.expected), None
    if item.kind == "fields":
        return grading.grade_fields(reply, item.expected)
    return grading.grade_answer(reply, item.expected), None


class Grader:
    """Grades the replies of one engine's run as they arrive (`extra` goes to the agents' done events) and sums them up."""

    def __init__(self, items: list[Item], clean: Callable[[str], str]):
        self.items, self.clean, self.results = items, clean, {}

    def extra(self, r: dict) -> dict:
        """The grading fields of an agent's done event; `r` is its result with its index `i`."""
        item = self.items[r["i"]]
        verdict, checks = grade(item, self.clean(r["text"]))
        self.results[r["i"]] = verdict
        out = {"question": item.question, "expected": item.shown(), "verdict": verdict, "document": item.doc}
        return {**out, "checks": checks} if checks is not None else out

    def metrics(self) -> dict:
        """The run's score and, per agent, its question, expected answer and verdict."""
        counts = {v: sum(x == v for x in self.results.values()) for v in grading.VERDICTS}
        asked = len({it.question for it in self.items})
        note = {"graded_note": f"{len(self.items) - asked} of {len(self.items)} agents repeat a question: this document has {asked}."}
        return {**(note if asked < len(self.items) and self.items[0].kind == "qa" and len({it.doc for it in self.items}) == 1 else {}),
                "graded": True, "correct": counts[grading.CORRECT], "wrong": counts[grading.WRONG], "empty": counts[grading.EMPTY],
                "graded_agents": [{"document": it.doc, "question": it.question, "expected": it.shown(), "verdict": self.results.get(i)}
                                  for i, it in enumerate(self.items)]}  # fmt: skip


def not_graded(scenario: str, message: str) -> dict:
    """What an ungraded run says of itself, plainly."""
    if scenario in GRADED:
        why = "you wrote your own question or labels" if message.strip() else "a document has no answer key"
        return {"graded": False, "graded_note": f"Not graded: {why}."}
    return {"graded": False, "graded_note": "Not graded: this test has no single right answer."}
