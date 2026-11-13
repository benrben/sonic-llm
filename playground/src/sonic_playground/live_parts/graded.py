"""The graded versions of the tests: which questions each agent gets from the answer key of the sample documents, and how its
reply is graded (grading.py). A run is graded when its test has a right answer, nobody typed their own question or labels, and
every document it reads has a key; otherwise it runs as an ordinary run and says so.

Every question of the key has a level (medium or hard) and a skill; the run's difficulty picks the level (mixed: both, taking
turns). The format rules of every graded question are in FORMAT, in this one place."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from .. import grading
from ..documents import answer_key, case_for, key_of
from .scenarios import QLINE

GRADED = ("swarm", "team", "followup", "shared", "classify", "extract", "long")
DIFFICULTIES = ("medium", "hard", "mixed")
FORMAT = ("Answer from the document only. Reply with just the value, on one line, and nothing else. Give numbers as digits, with the "
          "unit if the question names one. Give dates as written in the document or as YYYY-MM-DD. Give a duration as a number of "
          "days. If the document does not say, reply exactly: NOT STATED")
SYSTEM_FACT = "You are one agent of a team. Read the document. " + FORMAT
SYSTEM_TEAM_FACT = ("You are a research agent on a team. Read the document and answer each numbered question only from it. " + FORMAT +
                    " Reply with exactly one line per question: 'Q<n>: <value>'. If you cannot find it, reply 'Q<n>: NOT FOUND'.")
SYSTEM_MANAGER_FACT = ("You are the manager of a research team. Agents each read one document and reported what it says. Answer each "
                       "question using only their reports. " + FORMAT + ". Reply with exactly one line per question: 'Q<n>: <value>'.")
LONG_QUESTIONS = 6  # questions of a graded long document: an update and a negative first, then the difficulty's own


@dataclass(frozen=True)
class Item:
    """What one agent is asked about its document and what counts as right. `kind` is qa, label or fields."""
    doc: str
    question: str
    kind: str
    expected: object  # qa: the key's question {accept, type, ...}, label: the label, fields: {name: true value}
    level: str = "hard"
    skill: str = ""
    labels: tuple[str, ...] = ()  # label: the labels to choose from

    def shown(self) -> str:
        """The expected answer as a person reads it."""
        if self.kind == "qa":
            return self.expected["accept"][0]
        return self.expected if self.kind == "label" else json.dumps(self.expected, ensure_ascii=False)


def _qa(doc: str, q: dict, prefix: str = "") -> Item:
    return Item(doc, prefix + q["q"], "qa", q, q["level"], q["skill"])


def pick(key: dict, i: int, difficulty: str, follow: bool = False) -> dict:
    """The question agent `i` gets about a document: of the difficulty's level (mixed: medium and hard take turns, so each is half of
    the run), spread over the agents. A follow-up is hard, and never the question the agent had before."""
    qs = key["questions"]
    want = ("medium", "hard")[i % 2] if difficulty == "mixed" else difficulty
    pool = [q for q in qs if q["level"] == want] or qs
    first = pool[(i // 2 + i // 10) % len(pool)]
    if not follow:
        return first
    hard = [q for q in qs if q["level"] == "hard" and q is not first] or [q for q in qs if q is not first]
    return hard[(i // 2 + i // 10) % len(hard)]


def long_questions(key: dict, difficulty: str) -> list[dict]:
    """Several questions about one long document: its update and its negative, then the difficulty's own."""
    qs = key["questions"]
    first = [next(q for q in qs if q["skill"] == s) for s in ("update", "negative")]
    rest = [q for q in qs if q not in first and (difficulty == "mixed" or q["level"] == difficulty)]
    return (first + rest + [q for q in qs if q not in first + rest])[:LONG_QUESTIONS]


def _long_doc(docs: list[str]) -> str:
    return next((d for d in docs if key_of(d)["size"] == "long"), docs[0])


def _label_item(d: str, k: dict, difficulty: str) -> Item:
    if difficulty == "hard":  # fine-grained: the rating inside the kind, from the figures in the document
        return Item(d, "Which rating applies to this document?", "label", k["fine"]["label"], "hard", "classify", tuple(k["fine"]["options"]))
    return Item(d, "Which label fits this document best?", "label", k["kind"], "medium", "classify", tuple(labels()))


def plan(scenario: str, message: str, docs: list[str], shift: int = 0, agents: int | None = None, difficulty: str = "mixed") -> list[Item] | None:
    """One Item per agent (long: per question), or None when the run is not graded. `docs` are the documents the agents read, in
    agent order (shared: the one document, asked `agents` times); `shift` 1 is a follow-up, which asks a harder question."""
    keys = [key_of(d) for d in docs]
    if scenario not in GRADED or (message.strip() and scenario != "shared") or not docs or None in keys:
        return None
    if scenario == "long":
        d = _long_doc(docs)
        return [_qa(d, q, f"About {d.split('/')[-1]}: ") for q in long_questions(key_of(d), difficulty)]
    if scenario == "shared":
        pool = [q for q in keys[0]["questions"] if difficulty == "mixed" or q["level"] == difficulty] or keys[0]["questions"]
        return [_qa(docs[0], pool[i % len(pool)]) for i in range(agents or 1)]
    if scenario == "classify":
        return [_label_item(d, k, difficulty) for d, k in zip(docs, keys)]
    if scenario == "extract":
        return [Item(d, "Give these fields: " + ", ".join(k["fields"]), "fields", k["fields"], "hard", "extract") for d, k in zip(docs, keys)]
    return [_qa(d, pick(k, i, difficulty, bool(shift)), "") for i, (d, k) in enumerate(zip(docs, keys))]


def plan_case(docs: list[str]) -> list[Item] | None:
    """The questions of the case that `docs` make up as a whole (None when they are not exactly one case): they go to the manager."""
    case = case_for(docs)
    return [Item(case["id"], q["q"], "qa", q, q["level"], q["skill"]) for q in case["questions"]] if case else None


def labels() -> list[str]:
    return answer_key().get("labels", [])


def team_line(text: str, n: int) -> str:
    """The answer written after `Q<n>:` in a team reply ('' when there is no such line). A reply with no `Q<n>:` line at all is taken as
    the answer to question 1 as it stands."""
    lines = [QLINE.match(ln.strip()) for ln in text.splitlines()]
    for m in lines:
        if m and int(m.group(1)) == n:
            return m.group(2).strip(" *_")
    return text.strip() if n == 1 and not any(lines) else ""


def grade(item: Item, reply: str) -> tuple[str, str, dict[str, bool] | None]:
    """(verdict, reason, the fields that are right (extract only)) of one reply."""
    if item.kind == "label":
        verdict = grading.grade_label(reply, list(item.labels), item.expected)
        return verdict, "" if verdict == grading.CORRECT else grading.NO_ANSWER if verdict == grading.EMPTY else "named another label or two", None
    if item.kind == "fields":
        verdict, checks = grading.grade_fields(reply, item.expected)
        wrong = [k for k, ok in checks.items() if not ok]
        return verdict, ("" if not wrong else grading.NO_ANSWER if verdict == grading.EMPTY else f"{len(wrong)} of {len(checks)} fields wrong"), checks
    verdict, reason = grading.grade_question(reply, item.expected)
    return verdict, reason, None


def _tally(rows: list[dict], key: str) -> dict:
    out: dict[str, list[int]] = {}
    for r in rows:
        got = out.setdefault(r[key], [0, 0])
        got[0] += r["verdict"] == grading.CORRECT
        got[1] += 1
    return out


class Grader:
    """Grades the replies of one engine's run as they arrive (`extra` goes to the agents' done events) and sums them up. `case` are
    the questions of a whole case, answered by the manager (grade_case)."""

    def __init__(self, items: list[Item], clean: Callable[[str], str], case: list[Item] | None = None):
        self.items, self.clean, self.case = items, clean, case or []
        self.results: dict[int, tuple[str, str]] = {}
        self.fields: dict[int, dict[str, bool]] = {}
        self.case_results: list[tuple[str, str]] = []

    def extra(self, r: dict) -> dict:
        """The grading fields of an agent's done event; `r` is its result with its index `i`."""
        item = self.items[r["i"]]
        verdict, reason, checks = grade(item, self.clean(r["text"]))
        self.results[r["i"]] = verdict, reason
        out = {"question": item.question, "expected": item.shown(), "verdict": verdict, "document": item.doc, "level": item.level, "skill": item.skill,
               "reason": reason}
        if checks is not None:
            self.fields[r["i"]] = checks
        return {**out, "checks": checks} if checks is not None else out

    def grade_case(self, answer: str) -> list[dict]:
        """Grade the manager's final answer, one `Q<n>: value` line per case question. Returns the rows of the questions."""
        self.case_results = [grade(it, team_line(answer, n))[:2] for n, it in enumerate(self.case, 1)]
        return self._rows()[len(self.items):]

    def _rows(self) -> list[dict]:
        rows = [(it, self.results.get(i, (None, ""))) for i, it in enumerate(self.items)] + list(zip(self.case, self.case_results))
        return [{"document": it.doc, "question": it.question, "expected": it.shown(), "verdict": v[0], "reason": v[1], "level": it.level, "skill": it.skill}
                for it, v in rows]

    def metrics(self) -> dict:
        """The run's score and, per agent, its question, expected answer and verdict; by level and by skill; the reasons."""
        rows = self._rows()
        counts = {v: sum(r["verdict"] == v for r in rows) for v in grading.VERDICTS}
        asked = len({it.question for it in self.items})
        repeats = asked < len(self.items) and self.items[0].kind == "qa" and len({it.doc for it in self.items}) == 1
        extra = {"graded_note": f"{len(self.items) - asked} of {len(self.items)} agents repeat a question: this document has {asked}."} if repeats else {}
        if self.fields:
            extra |= {"fields_right": sum(sum(c.values()) for c in self.fields.values()), "fields_asked": sum(len(c) for c in self.fields.values())}
        reasons = {}
        for r in rows:
            if r["reason"]:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        return {**extra, "graded": True, "correct": counts[grading.CORRECT], "wrong": counts[grading.WRONG], "empty": counts[grading.EMPTY],
                "by_level": _tally(rows, "level"), "by_skill": _tally(rows, "skill"), "reasons": reasons, "case_asked": len(self.case),
                "case_correct": sum(v == grading.CORRECT for v, _ in self.case_results), "graded_agents": rows}  # fmt: skip


def not_graded(scenario: str, message: str) -> dict:
    """What an ungraded run says of itself, plainly."""
    if scenario in GRADED:
        why = "you wrote your own question or labels" if message.strip() else "a document has no answer key"
        return {"graded": False, "graded_note": f"Not graded: {why}."}
    return {"graded": False, "graded_note": "Not graded: this test has no single right answer."}
