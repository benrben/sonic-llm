#!/usr/bin/env python3
"""Write the sample library: 256 documents of eight kinds in three sizes and their answer key (answers.json), from structured data.

    python tools/make_samples.py [--out samples]

The output is the same on every run (a fixed seed for every document), uses only the standard library, and describes only
invented people, places and companies. 256 is the most agents the engine runs at once, so "all samples" fills it.
answers.json holds, per document: its kind (the label for classifying), its size, its fine-grained rating (the hard classify
test), its fields (for extraction) and its medium and hard questions with the answers a grader accepts and the traps it rejects;
and, under "cases", 16 sets of 4 to 8 documents with the questions only the whole set can answer. Every answer is computed from the
data the document was written from."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_cases import CASES, case_questions, items_of, members  # noqa: E402
from sample_questions import questions  # noqa: E402
from sample_render import KINDS, Doc, render  # noqa: E402
from sample_words import Ctx  # noqa: E402
from sample_world import make_world  # noqa: E402

COUNT = 256
SEED = 20261113  # the date this library was made; any other value gives another, equally valid library
NAME = re.compile(r"^\d{3}-[a-z-]+\.md$")
SIZES = ["short", "medium", "long"]
SHAPE = {"short": (6, 3, 3), "medium": (10, 4, 5), "long": (24, 7, 8)}  # items, teams, room bookings of a document written alone
CASE_ITEMS, CASE_TEAMS, CASE_ROOMS = 16, 5, 6
COMPUTED = {"line", "sum_status", "sum_team", "pct_more", "days", "diff", "sum_docs"}  # answers that must not be printed in the text


def _fields(d: Doc) -> dict:
    """What the extract test asks for: facts of the document, the last of them as revised."""
    w = d.w
    out = {"project": w.proj, "budget_eur": str(w.budget), "item_count": str(len(d.items)), f"lead_of_{w.teams[0].name.replace(' ', '_').lower()}": w.teams[0].lead}
    if d.has["sched"]:
        out |= {"audit_date": d.when(w.audit), "current_review_date": d.when(w.review_new)}
    else:
        out |= {f"current_due_date_{w.rev[0]}": d.when(w.rev[2]), f"current_unit_price_{w.err[0]}": str(w.err[2])}
    return out


def _valid(d: Doc, qs: list[dict]) -> bool:
    """A document is kept when no computed answer is printed in it, no trap is also an answer and no NOT STATED subject occurs."""
    low, words = d.text.lower(), set(re.findall(r"\d+(?:\.\d+)?", d.text.replace(",", "")))
    for q in qs:
        if q["type"] == "nostated" and q["subject"] in low:
            return False
        if set(q["accept"]) & set(q.get("reject", [])):
            return False
        if q.get("how", {}).get("op") in COMPUTED and re.search(r"\d+(?:\.\d+)?", q["accept"][0].replace(",", "")).group(0) in words:
            return False
    return len(qs) >= 7


def _entry(d: Doc, qs: list[dict]) -> dict:
    return {"kind": d.kind, "size": d.size, "fine": d.rating, "fields": _fields(d), "questions": qs}


def _name(index: int, kind: str) -> str:
    return f"{index + 1:03d}-{kind.replace(' ', '-')}.md"


def build(index: int, world=None, items=None) -> tuple[str, Doc, dict]:
    """(file name, document, key) of document `index` (0-based); a case document is written from its case's `world` and `items`.
    Kinds and sizes take turns, so any first few documents are mixed. A draw that fails the checks of _valid is drawn again."""
    kind, size = list(KINDS)[index % len(KINDS)], SIZES[index % 3]
    for attempt in range(50):
        c = Ctx(SEED * 1000 + index + attempt * 100_000)
        w = world or make_world(c, *SHAPE[size])
        d = render(kind, size, w, items or w.items, c, case=world is not None)
        qs = questions(d, c.r)
        if _valid(d, qs):
            return _name(index, kind), d, _entry(d, qs)
    raise RuntimeError(f"no valid draw for document {index}")


def build_all() -> tuple[dict, list[dict]]:
    """({file name: (document, key)}, cases): the cases are written first, then the other documents."""
    docs, cases = {}, []
    for j in range(CASES):
        c = Ctx(SEED * 1000 + 900 + j)
        w = make_world(c, CASE_ITEMS, CASE_TEAMS, CASE_ROOMS)
        mem = members(j)
        n_inv = sum(k == "invoice" for _, k in mem)
        nth, got = {}, []
        for index, kind in mem:
            nth[kind] = nth.get(kind, -1) + 1
            name, d, key = build(index, w, items_of(kind, w.items, w.rev[0], w.err[0], nth[kind], n_inv))
            docs[name] = (d, key)
            got.append((name, d))
        cases.append({"id": f"case-{j + 1:02d}", "name": f"Case {j + 1:02d}: {w.proj}", "documents": [n for n, _ in got], "questions": case_questions(got)})
    for index in range(COUNT):
        if _name(index, list(KINDS)[index % len(KINDS)]) not in docs:
            name, d, key = build(index)
            docs[name] = (d, key)
    return dict(sorted(docs.items())), cases


def write(out: Path) -> int:
    """Replace the library in `out` (only files this script makes: NNN-kind.md and answers.json). Returns the bytes written."""
    out.mkdir(parents=True, exist_ok=True)
    for old in out.iterdir():
        if NAME.match(old.name) or old.name == "answers.json":
            old.unlink()
    docs, cases = build_all()
    total = 0
    for name, (d, _) in docs.items():
        (out / name).write_text(d.text, encoding="utf-8")
        total += len(d.text.encode())
    dump = lambda v: json.dumps(v, ensure_ascii=False, separators=(",", ":"))  # noqa: E731
    # one line per document and per case: a diff and a grep stay readable
    lines = ",\n".join(f"{json.dumps(n)}: {dump(k)}" for n, (_, k) in docs.items())
    case_lines = ",\n".join(dump(c) for c in cases)
    answers = '{"labels": ' + json.dumps(list(KINDS)) + ', "documents": {\n' + lines + '\n},\n"cases": [\n' + case_lines + "\n]}\n"
    (out / "answers.json").write_text(answers, encoding="utf-8")
    return total + len(answers.encode())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "samples")
    args = ap.parse_args()
    print(f"wrote {COUNT} documents and answers.json to {args.out} ({write(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
