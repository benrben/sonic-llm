#!/usr/bin/env python3
"""Write the sample library: 256 documents of ten kinds and their answer key (answers.json), from templates and word lists.

    python tools/make_samples.py [--out samples]

The output is the same on every run (a fixed seed for every document), uses only the standard library, and describes only
invented people, places and companies. 256 is the most agents the engine runs at once, so "all samples" fills it.
answers.json holds, per document: its kind (the label for classifying), its fields (for extraction) and questions with the
answers a grader accepts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_extras import extras  # noqa: E402
from sample_kinds import KINDS  # noqa: E402
from sample_words import Ctx  # noqa: E402

COUNT = 256
SEED = 20261113  # the date this library was made; any other value gives another, equally valid library
NAME = re.compile(r"^\d{3}-[a-z-]+\.md$")


def build(index: int) -> tuple[str, str, dict]:
    """(file name, text, key) of document `index` (0-based); kinds take turns, so any first few documents are mixed."""
    kind = list(KINDS)[index % len(KINDS)]
    c = Ctx(SEED * 1000 + index)
    text, fields, qa = KINDS[kind](c)
    body, _, last = text.rstrip("\n").rpartition("\n\n")  # the extra paragraphs go before the closing one
    text = f"{body}\n\n{extras(c, kind)}\n\n{last}\n"
    key = {"kind": kind, "fields": fields, "questions": [{"q": q, "accept": a} for q, a in qa]}
    return f"{index + 1:03d}-{kind.replace(' ', '-')}.md", text, key


def write(out: Path) -> int:
    """Replace the library in `out` (only files this script makes: NNN-kind.md and answers.json). Returns the bytes written."""
    out.mkdir(parents=True, exist_ok=True)
    for old in out.iterdir():
        if NAME.match(old.name) or old.name == "answers.json":
            old.unlink()
    docs, total = {}, 0
    for i in range(COUNT):
        name, text, key = build(i)
        (out / name).write_text(text, encoding="utf-8")
        docs[name] = key
        total += len(text.encode())
    # one line per document: a diff and a grep stay readable
    lines = [f"{json.dumps(n)}: {json.dumps(k, ensure_ascii=False, separators=(',', ':'))}" for n, k in docs.items()]
    answers = '{"labels": ' + json.dumps(list(KINDS)) + ', "documents": {\n' + ",\n".join(lines) + "\n}}\n"
    (out / "answers.json").write_text(answers, encoding="utf-8")
    return total + len(answers.encode())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "samples")
    args = ap.parse_args()
    print(f"wrote {COUNT} documents and answers.json to {args.out} ({write(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
