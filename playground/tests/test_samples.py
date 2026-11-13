"""The sample library and its answer key: the generator reproduces the committed files byte for byte, and every key matches
its document. The property tests read the documents back as text (tables, sentences) and recompute every answer from them, so
a question whose answer the generator got wrong would be caught here."""

import datetime
import json
import re
import sys

import pytest

from sonic_playground import documents, grading
from sonic_playground.settings import PLAYGROUND_DIR

sys.path.insert(0, str(PLAYGROUND_DIR / "tools"))
import make_samples  # noqa: E402
from sample_render import KINDS  # noqa: E402

SAMPLES = PLAYGROUND_DIR / "samples"


@pytest.fixture(scope="module")
def key() -> dict:
    return json.loads((SAMPLES / "answers.json").read_text(encoding="utf-8"))


def test_the_generator_reproduces_the_committed_samples_byte_for_byte(tmp_path):
    out = tmp_path / "out"
    make_samples.write(out)
    made = {p.name: p.read_bytes() for p in out.iterdir()}
    committed = {p.name: p.read_bytes() for p in SAMPLES.iterdir()}
    assert sorted(made) == sorted(committed)
    assert [n for n in made if made[n] != committed[n]] == []


def test_the_generator_only_replaces_its_own_files(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "notes.md").write_text("mine")
    (out / "oldname-x.md").write_text("old")  # not NNN-kind: not the generator's
    (out / "999-gone.md").write_text("old")  # a name like those it makes: replaced
    make_samples.write(out)
    assert (out / "notes.md").exists() and (out / "oldname-x.md").exists() and not (out / "999-gone.md").exists()


def test_the_library_is_256_documents_of_eight_kinds_in_three_sizes_under_4_mb(key):
    docs = sorted(SAMPLES.glob("*.md"))
    by_size = {"short": [], "medium": [], "long": []}
    for p in docs:
        by_size[key["documents"][p.name]["size"]].append(p.stat().st_size)
    assert len(docs) == 256 and {k: len(v) for k, v in by_size.items()} == {"short": 86, "medium": 85, "long": 85}
    assert all(2500 <= n <= 5500 for n in by_size["short"]) and all(6000 <= n <= 13000 for n in by_size["medium"])
    assert all(15000 <= n <= 30000 for n in by_size["long"]), (min(by_size["long"]), max(by_size["long"]))
    assert sum(p.stat().st_size for p in SAMPLES.iterdir()) < 4_000_000
    assert set(key["documents"]) == {p.name for p in docs} and set(key["labels"]) == set(KINDS) and len(key["labels"]) == 8
    assert {k["kind"] for k in key["documents"].values()} == set(key["labels"])
    assert len({p.read_text() for p in docs}) == 256  # every document differs


def test_every_document_has_about_four_medium_and_four_hard_questions_with_an_update_and_a_negative(key):
    for name, k in key["documents"].items():
        levels = [q["level"] for q in k["questions"]]
        assert set(levels) == {"medium", "hard"} and 3 <= levels.count("medium") <= 5 and 3 <= levels.count("hard") <= 5, name
        assert 7 <= len(levels) <= 9, name
        skills = {q["skill"] for q in k["questions"]}
        assert {"update", "negative"} <= skills, name
        assert len(k["fields"]) >= 5 and k["fine"]["label"] in k["fine"]["options"] and len(k["fine"]["options"]) == 3, name
    every = {q["skill"] for k in key["documents"].values() for q in k["questions"]}
    assert every >= {"lookup", "multi-hop", "arithmetic", "compare", "date-math", "update", "table", "negative"}
    assert {q["skill"] for c in key["cases"] for q in c["questions"]} == {"cross-document"}


# -- the answers, recomputed from the text of the documents --------------------------------------------------------------------
def iso(text: str) -> str:
    return grading.normalize(text)


def read_tables(text: str) -> list[list[dict]]:
    """Every markdown table of a document as rows of {header: cell}."""
    tables, block = [], []
    for ln in text.splitlines() + [""]:
        if ln.startswith("|"):
            block.append([c.strip() for c in ln.strip("|").split("|")])
        elif block:
            head, _, *rows = block
            tables.append([dict(zip(head, r)) for r in rows])
            block = []
    return tables


class Read:
    """What a document says, read back from its text."""

    def __init__(self, name: str):
        self.name, self.text = name, (SAMPLES / name).read_text(encoding="utf-8")
        tables = read_tables(self.text)
        self.rows = next(t for t in tables if "Code" in t[0] and "Quantity" in t[0])
        self.bookings = next((t for t in tables if "Booked by" in t[0]), [])
        err = re.search(r"unit price of (\S+) is (\d+) euros, not (\d+) euros", self.text)
        self.fixed, self.name_col = {err[1]: int(err[2])}, list(self.rows[0])[1]  # the erratum's price; the column of the item names
        self.teams = re.findall(r"^(\w+ team) is led by (.+?)\.", self.text, re.M)
        self.status = KINDS[json.loads((SAMPLES / "answers.json").read_text())["documents"][name]["kind"]]["status"]

    def price(self, code: str) -> int:
        row = next(r for r in self.rows if r["Code"] == code)
        return self.fixed.get(code, int(row["Unit price (euros)"]))

    def cost(self, code: str) -> int:
        return int(next(r for r in self.rows if r["Code"] == code)["Quantity"]) * self.price(code)

    def late(self) -> str:
        prose = re.search(r"behind schedule: (\S+),", self.text)
        return prose[1] if prose else next(r["Code"] for r in self.rows if r.get("Status") == self.status[2])

    def lead_of_late(self) -> str:
        team = next(r["Team"] for r in self.rows if r["Code"] == self.late())
        return dict(self.teams)[team]


def derive(r: Read, how: dict) -> str:
    """The answer of `how` (the key's recipe), computed from document `r`: a number, an ISO date or a text."""
    op = how["op"]
    if op in ("line", "price"):
        return str(r.cost(how["code"]) if op == "line" else r.price(how["code"]))
    if op == "sum_status":
        return str(sum(r.cost(x["Code"]) for x in r.rows if x["Status"] == r.status[how["status"]]))
    if op == "sum_team":
        return str(sum(r.cost(x["Code"]) for x in r.rows if x["Team"] == r.teams[how["team"]][0]))
    if op == "pct_more":
        a, b = r.price(how["a"]), r.price(how["b"])
        return f"{100 * (a - b) / b:.1f}"
    if op == "diff":
        return str(r.cost(how["a"]) - r.cost(how["b"]))
    if op in ("min", "max"):
        field = {"price": "Unit price (euros)", "due": "Due", "qty": "Quantity"}[how["field"]]
        rows = [x for x in r.rows if x["Code"] in how["codes"]]
        value = (lambda x: iso(x[field])) if field == "Due" else (lambda x: int(x[field]))
        return (min if op == "min" else max)(rows, key=value)[r.name_col]
    if op == "max_cost":
        return max((x["Code"] for x in r.rows), key=r.cost)
    if op == "days":
        return str((datetime.date.fromisoformat(how["b"]) - datetime.date.fromisoformat(how["a"])).days)
    if op == "plus_days":
        return (datetime.date.fromisoformat(how["a"]) + datetime.timedelta(days=how["days"])).isoformat()
    if op == "weekday":
        return datetime.date.fromisoformat(how["a"]).strftime("%A")
    if op == "usd":
        return str(int(re.search(r"\(([\d,]+) US dollars", r.text)[1].replace(",", "")))
    if op == "approver":
        return re.search(rf"Change {how['code']} was approved by (.+?) on", r.text)[1]
    if op == "lead_of_late":
        return r.lead_of_late()
    if op == "room":
        return next(x["Room"] for x in r.bookings if iso(x["Date"]) == how["a"]).removesuffix(" room")
    raise AssertionError(op)


def same(got: str, q: dict, r: Read) -> bool:
    """Whether the derived answer `got` is what the key accepts."""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", got) or q["type"] in ("percent", "weekday"):
        return got.lower() in [iso(a).lower() for a in q["accept"]] or q["type"] == "percent" and abs(float(got) - float(q["accept"][0])) < 0.051
    if q["type"] == "number" or q["type"] == "days":
        return any(grading.tokens(a)[0] == got for a in q["accept"])
    return got in q["accept"]


def test_every_computed_answer_follows_from_the_text_of_its_document(key):
    checked = 0
    for name, k in key["documents"].items():
        r = Read(name)
        for q in k["questions"]:
            how = q.get("how")
            if how and how["op"] != "text":
                assert same(derive(r, how), q, r), (name, q["q"], derive(r, how), q["accept"])
                checked += 1
            elif how:  # a value the text states: the answer is written in it, after the question's subject
                assert any(grading._has(grading.tokens(r.text), grading._form(a)) for a in q["accept"]), (name, q["q"])
    assert checked > 1200


def test_no_reject_value_is_an_accepted_value_and_each_trap_is_graded_with_its_reason(key):
    reasons = {"old": grading.TRAP_REASON["old"], "currency": grading.TRAP_REASON["currency"], "similar": grading.TRAP_REASON["similar"]}
    traps = 0
    docs = [q for k in key["documents"].values() for q in k["questions"]] + [q for c in key["cases"] for q in c["questions"]]
    for q in docs:
        assert grading.grade_question(q["accept"][0], q) == (grading.CORRECT, ""), q
        if q.get("reject"):
            assert not set(q["reject"]) & set(q["accept"]), q
            assert grading.grade_question(q["reject"][0], q) == (grading.WRONG, reasons[q["trap"]]), q
            traps += 1
    assert traps > 900


def test_a_not_stated_question_is_about_something_the_document_does_not_say(key):
    negatives = 0
    for name, k in key["documents"].items():
        text = (SAMPLES / name).read_text(encoding="utf-8").lower()
        for q in k["questions"]:
            if q["type"] == "nostated":
                assert q["accept"] == ["NOT STATED"] and q["subject"] not in text, (name, q["q"])
                negatives += 1
    assert negatives >= 256


def test_a_computed_answer_is_never_printed_in_the_document(key):
    for name, k in key["documents"].items():
        words = set(re.findall(r"\d+(?:\.\d+)?", (SAMPLES / name).read_text(encoding="utf-8").replace(",", "")))
        for q in k["questions"]:
            if q.get("how", {}).get("op") in make_samples.COMPUTED:
                assert re.search(r"\d+(?:\.\d+)?", q["accept"][0].replace(",", ""))[0] not in words, (name, q["q"])


def fact_in(text: str, fact: dict) -> bool:
    return any(all(w in ln for w in fact["all"]) for ln in text.splitlines())


def test_every_case_question_needs_two_documents_and_no_single_document_has_all_its_facts(key):
    assert len(key["cases"]) == 16 and {len(c["documents"]) for c in key["cases"]} == {4, 5, 6, 7, 8}
    for case in key["cases"]:
        texts = {n: (SAMPLES / n).read_text(encoding="utf-8") for n in case["documents"]}
        assert len(case["questions"]) >= 3 and set(case["documents"]) <= set(key["documents"])
        for q in case["questions"]:
            assert len({f["doc"] for f in q["facts"]}) >= 2, q["q"]
            assert all(fact_in(texts[f["doc"]], f) for f in q["facts"]), q["q"]  # each fact is where the key says
            for name, text in texts.items():  # and no document alone has them all
                assert not all(fact_in(text, f) for f in q["facts"]), (case["id"], q["q"], name)


def test_the_case_answers_follow_from_the_text_of_the_documents(key):
    for case in key["cases"]:
        reads = {n: Read(n) for n in case["documents"]}
        for q in case["questions"]:
            how = q["how"]
            if how["op"] == "sum_docs":
                got = str(sum(r.cost(x["Code"]) for n in how["docs"] for r in [reads[n]] for x in r.rows))
            elif how["op"] == "approver":
                got = derive(next(r for r in reads.values() if f"Change {how['code']} was approved" in r.text), how)
            elif how["op"] == "lead_of_late":
                plan = next(r for n, r in reads.items() if "project-plan" in n)
                status = next(r for n, r in reads.items() if "status-report" in n)
                team = next(x["Team"] for x in plan.rows if x["Code"] == status.late())
                got = dict(plan.teams)[team]
            else:
                got = derive(next(iter(reads.values())), how)
            assert same(got, q, next(iter(reads.values()))), (case["id"], q["q"], got, q["accept"])


def test_the_key_is_not_a_document_and_documents_know_their_key(cfg):
    cfg.samples_dir.mkdir(exist_ok=True)
    make_samples.write(cfg.samples_dir)
    paths = [f["path"] for f in documents.list_files()]
    assert len(paths) == 256 and "samples/answers.json" not in paths
    k = documents.key_of(paths[0])
    assert k["kind"] == "project plan" and documents.key_of("uploads/" + paths[0].split("/")[1]) is None
    assert all(f["keyed"] for f in documents.list_files()) and {f["size"] for f in documents.list_files()} == {"short", "medium", "long"}
    cases = documents.cases()
    assert len(cases) == 16 and documents.case_for(cases[3]["documents"][::-1])["id"] == cases[3]["id"]
    assert documents.case_for(cases[3]["documents"][:-1]) is None and documents.case_for(paths[:4]) is None
    (cfg.documents_dir).mkdir(parents=True)
    documents.save_upload("mine.md", b"my text")
    assert [f["keyed"] for f in documents.list_files()][-1] is False
