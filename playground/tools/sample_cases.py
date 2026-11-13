"""Cases: sets of 4 to 8 documents that belong to one project (plan, minutes, invoices, incident, status), and the questions only
the whole set can answer. Each fact a question needs sits in a different document: the key lists them (`facts`: the document and the
words that must stand together in one line of it), and no single document holds them all."""

from __future__ import annotations

from sample_render import Doc
from sample_words import number_forms
from sample_world import Item

CASES = 16
BASE = ["project plan", "meeting minutes", "incident report", "status report"]
COMPOSITION = {4: BASE, 5: BASE + ["invoice"], 6: BASE + ["invoice", "invoice"], 7: BASE + ["invoice", "invoice", "meeting minutes"],
               8: BASE + ["invoice", "invoice", "meeting minutes", "status report"]}  # fmt: skip
KIND_ORDER = ["project plan", "meeting minutes", "invoice", "incident report", "status report", "event plan", "service agreement", "test report"]


def members(j: int) -> list[tuple[int, str]]:
    """(document index, kind) of case `j`: its documents come from the 16 documents j*16 .. j*16+15, where every kind comes twice."""
    seen, out = {}, []
    for kind in COMPOSITION[4 + j % 5]:
        nth = seen[kind] = seen.get(kind, -1) + 1
        out.append((16 * j + KIND_ORDER.index(kind) + 8 * nth, kind))
    return out


def items_of(kind: str, items: list[Item], rev: str, err: str, nth: int, invoices: int) -> list[Item]:
    """The items one document of a case lists: all of them, except an invoice, which lists its own lines (nobody pays a line twice)."""
    if kind != "invoice":
        return items
    other = [i.code for i in items if i.code not in (rev, err)]
    keep = {rev, err, *other[:4]} if invoices == 1 else {rev, *other[:4]} if nth == 0 else {err, *other[4:8]}
    return [i for i in items if i.code in keep]


def _fact(name: str, *words: str) -> dict:
    return {"doc": name, "all": list(words)}


def _q(q: str, accept: list[str], kind: str, how: dict, facts: list[dict], reject=()) -> dict:
    out = {"q": q, "accept": accept, "level": "hard", "skill": "cross-document", "type": kind, "how": how, "facts": facts}
    return {**out, "reject": list(reject), "trap": "similar"} if reject else out


def case_questions(docs: list[tuple[str, Doc]]) -> list[dict]:
    """The questions of a case from its documents [(file name, Doc)]."""
    by = {}
    for name, d in docs:
        by.setdefault(d.kind, []).append((name, d))
    w = docs[0][1].w
    out = []
    invoices = by.get("invoice", [])
    if len(invoices) > 1:
        total = sum(d.w.cost(i) for _, d in invoices for i in d.items)
        old = sum(i.qty * i.price for _, d in invoices for i in d.items)
        out.append(_q("What is the total amount of all the invoices together, with the corrected unit prices?", number_forms(total, "euros"), "number",
                      {"op": "sum_docs", "docs": [n for n, _ in invoices]}, [_fact(n, d.extra["number"]) for n, d in invoices],
                      number_forms(old, "euros") if old != total else ()))
    (mn, _), (inn, idoc) = by["meeting minutes"][0], by["incident report"][0]
    cr = w.cr
    out.append(_q("Who approved the change that caused the incident?", [cr["cause"]["approver"]], "text", {"op": "approver", "code": cr["cause"]["code"]},
                  [_fact(inn, cr["cause"]["code"], "caused"), _fact(mn, cr["cause"]["code"], cr["cause"]["approver"])], [cr["other"]["approver"]]))
    (pn, pd) = by["project plan"][0]
    out.append(_q("How many days after the audit did the incident happen?", number_forms((w.incident - w.audit).days, "days"), "days",
                  {"op": "days", "a": w.audit.isoformat(), "b": w.incident.isoformat()},
                  [_fact(pn, pd.when(w.audit), "audit"), _fact(inn, idoc.when(w.incident), "stopped")]))
    (sn, sd) = by["status report"][0]
    late = next(i for i in w.items if i.status == 2)
    team = w.teams[late.team]
    out.append(_q("Who leads the team that owns the item the status report shows as delayed?", [team.lead], "text", {"op": "lead_of_late", "code": late.code},
                  [_fact(sn, late.code, "delayed"), _fact(pn, late.code, team.name), _fact(pn, team.name, team.lead)],
                  [w.teams[1 - late.team].lead] if late.team < 2 else ()))
    return out
