"""Writes a document of one of the eight kinds from a World (sample_world.py): which sections it has, which table columns, how long
it is. The text prints the data the questions ask about, but never an answer that has to be computed (a total, a difference,
a weekday, a deadline). Returns the text and the facts the key needs (the document's rating for the hard classify test)."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Callable

from sample_extras import extras
from sample_words import Ctx
from sample_world import FEE_RATE, Item, World

# kind: item noun, table has a team column, status words (done, on track, delayed) or None, has dates and rooms, names the delayed
# item in words, lists who approved changes, the three ratings
KINDS = {
    "project plan": dict(noun="deliverable", team=True, status=None, sched=True, delayed=False, approvals=True, rating=("small", "medium", "large")),
    "meeting minutes": dict(noun="action", team=True, status=None, sched=True, delayed=True, approvals=True, rating=("routine", "notable", "major")),
    "invoice": dict(noun="line", team=False, status=("paid", "open", "overdue"), sched=False, delayed=False, approvals=False, rating=("paid", "overdue", "disputed")),
    "incident report": dict(noun="component", team=False, status=("fixed", "monitoring", "open"), sched=False, delayed=False, approvals=True, rating=("minor", "major", "critical")),
    "status report": dict(noun="work item", team=False, status=("done", "on track", "delayed"), sched=True, delayed=False, approvals=False, rating=("green", "amber", "red")),
    "event plan": dict(noun="task", team=True, status=None, sched=True, delayed=True, approvals=False, rating=("low", "standard", "premium")),
    "service agreement": dict(noun="service line", team=True, status=None, sched=True, delayed=True, approvals=True, rating=("bronze", "silver", "gold")),
    "test report": dict(noun="test", team=False, status=("passed", "queued", "failed"), sched=True, delayed=False, approvals=False, rating=("light", "normal", "heavy")),
}  # fmt: skip
SIZES = {"short": dict(extras=3, contacts=6, log=0, notes=False), "medium": dict(extras=4, contacts=20, log=20, notes=True),
         "long": dict(extras=5, contacts=60, log=230, notes=True)}  # fmt: skip
LONG_LOG_PER_ITEM = 4  # a long document has this many log rows fewer for every table row, so that documents of 16 and 24 items are about as long
INTRO = {
    "project plan": "{org} plans the {proj} project in this document: who leads which team, what each team delivers and when.",
    "meeting minutes": "Minutes of the {proj} steering meeting held by {org}. The notes record who attended, what was decided and what was agreed.",
    "invoice": "Invoice {inv} from {seller} to {org} for goods delivered to the {proj} project. The lines below make up the invoice.",
    "incident report": "Incident report {inv} for the {proj} project of {org}. It describes what failed, who was affected and what had changed.",
    "status report": "Weekly status report for the {proj} project of {org}: what is done, what is on track and what is late.",
    "event plan": "Event plan for the audit week of the {proj} project, organised by {org}: the tasks, the teams and the rooms.",
    "service agreement": "Summary of the service agreement between {org} and its suppliers for the {proj} project, in plain language.",
    "test report": "Test report for the {proj} project at {org}: what was tested and how each test ended.",
}


@dataclass
class Doc:
    kind: str
    size: str
    w: World
    items: list[Item]
    has: dict
    when: Callable[[datetime.date], str]
    text: str = ""
    rating: dict = field(default_factory=dict)  # the document's fine-grained label for the hard classify test: {label, options}
    extra: dict = field(default_factory=dict)  # facts of one kind: invoice number, incident number


def table(head: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def _band(value: int, t1: int, t2: int) -> int:
    return 0 if value <= t1 else 1 if value <= t2 else 2


def _thresholds(c: Ctx, value: int) -> tuple[int, int]:
    """Two limits, rounded to 50, that put `value` in a band picked at random."""
    k = c.r.randrange(3)
    lo, hi = {0: (1.3, 1.9), 1: (0.6, 1.3), 2: (0.3, 0.6)}[k]
    t1, t2 = (round(value * f / 50) * 50 for f in (lo, hi))
    return t1, max(t2, t1 + 50)


def _teams(w: World, c: Ctx, size: str, doc: Doc) -> str:
    if size != "long":
        return "\n".join(f"{t.name} is led by {t.lead}. Its members are {t.members[0]} and {t.members[1]}." for t in w.teams)
    paras = []
    for i, t in enumerate(w.teams):
        owned = ", ".join(x.code for x in doc.items if x.team == i) if doc.has["team"] else ""
        paras.append(f"{t.name} is led by {t.lead}. Its members are {t.members[0]} and {t.members[1]}. The team meets on weekday mornings in the "
                     f"{c.pick(['east', 'west', 'north'])} wing, desk block {c.n(3000, 8999)}." + (f" It owns {owned}." if owned else ""))
    return "\n\n".join(paras)


def _items(w: World, doc: Doc, spec: dict) -> str:
    head = ["Code", spec["noun"].capitalize()] + (["Team"] if spec["team"] else []) + ["Quantity", "Unit price (euros)", "Due"] + (["Status"] if spec["status"] else [])
    rows = [[i.code, i.name] + ([w.teams[i.team].name] if spec["team"] else []) + [i.qty, i.price, doc.when(i.due)]
            + ([spec["status"][i.status]] if spec["status"] else []) for i in doc.items]
    return table(head, rows)


def _delayed(doc: Doc, spec: dict) -> str:
    late = next(i for i in doc.items if i.status == 2)
    return f"Only one {spec['noun']} is behind schedule: {late.code}, the {late.name}."


def _schedule(w: World, doc: Doc) -> str:
    when = doc.when
    return (f"The project kicks off on {when(w.kickoff)}. The external audit takes place on {when(w.audit)}. The final report is due three weeks "
            f"after the audit. A review meeting is planned for {when(w.review_old)}.")


def _bookings(w: World, doc: Doc) -> str:
    return "Rooms are booked by day:\n\n" + table(["Date", "Room", "Booked by"], [[doc.when(d), f"{r} room", w.teams[t].name] for d, r, t in w.bookings])


def _changes(w: World, doc: Doc) -> str:
    a, b = w.cr["cause"], w.cr["other"]
    return (f"Change {a['code']} was approved by {a['approver']} on {doc.when(a['date'])}. "
            f"Change {b['code']} was approved by {b['approver']} on {doc.when(b['date'])}.")


def _email(w: World, doc: Doc) -> str:
    a, b = w.teams[0].lead, w.teams[2].lead
    return (f"From: {a}\nTo: {b}\nSubject: Review meeting\n\nHello {b.split()[0]}, the review meeting planned for {doc.when(w.review_old)} "
            f"cannot take place on that day. We will hold it on {doc.when(w.review_new)} instead. Regards, {a.split()[0]}")


def _special(kind: str, w: World, doc: Doc, c: Ctx) -> str:
    """The section only this kind has, and fills doc.extra."""
    when = doc.when
    if kind == "invoice":
        doc.extra["number"] = c.code("INV", 5)
        due, asof = c.date_between(w.kickoff, 30, 60), c.date(within=300)
        state = {"paid": f"The customer paid this invoice in full on {when(c.date_between(w.kickoff, 1, 29))}.",
                 "overdue": f"The invoice was due on {when(due)}. On {when(asof)} the payment had still not arrived.",
                 "disputed": f"The customer has disputed line {doc.items[-1].code} and has put the payment on hold."}[w.pay]
        return f"Issued by {c.company()}. Payment terms are thirty days.\n\n{state}"
    if kind == "incident report":
        doc.extra["number"] = c.code("INC")
        return (f"On {when(w.incident)} the service stopped answering for {c.n(12, 190)} minutes. {w.users:,} customers were affected. "
                f"The outage was caused by change {w.cr['cause']['code']}.")
    if kind == "meeting minutes":
        return f"{w.teams[0].lead} took the chair and {w.teams[1].lead} kept the minutes. {c.n(9, 24)} members came."
    return ""


def _rating(kind: str, w: World, doc: Doc, c: Ctx) -> str:
    spec = KINDS[kind]
    names = spec["rating"]
    if kind == "invoice":
        doc.rating = {"label": w.pay, "options": list(names)}
        return "Payment status: paid when the payment is recorded, overdue when the due date has passed without payment, disputed when the customer has put a line in dispute."
    if kind == "incident report":
        value, what = w.users, "customers affected"
    else:
        value, what = sum(w.cost(i) for i in doc.items), "total cost of the table (quantity times unit price, in euros)"
    t1, t2 = _thresholds(c, value)
    doc.rating = {"label": names[_band(value, t1, t2)], "options": list(names)}
    return (f"Rating by {what}: {names[0]} when it is at most {t1:,}; {names[1]} when it is above {t1:,} and at most {t2:,}; "
            f"{names[2]} when it is above {t2:,}.")


def _appendix(w: World, doc: Doc, c: Ctx, size: dict) -> str:
    parts = []
    people = [p for t in w.teams for p in [t.lead, *t.members]]
    if size["contacts"]:
        rows = [[p, w.teams[i % len(w.teams)].name, c.n(3000, 8999)] for i, p in enumerate(people[: size["contacts"]])]
        rows += [[c.person(), w.teams[j % len(w.teams)].name, c.n(3000, 8999)] for j in range(max(0, size["contacts"] - len(rows)))]
        parts.append("Appendix A: contact list\n\n" + table(["Person", "Team", "Desk"], rows))
    if size["log"]:
        n_rows = max(40, size["log"] - LONG_LOG_PER_ITEM * len(doc.items)) if size["log"] > 50 else size["log"]
        rows = [[doc.when(c.date(within=400)), c.person(), f"checked {doc.items[j % len(doc.items)].code}"] for j in range(n_rows)]
        parts.append("Appendix B: inspection log\n\n" + table(["Date", "Inspector", "Action"], rows))
    return "\n\n".join(parts)


def _notes(w: World, doc: Doc, c: Ctx) -> str:
    return "\n\n".join(f"Note on {i.code} ({i.name}): {c.company()} confirmed the order and packed it in {c.n(500, 2999)} cartons. "
                       f"The goods were checked on {doc.when(c.date(within=400))} by {c.person()}, who found nothing to report. They are stored in bay "
                       f"{c.n(3000, 8999)} of the warehouse and will be inspected again every {c.n(3, 99)} weeks; the supplier's reference for the "
                       f"order is {c.code('RF', 5)}." for i in doc.items)


def render(kind: str, size: str, w: World, items: list[Item], c: Ctx, case: bool = False) -> Doc:
    """The document: sections in an order where a later one may revise an earlier one."""
    spec, sz = KINDS[kind], SIZES[size]
    sched = spec["sched"]
    has = {"team": spec["team"], "status": bool(spec["status"]), "sched": sched, "email": sched, "bookings": sched,
           "delayed": spec["delayed"] and not case, "approvals": spec["approvals"] and (kind == "meeting minutes" or not case),
           "cause": kind == "incident report"}
    doc = Doc(kind, size, w, items, has, c.when)
    c.used |= w.used
    seller = c.company()
    special = _special(kind, w, doc, c)  # first: it fixes the invoice or incident number the title uses
    number = doc.extra.get("number", "")
    head = INTRO[kind].format(org=w.org, proj=w.proj, inv=number, seller=seller)
    title = {"invoice": f"Invoice {number}", "incident report": f"Incident report {number}"}.get(kind, f"{kind.capitalize()}: {w.proj}")
    out = [title, head, "Teams\n\n" + _teams(w, c, size, doc), f"{spec['noun'].capitalize()}s\n\n" + _items(w, doc, spec)]
    if sz["notes"]:
        out.append("Notes\n\n" + _notes(w, doc, c))
    if has["delayed"]:
        out.append(_delayed(doc, spec))
    if sched:
        out += ["Schedule\n\n" + _schedule(w, doc), "Bookings\n\n" + _bookings(w, doc)]
    out.append(f"Budget\n\nThe approved budget is {w.budget:,} euros ({w.usd:,} US dollars at the finance rate of {FEE_RATE}).")
    if special:
        out.append(special)
    if has["approvals"]:
        out.append("Changes\n\n" + _changes(w, doc))
    code, _, new = w.rev
    out.append(f"Revisions\n\nRevision: the due date of {code} was revised to {doc.when(new)}.")
    if has["email"]:
        out.append("Correspondence\n\n" + _email(w, doc))
    out.append(_rating(kind, w, doc, c))
    app = _appendix(w, doc, c, sz)
    if app:
        out.append(app)
    out.append(f"Erratum\n\nErratum to the table: the unit price of {w.err[0]} is {w.err[2]} euros, not {w.err[1]} euros.")
    if sz["extras"]:
        out.append(extras(c, kind, sz["extras"]))
    doc.text = "\n\n".join(out) + "\n"
    return doc
