"""The questions of a sample document, computed from its World: about four medium and four hard, each with the answers a grader
accepts, the answers that are traps (`reject`: a superseded value, a look-alike, the same amount in another currency), a `skill`,
and a `how` that says how the answer follows from the data. No question has a hand-typed answer."""

from __future__ import annotations

import random

from sample_render import KINDS, Doc
from sample_words import date_forms, number_forms, weekday_forms
from sample_world import TEAM_NAMES, Item

ABSENT = [("What is the telephone number of {lead}?", "telephone"), ("What is the warranty period of the {name}?", "warranty"),
          ("Which insurer covers the {proj} project?", "insurer"), ("What is the postal address of {org}?", "postal")]  # fmt: skip


def _q(q: str, accept: list[str], level: str, skill: str, kind: str = "text", **more) -> dict:
    return {"q": q, "accept": accept, "level": level, "skill": skill, "type": kind, **{k: v for k, v in more.items() if v}}


def _plain(d: Doc) -> list[Item]:
    """Items no revision or erratum touches: the questions about comparison and arithmetic use them."""
    return [i for i in d.items if i.code not in (d.w.rev[0], d.w.err[0])]


def _euro(n: int) -> list[str]:
    return number_forms(n, "euros")


def _name(i: Item) -> list[str]:
    return [i.name, i.code]


# -- medium ---------------------------------------------------------------------------------------------------------------
def m_late_lead(d: Doc, r) -> dict | None:
    if not (d.has["team"] and d.has["delayed"]):
        return None
    late = next(i for i in d.items if i.status == 2)
    lead = d.w.teams[late.team].lead
    twin = [t.lead for k, t in enumerate(d.w.teams[:2]) if k != late.team] if late.team < 2 else []
    return _q("Who leads the team that owns the item that is behind schedule?", [lead], "medium", "multi-hop", reject=twin, trap="similar",
              how={"op": "lead_of_late", "code": late.code})


def m_cause_approver(d: Doc, r) -> dict | None:
    if not (d.has["cause"] and d.has["approvals"]):
        return None
    cr = d.w.cr
    return _q("Who approved the change that caused the incident?", [cr["cause"]["approver"]], "medium", "multi-hop",
              reject=[cr["other"]["approver"]], trap="similar", how={"op": "approver", "code": cr["cause"]["code"]})


def m_compare(d: Doc, r) -> dict:
    pool = r.sample(_plain(d), min(5, len(_plain(d))))
    field, word, pick = r.choice([("price", "cheapest per unit", min), ("due", "due latest", max), ("qty", "ordered in the largest quantity", max)])
    best = pick(pool, key=lambda i: getattr(i, field))
    names = ", ".join(i.name for i in pool)
    return _q(f"Of these, which is {word}: {names}?", _name(best), "medium", "compare",
              how={"op": pick.__name__, "field": field, "codes": [i.code for i in pool]})


def m_line(d: Doc, r) -> dict:
    i = r.choice(_plain(d))
    return _q(f"How much do all the units of the {i.name} in the table cost in total?", _euro(i.qty * i.price), "medium", "arithmetic",
              reject=[str(i.price)], trap="similar", how={"op": "line", "code": i.code}, kind="number")


def m_weekday(d: Doc, r) -> dict | None:
    return _q("On what day of the week is the audit?", weekday_forms(d.w.audit), "medium", "date-math", "weekday",
              how={"op": "weekday", "a": d.w.audit.isoformat()}) if d.has["sched"] else None


def m_absent_fact(d: Doc, r) -> dict | None:
    for text, subject in r.sample(ABSENT, len(ABSENT)):
        if subject not in d.text.lower():
            q = text.format(lead=d.w.teams[0].lead, name=d.items[0].name, proj=d.w.proj, org=d.w.org)
            return _q(q, ["NOT STATED"], "medium", "negative", "nostated", subject=subject)
    return None


def m_revised_due(d: Doc, r) -> dict | None:
    code, old, new = d.w.rev
    if d.w.item(code) not in d.items:
        return None
    return _q(f"What is the current due date of {code}?", date_forms(new), "medium", "update", "date", reject=date_forms(old), trap="old",
              how={"op": "text", "value": new.isoformat()})


def m_twin_code(d: Doc, r) -> dict | None:
    a, b = d.w.items[0], d.w.items[1]
    if a not in d.items or b not in d.items:
        return None
    return _q(f"What is the unit price in euros of {a.code}?", number_forms(a.price), "medium", "table", "number", reject=number_forms(b.price),
              trap="similar", how={"op": "price", "code": a.code})


def m_twin_lead(d: Doc, r) -> dict:
    a, b = d.w.teams[0], d.w.teams[1]
    return _q(f"Who leads the {a.name}?", [a.lead], "medium", "lookup", reject=[b.lead], trap="similar", how={"op": "text", "value": a.lead})


def m_currency(d: Doc, r) -> dict:
    if r.random() < 0.5:
        return _q("How many US dollars is the approved budget?", number_forms(d.w.usd, "dollars"), "medium", "lookup", "number",
                  reject=number_forms(d.w.budget), trap="currency", how={"op": "usd", "eur": d.w.budget})
    return _q("How many euros is the approved budget?", _euro(d.w.budget), "medium", "lookup", "number", reject=number_forms(d.w.usd),
              trap="currency", how={"op": "text", "value": str(d.w.budget)})


# -- hard -----------------------------------------------------------------------------------------------------------------
def h_group_sum(d: Doc, r) -> dict | None:
    w = d.w
    if d.has["status"]:
        words = KINDS[d.kind]["status"]
        groups = {s: [i for i in d.items if i.status == s] for s in (0, 1)}
        s = max(groups, key=lambda k: len(groups[k]))
        which, group, how = f"all items whose status is {words[s]}", groups[s], {"op": "sum_status", "status": s}
    else:
        t = r.randrange(len(w.teams))
        which, group, how = f"all items owned by the {w.teams[t].name}", [i for i in d.items if i.team == t], {"op": "sum_team", "team": t}
    if len(group) < 2:
        return None
    total, old = sum(w.cost(i) for i in group), sum(i.qty * i.price for i in group)
    return _q(f"What is the total cost of {which}?", _euro(total), "hard", "arithmetic", "number", reject=_euro(old) if old != total else [],
              trap="old", how=how)


def h_percent(d: Doc, r) -> dict | None:
    pool = _plain(d)
    r.shuffle(pool)
    pairs = [(a, b) for a in pool for b in pool if a.price >= 1.1 * b.price]
    if not pairs:
        return None
    a, b = pairs[0]
    more = round(100 * (a.price - b.price) / b.price, 1)
    return _q(f"By about what percentage is the unit price of the {a.name} higher than that of the {b.name}? One decimal is enough.", [f"{more:.1f}"],
              "hard", "arithmetic", "percent", tol=0.5, how={"op": "pct_more", "a": a.code, "b": b.code})


def h_erratum(d: Doc, r) -> dict | None:
    code, old, new = d.w.err
    i = d.w.item(code)
    if i not in d.items:
        return None
    return _q(f"How much do all the units of the {i.name} cost in total?", _euro(i.qty * new), "hard", "update", "number",
              reject=_euro(i.qty * old), trap="old", how={"op": "line", "code": code})


def h_days(d: Doc, r) -> dict | None:
    n = (d.w.audit - d.w.kickoff).days
    return _q("How many days are there between the kickoff and the audit?", number_forms(n, "days"), "hard", "date-math", "days",
              how={"op": "days", "a": d.w.kickoff.isoformat(), "b": d.w.audit.isoformat()}) if d.has["sched"] else None


def h_deadline(d: Doc, r) -> dict | None:
    return _q("On what date is the final report due?", date_forms(d.w.report_due), "hard", "date-math", "date", reject=date_forms(d.w.audit),
              trap="similar", how={"op": "plus_days", "a": d.w.audit.isoformat(), "days": 21}) if d.has["sched"] else None


def h_review(d: Doc, r) -> dict | None:
    w = d.w
    return _q("On what date will the review meeting be held?", date_forms(w.review_new), "hard", "update", "date", reject=date_forms(w.review_old),
              trap="old", how={"op": "text", "value": w.review_new.isoformat()}) if d.has["email"] else None


def h_room(d: Doc, r) -> dict | None:
    if not d.has["bookings"]:
        return None
    w = d.w
    other = [room for day, room, _ in w.bookings if day != w.audit and room != w.audit_room]
    return _q("Which room is booked on the day of the audit?", [w.audit_room, f"{w.audit_room} room"], "hard", "multi-hop",
              reject=other[:2], trap="similar", how={"op": "room", "a": w.audit.isoformat()})


def h_difference(d: Doc, r) -> dict:
    a, b = sorted(r.sample(_plain(d), 2), key=lambda i: -i.qty * i.price)
    return _q(f"How much more do all the units of the {a.name} cost than all the units of the {b.name}?", _euro(a.qty * a.price - b.qty * b.price),
              "hard", "arithmetic", "number", how={"op": "diff", "a": a.code, "b": b.code})


def h_highest(d: Doc, r) -> dict | None:
    w = d.w
    costs = sorted(d.items, key=w.cost, reverse=True)
    old = sorted(d.items, key=lambda i: i.qty * i.price, reverse=True)[0]
    if w.cost(costs[0]) == w.cost(costs[1]):
        return None
    return _q("Which item has the highest total cost (quantity times unit price)?", _name(costs[0]), "hard", "compare",
              reject=_name(old) if old is not costs[0] else [], trap="old", how={"op": "max_cost"})


def h_absent_entity(d: Doc, r) -> dict | None:
    names = [n for n in TEAM_NAMES if f"{n.lower()}" not in d.text.lower()]
    if not names:
        return None
    n = r.choice(names)
    return _q(f"Who leads the {n} team?", ["NOT STATED"], "hard", "negative", "nostated", subject=n.lower())


MEDIUM = [m_late_lead, m_cause_approver, m_compare, m_line, m_weekday, m_absent_fact, m_revised_due, m_twin_code, m_twin_lead, m_currency]
HARD = [h_group_sum, h_percent, h_erratum, h_days, h_deadline, h_review, h_room, h_difference, h_highest, h_absent_entity]


def _forced(pool: list[dict], skill: str, r: random.Random) -> dict | None:
    have = [q for q in pool if q["skill"] == skill]
    return r.choice(have) if have else None


def questions(d: Doc, r: random.Random) -> list[dict]:
    """About 4 medium and 4 hard questions; always one update (the old value is the trap) and one negative (NOT STATED)."""
    med = [q for q in (f(d, r) for f in MEDIUM) if q]
    hard = [q for q in (f(d, r) for f in HARD) if q]
    chosen = [q for q in (_forced(med + hard, "update", r), _forced(med + hard, "negative", r)) if q]
    for level, pool in (("medium", med), ("hard", hard)):
        r.shuffle(pool)
        have = [q for q in chosen if q["level"] == level]
        chosen += [q for q in pool if q not in chosen][: max(0, 4 - len(have))]
    return sorted(chosen, key=lambda q: (q["level"] != "medium", q["skill"]))
