"""The structured data behind a sample document (a "world": a project with teams, items, dates, a budget, corrections) and the
eight kinds of document that are written from it. Every answer in the key is computed from this data, never typed by hand.
Facts that must be found together are split over sections, and a later section may revise an earlier one: the old value is the
trap, the final value is the answer."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

from sample_words import Ctx

TEAM_NAMES = ["Harbour", "Orchard", "Beacon", "Quarry", "Meadow", "Foundry", "Lantern", "Granary", "Tannery", "Willow", "Anchor", "Bridge"]
ADJ = ["steel", "copper", "spare", "sealed", "compact", "heavy", "outdoor", "modular", "rapid", "quiet", "bonded", "coated"]
NOUN = ["cable drum", "sensor pack", "relay board", "pump unit", "valve set", "filter bank", "battery rack", "panel frame",
        "signal lamp", "bracket kit", "fan module", "gauge set", "gasket roll", "pipe clamp"]  # fmt: skip
ROOMS = ["Linden", "Alder", "Rowan", "Hazel", "Birch", "Cedar", "Elm", "Maple", "Aspen", "Larch"]
PROJECTS = ["Harbour Lights", "Quiet Engine", "Green Ledger", "Blue Pantry", "Open Gate", "Slow River", "Bright Loom", "Iron Orchard"]
FEE_RATE = "1.10"  # US dollars per euro, as the finance team states it


@dataclass
class Team:
    name: str
    lead: str
    members: list[str]


@dataclass
class Item:
    code: str
    name: str
    team: int
    qty: int
    price: int  # euros, as the table prints it (an erratum may correct it)
    due: datetime.date  # as the table prints it (a revision may move it)
    status: int  # 0 done, 1 on track, 2 delayed


@dataclass
class World:
    proj: str
    org: str
    teams: list[Team]
    items: list[Item]
    kickoff: datetime.date
    audit: datetime.date
    review_old: datetime.date
    review_new: datetime.date
    incident: datetime.date
    bookings: list[tuple[datetime.date, str, int]]  # date, room, booked by (team index)
    budget: int
    rev: tuple[str, datetime.date, datetime.date]  # item code, due date printed, due date after the revision
    err: tuple[str, int, int]  # item code, unit price printed, unit price after the erratum
    cr: dict  # cause / other: change request code, approver, approval date
    users: int
    pay: str  # paid, overdue, disputed
    used: set = field(default_factory=set)  # everything the document's random source must not hand out again

    @property
    def usd(self) -> int:
        return self.budget * 11 // 10

    @property
    def report_due(self) -> datetime.date:
        return self.audit + datetime.timedelta(days=21)

    @property
    def audit_room(self) -> str:
        return next(room for day, room, _ in self.bookings if day == self.audit)

    def price(self, item: Item) -> int:
        """The unit price after the erratum."""
        return self.err[2] if item.code == self.err[0] else item.price

    def due(self, item: Item) -> datetime.date:
        return self.rev[2] if item.code == self.rev[0] else item.due

    def cost(self, item: Item) -> int:
        return item.qty * self.price(item)

    def item(self, code: str) -> Item:
        return next(i for i in self.items if i.code == code)


def _transposed(code: str) -> str:
    head, tail = code[:-2], code[-2:]
    return head + tail[::-1]


def _people(c: Ctx, n_teams: int) -> list[Team]:
    names = c.r.sample(TEAM_NAMES, n_teams)
    teams = [Team(f"{n} team", c.person(), [c.person(), c.person()]) for n in names]
    teams[1].lead = c.person_like(teams[0].lead)  # two leads with the same surname
    return teams


def _items(c: Ctx, n_items: int, n_teams: int) -> list[Item]:
    prefix = c.pick(["KX", "RT", "VN", "PM"])
    names = c.r.sample([f"{a} {n}" for a in ADJ for n in NOUN], n_items)
    items = [Item(f"{prefix}-{c.n(1000, 9999)}", names[i], i % n_teams, c.n(3, 160), c.n(12, 480), c.date(within=150), c.r.randint(0, 1))
             for i in range(n_items)]  # fmt: skip
    twin = _transposed(items[0].code)
    while twin == items[0].code or twin in c.used:
        items[0].code = f"{prefix}-{c.n(1000, 9999)}"
        twin = _transposed(items[0].code)
    items[1].code = twin  # two codes that differ by two digits swapped
    c.used.add(twin)
    items[c.r.randrange(2, n_items)].status = 2  # the one delayed item
    return items


def make_world(c: Ctx, n_items: int, n_teams: int, n_book: int) -> World:
    """One project: its teams, items, dates, budget, a revision, an erratum and two change requests."""
    teams, items = _people(c, n_teams), _items(c, n_items, n_teams)
    kick = c.date(within=30)
    audit = c.date_between(kick, 20, 90)
    days = [c.date_between(kick, 1, 120) for _ in range(n_book - 1)]
    rooms = c.r.sample(ROOMS, n_book)
    book = [(audit, rooms[0], c.r.randrange(n_teams))] + [(d, r, c.r.randrange(n_teams)) for d, r in zip(days, rooms[1:])]
    book[-1] = (book[-1][0], rooms[0], book[-1][2])  # the audit's room is booked on another day too
    review = c.date_between(kick, 5, 40)
    plain = [i for i in items[2:] if i.status != 2]
    rev, err = plain[0], plain[1]
    cr = {k: {"code": f"CR-{c.n(1000, 9999)}", "approver": c.person(), "date": c.date_between(kick, 1, 60)} for k in ("cause", "other")}
    w = World(c.pick(PROJECTS), c.company(), teams, items, kick, audit, review, c.date_between(review, 8, 25),
              c.date_between(audit, 8, 40), sorted(book), c.n(120, 480) * 100,
              (rev.code, rev.due, c.date_between(rev.due, 7, 40)), (err.code, err.price, c.n(12, 480)),
              cr, c.n(80, 9000), c.pick(["paid", "overdue", "disputed"]), set())
    w.used = set(c.used)
    return w
