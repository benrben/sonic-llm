"""Word lists and small helpers for the sample generator (make_samples.py). Everything here is invented: the names, places and
companies are made from syllables and common words, and mean no real person or business."""

from __future__ import annotations

import datetime
import random

FIRST = ["Marta", "Jonas", "Ilse", "Tomas", "Nadia", "Oskar", "Petra", "Rafael", "Selma", "Viktor", "Anouk", "Bruno", "Clara",
         "Dmitri", "Elena", "Felix", "Greta", "Hugo", "Irene", "Jakob", "Karin", "Leon", "Mira", "Nils", "Olga", "Pavel", "Rosa",
         "Sven", "Tilda", "Ulrich", "Vera", "Wim", "Yara", "Zoran", "Amara", "Bastian", "Celia", "Dario", "Esme", "Faruk"]  # fmt: skip
LAST = ["Okonkwo", "Lindqvist", "Marchetti", "Tavares", "Brandt", "Kowalczyk", "Haddad", "Ferreira", "Nakagawa", "Olsen", "Petrov",
        "Quinlan", "Rasmussen", "Salazar", "Thorne", "Ulbricht", "Varga", "Whitlock", "Yilmaz", "Zielinski", "Abernethy", "Castellan",
        "Dvorak", "Eriksen", "Fontaine", "Grasso", "Holloway", "Iversen", "Jankowski", "Kessler", "Laurent", "Moreau", "Novak",
        "Oyelaran", "Pellegrini", "Rossetti", "Sandoval", "Tennant", "Vasquez", "Wexler"]  # fmt: skip
PLACE_A = ["Brindle", "Marlow", "Oster", "Fenwick", "Calder", "Harrow", "Linden", "Thistle", "Ashby", "Corvin", "Dunmore", "Elmstead",
           "Galway", "Hollis", "Ivory", "Kestrel", "Larkin", "Maple", "Norwick", "Penrose", "Quarry", "Redmond", "Sable", "Tarn"]  # fmt: skip
PLACE_B = ["vale", "ford", "mouth", "bridge", "field", "haven", "cross", "wick", "stead", "ham", "port", "ridge", "burg", "holm"]
STREET = ["Mill", "Chapel", "Orchard", "Harbour", "Station", "Quay", "Willow", "Granary", "Tannery", "Beacon", "Lantern", "Foundry"]
COMPANY_A = ["Kestrel", "Copperleaf", "Northwind", "Lanternworks", "Bluepine", "Halcyon", "Ironbark", "Marigold", "Quillon", "Saltmarsh",
             "Tidewater", "Vantage", "Wrenfield", "Applegate", "Birchmoor", "Cinderlake"]  # fmt: skip
COMPANY_B = ["Supply", "Logistics", "Foods", "Systems", "Textiles", "Engineering", "Print", "Labs", "Dairy", "Studio", "Freight", "Tools"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]  # fmt: skip


class Ctx:
    """One document's random source. Every person, place, number, date and code it hands out is new for this document, so that
    a fact is stated once and a distractor can never be mistaken for it."""

    def __init__(self, seed: int):
        self.r = random.Random(seed)
        self.used: set = set()
        self.style = self.r.choice(["dmy", "mdy"])  # how dates are written in this document

    def pick(self, items):
        return self.r.choice(items)

    def _fresh(self, make):
        while True:
            v = make()
            if v not in self.used:
                self.used.add(v)
                return v

    def person(self) -> str:
        return self._fresh(lambda: f"{self.r.choice(FIRST)} {self.r.choice(LAST)}")

    def place(self) -> str:
        return self._fresh(lambda: self.r.choice(PLACE_A) + self.r.choice(PLACE_B))

    def street(self) -> str:
        return self._fresh(lambda: f"{self.r.choice(STREET)} Lane")

    def company(self) -> str:
        return self._fresh(lambda: f"{self.r.choice(COMPANY_A)} {self.r.choice(COMPANY_B)}")

    def n(self, lo: int, hi: int) -> int:
        return self._fresh(lambda: self.r.randint(lo, hi))

    def code(self, prefix: str, digits: int = 4) -> str:
        return self._fresh(lambda: f"{prefix}-{self.r.randint(10 ** (digits - 1), 10 ** digits - 1)}")

    def date(self, after: datetime.date | None = None, within: int = 120) -> datetime.date:
        base = after or datetime.date(2027, 1, 1)
        lo = 1 if after else 0
        return self._fresh(lambda: base + datetime.timedelta(days=self.r.randint(lo, within)))

    def date_between(self, after: datetime.date, lo: int, hi: int) -> datetime.date:
        """A date `lo` to `hi` days after `after`, new for this document."""
        return self._fresh(lambda: after + datetime.timedelta(days=self.r.randint(lo, hi)))

    def person_like(self, other: str) -> str:
        """A new person with the surname of `other` (two people who could be mixed up)."""
        return self._fresh(lambda: f"{self.r.choice(FIRST)} {other.split()[-1]}")

    def before(self, d: datetime.date, lo: int = 14, hi: int = 60) -> datetime.date:
        return self._fresh(lambda: d - datetime.timedelta(days=self.r.randint(lo, hi)))

    def when(self, d: datetime.date) -> str:
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}" if self.style == "dmy" else f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def weekday_forms(d: datetime.date) -> list[str]:
    return [WEEKDAYS[d.weekday()], WEEKDAYS[d.weekday()][:3]]


def date_forms(d: datetime.date) -> list[str]:
    """The ways a person writes a date, all accepted."""
    m = MONTHS[d.month - 1]
    return [f"{d.day} {m} {d.year}", f"{m} {d.day}, {d.year}", d.isoformat()]


def number_forms(n: int, unit: str = "") -> list[str]:
    """A count or amount: with its unit, bare, with a thousands comma and as a word when it is small."""
    forms = [f"{n:,} {unit}".strip(), f"{n:,}"] + ([f"{n} {unit}".strip(), str(n)] if n >= 1000 else [])
    if n < len(NUMBER_WORDS):
        forms.append(NUMBER_WORDS[n])
    return list(dict.fromkeys(forms))


def name_forms(full: str) -> list[str]:
    """A person: the full name, or the surname alone."""
    return [full, full.split()[-1]]
