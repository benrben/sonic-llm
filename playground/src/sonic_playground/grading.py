"""Deterministic grading of short answers: no model is the judge, only rules.

An answer is right when the final answer text (never the thinking) holds one of the accepted forms as a whole sequence of
words and numbers, and does not hedge. Case, punctuation, quotes, thousands separators, leading articles and the common ways
of writing a date ("14 March 2027", "March 14th, 2027", "2027-03-14") do not matter; a longer number or word that merely
contains the right digits or letters does not count.

Verdicts are the strings in VERDICTS. What the rules cannot see: a right fact in a wrong place of the text still passes, and
a free-text answer (a name) is only checked for the name, so two names given together pass if one is right.

A question of the sample key (grade_question) also has a type that says how to read the answer: a plain value (text, number, date,
weekday), a duration in days ("17 days", "seventeen days" and "2 weeks and 3 days" are all 17 days), a percentage (within the
key's tolerance of the key, 0.5, because the question asks for about one decimal) or NOT STATED (the reply says so and gives no
value). A wrong answer comes with a reason: the trap it fell into (`reject`: the old value, the look-alike, the other currency),
wrong arithmetic, an invented answer, two answers, or none."""

from __future__ import annotations

import json
import re
import unicodedata

CORRECT, WRONG, EMPTY = "correct", "wrong", "no answer"
VERDICTS = (CORRECT, WRONG, EMPTY)

_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.S | re.I)
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "‑": "-",
                         " ": " ", " ": " "})  # fmt: skip
_MONTHS = {m: i for i, names in enumerate(
    ["jan january", "feb february", "mar march", "apr april", "may", "jun june", "jul july", "aug august", "sep sept september",
     "oct october", "nov november", "dec december"], 1) for m in names.split()}  # fmt: skip
_DMY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]{3,9})\.?,?\s+(\d{4})\b")
_MDY = re.compile(r"\b([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_TOKEN = re.compile(r"\d{4}-\d{2}-\d{2}|\d+(?:\.\d+)?|[a-z0-9]+(?:[-'][a-z0-9]+)*")
_DATE, _NUMBER, _CODE = (re.compile(p) for p in (r"\d{4}-\d{2}-\d{2}", r"\d+(?:\.\d+)?", r"[a-z]{1,5}-\d{2,}"))
_ZERO_DECIMALS = re.compile(r"(?<=\d)\.0+(?!\d)")
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_SPELLED = re.compile(rf"\b((?:{'|'.join(_TENS)})(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|{'|'.join(_ONES)})(?=\s+(?:days?|weeks?)\b)")
_WEEKS = re.compile(r"\b(\d+)\s+weeks?(?:\s*(?:,|and)?\s*(\d+)\s+days?)?")
_NOT_STATED = ["not", "stated"]
_DOUBT = re.compile(r"(?:not|n't) (?:mention|give|say|state|provide|specify|know)\w*|no information|unknown|cannot|can't|unclear")
_DAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
NO_ANSWER, INVENTED, TWO_ANSWERS, WRONG_ARITHMETIC, WRONG_DATE = ("no answer", "invented an answer", "gave two answers", "wrong arithmetic",
                                                                "wrong date math")
NOT_SAID = "did not say NOT STATED"
TRAP_REASON = {"old": "gave the old value", "currency": "gave the amount in the other currency", "similar": "gave the look-alike"}
_ARTICLES = {"a", "an", "the"}
_LINKS = {"or", "and", "to", "vs", "versus", "either", "maybe", "perhaps", "not", "than", "instead"}  # between two numbers named side by side


def final_answer(reply: str) -> str:
    """The reply without its thinking: what follows the last </think>, with closed <think> blocks and an unfinished one removed."""
    text = reply.rsplit("</think>", 1)[-1] if "</think>" in reply.lower() else reply
    text = _THINK_CLOSED.sub("", text)
    return re.sub(r"<think>.*\Z", "", text, flags=re.S | re.I).strip()


def _iso(y: str, month: str, d: str, whole: re.Match) -> str:
    m = _MONTHS.get(month)
    return f"{int(y):04d}-{m:02d}-{int(d):02d}" if m else whole.group(0)


def normalize(text: str) -> str:
    """Lower case, plain quotes, thousands separators gone, dates as ISO (2027-03-14)."""
    s = unicodedata.normalize("NFKC", text).translate(_QUOTES).lower()
    s = _ZERO_DECIMALS.sub("", _THOUSANDS.sub("", s))
    s = _DMY.sub(lambda m: _iso(m[3], m[2], m[1], m), s)
    return _MDY.sub(lambda m: _iso(m[3], m[1], m[2], m), s)


def tokens(text: str) -> list[str]:
    """The words, numbers, dates (ISO) and codes of a text, normalized."""
    return _TOKEN.findall(normalize(text))


def _form(text: str) -> list[str]:
    t = tokens(text)
    while len(t) > 1 and t[0] in _ARTICLES:
        t = t[1:]
    return t


def _has(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return n > 0 and any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def _numbers_named(toks: list[str], unit: str | None) -> set[str]:
    """The numbers an answer puts forward: those next to the unit of the question or joined to another by 'or', 'and', 'not'."""
    out = set()
    for i, t in enumerate(toks):
        if not _NUMBER.fullmatch(t):
            continue
        before, after = toks[i - 1] if i else "", toks[i + 1] if i + 1 < len(toks) else ""
        if after == unit or before in _LINKS or after in _LINKS:
            out.add(t)
    return out


def _hedged(toks: list[str], forms: list[list[str]]) -> bool:
    """Whether the answer names two different weekdays, dates, codes or numbers of the kind that is asked for."""
    first = forms[0]
    if first[0] in _DAYS:
        return len(_DAYS & set(toks)) > 1
    if any(_DATE.fullmatch(t) for t in first):
        return len({t for t in toks if _DATE.fullmatch(t)}) > 1
    if any(_CODE.fullmatch(t) for t in first):
        return len({t for t in toks if _CODE.fullmatch(t)}) > 1
    if _NUMBER.fullmatch(first[0]):
        unit = next((f[1] for f in forms if len(f) > 1 and _NUMBER.fullmatch(f[0])), None)
        return len(_numbers_named(toks, unit)) > 1
    return False


def grade_answer(reply: str, accept: list[str]) -> str:
    """CORRECT when the final answer holds an accepted form and names no second answer of the same kind; EMPTY when there is
    no final answer; WRONG otherwise."""
    toks = tokens(final_answer(reply))
    if not toks:
        return EMPTY
    forms = [f for f in map(_form, accept) if f]
    return CORRECT if any(_has(toks, f) for f in forms) and not _hedged(toks, forms) else WRONG


def _spell_durations(text: str) -> str:
    """"seventeen days" -> "17 days"; "2 weeks and 3 days" -> "17 days"; "three weeks" -> "21 days"."""
    def digits(m: re.Match) -> str:
        words = re.split(r"[- ]", m[1])
        return str(sum(_TENS.get(w, 0) or _ONES.index(w) for w in words))
    text = _SPELLED.sub(digits, normalize(text))
    return _WEEKS.sub(lambda m: f"{7 * int(m[1]) + int(m[2] or 0)} days", text)


def _percent(toks: list[str], q: dict) -> bool:
    key = float(q["accept"][0])
    return any(abs(float(t) - key) <= q["tol"] + 1e-9 for t in toks if _NUMBER.fullmatch(t)) and not _hedged(toks, [[q["accept"][0]]])


def _not_stated(toks: list[str], q: dict) -> tuple[str, str]:
    """NOT STATED questions: right when the reply says it and gives no value (no number, date or code, and no long explanation)."""
    at = next((i for i in range(len(toks) - 1) if toks[i:i + 2] == _NOT_STATED), None)
    if at is None:
        return WRONG, NOT_SAID if _DOUBT.search(" ".join(toks)) else INVENTED
    rest = toks[:at] + toks[at + 2:]
    gives = any(_NUMBER.fullmatch(t) or _DATE.fullmatch(t) or _CODE.fullmatch(t) for t in rest) or len(rest) > 15
    return (WRONG, INVENTED) if gives else (CORRECT, "")


def _why_wrong(toks: list[str], q: dict, hedged: bool) -> str:
    if hedged:
        return TWO_ANSWERS
    if any(_has(toks, f) for f in map(_form, q.get("reject", [])) if f):
        return TRAP_REASON.get(q.get("trap"), TRAP_REASON["similar"])
    if q.get("skill") == "arithmetic" and any(_NUMBER.fullmatch(t) for t in toks):
        return WRONG_ARITHMETIC
    return WRONG_DATE if q.get("skill") == "date-math" else "wrong answer"


def grade_question(reply: str, q: dict) -> tuple[str, str]:
    """(verdict, reason) of a reply to a question of the sample key: {accept, type, skill, reject, trap, tol}. The reason is "" when
    correct, "no answer" for an empty reply, else why it is wrong."""
    answer = final_answer(reply)
    kind = q.get("type", "text")
    toks = tokens(_spell_durations(answer) if kind == "days" else answer)
    if not toks:
        return EMPTY, NO_ANSWER
    if kind == "nostated":
        return _not_stated(toks, q)
    if kind == "percent":
        return (CORRECT, "") if _percent(toks, q) else (WRONG, WRONG_ARITHMETIC)
    forms = [f for f in map(_form, q["accept"]) if f]
    hedged = _hedged(toks, forms)
    if any(_has(toks, f) for f in forms) and not hedged:
        return CORRECT, ""
    return WRONG, _why_wrong(toks, q, hedged and any(_has(toks, f) for f in forms))


def grade_label(reply: str, labels: list[str], right: str) -> str:
    """CORRECT when the final answer names exactly one of `labels` and it is `right`."""
    toks = tokens(final_answer(reply))
    if not toks:
        return EMPTY
    named = [x for x in labels if _has(toks, tokens(x))]
    return CORRECT if named == [right] else WRONG


def json_object(reply: str) -> dict | None:
    """The JSON object a reply is, in a code fence or not. None when it is not one."""
    try:
        v = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", final_answer(reply)))
    except ValueError:  # a reply that is not JSON is graded as such: every field wrong
        return None
    return v if isinstance(v, dict) else None


def _key(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def grade_fields(reply: str, fields: dict[str, str]) -> tuple[str, dict[str, bool]]:
    """(verdict, {field: right}): every field of the JSON object in the reply is compared with its true value by the same rules as
    an answer. A reply that is not a JSON object fails all fields. The verdict is CORRECT only when all fields are right."""
    if not tokens(final_answer(reply)):
        return EMPTY, {k: False for k in fields}
    obj = json_object(reply)
    given = {_key(k): v for k, v in (obj or {}).items()}
    checks = {k: grade_answer(_text(given.get(_key(k))), [v]) == CORRECT for k, v in fields.items()}
    return (CORRECT if all(checks.values()) else WRONG), checks


def _text(v) -> str:
    return " ".join(map(_text, v)) if isinstance(v, list) else "" if v is None else str(v)
