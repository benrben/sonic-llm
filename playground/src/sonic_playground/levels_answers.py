"""Reading a chat answer: the answer key two answers must share to agree, the short form shown on the page, the box and
bold-part extraction, and the two-answer agreement check of the normal and deep levels. All pure: no engine, no clock."""

from __future__ import annotations

import contextlib
import re
from decimal import Decimal, InvalidOperation

from .chat_template import split_thinking


def _brace_arg(text: str, start: int) -> str | None:
    """The text inside the {...} that opens at text[start] == "{" (nested braces counted), None when it never closes."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    return None


def last_boxed(text: str) -> str | None:
    """The content of the last \\boxed{...} in `text` (None without one, or when its brace never closes)."""
    for m in reversed(list(re.finditer(r"\\boxed\s*\{", text))):
        arg = _brace_arg(text, m.end() - 1)
        if arg is not None:
            return arg
    return None


def final_part(text: str) -> str:
    """The answer of a reply: the thinking (up to </think>, or harmony's final channel) cut off."""
    return split_thinking(text)[1].strip()


def _canon_box(s: str) -> str:
    s = re.sub(r"\\text\{([^}]*)\}|\\mathrm\{([^}]*)\}", lambda m: m.group(1) or m.group(2), s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\[,;:! ]", "", s)
    s = s.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"\s+", "", s).lower().rstrip(".")
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):  # 18, 18.0 and 18.00 are one answer
        with contextlib.suppress(InvalidOperation):
            d = Decimal(s).normalize()
            s = format(d, "f")
    return s


def normalize_text(text: str) -> str:
    """Lower-case, punctuation stripped, whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def answer_key(text: str) -> str:
    """What two answers must share to agree: the content of the last \\boxed{...} of the part after the thinking (the whole
    reply when it has no thinking), else the normalized text of that part. Prefixed "box:" / "txt:" so a box never equals prose."""
    final = final_part(text)
    box = last_boxed(final)
    if box is not None:
        return "box:" + _canon_box(box)
    # a chat answer without a box: the number in its last bold part ("moving for **184 minutes**"), so prose that words
    # the same result differently still agrees
    for bold in reversed(re.findall(r"\*\*(.+?)\*\*", final)):
        num = re.search(r"-?\d[\d,]*(?:\.\d+)?", bold)
        if num:
            return "box:" + _canon_box(num.group(0).replace(",", ""))
    return "txt:" + normalize_text(final)


def answer_shown(text: str) -> str:
    """A short human form of the answer (the box content, else the first characters of the answer)."""
    final = final_part(text)
    box = last_boxed(final)
    return box.strip() if box is not None else re.sub(r"\s+", " ", final)[:80]


def answer_core(text: str) -> str:
    """The part of a chat answer that states its result: the content of the last box, else the last bold part, else the
    last line that has text."""
    final = final_part(text)
    box = last_boxed(final)
    if box is not None:
        return box
    bold = re.findall(r"\*\*(.+?)\*\*", final)
    if bold:
        return bold[-1]
    lines = [ln for ln in final.splitlines() if ln.strip()]
    # "Final Answer: Canberra" states Canberra
    return re.sub(r"^\s*(?:the\s+)?(?:final\s+)?answer(?:\s+is)?\s*[:\-]?\s*", "", lines[-1], flags=re.IGNORECASE) if lines else ""


AGREE_WORDS = 6  # a worded result of at most this many words may agree by being part of the other answer's result


def answers_agree(one: str, tight: str) -> tuple[bool, str]:
    """Normal and deep: whether the ONE answer and the TIGHT answer give the same result, and why (human text for the page). They agree
    when their answer keys are equal (`answer_key`); else when the last number of each one's core (`answer_core`) is the same
    number; else, with no number in either core, when the shorter core (1 .. AGREE_WORDS words) is, word for word, part of
    the other. A chat answer is free text, so a wrong "do not agree" only costs the LONG answer. An empty answer never agrees."""
    yes = True, f"the two short answers agree ({answer_shown(tight)})"
    no = False, "the two short answers do not agree: answering again step by step"
    k1, k2 = answer_key(one), answer_key(tight)
    if k1 == k2 and k1.split(":", 1)[1]:
        return yes
    c1, c2 = answer_core(one), answer_core(tight)
    n1, n2 = (re.findall(r"-?\d[\d,]*(?:\.\d+)?", c) for c in (c1, c2))
    if n1 or n2:
        return yes if _same_last_number(n1, n2) else no
    return yes if _short_result_inside_other(c1, c2) else no


def _same_last_number(n1: list[str], n2: list[str]) -> bool:
    """Both cores have a number and the last number of each is the same one (compared by `_canon_box`)."""
    return bool(n1 and n2 and _canon_box(n1[-1].rstrip(",")) == _canon_box(n2[-1].rstrip(",")))


def _short_result_inside_other(c1: str, c2: str) -> bool:
    """Worded results: the shorter core (1 .. AGREE_WORDS words) is, word for word, part of the other."""
    w1, w2 = normalize_text(c1).split(), normalize_text(c2).split()
    short, long_ = (w1, w2) if len(w1) <= len(w2) else (w2, w1)
    return 1 <= len(short) <= AGREE_WORDS and any(long_[i:i + len(short)] == short for i in range(len(long_) - len(short) + 1))
