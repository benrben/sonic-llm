"""How a chat prompt is framed for a model: the system, user and reply markers of its own chat template, as text with special
tokens. The framing comes from the template stored in the model's GGUF, except for Gemma 3 and Qwen3 files, which use the
two hand-written framings below (thinking off keeps an empty think block in every reply so that the history only grows).

The end-of-turn token that stops a reply is not part of the history: the next turn starts by closing the previous one with
it, as the model's chat template does.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .gguf_meta import read_gguf_metadata

@dataclass(frozen=True)
class Template:
    """Chat framing as text with special tokens; the segments are tokenized with parse_special on, add_special off."""

    name: str
    bos: bool
    # segment 0: [BOS] + system_open + SYSTEM + system_close; segment 1: task_open + task + reply_open
    system_open: str
    system_close: str
    task_open: str
    reply_open: str
    # every later segment: reply_close + user_open + message + reply_open
    reply_close: str
    user_open: str
    eog: tuple[str, ...]  # the end-of-generation tokens, stripped from the stored reply

    def prefix_text(self, system: str) -> str:
        return self.system_open + system + self.system_close

    def task_text(self, message: str) -> str:
        return self.task_open + message + self.reply_open

    def turn_text(self, message: str) -> str:
        return self.reply_close + self.user_open + message + self.reply_open


GEMMA3 = Template(
    name="gemma3 (merged system into the first user turn, as the GGUF chat template does)",
    bos=True,
    system_open="<start_of_turn>user\n",
    system_close="\n\n",
    task_open="",
    reply_open="<end_of_turn>\n<start_of_turn>model\n",
    reply_close="<end_of_turn>\n",
    user_open="<start_of_turn>user\n",
    eog=("<end_of_turn>", "<eos>"),
)

QWEN3 = Template(
    name="qwen3 chatml, thinking off (empty think block in every reply, kept in history so the history is append-only)",
    bos=False,
    system_open="<|im_start|>system\n",
    system_close="<|im_end|>\n",
    task_open="<|im_start|>user\n",
    reply_open="<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    reply_close="<|im_end|>\n",
    user_open="<|im_start|>user\n",
    eog=("<|im_end|>", "<|endoftext|>"),
)


THINKING = ("off", "low", "medium", "high", "max")


def effort_of(thinking: str) -> str | None:
    """The `reasoning_effort` a chat template gets for a thinking level (None = thinking off). Templates know at most
    low / medium / high, so max is high; templates with only on / off (Qwen3, Gemma 4) read enable_thinking."""
    if thinking not in THINKING:
        raise ValueError(f"thinking must be one of {THINKING}")
    return None if thinking == "off" else "high" if thinking == "max" else thinking


# Where a reply's thinking ends: Qwen3-style </think>, and harmony's switch to the final channel (with its special tokens
# shown, or with them hidden as some detokenizers do: "...assistantfinal").
_THOUGHT_END = re.compile(r"</think>|(?:<\|end\|>)?(?:<\|start\|>)?assistant(?:<\|channel\|>)?final(?:<\|message\|>)?")
_THOUGHT_START = re.compile(r"^\s*(?:<think>|<\|channel\|>analysis<\|message\|>|analysis(?=\S))")


def split_thinking(text: str) -> tuple[str, str]:
    """(thinking, answer) of a reply; thinking is "" when the reply has none."""
    m = _THOUGHT_END.search(text)
    if not m or not (_THOUGHT_START.match(text) or m.group(0) == "</think>"):
        return "", text
    return _THOUGHT_START.sub("", text[: m.start()]).strip(), text[m.end():].lstrip()


def template_for(gguf: Path | str, thinking: str = "off") -> Template:
    """The framing for the model in file `gguf`. With thinking off, Gemma 3 and Qwen3 files (by file name) get the
    hand-written framing; any other model, and every framing with thinking on, is derived from the chat template stored
    in the GGUF."""
    path = Path(gguf)
    if thinking == "off":
        name = path.stem.lower()
        if name.startswith("gemma-3"):
            return GEMMA3
        if name.startswith("qwen3"):
            return QWEN3
    return template_from_gguf(path, thinking)


log = logging.getLogger("sonic.chat_template")

# Marker strings rendered through a model's own Jinja template; the text around them is the framing.
_S, _U, _A, _V = "\x01SYS\x01", "\x01USR\x01", "\x01ANS\x01", "\x01NXT\x01"


class TemplateError(ValueError):
    pass


def _render(src: str, messages: list[dict], specials: dict, effort: str | None = None) -> str:
    import datetime

    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise TemplateError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False: json.dumps(x, indent=indent, ensure_ascii=ensure_ascii)
    env.globals.update(raise_exception=raise_exception, strftime_now=lambda f: datetime.date(2026, 1, 1).strftime(f))
    return env.from_string(src).render(messages=messages, add_generation_prompt=True, enable_thinking=effort is not None,
                                       reasoning_effort=effort or "low", tools=None, **specials)  # fmt: skip


def _render_probe(src: str, specials: dict, merged: bool, effort: str | None) -> tuple[str, str] | None:
    """The template rendered for a one-turn and a two-turn conversation of marker texts; None when it cannot be rendered."""
    first = [{"role": "user", "content": _S + "\n\n" + _U}] if merged else [{"role": "system", "content": _S}, {"role": "user", "content": _U}]
    try:
        r1 = _render(src, first, specials, effort)
        r2 = _render(src, first + [{"role": "assistant", "content": _A}, {"role": "user", "content": _V}], specials, effort)
    except TemplateError:  # the template itself refuses this shape (raise_exception): routine, the caller tries the next one
        return None
    except Exception as exc:  # a jinja or render bug, not a template that cannot be split: still None, but visible
        log.warning("chat template cannot be split (merged=%s): %s: %s", merged, type(exc).__name__, exc)
        return None
    return r1, r2


def _renders_append_only(r1: str, r2: str) -> bool:
    """Both renders hold the markers in order, and the one-turn render is a prefix of the two-turn one up to the user text."""
    for r in (r1, r2):
        if not (0 <= r.find(_S) < r.find(_U)) or r.count(_S) != 1:
            return False
    return r1[: r1.index(_U)] == r2[: r2.index(_U)] and r2.index(_U) < r2.find(_A) < r2.find(_V)


def _split(src: str, specials: dict, merged: bool, effort: str | None = None) -> tuple[str, str, str, str] | None:
    """(system_open, system_close, reply_open, reply_close + user_open) when the template renders append-only, else None.
    `merged`: the system text goes into the first user message (templates without a system role). `effort`: thinking
    on at that reasoning effort (None = off)."""
    probe = _render_probe(src, specials, merged, effort)
    if probe is None:
        return None
    r1, r2 = probe
    if merged:
        r1, r2 = r1.replace(_S + "\n\n", _S + "\x02"), r2.replace(_S + "\n\n", _S + "\x02")
    if not _renders_append_only(r1, r2):
        return None
    sys_open, rest = r1.split(_S)
    sys_close, reply_open = rest.split(_U)
    if merged:
        sys_close = sys_close.replace("\x02", "\n\n", 1)
    hist_open = r2[r2.index(_U) + len(_U): r2.index(_A)]
    if effort is None and hist_open.startswith(reply_open) and len(hist_open) > len(reply_open):
        reply_open = hist_open  # history spells out a channel header (harmony's final channel): skip the reasoning channel
    turn = r2[r2.index(_A) + len(_A): r2.index(_V)]
    if r2[r2.index(_V) + len(_V):] != reply_open and not reply_open.startswith(r2[r2.index(_V) + len(_V):]):
        return None
    return sys_open, sys_close, reply_open, turn


def _vocab_token(meta: dict, tokens: list, key: str) -> str | None:
    i = meta.get(f"tokenizer.ggml.{key}")
    return tokens[i] if isinstance(i, int) and 0 <= i < len(tokens) else None


def _end_of_generation(meta: dict, tokens: list, eos_tok: str | None, turn: str) -> list[str]:
    """The special token that closes a reply in history, plus the vocabulary's eos/eot/eom."""
    eog = [t for t in (_vocab_token(meta, tokens, "eot_token_id"), _vocab_token(meta, tokens, "eom_token_id"), eos_tok) if t]
    types = meta.get("tokenizer.ggml.token_type") or []
    for i, t in enumerate(tokens):
        if i < len(types) and types[i] in (3, 4) and t and turn.startswith(t) and t not in eog:
            eog.insert(0, t)
            break
    return eog


def template_from_gguf(path: Path, thinking: str = "off") -> Template:
    meta = read_gguf_metadata(path)
    src = meta.get("tokenizer.chat_template")
    if not src:
        raise KeyError(f"{path.name} has no chat template")
    tokens = meta.get("tokenizer.ggml.tokens") or []

    bos_tok, eos_tok = _vocab_token(meta, tokens, "bos_token_id"), _vocab_token(meta, tokens, "eos_token_id")
    specials = {"bos_token": bos_tok or "", "eos_token": eos_tok or ""}
    effort = effort_of(thinking)
    parts = _split(src, specials, False, effort) or _split(src, specials, True, effort)
    if parts is None:
        raise KeyError(f"the chat template of {path.name} does not render append-only")
    sys_open, sys_close, reply_open, turn = parts
    bos = bool(bos_tok) and sys_open.startswith(bos_tok)
    if bos:
        sys_open = sys_open[len(bos_tok):]  # BOS goes in as its id (Template.bos), not as text
    eog = _end_of_generation(meta, tokens, eos_tok, turn)
    name = f"from the GGUF chat template of {path.name}" + (f", thinking {thinking}" if effort else "")
    return Template(name=name, bos=bos, system_open=sys_open, system_close=sys_close,
                    task_open="", reply_open=reply_open, reply_close=turn, user_open="", eog=tuple(dict.fromkeys(eog)))  # fmt: skip
