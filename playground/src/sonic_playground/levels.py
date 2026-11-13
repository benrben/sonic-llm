"""Answer levels: what the level selector (off / sonic / fast / normal / deep) does to a chat answer. The chat template's
thinking is off in every level. The recipes were measured on Qwen3.6-35B-A3B; on other models they run the same way.

* off     the model answers as it is: the system prompt of the test, no limit but the context.
* sonic   one answer with the system prompt ONE ("one short line"), at most 1024 tokens, streamed.
* fast    one answer with CHAT ("You are a helpful assistant.": an ordinary chat answer), at most 4096 tokens, streamed.
* normal  two short answers, one with ONE and one with TIGHT, both silent. When their answer keys agree (`answer_key`: the last
          box, else the number of the last bold part, else the whole text) the TIGHT answer is shown whole; else the model
          answers again with LONG (step by step, at most 6144 tokens, streamed). An empty LONG answer keeps the TIGHT one.
          Without equal keys the last number of each answer's result decides, and for worded results a short one that is part
          of the other (`answers_agree`).
* deep    for now the same as normal; a long-work level (code, research) is being built to replace it.

The level's cap is an upper bound: the reply limit is the smaller of it, what the context leaves and the run's own answer
limit. Only the chat answer (`live_parts.agents._chat_answer`) runs a level; the agents of a test run plain. The pure pieces
(`stage_system`, `level_caps`, `answers_agree`) are tested without a server.
"""

from __future__ import annotations

import time

from .levels_answers import answers_agree

LEVELS = ("off", "sonic", "fast", "normal", "deep")

# the system prompts of the answer levels (measured as written) and the most tokens each may write. ONE carries a sentence for
# code: without it a third of the coding tasks got a one-line description and no code.
ONE = ("Be extremely brief: one short line with only the essential step, then the final answer. "
       "If asked to write code, give only the complete code in one code block.")
TIGHT = "You are a helpful assistant. Be very brief: at most 3 short lines of key steps, then the final answer."
LONG = "You are a careful expert. Work step by step, check each step, then give the final answer."
CHAT = "You are a helpful assistant."
PROMPTS = {"ONE": ONE, "TIGHT": TIGHT, "LONG": LONG, "CHAT": CHAT}
CAPS = {"ONE": 1024, "TIGHT": 2048, "LONG": 6144, "CHAT": 4096}
# the stages of each level: normal answers with ONE and with TIGHT, and adds LONG when the two do not agree
STAGES = {"sonic": ("ONE",), "fast": ("CHAT",), "normal": ("ONE", "TIGHT", "LONG"), "deep": ("ONE", "TIGHT", "LONG")}


def norm_level(level: str | None) -> str:
    """The level's name: None is "off"."""
    return level or "off"


def framing_of(level: str, thinking: str = "off") -> str:
    """The chat-template thinking level a run at `level` frames its prompts with. Level "off" keeps the `thinking` choice
    (gpt-oss low / medium / high / max); every answer level frames with thinking off."""
    level = norm_level(level)
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, not {level!r}")
    return thinking if level == "off" else "off"


def supported(e) -> bool:
    """Whether the engine `e` is Sonic LLM's own engine, the one the levels were measured on."""
    return e.rt == "sllm" and e.kind == "next"


def stage_system(stage_prompt: str, caller_system: str, default_system: str) -> str:
    """The system prompt of one stage: the stage's own prompt alone when the caller's is the default chat system (or empty),
    else the stage prompt, a blank line, then the caller's system (a manager prompt keeps its instructions)."""
    if not caller_system or not caller_system.strip() or caller_system == default_system:
        return stage_prompt
    return f"{stage_prompt}\n\n{caller_system}"


def level_caps(level: str) -> list[int]:
    """The reply caps of the stages a level may run."""
    return [CAPS[name] for name in STAGES[norm_level(level)]]


class _AnswerRun:
    """One answer_generate call: its clock, the stages run so far, and when the page was first sent text (first-word time)."""

    def __init__(self, e, build, level: str, cap, lane: str | None, on_token, progress):
        self.e, self.build, self.level, self.cap, self.lane, self.on_token = e, build, level, cap, lane, on_token
        self.say = progress or (lambda *a, **k: None)
        self.t0 = time.monotonic()
        self.first_shown_t: float | None = None
        self.stages: list[dict] = []

    def to_page(self, text: str) -> None:
        if self.first_shown_t is None and text:
            self.first_shown_t = time.monotonic()
        if self.on_token is not None:
            self.on_token(text)

    async def stage(self, name: str, *, stream: bool) -> dict:
        ids = await self.build(PROMPTS[name])
        ts = time.monotonic()
        r = await self.e.generate(ids, self.cap(len(ids), CAPS[name]), lane=self.lane, on_token=self.to_page if stream else None)
        self.stages.append({"prompt": name, "tokens": r.get("n"), "wall_ms": (time.monotonic() - ts) * 1000})
        return r

    def finish(self, r: dict, check: str | None) -> dict:
        r["wall_ms"] = (time.monotonic() - self.t0) * 1000
        if self.first_shown_t is not None:
            r["ttft_ms"] = (self.first_shown_t - self.t0) * 1000
        r.update({"level": self.level, "stages": self.stages, "check": check})
        return r

    async def long_or_keep(self, kept: dict, why: str) -> dict:
        """Answer again with LONG (streamed); an empty LONG answer keeps `kept`, which is then sent to the page whole."""
        self.say(0, "answer", why)
        second = await self.stage("LONG", stream=True)
        if not (second.get("text") or "").strip():
            self.to_page(kept["text"])
            return self.finish(kept, why + " (the second answer was empty: kept the short one)")
        return self.finish(second, why)


async def _two_short_answers(run: _AnswerRun) -> dict:
    """normal / deep: ONE and TIGHT silently; the TIGHT one is shown when the two agree, else the LONG one streams."""
    run.say(0, "answer", "writing two short answers, then checking that they agree")
    one = await run.stage("ONE", stream=False)
    tight = await run.stage("TIGHT", stream=False)
    same, why = answers_agree(one["text"], tight["text"])
    if same:
        run.to_page(tight["text"])
        return run.finish(tight, why)
    return await run.long_or_keep(tight, why)


async def answer_generate(e, build, level: str, *, cap, lane: str | None, on_token=None, progress=None) -> dict:
    """The chat answer at an answer level (sonic / fast / normal / deep; "off" is the caller's own plain generate).

    e         the Engine.
    build     async callable `build(system_prompt) -> prompt ids`: the caller's prompt for that stage's system prompt, fitted.
    cap       `cap(prompt_len, level_cap) -> int`: the reply limit of a stage (the caller applies context room and its own limit).
    on_token  called with text: sonic / fast stream every token; normal / deep send the short answer whole once it is kept
              (the LONG answer streams).
    progress  progress(row, label, state): phase texts of normal / deep.

    Returns the Engine.generate dict of the answer shown, with wall_ms and ttft_ms counted from the start of the level (ttft:
    the first token of the shown answer reaching on_token), plus level, stages ([{prompt, tokens, wall_ms}]) and check (None
    for sonic / fast, else what the check found)."""
    level = norm_level(level)
    if level not in STAGES:
        raise ValueError(f"answer level must be one of {tuple(STAGES)}, not {level!r}")
    run = _AnswerRun(e, build, level, cap, lane, on_token, progress)
    if level in ("normal", "deep"):
        return await _two_short_answers(run)
    return run.finish(await run.stage(STAGES[level][0], stream=True), None)
