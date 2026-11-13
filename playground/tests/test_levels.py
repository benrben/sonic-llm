"""Answer levels (levels.py, levels_answers.py): reading an answer, the level recipes against fake engines, the session's
framing, and the chat answer that runs a level."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from sonic_playground import levels, levels_answers as answers, live
from sonic_playground.live_parts.agents import _chat_answer
from sonic_playground.live_parts.scenarios import SYSTEM_CHAT, RunReq
from sonic_playground.live_parts.session import Session


def run(coro):
    return asyncio.run(coro)


# -- reading an answer -------------------------------------------------------------------------------------------------
def test_answer_key_takes_the_last_box_after_the_thinking():
    t = "<think>maybe \\boxed{7}, no wait</think>\nThe total is \\boxed{18}."
    assert answers.answer_key(t) == "box:18"
    assert answers.answer_key("so \\boxed{1} and then \\boxed{ 2 }") == "box:2"  # no thinking: the whole reply, the last box
    assert answers.answer_key("<think>x</think>\\boxed{\\frac{3}{4}}") == "box:\\frac{3}{4}"  # nested braces


def test_answer_key_canonicalises_the_box():
    keys = {answers.answer_key(x) for x in ("\\boxed{18}", "\\boxed{18.0}", "\\boxed{ 18 }", "\\boxed{\\$18}", "\\boxed{18.}")}
    assert len(keys) == 1
    assert answers.answer_key("\\boxed{1,200}") == answers.answer_key("\\boxed{1200}")
    assert answers.answer_key("\\boxed{\\text{Yes}}") == answers.answer_key("\\boxed{yes}")
    assert answers.answer_key("\\boxed{18}") != answers.answer_key("\\boxed{19}")


def test_answer_key_without_a_box_is_the_normalised_final_text():
    a = answers.answer_key("<think>hmm</think>  The   answer is: PARIS!! ")
    b = answers.answer_key("the answer is paris")
    assert a == b == "txt:the answer is paris"


def test_a_chat_answer_without_a_box_agrees_on_the_number_in_its_last_bold_part():
    a = "<think>x</think> The total is 205 minutes, so the train is moving for **184 minutes**."
    b = "<think>y</think> Subtracting **21 minutes** of stops, it moves for **184** minutes in all."
    assert answers.answer_key(a) == answers.answer_key(b) == answers.answer_key("so \\boxed{184}")
    assert answers.answer_key("<think>z</think> It is about trains.").startswith("txt:")


def test_answers_agree_compares_the_answer_keys():
    assert answers.answers_agree("\\boxed{18}", "6 * 3 = 18, so \\boxed{18.0}") == (True, "the two short answers agree (18.0)")
    assert answers.answers_agree("it is **18 km**", "a\nb\n\\boxed{18}")[0]  # a bold number is a key too
    assert answers.answers_agree("\\boxed{18}", "\\boxed{19}") == (False, "the two short answers do not agree: answering again step by step")
    assert answers.answers_agree("Paris.", "paris")[0]
    # no equal keys: the last number of each result
    assert answers.answers_agree("240 / 4 = 60 km/h", "Total 240 km in 4 h.\n**60 km/h**")[0]
    assert not answers.answers_agree("240 / 4 = 60 km/h", "Total 240 km.\n**64 km/h**")[0]
    assert answers.answers_agree("so 1,200 dollars", "a\n**$1200.00**")[0] and not answers.answers_agree("it is 14", "four")[0]
    # worded results: a short one that is part of the other, word for word
    assert answers.answers_agree("Canberra", "Sydney is the largest city.\nThe capital is **Canberra**.")[0]
    assert answers.answers_agree("The capital of Australia is Canberra.", "**Canberra**")[0]
    assert not answers.answers_agree("Sydney", "The capital is **Canberra**.")[0] and not answers.answers_agree("no", "**not known**")[0]
    assert not answers.answers_agree("one two three four five six seven", "x\none two three four five six seven eight")[0]
    assert answers.answer_core("a\n\nlast line\n") == "last line" and answers.answer_core("") == ""
    assert answers.answer_core("x\nFinal Answer: Canberra") == "Canberra" and answers.answer_core("The answer is 4") == "4"
    assert not answers.answers_agree("", "")[0] and not answers.answers_agree("  ", " ")[0]


# -- the level names, prompts and caps ------------------------------------------------------------------------------------
def test_level_to_framing_map():
    assert levels.framing_of("off", "off") == "off"
    assert levels.framing_of("off", "high") == "high"  # gpt-oss keeps its own levels
    for lv in ("sonic", "fast", "normal", "deep"):  # no answer level thinks
        assert levels.framing_of(lv, "high") == "off"
    with pytest.raises(ValueError):
        levels.framing_of("turbo")


def test_supported_needs_the_own_engine():
    assert levels.supported(SimpleNamespace(rt="sllm", kind="next"))
    assert not levels.supported(SimpleNamespace(rt="sllm", kind="llama"))
    assert not levels.supported(SimpleNamespace(rt="llama", kind="next"))


def test_run_request_levels():
    r = RunReq(scenario="chat")
    assert r.level is None and r.thinking == "off"
    assert levels.LEVELS == ("off", "sonic", "fast", "normal", "deep")
    assert RunReq(level="sonic").level == "sonic" and levels.norm_level(None) == "off"
    with pytest.raises(Exception):
        RunReq(level="turbo")


def test_the_recipes_are_the_measured_ones():
    assert levels.ONE == ("Be extremely brief: one short line with only the essential step, then the final answer. "
                          "If asked to write code, give only the complete code in one code block.")
    assert levels.TIGHT == "You are a helpful assistant. Be very brief: at most 3 short lines of key steps, then the final answer."
    assert levels.LONG == "You are a careful expert. Work step by step, check each step, then give the final answer."
    assert levels.CHAT == "You are a helpful assistant." and levels.PROMPTS["CHAT"] == levels.CHAT
    assert levels.CAPS == {"ONE": 1024, "TIGHT": 2048, "LONG": 6144, "CHAT": 4096}
    assert levels.STAGES == {"sonic": ("ONE",), "fast": ("CHAT",), "normal": ("ONE", "TIGHT", "LONG"), "deep": ("ONE", "TIGHT", "LONG")}
    assert levels.level_caps("sonic") == [1024] and levels.level_caps("fast") == [4096]
    assert levels.level_caps("normal") == [1024, 2048, 6144] and levels.level_caps("deep") == [1024, 2048, 6144]


def test_stage_system_keeps_the_callers_instructions_unless_it_is_the_default():
    default = "You are a helpful assistant. Answer clearly and briefly."
    assert levels.stage_system(levels.TIGHT, default, default) == levels.TIGHT
    assert levels.stage_system(levels.TIGHT, "  ", default) == levels.TIGHT
    assert levels.stage_system(levels.LONG, "You manage agents. Reply in JSON.", default) == f"{levels.LONG}\n\nYou manage agents. Reply in JSON."


# -- the session's side: framing -------------------------------------------------------------------------------------------
class _Sess:
    """Just what Session.apply_level touches."""

    def __init__(self, kind="next", fail_on=()):
        self.engines = {"sllm": SimpleNamespace(rt="sllm", kind=kind)}
        self.used, self.fail_on = [], fail_on
        self.thinking, self.level = "off", "off"

    def use_thinking(self, lv):
        if lv in self.fail_on:
            raise HTTPException(422, "no thinking")
        self.used.append(lv)
        self.thinking = lv


def test_apply_level_frames_every_answer_level_with_thinking_off():
    apply = Session.apply_level
    s = _Sess()
    for lv in ("sonic", "fast", "normal", "deep"):
        assert apply(s, RunReq(scenario="chat", level=lv)) is None
        assert s.used[-1] == "off" and s.level == lv
    assert apply(s, RunReq(scenario="chat")) is None and s.used[-1] == "off" and s.level == "off"
    assert apply(s, RunReq(scenario="chat", thinking="high")) is None and s.used[-1] == "high" and s.level == "off"
    assert apply(s, RunReq(scenario="chat", level="off", thinking="low")) is None and s.used[-1] == "low"


def test_apply_level_says_so_once_where_it_is_not_what_was_asked():
    apply = Session.apply_level
    s = _Sess(kind="llama")
    note = apply(s, RunReq(scenario="chat", level="deep"))
    assert "own engine" in note and "unmeasured" in note and s.used[-1] == "off" and s.level == "deep"
    s = _Sess()
    note = apply(s, RunReq(scenario="exact", level="deep"))  # raw-engine tests run plain
    assert "does not apply" in note and s.level == "off" and s.used[-1] == "off"
    with pytest.raises(HTTPException):  # a thinking level the model's template cannot give is refused
        apply(_Sess(fail_on=("high",)), RunReq(scenario="chat", thinking="high"))


# -- the answer levels -------------------------------------------------------------------------------------------------------
class FakeAnswerEngine:
    """Engine.generate-shaped: `script` is a list of answers in order; records every call."""

    def __init__(self, script, kind="next", rt="sllm"):
        self.script, self.calls, self.kind, self.rt = list(script), [], kind, rt

    async def generate(self, ids, n, *, lane, on_token=None, ignore_eos=False):
        text = self.script.pop(0)
        self.calls.append({"ids": ids, "n": n, "lane": lane, "streams": on_token is not None})
        if on_token:
            for w in text.split(" "):
                on_token(w + " ")
        return {"text": text, "ids": list(range(len(text.split()))), "n": len(text.split()), "prompt_tokens": len(ids),
                "ttft_ms": 1.0, "wall_ms": 2.0, "decode_tok_s": 9.0, "stop": "eos"}


def answer(e, level):
    seen, got, phases = [], [], []

    async def build(sp):
        seen.append(sp)
        return [ord(c) for c in sp[:5]]

    r = run(levels.answer_generate(e, build, level, cap=lambda plen, c: c - plen, lane="interactive", on_token=got.append,
                                   progress=lambda row, label, state, **k: phases.append(state)))
    return r, seen, got, phases


def test_sonic_and_fast_stream_one_answer_with_their_prompt():
    for lv, prompt, cap in (("sonic", levels.ONE, 1024), ("fast", levels.CHAT, 4096)):
        e = FakeAnswerEngine(["a b c"])
        r, seen, got, phases = answer(e, lv)
        assert seen == [prompt] and "".join(got) == "a b c " and e.calls[0]["streams"] and e.calls[0]["n"] == cap - 5
        assert r["level"] == lv and r["check"] is None and [s["prompt"] for s in r["stages"]] == [{"sonic": "ONE", "fast": "CHAT"}[lv]]
        assert r["stages"][0]["tokens"] == 3 and not phases
        assert r["text"] == "a b c" and r["ttft_ms"] >= 0 and r["wall_ms"] >= r["ttft_ms"]


def test_normal_shows_the_tight_answer_when_the_two_short_answers_agree():
    for lv in ("normal", "deep"):
        e = FakeAnswerEngine(["\\boxed{18}", "6 * 3 so \\boxed{18}"])
        r, seen, got, phases = answer(e, lv)
        assert seen == [levels.ONE, levels.TIGHT] and got == ["6 * 3 so \\boxed{18}"] and r["text"] == "6 * 3 so \\boxed{18}"
        assert not e.calls[0]["streams"] and not e.calls[1]["streams"]
        assert e.calls[0]["n"] == 1024 - 5 and e.calls[1]["n"] == 2048 - 5
        assert phases == ["writing two short answers, then checking that they agree"]
        assert r["check"] == "the two short answers agree (18)" and [s["prompt"] for s in r["stages"]] == ["ONE", "TIGHT"]


def test_normal_answers_again_with_long_when_the_two_short_answers_differ():
    for lv in ("normal", "deep"):
        e = FakeAnswerEngine(["\\boxed{18}", "\\boxed{19}", "step one step two \\boxed{19}"])
        r, seen, got, phases = answer(e, lv)
        assert seen == [levels.ONE, levels.TIGHT, levels.LONG]
        assert phases[1] == "the two short answers do not agree: answering again step by step"
        assert e.calls[2]["streams"] and e.calls[2]["n"] == 6144 - 5  # LONG's own cap
        assert "".join(got) == "step one step two \\boxed{19} "  # only the LONG answer reaches the page
        assert r["text"] == "step one step two \\boxed{19}" and [s["prompt"] for s in r["stages"]] == ["ONE", "TIGHT", "LONG"]
        assert r["check"].startswith("the two short answers do not agree") and r["wall_ms"] >= r["ttft_ms"] > 0


def test_an_empty_second_answer_keeps_the_tight_one():
    for lv in ("normal", "deep"):
        e = FakeAnswerEngine(["x", "w w w", "  "])
        r, seen, got, phases = answer(e, lv)
        assert r["text"] == "w w w" and "".join(got).strip() == "w w w" and "kept the short one" in r["check"]
        assert len(r["stages"]) == 3 and r["ttft_ms"] > 0


def test_other_engines_run_the_same_levels():
    for kind, rt in (("llama", "sllm"), ("base", "llama")):
        e = FakeAnswerEngine(["a b", "a b"], kind=kind, rt=rt)
        r, seen, *_ = answer(e, "normal")
        assert seen == [levels.ONE, levels.TIGHT] and r["check"].startswith("the two short answers agree")


def test_an_unknown_level_is_refused():
    with pytest.raises(ValueError):
        run(levels.answer_generate(FakeAnswerEngine([]), None, "off", cap=None, lane=None))


def test_chat_answer_runs_the_level_with_the_stage_prompts_and_remembers_the_shown_answer(monkeypatch):
    monkeypatch.setattr(live, "BUS", bus := live.Bus())
    events = []
    monkeypatch.setattr(bus, "publish", events.append)

    class E(FakeAnswerEngine):
        ctx = 100_000

        async def tokenize(self, text):
            self.last = text
            return [ord(c) for c in text]

    e = E(["\\boxed{5}", "\\boxed{7}", "long \\boxed{6}"])  # ONE and TIGHT differ: LONG answers
    s = SimpleNamespace(engines={"sllm": e}, cells_max=100_000, train=None, lim={"answer": 300}, level="normal", history={"sllm": []},
                        template=SimpleNamespace(eog=()), chat_prompt=lambda rt, system, message: f"[{system}]{message}",
                        one_shot=lambda system, message: f"[{system}]{message}")
    r = run(_chat_answer(s, "sllm", "r1", SYSTEM_CHAT, "2+3?", remember=True))
    assert r["text"] == "long \\boxed{6}" and r["level"] == "normal" and r["stages"][2]["prompt"] == "LONG"
    assert [c["n"] for c in e.calls] == [300, 300, 300]  # the run's answer limit is under all the stage caps
    assert s.history["sllm"] == [("2+3?", "long \\boxed{6}")]  # the shown answer only
    assert e.last.startswith(f"[{levels.LONG}]")  # the default chat system is replaced by the stage prompt
    assert any(ev["type"] == "phase" and "answering again" in ev["msg"] for ev in events)
    assert "".join(ev["text"] for ev in events if ev["type"] == "token") == "long \\boxed{6} "
    # a manager prompt keeps its instructions after the stage prompt; level off is the caller's own prompt and plain
    s.level, s.history["sllm"] = "sonic", []
    e.script = ["ok"]
    run(_chat_answer(s, "sllm", "r1", "Manage the team.", "go", remember=False))
    assert e.last == f"[{levels.ONE}\n\nManage the team.]go"
    s.level, e.script = "off", ["plain"]
    r = run(_chat_answer(s, "sllm", "r1", SYSTEM_CHAT, "go", remember=False))
    assert e.last == f"[{SYSTEM_CHAT}]go" and "level" not in r
