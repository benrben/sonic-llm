"""sllm-server --engine next, the server features on a real model: grammars (against today's engine), prefix sharing (a live
agent's cells copied, a free slot's cells taken: the same tokens as without), compaction (the next turn equals a fresh
agent on the compacted window), rollover (a turn longer than the window), suspend / resume (from the SSD file: the next turn
equals an agent that never slept). usage: srv_feat.py [MODELFILE env]"""
import json
import os
import sys
import tempfile

sys.path.insert(0, "/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/rollover")
from common import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = MODELS + os.environ.get("MODELFILE", "Qwen3-4B-Q4_0.gguf")
BIN = os.environ.get("SLLM_BIN", "/Users/benreich/Projects/crazyidea/sllm-wt/shared-target/release/sllm-server")
DOC = json.load(open(f"{HERE}/{'g' if 'gemma' in MODEL else ''}docref.json"))["prompts"][0]
res = {}


def serve(args, tag):
    tmp = tempfile.mkdtemp()
    p = subprocess.Popen([BIN, "--model", MODEL, "--port", str(PORT), "--host", "127.0.0.1", "--prompt-cache", "off", "--agent-dir", tmp + "/a", *args],
                         stdout=open(f"/tmp/srv-feat-{tag}.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(600):
        try:
            get("/health")
            return p
        except Exception:
            time.sleep(0.2)
    raise SystemExit("server did not start")


def comp(ids, n, **kw):
    b = {"prompt": ids, "n_predict": n, "temperature": 0, "stream": False, "return_tokens": True}
    b.update(kw)
    return post("/completion", b)


def turn_ids(aid, ids, n):
    code, r = turn(aid, ids, n, return_tokens=True, ignore_eos=True)
    assert code == 200, (code, r)
    return r["tokens"]


GRAMMARS = ['root ::= ("yes" | "no")', 'root ::= "{\\"answer\\": " [0-9]+ "}"', 'root ::= [a-z]+ (" " [a-z]+)* "."']


def grammar_outputs():
    qs = [tok(t) for t in ["Is the sky blue? Answer:", "How many legs does a spider have? Reply in JSON:", "Describe the sea in a few words:"]]
    return [comp(q, 24, grammar=g)["content"] for q in qs for g in GRAMMARS]


CHAT = ("<start_of_turn>user\nSay hi.<end_of_turn>\n<start_of_turn>model\n" if "gemma" in MODEL
        else "<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def eos_outputs():
    """A reply that ends early, then the same with ignore_eos: (tokens without, tokens with)."""
    q = tok(CHAT)
    return comp(q, 48)["tokens"], comp(q, 48, ignore_eos=True)["tokens"]


# today's engine: the grammar reference
p = serve(["--ctx", "1024"], "today")
try:
    today = grammar_outputs()
    today_eos = eos_outputs()
finally:
    stop(p)

p = serve(["--engine", "next", "--ctx", "2048", "--kv-pages", "128"], "next")
try:
    new = grammar_outputs()
    res["grammar_same_as_today"] = f"{sum(a == b for a, b in zip(new, today))}/{len(today)}"
    res["grammar_samples"] = new[:3]
    # ignore_eos: an end-of-generation argmax is replaced by the best other token
    e1, e2 = eos_outputs()
    res["ignore_eos"] = {"reply_tokens": len(e1), "with_ignore_eos": len(e2), "eos_in_output": any(t in e2 for t in [1, 106, 151643, 151645]),
                         "same_as_today": [e1 == today_eos[0], e2 == today_eos[1]]}

    # prefix sharing, live: the second agent's system prompt is copied from the first's slot
    S = DOC[:700]
    a1 = newagent(system_ids=S)
    a2 = newagent(system_ids=S)
    m = get("/metrics")["prefix"]
    x = tok(" Summarize the text above in one sentence.")
    o1, o2 = turn_ids(a1, x, 32), turn_ids(a2, x, 32)
    res["live_share"] = {"attached_tokens": m["attached_tokens"], "same_tokens": o1 == o2}
    for a in (a1, a2):
        urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/agents/{a}", method="DELETE"))

    # prefix sharing, kept: a free slot holds the document; cold (cache_prompt false) and warm must give the same tokens
    qs = [DOC[:800] + tok(f" Question {i}: what is this text about? Answer:") for i in range(4)]
    cold = [comp(q, 24, cache_prompt=False) for q in qs]
    before = get("/metrics")["prefix"]["attached_tokens"]
    warm = [comp(q, 24) for q in qs]
    res["kept_share"] = {"same_tokens": f"{sum(a['tokens'] == b['tokens'] for a, b in zip(cold, warm))}/4",
                         "attached_tokens": get("/metrics")["prefix"]["attached_tokens"] - before,
                         "prompt_ms_cold": [round(r["timings"]["prompt_ms"]) for r in cold], "prompt_ms_warm": [round(r["timings"]["prompt_ms"]) for r in warm]}

    # compaction: cut to the system prompt + a summary + the tail; the next turn equals a fresh agent on that window
    S0 = tok("You are a helpful writer.")
    a = newagent(system_ids=S0)
    t1, t2 = tok(" Write about the moon."), tok(" Now about the sun.")
    g1 = turn_ids(a, t1, 80)
    g2 = turn_ids(a, t2, 80)
    ctx = S0 + t1 + g1 + t2 + g2
    summ = tok(" (Earlier: a text about the moon.)")
    tail_from = len(S0) + len(t1) + len(g1)
    rep = post(f"/v1/agents/{a}/compact", {"summary_ids": summ, "prefix_tokens": len(S0), "tail_from": tail_from})
    window = S0 + summ + ctx[tail_from:]
    y = tok(" And the stars?")
    oc = turn_ids(a, y, 32)
    f = newagent(system_ids=window)
    of = turn_ids(f, y, 32)
    res["compaction"] = {"facts_tokens": rep.get("facts_tokens"), "context_after": agent(a)["context_tokens"], "window": len(window) + 32,
                         "same_as_fresh": oc == of}
finally:
    stop(p)

# rollover: a turn longer than a 1024-cell window
p = serve(["--engine", "next", "--ctx", "1024", "--kv-pages", "32", "--rollover-tail", "256"], "roll")
try:
    a = newagent(system_ids=tok("You are a storyteller."))
    code, r = turn(a, tok(" Tell a very long story about a lighthouse keeper."), 1600, return_tokens=True, ignore_eos=True)
    res["rollover"] = {"status": code, "tokens": len(r.get("tokens") or []), "context_now": agent(a)["context_tokens"],
                       "rollovers": r.get("rollovers"), "reply_keys": sorted(k for k in r if "roll" in k or "compact" in k)}
finally:
    stop(p)

# suspend / resume through the SSD: 2 slots, the sleeping agent's slot is taken by another agent meanwhile
p = serve(["--engine", "next", "--ctx", "1024", "--kv-pages", "8"], "life")
try:
    S = DOC[:300]
    t1, x = tok(" What is this about?"), tok(" Say more.")
    b = newagent(system_ids=S)
    turn_ids(b, t1, 40)
    ref = turn_ids(b, x, 32)
    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/agents/{b}", method="DELETE"))
    a = newagent(system_ids=S)
    turn_ids(a, t1, 40)
    sus = post(f"/v1/agents/{a}/suspend", {"tier": "ssd"})
    time.sleep(1)
    c, d = newagent(system_ids=tok("Other agent one.")), newagent(system_ids=tok("Other agent two, a different prompt."))
    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/agents/{d}", method="DELETE"))
    wake = post(f"/v1/agents/{a}/wake", {})
    time.sleep(1)
    out = turn_ids(a, x, 32)
    m = get("/metrics")
    res["suspend_resume"] = {"suspend": sus, "wake": wake, "same_as_never_slept": out == ref, "lifecycle": m.get("lifecycle")}
finally:
    stop(p)
for k, v in res.items():
    print(k, json.dumps(v)[:600])
