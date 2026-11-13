"""Replay spec.rs's prompt lookup (propose/record, cooldowns, min_keep, start_at) over a fixed greedy output (spec_toks.py)
and count steps, drafts, kept tokens and time from a cost table (ms by rows: Gemma 3 4B's step_bench at 1k cells, one-row
kernels up to 3 rows, wide from 4, + 0.4 ms a step on the host). Drafting never changes the tokens, so one engine's text
can be replayed under any rule; two engines' texts that part at a near-tie keep different numbers of drafts.
usage: spec_replay.py TOKS.json ..."""
import json, sys

class St:
    def __init__(s): s.idx = {}; s.upto = 0; s.k_cur = 0; s.miss = 0; s.cooldown = 0; s.steps = 0; s.keep = 0.0; s.gated = 0
    def k_now(s, c): return c["k"] if s.k_cur == 0 else min(s.k_cur, c["k"])
    def propose(s, ctx, c):
        n = c["n"]; L = len(ctx)
        e = max(s.upto, n - 1)
        while e < L - 1:
            s.idx[tuple(ctx[e + 1 - n:e + 1])] = e; e += 1
        suf = tuple(ctx[L - n:]); found = s.idx.get(suf)
        s.idx[suf] = L - 1; s.upto = L
        if s.cooldown > 0:
            s.cooldown -= 1; return []
        if found is None: return []
        k = s.k_now(c); return ctx[found + 1:min(found + 1 + k, L)]
    def record(s, prop, acc, c):
        if prop == 0: return
        cur = s.k_now(c)
        if c["min_keep"] > 0:
            obs = acc / prop
            s.keep = obs if s.steps == 0 else 0.75 * s.keep + 0.25 * obs
            s.steps += 1
            if s.steps >= 3 and s.keep < c["min_keep"]:
                s.cooldown = max(s.cooldown, 16 << min(s.gated, 3)); s.gated += 1; s.keep = c["min_keep"]
        if acc == 0:
            s.miss += 1; s.k_cur = max(cur // 2, 1)
            if s.miss >= 3: s.cooldown = 1 << min(s.miss - 2, 5)
        else:
            s.miss = 0
            s.k_cur = min(cur + 1, c["k"]) if acc >= prop else (max(cur // 2, 1) if acc * 2 < prop else cur)

def run(prompt, out, c, cost):
    st = St(); i = 1; steps = drafts = drafted = kept = 0; t = 0.0
    while i < len(out):
        if c.get("start"): st.k_cur = st.k_cur or c["start"]
        d = st.propose(prompt + out[:i], c)[:max(0, len(out) - i - 1)] if c["k"] else []
        acc = 0
        while acc < len(d) and d[acc] == out[i + acc]: acc += 1
        st.record(len(d), acc, c)
        steps += 1; drafts += bool(d); drafted += len(d); kept += acc
        t += cost(len(d) + 1); i += acc + 1
    return dict(steps=steps, drafts=drafts, drafted=drafted, kept=kept, ms=round(t, 1), tok_s=round((len(out) - 1) / t * 1000, 1))

NARROW = {1: 10.76, 2: 11.96, 3: 12.93, 4: 17.34}
WIDE = {4: 15.09, 5: 15.42, 6: 15.73, 7: 16.15, 8: 16.42}
def ours(r): return (NARROW[r] if r <= 3 else WIDE[r]) + 0.4

if __name__ == "__main__":
    pols = {"ours now (k7 start2 keep.2)": dict(k=7, n=3, start=2, min_keep=0.2), "ggml's (k4 keep0)": dict(k=4, n=3, min_keep=0.0),
            "k7 start2 keep0": dict(k=7, n=3, start=2, min_keep=0.0), "k7 start4 keep.2": dict(k=7, n=3, start=4, min_keep=0.2),
            "k7 keep0 (start 7)": dict(k=7, n=3, min_keep=0.0), "off": dict(k=0, n=3, min_keep=0)}
    for f in sys.argv[1:]:
        d = json.load(open(f))
        for which in ("doc", "short"):
            print(f, which)
            for name, c in pols.items():
                print(f"   {name:30s}", run(d[which], d[which + "_out"], c, ours))
