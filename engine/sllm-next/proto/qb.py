"""qb: the quick bench over every model (qbench.m, one process a model, a few seconds each), against a saved reference
(bits and times) and stock ggml (llama-batched-bench b11669, measured once and cached).

  qb.py run  [-m qwen3,gemma3] [-e SN_X=1 ...] [--only w1,mix]   one pass: table, bits vs the reference, exactness
  qb.py ab   -e SN_X=1 [-m ...]                                  base / variant / variant / base: deltas, bits
  qb.py ref  [-m ...]                                            save this build's results as the reference
  qb.py ggml [-m ...]                                            measure stock ggml once (cached in ggml.json)

The kernels and the library come from the proto directory qb.py sits in (a worktree's own build). Data: QB_DATA
(default sllm-wt/qb-data). Every run holds the GPU lock."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GGUF = "/Users/benreich/Projects/crazyidea/models/gguf/"
DATA = os.environ.get("QB_DATA", "/Users/benreich/Projects/crazyidea/sllm-wt/qb-data")
LOCK = "/Users/benreich/Projects/crazyidea/sllm/engine/scripts/with_gpu_lock.py"
GGML_BENCH = "/Users/benreich/Projects/crazyidea/research/inference_runtime/vendor/llama-bin/llama-batched-bench"
MODELS = {  # key: (GGUF file, prompt tokens file in proto/)
    "qwen3": ("Qwen3-4B-Q4_0.gguf", "docref.json"),
    "gemma3": ("gemma-3-4b-it-qat-Q4_0.gguf", "gdocref.json"),
    "e2b": ("gemma-4-E2B-it-Q4_0.gguf", "gdocref.json"),
    "phi": ("Phi-3.5-mini-instruct-Q4_0.gguf", "pref.json"),
    "llama": ("Llama-3.1-8B-Instruct-Q4_0.gguf", "lref.json"),
    "mistral": ("Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", "mref.json"),
    "q35": ("Qwen3.5-4B-Q4_0.gguf", "docref.json"),
    "oss": ("gpt-oss-20b-MXFP4.gguf", "ossref.json"),
    "q36": ("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", "q36ref.json"),
}
SCEN = ["read", "w1", "w5", "b8", "b32", "mix"]
SERVER = "/Users/benreich/Projects/crazyidea/sllm/engine/target/release/sllm-server"  # main's (its libllama engine: ggml)
MULTI = {"qwen3": "ref128.json", "gemma3": "gref128.json", "e2b": "gref128.json", "phi": "pmulti.json", "llama": "lmmulti.json",
         "mistral": "mmulti.json", "q35": "q35multi.json", "oss": "ossmulti.json", "q36": "q36multi.json"}  # many-agent prompts


def build():
    """qbench from this directory's sn.m / core.h (rebuilt when they change)."""
    src = [os.path.join(HERE, f) for f in ("qbench.m", "sn.m", "core.h", "sn.h")]
    tag = hashlib.sha1(b"".join(open(f, "rb").read() for f in src)).hexdigest()[:12]
    out = os.path.join(DATA, "bin", f"qbench-{tag}")
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.run(["clang", "-O2", "-fobjc-arc", "-Wno-unused-variable", "-Wno-unused-function", src[0], src[1], "-o", out,
                        "-framework", "Metal", "-framework", "Foundation"], check=True, cwd=HERE)
    return out


def one(binary, key, env, only):
    gguf, ref = MODELS[key]
    e = dict(os.environ, **env)
    if only:
        e["QB_ONLY"] = only
    t0 = time.time()
    p = subprocess.run([sys.executable, LOCK, "--owner", "qb", "--", binary, GGUF + gguf, os.path.join(HERE, "kernels.metal"), os.path.join(HERE, ref)],
                       capture_output=True, text=True, env=e, cwd=HERE)
    line = [x for x in p.stdout.splitlines() if x.startswith("{")]
    r = json.loads(line[-1]) if line else {"error": (p.stderr or p.stdout)[-300:]}
    r["wall_s"] = round(time.time() - t0, 1)
    return r


def load(name):
    try:
        return json.load(open(os.path.join(DATA, name)))
    except (OSError, ValueError):
        return {}


def pct(a, b):
    return f"{(a / b - 1) * 100:+.0f}%" if a and b else ""


def cell(r, s):
    x = r.get("res", {}).get(s)
    return x.get("tok_s") if x else None


def bits(r, refr):
    """'=' every hashed row equals the reference's, 'DIFF: scen' otherwise, '?' with no reference."""
    if not refr or "hash" not in refr:
        return "?"
    bad = [s for s, h in r.get("hash", {}).items() if s in refr["hash"] and refr["hash"][s] != h and not s.endswith("_end")]
    return "=" if not bad else "DIFF " + ",".join(bad)


def table(results, refs, ggml, title):
    print(f"\n{title}")
    hdr = f"{'model':8} " + " ".join(f"{s:>14}" for s in SCEN) + "   bits   exact"
    print(hdr)
    for key, r in results.items():
        if "error" in r:
            print(f"{key:8} ERROR {r['error'][:150]}")
            continue
        cols = []
        for s in SCEN:
            v = cell(r, s)
            if v is None:
                cols.append(f"{'-':>14}")
                continue
            g = (ggml.get(key) or {}).get(s)
            cols.append(f"{v:>7} {pct(v, g):>6}" if g else f"{v:>7} {'':>6}")
        ex = r.get("exact", {})
        exs = "ok" if ex and all(ex.values()) else ("-" if not ex else "NO " + ",".join(k for k, v in ex.items() if not v))
        heat = r["res"].get("w1_end", {}).get("ms", 0) / max(r["res"].get("w1", {}).get("ms", 1), 1e-9)
        print(f"{key:8} " + " ".join(cols) + f"   {bits(r, refs.get(key)):5}  {exs}" + ("  HOT" if heat > 1.03 else "") + f"  ({r['wall_s']}s)")
    print("tok/s (mix: the chat's); % vs ggml b11669: read / w1 / b8 llama-batched-bench, b32 / mix sllm-server's libllama engine")


def cmd_run(a, save_ref=False):
    binary = build()
    env = dict(x.split("=", 1) for x in a.env)
    res = {k: one(binary, k, env, a.only) for k in a.models}
    refs, ggml = load("ref.json"), load("ggml.json")
    table(res, refs, ggml, f"qb run {' '.join(a.env) or '(default)'}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(DATA, "runs"), exist_ok=True)
    json.dump({"env": env, "res": res}, open(os.path.join(DATA, "runs", f"{stamp}.json"), "w"))
    if save_ref:
        refs.update({k: v for k, v in res.items() if "error" not in v})
        json.dump(refs, open(os.path.join(DATA, "ref.json"), "w"), indent=1)
        print(f"reference saved ({', '.join(res)})")


def cmd_ab(a):
    binary = build()
    var = dict(x.split("=", 1) for x in a.env)
    base_env = {k: os.environ.get("QB_BASE_" + k, "0") for k in var} if a.base_zero else {}
    print(f"qb ab: base {base_env or '(default)'} vs {var}, order base/var/var/base")
    for key in a.models:
        rs = [one(binary, key, e, a.only) for e in (base_env, var, var, base_env)]
        if any("error" in r for r in rs):
            print(f"{key:8} ERROR {[r.get('error', '')[:120] for r in rs if 'error' in r]}")
            continue
        line = []
        for s in SCEN + ["mix_bg"]:
            src = "mix" if s == "mix_bg" else s
            fld = "bg_tok_s" if s == "mix_bg" else "tok_s"
            b = [r["res"].get(src, {}).get(fld) for r in (rs[0], rs[3])]
            v = [r["res"].get(src, {}).get(fld) for r in (rs[1], rs[2])]
            if None in b or None in v:
                continue
            bm, vm = sum(b) / 2, sum(v) / 2
            line.append(f"{s} {bm:.0f}->{vm:.0f} {pct(vm, bm)}")
        same = bits(rs[1], rs[0])
        ex = rs[1].get("exact", {})
        print(f"{key:8} " + " | ".join(line) + f" | bits vs base: {same} | exact: {'ok' if all(ex.values()) else ex}")


def cmd_ggml(a):
    cache = load("ggml.json")
    for key in a.models:
        gguf = GGUF + MODELS[key][0]
        out = {}
        for npp, npl in ((1024, "1"), (512, "8,32")):
            p = subprocess.run([sys.executable, LOCK, "--owner", "qb-ggml", "--", GGML_BENCH, "-m", gguf, "-c", "20000", "-b", "2048", "-ub", "512",
                                "-npp", str(npp), "-ntg", "32", "-npl", npl, "-ngl", "99", "-fa", "on"], capture_output=True, text=True)
            for ln in p.stdout.splitlines():
                c = [x.strip() for x in ln.strip().strip("|").split("|")]
                if len(c) >= 10 and re.fullmatch(r"\d+", c[0] or "x"):
                    pp, tg, b = int(c[0]), int(c[1]), int(c[2])
                    if b == 1 and pp == 1024:
                        out["read"] = round(float(c[5]))
                        out["w1"] = round(float(c[7]))
                    elif b in (8, 32):
                        out[f"b{b}"] = round(float(c[7]))
            if not out:
                print(key, "ggml parse failed:", (p.stderr or p.stdout)[-400:])
        # 32 agents and the chat beside 31: ggml through sllm-server's libllama engine (stock llama-batched-bench runs a
        # 32-sequence step 4x its 8-sequence one: not the engine our server numbers compare against)
        env = dict(os.environ, ENGINE="today", MODELFILE=MODELS[key][0], PORT="19041", SLLM_BIN=SERVER)
        p = subprocess.run([sys.executable, LOCK, "--owner", "qb-ggml", "--", sys.executable, "-I", os.path.join(HERE, "qb_srv8.py"),
                            os.path.join(HERE, MULTI.get(key, MODELS[key][1])), "32", "64"], capture_output=True, text=True, env=env, cwd=HERE)
        r = [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")]
        if r and r[-1].get("write_tok_s"):
            out["b32"] = r[-1]["write_tok_s"]
        p = subprocess.run([sys.executable, LOCK, "--owner", "qb-ggml", "--", sys.executable, "-I", os.path.join(HERE, "srv_mix.py"),
                            os.path.join(HERE, MULTI.get(key, MODELS[key][1])), "31", "64", "192"], capture_output=True, text=True, env=env, cwd=HERE)
        r = [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")]
        if r and r[-1].get("chat_with_bg_tok_s"):
            out["mix"] = r[-1]["chat_with_bg_tok_s"]
            out["mix_bg"] = r[-1]["bg_tok_s_during_chat"]
        cache[key] = out
        print(key, out)
        os.makedirs(DATA, exist_ok=True)
        json.dump(cache, open(os.path.join(DATA, "ggml.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "ab", "ref", "ggml"])
    ap.add_argument("-m", "--models", default=",".join(MODELS))
    ap.add_argument("-e", "--env", action="append", default=[])
    ap.add_argument("--only", default=None)
    ap.add_argument("--base-zero", action="store_true", help="ab: the base sets each variant variable to 0 (QB_BASE_<VAR> to override)")
    a = ap.parse_args()
    a.models = [m for m in a.models.split(",") if m]
    {"run": cmd_run, "ref": lambda a: cmd_run(a, True), "ab": cmd_ab, "ggml": cmd_ggml}[a.cmd](a)
