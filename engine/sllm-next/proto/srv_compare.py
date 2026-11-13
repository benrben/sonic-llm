"""The same prompts on today's engine and on --engine next for one GGUF (MODELFILE): greedy tokens, how many sequences are
the same and where the others first differ. Plain-text prompts (no chat template). OLD_BIN set: --engine next on that
server binary instead of today's engine (a change's tokens against the build before it). LONG=N: 4 prompts of N tokens
cut from a long text (past the sliding windows) instead. LANE: the requests' lane (interactive / background).
usage: srv_compare.py N_OUT"""
import json
import os
import sys
import tempfile

sys.path.insert(0, "/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/rollover")
from common import *  # noqa: E402,F403

MODEL = MODELS + os.environ["MODELFILE"]
BIN = "/Users/benreich/Projects/crazyidea/sllm-wt/shared-target/release/sllm-server"
N_OUT = int(sys.argv[1])
TEXTS = ["The capital of France is", "Once upon a time, in a small village by the sea,", "The three laws of thermodynamics are",
         "def fibonacci(n):", "A good code review checks", "The main causes of the First World War were",
         "Photosynthesis is the process by which", "To make bread at home, you need"]


OLD_BIN = os.environ.get("OLD_BIN")
LONG = int(os.environ.get("LONG", "0"))
LANE = os.environ.get("LANE")
CTX = max(1024, LONG + N_OUT + 64)
DOCS = "/Users/benreich/Projects/crazyidea/research/inference_runtime/vendor/llama-src/docs/"
TEXT = "".join(open(DOCS + f).read() for f in ("build.md", "autoparser.md", "speculative.md", "function-calling.md"))


def run(engine, bin_=BIN):
    tmp = tempfile.mkdtemp()
    p = subprocess.Popen([bin_, "--model", MODEL, "--port", str(PORT), "--host", "127.0.0.1", "--ctx", str(CTX), "--kv-pages", str(max(16, 2 * CTX // 256)), "--prompt-cache", "off",
                          "--spec", "off", "--agent-dir", tmp + "/a"] + (["--engine", "next"] if engine == "next" else []),
                         stdout=open(f"/tmp/srv-compare-{engine}.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            try:
                get("/health")
                break
            except Exception:
                time.sleep(0.3)
        eng = get("/caps").get("engine")
        out = []
        prompts = [post("/tokenize", {"content": t, "add_special": True})["tokens"] for t in TEXTS]
        if LONG:
            ids = post("/tokenize", {"content": TEXT, "add_special": False})["tokens"]
            prompts = [p[:1] + ids[k * len(ids) // 5:][:LONG] for k, p in enumerate(prompts[:4])]
        for ids in prompts:
            body = {"prompt": ids, "n_predict": N_OUT, "temperature": 0, "return_tokens": True, "ignore_eos": True, "cache_prompt": False}
            if LANE:
                body["lane"] = LANE
            r = post("/completion", body)
            out.append(r["tokens"])
        return (eng.get("engine") if isinstance(eng, dict) else str(eng)), out
    finally:
        stop(p)


e1, today = run("next", OLD_BIN) if OLD_BIN else run("today")
e2, new = run("next")
first = [next((i for i, (a, b) in enumerate(zip(x, y)) if a != b), None) for x, y in zip(new, today)]
print(json.dumps({"model": os.path.basename(MODEL), "engines": [e1, e2], "long": LONG, "lane": LANE, "same": f"{sum(f is None for f in first)}/{len(first)}", "first_difference_at": first}))
