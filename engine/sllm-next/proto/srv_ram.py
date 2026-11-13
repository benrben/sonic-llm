"""One interactive agent's memory through sllm-server, the new engine against today's, alternating: the server's wired
memory peak over the idle machine (ab.py's measure) and its footprint peak, while it reads a short prompt and writes 256
tokens, then reads a 1k-token document and writes 128. Both engines get one agent's cache (--ctx 4096 = --kv-pages 16).
REF / DOCREF: prompts in the model's tokens (mkref.py); NEW_ARGS: more flags for the new engine. usage: srv_ram.py OUT.jsonl REPS [MODELFILE env]"""
import json
import os
import sys
import tempfile

sys.path.insert(0, "/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/rollover")
from common import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = MODELS + os.environ.get("MODELFILE", "Qwen3-4B-Q4_0.gguf")
G = "g" if "gemma" in MODEL else ""
BIN = "/Users/benreich/Projects/crazyidea/sllm-wt/shared-target/release/sllm-server"
SHORT = json.load(open(os.environ.get("REF", f"{HERE}/{G}ref128.json")))["prompts"][1]
DOC = json.load(open(os.environ.get("DOCREF", f"{HERE}/{G}docref.json")))["prompts"][0]
OUT, REPS = sys.argv[1], int(sys.argv[2])
GB = 2 ** 30


def idle_wired():
    """wired memory once the last run's memory is back: the lowest of a few seconds of samples"""
    v = []
    for _ in range(8):
        time.sleep(0.5)
        v.append(Mem(os.getpid(), 999, 999).wired() / GB)
    return min(v)


def gen(ids, n):
    post("/completion", {"prompt": ids, "n_predict": n, "temperature": 0, "lane": "interactive", "cache_prompt": False})


def run(engine):
    tmp = tempfile.mkdtemp()
    idle = idle_wired()
    args = [BIN, "--model", MODEL, "--port", str(PORT), "--host", "127.0.0.1", "--ctx", "4096", "--kv-pages", "16", "--prompt-cache", "off",
            "--agent-dir", tmp + "/a"] + (["--engine", "next"] + os.environ.get("NEW_ARGS", "").split() if engine == "new" else [])
    p = subprocess.Popen(args, stdout=open(f"/tmp/srv-ram-{engine}.log", "w"), stderr=subprocess.STDOUT)
    mem = Mem(p.pid, 40, 44, swap_mb=4096, every=0.1)
    mem.start()
    try:
        for _ in range(600):
            try:
                get("/health")
                break
            except Exception:
                time.sleep(0.2)
        gen(SHORT, 256)
        gen(DOC, 128)
    finally:
        mem.go = False
        mem.join(2)
        stop(p)
    m = mem.summary()
    return {"engine": engine, "wired_gb": round(m["wired_peak_gb"] - idle, 2), "footprint_gb": m["footprint_peak_gb"]}


rows = []
for i in range(REPS):
    for eng in ("new", "today"):
        rows.append(run(eng))
        print(json.dumps(rows[-1]), flush=True)
summary = {"model": os.path.basename(MODEL), **{k: {e: [r[k] for r in rows if r["engine"] == e] for e in ("new", "today")} for k in ("wired_gb", "footprint_gb")}}
print(json.dumps(summary))
with open(OUT, "a") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.write(json.dumps(summary) + "\n")
