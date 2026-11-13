"""Quality, scored, for any GGUF: quality.py's known-answer test (32 documents with six IMPORTANT lines hidden in each;
"list six" and "find one") through sllm-server's /v1/chat/completions (the model's own chat template), 32 requests at
once, on today's engine and on --engine next, the same scoring; and how many answers are the same text on both.
CONC: at most this many requests at once (default all 32); a 503 (the cache is full now) is retried after a pause.
usage: srv_quality.py OUT.jsonl [MODELFILE, SEED, CONC env]"""
import json
import os
import random as rnd
import re
import sys
import tempfile

sys.path.insert(0, "/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/rollover")
from common import *  # noqa: E402,F403

MODEL = MODELS + os.environ["MODELFILE"]
BIN = os.environ.get("SLLM_BIN", "/Users/benreich/Projects/crazyidea/sllm-wt/shared-target/release/sllm-server")
DOCS = "/Users/benreich/Projects/crazyidea/sllm/results/playground-live/files/SLLM/sllm-progress/decisions.md"
NAMES = ["Kestrel", "Marlin", "Osprey", "Heron", "Falcon", "Lynx", "Otter", "Badger", "Raven", "Stoat", "Puffin", "Ibex", "Bison", "Mink", "Crane", "Wren", "Shrike", "Tern"]
CITIES = ["Lisbon", "Oslo", "Quito", "Nairobi", "Hanoi", "Perth", "Lima", "Tunis", "Riga", "Cairo", "Dakar", "Bergen"]
N, N_OUT = 32, int(os.environ.get("N_OUT", "160"))
GATE = threading.Semaphore(int(os.environ.get("CONC", str(N))))
OUT = sys.argv[1]
TEXT = open(DOCS, errors="replace").read()


def build(r, i):
    """quality.py's build on text: a ~4,800-character cut of the document, six IMPORTANT lines spread through it"""
    ns = r.sample(NAMES, 6)
    vals = [str(r.randrange(1000, 9999)) for _ in ns]
    home = r.sample(CITIES, 6)
    cut = TEXT[i * 1200:i * 1200 + 4800]
    step = len(cut) // 6
    body = "".join(cut[k * step:(k + 1) * step] + f"\nIMPORTANT: project {ns[k]} is based in {home[k]} and its code is {vals[k]}.\n" for k in range(6))
    ask = r.randrange(6)
    q1 = "List every project named in an IMPORTANT line with its code, as: name code, one per line."
    q2 = f"Which project is based in {home[ask]}, and what is its code? Answer with the name and the code."
    doc = f"<document>\n{body}\n</document>\n\nQuestion: "
    return (doc + q1, vals, None), (doc + q2, [vals[ask]], ns[ask])


def score(items, texts):
    right = total = found = invented = 0
    for (p, want, name), got in zip(items, texts):
        total += len(want)
        right += sum(v in got for v in want)
        if name:
            found += name in got
        invented += len(set(re.findall(r"\b\d{4}\b", got)) - set(want)) if name is None else 0
    return [right, total, found, invented]


def run(engine, built):
    tmp = tempfile.mkdtemp()
    p = subprocess.Popen([BIN, "--model", MODEL, "--port", str(PORT), "--host", "127.0.0.1", "--ctx", "2560", "--kv-pages", str(N * 2560 // 256),
                          "--prompt-cache", "off", "--spec", "off", "--agent-dir", tmp + "/a"] + (["--engine", "next"] if engine == "next" else []),
                         stdout=open(f"/tmp/srv-quality-{engine}.log", "w"), stderr=subprocess.STDOUT)
    out = {}
    try:
        for _ in range(900):
            try:
                get("/health")
                break
            except Exception:
                time.sleep(0.3)
        for ti, test in enumerate(("list six", "find one")):
            items = [b[ti] for b in built]
            res = [None] * N

            def one(i):
                body = {"messages": [{"role": "user", "content": items[i][0]}], "max_tokens": N_OUT, "temperature": 0, "stream": False}
                with GATE:
                    for _ in range(600):
                        try:
                            r = post("/v1/chat/completions", body, timeout=600)
                            res[i] = r["choices"][0]["message"].get("content") or ""
                            break
                        except Exception as e:
                            res[i] = f"ERROR {e}"
                            if "503" not in str(e):
                                break
                            time.sleep(0.5)

            th = [threading.Thread(target=one, args=(i,)) for i in range(N)]
            [t.start() for t in th]
            [t.join() for t in th]
            out[test] = (items, res)
    finally:
        stop(p)
    return out


r = rnd.Random(int(os.environ.get("SEED", "77")))
built = [build(r, i) for i in range(N)]
today, new = run("today", built), run("next", built)
for test in ("list six", "find one"):
    items, a = today[test]
    _, b = new[test]
    row = {"model": os.path.basename(MODEL), "seed": os.environ.get("SEED", "77"), "test": test, "today": score(items, a), "next": score(items, b),
           "same_text": f"{sum(x == y for x, y in zip(a, b))}/{N}", "errors": [x[:80] for x in a + b if x.startswith("ERROR")][:2]}
    print(json.dumps(row), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
