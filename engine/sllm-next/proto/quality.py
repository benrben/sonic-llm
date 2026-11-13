"""Quality, scored: the known-answer test of results/limits/truth32.py (seed 77: six facts hidden in each of 32 real
documents, "list six" and "find one"), 32 agents at once on today's engine and on the new one, the same scoring. Today's
engine tokenizes for both; the new engine's tokens become text through the GGUF vocabulary. usage: quality.py OUT.jsonl"""
import json
import os
import random as rnd
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, "/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/rollover")
from common import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
FR = json.load(open("/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/limits/frames.json"))[os.environ.get("MODELKEY", "qwen")]
DOCS = "/Users/benreich/Projects/crazyidea/sllm/results/playground-live/files/SLLM/sllm-progress/decisions.md"
NAMES = ["Kestrel", "Marlin", "Osprey", "Heron", "Falcon", "Lynx", "Otter", "Badger", "Raven", "Stoat", "Puffin", "Ibex", "Bison", "Mink", "Crane", "Wren", "Shrike", "Tern"]
CITIES = ["Lisbon", "Oslo", "Quito", "Nairobi", "Hanoi", "Perth", "Lima", "Tunis", "Riga", "Cairo", "Dakar", "Bergen"]
N, N_OUT = 32, 110
EOS = {151645, 151643} if os.environ.get("MODELKEY", "qwen") == "qwen" else {1, 106}
log = open(OUT, "a")


def vocab(path):
    """tokenizer.ggml.tokens of a GGUF, and GPT-2's byte <-> unicode map reversed: token ids to text."""
    import struct
    f = open(path, "rb")
    u = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    s_ = lambda: f.read(u("Q")).decode("utf8", "replace")
    sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    assert f.read(4) == b"GGUF"
    u("I"); u("Q"); nk = u("Q")
    toks, model = None, "gpt2"
    for _ in range(nk):
        k = s_(); t = u("I")
        if t == 8:
            v = s_()
            if k == "tokenizer.ggml.model":
                model = v
        elif t == 9:
            at, n = u("I"), u("Q")
            if at == 8:
                arr = [s_() for _ in range(n)]
                if k == "tokenizer.ggml.tokens":
                    toks = arr
            else:
                f.seek(n * sz[at], 1)
        else:
            f.seek(sz[t], 1)
        if toks is not None and model:
            pass
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    back = {chr(c): b for b, c in zip(bs, cs)}
    if model != "gpt2":  # SentencePiece: \u2581 is a space, <0xNN> a raw byte
        def spm(ids):
            out = bytearray()
            for i in ids:
                t = toks[i]
                if len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
                    out.append(int(t[3:5], 16))
                else:
                    out += t.replace("\u2581", " ").encode("utf8")
            return out.decode("utf8", "replace")
        return spm
    return lambda ids: bytes(back.get(ch, 32) for i in ids for ch in toks[i]).decode("utf8", "replace")


DETOK = vocab(MODELS + FR["file"])


def emit(r):
    log.write(json.dumps(r) + "\n")
    log.flush()
    print(json.dumps(r), flush=True)


def build(doc, tk, r, i):  # truth32.py's build, unchanged
    a, b = FR["task"].split("@@MSG@@")
    pre = FR["prefix"]["agent"]
    ns = r.sample(NAMES, 6)
    vals = [str(r.randrange(1000, 9999)) for _ in ns]
    home = r.sample(CITIES, 6)
    cut = doc[i * 300:i * 300 + 1200]
    parts, step = [], len(cut) // 6
    for k in range(6):
        parts += cut[k * step:(k + 1) * step] + tk(f"\nIMPORTANT: project {ns[k]} is based in {home[k]} and its code is {vals[k]}.\n")
    ask = r.randrange(6)
    q1 = "List every project named in an IMPORTANT line with its code, as: name code, one per line."
    q2 = f"Which project is based in {home[ask]}, and what is its code? Answer with the name and the code."
    head = tk(pre + a + "<document>\n", True) + parts
    return (head + tk(f"\n</document>\n\nQuestion: {q1}" + b), vals, None), (head + tk(f"\n</document>\n\nQuestion: {q2}" + b), [vals[ask]], ns[ask])


def score(items, texts):
    right = total = found = invented = 0
    for (p, want, name), got in zip(items, texts):
        total += len(want)
        right += sum(v in got for v in want)
        if name:
            found += name in got
        invented += len(set(re.findall(r"\b\d{4}\b", got)) - set(want)) if name is None else 0
    return [right, total, found, invented]


tmp = tempfile.mkdtemp()
srv = start(MODELS + FR["file"], ["--host", "127.0.0.1", "--auto-mode", "--ctx", "2048", "--kv-pages", str(N * 2048 // 256), "--prompt-cache", "off",
                                  "--spec", "off", "--agent-dir", tmp + "/a", "--spill-dir", tmp + "/s", "--many-agents", "on"], OUT + ".srv.log")
try:
    doc = post("/tokenize", {"content": open(DOCS, errors="replace").read()[:120000], "add_special": False, "parse_special": False})["tokens"]
    tk = lambda t, first=False: post("/tokenize", {"content": t, "add_special": bool(first and FR["bos"]), "parse_special": True})["tokens"]
    r = rnd.Random(int(os.environ.get("SEED", "77")))
    built = [build(doc, tk, r, i) for i in range(N)]
    for ti, test in enumerate(("list six", "find one")):
        items = [b[ti] for b in built]
        res = [None] * N

        def one(i):
            res[i] = post("/completion", {"prompt": items[i][0], "n_predict": N_OUT, "temperature": 0, "stream": False, "return_tokens": True,
                                          "lane": "background", "cache_prompt": False}, timeout=300)

        th = [threading.Thread(target=one, args=(i,)) for i in range(N)]
        [t.start() for t in th]
        [t.join() for t in th]
        emit({"engine": "today", "seed": os.environ.get("SEED", "77"), "test": test, "at_32": score(items, [x.get("content", "") for x in res])})
        out = {}
        for mode in ("together", "solo"):
            job = tempfile.mktemp(suffix=".json")
            json.dump({"prompts": [x[0] for x in items], "n_out": N_OUT, "together": mode == "together"}, open(job, "w"))
            subprocess.run([f"{HERE}/next", MODELS + FR["file"], job, job + ".out"], check=True, stderr=subprocess.DEVNULL)
            out[mode] = json.load(open(job + ".out"))["tokens"]
        texts = []
        for t in out["together"]:
            cut = next((i for i, x in enumerate(t) if x in EOS), len(t))
            texts.append(DETOK(t[:cut]))
        emit({"engine": "new", "seed": os.environ.get("SEED", "77"), "test": test, "at_32": score(items, texts), "together_equals_solo": sum(a == b for a, b in zip(out["together"], out["solo"]))})
finally:
    stop(srv)
