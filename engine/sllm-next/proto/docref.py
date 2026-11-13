"""Reading reference: N real document prompts (DOC tokens of the bench document each, at different offsets) tokenized by
today's engine, its read speed for them (1 and N at once, first token wall time) and its greedy 32-token answer alone.
usage: docref.py OUT.json N DOC"""
import os
import sys
import tempfile

sys.path.insert(0, "/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/rollover")
from common import *  # noqa: E402,F403

OUT, N, DOC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
FR = json.load(open("/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/limits/frames.json"))[os.environ.get("MODELKEY", "qwen")]
TEXT = "/Users/benreich/Projects/crazyidea/sllm/results/playground-live/files/SLLM/sllm-progress/decisions.md"
tmp = tempfile.mkdtemp()
ctx = (DOC + 256) // 256 * 256 + 256
p = start(MODELS + FR["file"], ["--host", "127.0.0.1", "--ctx", str(ctx), "--kv-pages", str(N * ctx // 256), "--prompt-cache", "off", "--spec", "off",
                                "--agent-dir", tmp + "/a", "--spill-dir", tmp + "/s"], OUT + ".srv.log")
try:
    a, b = FR["task"].split("@@MSG@@")
    tk = lambda text, first=False: post("/tokenize", {"content": text, "add_special": bool(first and FR["bos"]), "parse_special": True})["tokens"]
    doc = post("/tokenize", {"content": open(TEXT, errors="replace").read()[:400000], "add_special": False, "parse_special": False})["tokens"]
    prompts = [tk(FR["prefix"]["summarize"] + a + f"<document part {i}>\n", True) + doc[i * 300:i * 300 + DOC] + tk("\n</document>\n\nSummarize this document in 3 sentences." + b) for i in range(N)]
    ref = [post("/completion", {"prompt": q, "n_predict": 32, "temperature": 0, "return_tokens": True, "cache_prompt": False, "ignore_eos": True})["tokens"] for q in prompts]
    speeds = {}
    for n in (1, N):
        res = [None] * n
        def one(i):
            t = time.monotonic()
            post("/completion", {"prompt": prompts[i], "n_predict": 1, "temperature": 0, "cache_prompt": False})
            res[i] = time.monotonic() - t
        for rep in range(2):
            t0 = time.monotonic()
            th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
            [x.start() for x in th]
            [x.join() for x in th]
            speeds.setdefault(n, []).append(round(sum(map(len, prompts[:n])) / (time.monotonic() - t0)))
    json.dump({"prompts": prompts, "ref": ref, "n_out": 32}, open(OUT, "w"))
    print(json.dumps({"engine": "today", "doc_tokens": len(prompts[0]), "read_tok_s": speeds}))
finally:
    stop(p)
