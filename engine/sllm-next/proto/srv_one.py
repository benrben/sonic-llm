"""One interactive agent through sllm-server, the new engine against today's, alternating: write tok/s over 256 tokens,
read tok/s of a short prompt and of a 1k-token document (time to first token), and the new engine's tokens with and
without queued steps (SLLM_NEXT_CHAIN=0). NEW_ARGS: more flags for the new engine's server; IGNORE_EOS=1: replies run
their full length.
usage: srv_one.py OUT.jsonl REPS [MODELFILE env]"""
import json
import os
import statistics
import sys
import tempfile

sys.path.insert(0, "/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/rollover")
from common import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = MODELS + os.environ.get("MODELFILE", "Qwen3-4B-Q4_0.gguf")
G = "g" if "gemma" in MODEL else ""
BIN = os.environ.get("SLLM_BIN", "/Users/benreich/Projects/crazyidea/sllm-wt/shared-target/release/sllm-server")
SHORT = json.load(open(os.environ.get("REF", f"{HERE}/{G}ref128.json")))["prompts"][1]
DOC = None if os.environ.get("DOCREF") == "none" else json.load(open(os.environ.get("DOCREF", f"{HERE}/{G}docref.json")))["prompts"][0]
# DOCS=N: also write 128 tokens after each of the first N document prompts and add them up (one document's greedy text
# can part from the other engine's at a near-tie and repeat the document more or less: drafts keep more or fewer tokens)
DOCS = json.load(open(os.environ.get("DOCREF", f"{HERE}/{G}docref.json")))["prompts"][:int(os.environ.get("DOCS", "0"))]
OUT, REPS = sys.argv[1], int(sys.argv[2])


def stream(ids, n):
    """(seconds to the first token, decode tok/s after it, token ids)"""
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=600)
    body = {"prompt": ids, "n_predict": n, "temperature": 0, "stream": True, "return_tokens": True, "lane": "interactive", "cache_prompt": False}
    if os.environ.get("IGNORE_EOS"):  # models that end at once on a raw prompt (Qwen2.5 on the document)
        body["ignore_eos"] = True
    t0 = time.monotonic()
    c.request("POST", "/completion", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    ts, ids_out, buf = [], [], b""
    while True:
        b = r.read1(65536)
        if not b:
            break
        now = time.monotonic()
        buf += b
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            if ev.startswith(b"data:") and ev[5:].strip() != b"[DONE]":
                k = json.loads(ev[5:]).get("tokens") or []
                ts += [now] * len(k)
                ids_out += k
    rate = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 else None
    stream.span = (len(ts) - 1, ts[-1] - ts[0])
    return ts[0] - t0, rate, ids_out


def run(engine, env=None):
    tmp = tempfile.mkdtemp()
    args = [BIN, "--model", MODEL, "--port", str(PORT), "--host", "127.0.0.1", "--ctx", "4096", "--kv-pages", "64", "--prompt-cache", "off",
            "--agent-dir", tmp + "/a"] + (["--engine", "next"] + os.environ.get("NEW_ARGS", "").split() if engine == "new" else [])
    p = subprocess.Popen(args, stdout=open(f"/tmp/srv-one-{engine}.log", "w"), stderr=subprocess.STDOUT, env=dict(os.environ, **(env or {})))
    try:
        for _ in range(600):
            try:
                get("/health")
                break
            except Exception:
                time.sleep(0.2)
        stream(SHORT, 8)  # warm
        t_short, rate, toks = stream(SHORT, 256)
        t_doc, rate_doc, dtoks = stream(DOC, 128) if DOC else (1.0, 0.0, [])
        n_all = s_all = 0
        docs_tokens = []
        for d in DOCS:
            docs_tokens.append(stream(d, 128)[2])
            n_all, s_all = n_all + stream.span[0], s_all + stream.span[1]
        m = get("/metrics").get("spec")
        return {"engine": engine, "write_tok_s": round(rate, 1), "write_after_doc_tok_s": round(rate_doc, 1),
                "read_short_tok_s": round(len(SHORT) / t_short), "read_doc_tok_s": round(len(DOC) / t_doc) if DOC else None,
                "write_after_docs_tok_s": round(n_all / s_all, 1) if DOCS else None, "spec": m, "tokens": toks, "doc_tokens": dtoks, "docs_tokens": docs_tokens}
    finally:
        stop(p)


rows = []
for i in range(REPS):
    for eng in ("new", "today"):
        r = run(eng)
        rows.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("tokens", "doc_tokens", "docs_tokens", "spec")}), json.dumps(r.get("spec"))[:120], flush=True)
sync = run("new", {"SLLM_NEXT_CHAIN": "0"})
new = [r for r in rows if r["engine"] == "new"]
summary = {"model": os.path.basename(MODEL), "same_tokens_as_plain_steps": [sync["tokens"] == new[0]["tokens"], sync["doc_tokens"] == new[0]["doc_tokens"]],
           "plain_write_tok_s": [sync["write_tok_s"], sync["write_after_doc_tok_s"]],
           "new_tokens_same_across_runs": all(r["tokens"] == new[0]["tokens"] for r in new),
           "first_token_differs_from_today_at": next((i for i, (a, b) in enumerate(zip(new[0]["tokens"], rows[1]["tokens"])) if a != b), None),
           "doc_differs_from_today_at": next((i for i, (a, b) in enumerate(zip(new[0]["doc_tokens"], rows[1]["doc_tokens"])) if a != b), None),
           "docs_differ_from_today_at": [next((i for i, (a, b) in enumerate(zip(x, y)) if a != b), None) for x, y in zip(new[0]["docs_tokens"], rows[1]["docs_tokens"])]}
for k in ("write_tok_s", "write_after_doc_tok_s", "read_short_tok_s", "read_doc_tok_s", "write_after_docs_tok_s"):
    summary[k] = {e: [r[k] for r in rows if r["engine"] == e] for e in ("new", "today")}
print(json.dumps(summary))
with open(OUT, "a") as f:
    for r in rows:
        f.write(json.dumps({k: v for k, v in r.items() if k not in ("tokens", "doc_tokens", "docs_tokens")}) + "\n")
    f.write(json.dumps(summary) + "\n")
