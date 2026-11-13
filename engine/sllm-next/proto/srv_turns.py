"""A chat's second turn on --engine next: turn 1 asks about a document, turn 2 sends the conversation back with a follow-up
(as a chat client does). Prints each turn's time to the first token and the prompt tokens the server read, for the new
build and OLD_BIN. usage: srv_turns.py [MODELFILE env]"""
import json
import os
import sys
import tempfile

sys.path.insert(0, "/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/rollover")
from common import *  # noqa: E402,F403

MODEL = MODELS + os.environ.get("MODELFILE", "Qwen3.5-4B-Q4_0.gguf")
BIN = "/Users/benreich/Projects/crazyidea/sllm-wt/shared-target/release/sllm-server"
DOC = open("/Users/benreich/Projects/crazyidea/research/inference_runtime/vendor/llama-src/docs/build.md").read()[:6000]


def ask(messages):
    t0 = time.monotonic()
    before = get("/metrics")
    r = post("/v1/chat/completions", {"messages": messages, "max_tokens": 48, "temperature": 0}, timeout=600)
    t1 = time.monotonic()
    after = get("/metrics")
    read = after["prompt_tokens_evaluated"] - before["prompt_tokens_evaluated"] if "prompt_tokens_evaluated" in after else None
    return r["choices"][0]["message"].get("content") or "", round((t1 - t0) * 1000), read


def run(bin_):
    tmp = tempfile.mkdtemp()
    p = subprocess.Popen([bin_, "--model", MODEL, "--port", str(PORT), "--host", "127.0.0.1", "--ctx", "8192", "--kv-pages", "64",
                          "--agent-dir", tmp + "/a", "--engine", "next"], stdout=open("/tmp/srv-turns.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            try:
                get("/health")
                break
            except Exception:
                time.sleep(0.2)
        m = [{"role": "user", "content": f"<document>\n{DOC}\n</document>\n\nIn one sentence: what does this document explain?"}]
        a1, ms1, read1 = ask(m)
        m += [{"role": "assistant", "content": a1}, {"role": "user", "content": "Name one build option it mentions."}]
        a2, ms2, read2 = ask(m)
        return {"turn1_ms": ms1, "turn1_read": read1, "turn2_ms": ms2, "turn2_read": read2, "answer2": a2[:60]}
    finally:
        stop(p)


old = os.environ.get("OLD_BIN", "/Users/benreich/Projects/crazyidea/sllm/engine/target/release/sllm-server")
for name, b in (("old", old), ("new", BIN), ("old", old), ("new", BIN)):
    print(name, json.dumps(run(b)), flush=True)
