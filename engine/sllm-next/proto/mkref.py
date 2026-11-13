"""Prompts in a model's own tokens for srv_one.py / srv_ram.py (REF= / DOCREF=): a short prompt (prompts[1]) and a
1k-token document (prompts[0]) cut from llama.cpp's docs, tokenized by sllm-server; DOCS8.json: 8 such documents (the
first the same) for srv_one.py's DOCS=8. usage: mkref.py OUT.json [DOCS8.json] [MODELFILE env]"""
import json
import os
import sys
import tempfile

sys.path.insert(0, "/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/rollover")
from common import *  # noqa: E402,F403

MODEL = MODELS + os.environ["MODELFILE"]
BIN = "/Users/benreich/Projects/crazyidea/sllm-wt/shared-target/release/sllm-server"
DOCS = "/Users/benreich/Projects/crazyidea/research/inference_runtime/vendor/llama-src/docs/"
tmp = tempfile.mkdtemp()
p = subprocess.Popen([BIN, "--model", MODEL, "--port", str(PORT), "--host", "127.0.0.1", "--ctx", "2048", "--agent-dir", tmp + "/a"],
                     stdout=open("/tmp/mkref.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(600):
        try:
            get("/health")
            break
        except Exception:
            time.sleep(0.3)
    doc = post("/tokenize", {"content": open(DOCS + "build.md").read(), "add_special": True})["tokens"][:1046]
    short = post("/tokenize", {"content": "Write a short story about a lighthouse keeper who finds a message in a bottle. " * 4, "add_special": True})["tokens"]
    json.dump({"prompts": [doc, short]}, open(sys.argv[1], "w"))
    print(len(doc), len(short))
    if len(sys.argv) > 2:
        files = ["build.md", "autoparser.md", "speculative.md", "function-calling.md", "ops.md", "docker.md", "build-s390x.md", "multi-gpu.md"]
        docs = [post("/tokenize", {"content": open(DOCS + f).read(), "add_special": True})["tokens"][:1046] for f in files]
        assert docs[0] == doc and all(len(d) == 1046 for d in docs)
        json.dump({"prompts": docs}, open(sys.argv[2], "w"))
finally:
    stop(p)
