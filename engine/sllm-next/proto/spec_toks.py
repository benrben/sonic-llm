"""The greedy tokens of one engine (new or today), drafting off: 128 after the first document prompt, 256 after the short
prompt, with the prompts, for spec_replay.py. env MODELFILE, REF, DOCREF, PORT. usage: spec_toks.py ENGINE OUT.json"""
import json, os, sys, tempfile
sys.path.insert(0, "/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/rollover")
from common import *  # noqa
P = "/Users/benreich/Projects/crazyidea/sllm-wt/newengine-20261104/engine/sllm-next/proto/"
MODEL = MODELS + os.environ["MODELFILE"]
G = "g" if "gemma" in MODEL else ""
BIN = os.environ.get("SLLM_BIN", "/Users/benreich/Projects/crazyidea/sllm-wt/shared-target/release/sllm-server")
SHORT = json.load(open(os.environ.get("REF", f"{P}{G}ref128.json")))["prompts"][1]
DOC = json.load(open(os.environ.get("DOCREF", f"{P}{G}docref.json")))["prompts"][0]
eng, out = sys.argv[1], sys.argv[2]
def gen(ids, n):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=600)
    c.request("POST", "/completion", json.dumps({"prompt": ids, "n_predict": n, "temperature": 0, "return_tokens": True, "ignore_eos": True, "cache_prompt": False}), {"Content-Type": "application/json"})
    return json.loads(c.getresponse().read())["tokens"]
tmp = tempfile.mkdtemp()
args = [BIN, "--model", MODEL, "--port", str(PORT), "--host", "127.0.0.1", "--ctx", "4096", "--kv-pages", "64", "--prompt-cache", "off", "--agent-dir", tmp + "/a"] + (["--engine", "next"] if eng == "new" else ["--spec", "off"])
p = subprocess.Popen(args, stdout=open(f"/tmp/toks-{eng}.log", "w"), stderr=subprocess.STDOUT, env=dict(os.environ, SLLM_NEXT_SPEC="off"))
try:
    for _ in range(600):
        try:
            get("/health"); break
        except Exception:
            time.sleep(0.2)
    json.dump({"doc": DOC, "doc_out": gen(DOC, 128), "short": SHORT, "short_out": gen(SHORT, 256)}, open(out, "w"))
finally:
    stop(p)
