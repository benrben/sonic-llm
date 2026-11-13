"""--engine next with a 512-cell cache and 4 agents of 200-token system prompts of their own under --pressure-relief: the
coldest idle agents go to the SSD to make room (the engine refuses a full cache as kv_pool_exhausted_error), a later turn
restores one with the same tokens. usage: srv_pressure.py"""
import json, os, sys, tempfile
sys.path.insert(0, "/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/rollover")
from common import *  # noqa
BIN = "/Users/benreich/Projects/crazyidea/sllm-wt/shared-target/release/sllm-server"
DOCS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "docref.json")))["prompts"]
tmp = tempfile.mkdtemp()
p = subprocess.Popen([BIN, "--model", MODELS + "Qwen3-4B-Q4_0.gguf", "--port", str(PORT), "--host", "127.0.0.1", "--engine", "next", "--ctx", "1024", "--kv-pages", "2",
                      "--prompt-cache", "off", "--agent-dir", tmp + "/a", "--pressure-relief", "on"], stdout=open("/tmp/srv-pressure.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(600):
        try: get("/health"); break
        except Exception: time.sleep(0.2)
    out = {}
    ids = []
    for i in range(4):
        try:
            a = newagent(system_ids=DOCS[i][:200])
            code, r = turn(a, tok(" Say hello."), 8, return_tokens=True)
            ids.append(a); out[f"agent{i}"] = [code, (r.get("content") or r.get("error", {}).get("message", ""))[:60]]
        except urllib.error.HTTPError as e:
            out[f"agent{i}"] = [e.code, e.read()[:160].decode()]
    time.sleep(1)
    out["states"] = {a: agent(a).get("state") for a in ids}
    code, r = turn(ids[0], tok(" Again?"), 8, return_tokens=True) if ids else (None, {})
    out["agent0_again"] = [code, (r.get("content") or str(r)[:80])[:60], r.get("tokens")]
    out["lifecycle"] = {k: v for k, v in get("/metrics").get("lifecycle", {}).items() if k in ("agents", "suspends", "restores", "pressure_relief")}
    print(json.dumps(out, indent=1))
finally:
    stop(p)
