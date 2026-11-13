"""Agents that share a long system prompt in a small cache, the new engine against today's: 32 agents created with the
same 700-token system prompt, each then takes a short turn of its own. Both engines get the same cache (--kv-pages) and
--pressure-relief on. Reports the turns answered, the agents parked on the SSD to make room, the cache in use and the
server's wired memory over idle. usage: srv_share.py OUT.jsonl [MODELFILE env]"""
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
SYS = json.load(open(f"{HERE}/{G}docref.json"))["prompts"][0][:700]
N, KV_PAGES = 32, int(os.environ.get("KV_PAGES", "32"))
OUT = sys.argv[1]
GB = 2 ** 30


def idle_wired():
    v = []
    for _ in range(8):
        time.sleep(0.5)
        v.append(Mem(os.getpid(), 999, 999).wired() / GB)
    return min(v)


def run(engine):
    tmp = tempfile.mkdtemp()
    idle = idle_wired()
    args = [BIN, "--model", MODEL, "--port", str(PORT), "--host", "127.0.0.1", "--ctx", "2048", "--kv-pages", str(KV_PAGES), "--prompt-cache", "off",
            "--agent-dir", tmp + "/a", "--pressure-relief", "on"] + (["--engine", "next"] if engine == "new" else [])
    p = subprocess.Popen(args, stdout=open(f"/tmp/srv-share-{engine}.log", "w"), stderr=subprocess.STDOUT)
    mem = Mem(p.pid, 40, 44, swap_mb=4096, every=0.1)
    mem.start()
    try:
        for _ in range(600):
            try:
                get("/health")
                break
            except Exception:
                time.sleep(0.2)
        ok, refused, outs = 0, [], []
        for i in range(N):
            try:
                a = newagent(system_ids=SYS)
                code, r = turn(a, tok(f" Question {i}: name one thing the text says."), 16, return_tokens=True, ignore_eos=True)
                ok += code == 200
                outs.append(r.get("tokens"))
                if code != 200:
                    refused.append((code, json.dumps(r)[:100]))
            except urllib.error.HTTPError as e:
                refused.append((e.code, e.read()[:100].decode()))
        m = get("/metrics")
        life = m.get("lifecycle", {})
        res = {"engine": engine, "turns_ok": f"{ok}/{N}", "parked": life.get("suspends", {}).get("pressure"), "agents": life.get("agents"),
               "pages_active": m.get("pages_active"), "pages_total": m.get("pages_reserved"), "refused": refused[:2], "outs": outs}
    finally:
        mem.go = False
        mem.join(2)
        stop(p)
    res["wired_gb"] = round(mem.summary()["wired_peak_gb"] - idle, 2)
    return res


rows = [run("new"), run("today")]
for r in rows:
    print(json.dumps({k: v for k, v in r.items() if k != "outs"}))
print("same answers new vs today:", sum(a == b for a, b in zip(rows[0]["outs"], rows[1]["outs"])), "/", N)
with open(OUT, "a") as f:
    for r in rows:
        f.write(json.dumps({"model": os.path.basename(MODEL), **{k: v for k, v in r.items() if k != "outs"}}) + "\n")
