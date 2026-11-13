"""sllm-server --engine next over HTTP: N reference prompts at once and one at a time (tokens equal?), the write rate with all
N writing, and an agent with two turns. usage: srv_test.py REF.json N N_OUT [extra server args...]"""
import json
import os
import sys
import tempfile

sys.path.insert(0, "/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/rollover")
from common import *  # noqa: E402,F403

REF, N, N_OUT = json.load(open(sys.argv[1])), int(sys.argv[2]), int(sys.argv[3])
MODEL = MODELS + os.environ.get("MODELFILE", "Qwen3-4B-Q4_0.gguf")
BIN = os.environ.get("SLLM_BIN", "/Users/benreich/Projects/crazyidea/sllm-wt/shared-target/release/sllm-server")


class Job(threading.Thread):
    def __init__(self, body, go):
        super().__init__(daemon=True)
        self.body, self.go, self.t, self.ids, self.err = body, go, [], [], None

    def run(self):
        self.go.wait()
        try:
            c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
            c.request("POST", "/completion", json.dumps(self.body), {"Content-Type": "application/json"})
            r = c.getresponse()
            if r.status != 200:
                self.err = f"{r.status} {r.read()[:200]!r}"
                return
            buf = b""
            while True:
                b = r.read1(65536)
                if not b:
                    return
                now = time.monotonic()
                buf += b
                while b"\n\n" in buf:
                    ev, buf = buf.split(b"\n\n", 1)
                    if ev.startswith(b"data:") and ev[5:].strip() != b"[DONE]":
                        k = json.loads(ev[5:]).get("tokens") or []
                        self.t += [now] * len(k)
                        self.ids += k
        except Exception as e:
            self.err = repr(e)


def batch(prompts):
    go = threading.Event()
    js = [Job({"prompt": p, "n_predict": N_OUT, "temperature": 0, "stream": True, "return_tokens": True, "lane": "background", "cache_prompt": False,
               "ignore_eos": True}, go) for p in prompts]
    [j.start() for j in js]
    go.set()
    [j.join(600) for j in js]
    errs = [j.err for j in js if j.err]
    ok = [j for j in js if j.t]
    a, b = max(j.t[0] for j in ok), min(j.t[-1] for j in ok)
    rate = round(sum(a <= x <= b for j in ok for x in j.t) / (b - a)) if b > a else None
    return [j.ids for j in js], rate, errs


ctx = int(os.environ.get("CTX", "1024"))
tmp = tempfile.mkdtemp()
env = dict(os.environ, SLLM_BIN=BIN)
ENGINE = [] if os.environ.get("ENGINE") == "today" else ["--engine", "next"]  # ENGINE=today: the libllama engine, same test
p = subprocess.Popen([BIN, "--model", MODEL, "--port", str(PORT), "--host", "127.0.0.1", *ENGINE, "--ctx", str(ctx), "--kv-pages", str(N * ctx // 256 + 4),
                      "--prompt-cache", "off", "--agent-dir", tmp + "/a", "--spill-dir", tmp + "/s", *sys.argv[4:]],
                     stdout=open("/tmp/srv-next.log", "w"), stderr=subprocess.STDOUT, env=env)
try:
    for _ in range(600):
        try:
            get("/health")
            break
        except Exception:
            time.sleep(0.2)
    prompts = REF["prompts"][:N]
    together, rate, e1 = batch(prompts)
    solo = []
    for q in prompts:
        ids, _, e2 = batch([q])
        solo.append(ids[0])
    ref = [r[:N_OUT] for r in REF["ref"][:N]]
    # an agent: create with a system prompt, two turns
    a = post("/v1/agents", {"system_ids": prompts[0][:8]})
    aid = a.get("id") or a.get("agent_id")
    t1 = post(f"/v1/agents/{aid}/turn", {"prompt": prompts[1][8:], "n_predict": 16, "temperature": 0, "return_tokens": True, "stream": False})
    t2 = post(f"/v1/agents/{aid}/turn", {"prompt": prompts[2][8:20], "n_predict": 16, "temperature": 0, "return_tokens": True, "stream": False})
    print(json.dumps({"engine": "today" if not ENGINE else "next", "agents": N, "write_tok_s": rate, "together_equals_solo": f"{sum(x == y for x, y in zip(together, solo))}/{N}",
                      "solo_equals_today": f"{sum(x == y for x, y in zip(solo, ref))}/{N}", "errors": (e1 + e2)[:2],
                      "agent": {"id": aid, "turn1": (t1.get("content") or "")[:60], "turn2_tokens": len(t2.get("tokens") or []), "ctx": get(f"/v1/agents/{aid}").get("context_tokens")}}))
finally:
    stop(p)
