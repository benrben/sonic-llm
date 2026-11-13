"""(qb.py ggml) N background agents writing at once: the write rate, and per library step the GPU time, the host time inside sn_step and
the step period (from /metrics deltas over the measured batch). usage: srv_8.py REF.json N N_OUT   env SLLM_BIN MODELFILE PORT ENGINE"""
import json
import os
import sys
import tempfile

sys.path.insert(0, "/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/rollover")
from common import *  # noqa: E402,F403

REF, N, N_OUT = json.load(open(sys.argv[1])), int(sys.argv[2]), int(sys.argv[3])
MODEL = MODELS + os.environ.get("MODELFILE", "Qwen3-4B-Q4_0.gguf")
BIN = os.environ["SLLM_BIN"]


class Job(threading.Thread):
    def __init__(self, body, go):
        super().__init__(daemon=True)
        self.body, self.go, self.t, self.err = body, go, [], None

    def run(self):
        self.go.wait()
        try:
            c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
            c.request("POST", "/completion", json.dumps(self.body), {"Content-Type": "application/json"})
            r, buf = c.getresponse(), b""
            while True:
                b = r.read1(65536)
                if not b:
                    return
                now = time.monotonic()
                buf += b
                while b"\n\n" in buf:
                    ev, buf = buf.split(b"\n\n", 1)
                    if ev.startswith(b"data:") and ev[5:].strip() != b"[DONE]":
                        self.t += [now] * len(json.loads(ev[5:]).get("tokens") or [])
        except Exception as e:
            self.err = repr(e)


def batch(prompts):
    go = threading.Event()
    js = [Job({"prompt": p, "n_predict": N_OUT, "temperature": 0, "stream": True, "return_tokens": True, "lane": "background",
               "cache_prompt": False, "ignore_eos": True}, go) for p in prompts]
    [j.start() for j in js]
    go.set()
    [j.join(600) for j in js]
    ok = [j for j in js if j.t]
    a, b = max(j.t[0] for j in ok), min(j.t[-1] for j in ok)
    return round(sum(a <= x <= b for j in ok for x in j.t) / (b - a)) if b > a else None, a, b


def spec():
    try:
        return get("/metrics").get("spec") or {}
    except Exception:
        return {}


ctx = 1024
tmp = tempfile.mkdtemp()
ENGINE = [] if os.environ.get("ENGINE") == "today" else ["--engine", "next"]
p = subprocess.Popen([BIN, "--model", MODEL, "--port", str(PORT), "--host", "127.0.0.1", *ENGINE, "--ctx", str(ctx), "--kv-pages", str(N * ctx // 256 + 4),
                      "--prompt-cache", "off", "--agent-dir", tmp + "/a", "--spill-dir", tmp + "/s"], stdout=open("/tmp/srv-8.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(600):
        try:
            get("/health")
            break
        except Exception:
            time.sleep(0.2)
    batch(REF["prompts"][-N:])  # warm-up on other prompts
    s0 = spec()
    t0 = time.monotonic()
    rate, a, b = batch(REF["prompts"][:N])
    t1 = time.monotonic()
    s1 = spec()
    out = {"engine": "today" if not ENGINE else "next", "agents": N, "write_tok_s": rate}
    if s1.get("lib_steps"):
        n = s1["lib_steps"] - s0["lib_steps"]
        out.update({"steps": n, "gpu_ms_step": round((s1["gpu_ms"] - s0["gpu_ms"]) / n, 2), "run_ms_step": round((s1["run_ms"] - s0["run_ms"]) / n, 2),
                    "period_ms": round((t1 - t0) * 1e3 / n, 2), "bg_drafted": s1.get("bg_drafted", 0) - s0.get("bg_drafted", 0),
                    "bg_kept": s1.get("bg_kept", 0) - s0.get("bg_kept", 0), "shared_kept": (s1.get("shared") or {}).get("kept", 0) - (s0.get("shared") or {}).get("kept", 0)})
    print(json.dumps(out))
finally:
    stop(p)
