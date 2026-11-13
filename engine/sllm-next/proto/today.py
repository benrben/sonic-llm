"""Today's engine on the same prompts as the prototype: N agents write together, the write rate inside the window where all
of them write (bench.py's method), and their tokens against the solo reference. --many-agents on above 16 agents, as the
playground runs it. usage: today.py REF.json N [N_OUT]"""
import sys
import tempfile

sys.path.insert(0, "/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/rollover")
from common import *  # noqa: E402,F403

REF, N = json.load(open(sys.argv[1])), int(sys.argv[2])
N_OUT = int(sys.argv[3]) if len(sys.argv) > 3 else REF["n_out"]
MODEL = MODELS + "Qwen3-4B-Q4_0.gguf"


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
            buf = b""
            while True:
                b = r.read1(65536)
                if not b:
                    return
                now = time.clock_gettime_ns(time.CLOCK_UPTIME_RAW) / 1e9
                buf += b
                while b"\n\n" in buf:
                    ev, buf = buf.split(b"\n\n", 1)
                    if ev.startswith(b"data:") and ev[5:].strip() != b"[DONE]":
                        k = json.loads(ev[5:]).get("tokens") or []
                        self.t += [now] * len(k)
                        self.ids += k
        except Exception as e:
            self.err = repr(e)


ctx = 256 if max(map(len, REF["prompts"])) + N_OUT <= 256 else 1024
tmp = tempfile.mkdtemp()
p = start(MODEL, ["--host", "127.0.0.1", "--auto-mode", "--ctx", str(ctx), "--kv-pages", str(max(N, 4) * ctx // 256), "--prompt-cache", "off", "--spec", "off",
                  "--agent-dir", tmp + "/a", "--spill-dir", tmp + "/s", *(["--many-agents", "on"] if N > 16 else [])], f"/tmp/today-{N}.srv.log")
try:
    rates, same = [], 0
    for rep in range(2):
        go = threading.Event()
        js = [Job({"prompt": REF["prompts"][i], "n_predict": N_OUT, "temperature": 0, "stream": True, "return_tokens": True, "lane": "background",
                   "cache_prompt": False, "ignore_eos": True}, go) for i in range(N)]
        for j in js:
            j.start()
        go.set()
        for j in js:
            j.join(300)
        a, b = max(j.t[0] for j in js), min(j.t[-1] for j in js)
        rates.append(round(sum(a <= x <= b for j in js for x in j.t) / (b - a)))
        same = sum(j.ids == REF["ref"][i][:N_OUT] for i, j in enumerate(js))
    print(json.dumps({"engine": "today", "agents": N, "tok_s": rates, "same_as_solo": f"{same}/{N}", "ctx": ctx}))
finally:
    stop(p)
