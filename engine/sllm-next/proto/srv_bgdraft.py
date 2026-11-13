"""Background agents drafting (SLLM_NEXT_BG_DRAFT, on by default) against not drafting (=0), same build, interleaved: N background agents at once,
total write tok/s and every agent's tokens (they must be equal with the switch on and off). One server per run (the switch is
read at start). Workloads: doc = the 1k-token documents of docref.json (the replies copy a lot: drafts pay), story = the
short prompts of ref128.json (little copying: must not get slower). usage: srv_bgdraft.py OUT.jsonl REPS "N,N,.." "doc,story"
env: SLLM_BIN (the server), MODELFILE, N_OUT (128), BG_ROWS / BG_K0 (SLLM_NEXT_BG_ROWS / SLLM_NEXT_BG_K0 for the on runs), PORT.
Each line of OUT: rep, agents, workload, mode, tok_s, tokens_equal_to_the_first_off_run, bg counters. The Mac heats after ~40 s,
so the order alternates off/on per rep; judge the medians."""
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
BIN = os.environ.get("SLLM_BIN", "/Users/benreich/Projects/crazyidea/sllm-wt/target-bgdraft/release/sllm-server")
N_OUT = int(os.environ.get("N_OUT", "128"))
OUT, REPS = sys.argv[1], int(sys.argv[2])
NS = [int(x) for x in sys.argv[3].split(",")]
WORK = sys.argv[4].split(",")
DOCS = json.load(open(f"{HERE}/{G}docref.json"))["prompts"]
STORY = json.load(open(f"{HERE}/{G}ref128.json"))["prompts"]


class Job(threading.Thread):
    def __init__(self, body, go):
        super().__init__(daemon=True)
        self.body, self.go, self.t, self.ids, self.err = body, go, [], [], None

    def run(self):
        self.go.wait()
        try:
            c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=600)
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
    js = [Job({"prompt": p, "n_predict": N_OUT, "temperature": 0, "stream": True, "return_tokens": True, "lane": "background",
               "cache_prompt": False, "ignore_eos": True}, go) for p in prompts]
    [j.start() for j in js]
    go.set()
    [j.join(900) for j in js]
    errs = [j.err for j in js if j.err]
    ok = [j for j in js if j.t]
    a, b = max(j.t[0] for j in ok), min(j.t[-1] for j in ok)  # all N writing
    rate = round(sum(a <= x <= b for j in ok for x in j.t) / (b - a), 1) if b > a else None
    return [j.ids for j in js], rate, errs


def one_run(mode, n, work):
    ctx = 2048
    tmp = tempfile.mkdtemp()
    env = dict(os.environ)
    if mode == "on":
        env["SLLM_NEXT_BG_DRAFT"] = "1"
        env["SLLM_NEXT_BG_ROWS"] = os.environ.get("BG_ROWS", "16")
        env["SLLM_NEXT_BG_K0"] = os.environ.get("BG_K0", "2")
    else:
        env["SLLM_NEXT_BG_DRAFT"] = "0"  # (on by default)
    p = subprocess.Popen([BIN, "--model", MODEL, "--port", str(PORT), "--host", "127.0.0.1", "--engine", "next", "--ctx", str(ctx),
                          "--kv-pages", str(n * ctx // 256 + 4), "--prompt-cache", "off", "--agent-dir", tmp + "/a", "--spill-dir", tmp + "/s"],
                         stdout=open("/tmp/srv-bgdraft.log", "w"), stderr=subprocess.STDOUT, env=env)
    try:
        for _ in range(600):
            try:
                get("/health")
                break
            except Exception:
                time.sleep(0.2)
        src = DOCS if work == "doc" else STORY
        prompts = [src[i % len(src)] for i in range(n)]
        batch([STORY[-1]])  # warm up (kernels compile, pages) on a prompt the runs do not take: not one shared drafts could replay
        ids, rate, errs = batch(prompts)
        stats = None
        try:
            stats = (get("/metrics").get("spec") or get("/v1/stats").get("spec"))
        except Exception:
            pass
        return ids, rate, errs, stats
    finally:
        stop(p)


ref = {}  # (n, work) -> the tokens of the first off run
rows = []
for rep in range(REPS):
    for n in NS:
        for work in WORK:
            for mode in (["off", "on"] if rep % 2 == 0 else ["on", "off"]):
                ids, rate, errs, stats = one_run(mode, n, work)
                first = ref.setdefault((n, work), ids) if mode == "off" else ref.get((n, work))
                same = None if first is None else f"{sum(a == b for a, b in zip(ids, first))}/{n}"
                row = {"rep": rep, "agents": n, "work": work, "mode": mode, "tok_s": rate, "same_as_off": same, "errors": errs[:2], "spec": stats}
                print(json.dumps(row), flush=True)
                rows.append(row)
                open(OUT, "a").write(json.dumps(row) + "\n")
for n in NS:
    for work in WORK:
        med = {m: statistics.median([r["tok_s"] for r in rows if r["agents"] == n and r["work"] == work and r["mode"] == m and r["tok_s"]] or [0])
               for m in ("off", "on")}
        print(f"summary agents={n} {work}: off {med['off']} on {med['on']} tok/s")
