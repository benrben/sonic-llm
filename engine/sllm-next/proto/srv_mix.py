"""One chat beside N background agents (the realistic mix): the chat's write rate alone and with the N writing, the
background agents' rate over the chat's writing, and whether the chat's tokens are the same both times.
env MODELFILE, ENGINE=today (the libllama engine), PORT. usage: srv_mix.py PROMPTS.json N [CHAT_OUT BG_OUT]"""
import json
import os
import sys
import tempfile

sys.path.insert(0, "/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/rollover")
from common import *  # noqa: E402,F403

REF, N = json.load(open(sys.argv[1])), int(sys.argv[2])
CHAT_OUT = int(sys.argv[3]) if len(sys.argv) > 3 else 128
BG_OUT = int(sys.argv[4]) if len(sys.argv) > 4 else 512
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


def body(p, n, lane):
    return {"prompt": p, "n_predict": n, "temperature": 0, "stream": True, "return_tokens": True, "lane": lane, "cache_prompt": False,
            "ignore_eos": True}


def chat_alone():
    go = threading.Event()
    j = Job(body(REF["prompts"][0], CHAT_OUT, "interactive"), go)
    j.start()
    go.set()
    j.join(600)
    return j


def rate(j):
    return round((len(j.t) - 1) / (j.t[-1] - j.t[0]), 1) if len(j.t) > 1 else None


ctx = 1024
tmp = tempfile.mkdtemp()
ENGINE = [] if os.environ.get("ENGINE") == "today" else ["--engine", "next"]
p = subprocess.Popen([BIN, "--model", MODEL, "--port", str(PORT), "--host", "127.0.0.1", *ENGINE, "--ctx", str(ctx), "--kv-pages", str((N + 1) * ctx // 256 + 4),
                      "--prompt-cache", "off", "--agent-dir", tmp + "/a", "--spill-dir", tmp + "/s"],
                     stdout=open("/tmp/srv-mix.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(600):
        try:
            get("/health")
            break
        except Exception:
            time.sleep(0.2)
    alone = chat_alone()
    go = threading.Event()
    bg = [Job(body(q, BG_OUT, "background"), go) for q in REF["prompts"][1:N + 1]]
    [j.start() for j in bg]
    go.set()
    time.sleep(1.5)  # the background agents writing
    chat = chat_alone()
    [j.join(600) for j in bg]
    if not chat.t or not alone.t:
        sys.exit(json.dumps({"errors": [alone.err, chat.err, *[j.err for j in bg if j.err][:3]]}))
    a, b = chat.t[0], chat.t[-1]
    bg_rate = round(sum(a <= x <= b for j in bg for x in j.t) / (b - a)) if b > a else None
    late = sum(j.t[-1] < b for j in bg if j.t)  # background agents done before the chat (the window is not all N)
    print(json.dumps({"engine": "today" if ENGINE == [] else "next", "bg_agents": N, "chat_alone_tok_s": rate(alone), "chat_with_bg_tok_s": rate(chat),
                      "bg_tok_s_during_chat": bg_rate, "bg_done_before_chat": late, "chat_same_tokens": alone.ids == chat.ids,
                      "errors": [j.err for j in [alone, chat, *bg] if j.err]}))
finally:
    stop(p)
