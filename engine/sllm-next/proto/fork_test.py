"""Agent fork (POST /v1/agents {"from_agent": id}) on sllm-server --engine next: a source agent reads a prompt and thinks
silently; forks of it answer the same question bit for bit as the source does, and as a fresh agent that reads and thinks
the same way; a fork's later turns do not touch the source. Prints one JSON line; exit 1 on a failed check.

  SLLM_BIN=... MODEL=... python3 fork_test.py
"""
import json, os, subprocess, sys, tempfile, time, urllib.error, urllib.request

BIN = os.environ.get("SLLM_BIN", "/Users/benreich/Projects/crazyidea/sllm-wt/shared-target/release/sllm-server")
MODEL = os.environ.get("MODEL", "/Users/benreich/Projects/crazyidea/models/gguf/Qwen3-4B-Q4_0.gguf")
PORT = int(os.environ.get("PORT", "8793"))


def http(method, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            t = r.read()
            return r.status, (json.loads(t) if t else {})
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def chat(u):
    return f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def ids(t):
    return http("POST", "/tokenize", {"content": t, "add_special": False, "parse_special": True})[1]["tokens"]


def agent(body):
    s, a = http("POST", "/v1/agents", {"lane": "background", **body})
    assert s in (200, 201), (s, a)
    return a["id"]


def turn(aid, prompt, **kw):
    s, r = http("POST", f"/v1/agents/{aid}/turn", {"prompt": prompt, "temperature": 0, "stream": False, "return_tokens": True, **kw})
    assert s == 200, (s, r)
    return r.get("tokens") or []


def ctx(aid):
    return http("GET", f"/v1/agents/{aid}")[1].get("context_tokens")


tmp = tempfile.mkdtemp()
p = subprocess.Popen([BIN, "--model", MODEL, "--port", str(PORT), "--host", "127.0.0.1", "--engine", "next", "--ctx", "4096", "--kv-pages", "200",
                      "--prompt-cache", "off", "--agent-dir", tmp + "/a", "--spill-dir", tmp + "/s"], stdout=open(tmp + "/srv.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(600):
        try:
            if http("GET", "/health")[0] == 200:
                break
        except Exception:
            time.sleep(0.2)
    read = ids(chat("Here are facts: the red box holds 7 apples, the blue box holds 3 more than the red one, the green box twice the blue one. Think about them."))
    q = ids(chat("How many apples are in the green box? Answer with the number and one sentence."))
    q2 = ids(chat("Now name the box with the fewest apples."))
    checks = {}
    src = agent({})
    turn(src, read, latent_steps=24)
    n_src = ctx(src)
    f1, f2 = agent({"from_agent": src}), agent({"from_agent": src})
    checks["fork_context_equal"] = ctx(f1) == ctx(f2) == n_src
    a_f1 = turn(f1, q, n_predict=48)
    a_f2 = turn(f2, q, n_predict=48)
    checks["forks_same_answer"] = a_f1 == a_f2 and len(a_f1) > 0
    turn(f1, q2, n_predict=24)  # a fork's later turn: the source keeps its own context
    checks["source_untouched"] = ctx(src) == n_src
    a_src = turn(src, q, n_predict=48)
    checks["source_same_answer"] = a_src == a_f1
    fresh = agent({})
    turn(fresh, read, latent_steps=24)
    checks["fresh_same_answer"] = turn(fresh, q, n_predict=48) == a_f1
    s_busy, _ = http("POST", "/v1/agents", {"from_agent": "ag_999"})
    checks["unknown_source_refused"] = s_busy == 404
    s_both, _ = http("POST", "/v1/agents", {"from_agent": src, "system_ids": [1, 2]})
    checks["fork_with_system_refused"] = s_both in (400, 422)
    print(json.dumps({"ok": all(checks.values()), **checks, "context": n_src, "answer_tokens": len(a_f1)}))
    sys.exit(0 if all(checks.values()) else 1)
finally:
    p.terminate()
    p.wait()
