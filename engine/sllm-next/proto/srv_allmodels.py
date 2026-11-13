"""--engine next on every GGUF in models/gguf: which engine runs it (next, or the libllama engine with the reason) and a
short completion. usage: srv_allmodels.py"""
import glob, json, os, subprocess, sys, tempfile, time, urllib.request
BIN = "/Users/benreich/Projects/crazyidea/sllm-wt/shared-target/release/sllm-server"
PORT = 19014
def post(path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=300))
def get(path):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=30))
for f in sorted(glob.glob("/Users/benreich/Projects/crazyidea/models/gguf/*.gguf")):
    name = os.path.basename(f)
    tmp = tempfile.mkdtemp()
    log = f"/tmp/allmodels-{name}.log"
    p = subprocess.Popen([BIN, "--model", f, "--port", str(PORT), "--host", "127.0.0.1", "--engine", "next", "--ctx", "1024", "--kv-pages", "8",
                          "--prompt-cache", "off", "--agent-dir", tmp + "/a"], stdout=open(log, "w"), stderr=subprocess.STDOUT)
    res = {"model": name}
    try:
        t0 = time.time()
        while time.time() - t0 < 240:
            if p.poll() is not None:
                res["error"] = "exited: " + open(log).read()[-200:].replace("\n", " ")
                break
            try:
                get("/health"); break
            except Exception:
                time.sleep(0.5)
        if "error" not in res:
            caps = get("/caps")
            eng = caps.get("engine")
            res["engine"] = eng.get("engine") if isinstance(eng, dict) else str(eng)[:20]
            ids = post("/tokenize", {"content": "The capital of France is", "add_special": True})["tokens"]
            r = post("/completion", {"prompt": ids, "n_predict": 8, "temperature": 0})
            res["text"] = r.get("content", "")[:40]
            why = [l for l in open(log).read().splitlines() if "--engine next:" in l]
            if why: res["fallback_reason"] = why[0].split(": ", 2)[-1][:110]
    except Exception as e:
        res["error"] = repr(e)[:150]
    finally:
        p.terminate()
        try: p.wait(30)
        except Exception: p.kill()
    print(json.dumps(res), flush=True)
