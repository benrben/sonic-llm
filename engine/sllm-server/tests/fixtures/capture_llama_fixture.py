#!/usr/bin/env python3
"""Capture real llama-server /completion responses as contract fixtures for sllm-server.

Run it under the GPU lock (rule 094), waiting for the lock if it is held:

    python3 engine/scripts/with_gpu_lock.py --owner sllm-playground-capture-llama-fixture --timeout 7200 -- \
        python3 engine/sllm-server/tests/fixtures/capture_llama_fixture.py

It starts research/inference_runtime/vendor/llama-bin/llama-server on qwen3-0.6b-Q4_0 (-ngl 99 -fa on -c 512 -np 1),
tokenizes a fixed sentence with its /tokenize, and records /completion with `prompt` = token ids, `n_predict` 8,
`temperature` 0, `ignore_eos` true, `return_tokens` true, once streamed and once not. The response BODY is saved byte for
byte (the SSE framing is part of the contract) and the status line and headers go into a sidecar JSON.
Outputs, next to this script: llama_completion_stream.sse, llama_completion_nonstream.json, llama_capture_meta.json.
"""
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = Path("/Users/benreich/Projects/crazyidea/research/inference_runtime")
SERVER = RESEARCH / "vendor" / "llama-bin" / "llama-server"
GGUF = RESEARCH / "converted" / "qwen3-0.6b-Q4_0.gguf"
TEXT = "The quick brown fox jumps over the lazy dog. Then it"
N_PREDICT = 8


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def request(port, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    data = None if body is None else json.dumps(body)
    c.request(method, path, body=data, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    raw = r.read()  # http.client de-chunks; the SSE events themselves are the framing under test
    return r.status, r.reason, dict(r.getheaders()), raw


def main():
    lock = json.load(open("/tmp/sllm-gpu.lock"))
    assert lock["pid"] == os.getppid() or lock["pid"] == os.getpid() or lock["owner"].startswith("sllm-playground-"), lock
    port = free_port()
    argv = [str(SERVER), "-m", str(GGUF), "--host", "127.0.0.1", "--port", str(port), "-ngl", "99", "-fa", "on", "-c", "512", "-np", "1"]
    log = tempfile.TemporaryFile()
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(600):
            try:
                st, *_ = request(port, "GET", "/health")
                if st == 200:
                    break
            except OSError:
                pass
            if proc.poll() is not None:
                raise SystemExit(f"llama-server exited early rc={proc.returncode}")
            time.sleep(0.2)
        else:
            raise SystemExit("llama-server not ready")
        _, _, _, raw = request(port, "POST", "/tokenize", {"content": TEXT, "add_special": False, "parse_special": False})
        ids = json.loads(raw)["tokens"]
        base = {"prompt": ids, "n_predict": N_PREDICT, "temperature": 0.0, "cache_prompt": False, "ignore_eos": True,
                "return_tokens": True, "return_progress": False}
        st, reason, hdr, sse = request(port, "POST", "/completion", {**base, "stream": True})
        assert st == 200, (st, sse[:300])
        st2, reason2, hdr2, js = request(port, "POST", "/completion", {**base, "stream": False})
        assert st2 == 200, (st2, js[:300])
        # a few error shapes, so the contract tests can compare status codes and the error body keys
        errs = {}
        for name, body in {"missing_prompt": {"n_predict": 4}, "bad_n_predict_type": {"prompt": ids, "n_predict": "x"}}.items():
            s, r, h, b = request(port, "POST", "/completion", body)
            errs[name] = {"status": s, "content_type": h.get("Content-Type"), "body": b.decode(errors="replace")}
        s, r, h, b = request(port, "GET", "/nope")
        errs["unknown_path"] = {"status": s, "content_type": h.get("Content-Type"), "body": b.decode(errors="replace")}
        _, _, _, vraw = request(port, "GET", "/health")
        (HERE / "llama_completion_stream.sse").write_bytes(sse)
        (HERE / "llama_completion_nonstream.json").write_bytes(js)
        meta = {
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "server": str(SERVER), "argv": argv, "model": str(GGUF),
            "request": {**base, "stream": True}, "text": TEXT,
            "stream_response": {"status": st, "reason": reason, "headers": hdr},
            "nonstream_response": {"status": st2, "reason": reason2, "headers": hdr2},
            "health_body": vraw.decode(), "errors": errs,
        }
        (HERE / "llama_capture_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print("captured", len(sse), "bytes of SSE,", len(js), "bytes of JSON; prompt ids:", len(ids))
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


if __name__ == "__main__":
    sys.exit(main())
