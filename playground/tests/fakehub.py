"""A small local HTTP server that answers like the Hugging Face Hub: the search and tree APIs (response shapes as recorded from
huggingface.co) and file downloads with or without Range support. No real network is used by any test."""

from __future__ import annotations

import json
import re
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


def gguf_bytes(size: int, seed: int = 7) -> bytes:
    """A fake model file: a valid GGUF header (qwen3, no tensors), then bytes that tell their own position (so a wrong resume is
    caught by the content)."""
    def text(x: str) -> bytes:
        return struct.pack("<Q", len(x)) + x.encode()

    head = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + text("general.architecture") + struct.pack("<I", 8) + text("qwen3")
    return head + bytes((i * 31 + seed) % 251 for i in range(size - len(head)))


class FakeHub:
    def __init__(self):
        self.files: dict[str, bytes] = {}  # "owner/repo/file.gguf" -> content
        self.ranges = True  # honour Range requests
        self.delay = 0.0  # seconds between 64 KB pieces of a download
        self.token: str | None = None  # when set, downloads and gated repos need `Authorization: Bearer <token>`
        self.gated_repos: set[str] = set()
        self.rate_limited = False
        self.hang = 0.0  # seconds every API answer waits
        self.search_rows = [{"_id": "x", "id": "unsloth/Qwen3-4B-GGUF", "likes": 100, "private": False, "downloads": 5000,
                             "tags": ["gguf", "qwen3", "license:apache-2.0"], "pipeline_tag": "text-generation", "library_name": "gguf",
                             "createdAt": "2026-01-01T00:00:00.000Z", "modelId": "unsloth/Qwen3-4B-GGUF"}]
        self.repo_info = {"id": "unsloth/Qwen3-4B-GGUF", "private": False, "gated": False, "downloads": 5000, "likes": 100,
                          "gguf": {"total": 4022468096, "architecture": "qwen3", "context_length": 40960}}
        self.tree_pages: list[list[dict]] = [[]]
        self.requests: list[dict] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def add_file(self, repo: str, name: str, content: bytes) -> None:
        self.files[f"{repo}/{name}"] = content

    def _handler(self):
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                url = urlparse(self.path)
                hub.requests.append({"path": url.path, "query": url.query, "headers": dict(self.headers)})
                if url.path.startswith("/api/"):
                    return self.api(url)
                return self.resolve(url.path)

            def send_json(self, status, body, headers=()):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in headers:
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def authorised(self) -> bool:
                return hub.token is None or self.headers.get("Authorization") == f"Bearer {hub.token}"

            def api(self, url):
                if hub.hang:
                    time.sleep(hub.hang)
                if hub.rate_limited:
                    return self.send_json(429, {"error": "Rate limit"}, [("Retry-After", "30")])
                if url.path == "/api/models":
                    return self.send_json(200, hub.search_rows)
                m = re.fullmatch(r"/api/models/(.+?)(/tree/main)?", url.path)
                repo = m.group(1)
                if repo in hub.gated_repos and self.headers.get("Authorization") != "Bearer good":
                    return self.send_json(401, {"error": "Invalid credentials in Authorization header"})
                if m.group(2):
                    page = int(parse_qs(url.query).get("cursor", ["0"])[0])
                    link = [("Link", f'<{hub.url}{url.path}?recursive=true&cursor={page + 1}>; rel="next"')] if page + 1 < len(hub.tree_pages) else []
                    return self.send_json(200, hub.tree_pages[page], link)
                return self.send_json(200, hub.repo_info)

            def resolve(self, path):
                key = path.replace("/resolve/main", "").lstrip("/")
                if hub.token is not None and not self.authorised():
                    return self.send_json(401, {"error": "Access to model is restricted"})
                content = hub.files.get(key)
                if content is None:
                    return self.send_json(404, {"error": "Entry not found"})
                start, status = 0, 200
                rng = re.fullmatch(r"bytes=(\d+)-", self.headers.get("Range", "")) if hub.ranges else None
                if rng:
                    start, status = int(rng.group(1)), 206
                    if start >= len(content):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(content)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                body = content[start:]
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{len(content) - 1}/{len(content)}")
                self.end_headers()
                try:
                    for i in range(0, len(body), 65536):
                        self.wfile.write(body[i:i + 65536])
                        self.wfile.flush()
                        if hub.delay:
                            time.sleep(hub.delay)
                except (BrokenPipeError, ConnectionResetError):  # the client cancelled
                    pass

        return Handler
