"""The page, start to finish, with real clicks in Chromium (Playwright), on the engine's `--fake` mode: no model, no GPU, no
network. The playground runs as its own process (`python -m sonic_playground`) with a fake model file in a temporary models folder
and a fake Hugging Face on a local port. The real engine binary is SONIC_SERVER_BIN, else ../engine/target/release/sllm-server;
the test is skipped when there is none.

Ports: the playground takes one of 8720-8729."""

from __future__ import annotations

import hashlib
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from conftest import write_gguf
from fakehub import FakeHub, gguf_bytes

sync_api = pytest.importorskip("playwright.sync_api")

PLAYGROUND = Path(__file__).resolve().parents[1]
REAL_ENGINE = Path(os.environ.get("SONIC_SERVER_BIN") or PLAYGROUND.parent / "engine" / "target" / "release" / "sllm-server")


def free_port() -> int:
    for port in range(8720, 8730):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port in 8720-8729")


@pytest.fixture
def playground(tmp_path):
    if not REAL_ENGINE.is_file():
        pytest.skip(f"no engine binary at {REAL_ENGINE} (set SONIC_SERVER_BIN)")
    models = tmp_path / "models"
    write_gguf(models / "qwen3-fake-Q4_K_M.gguf")
    hub = FakeHub()
    content = gguf_bytes(300_000)
    hub.add_file("owner/tiny-GGUF", "tiny-Q4_0.gguf", content)
    hub.search_rows[0]["id"] = "owner/tiny-GGUF"
    hub.tree_pages = [[{"type": "file", "oid": "a", "size": 300_000, "lfs": {"oid": hashlib.sha256(content).hexdigest()}, "path": "tiny-Q4_0.gguf"}]]
    hub.repo_info = {**hub.repo_info, "id": "owner/tiny-GGUF"}
    port = free_port()
    env = {**os.environ, "SONIC_SERVER_BIN": str(PLAYGROUND / "tests" / "fake_engine.py"), "SONIC_REAL_SERVER_BIN": str(REAL_ENGINE),
           "SONIC_MODELS": str(models), "SONIC_PLAYGROUND_DATA": str(tmp_path / "data"), "SONIC_PLAYGROUND_PORT": str(port),
           "HF_ENDPOINT": hub.url, "LLAMA_SERVER_BIN": str(tmp_path / "no-llama-server")}
    proc = subprocess.Popen([sys.executable, "-m", "sonic_playground"], cwd=PLAYGROUND, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    try:
        end = time.monotonic() + 20
        while time.monotonic() < end:
            try:
                urllib.request.urlopen(base + "/api/setup", timeout=1)
                break
            except OSError:
                assert proc.poll() is None, proc.stdout.read().decode()
                time.sleep(0.2)
        yield base, tmp_path
    finally:
        proc.terminate()
        proc.wait(timeout=15)
        hub.close()


def test_chat_swarm_and_burst_with_real_clicks(playground):
    base, tmp_path = playground
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base)
        page.wait_for_function("document.querySelectorAll('#model option').length > 0")
        assert page.locator("#model").input_value() == "qwen3-fake-Q4_K_M"
        assert page.locator("#compare").is_disabled() and "llama.cpp was not found" in page.locator("#compareNote").inner_text()

        page.click("#load")  # load the fake model
        page.wait_for_function("document.getElementById('status').textContent.startsWith('Ready')", timeout=30_000)

        # chat
        page.fill("#input", "Say something short.")
        page.click("#send")
        page.wait_for_selector(".msg.bot .answer .stats b", timeout=30_000)
        assert page.locator(".msg.bot .answer .text").first.inner_text().strip()

        # agent swarm over the sample documents: one agent per file
        page.click("#taskBtn")
        page.click("#taskMenu [data-s=swarm]")
        assert "agent" in page.locator("#plan").inner_text()
        page.click("#send")
        page.wait_for_selector("#swarm .cell.done", timeout=30_000)
        page.wait_for_function("document.querySelectorAll('#swarm .cell').length > 0 && "
                               "document.querySelectorAll('#swarm .cell:not(.done)').length === 0", timeout=60_000)
        page.wait_for_function("document.getElementById('send').textContent === 'Run'", timeout=30_000)
        assert page.locator("#swarm .cell").count() == 9  # the nine sample documents

        # throughput burst with 8 agents
        page.click("#taskBtn")
        page.click("#taskMenu [data-s=burst]")
        page.fill("#agents", "8")
        page.click("#send")
        page.wait_for_function("document.querySelectorAll('#swarm .cell.done').length === 8", timeout=60_000)
        page.wait_for_function("document.getElementById('send').textContent === 'Run'", timeout=30_000)
        assert "wall" in page.locator("#lastCard").inner_text().lower() or page.locator("#lastCard").is_visible()

        # upload a document of your own
        page.click("#attach")
        page.set_input_files("#upload", {"name": "mine.md", "mimeType": "text/markdown", "buffer": b"# Mine\nSome words."})
        page.wait_for_function("document.getElementById('filesList').textContent.includes('uploads')")
        page.click("#filesClose")
        assert (tmp_path / "data" / "documents" / "mine.md").read_text().startswith("# Mine")

        # models: search the (fake) Hub, download, and the model appears in the picker
        page.click("#modelsBtn")
        page.wait_for_function("document.querySelectorAll('#recList .mrow2').length === 8")
        page.fill("#hubSearch", "tiny")
        page.click("#hubResults [data-repo]")
        page.click("#hubFiles [data-file]")
        page.wait_for_function("document.getElementById('dlList').textContent.includes('Done')", timeout=30_000)
        page.click("#modelsClose")
        page.wait_for_function("[...document.querySelectorAll('#model option')].some(o => o.value === 'tiny-Q4_0')")
        assert (tmp_path / "models" / "tiny-Q4_0.gguf").exists()

        # remove it again, with the confirmation
        page.once("dialog", lambda d: d.accept())
        page.click("#modelsBtn")
        page.click("#installedList [data-remove='tiny-Q4_0']")
        page.wait_for_function("![...document.querySelectorAll('#model option')].some(o => o.value === 'tiny-Q4_0')")
        browser.close()
    assert not errors, errors
