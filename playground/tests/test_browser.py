"""The page, start to finish, with real clicks in Chromium (Playwright), on the engine's `--fake` mode: no model, no GPU, no
network. The playground runs as its own process (`python -m sonic_playground`) with a fake model file in a temporary models folder
and a fake Hugging Face on a local port; the comparison runs against the same fake engine standing in for llama-server. The real
engine binary is SONIC_SERVER_BIN, else ../engine/target/release/sllm-server; the test is skipped when there is none.

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


def _start(tmp_path, with_llama: bool):
    """The playground as a process on the fake engine, a fake Hub, and (when asked) the fake engine as llama-server."""
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
    fake = str(PLAYGROUND / "tests" / "fake_engine.py")
    env = {**os.environ, "SONIC_SERVER_BIN": fake, "SONIC_REAL_SERVER_BIN": str(REAL_ENGINE),
           "SONIC_MODELS": str(models), "SONIC_PLAYGROUND_DATA": str(tmp_path / "data"), "SONIC_PLAYGROUND_PORT": str(port),
           "HF_ENDPOINT": hub.url, "LLAMA_SERVER_BIN": fake if with_llama else str(tmp_path / "no-llama-server")}
    proc = subprocess.Popen([sys.executable, "-m", "sonic_playground"], cwd=PLAYGROUND, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    end = time.monotonic() + 20
    while time.monotonic() < end:
        try:
            urllib.request.urlopen(base + "/api/setup", timeout=1)
            return proc, hub, base
        except OSError:
            assert proc.poll() is None, proc.stdout.read().decode()
            time.sleep(0.2)
    raise RuntimeError("the playground did not start")


def _stop(proc, hub):
    proc.terminate()
    proc.wait(timeout=15)
    hub.close()


@pytest.fixture
def playground(tmp_path):
    proc, hub, base = _start(tmp_path, with_llama=False)
    try:
        yield base, tmp_path
    finally:
        _stop(proc, hub)


@pytest.fixture
def compared(tmp_path):
    proc, hub, base = _start(tmp_path, with_llama=True)
    try:
        yield base, tmp_path
    finally:
        _stop(proc, hub)


def _watch(page) -> list[str]:
    """Every console error and uncaught page error from here on."""
    errors: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    return errors


def _ready(page, base):
    page.goto(base)
    page.wait_for_function("document.querySelectorAll('#model option').length > 0")
    assert page.locator("#model").input_value() == "qwen3-fake-Q4_K_M"


def _load(page, compare: bool = False):
    if compare:
        page.click(".switch")
    page.click("#load")
    page.wait_for_function("document.getElementById('statusText').textContent.startsWith('Ready')", timeout=30_000)


def _pick(page, name):
    page.click(f"#tests .pick:has-text('{name}')")


def _wait_idle(page, timeout=60_000):
    page.wait_for_function("document.getElementById('send').textContent === 'Run'", timeout=timeout)


def test_chat_swarm_and_burst_with_real_clicks(playground):
    base, tmp_path = playground
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        errors = _watch(page)
        _ready(page, base)

        # the first screen teaches: what this is, three steps, and why the comparison is off
        assert "Watch many AI agents run on your Mac" in page.locator("#intro").inner_text()
        assert page.locator("#intro .step").count() == 3
        assert page.locator("#tests input[name=test]").count() == 15
        assert page.locator("#compare").is_disabled()
        page.click("#compareWhy")
        assert "llama-server" in page.locator("#comparePanel").inner_text()
        page.keyboard.press("Escape")
        assert page.locator("#comparePanel").is_hidden()

        _load(page)  # load the fake model

        # chat: the only controls are the question, the documents and the answer level
        page.fill("#input", "Say something short.")
        assert page.locator("#agentsRow").is_hidden() and page.locator("#levelBox").is_visible()
        page.click("#send")
        page.wait_for_function("document.getElementById('runState').textContent.includes('Done')", timeout=30_000)
        assert page.locator("#messages .answer .text").first.inner_text().strip()
        tiles = page.locator("#tiles").inner_text()
        assert "Total time" in tiles and "First word" in tiles and "Write speed" in tiles and "faster" not in tiles
        assert "One live run on this machine, not a benchmark." in page.locator("#run").inner_text()
        assert page.locator("#intro").is_hidden()

        # agent swarm over the sample documents: one agent per document
        _pick(page, "Agent swarm")
        assert page.locator("#agentsRow").is_visible() and page.locator("#askBox").is_visible()
        assert "agent" in page.locator("#plan").inner_text()
        page.click("#send")
        page.wait_for_selector("#swarm .cell.done", timeout=30_000)
        page.wait_for_function("document.querySelectorAll('#swarm .cell').length > 0 && "
                               "document.querySelectorAll('#swarm .cell:not(.done)').length === 0", timeout=60_000)
        _wait_idle(page, 30_000)
        assert page.locator("#swarm .cell").count() == 8  # the default of eight agents, on the biggest samples
        assert "Read speed" in page.locator("#tiles").inner_text()

        # throughput burst with 8 agents: only the number of agents is asked
        _pick(page, "Throughput burst")
        assert page.locator("#askBox").is_hidden() and page.locator("#docsBox").is_hidden()
        page.fill("#agents", "8")
        page.click("#send")
        page.wait_for_function("document.querySelectorAll('#swarm .cell.done').length === 8", timeout=60_000)
        _wait_idle(page, 30_000)
        page.click("#allNums summary")
        assert "Total time" in page.locator("#last").inner_text()

        # your own document: add it, pick it, remove it (after being asked once more)
        page.click("#tests .pick:has-text('Classify')")
        page.click("#attach")
        page.set_input_files("#upload", {"name": "mine.md", "mimeType": "text/markdown", "buffer": b"# Mine\nSome words."})
        page.wait_for_function("document.getElementById('filesList').textContent.includes('mine.md')")
        assert (tmp_path / "data" / "documents" / "mine.md").read_text().startswith("# Mine")
        page.click("[data-ask='uploads/mine.md']")
        assert (tmp_path / "data" / "documents" / "mine.md").exists()  # not yet: it asked first
        page.click("[data-del='uploads/mine.md']")
        page.wait_for_function("!document.getElementById('filesList').textContent.includes('mine.md')")
        assert not (tmp_path / "data" / "documents" / "mine.md").exists()
        page.keyboard.press("Escape")  # a dialog closes on Escape and gives the focus back
        assert not page.locator("#files").evaluate("d => d.open")
        assert page.evaluate("document.activeElement.id") == "attach"

        # earlier runs are in the drawer; one opens as a saved result
        page.click("#historyBtn")
        page.wait_for_function("document.querySelectorAll('#historyList .hitem').length >= 3")
        page.locator("#historyList .hitem button").first.click()
        assert "Saved run" in page.locator("#runState").inner_text()
        assert not page.locator("#history").evaluate("d => d.open")

        # the theme switch flips and keeps its choice
        before = page.evaluate("document.documentElement.getAttribute('data-theme')")
        page.click("#theme")
        assert page.evaluate("document.documentElement.getAttribute('data-theme')") in ("light", "dark") != before

        # the same page at the width of a phone: nothing sticks out sideways
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        browser.close()
    assert not errors, errors


def test_models_panel_search_download_load_and_remove(playground):
    base, tmp_path = playground
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        errors = _watch(page)
        _ready(page, base)
        page.click("#modelsBtn")
        page.wait_for_selector("#installedList .mrow2")
        assert page.locator("#installedList .mrow2").count() == 1  # the fake model on this Mac

        # recommended: eight measured models, each with its fit and engine hints as words
        page.click("#tabRec")
        page.wait_for_function("document.querySelectorAll('#recList .mrow2').length === 8")
        assert "Memory:" in page.locator("#recList").inner_text() and "Engine:" in page.locator("#recList").inner_text()

        # search the (fake) Hub, open the repository, download its file
        page.click("#tabHub")
        page.fill("#hubSearch", "tiny")
        page.click("#hubResults [data-repo]")
        page.click("#hubFiles [data-file]")
        page.wait_for_function("document.getElementById('dlStrip').textContent.includes('Done')", timeout=30_000)
        page.click("#modelsClose")
        page.wait_for_function("[...document.querySelectorAll('#model option')].some(o => o.value === 'tiny-Q4_0')")
        assert (tmp_path / "models" / "tiny-Q4_0.gguf").exists()

        # remove it again: the button asks once more before the file goes
        page.click("#modelsBtn")
        page.click("#tabInstalled")
        page.click("#installedList [data-ask='tiny-Q4_0']")
        assert (tmp_path / "models" / "tiny-Q4_0.gguf").exists()
        page.click("#installedList [data-remove='tiny-Q4_0']")
        page.wait_for_function("![...document.querySelectorAll('#model option')].some(o => o.value === 'tiny-Q4_0')")
        assert not (tmp_path / "models" / "tiny-Q4_0.gguf").exists()

        # load the other model from the panel itself
        page.click("#installedList [data-load='qwen3-fake-Q4_K_M']")
        page.wait_for_function("document.getElementById('statusText').textContent.startsWith('Ready')", timeout=30_000)
        browser.close()
    assert not errors, errors


def test_comparison_shows_both_engines_and_the_ratio_in_words(compared):
    base, _ = compared
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        errors = _watch(page)
        _ready(page, base)
        assert page.locator("#compare").is_enabled() and page.locator("#compareWhy").is_hidden()
        _load(page, compare=True)
        _pick(page, "Agent swarm")
        page.click("#send")
        page.wait_for_function("document.getElementById('runState').textContent.includes('Done')", timeout=90_000)
        assert page.locator("#messages .answer").count() == 2
        assert page.locator("#swarm .swarmrow").count() == 2
        tiles = page.locator("#tiles").inner_text()
        assert "faster" in tiles or "slower" in tiles or "about the same" in tiles
        speed = " ".join(page.locator("#tiles .tile:not(:has-text('Correct'))").all_inner_texts())
        assert "%" not in speed  # a ratio in words, never a bare percentage (the score tile has its percentage)
        score = page.locator("#tiles .tile:has-text('Correct')").inner_text()
        assert "Sonic LLM: 0 of 8" in " ".join(score.split()) and "llama.cpp: 0 of 8" in " ".join(score.split())  # the fake engine writes token ids: all wrong, for both
        assert page.locator("#runGrade").inner_text() == "Both engines got 0 of 8 right."
        assert page.locator("#runVerdict").is_visible()
        browser.close()
    assert not errors, errors


def test_graded_swarm_shows_the_score_the_answers_and_the_wrong_only_filter(playground):
    base, _ = playground
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        errors = _watch(page)
        _ready(page, base)
        _load(page)
        _pick(page, "Agent swarm")

        # the most agents that fit, from the memory plan: a button next to the field sets it (the defaults stay modest)
        page.wait_for_function("document.getElementById('agentRoom').textContent.includes('up to')")
        assert page.locator("#agents").input_value() == "8"
        most = int(page.locator("#agentRoom").inner_text().split("up to ")[1].split(" ")[0])
        assert 8 < most <= 256 and "Graded" in page.locator("#plan").inner_text()

        # the documents window: shortcuts and a count
        page.click("#attach")
        page.click("#pickFirst [data-first='32']")
        assert "32 documents picked" in page.locator("#filesSel").inner_text()
        assert page.locator("#pickAll").inner_text() == "Pick all 256"
        page.keyboard.press("Escape")
        assert page.locator("#nfiles").inner_text() == "32 documents picked" and page.locator("#agents").input_value() == "32"

        page.click("#send")
        page.wait_for_function("document.querySelectorAll('#swarm .cell').length === 32 && "
                               "document.querySelectorAll('#swarm .cell:not(.done)').length === 0", timeout=60_000)
        _wait_idle(page, 60_000)

        # the score: all 32 wrong (the fake engine writes token ids), said in words and with the cross in every tile
        
        score = " ".join(page.locator("#tiles .tile:has-text('Correct')").inner_text().split())
        assert "0 of 32" in score and "0%" in score and "32 wrong" in score and "0 no answer" in score
        assert page.locator("#runGrade").inner_text() == "Sonic LLM got 0 of 32 right."
        assert page.locator("#swarm .cell.v-bad").count() == 32 and page.locator("#legendGrade").is_visible()

        # an agent's row opens to its document, question, answer, expected answer and verdict
        page.locator("#messages .arow summary").first.click()
        facts = page.locator("#messages .arow[open] .facts").first.inner_text()
        assert "Document" in facts and "Question" in facts and "Expected" in facts and "Verdict" in facts and "Wrong" in facts
        assert page.locator("#messages .arow[open] pre").first.inner_text().strip()

        # the filter: wrong only keeps all 32, no answer keeps none
        page.click("#messages .filter label:has-text('Wrong only')")
        assert page.locator("#messages .arow:visible").count() == 32
        page.click("#messages .filter label:has-text('No answer')")
        assert page.locator("#messages .arow:visible").count() == 0 and "No answer matches" in page.locator("#messages .agentlist").evaluate("e => e.dataset.empty")

        # more than a page: 64 agents show 50 rows and a pager
        page.click("#attach")
        page.click("#pickFirst [data-first='64']")
        page.keyboard.press("Escape")
        page.click("#send")
        page.wait_for_function("document.querySelectorAll('#swarm .cell').length === 64 && "
                               "document.querySelectorAll('#swarm .cell:not(.done)').length === 0", timeout=90_000)
        _wait_idle(page, 60_000)
        assert page.locator("#messages .arow:visible").count() == 50 and "1–50 of 64" in page.locator("#messages .pg").inner_text()
        page.click("#messages [data-page='1']")
        assert page.locator("#messages .arow:visible").count() == 14

        # Max: one agent per document tests pick that many documents
        page.click("#agentsMax")
        assert page.locator("#nfiles").inner_text() == f"{min(most, 256)} documents picked"

        # earlier runs show their score
        page.click("#historyBtn")
        page.wait_for_function("document.querySelectorAll('#historyList .hitem').length >= 2")
        assert "got 0 of 64 right" in page.locator("#historyList .hitem").first.inner_text()
        browser.close()
    assert not errors, errors
