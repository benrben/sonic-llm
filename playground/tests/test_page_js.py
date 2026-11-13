"""The page's run and session state in node (tests/js: plain node, no packages): the live page's reducer and its boxes, and the
shared event stream (events.js)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

JS = Path(__file__).resolve().parent / "js"


def test_page_state_in_node():
    node = shutil.which("node")
    assert node, "node is not installed: the page tests in tests/js need it (brew install node)"
    files = sorted(str(p) for p in JS.glob("*.test.js"))
    assert files, f"no node tests in {JS}"
    r = subprocess.run([node, "--test", *files], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-2000:]
