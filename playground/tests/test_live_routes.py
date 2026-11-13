"""The routes that need no engine: the page, setup, models, state, documents, history, and the refusals without a session."""

import json

import pytest
from conftest import write_gguf
from fastapi.testclient import TestClient

from sonic_playground import live
from sonic_playground.app import app


@pytest.fixture
def client(cfg, monkeypatch):
    cfg.samples_dir.mkdir()
    (cfg.samples_dir / "a.md").write_text("# A\n" + "word " * 400)
    (cfg.samples_dir / "b.txt").write_text("short")
    monkeypatch.setattr(live, "SESSION", None)
    monkeypatch.setattr("sonic_playground.app.stop_orphans", lambda: [])
    with TestClient(app) as c:
        yield c


def test_the_page_is_served_with_the_product_name(client):
    page = client.get("/").text
    assert "<title>Sonic LLM playground</title>" in page
    for gone in ("SLLM", "Symbiknow", "Classic lab", "/runs"):
        assert gone not in page
    assert client.get("/static/live.js").status_code == 200


def test_state_lists_the_tests_and_the_library(client):
    d = client.get("/api/live/state").json()
    assert d["session"] is None and d["files"]["n"] == 2 and "gpu_lock" not in d
    assert [s["id"] for s in d["scenarios"]] == ["chat", "long", "swarm", "team", "followup", "relay", "turns", "shared",
                                                  "classify", "extract", "summarize", "compress", "load", "burst", "exact"]
    assert client.get("/api/live/scenarios").json()["scenarios"] == d["scenarios"]


def test_setup_says_what_is_missing(client, cfg):
    d = client.get("/api/setup").json()
    assert d["engine"] == {"found": True, "path": str(cfg.server_bin), "build": "cargo build --release -p sllm-server"}
    assert d["llama"] == {"found": False} and d["models_dir"] == str(cfg.models_dir)
    cfg.server_bin.unlink()
    assert client.get("/api/setup").json()["engine"]["found"] is False


def test_models_are_listed_from_the_models_folder(client, cfg):
    assert client.get("/api/models").json() == {"models": [], "dir": str(cfg.models_dir)}
    write_gguf(cfg.models_dir / "qwen3-4b-Q4_K_M.gguf")
    (m,) = client.get("/api/models").json()["models"]
    assert (m["id"], m["arch"], m["quant"]) == ("qwen3-4b-Q4_K_M", "qwen3", "Q4_K_M")


def test_documents_are_text_only_with_token_estimates_and_can_be_added(client):
    fs = client.get("/api/live/files").json()["files"]
    assert [f["path"] for f in fs] == ["samples/a.md", "samples/b.txt"] and fs[0]["est_tokens"] == fs[0]["bytes"] // 4
    r = client.put("/api/live/files/mine.md", content=b"my own notes")
    assert r.json() == {"path": "uploads/mine.md"}
    assert [f["path"] for f in client.get("/api/live/files").json()["files"]][-1] == "uploads/mine.md"
    assert client.put("/api/live/files/run.exe", content=b"x").status_code == 422
    assert client.put("/api/live/files/big.md", content=b"x" * 2_000_001).status_code == 413


def test_a_document_of_your_own_can_be_removed_and_a_sample_cannot(client):
    client.put("/api/live/files/mine.md", content=b"my own notes")
    assert client.delete("/api/live/files/mine.md").json() == {"path": "uploads/mine.md"}
    assert [f["path"] for f in client.get("/api/live/files").json()["files"]] == ["samples/a.md", "samples/b.txt"]
    assert client.delete("/api/live/files/mine.md").status_code == 404  # already gone
    assert client.delete("/api/live/files/a.md").status_code == 404  # a sample is not the user's to remove
    assert client.get("/api/live/files").json()["files"][0]["path"] == "samples/a.md"


def test_history_is_newest_first_and_a_damaged_line_is_skipped_and_logged(client, cfg, monkeypatch):
    lines = []
    monkeypatch.setattr(live, "log", lines.append)
    cfg.runs_file.parent.mkdir(parents=True)
    cfg.runs_file.write_text("\n".join([json.dumps({"id": "r0"}), "{not json", json.dumps({"id": "r1"}), json.dumps({"id": "r2"})]))
    assert [r["id"] for r in client.get("/api/live/history").json()["runs"]] == ["r2", "r1", "r0"]
    assert client.get("/api/live/history/r1").json()["id"] == "r1"
    assert client.get("/api/live/history/nope").status_code == 404
    assert "line 2" in lines[0]


def test_a_run_without_a_model_is_refused(client):
    r = client.post("/api/live/run", json={"scenario": "chat", "message": "hi"})
    assert r.status_code == 409 and "no model" in r.json()["detail"]


def test_loading_a_model_is_refused_with_a_reason(client, cfg):
    write_gguf(cfg.models_dir / "qwen3-4b.gguf")
    post = lambda **kw: client.post("/api/live/session", json={"model_id": "qwen3-4b", **kw})
    r = post(compare=True)
    assert r.status_code == 409 and "llama-server was not found" in r.json()["detail"]
    assert post(model_id="nope").status_code == 404
    write_gguf(cfg.models_dir / "bare-model.gguf", arch="llama")  # a model with no chat template
    r = client.post("/api/live/session", json={"model_id": "bare-model"})
    assert r.status_code == 422 and "no chat template" in r.json()["detail"]
    cfg.server_bin.unlink()
    r = post()
    assert r.status_code == 409 and "not built" in r.json()["detail"] and "cargo build --release" in r.json()["detail"]


def test_stopping_the_playground_stops_the_loaded_models_engines(cfg, monkeypatch):
    stopped = []

    class S:
        model = {"id": "m"}
        run_task = None

        async def stop(self):
            stopped.append(1)

    monkeypatch.setattr(live, "SESSION", S())
    monkeypatch.setattr("sonic_playground.app.stop_orphans", lambda: [])
    with TestClient(app):
        assert stopped == []
    assert stopped == [1] and live.SESSION is None
