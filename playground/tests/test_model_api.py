"""The model routes: recommended list, Hub search and repository files, downloads with progress and cancel, removal, offline mode."""

import time

import pytest
from conftest import write_gguf
from fakehub import gguf_bytes
from fastapi.testclient import TestClient

from sonic_playground import catalog, live, settings
from sonic_playground.app import app
from test_hub import TREE


@pytest.fixture
def client(hub_server, monkeypatch):
    monkeypatch.setattr(live, "SESSION", None)
    monkeypatch.setattr("sonic_playground.app.stop_orphans", lambda: [])
    with TestClient(app) as c:
        yield c


def wait_for(client, state, seconds=10):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        d = client.get("/api/downloads").json()["downloads"][0]
        if d["state"] == state:
            return d
        time.sleep(0.02)
    raise AssertionError(f"never reached {state}: {d}")


def test_the_recommended_list_has_the_eight_benchmarked_models_with_sizes_and_memory(client, cfg):
    write_gguf(cfg.models_dir / "Qwen3-4B-Q4_0.gguf")
    d = client.get("/api/catalog").json()
    rows = {m["id"]: m for m in d["models"]}
    assert len(rows) == 8 and set(rows) == {m["id"] for m in catalog.CATALOG}
    assert rows["Qwen3-4B-Q4_0"]["installed"] is True and rows["gpt-oss-20b-MXFP4"]["installed"] is False
    assert rows["gpt-oss-20b-MXFP4"]["size_bytes"] == 12109566624 and rows["gpt-oss-20b-MXFP4"]["ram_gb"] == 16
    assert rows["Qwen3-4B-Q4_0"]["engine"]["engine"] == "own" and "repo" in rows["Qwen3-4B-Q4_0"] and d["ram_bytes"]
    assert all(m["sha256"] and len(m["sha256"]) == 64 and m["repo"].count("/") == 1 for m in rows.values())


def test_search_and_repository_routes(client, hub_server):
    assert client.get("/api/hub/search", params={"q": "qwen"}).json()["results"][0]["id"] == "unsloth/Qwen3-4B-GGUF"
    hub_server.tree_pages = [TREE]
    d = client.get("/api/hub/repo", params={"id": "unsloth/Qwen3-4B-GGUF"}).json()
    assert d["arch"] == "qwen3" and d["engine"]["engine"] == "own"
    assert [f["name"] for f in d["files"]] == ["Qwen3-4B-Q4_0.gguf", "Qwen3-4B-Q8_0.gguf"] and d["hidden"] == {"split": 2, "projector": 1}
    assert {"quant", "fit", "ram_gb", "sha256", "size"} <= set(d["files"][0])


def test_hub_errors_come_back_with_their_plain_message_and_status(client, hub_server, monkeypatch):
    hub_server.rate_limited = True
    r = client.get("/api/hub/search", params={"q": "x"})
    assert r.status_code == 429 and "limiting requests" in r.json()["detail"]
    hub_server.rate_limited = False
    hub_server.gated_repos = {"a/b"}
    monkeypatch.delenv("HF_TOKEN", raising=False)
    r = client.get("/api/hub/repo", params={"id": "a/b"})
    assert r.status_code == 403 and "gated or private" in r.json()["detail"]


def test_a_fallback_architecture_is_said_plainly_and_does_not_block(client, hub_server):
    hub_server.repo_info = {**hub_server.repo_info, "gguf": {"architecture": "mamba2"}}
    hub_server.tree_pages = [TREE]
    d = client.get("/api/hub/repo", params={"id": "a/b"}).json()
    assert d["engine"]["engine"] == "fallback" and "fallback engine" in d["engine"]["text"] and d["files"]


def test_offline_mode_turns_search_and_downloads_off_but_keeps_local_models(client, cfg, monkeypatch):
    write_gguf(cfg.models_dir / "mine.gguf")
    monkeypatch.setattr(settings, "_current", settings.Settings(**{**settings.get().__dict__, "offline": True}))
    assert client.get("/api/setup").json()["offline"] is True
    assert client.get("/api/hub/search").status_code == 409 and client.get("/api/hub/repo", params={"id": "a/b"}).status_code == 409
    assert client.post("/api/downloads", json={"source": "a/b", "file": "x.gguf"}).status_code == 409
    assert [m["id"] for m in client.get("/api/models").json()["models"]] == ["mine"]


def test_a_download_by_repo_and_by_link_runs_and_the_model_appears_in_the_list(client, hub_server, cfg):
    hub_server.add_file("owner/repo", "tiny-Q4_0.gguf", gguf_bytes(200_000))
    hub_server.add_file("owner/repo", "link-Q8_0.gguf", gguf_bytes(100_000))
    assert client.get("/api/models").json()["models"] == []
    r = client.post("/api/downloads", json={"source": "owner/repo", "file": "tiny-Q4_0.gguf"})
    assert r.status_code == 200 and r.json()["name"] == "tiny-Q4_0.gguf"
    wait_for(client, "done")
    link = f"{hub_server.url}/owner/repo/resolve/main/link-Q8_0.gguf"
    assert client.post("/api/downloads", json={"source": "https://huggingface.co/owner/repo/resolve/main/link-Q8_0.gguf"}).status_code == 200
    wait_for(client, "done")
    ids = [m["id"] for m in client.get("/api/models").json()["models"]]
    assert ids == ["link-Q8_0", "tiny-Q4_0"] and link


def test_bad_sources_are_refused_and_a_running_download_can_be_cancelled(client, hub_server, cfg):
    assert client.post("/api/downloads", json={"source": "not a source"}).status_code == 422
    assert client.post("/api/downloads", json={"source": "owner/repo"}).status_code == 422  # which file?
    assert client.post("/api/downloads/nope/cancel").status_code == 404
    hub_server.add_file("owner/repo", "slow-Q4_0.gguf", gguf_bytes(2_000_000))
    hub_server.delay = 0.05
    d = client.post("/api/downloads", json={"source": "owner/repo", "file": "slow-Q4_0.gguf"}).json()
    time.sleep(0.2)
    assert client.post(f"/api/downloads/{d['id']}/cancel").json()["state"] == "cancelled"
    assert (cfg.models_dir / "slow-Q4_0.gguf.part").exists() and client.get("/api/models").json()["models"] == []


def test_a_model_can_be_removed_only_from_the_models_folder_and_not_while_loaded(client, cfg, monkeypatch, tmp_path):
    write_gguf(cfg.models_dir / "old-Q4_0.gguf")
    write_gguf(tmp_path / "elsewhere.gguf")
    monkeypatch.setattr(live, "SESSION", type("S", (), {"model": {"id": "old-Q4_0"}})())
    assert client.delete("/api/models/old-Q4_0").status_code == 409
    monkeypatch.setattr(live, "SESSION", None)
    assert client.delete("/api/models/elsewhere").status_code == 404 and (tmp_path / "elsewhere.gguf").exists()
    assert client.delete("/api/models/old-Q4_0").json() == {"removed": "old-Q4_0.gguf"}
    assert not (cfg.models_dir / "old-Q4_0.gguf").exists()
    assert client.delete("/api/models/old-Q4_0").status_code == 404


def test_a_model_in_a_subfolder_can_be_removed_by_its_id(client, cfg):
    write_gguf(cfg.models_dir / "a" / "m.gguf")
    write_gguf(cfg.models_dir / "b" / "m.gguf", ctx=4096)
    assert client.delete("/api/models/b/m").json() == {"removed": "m.gguf"}
    assert (cfg.models_dir / "a" / "m.gguf").exists() and not (cfg.models_dir / "b" / "m.gguf").exists()
