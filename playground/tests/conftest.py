"""Shared fixtures: every test gets its own settings (folders inside tmp_path), so no test touches the real data folder."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # this checkout's package first

from sonic_playground import settings  # noqa: E402


@pytest.fixture(autouse=True)
def cfg(tmp_path, monkeypatch) -> settings.Settings:
    """Settings that point into tmp_path. The engine binary is a placeholder file that is never run."""
    server = tmp_path / "bin" / "sllm-server"
    server.parent.mkdir()
    server.write_text("#!/bin/sh\n")
    server.chmod(0o755)
    s = settings.Settings(server_bin=server, models_dir=tmp_path / "models", llama_bin=None, data_dir=tmp_path / "data",
                          samples_dir=tmp_path / "samples")
    monkeypatch.setattr(settings, "_current", s)
    return s


def _kv(key: str, value) -> bytes:
    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    if isinstance(value, str):
        return s(key) + struct.pack("<I", 8) + s(value)
    return s(key) + struct.pack("<II", 4, value)


def write_gguf(path: Path, arch: str = "qwen3", ctx: int = 32768, file_type: int = 15, **extra) -> Path:
    """A GGUF file with a header and no tensors: enough for the model list, the trained context and the chat template lookup."""
    kvs = {"general.architecture": arch, f"{arch}.context_length": ctx, "general.file_type": file_type, **extra}
    body = b"".join(_kv(k, v) for k, v in kvs.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + body)
    return path


@pytest.fixture
def hub_server(cfg, monkeypatch):
    """A fake Hugging Face Hub the playground talks to instead of huggingface.co (the settings, the Hub client and the downloads
    of this test all point at it)."""
    from fakehub import FakeHub
    from sonic_playground import downloads, hub, model_api

    fake = FakeHub()
    monkeypatch.setattr(settings, "_current", settings.Settings(**{**cfg.__dict__, "hf_endpoint": fake.url}))
    monkeypatch.setattr(hub, "_client", None)
    monkeypatch.setattr(hub, "TOKEN_HOSTS", ("127.0.0.1",))  # the real list is huggingface.co only
    monkeypatch.setattr(model_api, "DOWNLOADS", downloads.Downloads())
    monkeypatch.setattr(downloads, "CHUNK", 65536)
    yield fake
    fake.close()
