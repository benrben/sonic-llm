"""The settings: flag, then environment variable, then default; nothing outside the checkout is assumed."""

import os
from pathlib import Path
import stat

import pytest

from sonic_playground import settings


def test_defaults_are_relative_to_the_checkout_and_the_home_folder():
    s = settings.load([], {})
    assert s.server_bin == (settings.PLAYGROUND_DIR.parent / "engine" / "target" / "release" / "sllm-server").resolve()
    assert s.models_dir == (Path("~/models").expanduser().resolve())
    assert s.data_dir == settings.PLAYGROUND_DIR / ".data"
    assert s.samples_dir == settings.PLAYGROUND_DIR / "samples"
    assert (s.host, s.port, s.engine_args) == ("127.0.0.1", settings.DEFAULT_PORT, ())


def test_environment_beats_the_default_and_a_flag_beats_the_environment(tmp_path):
    env = {"SONIC_SERVER_BIN": str(tmp_path / "a"), "SONIC_MODELS": str(tmp_path / "m1"), "SONIC_PLAYGROUND_DATA": str(tmp_path / "d"),
           "SONIC_ENGINE_ARGS": "--threads 8 --chunk '64'", "SONIC_PLAYGROUND_PORT": "8123"}
    s = settings.load([], env)
    assert (s.server_bin, s.models_dir, s.data_dir) == (tmp_path / "a", tmp_path / "m1", tmp_path / "d")
    assert s.engine_args == ("--threads", "8", "--chunk", "64") and s.port == 8123
    s = settings.load(["--server-bin", str(tmp_path / "b"), "--models", str(tmp_path / "m2"), "--port", "9000"], env)
    assert (s.server_bin, s.models_dir, s.port) == (tmp_path / "b", tmp_path / "m2", 9000)


def test_llama_server_is_found_by_path_or_given_and_absent_otherwise(tmp_path, monkeypatch):
    llama = tmp_path / "llama-server"
    llama.write_text("#!/bin/sh\n")
    llama.chmod(0o755)
    assert settings.load([], {"LLAMA_SERVER_BIN": str(llama)}).llama_bin == llama
    assert settings.load(["--llama-bin", str(llama)], {}).llama_bin == llama
    monkeypatch.setenv("PATH", str(tmp_path))
    assert settings.load([], {}).llama_bin == llama  # on PATH
    llama.chmod(stat.S_IRUSR | stat.S_IWUSR)  # not executable: not usable
    assert settings.load([], {}).llama_bin is None
    assert settings.load([], {"LLAMA_SERVER_BIN": str(tmp_path / "missing")}).llama_bin is None


def test_the_engine_is_found_only_when_the_file_runs(tmp_path):
    s = settings.load([], {"SONIC_SERVER_BIN": str(tmp_path / "nope")})
    assert not s.engine_found()
    (tmp_path / "ok").write_text("x")
    os.chmod(tmp_path / "ok", 0o755)
    assert settings.load([], {"SONIC_SERVER_BIN": str(tmp_path / "ok")}).engine_found()


def test_derived_paths_live_in_the_data_folder(tmp_path):
    s = settings.load([], {"SONIC_PLAYGROUND_DATA": str(tmp_path)})
    assert {s.documents_dir, s.runs_file, s.logs_dir, s.engines_file} == {tmp_path / "documents", tmp_path / "runs.jsonl", tmp_path / "logs", tmp_path / "engines.json"}


def test_an_unknown_flag_is_refused():
    with pytest.raises(SystemExit):
        settings.load(["--nope"], {})
