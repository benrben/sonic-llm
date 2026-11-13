"""Where the playground finds things: the engine, the models, llama.cpp and its own data folder.

Each setting is read in this order: a command-line flag, an environment variable, a default. Nothing else in the package
looks at the environment or at a fixed path; it asks `get()` for the settings.

    flag             environment variable        default
    --server-bin     SONIC_SERVER_BIN            ../engine/target/release/sllm-server (next to this folder)
    --models         SONIC_MODELS                ~/models
    --llama-bin      LLAMA_SERVER_BIN            llama-server on PATH
    --data           SONIC_PLAYGROUND_DATA       <playground folder>/.data
    --engine-args    SONIC_ENGINE_ARGS           none (extra flags for sllm-server, split like a shell does)
    --host, --port   SONIC_PLAYGROUND_HOST/PORT  127.0.0.1, 8000
    --offline        SONIC_PLAYGROUND_OFFLINE    off; 1 turns model search and downloads off
    (environment only)  HF_ENDPOINT              https://huggingface.co, where model downloads come from
    (environment only)  HF_TOKEN                 none; sent to Hugging Face for gated models, never stored or logged
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

PLAYGROUND_DIR = Path(__file__).resolve().parents[2]  # the playground/ folder of a checkout
BUILD_COMMAND = "cargo build --release -p sllm-server"  # run in the engine/ folder
DEFAULT_PORT = 8000


@dataclass(frozen=True)
class Settings:
    server_bin: Path
    models_dir: Path
    llama_bin: Path | None  # None: llama-server was not found, the page runs Sonic LLM alone
    data_dir: Path
    engine_args: tuple[str, ...] = ()
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    hf_endpoint: str = "https://huggingface.co"
    offline: bool = False  # no network at all: no model search, no downloads
    samples_dir: Path = field(default=PLAYGROUND_DIR / "samples")

    @property
    def documents_dir(self) -> Path:
        """Where the documents a user uploads are kept."""
        return self.data_dir / "documents"

    @property
    def runs_file(self) -> Path:
        return self.data_dir / "runs.jsonl"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def engines_file(self) -> Path:
        """The pids of the engine processes this playground started and has not stopped."""
        return self.data_dir / "engines.json"

    def engine_found(self) -> bool:
        return self.server_bin.is_file() and os.access(self.server_bin, os.X_OK)


def _expand(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _llama_bin(value: str | None) -> Path | None:
    """The llama-server to compare with: the given path, else the one on PATH. None when there is none that runs."""
    found = value or shutil.which("llama-server")
    if not found:
        return None
    path = _expand(found)
    return path if path.is_file() and os.access(path, os.X_OK) else None


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sonic-playground", description="Sonic LLM playground: run live tests on your models.")
    p.add_argument("--server-bin", help="the sllm-server binary (env SONIC_SERVER_BIN)")
    p.add_argument("--models", help="folder with your .gguf models, searched recursively (env SONIC_MODELS)")
    p.add_argument("--llama-bin", help="llama-server, for the side-by-side comparison (env LLAMA_SERVER_BIN)")
    p.add_argument("--data", help="folder for run history, logs and uploaded documents (env SONIC_PLAYGROUND_DATA)")
    p.add_argument("--engine-args", help="extra flags for sllm-server, for example '--threads 8' (env SONIC_ENGINE_ARGS)")
    p.add_argument("--offline", action="store_true", default=None, help="no model search or downloads (env SONIC_PLAYGROUND_OFFLINE=1)")
    p.add_argument("--host", help="address to listen on (env SONIC_PLAYGROUND_HOST)")
    p.add_argument("--port", type=int, help="port to listen on (env SONIC_PLAYGROUND_PORT)")
    return p


def load(argv: Sequence[str] = (), environ: Mapping[str, str] | None = None) -> Settings:
    """The settings from command-line flags `argv`, then the environment `environ`, then the defaults."""
    env = os.environ if environ is None else environ
    args = _parser().parse_args(list(argv))

    def pick(flag: str | None, var: str) -> str | None:
        return flag if flag is not None else env.get(var)

    server = pick(args.server_bin, "SONIC_SERVER_BIN")
    models = pick(args.models, "SONIC_MODELS")
    data = pick(args.data, "SONIC_PLAYGROUND_DATA")
    extra = pick(args.engine_args, "SONIC_ENGINE_ARGS")
    port = pick(None if args.port is None else str(args.port), "SONIC_PLAYGROUND_PORT")
    return Settings(
        server_bin=_expand(server) if server else (PLAYGROUND_DIR.parent / "engine" / "target" / "release" / "sllm-server").resolve(),
        models_dir=_expand(models or "~/models"),
        llama_bin=_llama_bin(pick(args.llama_bin, "LLAMA_SERVER_BIN")),
        data_dir=_expand(data) if data else PLAYGROUND_DIR / ".data",
        engine_args=tuple(shlex.split(extra)) if extra else (),
        host=pick(args.host, "SONIC_PLAYGROUND_HOST") or "127.0.0.1",
        port=int(port) if port else DEFAULT_PORT,
        offline=bool(args.offline) or env.get("SONIC_PLAYGROUND_OFFLINE", "") not in ("", "0"),
        hf_endpoint=(env.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/"),
    )


def hf_token() -> str | None:
    """The Hugging Face token for gated models, read from HF_TOKEN each time it is needed and kept nowhere."""
    return os.environ.get("HF_TOKEN") or None


_current: Settings | None = None


def configure(settings: Settings) -> None:
    """Use `settings` from now on (the command line does this before the server starts; tests do it per test)."""
    global _current
    _current = settings


def get() -> Settings:
    """The settings in use: the ones `configure` gave, else the environment and the defaults."""
    if _current is None:
        configure(load())
    return _current
