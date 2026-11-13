"""The `sonic-playground` command: read the settings, start the server."""

from __future__ import annotations

import sys
from collections.abc import Sequence

import uvicorn

from . import settings


def main(argv: Sequence[str] | None = None) -> None:
    cfg = settings.load(sys.argv[1:] if argv is None else argv)
    settings.configure(cfg)
    print(f"Sonic LLM playground: http://{cfg.host}:{cfg.port}")
    print(f"  engine  {cfg.server_bin}" + ("" if cfg.engine_found() else "  (not built yet)"))
    print(f"  models  {cfg.models_dir}")
    print(f"  llama   {cfg.llama_bin or 'not found (no comparison)'}")
    uvicorn.run("sonic_playground.app:app", host=cfg.host, port=cfg.port, log_level="warning")
