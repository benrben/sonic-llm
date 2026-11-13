"""The playground server: the page (static/live.html), the live routes (live.py), the model list and the setup status."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import fit, live, model_api, models, settings
from .live_parts.engine import stop_orphans

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """At start: stop the engines an earlier run of this playground left behind. At the end: stop the loaded model's engines
    (uvicorn ends the process by the signal it was given, so the exit hook of live.py would not run)."""
    gone = await asyncio.to_thread(stop_orphans)
    if gone:
        live.log(f"stopped engines left by an earlier playground run: pids {gone}")
    yield
    await live.stop_session()


app = FastAPI(title="Sonic LLM playground", lifespan=lifespan)


@app.get("/")
async def page() -> HTMLResponse:
    return HTMLResponse((STATIC / "live.html").read_text())


@app.get("/api/models")
async def get_models() -> dict:
    """The models under the models folder (see models.scan); a model whose header cannot be read has an `error`."""
    cfg = settings.get()
    rows = await asyncio.to_thread(models.scan, cfg.models_dir)
    return {"models": [{**r, "engine": fit.engine_hint(r["arch"])} for r in rows], "dir": str(cfg.models_dir)}


@app.get("/api/setup")
async def get_setup() -> dict:
    """What the page needs to tell the user: is the engine built, is llama.cpp there for the comparison, where do models go."""
    cfg = settings.get()
    return {"engine": {"found": cfg.engine_found(), "path": str(cfg.server_bin), "build": settings.BUILD_COMMAND},
            "llama": {"found": cfg.llama_bin is not None}, "offline": cfg.offline, "hf_token": settings.hf_token() is not None,
            "models_dir": str(cfg.models_dir), "data_dir": str(cfg.data_dir)}


app.include_router(live.router)
app.include_router(model_api.router)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
