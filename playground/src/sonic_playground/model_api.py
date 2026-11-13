"""The model routes of the page: the recommended list, Hugging Face search, a repository's files, downloads, and removing a model."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import catalog, downloads, fit, hf_source, hub, live, models, settings

router = APIRouter(prefix="/api")
DOWNLOADS = downloads.Downloads()


def _online() -> None:
    if settings.get().offline:
        raise HTTPException(409, "The playground is offline (SONIC_PLAYGROUND_OFFLINE): model search and downloads are off.")


def _hub_failed(e: hub.HubError) -> HTTPException:
    return HTTPException({"not_found": 404, "gated": 403, "rate_limited": 429}.get(e.kind, 502), str(e))


@router.get("/catalog")
async def get_catalog() -> dict:
    """The recommended models, with this machine's memory so the page can say what fits."""
    cfg = settings.get()
    return {"models": await asyncio.to_thread(catalog.listing, cfg.models_dir), "ram_bytes": fit.total_ram_bytes()}


@router.get("/hub/search")
async def hub_search(q: str = "") -> dict:
    """The most downloaded GGUF repositories on Hugging Face matching `q` (the top ones when it is empty)."""
    _online()
    try:
        return {"results": await hub.client().search(q)}
    except hub.HubError as e:
        raise _hub_failed(e) from e


@router.get("/hub/repo")
async def hub_repo(id: str) -> dict:
    """One repository: its facts, the engine hint for its architecture, and its offered .gguf files."""
    _online()
    try:
        info, files = await asyncio.gather(hub.client().repo(id), hub.client().files(id))
    except hub.HubError as e:
        raise _hub_failed(e) from e
    return {**info, "engine": fit.engine_hint(info["arch"]), **hub.model_files(files, fit.total_ram_bytes())}


class DownloadReq(BaseModel):
    source: str  # a Hugging Face link to a .gguf file, or owner/repo
    file: str = ""  # the file name, when `source` is a repo id
    sha256: str | None = None  # the file's checksum, when the caller knows it (the page passes what the Hub said)


@router.post("/downloads")
async def post_download(req: DownloadReq) -> dict:
    try:
        repo, file = hf_source.parse_source(req.source, req.file)
        return DOWNLOADS.start(repo, file, req.sha256).to_dict()
    except hf_source.SourceError as e:
        raise HTTPException(422, str(e)) from e
    except downloads.DownloadError as e:
        raise HTTPException(409, str(e)) from e


@router.get("/downloads")
async def get_downloads() -> dict:
    return {"downloads": DOWNLOADS.listing()}


@router.post("/downloads/{download_id}/cancel")
async def cancel_download(download_id: str) -> dict:
    """Stop a download. Its `.part` file stays, so starting the same download again continues where it stopped."""
    try:
        return (await DOWNLOADS.cancel(download_id)).to_dict()
    except KeyError:
        raise HTTPException(404, f"no download {download_id!r}") from None


@router.delete("/models/{model_id:path}")
async def delete_model(model_id: str) -> dict:
    """Remove a model file from the models folder. Refused while that model is loaded."""
    if live.SESSION is not None and live.SESSION.model["id"] == model_id:
        raise HTTPException(409, "This model is loaded. Unload it first.")
    try:
        name = await asyncio.to_thread(models.remove, settings.get().models_dir, model_id)
    except models.RemoveError as e:
        raise HTTPException(404, str(e)) from e
    live.log(f"removed {name} from the models folder")
    return {"removed": name}
