"""Model downloads into the models folder: progress, cancel, resume, a `.part` file renamed when complete, a disk-space check.

A download writes `<name>.gguf.part` and renames it to `<name>.gguf` only when the whole file is there (and, when its SHA-256 is
known, matches). Cancelling or a lost connection keeps the `.part` file: starting the same download again asks the server for
the rest (a Range request) and appends, or starts over when the server does not support ranges. The Hugging Face token comes from
HF_TOKEN, goes only to huggingface.co, and is not kept anywhere.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import time
import uuid
from collections import deque
from pathlib import Path
from urllib.parse import quote

import httpx

from . import hub, settings
from .hf_source import SourceError, plain_name

RESERVE_BYTES = 1 << 30  # free space that must stay after the download
CHUNK = 1 << 20
SPEED_WINDOW_S = 3.0
TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)|bytes \*/(\d+)")
ACTIVE = ("starting", "downloading", "verifying")


class DownloadError(Exception):
    """A download that cannot go on; the message is for the person who started it."""


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


class Download:
    def __init__(self, repo: str, file: str, sha256: str | None):
        self.id, self.repo, self.file, self.sha256 = uuid.uuid4().hex[:8], repo, file, sha256
        self.name = plain_name(file)
        self.state, self.error, self.resumed = "starting", None, False
        self.bytes_done, self.bytes_total = 0, 0
        self.task: asyncio.Task | None = None
        self._samples: deque[tuple[float, int]] = deque()

    def note_progress(self, now: float) -> None:
        """Remember the byte count for the speed: the average over the last SPEED_WINDOW_S seconds."""
        self._samples.append((now, self.bytes_done))
        while len(self._samples) > 2 and now - self._samples[0][0] > SPEED_WINDOW_S:
            self._samples.popleft()

    @property
    def speed_bps(self) -> float:
        if self.state != "downloading" or len(self._samples) < 2:
            return 0.0
        (t0, b0), (t1, b1) = self._samples[0], self._samples[-1]
        return (b1 - b0) / (t1 - t0) if t1 > t0 else 0.0

    def to_dict(self) -> dict:
        pct = round(100 * self.bytes_done / self.bytes_total, 1) if self.bytes_total else None
        return {"id": self.id, "repo": self.repo, "file": self.file, "name": self.name, "state": self.state, "error": self.error,
                "bytes_done": self.bytes_done, "bytes_total": self.bytes_total, "percent": pct, "speed_bps": round(self.speed_bps),
                "resumed": self.resumed}  # fmt: skip


class Downloads:
    """The downloads of this playground run, by id."""

    def __init__(self):
        self.items: dict[str, Download] = {}

    def start(self, repo: str, file: str, sha256: str | None = None) -> Download:
        """Begin downloading `file` of `repo` into the models folder. Refused (DownloadError) when offline, when the file is
        already there, or when it is being downloaded now. Needs a running event loop."""
        cfg = settings.get()
        if cfg.offline:
            raise DownloadError("The playground is offline (SONIC_PLAYGROUND_OFFLINE): downloads are off.")
        try:
            d = Download(repo, file, sha256)
        except SourceError as e:
            raise DownloadError(str(e)) from e
        if (cfg.models_dir / d.name).exists():
            raise DownloadError(f"{d.name} is already in the models folder.")
        if any(x.name == d.name and x.state in ACTIVE for x in self.items.values()):
            raise DownloadError(f"{d.name} is already being downloaded.")
        self.items[d.id] = d
        d.task = asyncio.create_task(self._run(d))
        return d

    async def cancel(self, download_id: str) -> Download:
        """Stop a download and wait until it has stopped. Raises KeyError for an id that is not known."""
        d = self.items[download_id]
        if d.task is not None and not d.task.done():
            d.task.cancel()
            await asyncio.gather(d.task, return_exceptions=True)
        return d

    def listing(self) -> list[dict]:
        return [d.to_dict() for d in reversed(self.items.values())]

    async def _run(self, d: Download) -> None:
        try:
            await _download(d)
            d.state = "done"
        except asyncio.CancelledError:
            d.state = "cancelled"
            raise
        except DownloadError as e:
            d.state, d.error = "failed", str(e)
        except httpx.TimeoutException:
            d.state, d.error = "failed", "The server stopped answering. Start the download again to continue where it stopped."
        except httpx.TransportError:
            d.state, d.error = "failed", "The connection was lost. Start the download again to continue where it stopped."
        except OSError as e:
            d.state, d.error = "failed", f"Could not write the file: {e}"


def _url(d: Download) -> str:
    return f"{settings.get().hf_endpoint}/{d.repo}/resolve/main/{quote(d.file)}"


async def _download(d: Download) -> None:
    models_dir = settings.get().models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    dest, part = models_dir / d.name, models_dir / (d.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    url = _url(d)
    headers = hub.auth_headers(url)
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        for attempt in range(2):
            req_headers = {**headers, **({"Range": f"bytes={have}-"} if have else {})}
            async with client.stream("GET", url, headers=req_headers) as r:
                if r.status_code == 416 and have:  # the part is longer than the file, or it is the whole file
                    total = _range_total(r)
                    if total == have:
                        d.bytes_done = d.bytes_total = have
                        break
                    part.unlink()
                    have = 0
                    continue
                _check_status(r, d, bool(headers))
                await _receive(r, d, part, have, models_dir)
                break
    await _finish(d, part, dest)


def _check_status(r: httpx.Response, d: Download, has_token: bool) -> None:
    if r.status_code in (200, 206):
        return
    raise DownloadError(hub.error_for(r.status_code, r.headers, d.name, has_token).args[0])


def _range_total(r: httpx.Response) -> int | None:
    m = _RANGE.fullmatch(r.headers.get("content-range", ""))
    return int(m.group(3) or m.group(4)) if m else None


async def _receive(r: httpx.Response, d: Download, part: Path, have: int, models_dir: Path) -> None:
    """Write the body of response `r` to `part`: appended when the server answered the Range request (206), else from the start."""
    resumed = r.status_code == 206
    if resumed:
        m = _RANGE.fullmatch(r.headers.get("content-range", ""))
        if not m or m.group(1) is None or int(m.group(1)) != have:
            raise DownloadError("The server answered a resume request with the wrong part of the file. Delete the .part file and try again.")
        total = int(m.group(3))
    else:
        have, total = 0, int(r.headers.get("content-length") or 0)
    d.resumed, d.bytes_done, d.bytes_total = resumed and have > 0, have, total
    _check_disk(models_dir, total - have, d)
    d.state = "downloading"
    first = have == 0
    with open(part, "ab" if resumed else "wb") as out:
        async for chunk in r.aiter_bytes(CHUNK):
            if first and not chunk.startswith(b"GGUF"):
                out.close()
                part.unlink()
                raise DownloadError("This file is not a GGUF model (the server sent something else).")
            first = False
            out.write(chunk)
            d.bytes_done += len(chunk)
            d.note_progress(time.monotonic())
    if total and d.bytes_done != total:
        raise DownloadError(f"The download ended early ({_gb(d.bytes_done)} of {_gb(total)}). Start it again to continue.")


def _check_disk(models_dir: Path, needed: int, d: Download) -> None:
    free = shutil.disk_usage(models_dir).free
    if needed > 0 and free < needed + RESERVE_BYTES:
        raise DownloadError(f"Not enough disk space: {d.name} needs {_gb(needed)} more, and {_gb(free)} is free "
                            f"(keeping {_gb(RESERVE_BYTES)} spare). Free some space or use another SONIC_MODELS folder.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


async def _finish(d: Download, part: Path, dest: Path) -> None:
    """Check the finished `.part` file against its SHA-256 (when known; read off the event loop) and rename it to its name."""
    if d.sha256:
        d.state = "verifying"
        if await asyncio.to_thread(_sha256, part) != d.sha256:
            part.unlink()
            raise DownloadError("The file does not match its checksum, so it was removed. Try the download again.")
    if dest.exists():
        raise DownloadError(f"{dest.name} appeared in the models folder during the download; it was not replaced.")
    os.replace(part, dest)
