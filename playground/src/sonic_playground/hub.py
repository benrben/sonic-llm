"""The Hugging Face Hub as the playground uses it: search for GGUF repositories, read one repository's facts and its files.

Everything goes through one small class, `HubClient`, from the playground's backend (never from the browser). Results are kept in
memory for a few minutes. The token in HF_TOKEN goes only to huggingface.co. Calls used (all read-only):

  GET {endpoint}/api/models?search=<q>&filter=gguf&sort=downloads&direction=-1&limit=20    a JSON list of {id, downloads, likes, tags}
  GET {endpoint}/api/models/<repo>                  {id, private, gated, downloads, likes, gguf: {architecture, context_length}}
  GET {endpoint}/api/models/<repo>/tree/main?recursive=true    a JSON list of {type, path, size, lfs: {oid}} (a `Link` header
                                                               with rel="next" says there is another page)
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote, urlparse

import httpx

from . import fit, models, settings

CACHE_SECONDS = 300
TIMEOUT = httpx.Timeout(10.0, connect=5.0)
TOKEN_HOSTS = ("huggingface.co",)  # the only hosts the token is sent to
_SPLIT = re.compile(r"-\d{5}-of-\d{5}")
_NEXT_LINK = re.compile(r'<([^>]+)>\s*;\s*rel="next"')


class HubError(Exception):
    """A Hub call that failed. `kind` is one of offline, rate_limited, gated, not_found, timeout, http; the message is plain."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def auth_headers(url: str) -> dict[str, str]:
    """The Authorization header for a request to `url`: the HF_TOKEN, but only when the host is one of TOKEN_HOSTS."""
    token, host = settings.hf_token(), urlparse(url).hostname or ""
    if token and any(host == h or host.endswith("." + h) for h in TOKEN_HOSTS):
        return {"Authorization": f"Bearer {token}"}
    return {}


def error_for(status: int, headers, what: str, has_token: bool) -> HubError:
    """The plain-language error of a failed answer."""
    if status in (401, 403):
        if has_token:
            return HubError("gated", f"{what} needs access your HF_TOKEN does not have. Ask for access on huggingface.co.")
        return HubError("gated", f"{what} is gated or private. Accept its terms on huggingface.co and start the playground with HF_TOKEN set.")
    if status == 404:
        return HubError("not_found", f"{what} was not found on Hugging Face.")
    if status == 429:
        wait = headers.get("retry-after")
        return HubError("rate_limited", "Hugging Face is limiting requests." + (f" Try again in {wait} seconds." if wait else " Try again in a few minutes."))
    return HubError("http", f"Hugging Face answered {status} for {what}.")


class HubClient:
    def __init__(self, endpoint: str, *, ttl: float = CACHE_SECONDS, timeout: httpx.Timeout = TIMEOUT, clock: Callable[[], float] = time.monotonic):
        self.endpoint, self.ttl, self.timeout, self.clock = endpoint.rstrip("/"), ttl, timeout, clock
        self._cache: dict[str, tuple[float, object]] = {}

    async def _get(self, url: str, what: str) -> httpx.Response:
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as c:
                r = await c.get(url, headers=auth_headers(url))
        except httpx.TimeoutException as e:
            raise HubError("timeout", "Hugging Face did not answer in time. Check your connection and try again.") from e
        except httpx.TransportError as e:
            raise HubError("offline", "Cannot reach Hugging Face. Are you offline?") from e
        if r.status_code != 200:
            raise error_for(r.status_code, r.headers, what, bool(auth_headers(url)))
        return r

    async def _cached(self, key: str, load):
        hit = self._cache.get(key)
        if hit and self.clock() - hit[0] < self.ttl:
            return hit[1]
        value = await load()
        self._cache[key] = (self.clock(), value)
        return value

    async def search(self, query: str, limit: int = 20) -> list[dict]:
        """The most downloaded GGUF repositories matching `query` (all of them when it is empty): [{id, downloads, likes, tags}]."""
        params = {"filter": "gguf", "sort": "downloads", "direction": "-1", "limit": str(limit)}
        if query.strip():
            params["search"] = query.strip()
        url = f"{self.endpoint}/api/models?" + "&".join(f"{k}={quote(v)}" for k, v in params.items())

        async def load():
            rows = (await self._get(url, "the search")).json()
            return [{"id": r["id"], "downloads": r.get("downloads", 0), "likes": r.get("likes", 0), "tags": r.get("tags", [])[:8]} for r in rows]

        return await self._cached(url, load)

    async def repo(self, repo: str) -> dict:
        """One repository: {id, private, gated, downloads, likes, arch, context_length}. `arch` is None when the Hub has no GGUF facts."""
        url = f"{self.endpoint}/api/models/{quote(repo, safe='/')}"

        async def load():
            d = (await self._get(url, repo)).json()
            gguf = d.get("gguf") or {}
            return {"id": d.get("id", repo), "private": bool(d.get("private")), "gated": d.get("gated") or False,
                    "downloads": d.get("downloads", 0), "likes": d.get("likes", 0), "arch": gguf.get("architecture"),
                    "context_length": gguf.get("context_length")}  # fmt: skip

        return await self._cached(url, load)

    async def files(self, repo: str) -> list[dict]:
        """The files of a repository's main branch: [{path, size, sha256}] (sha256 is None for files that are not LFS)."""
        url = f"{self.endpoint}/api/models/{quote(repo, safe='/')}/tree/main?recursive=true"

        async def load():
            out, next_url = [], url
            while next_url:
                r = await self._get(next_url, repo)
                out += [{"path": f["path"], "size": f.get("size", 0), "sha256": (f.get("lfs") or {}).get("oid")}
                        for f in r.json() if f.get("type") == "file"]
                m = _NEXT_LINK.search(r.headers.get("link", ""))
                next_url = m.group(1) if m else None
            return out

        return await self._cached(url, load)


_client: HubClient | None = None


def client() -> HubClient:
    """The playground's Hub client for the endpoint in the settings (kept, so its cache lasts)."""
    global _client
    endpoint = settings.get().hf_endpoint
    if _client is None or _client.endpoint != endpoint:
        _client = HubClient(endpoint)
    return _client


def model_files(files: list[dict], total_ram: int | None) -> dict:
    """The .gguf files of a repository the playground offers, and how many it leaves out: {files, hidden: {split, projector}}. Files
    in several parts (`-00001-of-00003`) and multimodal projectors (mmproj) are not offered: the playground loads one file."""
    out, hidden = [], {"split": 0, "projector": 0}
    for f in files:
        name = f["path"].rsplit("/", 1)[-1]
        if not name.endswith(".gguf"):
            continue
        if "mmproj" in name.lower():
            hidden["projector"] += 1
        elif _SPLIT.search(name):
            hidden["split"] += 1
        else:
            out.append({**f, "name": name, "quant": models.quant_of(Path(name), {}), "fit": fit.fit_hint(f["size"], total_ram),
                        "ram_gb": fit.ram_gb(f["size"])})
    return {"files": sorted(out, key=lambda f: f["size"]), "hidden": hidden}
