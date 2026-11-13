"""Where a model comes from: a Hugging Face link, or a repo id and a file name, turned into (repo, file)."""

from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

_REPO = re.compile(r"[\w.-]+/[\w.-]+")


class SourceError(ValueError):
    """A source the playground cannot make a download of."""


def plain_name(file: str) -> str:
    """The name a downloaded file is saved under: the last part of its path in the repo, which must be a .gguf name."""
    name = file.rsplit("/", 1)[-1]
    if not name.endswith(".gguf") or name.startswith(".") or "\\" in name:
        raise SourceError("the file must be a .gguf file")
    return name


def parse_source(text: str, file: str = "") -> tuple[str, str]:
    """(repo, file) from what the user typed: a link to the file (`https://huggingface.co/owner/repo/resolve/main/x.gguf`, or
    its `blob` page), or `owner/repo` with the file name given apart (or after the repo: `owner/repo/x.gguf`)."""
    text, file = text.strip(), file.strip()
    if text.startswith(("http://", "https://")):
        return _from_link(text)
    if not _REPO.match(text):
        raise SourceError("write a link to a .gguf file on huggingface.co, or owner/repo and the file name")
    repo = "/".join(text.split("/")[:2])
    rest = "/".join(text.split("/")[2:]) or file
    if not rest:
        raise SourceError(f"which file of {repo}? Give the .gguf file name")
    plain_name(rest)
    return repo, rest


def _from_link(link: str) -> tuple[str, str]:
    parts = [unquote(p) for p in urlparse(link).path.strip("/").split("/")]
    if len(parts) < 5 or parts[2] not in ("resolve", "blob"):
        raise SourceError("the link must point at a file: https://huggingface.co/owner/repo/resolve/main/file.gguf")
    repo, file = "/".join(parts[:2]), "/".join(parts[4:])
    plain_name(file)
    return repo, file
