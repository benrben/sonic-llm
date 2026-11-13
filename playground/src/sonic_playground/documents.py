"""The documents the tests read: the sample files that come with the playground and the files a user uploads.

A document's path is `<folder>/<file name>`: `samples/` is the read-only folder of the playground, `uploads/` is the user's
own folder inside the data folder. Only plain text and code files are offered (TEXT_EXT).
"""

from __future__ import annotations

import re
from pathlib import Path

from . import settings

TEXT_EXT = {".md", ".txt", ".json", ".csv", ".py", ".rs", ".toml", ".yaml", ".yml"}
MAX_UPLOAD_BYTES = 2_000_000
_SAFE_NAME = re.compile(r"[^\w.\- ]+")


class DocumentError(Exception):
    """A document that is not in the library, or an upload that cannot be kept."""


def _roots() -> dict[str, Path]:
    cfg = settings.get()
    return {"samples": cfg.samples_dir, "uploads": cfg.documents_dir}


def list_files() -> list[dict]:
    """Every document: {path, bytes, est_tokens} (about 4 bytes of text to a token), samples first."""
    out = []
    for folder, root in _roots().items():
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in TEXT_EXT:
                n = p.stat().st_size
                out.append({"path": f"{folder}/{p.relative_to(root).as_posix()}", "bytes": n, "est_tokens": n // 4})
    return out


def read_file(rel: str, limit: int | None = None) -> str:
    """The text of document `rel`, cut to `limit` characters when given. Only files inside the two folders are read."""
    folder, _, name = rel.partition("/")
    root = _roots().get(folder)
    p = (root / name).resolve() if root is not None and name else None
    if p is None or root.resolve() not in p.parents or not p.is_file():
        raise DocumentError(f"no document {rel!r} in the library")
    text = p.read_text(errors="replace")
    return text[:limit] if limit else text


def clean_name(name: str) -> str:
    """A safe file name for an upload: no folders, no odd characters; a name must keep a text extension."""
    base = _SAFE_NAME.sub("_", Path(name).name).strip(" .")
    if Path(base).suffix.lower() not in TEXT_EXT:
        raise DocumentError(f"only text files can be added ({', '.join(sorted(TEXT_EXT))})")
    return base


def save_upload(name: str, data: bytes) -> str:
    """Keep an uploaded document in the uploads folder (an existing file of that name is replaced). Returns its path."""
    base = clean_name(name)
    if not data:
        raise DocumentError("the file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise DocumentError(f"the file is larger than {MAX_UPLOAD_BYTES // 1_000_000} MB")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise DocumentError("the file is not UTF-8 text") from e
    folder = settings.get().documents_dir
    folder.mkdir(parents=True, exist_ok=True)
    (folder / base).write_bytes(data)
    return f"uploads/{base}"
