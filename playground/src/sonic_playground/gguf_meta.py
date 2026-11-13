"""Minimal GGUF header reader: parameter count (sum of tensor elements) and a few metadata keys.

Only the header is read (metadata KVs and tensor infos), never tensor data, so it is cheap
even for a 12 GB file.
"""

from __future__ import annotations

import logging
import os
import struct
from pathlib import Path

log = logging.getLogger("sonic.gguf")

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
KEEP = ("general.architecture", "general.file_type", "general.size_label")  # what read_gguf_header keeps of the metadata


class _Reader:
    def __init__(self, f):
        self.f = f
        self.size = os.fstat(f.fileno()).st_size

    def take(self, n: int) -> bytes:
        if n > self.size:  # a length no file of this size can hold: damaged, and too big to read
            raise ValueError("damaged GGUF header")
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError("truncated GGUF header")
        return b

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.unpack("<Q")
        return self.take(n).decode("utf-8", errors="replace")

    def skip_value(self, t: int) -> None:
        if t in _SCALAR:
            self.take(struct.calcsize(_SCALAR[t]))
        elif t == _STRING:
            self.take(self.unpack("<Q"))
        elif t == _ARRAY:
            et, n = self.unpack("<I"), self.unpack("<Q")
            if et in _SCALAR:
                self.take(struct.calcsize(_SCALAR[et]) * n)
            else:
                for _ in range(n):
                    self.skip_value(et)
        else:
            raise ValueError(f"unknown GGUF value type {t}")

    def value(self, t: int):
        if t in _SCALAR:
            return self.unpack(_SCALAR[t])
        if t == _STRING:
            return self.string()
        self.skip_value(t)
        return None


def read_gguf_header(path: Path) -> dict:
    """{'version', 'n_tensors', 'n_params', 'meta': {...kept keys}}."""
    with open(path, "rb") as f:
        r = _Reader(f)
        if r.take(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = r.unpack("<I")
        n_tensors, n_kv = r.unpack("<Q"), r.unpack("<Q")
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            t = r.unpack("<I")
            if key in KEEP:
                meta[key] = r.value(t)
            else:
                r.skip_value(t)
        n_params = 0
        for _ in range(n_tensors):
            r.string()
            n_dims = r.unpack("<I")
            elems = 1
            for _ in range(n_dims):
                elems *= r.unpack("<Q")
            r.unpack("<I")  # ggml type
            r.unpack("<Q")  # data offset
            n_params += elems
    return {"version": version, "n_tensors": n_tensors, "n_params": n_params, "meta": meta}


def read_gguf_metadata(path: Path, full_prefix: str = "tokenizer.") -> dict:
    """Every metadata KV of the header. Arrays whose key starts with `full_prefix` are read in full; other arrays are given
    as their length (the playground reads the chat template and the vocabulary without any other library)."""

    def array(r: _Reader, full: bool):
        et, n = r.unpack("<I"), r.unpack("<Q")
        if not full:
            if et in _SCALAR:
                r.take(struct.calcsize(_SCALAR[et]) * n)
            else:
                for _ in range(n):
                    r.skip_value(et)
            return n
        if et in _SCALAR:
            fmt = _SCALAR[et]
            return list(struct.unpack(f"<{n}{fmt[1]}", r.take(struct.calcsize(fmt) * n)))
        return [r.value(et) for _ in range(n)]

    with open(path, "rb") as f:
        r = _Reader(f)
        if r.take(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        r.unpack("<I")
        r.unpack("<Q")
        n_kv = r.unpack("<Q")
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            t = r.unpack("<I")
            meta[key] = array(r, key.startswith(full_prefix)) if t == _ARRAY else r.value(t)
    return meta


def train_ctx(gguf: Path) -> int | None:
    """The model's trained context (`<arch>.context_length` in the GGUF); None when the file lacks it or cannot be read."""
    try:
        meta = read_gguf_metadata(Path(gguf))
    except (OSError, ValueError, struct.error) as exc:
        log.warning("cannot read the GGUF metadata of %s: %s: %s", gguf, type(exc).__name__, exc)
        return None
    value = meta.get(f"{meta.get('general.architecture')}.context_length")
    return int(value) if isinstance(value, int) else None
