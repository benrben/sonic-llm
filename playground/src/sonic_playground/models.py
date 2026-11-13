"""The models the page offers: every .gguf file under the models folder, with what its header says of it."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .gguf_meta import read_gguf_header

log = logging.getLogger("sonic.models")

# llama.cpp LLAMA_FTYPE values (general.file_type)
FTYPE = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
         13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 32: "BF16", 38: "MXFP4"}
_QUANT_RE = re.compile(r"(BF16|F16|F32|MXFP4|IQ\d_[A-Z]+|Q\d_K(?:_[SML])?|Q\d_[01])", re.I)

_HEADER_CACHE: dict = {}  # (path, size, mtime_ns) -> row; a changed or new file is read again on the next call


def quant_of(path: Path, meta: dict) -> str:
    """The quantisation of a model: the header's file type, else the last quant word in the file name."""
    ft = meta.get("general.file_type")
    if isinstance(ft, int) and ft in FTYPE:
        return FTYPE[ft]
    found = _QUANT_RE.findall(path.stem)
    return found[-1].upper() if found else "unknown"


def params_label(meta: dict, n_params: int) -> str | None:
    """"8B" or "600M": the header's size label, else the tensors' element count."""
    if meta.get("general.size_label"):
        return str(meta["general.size_label"])
    if n_params:
        b = n_params / 1e9
        return f"{b:.1f}B" if b >= 1 else f"{n_params / 1e6:.0f}M"
    return None


def _file_row(f: Path) -> dict:
    st = f.stat()
    key = (str(f), st.st_size, st.st_mtime_ns)
    if key not in _HEADER_CACHE:
        row = {"id": f.stem, "file": str(f), "size_bytes": st.st_size, "arch": None, "quant": quant_of(f, {}), "params": None,
               "error": None}  # fmt: skip
        try:
            h = read_gguf_header(f)
        except (ValueError, OSError) as e:
            row["error"] = f"cannot read the GGUF header: {e}"
            log.warning("%s: %s", f, row["error"])
        else:
            row.update(arch=h["meta"].get("general.architecture"), quant=quant_of(f, h["meta"]),
                       params=params_label(h["meta"], h["n_params"]))
        _HEADER_CACHE[key] = row
    return _HEADER_CACHE[key]


def scan(models_dir: Path) -> list[dict]:
    """The models under `models_dir` (searched recursively), smallest first. Files that are not models are left out: `.part`
    files (a download under way) are not matched, and multimodal projectors (mmproj) cannot answer a prompt. A file whose
    header cannot be read is listed with its `error`, and the page does not offer it. Ids are the file names without
    `.gguf`; two files with one name get their folder in front of the second."""
    if not models_dir.is_dir():
        return []
    rows, seen = [], set()
    for f in sorted(models_dir.rglob("*.gguf")):
        if "mmproj" in f.name.lower():
            continue
        row = dict(_file_row(f))
        if row["id"] in seen:
            row["id"] = f"{f.parent.name}/{f.stem}"
        seen.add(row["id"])
        rows.append(row)
    return sorted(rows, key=lambda r: r["size_bytes"])


class RemoveError(Exception):
    """A model that cannot be removed."""


def remove(models_dir: Path, model_id: str) -> str:
    """Delete the model file with id `model_id`. Only a .gguf file inside `models_dir` is ever removed. Returns its file name."""
    row = next((r for r in scan(models_dir) if r["id"] == model_id), None)
    if row is None:
        raise RemoveError(f"no model {model_id!r} in the models folder")
    path = Path(row["file"]).resolve()
    if models_dir.resolve() not in path.parents or path.suffix != ".gguf":
        raise RemoveError(f"{path.name} is not inside the models folder; it is not removed")
    path.unlink()
    return path.name
