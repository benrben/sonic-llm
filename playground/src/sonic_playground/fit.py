"""Will a model fit this Mac, and does Sonic LLM's own engine run it: the plain hints the model pages show."""

from __future__ import annotations

import os

GIB = 2**30
# the architectures Sonic LLM's own engine runs; any other GGUF still works, on the fallback engine inside sllm-server
NEXT_ARCHS = {"qwen3", "qwen2", "qwen35", "qwen35moe", "llama", "smollm3", "phi3", "gemma3", "gemma4", "gpt-oss"}


def total_ram_bytes() -> int | None:
    """This machine's memory in bytes; None where the system does not say."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def ram_gb(size_bytes: int) -> int:
    """About how much memory a model needs while it runs: its weights, room for the agents' memory, and the engine (rounded up)."""
    return -(-int(size_bytes * 1.25 + 1.5 * GIB) // GIB)


def fit_hint(size_bytes: int, total_ram: int | None) -> str:
    """"fits easily", "tight" or "too large" for a model file of `size_bytes` on a machine with `total_ram` bytes ("unknown"
    when the memory is not known). Up to 60% of the memory is easy, up to 85% is tight, since the system needs the rest."""
    if not total_ram:
        return "unknown"
    need = ram_gb(size_bytes) * GIB
    return "fits easily" if need <= 0.6 * total_ram else "tight" if need <= 0.85 * total_ram else "too large"


def engine_hint(arch: str | None) -> dict:
    """Which engine runs a model of architecture `arch`: {"engine": "own" | "fallback" | "unknown", "text": ...}."""
    if not arch:
        return {"engine": "unknown", "text": "The architecture is not known until the file is read."}
    if arch in NEXT_ARCHS:
        return {"engine": "own", "text": f"Sonic LLM's own engine runs {arch} models."}
    return {"engine": "fallback", "text": f"Sonic LLM's own engine does not run {arch} yet: it works, on the fallback engine, without the speed-ups."}
