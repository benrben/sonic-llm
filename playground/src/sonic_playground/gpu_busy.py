"""How busy the GPU is, read from macOS (`ioreg`, no root needed). Other systems have no reading."""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys

_UTIL_RE = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')


def parse_gpu_utilisation(text: str) -> int | None:
    """"Device Utilization %" (0-100) from `ioreg -r -c AGXAccelerator -d 1`, or None when the text has none."""
    m = _UTIL_RE.search(text or "")
    return int(m.group(1)) if m else None


async def gpu_utilisation() -> int | None:
    """The GPU's utilisation now, in percent; None when this is not macOS or `ioreg` says nothing."""
    if sys.platform != "darwin":
        return None
    out = await asyncio.to_thread(subprocess.run, ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"],
                                  capture_output=True, text=True)  # fmt: skip
    return parse_gpu_utilisation(out.stdout)
