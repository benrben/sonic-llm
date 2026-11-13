#!/usr/bin/env python3
"""Compare the embedded Metal kernel sources of two libggml-metal builds, kind by kind.

    verify_embed.py <official libggml-metal> <patched libggml-metal>

Each kind (kernels/<kind>.metal, flattened the CMake way) sits between the symbols _ggml_metallib_<kind>_{start,end}
in the __DATA,__ggml_metallib section. Exits non-zero unless every official kind is byte-identical in the patched
build and the only extra kinds are SLLM's own kernels, qmm, cascade and prefill (D042: the patches add kernels and change no
ggml kernel).
"""
import hashlib
import re
import subprocess
import sys


def kinds(path):
    out = subprocess.run(["otool", "-l", path], capture_output=True, text=True, check=True).stdout
    m = re.search(r"sectname __ggml_metallib\n\s+segname __DATA\n\s+addr (\S+)\n\s+size \S+\n\s+offset (\d+)", out)
    if not m:
        sys.exit(f"{path}: no __ggml_metallib section")
    addr, off = int(m.group(1), 16), int(m.group(2))
    data = open(path, "rb").read()
    syms = {}
    for line in subprocess.run(["nm", "-gU", path], capture_output=True, text=True, check=True).stdout.splitlines():
        a, _, name = line.split()
        mm = re.match(r"_ggml_metallib_(.*)_(start|end)$", name)
        if mm:
            syms.setdefault(mm.group(1), {})[mm.group(2)] = int(a, 16)
    return {k: hashlib.sha256(data[v["start"] - addr + off:v["end"] - addr + off]).hexdigest() for k, v in syms.items()}


def main():
    a, b = kinds(sys.argv[1]), kinds(sys.argv[2])
    bad = [k for k in a if a[k] != b.get(k)]
    extra = sorted(set(b) - set(a))
    for k in bad:
        print(f"DIFF {k}")
    print(f"{len(a) - len(bad)}/{len(a)} official kernel sources byte-identical; extra kinds: {extra}")
    sys.exit(0 if not bad and extra == ["cascade", "prefill", "qmm"] else 1)


if __name__ == "__main__":
    main()
