"""Models known to work with Sonic LLM: public GGUF files on Hugging Face, with their size on disk and the memory they need.

The eight models of docs/benchmarks.md. Each entry names the file exactly (repo id and file name), its architecture, and the
file's size and SHA-256 as the Hugging Face Hub reports them (all eight were checked against the Hub). A finished download
is checked against the hash.
"""

from __future__ import annotations

from pathlib import Path

from . import fit

CATALOG = [
    {"id": "gemma-4-E2B-it-Q4_0", "arch": "gemma4", "name": "Gemma 4 E2B (Q4_0)", "repo": "unsloth/gemma-4-E2B-it-GGUF", "file": "gemma-4-E2B-it-Q4_0.gguf",
     "size_bytes": 3041378400, "sha256": "31d3a3c630d4e71a7416498c42660dd3805066948acaec76a47e1ffac7010132"},
    {"id": "Qwen3-4B-Q4_0", "arch": "qwen3", "name": "Qwen3 4B (Q4_0)", "repo": "unsloth/Qwen3-4B-GGUF", "file": "Qwen3-4B-Q4_0.gguf",
     "size_bytes": 2375773472, "sha256": "92f4a83cdfe691b216a949f915e47f35f0ed85b833aa03490c719160ac5cdf7a"},
    {"id": "gemma-3-4b-it-qat-Q4_0", "arch": "gemma3", "name": "Gemma 3 4B QAT (Q4_0)", "repo": "unsloth/gemma-3-4b-it-qat-GGUF",
     "file": "gemma-3-4b-it-qat-Q4_0.gguf", "size_bytes": 2370065536,
     "sha256": "b4987b5c1da6e1443290cb16b046dc63cd38988a3db509841a7eed32e4d743d4"},
    {"id": "Phi-3.5-mini-instruct-Q4_0", "arch": "phi3", "name": "Phi 3.5 mini (Q4_0)", "repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
     "file": "Phi-3.5-mini-instruct-Q4_0.gguf", "size_bytes": 2182468896,
     "sha256": "6e0a4c93fe5db9408348cd2cbd15eee6f448efe6f91255a208c2d28701259b1d"},
    {"id": "Mistral-7B-Instruct-v0.3-Q4_K_M", "arch": "llama", "name": "Mistral 7B Instruct v0.3 (Q4_K_M)", "repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
     "file": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", "size_bytes": 4372812000,
     "sha256": "1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6"},
    {"id": "Llama-3.1-8B-Instruct-Q4_0", "arch": "llama", "name": "Llama 3.1 8B Instruct (Q4_0)", "repo": "unsloth/Llama-3.1-8B-Instruct-GGUF",
     "file": "Llama-3.1-8B-Instruct-Q4_0.gguf", "size_bytes": 4675896704,
     "sha256": "88e2c6002459f892d77c7e5e219868ef08a57c4e92354c7271aaf64f47ab0eaa"},
    {"id": "gemma-4-12b-it-qat-q4_0", "arch": "gemma4", "name": "Gemma 4 12B QAT (Q4_0)", "repo": "google/gemma-4-12B-it-qat-q4_0-gguf",
     "file": "gemma-4-12b-it-qat-q4_0.gguf", "size_bytes": 6975879296,
     "sha256": "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b"},
    {"id": "gpt-oss-20b-MXFP4", "arch": "gpt-oss", "name": "gpt-oss 20B (MXFP4)", "repo": "ggml-org/gpt-oss-20b-GGUF", "file": "gpt-oss-20b-MXFP4.gguf",
     "size_bytes": 12109566624, "sha256": "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901"},
]


def listing(models_dir: Path) -> list[dict]:
    """The catalog for the page: each entry with its RAM estimate, the fit hint for this machine, the engine hint and whether
    its file is already in the models folder."""
    have = {p.name for p in models_dir.rglob("*.gguf")} if models_dir.is_dir() else set()
    ram = fit.total_ram_bytes()
    return [{**m, "ram_gb": fit.ram_gb(m["size_bytes"]), "fit": fit.fit_hint(m["size_bytes"], ram), "engine": fit.engine_hint(m["arch"]),
             "installed": m["file"] in have} for m in CATALOG]
