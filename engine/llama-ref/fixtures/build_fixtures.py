"""Build the fixed parity inputs (phase 1 task 4a). Run from the playground venv, read only on the datasets:

    cd sllm/playground && uv run python ../../sllm-wt/phase1-task4a/engine/llama-ref/fixtures/build_fixtures.py \
        --llama-ref <path to target/release/llama-ref> --out <fixtures dir>

Writes, per pair, wikitext2k-<pair>.json (first 2,000 WikiText-2 raw test tokens, tokenized by llama.cpp with the
GGUF's own tokenizer) and gsm8k20-<pair>.json (first 20 GSM8K test questions, chat templated by the quality probe's
own renderer, ids from the HF tokenizer and cross-checked against llama.cpp). Refuses to overwrite.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from sllm_playground import quality_data, quality_run, quality_tasks
from sllm_playground.adapters.base import CRAZYIDEA_ROOT, ModelSpec
from sllm_playground.bos import LlamaCliTokenizer
from sllm_playground.pairs import load_registry
from sllm_playground.prompts import load_hf_tokenizer

WIKITEXT = {
    "repo": "Salesforce/wikitext",
    "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "file": "wikitext-2-raw-v1/test-00000-of-00001.parquet",
    "local": str(quality_data.DATA_DIR / "wikitext-2-raw-v1-test.parquet"),
}
N_WIKI = 2000
N_GSM = 20
PAIRS = ("qwen3-0.6b-q4", "gemma-3-4b-it-qat-4bit")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_new(path: Path, obj) -> None:
    with open(path, "x") as f:  # 'x': never overwrite
        json.dump(obj, f, separators=(",", ":"))
        f.write("\n")


def llama_tokenize(binary: str, gguf: Path, text: str, add_special: bool, parse_special: bool) -> dict:
    with tempfile.TemporaryDirectory() as d:
        t, o = Path(d, "t.txt"), Path(d, "o.json")
        t.write_text(text, encoding="utf-8")
        argv = [binary, "tokenize", str(gguf), str(t), str(o)]
        argv += ["--add-special"] if add_special else []
        argv += ["--parse-special"] if parse_special else []
        subprocess.run(argv, check=True, capture_output=True)
        return json.loads(o.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # -- WikiText-2 text: the non-empty rows of `text`, concatenated as they are (each row keeps its newline) ------
    wsha = quality_data.sha256_file(Path(WIKITEXT["local"]))
    rows = pq.read_table(WIKITEXT["local"]).column("text").to_pylist()
    text = "".join(r for r in rows if r != "")
    text_sha = sha256_bytes(text.encode("utf-8"))
    print(f"wikitext parquet sha256 {wsha}; {len(rows)} rows, {sum(1 for r in rows if r)} non-empty; text {len(text)} chars sha256 {text_sha}")

    # -- GSM8K questions (hash-checked by load_rows) -------------------------------------------------------------
    gsm_items = quality_tasks.gsm8k_items(N_GSM)
    gsm_ds = quality_data.GSM8K

    reg = {p.name: p for p in load_registry()}
    for name in PAIRS:
        p = reg[name]
        gguf = CRAZYIDEA_ROOT / p.gguf.path
        spec = ModelSpec(p.name, CRAZYIDEA_ROOT / p.mlx_dir, gguf, p.quant, add_bos=p.add_bos, bos_id=p.bos_id)
        bos = {"add_bos": p.add_bos, "bos_id": p.bos_id,
               "policy": "BOS once at position 0 when the model declares one (Gemma yes, Qwen3 no)"}
        gguf_info = {"path": str(p.gguf.path), "sha256": p.gguf.sha256, "bytes": p.gguf.bytes}

        # WikiText: llama.cpp's tokenizer, add_special on (the GGUF adds BOS exactly when the model wants it).
        tok = llama_tokenize(a.llama_ref, gguf, text, add_special=True, parse_special=False)
        ids = tok["ids"]
        assert tok["model_adds_bos"] == p.add_bos, (name, tok["model_adds_bos"], p.add_bos)
        if p.add_bos:
            assert ids[0] == p.bos_id and ids[1:].count(p.bos_id) == 0, "BOS must be once at position 0"
        else:
            assert p.bos_id is None or p.bos_id not in ids[:N_WIKI]
        assert len(ids) >= N_WIKI
        w = {
            "pair": name, "kind": "wikitext2k", "n_tokens": N_WIKI, "ids": ids[:N_WIKI],
            "ids_sha256": sha256_bytes(json.dumps(ids[:N_WIKI]).encode()),
            "total_tokens_in_text": len(ids),
            "dataset": {"repo": WIKITEXT["repo"], "revision": WIKITEXT["revision"], "file": WIKITEXT["file"],
                        "parquet_sha256": wsha, "split": "test"},
            "text_recipe": "concatenate the non-empty `text` rows of the parquet as they are (rows carry their own newlines)",
            "text_sha256": text_sha, "text_chars": len(text),
            "tokenizer": {"tool": "llama-ref tokenize (llama.cpp b11669 llama_tokenize)", "add_special": True,
                          "parse_special": False, "gguf": gguf_info, "n_vocab": tok["n_vocab"]},
            "bos": bos,
        }
        write_new(out / f"wikitext2k-{name}.json", w)
        print(f"{name}: wikitext2k ids[0:4]={w['ids'][:4]}")

        # GSM8K: the quality probe's renderer (HF chat template of the MLX dir), then the probe's own token check.
        hf_tok = load_hf_tokenizer(spec.mlx_dir)
        llama_cli = LlamaCliTokenizer(gguf, parse_special=True)
        prompts = []
        for it in gsm_items:
            r = quality_run.render(it, hf_tok, spec)
            why = quality_run.check_rendered(r, spec, hf_tok, llama_cli)
            assert not why, (name, it.item_id, why)
            # Same thing through llama-ref's own tokenizer (parse_special, BOS per policy).
            mine = llama_tokenize(a.llama_ref, gguf, r.text, add_special=p.add_bos, parse_special=True)["ids"]
            assert mine == r.ids, (name, it.item_id, "llama-ref tokenize differs")
            prompts.append({"index": it.meta["index"], "item_id": it.item_id, "ids": r.ids,
                            "ids_sha256": r.ids_sha256, "n": len(r.ids), "gold": it.meta["gold"],
                            "templated_text_sha256": sha256_bytes(r.full_text.encode("utf-8"))})
        g = {
            "pair": name, "kind": "gsm8k20", "n_prompts": N_GSM, "prompts": prompts,
            "dataset": {"repo": gsm_ds.repo, "revision": gsm_ds.revision, "file": gsm_ds.file,
                        "parquet_sha256": gsm_ds.sha256, "split": "test", "rows": [0, N_GSM]},
            "prompt_recipe": {"instruction": quality_tasks.GSM8K_INSTRUCTION,
                              "message": "<instruction>\\n\\nProblem: <question>", "role": "user",
                              "renderer": "sllm_playground.quality_run.render (HF apply_chat_template, add_generation_prompt)",
                              "template_kwargs": quality_run.template_kwargs(spec, hf_tok),
                              "mlx_dir": str(p.mlx_dir)},
            "token_check": "HF re-encode, mlx_lm.server encode, llama-tokenize and llama-ref tokenize all give the same ids",
            "gguf": gguf_info, "bos": bos,
        }
        write_new(out / f"gsm8k20-{name}.json", g)
        print(f"{name}: gsm8k20 lengths {[x['n'] for x in prompts]}")


if __name__ == "__main__":
    sys.exit(main())
