"""The model list: every .gguf under the models folder, read from its header."""

from conftest import write_gguf

from sonic_playground import models
from sonic_playground.gguf_meta import read_gguf_metadata, train_ctx


def test_scan_lists_ggufs_recursively_smallest_first_and_skips_projectors(cfg):
    write_gguf(cfg.models_dir / "qwen3-4b-Q4_K_M.gguf")
    write_gguf(cfg.models_dir / "family" / "gemma-3-1b-it.gguf", arch="gemma3", file_type=7)
    write_gguf(cfg.models_dir / "mmproj-gemma-3.gguf", arch="clip")
    (cfg.models_dir / "half.gguf.part").write_bytes(b"x")
    rows = models.scan(cfg.models_dir)
    assert [r["id"] for r in rows] == sorted((r["id"] for r in rows), key=lambda i: next(r["size_bytes"] for r in rows if r["id"] == i))
    by_id = {r["id"]: r for r in rows}
    assert set(by_id) == {"qwen3-4b-Q4_K_M", "gemma-3-1b-it"}
    assert (by_id["gemma-3-1b-it"]["arch"], by_id["gemma-3-1b-it"]["quant"]) == ("gemma3", "Q8_0")
    assert by_id["qwen3-4b-Q4_K_M"]["quant"] == "Q4_K_M" and by_id["qwen3-4b-Q4_K_M"]["error"] is None


def test_a_missing_folder_has_no_models(cfg):
    assert models.scan(cfg.models_dir) == []


def test_a_file_that_is_not_a_gguf_is_listed_with_its_error(cfg):
    (cfg.models_dir).mkdir()
    (cfg.models_dir / "broken-Q8_0.gguf").write_text("not a model")
    (row,) = models.scan(cfg.models_dir)
    assert "not a GGUF" in row["error"] and row["quant"] == "Q8_0"


def test_two_files_with_one_name_get_different_ids(cfg):
    write_gguf(cfg.models_dir / "a" / "m.gguf")
    write_gguf(cfg.models_dir / "b" / "m.gguf", ctx=4096)
    assert sorted(r["id"] for r in models.scan(cfg.models_dir)) == ["b/m", "m"]


def test_quant_comes_from_the_header_then_from_the_name(cfg):
    assert models.quant_of(cfg.models_dir / "x-q6_k.gguf", {"general.file_type": 18}) == "Q6_K"
    assert models.quant_of(cfg.models_dir / "x-q6_k.gguf", {}) == "Q6_K"
    assert models.quant_of(cfg.models_dir / "x.gguf", {}) == "unknown"


def test_params_label_prefers_the_size_label():
    assert models.params_label({"general.size_label": "35B-A3B"}, 1) == "35B-A3B"
    assert models.params_label({}, 4_020_000_000) == "4.0B" and models.params_label({}, 600_000_000) == "600M"
    assert models.params_label({}, 0) is None


def test_trained_context_is_read_from_the_header(cfg):
    g = write_gguf(cfg.models_dir / "m.gguf", arch="llama", ctx=131072)
    assert train_ctx(g) == 131072
    assert read_gguf_metadata(g)["general.architecture"] == "llama"
    assert train_ctx(cfg.models_dir / "missing.gguf") is None
    (cfg.models_dir / "bad.gguf").write_text("nope")
    assert train_ctx(cfg.models_dir / "bad.gguf") is None
