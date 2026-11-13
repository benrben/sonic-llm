"""What a person may type to name a model: a link, owner/repo and a file name, or owner/repo/file."""

import pytest

from sonic_playground.hf_source import SourceError, parse_source, plain_name


@pytest.mark.parametrize("text,file,want", [
    ("https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_0.gguf?download=true", "", ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_0.gguf")),
    ("https://huggingface.co/a/b/blob/main/sub/x%20y-Q4_0.gguf", "", ("a/b", "sub/x y-Q4_0.gguf")),
    ("owner/repo", "m-Q4_0.gguf", ("owner/repo", "m-Q4_0.gguf")),
    ("  owner/repo/m-Q4_0.gguf  ", "", ("owner/repo", "m-Q4_0.gguf")),
])
def test_sources_that_name_one_file(text, file, want):
    assert parse_source(text, file) == want


@pytest.mark.parametrize("text,file,why", [
    ("owner/repo", "", "which file"), ("hello", "", "write a link"), ("https://huggingface.co/owner/repo", "", "point at a file"),
    ("https://huggingface.co/a/b/resolve/main/model.bin", "", ".gguf"), ("owner/repo", "model.safetensors", ".gguf"),
])
def test_sources_that_do_not(text, file, why):
    with pytest.raises(SourceError, match=why):
        parse_source(text, file)


def test_plain_name_is_the_last_part_of_the_path():
    assert plain_name("a/b/c.gguf") == "c.gguf"
    with pytest.raises(SourceError):
        plain_name(".hidden.gguf")
