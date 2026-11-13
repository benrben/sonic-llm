"""Chat framing: the thinking split, the hand-written framings by file name, and the framing derived from a GGUF template."""

import pytest
from conftest import write_gguf

from sonic_playground.chat_template import GEMMA3, QWEN3, effort_of, split_thinking, template_for


def test_split_thinking():
    assert split_thinking("<think>\nhmm 17*23\n</think>\n\n391") == ("hmm 17*23", "391")
    assert split_thinking("<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final<|message|>42") == ("think", "42")
    assert split_thinking("analysisthink hardassistantfinal42") == ("think hard", "42")
    assert split_thinking("plain answer about the final assistant") == ("", "plain answer about the final assistant")


def test_gemma_3_and_qwen3_files_use_their_hand_written_framing(cfg):
    assert template_for(cfg.models_dir / "gemma-3-4b-it-Q4_K_M.gguf") is GEMMA3
    assert template_for("Qwen3.6-35B-A3B-UD.gguf") is QWEN3


def test_any_other_model_takes_the_template_in_its_gguf(cfg):
    chatml = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
              "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    g = write_gguf(cfg.models_dir / "mymodel.gguf", **{"tokenizer.chat_template": chatml})
    t = template_for(g)
    assert (t.system_open, t.system_close, t.reply_open) == ("<|im_start|>system\n", "<|im_end|>\n<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n")
    assert t.prefix_text("S") + t.task_text("Q") == "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n"


def test_a_model_without_a_chat_template_is_refused(cfg):
    with pytest.raises(KeyError, match="no chat template"):
        template_for(write_gguf(cfg.models_dir / "bare.gguf"))


def test_thinking_on_needs_the_gguf_template_even_for_a_hand_written_model(cfg):
    with pytest.raises(KeyError):
        template_for(write_gguf(cfg.models_dir / "qwen3-x.gguf"), "high")


def test_effort_of():
    assert [effort_of(t) for t in ("off", "low", "medium", "high", "max")] == [None, "low", "medium", "high", "high"]
    with pytest.raises(ValueError):
        effort_of("lots")
