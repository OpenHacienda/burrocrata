"""Tests for prompt templates (no real tokenizer)."""

import pytest

from burrocrata.training.prompts import REGISTRY, get
from burrocrata.training.prompts.dgt_sft import render


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        return " | ".join(f"{m['role']}: {m['content']}" for m in messages)


def test_registry_lists_dgt_sft():
    assert "dgt_sft" in REGISTRY
    assert get("dgt_sft") is render


def test_dgt_sft_renders_messages():
    row = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ],
        "metadata": {},
    }
    out = render(row, _FakeTokenizer())
    assert out == {"text": "system: sys | user: u | assistant: a"}


def test_unknown_template_raises():
    with pytest.raises(KeyError):
        get("nope")
