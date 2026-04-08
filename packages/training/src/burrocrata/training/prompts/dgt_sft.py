"""DGT SFT prompt template.

Input row shape (produced by ``burrocrata-datasets build dgt-sft``):
    {"messages": [{"role": ..., "content": ...}, ...], "metadata": {...}}

We apply the tokenizer's chat template to flatten ``messages`` into a
single string. No manual system-prompt concatenation here — the system
prompt is already the first message.
"""

from __future__ import annotations

from typing import Any


def render(row: dict, tokenizer: Any) -> dict:
    text = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}
