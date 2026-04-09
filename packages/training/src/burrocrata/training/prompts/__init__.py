"""Prompt template registry.

Each template is a callable ``(row, tokenizer) -> {"text": str}`` that
renders one dataset row into a single string the SFTTrainer can
tokenize. We indirect through a registry so experiments can pick a
template by name in their YAML.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import dgt_sft

Template = Callable[[dict, Any], dict]

REGISTRY: dict[str, Template] = {
    "dgt_sft": dgt_sft.render,
}


def get(name: str) -> Template:
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown prompt template {name!r}. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]
