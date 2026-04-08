"""Builder registry. Look up by name."""

from __future__ import annotations

from collections.abc import Callable

from ..card import BuildResult
from . import dgt_sft

Builder = Callable[..., BuildResult]

REGISTRY: dict[str, Builder] = {
    dgt_sft.BUILDER_NAME: dgt_sft.build,
}


def get(name: str) -> Builder:
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown builder {name!r}. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]


def names() -> list[str]:
    return sorted(REGISTRY)
