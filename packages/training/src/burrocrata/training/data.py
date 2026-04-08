"""Dataset loading for training.

Supports two sources:
- ``local``: a directory written by ``burrocrata-datasets build``
  (HF ``save_to_disk`` format).
- ``hub``: an HF Hub repo id, loaded via ``datasets.load_dataset``.

Also extracts the dataset fingerprint from a sibling ``dataset_card.md``
(produced by the datasets package) so the training run can record
exactly which corpus it trained on.
"""

from __future__ import annotations

import re
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from .config import DatasetConfig

FINGERPRINT_RE = re.compile(r"Source fingerprint\s*\|\s*`([0-9a-f]+)`")


def load_dataset_for_config(cfg: DatasetConfig) -> DatasetDict | Dataset:
    """Load the dataset named by ``cfg``. Returns a ``DatasetDict`` when
    splits are present, else a single ``Dataset``."""
    if cfg.source == "local":
        return load_from_disk(cfg.path)
    if cfg.source == "hub":
        return load_dataset(cfg.path)
    raise ValueError(f"Unknown dataset source: {cfg.source}")


def extract_fingerprint(cfg: DatasetConfig) -> str | None:
    """Return the dataset fingerprint from ``dataset_card.md`` if present.

    Only meaningful for ``source: local`` — returns ``None`` for Hub
    datasets (those are identified by the repo + revision instead).
    """
    if cfg.source != "local":
        return None
    card = Path(cfg.path) / "dataset_card.md"
    if not card.exists():
        return None
    text = card.read_text(encoding="utf-8")
    m = FINGERPRINT_RE.search(text)
    return m.group(1) if m else None


def get_splits(
    dsd: DatasetDict | Dataset, cfg: DatasetConfig
) -> tuple[Dataset, Dataset | None]:
    """Return ``(train, eval_or_none)`` based on the config's split names."""
    if isinstance(dsd, Dataset):
        return dsd, None
    train = dsd[cfg.split]
    eval_ds = (
        dsd[cfg.eval_split]
        if cfg.eval_split and cfg.eval_split in dsd
        else None
    )
    return train, eval_ds
