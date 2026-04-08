"""Generic JSON checkpoint helpers for resumable scrapes."""

from __future__ import annotations

import json
from pathlib import Path


def load_checkpoint(data_dir: Path) -> dict:
    """Load a checkpoint dict from ``<data_dir>/checkpoint.json``, or {}."""
    cp_path = data_dir / "checkpoint.json"
    if cp_path.exists():
        return json.loads(cp_path.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data_dir: Path, checkpoint: dict) -> None:
    """Persist a checkpoint dict to ``<data_dir>/checkpoint.json``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    cp_path = data_dir / "checkpoint.json"
    cp_path.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
    )
