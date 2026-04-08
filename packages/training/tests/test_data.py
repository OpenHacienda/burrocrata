"""Tests for training data loading."""

from pathlib import Path

from datasets import Dataset, DatasetDict

from burrocrata.training.config import DatasetConfig
from burrocrata.training.data import (
    extract_fingerprint,
    get_splits,
    load_dataset_for_config,
)


def _make_local_dataset(tmp_path: Path) -> Path:
    out = tmp_path / "ds"
    dsd = DatasetDict(
        train=Dataset.from_list([{"text": "a"}, {"text": "b"}, {"text": "c"}]),
        validation=Dataset.from_list([{"text": "d"}]),
    )
    dsd.save_to_disk(str(out))
    return out


def test_load_local_dataset(tmp_path):
    path = _make_local_dataset(tmp_path)
    cfg = DatasetConfig(source="local", path=str(path))
    dsd = load_dataset_for_config(cfg)
    assert set(dsd.keys()) == {"train", "validation"}


def test_get_splits_returns_train_and_eval(tmp_path):
    path = _make_local_dataset(tmp_path)
    cfg = DatasetConfig(source="local", path=str(path))
    dsd = load_dataset_for_config(cfg)
    train, eval_ds = get_splits(dsd, cfg)
    assert len(train) == 3
    assert eval_ds is not None and len(eval_ds) == 1


def test_get_splits_no_eval_when_missing(tmp_path):
    path = _make_local_dataset(tmp_path)
    cfg = DatasetConfig(
        source="local", path=str(path), eval_split="nonexistent"
    )
    dsd = load_dataset_for_config(cfg)
    _, eval_ds = get_splits(dsd, cfg)
    assert eval_ds is None


def test_extract_fingerprint_from_card(tmp_path):
    path = _make_local_dataset(tmp_path)
    (path / "dataset_card.md").write_text(
        "| Source fingerprint | `abc123def` |\n", encoding="utf-8"
    )
    cfg = DatasetConfig(source="local", path=str(path))
    assert extract_fingerprint(cfg) == "abc123def"


def test_extract_fingerprint_none_when_missing(tmp_path):
    path = _make_local_dataset(tmp_path)
    cfg = DatasetConfig(source="local", path=str(path))
    assert extract_fingerprint(cfg) is None


def test_extract_fingerprint_hub_returns_none():
    cfg = DatasetConfig(source="hub", path="user/repo")
    assert extract_fingerprint(cfg) is None
