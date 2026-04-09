"""Tests for the training config schema."""

from pathlib import Path

import pytest
import yaml

from burrocrata.training.config import Config, load_config


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def _valid(**overrides) -> dict:
    base = {
        "name": "test-run",
        "base_model": "unsloth/Qwen3-8B-bnb-4bit",
        "dataset": {"source": "local", "path": "datasets/dgt-sft"},
        "prompt": "dgt_sft",
    }
    base.update(overrides)
    return base


def test_minimal_config_round_trips(tmp_path):
    p = _write_yaml(tmp_path, _valid())
    cfg = load_config(p)
    assert cfg.name == "test-run"
    assert cfg.dataset.source == "local"
    assert cfg.lora.r == 16  # default
    assert cfg.train.seed == 42  # default
    assert cfg.hub.push is False  # default


def test_unknown_field_rejected(tmp_path):
    p = _write_yaml(tmp_path, _valid(unknown_toplevel_key=1))
    with pytest.raises(Exception):
        load_config(p)


def test_hub_push_requires_repo(tmp_path):
    p = _write_yaml(tmp_path, _valid(hub={"push": True}))
    with pytest.raises(Exception):
        load_config(p)


def test_hub_push_with_repo_ok(tmp_path):
    p = _write_yaml(tmp_path, _valid(hub={"push": True, "repo": "user/model"}))
    cfg = load_config(p)
    assert cfg.hub.repo == "user/model"


def test_resolved_output_substitutes_name():
    cfg = Config.model_validate(_valid())
    out = cfg.resolved_output("2026-04-08T12-00-00Z")
    assert str(out) == "experiments/test-run/2026-04-08T12-00-00Z"


def test_dataset_source_enforced(tmp_path):
    data = _valid(dataset={"source": "bogus", "path": "x"})
    p = _write_yaml(tmp_path, data)
    with pytest.raises(Exception):
        load_config(p)
