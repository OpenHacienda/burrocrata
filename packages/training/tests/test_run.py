"""Tests for the run metadata writer."""

import json
from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict

from burrocrata.training.config import Config
from burrocrata.training.run import prepare_run, write_metrics


def _cfg(tmp_path: Path) -> Config:
    ds = tmp_path / "ds"
    DatasetDict(train=Dataset.from_list([{"text": "a"}])).save_to_disk(str(ds))
    (ds / "dataset_card.md").write_text(
        "| Source fingerprint | `deadbeef` |\n", encoding="utf-8"
    )
    return Config.model_validate(
        {
            "name": "unit-run",
            "base_model": "unsloth/Qwen3-8B-bnb-4bit",
            "dataset": {"source": "local", "path": str(ds)},
            "prompt": "dgt_sft",
            "output": str(tmp_path / "experiments" / "{name}") + "/",
        }
    )


def test_prepare_run_creates_snapshot(tmp_path):
    cfg = _cfg(tmp_path)
    ctx = prepare_run(cfg, now="2026-04-08T12-00-00Z")

    assert ctx.output_dir.exists()
    assert ctx.output_dir.name == "2026-04-08T12-00-00Z"
    assert ctx.dataset_fingerprint == "deadbeef"

    cfg_out = yaml.safe_load((ctx.output_dir / "config.yaml").read_text())
    assert cfg_out["name"] == "unit-run"

    run_out = json.loads((ctx.output_dir / "run.json").read_text())
    assert run_out["name"] == "unit-run"
    assert run_out["dataset"]["fingerprint"] == "deadbeef"
    assert "git_sha" in run_out


def test_write_metrics(tmp_path):
    cfg = _cfg(tmp_path)
    ctx = prepare_run(cfg, now="2026-04-08T12-00-00Z")
    p = write_metrics(ctx, {"final_loss": 1.23, "final_ppl": 3.42})
    data = json.loads(p.read_text())
    assert data == {"final_loss": 1.23, "final_ppl": 3.42}
