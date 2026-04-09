"""Per-run reproducibility snapshot writer.

Creates ``experiments/<name>/<timestamp>/`` and drops:
- ``config.yaml``: resolved config (post-defaults)
- ``run.json``: git SHA, dirty flag, dataset fingerprint, dataset source
- ``metrics.json``: written after training completes

Separate from the actual training loop so it can be unit-tested without
touching torch.
"""

from __future__ import annotations

import datetime
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import Config
from .data import extract_fingerprint


def _git_sha_and_dirty() -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return sha, dirty
    except Exception:
        return "unknown", False


def _timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


@dataclass
class RunContext:
    config: Config
    output_dir: Path
    timestamp: str
    git_sha: str
    git_dirty: bool
    dataset_fingerprint: str | None
    started_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


def prepare_run(config: Config, now: str | None = None) -> RunContext:
    """Create the output directory and write the initial snapshot."""
    ts = now or _timestamp()
    out = config.resolved_output(ts)
    out.mkdir(parents=True, exist_ok=True)

    sha, dirty = _git_sha_and_dirty()
    fingerprint = extract_fingerprint(config.dataset)

    ctx = RunContext(
        config=config,
        output_dir=out,
        timestamp=ts,
        git_sha=sha,
        git_dirty=dirty,
        dataset_fingerprint=fingerprint,
    )

    # Resolved config dump
    (out / "config.yaml").write_text(
        yaml.safe_dump(
            config.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    # Provenance snapshot
    (out / "run.json").write_text(
        json.dumps(
            {
                "name": config.name,
                "timestamp": ts,
                "started_at": ctx.started_at,
                "git_sha": sha,
                "git_dirty": dirty,
                "dataset": {
                    "source": config.dataset.source,
                    "path": config.dataset.path,
                    "fingerprint": fingerprint,
                },
                "base_model": config.base_model,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return ctx


def write_metrics(ctx: RunContext, metrics: dict) -> Path:
    """Persist final training metrics after the loop completes."""
    p = ctx.output_dir / "metrics.json"
    p.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return p
