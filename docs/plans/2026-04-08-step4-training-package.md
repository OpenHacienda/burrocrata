# Step 4: `packages/training/` — Unsloth fine-tuning

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a third workspace member `packages/training/` (`burrocrata-training`) providing a config-driven Unsloth fine-tuning pipeline: load a built HF dataset, train a LoRA adapter, write a reproducibility snapshot to `experiments/<name>/<timestamp>/`. Eval is Phase 1 only (loss/perplexity via SFTTrainer's built-in eval). CLI: `burrocrata-train run | eval | merge | push`.

**Architecture:**
- **Heavy ML deps are in an opt-in dependency group** (`[dependency-groups] heavy`). Default `uv sync` does NOT install torch/unsloth/trl/peft/bitsandbytes. This keeps laptop workflows cheap — you only install the GPU stack on the GPU box.
- **Lazy imports**: `train.py`, `eval.py`, `merge.py`, `push.py` import torch/unsloth/trl inside their `run(...)` / `main(...)` functions, never at module top-level. This way `import burrocrata.training.train` and `burrocrata-train --help` work with zero heavy deps installed.
- **Pydantic config schema**: every experiment is a YAML file validated against `Config`. Bad configs fail fast before any model load.
- **Reproducibility snapshot** written at train start into `experiments/<name>/<timestamp>/`: resolved config, git SHA + dirty flag, dataset fingerprint extracted from the dataset's `dataset_card.md`, `uv.lock` copy, final metrics JSON (written after training).
- No Nix packaging for training this step — deferred (see "Not in scope"). Training runs via `uv sync --group heavy && uv run burrocrata-train run ...` on the GPU box.

**Tech Stack (light, always installed):** Python 3.11+, `pydantic>=2`, `pyyaml`, `click`, `datasets`.
**Tech Stack (heavy, opt-in):** `torch`, `unsloth`, `trl`, `peft`, `transformers`, `bitsandbytes`, `accelerate`.

---

## Task 1: scaffold `packages/training/`

**Files:**
- Create: `packages/training/pyproject.toml`
- Create: `packages/training/src/burrocrata/training/__init__.py`
- Create: `packages/training/src/burrocrata/training/prompts/__init__.py`
- Create: `packages/training/src/burrocrata/training/configs/base.yaml`
- Create: `packages/training/tests/`

**Step 1: pyproject.toml**

```toml
[project]
name = "burrocrata-training"
version = "0.1.0"
description = "Unsloth-based fine-tuning pipeline for burrocrata datasets"
requires-python = ">=3.11"
dependencies = [
  "pydantic>=2.6",
  "pyyaml>=6.0",
  "click>=8.1.0",
  "datasets>=2.18.0",
]

[project.scripts]
burrocrata-train = "burrocrata.training.cli:main"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
include = ["burrocrata*"]
namespaces = true

[tool.setuptools.package-data]
"burrocrata.training" = ["configs/*.yaml"]

[dependency-groups]
dev = ["pytest>=8.0"]
# Heavy GPU-side deps. Install with: `uv sync --group heavy`
# on the training machine.
heavy = [
  "torch>=2.3",
  "transformers>=4.44",
  "trl>=0.9",
  "peft>=0.11",
  "accelerate>=0.33",
  "bitsandbytes>=0.43; sys_platform == 'linux'",
  # Unsloth is installed from their index; leave as an opt-in
  # install the user can do manually per their CUDA version:
  #   uv pip install unsloth
]
```

Rationale for leaving `unsloth` as a manual install: its wheels are keyed to specific CUDA / torch versions and upstream recommends their own install command. Putting it in the group would force a specific pin that goes stale. Documented in CLAUDE.md in Task 10.

**Step 2: empty modules**

Each `__init__.py` is a one-line docstring. **Do NOT create `packages/training/src/burrocrata/__init__.py`** — PEP 420 namespace package.

`configs/base.yaml`: see Task 8.

**Step 3: Verify**

```bash
uv sync
uv run python -c "import burrocrata.training; import burrocrata.datasets; import burrocrata.scrapers; print('OK')"
uv run burrocrata-train --help 2>&1 | head -5
```

The last command will fail (no `cli.py` yet) — expected. Just confirm the package installs.

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(training): scaffold burrocrata-training workspace package"
```

---

## Task 2: pydantic config schema + loader

**Files:**
- Create: `packages/training/src/burrocrata/training/config.py`
- Create: `packages/training/tests/test_config.py`

**Step 1: Implementation**

```python
"""Pydantic schema for training configs.

Every experiment is a YAML file that round-trips through :class:`Config`.
Bad configs fail fast before any model is loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["local", "hub"]
    path: str
    split: str = "train"
    eval_split: str | None = "validation"


class LoraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target: str = "all-linear"


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: float = 1.0
    lr: float = 2e-4
    batch_size: int = 2
    grad_accum: int = 8
    max_seq: int = 4096
    seed: int = 42
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    every_steps: int = 200


class HubConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    push: bool = False
    repo: str | None = None

    @model_validator(mode="after")
    def _repo_required_if_push(self) -> HubConfig:
        if self.push and not self.repo:
            raise ValueError("hub.push is true but hub.repo is not set")
        return self


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Experiment name; used in output path")
    base_model: str
    dataset: DatasetConfig
    prompt: str = Field(..., description="Prompt template name (key in prompts registry)")
    lora: LoraConfig = Field(default_factory=LoraConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    output: str = "experiments/{name}/"
    hub: HubConfig = Field(default_factory=HubConfig)

    def resolved_output(self, timestamp: str) -> Path:
        """Return the concrete output directory for a run."""
        base = Path(self.output.format(name=self.name))
        return base / timestamp


def load_config(path: Path) -> Config:
    """Load a YAML file and validate it against :class:`Config`."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must be a YAML mapping")
    return Config.model_validate(data)
```

**Step 2: Tests**

```python
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
```

**Step 3: Run**

```bash
uv sync
uv run --with pytest pytest packages/training/tests/test_config.py -v
```
Expected: 6 passed.

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(training): add pydantic config schema and loader"
```

---

## Task 3: dataset loader + fingerprint extraction

**Files:**
- Create: `packages/training/src/burrocrata/training/data.py`
- Create: `packages/training/tests/test_data.py`

**Step 1: Implementation**

```python
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
```

**Step 2: Tests**

```python
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
```

**Step 3: Run**

```bash
uv run --with pytest pytest packages/training/tests/test_data.py -v
```
Expected: 6 passed.

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(training): add dataset loader with fingerprint extraction"
```

---

## Task 4: prompt templates

**Files:**
- Create: `packages/training/src/burrocrata/training/prompts/__init__.py`
- Create: `packages/training/src/burrocrata/training/prompts/dgt_sft.py`

**Step 1: Registry + `dgt_sft` template**

Our datasets already produce `messages` rows in ChatML shape (system/user/assistant). The "prompt" step here is just applying the tokenizer's chat template at train time — no manual prompt building. But we keep the indirection because future datasets might need something else.

`prompts/__init__.py`:

```python
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
```

`prompts/dgt_sft.py`:

```python
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
```

**Step 2: Smoke test**

```python
# packages/training/tests/test_prompts.py
"""Tests for prompt templates (no real tokenizer)."""

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
    import pytest
    with pytest.raises(KeyError):
        get("nope")
```

**Step 3: Run**

```bash
uv run --with pytest pytest packages/training/tests/test_prompts.py -v
```
Expected: 3 passed.

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(training): add prompt template registry + dgt_sft"
```

---

## Task 5: run metadata writer (reproducibility snapshot)

**Files:**
- Create: `packages/training/src/burrocrata/training/run.py`
- Create: `packages/training/tests/test_run.py`

**Step 1: Implementation**

```python
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
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H-%M-%SZ"
    )


@dataclass
class RunContext:
    config: Config
    output_dir: Path
    timestamp: str
    git_sha: str
    git_dirty: bool
    dataset_fingerprint: str | None
    started_at: str = field(
        default_factory=lambda: datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
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
    p.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return p
```

**Step 2: Tests**

```python
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
```

**Step 3: Run**

```bash
uv run --with pytest pytest packages/training/tests/test_run.py -v
```
Expected: 2 passed.

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(training): add run metadata writer for reproducibility"
```

---

## Task 6: training loop (`train.py`) — lazy imports only

**Files:**
- Create: `packages/training/src/burrocrata/training/train.py`

This is the first file that touches the heavy ML stack. **All torch/unsloth/trl imports go inside the function body**, not at module top-level. The file must byte-compile with zero heavy deps installed.

**Step 1: Implementation**

```python
"""SFT training loop using Unsloth + TRL.

Heavy imports (torch, unsloth, trl, peft) are all **inside** :func:`run`
so that the module can be imported on machines without the GPU stack
(for CLI help, config validation, tests, etc).
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import Config
from .data import get_splits, load_dataset_for_config
from .prompts import get as get_prompt
from .run import RunContext, prepare_run, write_metrics

logger = logging.getLogger(__name__)


def run(config: Config) -> RunContext:
    """Execute one training run described by ``config``.

    Returns the :class:`RunContext` with metrics persisted.
    """
    # ---- Lazy heavy imports -------------------------------------------
    import torch  # noqa: F401  (ensures CUDA is available)
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    # ---- Prepare run dir + snapshot -----------------------------------
    ctx = prepare_run(config)
    logger.info("Run directory: %s", ctx.output_dir)
    logger.info("Git SHA: %s%s", ctx.git_sha, " (dirty)" if ctx.git_dirty else "")
    logger.info("Dataset fingerprint: %s", ctx.dataset_fingerprint or "n/a")

    # ---- Load model + tokenizer ---------------------------------------
    logger.info("Loading base model: %s", config.base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.train.max_seq,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target,
        use_gradient_checkpointing="unsloth",
        random_state=config.train.seed,
    )

    # ---- Load + format dataset ----------------------------------------
    logger.info("Loading dataset: %s (%s)", config.dataset.path, config.dataset.source)
    dsd = load_dataset_for_config(config.dataset)
    train_ds, eval_ds = get_splits(dsd, config.dataset)

    prompt_fn = get_prompt(config.prompt)
    train_ds = train_ds.map(lambda row: prompt_fn(row, tokenizer))
    if eval_ds is not None:
        eval_ds = eval_ds.map(lambda row: prompt_fn(row, tokenizer))

    # ---- Train ---------------------------------------------------------
    sft_cfg = SFTConfig(
        output_dir=str(ctx.output_dir / "checkpoints"),
        per_device_train_batch_size=config.train.batch_size,
        gradient_accumulation_steps=config.train.grad_accum,
        num_train_epochs=config.train.epochs,
        learning_rate=config.train.lr,
        warmup_ratio=config.train.warmup_ratio,
        weight_decay=config.train.weight_decay,
        logging_steps=config.train.logging_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=config.eval.every_steps if eval_ds is not None else None,
        seed=config.train.seed,
        max_seq_length=config.train.max_seq,
        dataset_text_field="text",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    logger.info("Starting training")
    train_result = trainer.train()

    # ---- Save adapter + metrics ---------------------------------------
    adapter_dir = ctx.output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = {
        "final_loss": float(train_result.training_loss),
        **{
            k: float(v)
            for k, v in train_result.metrics.items()
            if isinstance(v, (int, float))
        },
    }
    write_metrics(ctx, metrics)
    logger.info("Metrics: %s", metrics)

    return ctx
```

**Step 2: Syntax check (without heavy deps)**

```bash
uv run python -m py_compile packages/training/src/burrocrata/training/train.py
uv run python -c "import burrocrata.training.train; print('imported OK')"
```

Both must succeed. The `import` works because all heavy deps are inside `run()`. If you get an `ImportError` at import time, a stray top-level heavy import slipped in — find and move it.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(training): add Unsloth SFT training loop with lazy imports"
```

---

## Task 7: `eval.py`, `merge.py`, `push.py` — lazy stubs

**Files:**
- Create: `packages/training/src/burrocrata/training/eval.py`
- Create: `packages/training/src/burrocrata/training/merge.py`
- Create: `packages/training/src/burrocrata/training/push.py`

Same pattern as `train.py`: lazy heavy imports, light module-level logic.

**Step 1: `eval.py`**

```python
"""Phase 1 eval: loss + perplexity on the validation split.

Re-uses the Unsloth loader to avoid OOMs on laptop machines: this path
is intended to be invoked against a saved adapter on the same GPU that
trained it.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from .config import Config
from .data import get_splits, load_dataset_for_config
from .prompts import get as get_prompt

logger = logging.getLogger(__name__)


def run(config: Config, adapter_dir: Path) -> dict:
    """Compute loss + ppl on the eval split. Writes ``eval.json`` next to the adapter."""
    import torch
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=config.train.max_seq,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    dsd = load_dataset_for_config(config.dataset)
    _, eval_ds = get_splits(dsd, config.dataset)
    if eval_ds is None:
        raise RuntimeError("Config has no eval split configured")

    prompt_fn = get_prompt(config.prompt)
    eval_ds = eval_ds.map(lambda row: prompt_fn(row, tokenizer))

    sft_cfg = SFTConfig(
        output_dir=str(adapter_dir / "_eval_tmp"),
        per_device_eval_batch_size=1,
        max_seq_length=config.train.max_seq,
        dataset_text_field="text",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_cfg,
        train_dataset=eval_ds,  # dummy; we only call evaluate
        eval_dataset=eval_ds,
    )
    metrics = trainer.evaluate()
    loss = float(metrics.get("eval_loss", float("nan")))
    ppl = math.exp(loss) if not math.isnan(loss) else float("nan")
    out = {"eval_loss": loss, "eval_ppl": ppl}

    (adapter_dir / "eval.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    logger.info("Eval: loss=%.4f ppl=%.3f", loss, ppl)
    return out
```

**Step 2: `merge.py`**

```python
"""Merge a trained LoRA adapter into its base model weights."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import Config

logger = logging.getLogger(__name__)


def run(config: Config, adapter_dir: Path, output_dir: Path) -> Path:
    """Load ``adapter_dir``, merge into base, save to ``output_dir``."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=config.train.max_seq,
        load_in_4bit=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(output_dir), tokenizer, save_method="merged_16bit")
    logger.info("Merged weights saved to %s", output_dir)
    return output_dir
```

**Step 3: `push.py`**

```python
"""Push an adapter or merged model to HuggingFace Hub."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import Config

logger = logging.getLogger(__name__)


def run(config: Config, source_dir: Path) -> str:
    """Upload ``source_dir`` to ``config.hub.repo``. Returns the repo id."""
    if not config.hub.push or not config.hub.repo:
        raise RuntimeError(
            "hub.push is false or hub.repo is unset; nothing to push"
        )
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(config.hub.repo, exist_ok=True)
    api.upload_folder(
        folder_path=str(source_dir),
        repo_id=config.hub.repo,
        repo_type="model",
    )
    logger.info("Pushed %s → %s", source_dir, config.hub.repo)
    return config.hub.repo
```

**Step 4: Verify imports (no heavy deps)**

```bash
uv run python -c "
import burrocrata.training.eval
import burrocrata.training.merge
import burrocrata.training.push
print('all imported OK')
"
```

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(training): add eval, merge, push stubs with lazy imports"
```

---

## Task 8: base config YAML + DGT example

**Files:**
- Create: `packages/training/src/burrocrata/training/configs/base.yaml`
- Create: `packages/training/src/burrocrata/training/configs/dgt-qwen3-8b-lora.yaml`

**Step 1: `base.yaml`** (reference only — not consumed directly; shows defaults)

```yaml
# Reference config showing all defaults applied by the pydantic schema.
# Not used at runtime. Copy-paste as a starting point for new experiments.
name: REQUIRED
base_model: REQUIRED
dataset:
  source: local          # or "hub"
  path: datasets/dgt-sft
  split: train
  eval_split: validation
prompt: dgt_sft
lora:
  r: 16
  alpha: 32
  dropout: 0.0
  target: all-linear
train:
  epochs: 1.0
  lr: 2.0e-4
  batch_size: 2
  grad_accum: 8
  max_seq: 4096
  seed: 42
  warmup_ratio: 0.03
  weight_decay: 0.0
  logging_steps: 10
eval:
  every_steps: 200
output: experiments/{name}/
hub:
  push: false
  repo: null
```

**Step 2: `dgt-qwen3-8b-lora.yaml`**

```yaml
name: dgt-qwen3-8b-lora-v1
base_model: unsloth/Qwen3-8B-bnb-4bit
dataset:
  source: local
  path: datasets/dgt-sft
  split: train
  eval_split: validation
prompt: dgt_sft
lora:
  r: 16
  alpha: 32
  dropout: 0.0
  target: all-linear
train:
  epochs: 1.0
  lr: 2.0e-4
  batch_size: 2
  grad_accum: 8
  max_seq: 4096
  seed: 42
eval:
  every_steps: 200
hub:
  push: false
```

**Step 3: Verify the example validates**

```bash
uv run python -c "
from pathlib import Path
from burrocrata.training.config import load_config
p = Path('packages/training/src/burrocrata/training/configs/dgt-qwen3-8b-lora.yaml')
cfg = load_config(p)
print(cfg.model_dump_json(indent=2)[:200])
"
```
Expected: JSON output starting with `{"name": "dgt-qwen3-8b-lora-v1"...`.

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(training): add base and dgt-qwen3-8b-lora reference configs"
```

---

## Task 9: CLI

**Files:**
- Create: `packages/training/src/burrocrata/training/cli.py`
- Create: `packages/training/src/burrocrata/training/__main__.py`

The CLI must `--help` successfully with zero heavy deps installed. Subcommand bodies call into the lazy-import modules; they only fail if the user actually tries to run them without the GPU stack.

**Step 1: `cli.py`**

```python
"""CLI for burrocrata-training.

The ``--help`` path does not touch any heavy dependency; the subcommands
call into lazy-import modules that only import torch/unsloth when
actually invoked.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from .config import load_config

logger = logging.getLogger("burrocrata.training")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """Train, eval, merge, and push models for burrocrata."""
    _setup_logging(verbose)


@cli.command("validate")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
def validate_cmd(config_path: Path) -> None:
    """Validate a training config YAML without loading any model."""
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        click.echo(f"Invalid config: {exc}", err=True)
        sys.exit(1)
    click.echo(f"OK: {cfg.name} ({cfg.base_model})")


@cli.command("run")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
def run_cmd(config_path: Path) -> None:
    """Run a training experiment described by CONFIG_PATH."""
    from .train import run as run_training

    cfg = load_config(config_path)
    ctx = run_training(cfg)
    click.echo(f"Done. Run dir: {ctx.output_dir}")


@cli.command("eval")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--adapter",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Trained adapter directory",
)
def eval_cmd(config_path: Path, adapter: Path) -> None:
    """Compute loss + perplexity on the eval split."""
    from .eval import run as run_eval

    cfg = load_config(config_path)
    out = run_eval(cfg, adapter)
    click.echo(f"eval_loss={out['eval_loss']:.4f} eval_ppl={out['eval_ppl']:.3f}")


@cli.command("merge")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--adapter",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
)
def merge_cmd(config_path: Path, adapter: Path, output_dir: Path) -> None:
    """Merge a LoRA adapter into its base model."""
    from .merge import run as run_merge

    cfg = load_config(config_path)
    out = run_merge(cfg, adapter, output_dir)
    click.echo(f"Merged → {out}")


@cli.command("push")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--source",
    "source_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
def push_cmd(config_path: Path, source_dir: Path) -> None:
    """Upload an adapter or merged model to HF Hub."""
    from .push import run as run_push

    cfg = load_config(config_path)
    repo = run_push(cfg, source_dir)
    click.echo(f"Pushed → {repo}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
```

**Step 2: `__main__.py`**

```python
"""Allow running as `python -m burrocrata.training`."""

from .cli import main

main()
```

**Step 3: Verify** (no heavy deps installed)

```bash
uv sync
uv run burrocrata-train --help
uv run burrocrata-train validate --help
uv run burrocrata-train run --help
uv run burrocrata-train eval --help
uv run burrocrata-train merge --help
uv run burrocrata-train push --help

# Validate the example config
uv run burrocrata-train validate \
  packages/training/src/burrocrata/training/configs/dgt-qwen3-8b-lora.yaml
```

The `validate` command must print `OK: dgt-qwen3-8b-lora-v1 (unsloth/Qwen3-8B-bnb-4bit)`.

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(training): add burrocrata-train CLI (validate/run/eval/merge/push)"
```

---

## Task 10: update CLAUDE.md + final verification

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Append the training section**

Add a new section to the "Layout" bullet list — update the entry for `packages/training/` from "planned" to real, and add an "Installing heavy deps" subsection under "Common commands".

Update the layout bullet:

```markdown
- `packages/training/` — `burrocrata-training`, namespace `burrocrata.training.*`.
  Unsloth-based fine-tuning pipeline (CLI: `burrocrata-train`). Heavy ML
  deps (torch, unsloth, trl) live in an opt-in `heavy` dependency group.
```

Add to the commands section:

````markdown
### Training (GPU box)

Training pulls in torch, unsloth, and friends. These live in an opt-in
dependency group so laptop workflows stay cheap. On a machine with CUDA:

```bash
uv sync --group heavy                   # install torch/trl/peft/...
uv pip install unsloth                  # installed manually per CUDA version
uv run burrocrata-train validate packages/training/src/burrocrata/training/configs/dgt-qwen3-8b-lora.yaml
uv run burrocrata-train run      packages/training/src/burrocrata/training/configs/dgt-qwen3-8b-lora.yaml
```

Training artifacts land in `experiments/<name>/<timestamp>/`:
- `config.yaml`: resolved config
- `run.json`: git SHA, dataset fingerprint, start time
- `checkpoints/`: TRL checkpoints
- `adapter/`: saved LoRA adapter
- `metrics.json`: final training metrics
````

**Step 2: Final workspace-wide checks**

```bash
uv sync
uv run --with pytest pytest packages/scrapers/tests packages/datasets/tests packages/training/tests -v
uv run ruff check packages/scrapers packages/datasets packages/training
nix develop -c treefmt --fail-on-change

uv run burrocrata-dgt --help
uv run burrocrata-datasets --help
uv run burrocrata-train --help
uv run burrocrata-train validate \
  packages/training/src/burrocrata/training/configs/dgt-qwen3-8b-lora.yaml
```

Expected:
- All three CLIs print `--help`.
- Total test count across the three packages should be around 22 (5 scrapers + 5 datasets + ~12 training).
- All ruff + treefmt clean.
- Validate command prints `OK: ...`.

**Step 3: Orphan sweep**

```bash
rg -n 'import torch|import unsloth|from trl|from peft' \
  --glob '!docs/**' --glob '!.worktrees/**' packages/training/src
```
Expected: matches must be **inside function bodies only**, never at module top level. The grep itself won't distinguish — visually inspect each line and confirm it's indented (inside a def).

**Step 4: Commit any cleanup (if needed)**

If treefmt reformatted anything:
```bash
git add -A
git commit -m "chore: treefmt"
```

---

## Done criteria for Step 4

- [ ] `packages/training/` exists with pydantic config schema, data loader, prompts, run writer, train/eval/merge/push modules, CLI, and example configs
- [ ] Default `uv sync` (no `--group heavy`) installs the package cleanly — no torch/unsloth required
- [ ] `burrocrata-train --help` works with zero heavy deps installed
- [ ] `burrocrata-train validate <config>` validates the example config successfully
- [ ] `pytest packages/training/tests` passes (≈ 17 tests across config, data, prompts, run)
- [ ] All heavy imports (torch, unsloth, trl, peft) are inside function bodies, not at module top level
- [ ] CLAUDE.md documents the `--group heavy` install dance
- [ ] ruff + treefmt clean across all three packages

## Not in scope

- **`nix develop .#training` CUDA devshell.** Designing a reproducible CUDA shell is its own project; users install via `uv sync --group heavy` on their GPU box. Will revisit if the repo starts targeting multiple CUDA versions.
- **Pinning Unsloth.** Their wheels are keyed to specific CUDA × torch versions and upstream recommends their own installer. Left as a manual `uv pip install unsloth` per the CLAUDE.md.
- **Actual training smoke run.** The plan executor may not have a GPU; training correctness is verified at first real use on the GPU box.
- **Rubric eval.** Phase 2 — after first successful training run.
- **HF Hub dataset loading tests.** Would require network; the `source: hub` branch is type-checked only.
- **Multi-GPU / DeepSpeed / FSDP.** YAGNI until single-GPU works end-to-end.
- **Wandb / Tensorboard reporting.** `report_to="none"` is hardcoded. Easy to add as a config field later.
