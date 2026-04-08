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
