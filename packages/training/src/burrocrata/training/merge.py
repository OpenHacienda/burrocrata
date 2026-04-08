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
