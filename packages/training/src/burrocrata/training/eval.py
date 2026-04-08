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
    import torch  # noqa: F401
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

    (adapter_dir / "eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Eval: loss=%.4f ppl=%.3f", loss, ppl)
    return out
