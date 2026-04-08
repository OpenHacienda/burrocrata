"""SFT training loop using Unsloth + TRL.

Heavy imports (torch, unsloth, trl, peft) are all **inside** :func:`run`
so that the module can be imported on machines without the GPU stack
(for CLI help, config validation, tests, etc).
"""

from __future__ import annotations

import logging

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
