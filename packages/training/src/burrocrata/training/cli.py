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
