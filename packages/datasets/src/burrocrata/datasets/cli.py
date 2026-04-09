"""CLI for burrocrata-datasets."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from .builders import get, names
from .card import write_card

logger = logging.getLogger("burrocrata.datasets")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """Build datasets from scraped corpora."""
    _setup_logging(verbose)


@cli.command("list")
def list_builders() -> None:
    """List available builders."""
    for n in names():
        click.echo(n)


@cli.command("build")
@click.argument("builder", type=click.Choice(names(), case_sensitive=False))
@click.option(
    "--input",
    "input_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Source corpus directory (e.g. data/dgt)",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Where to write the HF dataset",
)
@click.option("--seed", type=int, default=42, help="Split seed (default: 42)")
@click.option(
    "--val-frac",
    type=float,
    default=0.02,
    help="Validation fraction (default: 0.02)",
)
def build(
    builder: str, input_dir: Path, output_dir: Path, seed: int, val_frac: float
) -> None:
    """Build a dataset using BUILDER."""
    fn = get(builder)
    try:
        result = fn(
            input_dir=input_dir,
            output_dir=output_dir,
            seed=seed,
            val_frac=val_frac,
        )
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except RuntimeError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    card_path = write_card(result)
    click.echo(
        f"Built {result.builder} → {output_dir}\n"
        f"  rows: {result.n_rows} (train={result.n_train}, val={result.n_val})\n"
        f"  card: {card_path}"
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
