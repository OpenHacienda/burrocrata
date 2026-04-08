# burrocrata

Scraper + dataset-building + fine-tuning pipeline for Spanish tax/legal
sources (DGT consultas vinculantes, future: BOE, TEAC, AEAT).

## Layout

This is a **uv workspace**. Packages live under `packages/`:

- `packages/scrapers/` — `burrocrata-scrapers`, namespace `burrocrata.scrapers.*`.
  Currently ships the DGT scraper (CLI: `burrocrata-dgt`).
- `packages/datasets/` — `burrocrata-datasets`, namespace `burrocrata.datasets.*`
  (CLI: `burrocrata-datasets`).
- `packages/training/` — `burrocrata-training`, namespace `burrocrata.training.*`.
  Unsloth-based fine-tuning pipeline (CLI: `burrocrata-train`). Heavy ML
  deps (torch, unsloth, trl) live in an opt-in `heavy` dependency group.

Scraped data lives in `data/` (gitignored). Built datasets in `datasets/`
(gitignored). Training runs in `experiments/` (gitignored).

See `docs/plans/2026-04-08-repo-restructure-design.md` for the full design.

## Common commands

```bash
uv sync                                    # install workspace
uv run burrocrata-dgt --help               # DGT scraper CLI
uv run burrocrata-dgt stats                # corpus stats
nix develop                                # lightweight dev shell
nix build .#scrapers                       # build scrapers package
```

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

## Conventions

- Use `rg` (ripgrep) instead of `grep`; `fd` instead of `find`.
- Namespace packages: never add `packages/scrapers/src/burrocrata/__init__.py`.
- All imports under `burrocrata.*`.
