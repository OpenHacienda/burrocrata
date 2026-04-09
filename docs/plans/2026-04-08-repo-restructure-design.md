# Repo restructure for multi-source scraping + Unsloth fine-tuning

**Date:** 2026-04-08
**Status:** Design approved, ready for implementation

## Goal

Evolve the repo from a single DGT scraper into a three-stage pipeline
(**scrape → build dataset → fine-tune**) that supports multiple Spanish
tax/legal sources and Unsloth-based training, with HF Hub as the
primary artifact store and full local-only support.

## Scope

- Scraper + dataset building + training code all in this repo.
- Model weights and built datasets live on HF Hub (opt-in) or locally.
- Multiple scrapers anticipated (DGT first, BOE/TEAC/AEAT later).

## Top-level layout

```
burrocrata/
├── flake.nix, nix/              # devshell + packages (cheap shell + CUDA shell)
├── pyproject.toml               # uv workspace root
├── packages/
│   ├── scrapers/                # burrocrata-scrapers
│   │   └── src/burrocrata/scrapers/
│   │       ├── core/            # Session base, AIMD rate limit, checkpoint, ntfy
│   │       ├── dgt/             # current dgt/ contents
│   │       ├── boe/             # future
│   │       └── teac/            # future
│   ├── datasets/                # burrocrata-datasets
│   │   └── src/burrocrata/datasets/
│   │       ├── builders/        # raw md -> HF datasets (one per source)
│   │       ├── formats/         # sft, dpo, rag chunks
│   │       └── cli.py
│   └── training/                # burrocrata-training (heavy deps)
│       └── src/burrocrata/training/
│           ├── configs/         # YAML per experiment
│           ├── prompts/         # chat templates / system prompts
│           ├── data.py, train.py, eval.py, merge.py, push.py
│           └── cli.py
├── data/                        # gitignored; raw scraped corpora
├── datasets/                    # gitignored; built HF datasets / jsonl
├── experiments/                 # gitignored; runs, checkpoints, wandb
├── notebooks/
└── docs/
    ├── plans/
    └── sources/
```

**Naming:**
- Packages: `burrocrata-scrapers`, `burrocrata-datasets`, `burrocrata-training`.
- Namespace: `burrocrata.*` (e.g. `from burrocrata.scrapers.dgt import DGTSession`).
- CLIs: `burrocrata-dgt`, `burrocrata-datasets`, `burrocrata-train`.

**Why a uv workspace of 3 packages:** training's heavy deps
(torch, unsloth, bitsandbytes) don't pollute the scraper devshell.
`uv sync --package burrocrata-scrapers` on a cheap box, full workspace
on the GPU box.

## Data flow (three idempotent stages)

```
  [scrapers]          [datasets]              [training]
  PETETE ──▶ data/dgt/raw/*.html     ┐
             data/dgt/consultas/*.md │──▶ datasets/dgt-sft/    ──▶ experiments/
                                     │    (parquet + card)         runs/lora/
  BOE    ──▶ data/boe/...            ┘                             merged/
                                                                    ▲
                                                          push_to_hub.py
```

1. **Scrape** — source of truth: `data/<source>/` with raw HTML +
   markdown+frontmatter per document. Everything downstream is
   regenerable from here.
2. **Build dataset** — `burrocrata-datasets build dgt-sft` reads md,
   applies a formatter (`sft_chat`, `sft_completion`, `dpo`,
   `rag_chunks`), writes HF `datasets` parquet + `dataset_card.md`
   recording: source file count, content hash, split seed, filters,
   build timestamp, git SHA. Splits (train/validation) done here with
   a fixed seed stored in the card. Filters (drop anulados, dedupe,
   length caps, PII scrub) live here.
3. **Train** — Unsloth consumes `datasets/<name>/` by path or from HF
   Hub. Never touches `data/` directly. This firewall is what keeps
   training reproducible.

**Artifact storage:** `data/`, `datasets/`, `experiments/` all
gitignored. HF Hub primary for sharing (private dataset and model
repos); local-only fully supported — `push_to_hub` is an opt-in step,
never required by training.

## Training package details

Config-driven, not notebook-driven. Notebooks are for exploration
only.

**Config schema** (pydantic-validated):

```yaml
name: dgt-qwen3-8b-lora-v1
base_model: unsloth/Qwen3-8B-bnb-4bit
dataset:
  source: local          # or "hub"
  path: datasets/dgt-sft # or hub repo id
  split: train
prompt: dgt_sft
lora: {r: 16, alpha: 32, dropout: 0.0, target: all-linear}
train: {epochs: 1, lr: 2e-4, batch_size: 2, grad_accum: 8, max_seq: 4096, seed: 42}
eval: {every_steps: 200, held_out_frac: 0.02}
output: experiments/{name}/
hub: {push: false, repo: null}
```

**Reproducibility hooks** saved into `experiments/<name>/<timestamp>/`:
- Resolved config (after defaults merge)
- Git commit SHA + dirty flag
- Dataset fingerprint (hash from dataset card)
- `uv.lock` snapshot
- Final metrics JSON

**Dev ergonomics:** `nix develop .#training` devshell with CUDA wired
up, separate from the lightweight scraper shell.

## Eval strategy

**Phase 1 — ship with cheap eval only:** held-out loss + perplexity on
a 1–2% slice split at dataset-build time (fixed seed in the card).
Goal: detect "is the model learning / overfitting" — nothing more.

**Phase 2 — rubric eval after first run:** add `dgt-eval` builder
producing a frozen ~100-row eval set (question, reference consulta,
reference answer) held out of training. Add `eval.py rubric`
subcommand: generates answers, calls a judge model with a rubric
(factual accuracy, correct normativa, correct conclusion), writes
scored JSONL + summary. Eval set versioned on HF Hub.

**Discipline:** eval set built once and frozen. Never tune on it.
Never regenerate between runs.

## Migration plan (5 separate PRs)

**Step 0 — prep:** branch `restructure/workspace`, this design
committed.

**Step 1 — uv workspace shell + scrapers package:**
- Root `pyproject.toml` declaring uv workspace with `packages/scrapers`.
- Move `python/` → `packages/scrapers/src/burrocrata/scrapers/`.
- Rename package `scrapper-dgt` → `burrocrata-scrapers`, script
  `scrapper-dgt` → `burrocrata-dgt`.
- Imports: `scraper.dgt...` → `burrocrata.scrapers.dgt...`.
- Update `nix/packages/scrapper/` to new path/name.
- **Verify:** `uv run burrocrata-dgt test` and `... stats` work.

**Step 2 — extract `scrapers/core/`:** move generic bits from
`dgt/scraper.py` (AIMD rate limiter, retry wrapper, checkpoint
helpers, ntfy) into `burrocrata.scrapers.core`. `dgt/scraper.py`
imports from `core`. No behavior change. **Verify:**
`burrocrata-dgt test` still works; add a unit test for the rate
limiter.

**Step 3 — `datasets` package:**
- New workspace member `packages/datasets/`, no dep on scrapers
  (reads md+frontmatter directly).
- Port `export_sft_from_markdowns` → `burrocrata.datasets.builders.dgt_sft`.
- CLI: `burrocrata-datasets build dgt-sft --input data/dgt --output datasets/dgt-sft`.
- Output: HF `datasets` parquet + `dataset_card.md`.
- Delete the old exporter path.
- **Verify:** `datasets.load_from_disk("datasets/dgt-sft")` works; row
  count ≈ md files minus anulados.

**Step 4 — `training` package:**
- New workspace member `packages/training/`, heavy deps isolated.
- Implement `data.py`, `prompts/dgt_sft.py`, `train.py`, minimal
  `eval.py` (loss/perplexity only).
- `configs/base.yaml` + `configs/dgt-qwen3-8b-lora.yaml`.
- `nix develop .#training` devshell with CUDA.
- **Verify:** smoke run with `train.epochs: 0` loads model, loads
  dataset, runs one forward pass, writes `experiments/...`. Do NOT
  gate the PR on a full training run.

**Step 5 — docs & cleanup:** update `CLAUDE.md` with three-stage
flow and commands; delete any remnants of `python/`.

## Non-goals for the migration

- No second scraper (BOE/TEAC) — structure supports it, YAGNI until
  needed.
- No rubric eval — Phase 2.
- No Hub pushing wired up — add when first needed (~30 lines).
- No CI changes beyond making existing checks pass.

## Rollback

Each step is a separate PR. Steps 1–3 are independently useful; if
Step 4 goes sideways they can merge on their own.
