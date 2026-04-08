# burrocrata

Scraper + dataset-building + fine-tuning pipeline for Spanish tax/legal
sources (DGT consultas vinculantes, future: BOE, TEAC, AEAT).

## Layout

This is a **uv workspace**. Packages live under `packages/`:

- `packages/scrapers/` — `burrocrata-scrapers`, namespace `burrocrata.scrapers.*`.
  Currently ships the DGT scraper (CLI: `burrocrata-dgt`).
- `packages/datasets/` — *planned*, Step 3 of restructure.
- `packages/training/` — *planned*, Step 4 of restructure.

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

## Conventions

- Use `rg` (ripgrep) instead of `grep`; `fd` instead of `find`.
- Namespace packages: never add `packages/scrapers/src/burrocrata/__init__.py`.
- All imports under `burrocrata.*`.
