# Step 1: uv workspace shell + scrapers package rename

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert the single-package `python/` layout into a uv workspace rooted at the repo, with the DGT scraper living at `packages/scrapers/` under the `burrocrata.scrapers` namespace, renamed from `scrapper-dgt` → `burrocrata-scrapers` (CLI: `burrocrata-dgt`). No behavior changes.

**Architecture:** uv workspace with a root `pyproject.toml` listing `packages/scrapers` as the only member for now. `src/`-layout inside the package. All imports are already relative (`from .parser import ...`), so rename is primarily directory moves + `pyproject.toml` + Nix packaging + logger name string. `DGTSession`, parsers, exporters, CLI code itself are untouched.

**Tech Stack:** Python 3.11+, uv workspaces, click, setuptools, Nix (blueprint-based flake).

**Scope exclusions (explicit):**
- No `core/` extraction — that's Step 2.
- No test suite creation beyond one smoke import test.
- No change to scraping logic, CLI flags, on-disk data layout, or `data/dgt/` paths.
- The user has pre-existing uncommitted edits on `main` (`cli.py`, `scraper.py`, `flake.lock`). This plan runs on the `restructure/workspace` branch in the worktree, which is based off the committed state of `main` — do NOT try to incorporate those uncommitted edits. They will be resolved by the user separately.

---

## Preconditions

- Working directory: `/home/aldo/Dev/openhacienda/burrocrata/.worktrees/restructure-workspace`
- Branch: `restructure/workspace`
- The worktree has `docs/plans/2026-04-08-repo-restructure-design.md` committed (design doc).
- `uv` is available (via Nix devshell or system).

## Verification strategy

No real test suite exists today. Each task's verification is a mix of:
1. **Import smoke test**: `uv run python -c "import burrocrata.scrapers.dgt"` must succeed.
2. **CLI smoke test**: `uv run burrocrata-dgt --help` must print the click help.
3. **CLI functional test**: `uv run burrocrata-dgt stats --data-dir /tmp/nonexistent` must print `"No data found..."` and exit 0 (exercises click wiring + imports without touching the network).
4. **Nix build**: `nix build .#scrapers` must succeed (rename of the nix package).

Only network-dependent test (`burrocrata-dgt test`) is **not** part of verification — we don't want to hit PETETE in the loop.

---

## Task 1: Create workspace root `pyproject.toml`

**Files:**
- Create: `pyproject.toml` (repo root)

**Step 1: Write root pyproject**

```toml
[tool.uv.workspace]
members = ["packages/*"]
```

That's the whole file. No project table — the root is a workspace, not a package.

**Step 2: Verify**

Run: `uv --version` (sanity). Nothing else to verify yet; workspace is empty of members until Task 2.

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add uv workspace root"
```

---

## Task 2: Move `python/` → `packages/scrapers/` with src-layout

**Files:**
- Move tree: `python/scraper/` → `packages/scrapers/src/burrocrata/scrapers/`
- Move: `python/pyproject.toml` → `packages/scrapers/pyproject.toml` (and rewrite)
- Delete: `python/scraper/dgt/requirements.txt` (stale; deps live in pyproject)
- Delete: empty `python/` directory

**Step 1: Create new directory structure and move files**

```bash
mkdir -p packages/scrapers/src/burrocrata
git mv python/scraper packages/scrapers/src/burrocrata/scrapers
git mv python/pyproject.toml packages/scrapers/pyproject.toml
git rm packages/scrapers/src/burrocrata/scrapers/dgt/requirements.txt
rmdir python
```

**Step 2: Create namespace package marker**

`burrocrata` must be importable as a namespace package so future `burrocrata.datasets` / `burrocrata.training` can coexist. With PEP 420 implicit namespace packages, **do NOT** create `packages/scrapers/src/burrocrata/__init__.py`. Leave that directory without an `__init__.py`.

Verify:
```bash
ls packages/scrapers/src/burrocrata/
# should show: scrapers/
# should NOT show: __init__.py
test ! -f packages/scrapers/src/burrocrata/__init__.py && echo OK
```

**Step 3: Rewrite `packages/scrapers/pyproject.toml`**

Replace the entire file with:

```toml
[project]
name = "burrocrata-scrapers"
version = "0.1.0"
description = "Scrapers for Spanish tax/legal sources (DGT, ...)"
requires-python = ">=3.11"
dependencies = [
  "requests>=2.31.0",
  "beautifulsoup4>=4.12.0",
  "click>=8.1.0",
  "python-frontmatter>=1.1.0",
]

[project.scripts]
burrocrata-dgt = "burrocrata.scrapers.dgt.cli:main"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
include = ["burrocrata*"]
namespaces = true
```

Key changes from the old file:
- `name`: `scrapper-dgt` → `burrocrata-scrapers`
- `scripts`: `scrapper-dgt = "scraper.dgt.cli:main"` → `burrocrata-dgt = "burrocrata.scrapers.dgt.cli:main"`
- `setuptools.packages.find.where`: `["."]` → `["src"]` (src-layout)
- `include`: `["scraper*"]` → `["burrocrata*"]`
- `namespaces = true` for PEP 420 namespace packages

**Step 4: Update the logger name string in `cli.py`**

File: `packages/scrapers/src/burrocrata/scrapers/dgt/cli.py:22`

Change:
```python
logger = logging.getLogger("scraper.dgt")
```
to:
```python
logger = logging.getLogger("burrocrata.scrapers.dgt")
```

**Step 5: Update `__main__.py` docstrings** (cosmetic — module paths in docstrings are stale)

File: `packages/scrapers/src/burrocrata/scrapers/__main__.py`
Change docstring:
```python
"""Allow running as `python -m burrocrata.scrapers`."""
```

File: `packages/scrapers/src/burrocrata/scrapers/dgt/__main__.py`
Change docstring:
```python
"""Allow running as `python -m burrocrata.scrapers.dgt`."""
```

**Step 6: Sync workspace**

Run: `uv sync`
Expected: creates `.venv/` at repo root, installs `burrocrata-scrapers` as editable. No errors.

**Step 7: Import smoke test**

Run: `uv run python -c "import burrocrata.scrapers.dgt; import burrocrata.scrapers.dgt.cli; import burrocrata.scrapers.dgt.scraper; import burrocrata.scrapers.dgt.parser; import burrocrata.scrapers.dgt.exporter; print('OK')"`
Expected: prints `OK`, exit 0.

**Step 8: CLI smoke test**

Run: `uv run burrocrata-dgt --help`
Expected: prints click help listing subcommands `test`, `fetch`, `export-sft`, `stats`, `reparse`.

Run: `uv run burrocrata-dgt stats --data-dir /tmp/burrocrata-nonexistent`
Expected: prints `No data found. Run 'fetch' first.` and exits 0.

**Step 9: Commit**

```bash
git add -A
git commit -m "refactor: move scraper to packages/scrapers under burrocrata namespace

Renames scrapper-dgt → burrocrata-scrapers (CLI burrocrata-dgt).
Adopts src-layout with PEP 420 namespace package burrocrata.*
to allow future burrocrata.datasets / burrocrata.training siblings.
No behavior change; relative imports untouched."
```

---

## Task 3: Update Nix packaging

**Files:**
- Rename dir: `nix/packages/scrapper/` → `nix/packages/scrapers/`
- Modify: `nix/packages/scrapers/package.nix`
- Modify: `nix/packages/scrapers/default.nix` (no content change, but verify path)

**Step 1: Rename the package directory**

```bash
git mv nix/packages/scrapper nix/packages/scrapers
```

This renames the flake output from `packages.<system>.scrapper` to `packages.<system>.scrapers` (blueprint derives names from directory).

**Step 2: Rewrite `nix/packages/scrapers/package.nix`**

Replace with:

```nix
{
  lib,
  python3Packages,
}:
python3Packages.buildPythonApplication {
  pname = "burrocrata-scrapers";
  version = "0.1.0";
  pyproject = true;

  src = lib.cleanSource ../../../packages/scrapers;

  build-system = [ python3Packages.setuptools ];

  dependencies = with python3Packages; [
    requests
    beautifulsoup4
    click
    python-frontmatter
  ];

  doCheck = false;

  meta = {
    description = "Scrapers for Spanish tax/legal sources (DGT, ...)";
    mainProgram = "burrocrata-dgt";
  };
}
```

Changes from old file:
- `pname`: `scrapper-dgt` → `burrocrata-scrapers`
- `src`: `../../../python` → `../../../packages/scrapers`
- `mainProgram`: `scrapper-dgt` → `burrocrata-dgt`
- `meta.description` updated

**Step 3: Verify `default.nix` is unchanged and still correct**

File: `nix/packages/scrapers/default.nix` should still be:
```nix
{ pkgs, ... }:
pkgs.callPackage ./package.nix { }
```
No change needed — just confirm.

**Step 4: Nix build smoke test**

Run: `nix build .#scrapers -L`
Expected: builds successfully, produces `result/bin/burrocrata-dgt`.

Run: `./result/bin/burrocrata-dgt --help`
Expected: click help output (same as the uv smoke test).

Run: `./result/bin/burrocrata-dgt stats --data-dir /tmp/burrocrata-nonexistent`
Expected: `No data found. Run 'fetch' first.`

Clean up: `rm result`

**Step 5: Commit**

```bash
git add -A
git commit -m "nix: rename scrapper package to scrapers, point at packages/scrapers

Flake output packages.<system>.scrapper → packages.<system>.scrapers.
Built binary: burrocrata-dgt."
```

---

## Task 4: Verify workspace-level tooling still works

**Files:** none (verification only)

**Step 1: Ruff check**

Run: `uv run ruff check packages/scrapers`
Expected: either clean, or at worst the same warnings as before the move. If new warnings appear solely due to the move (e.g. import order), fix them and re-run.

**Step 2: Treefmt check**

Run: `nix develop -c treefmt --fail-on-change`
Expected: clean. If it reformats anything, commit the reformat as a separate chore commit.

**Step 3: Final holistic smoke test**

Run:
```bash
uv sync
uv run burrocrata-dgt --help
uv run burrocrata-dgt stats --data-dir /tmp/burrocrata-nonexistent
nix build .#scrapers -L && ./result/bin/burrocrata-dgt --help && rm result
```
Expected: all four succeed.

**Step 4: Confirm no orphaned references to old names**

Run: `rg -n "scrapper-dgt|python/scraper|from scraper|scraper\.dgt\.cli" --glob '!docs/**' --glob '!.worktrees/**' --glob '!uv.lock'`
Expected: **no matches**. (Matches inside `docs/` are OK — they may be historical references in the design doc.)

If any non-doc matches remain, fix them before moving on.

**Step 5: Commit any cleanup (if needed)**

If steps 1–4 produced follow-up fixes:
```bash
git add -A
git commit -m "chore: post-rename cleanup"
```

Otherwise skip.

---

## Task 5: Update CLAUDE.md (create if missing)

**Files:**
- Create or modify: `CLAUDE.md` (repo root)

**Step 1: Check if CLAUDE.md exists**

```bash
ls CLAUDE.md 2>/dev/null && echo EXISTS || echo MISSING
```

**Step 2: Write or append the following content**

If creating from scratch, the full file:

```markdown
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
```

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md with workspace layout and conventions"
```

---

## Done criteria for Step 1

- [ ] `uv sync` succeeds from repo root
- [ ] `uv run burrocrata-dgt --help` works
- [ ] `uv run burrocrata-dgt stats --data-dir /tmp/nonexistent` works
- [ ] `nix build .#scrapers` succeeds and produces `burrocrata-dgt` binary
- [ ] No references to `scrapper-dgt` or `scraper.dgt` outside `docs/`
- [ ] CLAUDE.md present
- [ ] Branch `restructure/workspace` has a clean commit history (one commit per task)

## Not in scope (reminders)

- `core/` extraction → Step 2
- `datasets` package → Step 3
- `training` package → Step 4
- `nix develop .#training` CUDA shell → Step 4
- Merging back to `main` → handled after all 5 steps via PR chain
