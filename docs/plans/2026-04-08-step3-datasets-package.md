# Step 3: `packages/datasets/` — HF datasets builders

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a second workspace member `packages/datasets/` (`burrocrata-datasets`) that reads scraped markdown corpora and produces HuggingFace `datasets`-format parquet shards plus a `dataset_card.md`. First builder: `dgt-sft` (chat-style SFT). Remove the old `export_sft_from_markdowns` JSONL exporter from `scrapers/dgt/exporter.py` and the `export-sft` command from `burrocrata-dgt`.

**Architecture:**
- `burrocrata-datasets` is **independent of `burrocrata-scrapers`**: it reads markdown + frontmatter directly, no Python import from scrapers. This keeps the two packages decoupled and lets you build datasets on machines that never installed scraper deps.
- Builder pattern: each source/format pair is a function `build(input_dir, output_dir, **opts) -> BuildResult` registered by name.
- Output is **HuggingFace `datasets` save_to_disk format** (parquet shards + `dataset_info.json`) accompanied by a sibling `dataset_card.md` recording: source file count, content hash, split seed, filters applied, build timestamp, git SHA, builder name + version.
- Splits done at build time with a fixed seed; recorded in the card so training is reproducible.
- One CLI: `burrocrata-datasets build <builder> --input <dir> --output <dir> [--seed N] [--val-frac F]`.

**Tech Stack:** Python 3.11+, `datasets` (HF), `python-frontmatter`, `click`, `pytest`.

---

## What lives where after Step 3

| Concern | Location |
|---|---|
| Render `Consulta` → markdown file (scraper output) | `scrapers/dgt/exporter.py` (`consulta_to_markdown`, `save_markdown`, `save_raw_html`, `load_raw_html`) — **stays** |
| Read markdown back into a row dict | `datasets/loaders/dgt.py` (new) |
| SFT chat formatter (system/user/assistant) | `datasets/formats/sft_chat.py` (new) — generic, not DGT-specific |
| DGT SFT builder | `datasets/builders/dgt_sft.py` (new) |
| Dataset card writer | `datasets/card.py` (new) |
| Builder registry | `datasets/builders/__init__.py` (new) |
| CLI | `datasets/cli.py` (new) |

**Removed in this step:** `consulta_to_sft`, `export_sft_from_markdowns`, `_parse_markdown`, `SFT_SYSTEM_PROMPT` from `scrapers/dgt/exporter.py`. The `export-sft` click command from `scrapers/dgt/cli.py`.

---

## Task 1: scaffold `packages/datasets/`

**Files:**
- Create: `packages/datasets/pyproject.toml`
- Create: `packages/datasets/src/burrocrata/datasets/__init__.py`
- Create: `packages/datasets/src/burrocrata/datasets/builders/__init__.py`
- Create: `packages/datasets/src/burrocrata/datasets/formats/__init__.py`
- Create: `packages/datasets/src/burrocrata/datasets/loaders/__init__.py`
- Create: `packages/datasets/tests/__init__.py`

**Step 1: pyproject.toml**

```toml
[project]
name = "burrocrata-datasets"
version = "0.1.0"
description = "Dataset builders for burrocrata corpora (HF datasets format)"
requires-python = ">=3.11"
dependencies = [
  "datasets>=2.18.0",
  "python-frontmatter>=1.1.0",
  "click>=8.1.0",
]

[project.scripts]
burrocrata-datasets = "burrocrata.datasets.cli:main"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
include = ["burrocrata*"]
namespaces = true

[dependency-groups]
dev = ["pytest>=8.0"]
```

**Step 2: empty `__init__.py` modules**

Each of the four `__init__.py` files in `src/burrocrata/datasets/{,builders,formats,loaders}/` is a one-line docstring:

```python
"""<short description>"""
```

`tests/__init__.py` is empty.

**Important:** Do NOT create `packages/datasets/src/burrocrata/__init__.py`. PEP 420 namespace package — same rule as scrapers.

**Step 3: workspace already auto-discovers `packages/*`**

The root `pyproject.toml` from Step 1 says `members = ["packages/*"]`, so no edit is needed there.

**Step 4: Verify**

```bash
uv sync
uv run python -c "import burrocrata.datasets; import burrocrata.scrapers; print('OK')"
```
Both must import. (`datasets` is the new package; `scrapers` must still work alongside it — this is the namespace-package coexistence test.)

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(datasets): scaffold burrocrata-datasets workspace package"
```

---

## Task 2: dataset card writer

**Files:**
- Create: `packages/datasets/src/burrocrata/datasets/card.py`

**Step 1: Implementation**

```python
"""Dataset card writer.

Records reproducibility metadata next to a built dataset:
- builder identity and version
- source file fingerprint (count + sha256 of sorted hashes)
- split seed and fractions
- filters applied (free-form list of strings)
- build timestamp + git SHA of the burrocrata repo
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BuildResult:
    """Returned by every builder; consumed by the CLI for the card + summary."""

    builder: str
    builder_version: str
    input_dir: Path
    output_dir: Path
    n_rows: int
    n_train: int
    n_val: int
    split_seed: int
    val_frac: float
    source_files: list[Path] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def _source_fingerprint(files: list[Path]) -> str:
    """Stable sha256 over sorted file content hashes."""
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def write_card(result: BuildResult) -> Path:
    """Write `dataset_card.md` next to the built dataset; return its path."""
    fingerprint = _source_fingerprint(result.source_files)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    git_sha = _git_sha()

    body = f"""# {result.builder}

Built dataset for the burrocrata pipeline.

## Provenance

| Field | Value |
|---|---|
| Builder | `{result.builder}` |
| Builder version | `{result.builder_version}` |
| Input directory | `{result.input_dir}` |
| Source files | {len(result.source_files)} |
| Source fingerprint | `{fingerprint}` |
| Build timestamp (UTC) | {timestamp} |
| burrocrata git SHA | `{git_sha}` |

## Splits

| Split | Rows |
|---|---|
| train | {result.n_train} |
| validation | {result.n_val} |
| **total** | **{result.n_rows}** |

- Seed: `{result.split_seed}`
- Validation fraction: `{result.val_frac}`

## Filters applied

{chr(10).join(f"- {f}" for f in result.filters) if result.filters else "_none_"}

## Extra

```json
{json.dumps(result.extra, indent=2, ensure_ascii=False)}
```

## Loading

```python
from datasets import load_from_disk
ds = load_from_disk("{result.output_dir}")
```
"""
    out = result.output_dir / "dataset_card.md"
    out.write_text(body, encoding="utf-8")
    return out
```

**Step 2: Verify imports**

```bash
uv run python -c "from burrocrata.datasets.card import BuildResult, write_card; print('OK')"
```

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(datasets): add dataset card writer with provenance metadata"
```

---

## Task 3: DGT markdown loader

**Files:**
- Create: `packages/datasets/src/burrocrata/datasets/loaders/dgt.py`

This is the only DGT-specific reader. Independent of `burrocrata.scrapers` — it reads markdown + frontmatter directly.

**Step 1: Implementation**

```python
"""Read DGT consulta markdown files (with frontmatter) into row dicts.

Mirror of the scraper's exporter format, but standalone — datasets does
not depend on scrapers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import frontmatter

logger = logging.getLogger(__name__)


@dataclass
class ConsultaRow:
    numero: str
    organo: str
    fecha_iso: str
    normativa: str
    hechos: str
    cuestion: str
    contestacion: str
    source_path: Path


def _extract_section(body: str, header: str) -> str:
    pattern = rf"## {re.escape(header)}\n\n(.*?)(?=\n## |\Z)"
    m = re.search(pattern, body, re.DOTALL)
    return m.group(1).strip() if m else ""


def load_consulta(path: Path) -> ConsultaRow | None:
    """Parse a single DGT consulta markdown file. Returns None on failure."""
    try:
        post = frontmatter.load(path)
    except Exception:
        logger.warning("Failed to parse frontmatter from %s", path)
        return None

    fm = post.metadata
    body = post.content

    return ConsultaRow(
        numero=str(fm.get("numero", "")),
        organo=str(fm.get("organo", "")),
        fecha_iso=str(fm.get("fecha", "")),
        normativa=str(fm.get("normativa", "")),
        hechos=_extract_section(body, "Descripcion de hechos"),
        cuestion=_extract_section(body, "Cuestion planteada"),
        contestacion=_extract_section(body, "Contestacion"),
        source_path=path,
    )


def iter_consultas(input_dir: Path) -> list[ConsultaRow]:
    """Walk ``<input_dir>/consultas/**/*.md`` and load each file.

    Skips files whose stem ends in 'anulado' marker (the scraper saves
    anulados with the suffix in the filename).
    """
    consultas_dir = input_dir / "consultas"
    if not consultas_dir.exists():
        raise FileNotFoundError(f"No 'consultas' subdirectory under {input_dir}")

    rows: list[ConsultaRow] = []
    for md_path in sorted(consultas_dir.rglob("*.md")):
        # The scraper writes anulado markers into the filename like
        # "V0119-26 Número de consulta anulado.md" — skip them at load time.
        if "anulado" in md_path.name.lower():
            continue
        row = load_consulta(md_path)
        if row is None:
            continue
        rows.append(row)
    return rows
```

**Step 2: Verify against the real corpus**

```bash
uv run python -c "
from pathlib import Path
from burrocrata.datasets.loaders.dgt import iter_consultas
rows = iter_consultas(Path('data/dgt'))
print(f'{len(rows)} rows; first numero: {rows[0].numero if rows else None}')
"
```

Expected: a non-zero count (the worktree has scraped 2026 consultas under `data/dgt/consultas/2026/`). If 0, investigate before moving on.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(datasets): add standalone DGT markdown loader"
```

---

## Task 4: SFT chat formatter (generic)

**Files:**
- Create: `packages/datasets/src/burrocrata/datasets/formats/sft_chat.py`

**Step 1: Implementation**

```python
"""Generic SFT chat-format helpers (system/user/assistant messages).

Source-agnostic. Each builder calls ``to_chat_example`` with strings.
"""

from __future__ import annotations


def to_chat_example(
    system: str,
    user: str,
    assistant: str,
    metadata: dict | None = None,
) -> dict:
    """Build one ChatML-style row: messages + metadata."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": metadata or {},
    }
```

**Step 2: Commit**

```bash
git add -A
git commit -m "feat(datasets): add generic SFT chat formatter"
```

---

## Task 5: DGT SFT builder

**Files:**
- Create: `packages/datasets/src/burrocrata/datasets/builders/dgt_sft.py`
- Modify: `packages/datasets/src/burrocrata/datasets/builders/__init__.py` (registry)

**Step 1: Builder implementation**

```python
"""Builder: DGT consultas → SFT chat dataset (HF datasets format)."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict

from ..card import BuildResult
from ..formats.sft_chat import to_chat_example
from ..loaders.dgt import iter_consultas

BUILDER_NAME = "dgt-sft"
BUILDER_VERSION = "1"

SYSTEM_PROMPT = (
    "Eres un asistente juridico experto en derecho tributario espanol. "
    "Respondes como la Direccion General de Tributos, citando la normativa aplicable."
)


def build(
    input_dir: Path,
    output_dir: Path,
    val_frac: float = 0.02,
    seed: int = 42,
) -> BuildResult:
    """Build the DGT SFT dataset and persist it to ``output_dir``.

    Filters:
    - skips entries with empty contestacion
    - skips files whose name marks them as anulado (handled in loader)
    """
    rows = iter_consultas(input_dir)

    examples: list[dict] = []
    source_files: list[Path] = []
    skipped_empty_contestacion = 0

    for r in rows:
        if not r.contestacion.strip():
            skipped_empty_contestacion += 1
            continue
        user_content = r.hechos
        if r.cuestion:
            user_content += "\n\n" + r.cuestion
        examples.append(
            to_chat_example(
                system=SYSTEM_PROMPT,
                user=user_content,
                assistant=r.contestacion,
                metadata={
                    "numero": r.numero,
                    "fecha": r.fecha_iso,
                    "normativa": r.normativa,
                    "organo": r.organo,
                },
            )
        )
        source_files.append(r.source_path)

    if not examples:
        raise RuntimeError(f"No usable consultas found under {input_dir}")

    ds = Dataset.from_list(examples)
    splits = ds.train_test_split(test_size=val_frac, seed=seed, shuffle=True)
    dsd = DatasetDict(train=splits["train"], validation=splits["test"])

    output_dir.mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(str(output_dir))

    return BuildResult(
        builder=BUILDER_NAME,
        builder_version=BUILDER_VERSION,
        input_dir=input_dir,
        output_dir=output_dir,
        n_rows=len(examples),
        n_train=len(dsd["train"]),
        n_val=len(dsd["validation"]),
        split_seed=seed,
        val_frac=val_frac,
        source_files=source_files,
        filters=[
            "drop entries with empty contestacion",
            "drop files whose name contains 'anulado'",
        ],
        extra={
            "skipped_empty_contestacion": skipped_empty_contestacion,
            "system_prompt_chars": len(SYSTEM_PROMPT),
        },
    )
```

**Step 2: Builder registry**

`packages/datasets/src/burrocrata/datasets/builders/__init__.py`:

```python
"""Builder registry. Look up by name."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from . import dgt_sft
from ..card import BuildResult

Builder = Callable[..., BuildResult]

REGISTRY: dict[str, Builder] = {
    dgt_sft.BUILDER_NAME: dgt_sft.build,
}


def get(name: str) -> Builder:
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown builder {name!r}. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]


def names() -> list[str]:
    return sorted(REGISTRY)
```

**Step 3: Verify**

```bash
uv sync
uv run python -c "
from burrocrata.datasets.builders import REGISTRY, get, names
print(names())
build = get('dgt-sft')
print(build)
"
```
Expected: `['dgt-sft']` and a function reference.

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(datasets): add dgt-sft builder + registry"
```

---

## Task 6: CLI

**Files:**
- Create: `packages/datasets/src/burrocrata/datasets/cli.py`
- Create: `packages/datasets/src/burrocrata/datasets/__main__.py`

**Step 1: CLI**

```python
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
```

**Step 2: `__main__.py`**

```python
"""Allow running as `python -m burrocrata.datasets`."""

from .cli import main

main()
```

**Step 3: Verify**

```bash
uv sync
uv run burrocrata-datasets --help
uv run burrocrata-datasets list
uv run burrocrata-datasets build --help
```

The `list` command must print `dgt-sft`. The `build --help` must show all four options.

**Real build dry-run** (should succeed against the existing scraped corpus):

```bash
uv run burrocrata-datasets build dgt-sft --input data/dgt --output /tmp/burrocrata-test-dgt-sft
ls /tmp/burrocrata-test-dgt-sft/
cat /tmp/burrocrata-test-dgt-sft/dataset_card.md
uv run python -c "
from datasets import load_from_disk
ds = load_from_disk('/tmp/burrocrata-test-dgt-sft')
print(ds)
print(ds['train'][0]['messages'][1]['content'][:200])
"
rm -rf /tmp/burrocrata-test-dgt-sft
```

Expected:
- Build succeeds, prints row counts.
- Output directory contains `train/`, `validation/`, `dataset_dict.json`, and `dataset_card.md`.
- `load_from_disk` round-trips fine; first user message is non-empty Spanish text.

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(datasets): add burrocrata-datasets CLI (list, build)"
```

---

## Task 7: remove old SFT exporter from scrapers

**Files:**
- Modify: `packages/scrapers/src/burrocrata/scrapers/dgt/exporter.py`
- Modify: `packages/scrapers/src/burrocrata/scrapers/dgt/cli.py`

**Step 1: Trim `exporter.py`**

Remove from `dgt/exporter.py`:
- `import json` (no longer needed if nothing else uses it; check first)
- `import re` (only used by `_parse_markdown`)
- `SFT_SYSTEM_PROMPT`
- `consulta_to_sft`
- `export_sft_from_markdowns`
- `_parse_markdown`

Keep:
- `consulta_to_markdown`
- `save_markdown`
- `save_raw_html`
- `load_raw_html`
- The `BASE_URL` constant
- `frontmatter` import (used by `consulta_to_markdown`)

After the trim, `re` and `json` are unused in this file — remove them. The `logging` and `Path` imports stay; `frontmatter` stays.

**Step 2: Trim `cli.py`**

In `packages/scrapers/src/burrocrata/scrapers/dgt/cli.py`:
- Remove the `export_sft` click command (around the `@cli.command("export-sft")` decorator).
- Remove `export_sft_from_markdowns` from the `from .exporter import ...` line.

Verify no orphan reference: `rg -n 'export_sft|export-sft' packages/scrapers/src` — should be empty.

**Step 3: Verify scrapers still work**

```bash
uv sync
uv run burrocrata-dgt --help          # 'export-sft' must NOT appear
uv run burrocrata-dgt fetch --help    # unchanged
uv run burrocrata-dgt stats --data-dir /tmp/burrocrata-nonexistent
uv run --with pytest pytest packages/scrapers/tests -v
uv run ruff check packages/scrapers
```

The `--help` listing must show: `fetch`, `reparse`, `stats`, `test` — and **no longer** show `export-sft`.

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor(scrapers): remove SFT exporter (moved to burrocrata-datasets)

The export-sft command and consulta_to_sft / export_sft_from_markdowns
helpers are gone from scrapers. Use \`burrocrata-datasets build dgt-sft\`
instead, which produces HF datasets parquet format with a dataset card."
```

---

## Task 8: unit tests for the loader and builder

**Files:**
- Create: `packages/datasets/tests/fixtures/data/dgt/consultas/2099/V0001-99.md`
- Create: `packages/datasets/tests/fixtures/data/dgt/consultas/2099/V0002-99.md`
- Create: `packages/datasets/tests/fixtures/data/dgt/consultas/2099/V0003-99 Número de consulta anulado.md`
- Create: `packages/datasets/tests/fixtures/data/dgt/consultas/2099/V0004-99.md` (empty contestacion)
- Create: `packages/datasets/tests/test_dgt_loader.py`
- Create: `packages/datasets/tests/test_dgt_sft_builder.py`

**Step 1: Fixture markdown files**

`V0001-99.md`:

```markdown
---
numero: V0001-99
organo: Test Organo
fecha: 2099-01-01
normativa: Test Normativa
url: https://example.test/?num_consulta=V0001-99
---
# Consulta Vinculante V0001-99

## Descripcion de hechos

Hechos del primer caso de prueba.

## Cuestion planteada

Cual es la cuestion?

## Contestacion

Esta es la contestacion del primer caso.
```

`V0002-99.md`: same shape, `numero: V0002-99`, different text in each section.

`V0003-99 Número de consulta anulado.md`: same shape but with anulado in the filename — should be skipped by the loader.

`V0004-99.md`: valid frontmatter and sections, but the **Contestacion** section body is empty (just `## Contestacion\n\n\n## ...` or end-of-file). The loader returns it; the builder filters it out as `skipped_empty_contestacion`.

**Step 2: Loader test**

`packages/datasets/tests/test_dgt_loader.py`:

```python
"""Tests for the standalone DGT markdown loader."""

from pathlib import Path

from burrocrata.datasets.loaders.dgt import iter_consultas, load_consulta

FIXTURE = Path(__file__).parent / "fixtures" / "data" / "dgt"


def test_iter_skips_anulado_files():
    rows = iter_consultas(FIXTURE)
    numeros = {r.numero for r in rows}
    assert "V0001-99" in numeros
    assert "V0002-99" in numeros
    assert "V0003-99" not in numeros  # anulado in filename
    # V0004-99 has empty contestacion but the loader still returns it;
    # filtering happens at the builder layer.
    assert "V0004-99" in numeros


def test_load_consulta_extracts_sections():
    p = FIXTURE / "consultas" / "2099" / "V0001-99.md"
    row = load_consulta(p)
    assert row is not None
    assert row.numero == "V0001-99"
    assert row.organo == "Test Organo"
    assert row.fecha_iso == "2099-01-01"
    assert "primer caso" in row.hechos
    assert "cuestion" in row.cuestion.lower()
    assert "contestacion" in row.contestacion.lower()


def test_iter_raises_when_no_consultas_dir(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        iter_consultas(tmp_path)
```

**Step 3: Builder test**

`packages/datasets/tests/test_dgt_sft_builder.py`:

```python
"""Tests for the dgt-sft builder."""

from pathlib import Path

from datasets import load_from_disk

from burrocrata.datasets.builders.dgt_sft import build, BUILDER_NAME
from burrocrata.datasets.card import write_card

FIXTURE = Path(__file__).parent / "fixtures" / "data" / "dgt"


def test_build_produces_loadable_dataset_and_filters_empty(tmp_path):
    out = tmp_path / "out"
    result = build(input_dir=FIXTURE, output_dir=out, val_frac=0.5, seed=7)

    assert result.builder == BUILDER_NAME
    # 2 valid rows: V0001 and V0002. V0003 anulado, V0004 empty contestacion.
    assert result.n_rows == 2
    assert result.n_train + result.n_val == 2
    assert result.extra["skipped_empty_contestacion"] == 1

    dsd = load_from_disk(str(out))
    assert set(dsd.keys()) == {"train", "validation"}

    sample = dsd["train"][0] if len(dsd["train"]) else dsd["validation"][0]
    roles = [m["role"] for m in sample["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert sample["metadata"]["numero"].startswith("V000")


def test_card_writes_with_provenance(tmp_path):
    out = tmp_path / "out"
    result = build(input_dir=FIXTURE, output_dir=out, val_frac=0.5, seed=7)
    card_path = write_card(result)
    assert card_path.exists()
    text = card_path.read_text(encoding="utf-8")
    assert "dgt-sft" in text
    assert "Source fingerprint" in text
    assert "Splits" in text
```

**Step 4: Run the tests**

```bash
uv run --with pytest pytest packages/datasets/tests -v
```

Expected: 5 passed (3 loader + 2 builder).

**Step 5: Commit**

```bash
git add -A
git commit -m "test(datasets): add loader + dgt-sft builder unit tests with fixtures"
```

---

## Task 9: Nix package for datasets (optional but expected)

Mirror the existing `nix/packages/scrapers/` for the new package so `nix build .#datasets` works.

**Files:**
- Create: `nix/packages/datasets/default.nix`
- Create: `nix/packages/datasets/package.nix`

**Step 1: `default.nix`**

```nix
{ pkgs, ... }:
pkgs.callPackage ./package.nix { }
```

**Step 2: `package.nix`**

```nix
{
  lib,
  python3Packages,
}:
python3Packages.buildPythonApplication {
  pname = "burrocrata-datasets";
  version = "0.1.0";
  pyproject = true;

  src = lib.cleanSource ../../../packages/datasets;

  build-system = [ python3Packages.setuptools ];

  dependencies = with python3Packages; [
    datasets
    python-frontmatter
    click
  ];

  doCheck = false;

  meta = {
    description = "Dataset builders for burrocrata corpora";
    mainProgram = "burrocrata-datasets";
  };
}
```

**Step 3: Verify**

```bash
nix build .#datasets -L
./result/bin/burrocrata-datasets --help
./result/bin/burrocrata-datasets list
rm result
```

If `python3Packages.datasets` is not in nixpkgs by that name, the build will fail with an attribute error. In that case, **STOP and report** — don't try to add overlays. The user will decide whether to skip Nix packaging for datasets in this PR or do something else.

**Step 4: Commit**

```bash
git add -A
git commit -m "nix: package burrocrata-datasets"
```

---

## Task 10: final verification

**Step 1: Workspace-wide checks**

```bash
uv sync
uv run burrocrata-dgt --help          # no export-sft
uv run burrocrata-datasets --help
uv run burrocrata-datasets list       # dgt-sft
uv run --with pytest pytest packages/scrapers/tests packages/datasets/tests -v
uv run ruff check packages/scrapers packages/datasets
nix develop -c treefmt --fail-on-change
```

**Step 2: End-to-end real-corpus build**

```bash
uv run burrocrata-datasets build dgt-sft --input data/dgt --output /tmp/burrocrata-real-dgt-sft
cat /tmp/burrocrata-real-dgt-sft/dataset_card.md
rm -rf /tmp/burrocrata-real-dgt-sft
```

Expected: build succeeds, card shows the real source file count from `data/dgt/consultas/`.

**Step 3: Orphan sweep**

```bash
rg -n 'export_sft|consulta_to_sft|SFT_SYSTEM_PROMPT|_parse_markdown' \
  --glob '!docs/**' --glob '!.worktrees/**' packages/
```
Expected: matches **only** in:
- `packages/datasets/src/burrocrata/datasets/loaders/dgt.py` — no, this file doesn't reference any of the old names
- (none — all of these names should be gone outside of docs)

If matches appear, fix before proceeding.

**Step 4: treefmt cleanup if needed**

If `treefmt --fail-on-change` reformatted anything, commit as `chore: treefmt` and re-run until clean.

---

## Done criteria for Step 3

- [ ] `packages/datasets/` workspace member exists with `pyproject.toml`, `src/burrocrata/datasets/{builders,formats,loaders,card,cli,__main__}.py`, `tests/`
- [ ] `burrocrata-datasets list` prints `dgt-sft`
- [ ] `burrocrata-datasets build dgt-sft --input data/dgt --output X` produces a valid HF dataset + `dataset_card.md`
- [ ] `pytest packages/datasets/tests` passes (≥ 5 tests)
- [ ] Old `export_sft_from_markdowns`, `consulta_to_sft`, `_parse_markdown`, `SFT_SYSTEM_PROMPT` removed from scrapers
- [ ] `burrocrata-dgt --help` no longer shows `export-sft`
- [ ] `nix build .#datasets` succeeds (or, if `python3Packages.datasets` is missing, this task is deferred per the stop rule)
- [ ] All ruff and treefmt checks clean

## Not in scope

- Other builders (BOE, TEAC, RAG chunks, DPO) — wait until first need.
- HF Hub push — Step 4 or later.
- `dgt-eval` rubric eval set — Phase 2 of the eval strategy, after first training run.
- Reading scraped data straight from HTML — markdowns are the canonical source.
- Renaming `data/` or `datasets/` directory layout.
