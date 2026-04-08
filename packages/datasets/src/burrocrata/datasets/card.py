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
