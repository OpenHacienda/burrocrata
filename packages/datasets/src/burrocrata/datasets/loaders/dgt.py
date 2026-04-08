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
        if "anulado" in md_path.name.lower():
            continue
        row = load_consulta(md_path)
        if row is None:
            continue
        rows.append(row)
    return rows
