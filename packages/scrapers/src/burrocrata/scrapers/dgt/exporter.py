"""Export consultas to Markdown files."""

import logging
from pathlib import Path

import frontmatter

from .parser import Consulta

logger = logging.getLogger(__name__)

BASE_URL = "https://petete.tributos.hacienda.gob.es"


def consulta_to_markdown(c: Consulta) -> str:
    """Render a Consulta as a Markdown string with YAML frontmatter."""
    body = "\n".join(
        [
            f"# Consulta Vinculante {c.numero}",
            "",
            "## Descripcion de hechos",
            "",
            c.hechos,
            "",
            "## Cuestion planteada",
            "",
            c.cuestion,
            "",
            "## Contestacion",
            "",
            c.contestacion,
            "",
        ]
    )
    post = frontmatter.Post(
        body,
        numero=c.numero,
        organo=c.organo,
        fecha=c.fecha_iso,
        normativa=c.normativa,
        url=f"{BASE_URL}/consultas/?num_consulta={c.numero}",
    )
    return frontmatter.dumps(post) + "\n"


def save_markdown(c: Consulta, data_dir: Path) -> Path:
    """Save a consulta as a Markdown file, returning the path."""
    year_dir = data_dir / "consultas" / c.year
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / f"{c.numero}.md"
    path.write_text(consulta_to_markdown(c), encoding="utf-8")
    return path


def save_raw_html(html_content: str, numero: str, year: str, data_dir: Path) -> Path:
    """Save the raw HTML of a document for later re-parsing."""
    raw_dir = data_dir / "raw" / year
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{numero}.html"
    path.write_text(html_content, encoding="utf-8")
    return path


def load_raw_html(numero: str, year: str, data_dir: Path) -> str | None:
    """Load cached raw HTML for a document, or None if not cached."""
    path = data_dir / "raw" / year / f"{numero}.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None
