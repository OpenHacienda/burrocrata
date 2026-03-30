"""Export consultas to Markdown files and JSONL dataset."""

import json
import logging
import re
from pathlib import Path

import frontmatter

from .parser import Consulta

logger = logging.getLogger(__name__)

BASE_URL = "https://petete.tributos.hacienda.gob.es"

SFT_SYSTEM_PROMPT = (
    "Eres un asistente juridico experto en derecho tributario espanol. "
    "Respondes como la Direccion General de Tributos, citando la normativa aplicable."
)


def consulta_to_markdown(c: Consulta) -> str:
    """Render a Consulta as a Markdown string with YAML frontmatter."""
    body = "\n".join([
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
    ])
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


def consulta_to_sft(c: Consulta) -> dict:
    """Convert a consulta to a ChatML-style SFT training example."""
    user_content = c.hechos
    if c.cuestion:
        user_content += "\n\n" + c.cuestion
    return {
        "messages": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": c.contestacion},
        ],
        "metadata": {
            "numero": c.numero,
            "fecha": c.fecha_iso,
            "normativa": c.normativa,
            "organo": c.organo,
        },
    }


def export_sft_from_markdowns(data_dir: Path, output_path: Path) -> int:
    """Read all saved .md files and produce a JSONL dataset. Returns count."""
    consultas_dir = data_dir / "consultas"
    if not consultas_dir.exists():
        logger.error("No consultas directory found at %s", consultas_dir)
        return 0

    md_files = sorted(consultas_dir.rglob("*.md"))
    logger.info("Found %d markdown files", len(md_files))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for md_path in md_files:
            c = _parse_markdown(md_path)
            if c is None:
                continue
            if not c.contestacion.strip():
                logger.debug("Skipping %s (empty contestacion)", c.numero)
                continue
            record = consulta_to_sft(c)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Exported %d examples to %s", count, output_path)
    return count


def _parse_markdown(path: Path) -> Consulta | None:
    """Parse a saved Markdown file back into a Consulta."""
    try:
        post = frontmatter.load(path)
    except Exception:
        logger.warning("Failed to parse frontmatter from %s", path)
        return None

    fm = post.metadata
    body = post.content

    def _extract_section(header: str) -> str:
        pattern = rf"## {re.escape(header)}\n\n(.*?)(?=\n## |\Z)"
        m = re.search(pattern, body, re.DOTALL)
        return m.group(1).strip() if m else ""

    # Convert fecha back to DD/MM/YYYY for Consulta
    fecha_iso = str(fm.get("fecha", ""))
    fecha_parts = fecha_iso.split("-")
    fecha = f"{fecha_parts[2]}/{fecha_parts[1]}/{fecha_parts[0]}" if len(fecha_parts) == 3 else fecha_iso

    return Consulta(
        numero=str(fm.get("numero", "")),
        organo=str(fm.get("organo", "")),
        fecha=fecha,
        normativa=str(fm.get("normativa", "")),
        hechos=_extract_section("Descripcion de hechos"),
        cuestion=_extract_section("Cuestion planteada"),
        contestacion=_extract_section("Contestacion"),
    )
