"""CLI for the DGT consultas vinculantes scraper."""

import json
import logging
import sys
import time
from pathlib import Path

import click

from .exporter import export_sft_from_markdowns, save_markdown
from .parser import parse_document, parse_search_results
from .scraper import DGTSession

DEFAULT_DATA_DIR = Path("data/dgt")

logger = logging.getLogger("scraper.dgt")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_checkpoint(data_dir: Path) -> dict:
    cp_path = data_dir / "checkpoint.json"
    if cp_path.exists():
        return json.loads(cp_path.read_text(encoding="utf-8"))
    return {}


def _save_checkpoint(data_dir: Path, checkpoint: dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    cp_path = data_dir / "checkpoint.json"
    cp_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")


def _existing_numeros(data_dir: Path) -> set[str]:
    """Return the set of already-downloaded consulta numbers."""
    consultas_dir = data_dir / "consultas"
    if not consultas_dir.exists():
        return set()
    return {p.stem for p in consultas_dir.rglob("*.md")}


def _format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h{mins:02d}m"


def _fetch_year(
    session: DGTSession,
    year: int,
    data_dir: Path,
    force: bool = False,
    existing: set[str] | None = None,
    checkpoint: dict | None = None,
) -> tuple[int, int]:
    """Fetch all consultas for a given year. Returns (fetched, errors)."""
    date_start = f"01/01/{year}"
    date_end = f"31/12/{year}"

    if existing is None:
        existing = _existing_numeros(data_dir) if not force else set()
    if checkpoint is None:
        checkpoint = _load_checkpoint(data_dir)

    # First search to get totals
    html = session.search(page=1, date_start=date_start, date_end=date_end)
    page_data = parse_search_results(html)
    total_results = page_data.total_results
    total_pages = page_data.total_pages

    if total_results == 0:
        logger.info("[%d] No results found", year)
        return 0, 0

    logger.info("[%d] Found %d consultas across %d pages", year, total_results, total_pages)

    fetched = 0
    skipped = 0
    errors = 0
    start_time = time.monotonic()

    cp_key = f"year_{year}"
    start_page = checkpoint.get(cp_key, {}).get("page", 1)
    if start_page > 1:
        logger.info("[%d] Resuming from page %d", year, start_page)

    for page_num in range(start_page, total_pages + 1):
        if page_num != 1 or start_page > 1:
            html = session.search(page=page_num, date_start=date_start, date_end=date_end)
            page_data = parse_search_results(html)

        for result in page_data.results:
            if result.numero and result.numero in existing:
                logger.debug("Skipping %s (already exists)", result.numero)
                skipped += 1
                continue

            try:
                doc_html = session.fetch_document(result.doc_id)
                consulta = parse_document(doc_html)

                if not consulta.numero:
                    # Use numero from search results as fallback
                    consulta.numero = result.numero

                if not consulta.numero:
                    logger.warning("Document %s has no numero, skipping", result.doc_id)
                    errors += 1
                    continue

                path = save_markdown(consulta, data_dir)
                fetched += 1
                logger.debug("Saved %s -> %s", consulta.numero, path)
            except Exception:
                logger.exception("Error fetching document %s", result.doc_id)
                errors += 1

        # Progress
        elapsed = time.monotonic() - start_time
        rate = fetched / elapsed if elapsed > 0 else 0
        remaining = total_results - fetched - skipped - errors
        eta_str = _format_eta(remaining / rate) if rate > 0 else "--"
        click.echo(
            f"[{year}] Page {page_num}/{total_pages} | "
            f"Fetched: {fetched}/{total_results} | "
            f"Skipped: {skipped} | "
            f"Errors: {errors} | "
            f"ETA: {eta_str}"
        )

        # Save checkpoint
        checkpoint[cp_key] = {"page": page_num + 1}
        _save_checkpoint(data_dir, checkpoint)

    # Clear checkpoint for this year on success
    checkpoint.pop(cp_key, None)
    _save_checkpoint(data_dir, checkpoint)

    return fetched, errors


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """DGT Consultas Vinculantes scraper."""
    _setup_logging(verbose)


@cli.command()
@click.option("--rate-limit", default=0.5, help="Requests per second (default: 0.5)")
def test(rate_limit: float) -> None:
    """Test connectivity to the PETETE server."""
    session = DGTSession(rate_limit=rate_limit)
    click.echo("Initializing session…")
    try:
        session.init()
        click.echo("Session OK. Testing search…")
        html = session.search(page=1)
        page = parse_search_results(html)
        click.echo(
            f"Search OK. Total consultas vinculantes: {page.total_results} "
            f"({page.total_pages} pages)"
        )
        if page.results:
            r = page.results[0]
            click.echo(f"First result: {r.numero} (doc_id={r.doc_id})")
            click.echo("Fetching first document…")
            doc_html = session.fetch_document(r.doc_id)
            consulta = parse_document(doc_html)
            click.echo(f"Document OK: {consulta.numero} ({consulta.fecha})")
            click.echo(f"  Organo: {consulta.organo}")
            click.echo(f"  Normativa: {consulta.normativa}")
            click.echo(f"  Hechos: {consulta.hechos[:120]}…")
        click.echo("All connectivity tests passed.")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--year", type=int, help="Download a specific year")
@click.option("--all", "fetch_all", is_flag=True, help="Download all years (1997-present)")
@click.option("--update", is_flag=True, help="Download only the current year (incremental)")
@click.option("--data-dir", type=click.Path(), default=str(DEFAULT_DATA_DIR))
@click.option("--rate-limit", default=0.5, help="Requests per second")
@click.option("--force", is_flag=True, help="Re-download even if file exists")
def fetch(
    year: int | None,
    fetch_all: bool,
    update: bool,
    data_dir: str,
    rate_limit: float,
    force: bool,
) -> None:
    """Fetch consultas vinculantes from PETETE."""
    data_path = Path(data_dir)
    session = DGTSession(rate_limit=rate_limit)

    if not year and not fetch_all and not update:
        click.echo("Specify --year YYYY, --all, or --update", err=True)
        sys.exit(1)

    if year:
        years = [year]
    elif update:
        from datetime import date
        years = [date.today().year]
    else:
        from datetime import date
        years = list(range(1997, date.today().year + 1))

    existing = _existing_numeros(data_path) if not force else set()
    checkpoint = _load_checkpoint(data_path)

    total_fetched = 0
    total_errors = 0

    for y in years:
        fetched, errors = _fetch_year(
            session, y, data_path, force=force, existing=existing, checkpoint=checkpoint,
        )
        total_fetched += fetched
        total_errors += errors

    click.echo(f"\nDone. Fetched: {total_fetched} | Errors: {total_errors}")

    # Save metadata
    _save_metadata(data_path, total_fetched, total_errors)


def _save_metadata(data_dir: Path, fetched: int, errors: int) -> None:
    import datetime
    meta_path = data_dir / "metadata.json"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Load existing metadata to accumulate lifetime stats
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            pass
    meta["last_run"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta["last_fetched"] = fetched
    meta["last_errors"] = errors
    meta["lifetime_fetched"] = meta.get("lifetime_fetched", 0) + fetched
    meta["lifetime_errors"] = meta.get("lifetime_errors", 0) + errors
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


@cli.command("export-sft")
@click.option("--data-dir", type=click.Path(), default=str(DEFAULT_DATA_DIR))
@click.option("--output", "-o", type=click.Path(), default=str(DEFAULT_DATA_DIR / "sft_dataset.jsonl"))
def export_sft(data_dir: str, output: str) -> None:
    """Export downloaded consultas to JSONL for SFT fine-tuning."""
    count = export_sft_from_markdowns(Path(data_dir), Path(output))
    click.echo(f"Exported {count} examples to {output}")


@cli.command()
@click.option("--data-dir", type=click.Path(), default=str(DEFAULT_DATA_DIR))
def stats(data_dir: str) -> None:
    """Show stats about the downloaded corpus."""
    data_path = Path(data_dir)
    consultas_dir = data_path / "consultas"

    if not consultas_dir.exists():
        click.echo("No data found. Run 'fetch' first.")
        return

    total = 0
    by_year: dict[str, int] = {}
    for md in consultas_dir.rglob("*.md"):
        total += 1
        year = md.parent.name
        by_year[year] = by_year.get(year, 0) + 1

    click.echo(f"Total consultas: {total}")
    click.echo(f"Years covered: {len(by_year)}")
    for y in sorted(by_year):
        click.echo(f"  {y}: {by_year[y]}")

    # Size
    total_size = sum(f.stat().st_size for f in consultas_dir.rglob("*.md"))
    if total_size > 1_000_000_000:
        size_str = f"{total_size / 1_000_000_000:.1f} GB"
    elif total_size > 1_000_000:
        size_str = f"{total_size / 1_000_000:.1f} MB"
    else:
        size_str = f"{total_size / 1_000:.1f} KB"
    click.echo(f"Total size: {size_str}")

    # Metadata
    meta_path = data_path / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        click.echo(f"Last run: {meta.get('last_run', 'unknown')}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
