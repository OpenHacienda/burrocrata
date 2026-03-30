"""CLI for the DGT consultas vinculantes scraper."""

import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click

from .exporter import export_sft_from_markdowns, save_markdown, save_raw_html
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


def _load_search_index(data_dir: Path, year: int) -> list[dict] | None:
    """Load cached search index for a year, or None if not cached."""
    path = data_dir / "search_index" / f"{year}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            return None
    return None


def _save_search_index(data_dir: Path, year: int, entries: list[dict]) -> None:
    """Save the full search index for a year."""
    idx_dir = data_dir / "search_index"
    idx_dir.mkdir(parents=True, exist_ok=True)
    path = idx_dir / f"{year}.json"
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")


def _existing_numeros(data_dir: Path) -> set[str]:
    """Return the set of already-downloaded consulta numbers (md or raw html)."""
    numeros: set[str] = set()
    consultas_dir = data_dir / "consultas"
    if consultas_dir.exists():
        numeros.update(p.stem for p in consultas_dir.rglob("*.md"))
    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        numeros.update(p.stem for p in raw_dir.rglob("*.html"))
    return numeros


def _format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h{mins:02d}m"


def _build_search_index(
    session: DGTSession,
    year: int,
    data_dir: Path,
) -> list[dict]:
    """Paginate through all search results for a year and return a doc index.

    Uses a cached index if available; otherwise paginates the server and
    caches the result so subsequent runs skip the pagination entirely.
    """
    cached = _load_search_index(data_dir, year)
    if cached is not None:
        logger.info("[%d] Using cached search index (%d entries)", year, len(cached))
        return cached

    date_start = f"01/01/{year}"
    date_end = f"31/12/{year}"

    html = session.search(page=1, date_start=date_start, date_end=date_end)
    page_data = parse_search_results(html)
    total_pages = page_data.total_pages

    if page_data.total_results == 0:
        _save_search_index(data_dir, year, [])
        return []

    logger.info(
        "[%d] Building search index: %d results across %d pages",
        year, page_data.total_results, total_pages,
    )

    entries: list[dict] = []
    for result in page_data.results:
        entries.append({"doc_id": result.doc_id, "numero": result.numero})

    for page_num in range(2, total_pages + 1):
        html = session.search(page=page_num, date_start=date_start, date_end=date_end)
        page_data = parse_search_results(html)
        for result in page_data.results:
            entries.append({"doc_id": result.doc_id, "numero": result.numero})
        if page_num % 10 == 0 or page_num == total_pages:
            click.echo(f"[{year}] Indexing page {page_num}/{total_pages}")

    _save_search_index(data_dir, year, entries)
    logger.info("[%d] Search index cached (%d entries)", year, len(entries))
    return entries


def _fetch_one_doc(
    session: DGTSession,
    doc_id: str,
    numero_hint: str,
    data_dir: Path,
) -> tuple[str | None, str | None]:
    """Fetch and save a single document. Returns (numero, error_msg)."""
    doc_html = session.fetch_document(doc_id)
    consulta = parse_document(doc_html)

    if not consulta.numero:
        consulta.numero = numero_hint

    if not consulta.numero:
        return None, f"Document {doc_id} has no numero"

    save_raw_html(doc_html, consulta.numero, consulta.year, data_dir)
    save_markdown(consulta, data_dir)
    return consulta.numero, None


def _fetch_year(
    session: DGTSession,
    year: int,
    data_dir: Path,
    force: bool = False,
    existing: set[str] | None = None,
    checkpoint: dict | None = None,
    concurrency: int = 1,
) -> tuple[int, int]:
    """Fetch all consultas for a given year. Returns (fetched, errors)."""
    if existing is None:
        existing = _existing_numeros(data_dir) if not force else set()
    if checkpoint is None:
        checkpoint = _load_checkpoint(data_dir)

    entries = _build_search_index(session, year, data_dir)
    total_results = len(entries)

    if total_results == 0:
        logger.info("[%d] No results found", year)
        return 0, 0

    fetched = 0
    skipped = 0
    errors = 0
    start_time = time.monotonic()

    # Checkpoint: resume from last doc index
    cp_key = f"year_{year}"
    start_idx = checkpoint.get(cp_key, {}).get("doc_idx", 0)
    if start_idx > 0:
        logger.info("[%d] Resuming from document %d/%d", year, start_idx, total_results)

    # Filter out already-existing entries upfront
    work: list[tuple[int, dict]] = []
    for idx in range(start_idx, total_results):
        entry = entries[idx]
        if entry["numero"] and entry["numero"] in existing:
            logger.debug("Skipping %s (already exists)", entry["numero"])
            skipped += 1
        else:
            work.append((idx, entry))

    if not work:
        logger.info("[%d] All %d consultas already downloaded", year, total_results)
        checkpoint.pop(cp_key, None)
        _save_checkpoint(data_dir, checkpoint)
        return 0, 0

    logger.info("[%d] %d to fetch, %d skipped (concurrency=%d)", year, len(work), skipped, concurrency)

    # Thread-safe counters
    lock = threading.Lock()

    def _progress() -> None:
        done = fetched + errors
        if done % 20 == 0 or done == len(work):
            elapsed = time.monotonic() - start_time
            rate = fetched / elapsed if elapsed > 0 else 0
            remaining = len(work) - done
            eta_str = _format_eta(remaining / rate) if rate > 0 else "--"
            click.echo(
                f"[{year}] Done {done}/{len(work)} (of {total_results} total) | "
                f"Fetched: {fetched} | "
                f"Skipped: {skipped} | "
                f"Errors: {errors} | "
                f"ETA: {eta_str}"
            )

    if concurrency <= 1:
        # Sequential path — simpler, preserves order for checkpointing
        for work_pos, (idx, entry) in enumerate(work):
            try:
                numero, err = _fetch_one_doc(
                    session, entry["doc_id"], entry["numero"], data_dir,
                )
                if err or numero is None:
                    logger.warning(err or f"Document {entry['doc_id']} returned no numero")
                    errors += 1
                else:
                    existing.add(numero)
                    fetched += 1
                    logger.debug("Saved %s", numero)
            except Exception:
                logger.exception("Error fetching document %s", entry["doc_id"])
                errors += 1

            _progress()

            if (work_pos + 1) % 20 == 0:
                checkpoint[cp_key] = {"doc_idx": idx + 1}
                _save_checkpoint(data_dir, checkpoint)
    else:
        # Concurrent path — pipeline requests to overlap network latency
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_entry = {}
            for idx, entry in work:
                fut = pool.submit(
                    _fetch_one_doc,
                    session, entry["doc_id"], entry["numero"], data_dir,
                )
                future_to_entry[fut] = (idx, entry)

            for fut in as_completed(future_to_entry):
                idx, entry = future_to_entry[fut]
                try:
                    numero, err = fut.result()
                    with lock:
                        if err or numero is None:
                            logger.warning(err or f"Document {entry['doc_id']} returned no numero")
                            errors += 1
                        else:
                            existing.add(numero)
                            fetched += 1
                            logger.debug("Saved %s", numero)
                        _progress()
                except Exception:
                    with lock:
                        logger.exception("Error fetching document %s", entry["doc_id"])
                        errors += 1
                        _progress()

        # Checkpoint at the end for concurrent mode (order is non-deterministic)
        checkpoint[cp_key] = {"doc_idx": total_results}

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
@click.option("--concurrency", default=1, help="Concurrent document fetches (default: 1)")
@click.option("--force", is_flag=True, help="Re-download even if file exists")
def fetch(
    year: int | None,
    fetch_all: bool,
    update: bool,
    data_dir: str,
    rate_limit: float,
    concurrency: int,
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
            session, y, data_path, force=force, existing=existing,
            checkpoint=checkpoint, concurrency=concurrency,
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


@cli.command()
@click.option("--data-dir", type=click.Path(), default=str(DEFAULT_DATA_DIR))
def reparse(data_dir: str) -> None:
    """Re-generate all markdown files from cached raw HTML."""
    data_path = Path(data_dir)
    raw_dir = data_path / "raw"

    if not raw_dir.exists():
        click.echo("No raw HTML cache found. Run 'fetch' first.")
        return

    html_files = sorted(raw_dir.rglob("*.html"))
    click.echo(f"Found {len(html_files)} cached HTML files")

    reparsed = 0
    errors = 0
    for html_path in html_files:
        try:
            html_content = html_path.read_text(encoding="utf-8")
            consulta = parse_document(html_content)
            if not consulta.numero:
                consulta.numero = html_path.stem
            save_markdown(consulta, data_path)
            reparsed += 1
        except Exception:
            logger.exception("Error re-parsing %s", html_path)
            errors += 1

    click.echo(f"Done. Re-parsed: {reparsed} | Errors: {errors}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
