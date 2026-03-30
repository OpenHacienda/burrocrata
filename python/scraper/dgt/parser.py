"""HTML parsing for DGT PETETE search results and document pages."""

import html
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag


@dataclass
class SearchResult:
    doc_id: str
    numero: str
    preview_hechos: str
    preview_cuestion: str


@dataclass
class SearchPage:
    total_results: int
    total_pages: int
    results: list[SearchResult]


@dataclass
class Consulta:
    numero: str
    organo: str
    fecha: str  # DD/MM/YYYY as returned by PETETE
    normativa: str
    hechos: str
    cuestion: str
    contestacion: str

    @property
    def fecha_iso(self) -> str:
        """Convert DD/MM/YYYY to YYYY-MM-DD."""
        parts = self.fecha.split("/")
        if len(parts) == 3:
            return f"{parts[2]}-{parts[1]}-{parts[0]}"
        return self.fecha

    @property
    def year(self) -> str:
        """Extract 4-digit year from the consulta numero (e.g. V2653-24 -> 2024)."""
        m = re.search(r"-(\d{2})$", self.numero)
        if m:
            yy = int(m.group(1))
            return str(2000 + yy) if yy < 80 else str(1900 + yy)
        # Fallback: use fecha
        return self.fecha_iso[:4]


def _clean_html(text: str) -> str:
    """Strip HTML tags and unescape entities."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_search_results(html_content: str) -> SearchPage:
    """Parse a search results page and return structured data."""
    # Total results: updateNumResults("2", "NNNNN")
    total_match = re.search(r'updateNumResults\("2",\s*"(\d+)"\)', html_content)
    total_results = int(total_match.group(1)) if total_match else 0

    # Total pages
    pages_match = re.search(r'<span id="total_pages">(\d+)</span>', html_content)
    total_pages = int(pages_match.group(1)) if pages_match else 1

    # Document entries
    results: list[SearchResult] = []
    doc_pattern = re.compile(
        r'<td id="doc_(\d+)"[^>]*onClick="return viewDocument\(\d+,\s*2\);">'
    )
    soup = BeautifulSoup(html_content, "html.parser")

    for td in soup.find_all("td", id=re.compile(r"^doc_\d+$")):
        td_id = td.get("id", "")
        doc_id = td_id.replace("doc_", "")

        # Walk up to the containing row / sibling area to find metadata
        container = td.parent or td
        num_span = container.find("span", class_="NUM-CONSULTA")
        numero = ""
        if num_span:
            strong = num_span.find("strong")
            numero = (strong.get_text(strip=True) if strong else num_span.get_text(strip=True))

        hechos_span = container.find("span", class_="DESCRIPCION-HECHOS")
        preview_hechos = hechos_span.get_text(strip=True) if hechos_span else ""

        cuestion_span = container.find("span", class_="CUESTION-PLANTEADA")
        preview_cuestion = cuestion_span.get_text(strip=True) if cuestion_span else ""

        results.append(
            SearchResult(
                doc_id=doc_id,
                numero=numero,
                preview_hechos=preview_hechos,
                preview_cuestion=preview_cuestion,
            )
        )

    # Fallback: use regex if BS4 found nothing (some pages have odd HTML)
    if not results:
        for m in doc_pattern.finditer(html_content):
            doc_id = m.group(1)
            results.append(
                SearchResult(doc_id=doc_id, numero="", preview_hechos="", preview_cuestion="")
            )

    return SearchPage(total_results=total_results, total_pages=total_pages, results=results)


def _extract_multi_paragraph(soup: BeautifulSoup, class_name: str) -> str:
    """Extract and join all <p> elements with the given class."""
    paragraphs = soup.find_all("p", class_=class_name)
    parts: list[str] = []
    for p in paragraphs:
        text = _clean_html(str(p))
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _extract_single_field(soup: BeautifulSoup, class_name: str) -> str:
    """Extract a single-value field."""
    tag = soup.find("p", class_=class_name)
    if tag:
        return _clean_html(str(tag))
    # Fallback: look in <tr> with the class
    tr = soup.find("tr", class_=class_name)
    if tr:
        return _clean_html(tr.get_text())
    return ""


def parse_document(html_content: str) -> Consulta:
    """Parse a full document page into a Consulta."""
    soup = BeautifulSoup(html_content, "html.parser")
    return Consulta(
        numero=_extract_single_field(soup, "NUM-CONSULTA"),
        organo=_extract_single_field(soup, "ORGANO"),
        fecha=_extract_single_field(soup, "FECHA-SALIDA"),
        normativa=_extract_single_field(soup, "NORMATIVA"),
        hechos=_extract_multi_paragraph(soup, "DESCRIPCION-HECHOS"),
        cuestion=_extract_multi_paragraph(soup, "CUESTION-PLANTEADA"),
        contestacion=_extract_multi_paragraph(soup, "CONTESTACION-COMPL"),
    )
