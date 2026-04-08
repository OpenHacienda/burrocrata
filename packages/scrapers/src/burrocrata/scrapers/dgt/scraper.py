"""DGT PETETE scraper – session management, search pagination, and document fetching."""

import logging
import random
import threading
import time

import requests
import urllib3

from burrocrata.scrapers.core.ratelimit import TokenBucket

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

BASE_URL = "https://petete.tributos.hacienda.gob.es"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"{BASE_URL}/consultas/",
    "Content-Type": "application/x-www-form-urlencoded",
}

MAX_RETRIES = 3
RETRY_BACKOFFS = [5, 15, 45]


class DGTSession:
    """Manages an HTTP session against the PETETE server."""

    def __init__(
        self,
        rate_limit: float = 1.0,
        min_rate: float = 0.2,
        max_rate: float = 4.0,
    ) -> None:
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.session.verify = False
        self.bucket = TokenBucket(rate=rate_limit, min_rate=min_rate, max_rate=max_rate)
        self._initialized = False
        self._init_lock = threading.Lock()

    def init(self) -> None:
        """Visit the landing page to obtain session cookies."""
        with self._init_lock:
            if self._initialized:
                return
            for attempt in range(MAX_RETRIES + 1):
                self.bucket.wait()
                try:
                    resp = self.session.get(f"{BASE_URL}/consultas/")
                except requests.RequestException as exc:
                    if attempt < MAX_RETRIES:
                        wait = RETRY_BACKOFFS[attempt]
                        logger.warning(
                            "Init request error (%s), retrying in %ds…", exc, wait
                        )
                        time.sleep(wait)
                        continue
                    raise
                if resp.status_code == 503:
                    if attempt < MAX_RETRIES:
                        wait = RETRY_BACKOFFS[attempt]
                        logger.warning("Init got 503, retrying in %ds…", wait)
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                resp.raise_for_status()
                self._initialized = True
                logger.info("Session initialized (cookies obtained)")
                return
            raise RuntimeError("Failed to initialize session after retries")

    def _ensure_init(self) -> None:
        if not self._initialized:
            self.init()

    def _backoff(self, attempt: int, resp: requests.Response | None = None) -> float:
        """Compute backoff with jitter. Honors Retry-After when present."""
        if resp is not None:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass  # HTTP-date form — ignore, fall through
        base = RETRY_BACKOFFS[min(attempt, len(RETRY_BACKOFFS) - 1)]
        return base * (1 + random.random() * 0.3)

    def _post_with_retry(self, url: str, data: dict) -> requests.Response:
        self._ensure_init()
        for attempt in range(MAX_RETRIES + 1):
            self.bucket.wait()
            try:
                resp = self.session.post(url, data=data)
            except requests.RequestException as exc:
                self.bucket.record_failure()
                if attempt < MAX_RETRIES:
                    wait = self._backoff(attempt)
                    logger.warning("Request error (%s), retrying in %.1fs…", exc, wait)
                    time.sleep(wait)
                    continue
                raise
            if resp.status_code == 401:
                if attempt < MAX_RETRIES:
                    logger.warning("Got 401, reinitializing session…")
                    with self._init_lock:
                        self._initialized = False
                    self.init()
                    continue
                resp.raise_for_status()
            if resp.status_code == 429 or resp.status_code >= 500:
                self.bucket.record_failure()
                if attempt < MAX_RETRIES:
                    wait = self._backoff(attempt, resp)
                    logger.warning("Got %d, retrying in %.1fs…", resp.status_code, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
            self.bucket.record_success()
            return resp
        raise RuntimeError("Exceeded max retries")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        page: int = 1,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> str:
        """Run a paginated search and return the raw HTML."""
        data: dict[str, str] = {
            "type2": "on",
            "NMCMP_1": "NUM-CONSULTA",
            "VLCMP_1": "",
            "OPCMP_1": ".Y",
            "NMCMP_2": "FECHA-SALIDA",
            "OPCMP_2": ".Y",
            "cmpOrder": "FECHA-SALIDA",
            "dirOrder": "1",
            "tab": "2",
            "page": str(page),
        }
        if date_start and date_end:
            data["dateIni_2"] = date_start
            data["dateEnd_2"] = date_end
            data["VLCMP_2"] = f"{date_start}..{date_end}"
        resp = self._post_with_retry(f"{BASE_URL}/consultas/do/search", data)
        return resp.text

    def fetch_document(self, doc_id: str) -> str:
        """Fetch a single document by its internal numeric ID."""
        data = {
            "doc": doc_id,
            "tab": "2",
            "query": ".T",
        }
        resp = self._post_with_retry(f"{BASE_URL}/consultas/do/document", data)
        return resp.text
