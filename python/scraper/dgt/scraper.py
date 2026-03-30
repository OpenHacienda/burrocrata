"""DGT PETETE scraper – session management, search pagination, and document fetching."""

import logging
import time
from dataclasses import dataclass, field

import requests
import urllib3

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


@dataclass
class TokenBucket:
    """Simple token-bucket rate limiter."""

    rate: float = 0.5  # tokens per second
    burst: int = 5
    _tokens: float = field(init=False, default=0.0)
    _last: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self._tokens = float(self.burst)
        self._last = time.monotonic()

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last = now
        if self._tokens < 1:
            sleep_time = (1 - self._tokens) / self.rate
            logger.debug("Rate limit: sleeping %.2fs", sleep_time)
            time.sleep(sleep_time)
            self._tokens = 0
            self._last = time.monotonic()
        else:
            self._tokens -= 1


class DGTSession:
    """Manages an HTTP session against the PETETE server."""

    def __init__(self, rate_limit: float = 0.5) -> None:
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.session.verify = False
        self.bucket = TokenBucket(rate=rate_limit)
        self._initialized = False

    def init(self) -> None:
        """Visit the landing page to obtain session cookies."""
        self.bucket.wait()
        resp = self.session.get(f"{BASE_URL}/consultas/")
        resp.raise_for_status()
        self._initialized = True
        logger.info("Session initialized (cookies obtained)")

    def _ensure_init(self) -> None:
        if not self._initialized:
            self.init()

    def _post_with_retry(self, url: str, data: dict) -> requests.Response:
        self._ensure_init()
        for attempt in range(MAX_RETRIES + 1):
            self.bucket.wait()
            try:
                resp = self.session.post(url, data=data)
            except requests.RequestException as exc:
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFFS[attempt]
                    logger.warning("Request error (%s), retrying in %ds…", exc, wait)
                    time.sleep(wait)
                    continue
                raise
            if resp.status_code == 401:
                logger.warning("Got 401, reinitializing session…")
                self._initialized = False
                self.init()
                continue
            if resp.status_code == 503:
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFFS[attempt]
                    logger.warning("Got 503, retrying in %ds…", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
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
