"""DGT PETETE scraper – session management, search pagination, and document fetching."""

import logging
import random
import threading
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
    """Thread-safe AIMD token-bucket rate limiter.

    Additive-increase, multiplicative-decrease: each success nudges the rate
    up by ``aimd_step`` (capped at ``max_rate``); each failure halves it
    (floored at ``min_rate``). Same shape as TCP congestion control — the
    bucket converges on whatever sustained rate the server tolerates.
    """

    rate: float = 1.0  # tokens per second (current target)
    burst: int = 5
    min_rate: float = 0.2
    max_rate: float = 4.0
    aimd_step: float = 0.1
    success_threshold: int = 10  # successes between additive bumps
    _tokens: float = field(init=False, default=0.0)
    _last: float = field(init=False, default=0.0)
    _success_streak: int = field(init=False, default=0)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.rate = max(self.min_rate, min(self.max_rate, self.rate))
        self._tokens = float(self.burst)
        self._last = time.monotonic()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last = now
            if self._tokens < 1:
                sleep_time = (1 - self._tokens) / self.rate
                self._tokens = 0
            else:
                self._tokens -= 1
                sleep_time = 0.0
        if sleep_time > 0:
            logger.debug("Rate limit: sleeping %.2fs (rate=%.2f/s)", sleep_time, self.rate)
            time.sleep(sleep_time)
            with self._lock:
                self._last = time.monotonic()

    def record_success(self) -> None:
        """Additive increase after a streak of successes."""
        with self._lock:
            self._success_streak += 1
            if self._success_streak >= self.success_threshold and self.rate < self.max_rate:
                old = self.rate
                self.rate = min(self.max_rate, self.rate + self.aimd_step)
                self._success_streak = 0
                logger.info("AIMD: rate %.2f -> %.2f/s", old, self.rate)

    def record_failure(self) -> None:
        """Multiplicative decrease on server errors."""
        with self._lock:
            self._success_streak = 0
            if self.rate > self.min_rate:
                old = self.rate
                self.rate = max(self.min_rate, self.rate * 0.5)
                logger.warning("AIMD: rate %.2f -> %.2f/s (server pressure)", old, self.rate)


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
        self.bucket = TokenBucket(
            rate=rate_limit, min_rate=min_rate, max_rate=max_rate
        )
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
                        logger.warning("Init request error (%s), retrying in %ds…", exc, wait)
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
                    logger.warning(
                        "Got %d, retrying in %.1fs…", resp.status_code, wait
                    )
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
