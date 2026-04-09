# Step 2: extract `burrocrata.scrapers.core`

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move source-agnostic scraping primitives out of `dgt/` into a new `burrocrata.scrapers.core` subpackage so future scrapers (BOE, TEAC, …) don't copy-paste. **No behavior change.**

**Architecture:** `core/` is a leaf package with no dependencies on any specific source. `dgt/scraper.py` and `dgt/cli.py` import from it. Every extraction is a pure move: no refactor, no generalization, no feature additions.

**Tech Stack:** Same as Step 1 — Python 3.11+, stdlib + requests.

---

## What moves to `core/`

| From | To | Symbol |
|---|---|---|
| `dgt/scraper.py` | `core/ratelimit.py` | `TokenBucket` (AIMD) |
| `dgt/scraper.py` | `core/retry.py` | `MAX_RETRIES`, `RETRY_BACKOFFS`, `compute_backoff` (extracted from `DGTSession._backoff`) |
| `dgt/cli.py` | `core/notify.py` | `_notify_ntfy` → public `notify_ntfy` |
| `dgt/cli.py` | `core/checkpoint.py` | `_load_checkpoint`, `_save_checkpoint` → `load_checkpoint`, `save_checkpoint` |

**Deliberately NOT moved:**
- `DGTSession`, `HEADERS`, `BASE_URL` — DGT-specific.
- `_post_with_retry`, `init`, `search`, `fetch_document` — DGT-specific session logic that *uses* the core primitives but isn't itself generic. Generalizing a base `Session` class is speculative until BOE exists (YAGNI).
- `_load_search_index` / `_save_search_index` — could be generic, but currently hardcodes doc_id/numero shape. Leave in `cli.py` until a second source proves the shape.
- `_existing_numeros` — tied to DGT's "numero" identifier.

---

## Task 1: create `core/` subpackage

**Files:**
- Create: `packages/scrapers/src/burrocrata/scrapers/core/__init__.py`

**Step 1: Create empty package**

```python
"""Source-agnostic scraping primitives shared across all sources."""
```

That's the whole file. No re-exports yet — force call sites to use the submodule path (`from burrocrata.scrapers.core.ratelimit import TokenBucket`) so the dependency is explicit.

**Step 2: Verify import**

```bash
uv run python -c "import burrocrata.scrapers.core; print('OK')"
```
Expected: `OK`.

**Step 3: Commit**

```bash
git add packages/scrapers/src/burrocrata/scrapers/core/__init__.py
git commit -m "refactor: add empty burrocrata.scrapers.core subpackage"
```

---

## Task 2: extract `TokenBucket` → `core/ratelimit.py`

**Files:**
- Create: `packages/scrapers/src/burrocrata/scrapers/core/ratelimit.py`
- Modify: `packages/scrapers/src/burrocrata/scrapers/dgt/scraper.py`

**Step 1: Create `core/ratelimit.py`**

Copy the `TokenBucket` dataclass **verbatim** from `dgt/scraper.py` (currently lines ~34–104), plus the module docstring, logger, `threading`, `time`, and `dataclass`/`field` imports it needs. Do NOT change any logic, constants, log message, or defaults.

File contents:

```python
"""Thread-safe AIMD token-bucket rate limiter.

Source-agnostic: used by all scrapers in this package.
"""

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
            logger.debug(
                "Rate limit: sleeping %.2fs (rate=%.2f/s)", sleep_time, self.rate
            )
            time.sleep(sleep_time)
            with self._lock:
                self._last = time.monotonic()

    def record_success(self) -> None:
        """Additive increase after a streak of successes."""
        with self._lock:
            self._success_streak += 1
            if (
                self._success_streak >= self.success_threshold
                and self.rate < self.max_rate
            ):
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
                logger.warning(
                    "AIMD: rate %.2f -> %.2f/s (server pressure)", old, self.rate
                )
```

**Step 2: Remove `TokenBucket` from `dgt/scraper.py`**

Delete the `@dataclass class TokenBucket:` block and its body. Keep `MAX_RETRIES` and `RETRY_BACKOFFS` for now (they move in Task 3). Also remove the `dataclass, field` import from `dgt/scraper.py` — no longer needed there.

**Step 3: Add import to `dgt/scraper.py`**

At the top, add:

```python
from burrocrata.scrapers.core.ratelimit import TokenBucket
```

No other changes; `DGTSession.__init__` already uses `TokenBucket` by name.

**Step 4: Verify**

```bash
uv run python -c "from burrocrata.scrapers.core.ratelimit import TokenBucket; from burrocrata.scrapers.dgt.scraper import DGTSession; s = DGTSession(); print(type(s.bucket).__module__)"
```
Expected: `burrocrata.scrapers.core.ratelimit`

```bash
uv run burrocrata-dgt --help
uv run burrocrata-dgt stats --data-dir /tmp/burrocrata-nonexistent
uv run ruff check packages/scrapers
```

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move TokenBucket to burrocrata.scrapers.core.ratelimit

Pure move — identical code, identical defaults. Reusable across
future scraper sources."
```

---

## Task 3: extract retry constants + backoff → `core/retry.py`

**Files:**
- Create: `packages/scrapers/src/burrocrata/scrapers/core/retry.py`
- Modify: `packages/scrapers/src/burrocrata/scrapers/dgt/scraper.py`

**Step 1: Create `core/retry.py`**

This one requires **one tiny refactor**: `DGTSession._backoff` is a method that doesn't use `self`. Extract it to a module-level function `compute_backoff(attempt, resp=None)`.

```python
"""Retry constants and backoff computation for HTTP scrapers."""

from __future__ import annotations

import random

import requests

MAX_RETRIES = 3
RETRY_BACKOFFS = [5, 15, 45]


def compute_backoff(
    attempt: int, resp: requests.Response | None = None
) -> float:
    """Compute backoff seconds with jitter. Honors Retry-After when present."""
    if resp is not None:
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass  # HTTP-date form — ignore, fall through
    base = RETRY_BACKOFFS[min(attempt, len(RETRY_BACKOFFS) - 1)]
    return base * (1 + random.random() * 0.3)
```

**Step 2: Update `dgt/scraper.py`**

- Remove the `MAX_RETRIES` and `RETRY_BACKOFFS` module constants.
- Remove the `_backoff` method from `DGTSession`.
- Remove `import random`.
- Add: `from burrocrata.scrapers.core.retry import MAX_RETRIES, RETRY_BACKOFFS, compute_backoff`
- Replace the two call sites `self._backoff(attempt)` and `self._backoff(attempt, resp)` with `compute_backoff(attempt)` and `compute_backoff(attempt, resp)` respectively.

Double-check: `RETRY_BACKOFFS` is still referenced in the `init` method for the 503/RequestException retry loops — those references continue to work via the import.

**Step 3: Verify**

```bash
uv run python -c "
from burrocrata.scrapers.core.retry import compute_backoff, MAX_RETRIES, RETRY_BACKOFFS
from burrocrata.scrapers.dgt.scraper import DGTSession
assert MAX_RETRIES == 3
assert RETRY_BACKOFFS == [5, 15, 45]
print('OK')
"
uv run burrocrata-dgt --help
uv run burrocrata-dgt stats --data-dir /tmp/burrocrata-nonexistent
uv run ruff check packages/scrapers
```

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: move retry constants and compute_backoff to core.retry

Extracts DGTSession._backoff to a module-level function since it
never used self. No behavior change."
```

---

## Task 4: extract ntfy notifier → `core/notify.py`

**Files:**
- Create: `packages/scrapers/src/burrocrata/scrapers/core/notify.py`
- Modify: `packages/scrapers/src/burrocrata/scrapers/dgt/cli.py`

**Step 1: Create `core/notify.py`**

```python
"""Notification helpers (currently: ntfy)."""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)


def notify_ntfy(
    topic: str | None,
    title: str,
    message: str,
    priority: str = "default",
    server: str | None = None,
) -> None:
    """Send a notification to an ntfy topic. Silently no-ops if topic is falsy."""
    if not topic:
        return
    server = (server or os.environ.get("NTFY_SERVER") or "https://ntfy.sh").rstrip("/")
    url = f"{server}/{topic}"
    try:
        requests.post(
            url,
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": priority, "Tags": "scroll"},
            timeout=10,
        )
    except requests.RequestException as exc:
        logger.warning("Failed to send ntfy notification: %s", exc)
```

Note the rename: `_notify_ntfy` → `notify_ntfy` (no leading underscore; it's public now). Behavior identical.

**Step 2: Update `dgt/cli.py`**

- Delete the `_notify_ntfy` function (currently ~lines 469–489).
- Add: `from burrocrata.scrapers.core.notify import notify_ntfy`
- Replace both call sites `_notify_ntfy(ntfy_topic, ...)` with `notify_ntfy(ntfy_topic, ...)`.
- If nothing else in `cli.py` uses `requests` or `os` after this, remove those imports. (Check: `os.environ.get` is still used for the click option defaults; keep `os`. `requests` — check with `rg -n 'requests' packages/scrapers/src/burrocrata/scrapers/dgt/cli.py` and drop the import if unused.)

**Step 3: Verify**

```bash
uv run python -c "from burrocrata.scrapers.core.notify import notify_ntfy; notify_ntfy(None, 't', 'm'); print('OK')"
uv run burrocrata-dgt --help
uv run burrocrata-dgt fetch --help
uv run burrocrata-dgt stats --data-dir /tmp/burrocrata-nonexistent
uv run ruff check packages/scrapers
```

The `fetch --help` call must still show `--ntfy-topic` and `--ntfy-server` options.

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: move ntfy notifier to core.notify (and make public)

Renames _notify_ntfy → notify_ntfy. No behavior change."
```

---

## Task 5: extract checkpoint helpers → `core/checkpoint.py`

**Files:**
- Create: `packages/scrapers/src/burrocrata/scrapers/core/checkpoint.py`
- Modify: `packages/scrapers/src/burrocrata/scrapers/dgt/cli.py`

**Step 1: Create `core/checkpoint.py`**

```python
"""Generic JSON checkpoint helpers for resumable scrapes."""

from __future__ import annotations

import json
from pathlib import Path


def load_checkpoint(data_dir: Path) -> dict:
    """Load a checkpoint dict from ``<data_dir>/checkpoint.json``, or {}."""
    cp_path = data_dir / "checkpoint.json"
    if cp_path.exists():
        return json.loads(cp_path.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data_dir: Path, checkpoint: dict) -> None:
    """Persist a checkpoint dict to ``<data_dir>/checkpoint.json``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    cp_path = data_dir / "checkpoint.json"
    cp_path.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
    )
```

Rename: `_load_checkpoint` → `load_checkpoint`, `_save_checkpoint` → `save_checkpoint`. Identical bodies.

**Step 2: Update `dgt/cli.py`**

- Delete the `_load_checkpoint` and `_save_checkpoint` functions.
- Add: `from burrocrata.scrapers.core.checkpoint import load_checkpoint, save_checkpoint`
- Replace all call sites: `_load_checkpoint(` → `load_checkpoint(`, `_save_checkpoint(` → `save_checkpoint(`.

Verify with `rg` that no private-named references remain:
```bash
rg -n '_load_checkpoint|_save_checkpoint' packages/scrapers
```
Expected: no matches.

**Step 3: Verify**

```bash
uv run python -c "from burrocrata.scrapers.core.checkpoint import load_checkpoint, save_checkpoint; print('OK')"
uv run burrocrata-dgt --help
uv run burrocrata-dgt stats --data-dir /tmp/burrocrata-nonexistent
uv run ruff check packages/scrapers
```

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: move checkpoint helpers to core.checkpoint

Renames _load_checkpoint/_save_checkpoint to public names. No
behavior change."
```

---

## Task 6: minimal unit test for `TokenBucket`

Rationale: Step 2 of the design explicitly calls for "add a unit test for the rate limiter". This is the only new test Step 2 introduces — enough to catch regressions if anyone refactors AIMD semantics.

**Files:**
- Create: `packages/scrapers/tests/__init__.py` (empty)
- Create: `packages/scrapers/tests/test_ratelimit.py`
- Modify: `packages/scrapers/pyproject.toml` (add `pytest` to an optional dev group)

**Step 1: Add pytest as a dev dependency**

Append to `packages/scrapers/pyproject.toml`:

```toml
[dependency-groups]
dev = ["pytest>=8.0"]
```

**Step 2: Create the test**

`packages/scrapers/tests/__init__.py` — empty file.

`packages/scrapers/tests/test_ratelimit.py`:

```python
"""Unit tests for the AIMD TokenBucket."""

from burrocrata.scrapers.core.ratelimit import TokenBucket


def test_initial_rate_clamped_to_bounds():
    b = TokenBucket(rate=100.0, min_rate=0.5, max_rate=2.0)
    assert b.rate == 2.0

    b2 = TokenBucket(rate=0.01, min_rate=0.5, max_rate=2.0)
    assert b2.rate == 0.5


def test_record_failure_halves_rate_floored_at_min():
    b = TokenBucket(rate=2.0, min_rate=0.5, max_rate=4.0)
    b.record_failure()
    assert b.rate == 1.0
    b.record_failure()
    assert b.rate == 0.5
    b.record_failure()  # already at floor
    assert b.rate == 0.5


def test_record_success_increases_after_threshold():
    b = TokenBucket(
        rate=1.0, min_rate=0.2, max_rate=4.0, aimd_step=0.1, success_threshold=3
    )
    for _ in range(2):
        b.record_success()
    assert b.rate == 1.0  # not yet
    b.record_success()
    assert b.rate == 1.1
    # streak resets; needs 3 more to bump again
    for _ in range(2):
        b.record_success()
    assert b.rate == 1.1
    b.record_success()
    assert b.rate == 1.2000000000000002 or abs(b.rate - 1.2) < 1e-9


def test_record_success_capped_at_max():
    b = TokenBucket(
        rate=3.95, min_rate=0.1, max_rate=4.0, aimd_step=0.1, success_threshold=1
    )
    b.record_success()
    assert b.rate == 4.0
    b.record_success()
    assert b.rate == 4.0


def test_failure_resets_success_streak():
    b = TokenBucket(
        rate=1.0, min_rate=0.2, max_rate=4.0, aimd_step=0.1, success_threshold=3
    )
    b.record_success()
    b.record_success()
    b.record_failure()
    assert b.rate == 0.5
    # streak was reset; need full threshold again to bump
    for _ in range(2):
        b.record_success()
    assert b.rate == 0.5
    b.record_success()
    assert b.rate == 0.6000000000000001 or abs(b.rate - 0.6) < 1e-9
```

**Step 3: Run tests**

```bash
uv sync
uv run --group dev pytest packages/scrapers/tests -v
```

Expected: 5 passed.

If `uv run --group dev pytest` fails because of the group flag syntax, fall back to `uv run --with pytest pytest packages/scrapers/tests -v`. Either works.

**Step 4: Commit**

```bash
git add -A
git commit -m "test: add unit tests for AIMD TokenBucket"
```

---

## Task 7: final verification sweep

**No files changed.** Pure verification.

**Step 1: Workspace smoke tests**

```bash
uv sync
uv run burrocrata-dgt --help
uv run burrocrata-dgt fetch --help   # must still show --ntfy-topic, --min-rate, --max-rate
uv run burrocrata-dgt stats --data-dir /tmp/burrocrata-nonexistent
uv run ruff check packages/scrapers
uv run --with pytest pytest packages/scrapers/tests -v
```

**Step 2: Nix build**

```bash
nix build .#scrapers -L
./result/bin/burrocrata-dgt --help
./result/bin/burrocrata-dgt fetch --help
rm result
```

**Step 3: treefmt**

```bash
nix develop -c treefmt --fail-on-change
```

If it reformats anything, commit as `chore: treefmt` and re-run.

**Step 4: Orphan sweep**

No references to the old private names or to in-`scraper.py` `TokenBucket`/`_backoff` should remain outside of `docs/`:

```bash
rg -n 'class TokenBucket|def _backoff|def _notify_ntfy|def _load_checkpoint|def _save_checkpoint' \
  --glob '!docs/**' --glob '!.worktrees/**' packages/
```
Expected: only matches inside `core/` (the definitions themselves) — no matches in `dgt/` and no private-named definitions anywhere.

**Step 5: Commit any cleanup (if needed)**

Otherwise nothing to commit — the branch state after Task 6 is the final state.

---

## Done criteria for Step 2

- [ ] `packages/scrapers/src/burrocrata/scrapers/core/` exists with `ratelimit.py`, `retry.py`, `notify.py`, `checkpoint.py`
- [ ] `dgt/scraper.py` imports `TokenBucket`, `MAX_RETRIES`, `RETRY_BACKOFFS`, `compute_backoff` from `core.*`
- [ ] `dgt/cli.py` imports `notify_ntfy`, `load_checkpoint`, `save_checkpoint` from `core.*`
- [ ] No private `_load_checkpoint` / `_save_checkpoint` / `_notify_ntfy` / `_backoff` definitions remain
- [ ] `pytest packages/scrapers/tests` passes (5 tests)
- [ ] `uv run burrocrata-dgt --help`, `fetch --help`, `stats` all work
- [ ] `nix build .#scrapers` succeeds
- [ ] `ruff check` and `treefmt --fail-on-change` clean
- [ ] One commit per task (7 total, + optional treefmt cleanup)

## Not in scope

- No base `Session` class / `HttpSession` abstraction — speculative until a 2nd source exists.
- No generic search-index helper — hardcoded doc_id/numero shape; revisit when BOE lands.
- No `_existing_numeros` extraction — tied to DGT identifier.
- No change to CLI flags, data layout, or scraping logic.
