"""Optional API-key authentication and per-client rate limiting.

Both are off by default so local development needs no setup:
* API_KEYS            comma-separated keys required (X-API-Key header) on compute endpoints
* ADMIN_API_KEYS      comma-separated keys for /api/v1/admin; admin API is disabled when empty
* RATE_LIMIT_PER_MINUTE  requests per client per rolling minute on compute endpoints (0 = off)

The rate limiter is in-process: with several workers or replicas each keeps its own window,
so put a shared limiter (e.g. at the proxy) in front for strict global limits.
"""

from __future__ import annotations

import math
import secrets
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request, status

from app.core.config import get_settings


def _keys(raw: str) -> list[str]:
    return [k.strip() for k in raw.split(",") if k.strip()]


def _matches(candidate: str | None, keys: list[str]) -> bool:
    # Compare against every key so timing does not reveal which (or whether a) prefix matched.
    ok = False
    for key in keys:
        ok |= secrets.compare_digest((candidate or "").encode(), key.encode())
    return ok


def require_api_key(x_api_key: str | None = Header(default=None, description="Required when API_KEYS is set")):
    keys = _keys(get_settings().api_keys)
    if keys and not _matches(x_api_key, keys):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"}
        )


def require_admin_key(x_api_key: str | None = Header(default=None, description="An ADMIN_API_KEYS value")):
    keys = _keys(get_settings().admin_api_keys)
    if not keys:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin API is disabled; set ADMIN_API_KEYS to enable it")
    if not _matches(x_api_key, keys):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Missing or invalid admin API key", headers={"WWW-Authenticate": "ApiKey"}
        )


class SlidingWindowLimiter:
    def __init__(self, window_s: float = 60.0):
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, client: str, limit: int, now: float | None = None) -> float | None:
        """Record a request; return seconds to wait if the client is over `limit`, else None."""
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits[client]
            while q and now - q[0] >= self.window_s:
                q.popleft()
            if len(q) >= limit:
                return max(0.0, self.window_s - (now - q[0]))
            q.append(now)
            if len(self._hits) > 50_000:  # bound memory under many distinct clients
                for key in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window_s]:
                    del self._hits[key]
            return None

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = SlidingWindowLimiter()


def rate_limit(request: Request, x_api_key: str | None = Header(default=None, include_in_schema=False)):
    limit = get_settings().rate_limit_per_minute
    if limit <= 0:
        return
    client = f"key:{x_api_key}" if x_api_key else f"ip:{request.client.host if request.client else 'unknown'}"
    retry_after = limiter.hit(client, limit)
    if retry_after is not None:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Rate limit of {limit} requests per minute exceeded",
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )
