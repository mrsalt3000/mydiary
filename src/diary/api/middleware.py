"""HTTP middleware that tags every request with a request id (Task 19).

Each HTTP request receives a fresh id (``uuid4().hex``) bound to the current
asyncio context for the whole request lifecycle. Application log records
emitted while the request is handled carry it via the log format's
``request_id=...`` field (see ``diary.core.logging_config``).

Privacy contract (AGENTS.md §8, ``docs/privacy-and-safety.md`` §4): the
middleware logs only safe operational metadata — method, URL path, HTTP status
and duration. It never logs request bodies, headers, query strings or diary
content. URL paths carry identifiers (message/export/job ids), never diary
text.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from diary.core.request_context import (
    attach_request_id_filter,
    new_request_id,
    reset_request_id,
    set_request_id,
)

logger = logging.getLogger("diary.api")
attach_request_id_filter(logger)


class RequestContextMiddleware:
    """Pure-ASGI middleware assigning and logging one request id per request.

    A pure ASGI middleware (rather than ``BaseHTTPMiddleware``) keeps the whole
    downstream app inside the same asyncio task, so the request id bound via
    the ``ContextVar`` is visible to synchronous endpoints dispatched through
    ``anyio``'s context-copying thread pool and to any nested task.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        token = set_request_id(new_request_id())
        method = scope.get("method", "")
        path = scope.get("path", "")
        started = time.monotonic()
        status_code = 500
        logger.info("http_request method=%s path=%s", method, path)

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", status_code))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.monotonic() - started) * 1000.0
            logger.info(
                "http_request method=%s path=%s status=%s duration_ms=%.1f",
                method,
                path,
                status_code,
                duration_ms,
            )
            reset_request_id(token)


__all__ = ["RequestContextMiddleware"]
