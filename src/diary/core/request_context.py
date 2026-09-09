"""Request-id plumbing for structured logs (Task 19).

Every HTTP request gets a fresh request id in a ``contextvars.ContextVar`` so
all log records emitted while the request is being handled can carry it. The
worker process has no request context and logs ``request_id=-`` (the default).

Privacy (AGENTS.md §7–8, ``docs/privacy-and-safety.md`` §4): only the request
id travels with a log record — never the request body, diary content, prompts,
secrets or provider payloads. The middleware that assigns the id
(``diary.api.middleware``) logs only method/path/status/duration; this module
knows nothing about request content.

Why a ``ContextVar`` and not a thread-local: FastAPI serves concurrent
requests on the same event-loop thread and dispatches synchronous handlers to
worker threads. ``anyio``'s thread pool copies the current context, so a
``ContextVar`` set by the middleware is visible to the endpoint, its logs and
any nested task — while two interleaved requests can never see each other's id.
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar, Token

#: Per-request request id; empty string when no HTTP request is being handled
#: (worker process, background code, tests).
_request_id_var: ContextVar[str] = ContextVar("diary_request_id", default="")

#: Value rendered by the log formatter when no request is in scope.
NO_REQUEST_ID = "-"


def new_request_id() -> str:
    """A fresh, unpredictable request id (32 lowercase hex chars)."""
    return uuid.uuid4().hex


def current_request_id() -> str:
    """The request id of the current context, or ``""`` outside a request."""
    return _request_id_var.get()


def set_request_id(request_id: str) -> Token[str]:
    """Bind ``request_id`` to the current context; return the reset token."""
    return _request_id_var.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    """Restore the request-id value that was active before ``token``."""
    _request_id_var.reset(token)


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every log record of a logger.

    The attribute is always present (``-`` outside an HTTP request), so log
    formatters may reference ``%(request_id)s`` unconditionally and caplog
    tests can read ``record.request_id`` on every captured record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id() or NO_REQUEST_ID
        return True


def attach_request_id_filter(logger: logging.Logger) -> None:
    """Add a :class:`RequestIdFilter` to ``logger`` (idempotent)."""
    if not any(isinstance(item, RequestIdFilter) for item in logger.filters):
        logger.addFilter(RequestIdFilter())


__all__ = [
    "NO_REQUEST_ID",
    "RequestIdFilter",
    "attach_request_id_filter",
    "current_request_id",
    "new_request_id",
    "reset_request_id",
    "set_request_id",
]
