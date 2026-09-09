"""Unit tests for the shared logging configuration (Task 01 baseline, Task 19).

The rendered log line must always carry a ``request_id=...`` field: the real
id while an HTTP request is in scope (bound by the request middleware) and the
``-`` placeholder otherwise. ``configure_logging`` must stay idempotent so the
web process and the worker can both call it safely.
"""

from __future__ import annotations

import io
import logging

from diary.core.logging_config import DEFAULT_LEVEL, LOG_FORMAT, configure_logging
from diary.core.request_context import (
    NO_REQUEST_ID,
    RequestIdFilter,
    attach_request_id_filter,
    reset_request_id,
    set_request_id,
)


def test_configure_logging_is_idempotent() -> None:
    root = logging.getLogger()
    configure_logging()
    after_first = len(root.handlers)
    configure_logging()
    configure_logging()
    assert len(root.handlers) == after_first  # never duplicates the handler
    assert root.level == DEFAULT_LEVEL
    # The request-id filter is installed on the root logger.
    assert any(isinstance(item, RequestIdFilter) for item in root.filters)


def test_log_format_contains_request_id_field() -> None:
    assert "%(request_id)s" in LOG_FORMAT


def test_rendered_line_carries_request_id_when_in_scope() -> None:
    logger = logging.getLogger("diary.logging-config-test")
    attach_request_id_filter(logger)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, defaults={"request_id": NO_REQUEST_ID}))
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    try:
        token = set_request_id("abc123")
        try:
            logger.info("inside request")
        finally:
            reset_request_id(token)
        logger.info("outside request")
    finally:
        logger.removeHandler(handler)
        logger.propagate = True
    output = stream.getvalue()
    assert "request_id=abc123 | inside request" in output
    assert f"request_id={NO_REQUEST_ID} | outside request" in output
