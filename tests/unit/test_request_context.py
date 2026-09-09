"""Unit tests for the request-id context plumbing (Task 19)."""

from __future__ import annotations

import logging

from diary.core.request_context import (
    NO_REQUEST_ID,
    RequestIdFilter,
    attach_request_id_filter,
    current_request_id,
    new_request_id,
    reset_request_id,
    set_request_id,
)


def test_new_request_id_is_a_32_char_hex_string() -> None:
    first = new_request_id()
    second = new_request_id()
    assert len(first) == 32
    assert int(first, 16) >= 0  # valid hex
    assert first != second


def test_current_request_id_is_empty_outside_a_request() -> None:
    assert current_request_id() == ""


def test_set_and_reset_roundtrip() -> None:
    assert current_request_id() == ""
    token = set_request_id("req-1")
    assert current_request_id() == "req-1"
    reset_request_id(token)
    assert current_request_id() == ""


def test_request_id_filter_attaches_current_value_to_records() -> None:
    record = logging.LogRecord("diary.test", logging.INFO, "test.py", 1, "message", (), None)
    log_filter = RequestIdFilter()
    assert log_filter.filter(record) is True
    assert getattr(record, "request_id") == NO_REQUEST_ID  # outside a request

    token = set_request_id("req-42")
    try:
        assert log_filter.filter(record) is True
        assert getattr(record, "request_id") == "req-42"
    finally:
        reset_request_id(token)


def test_attach_request_id_filter_is_idempotent() -> None:
    logger = logging.getLogger("diary.request-context-test")
    original_count = len(logger.filters)
    attach_request_id_filter(logger)
    attach_request_id_filter(logger)
    attach_request_id_filter(logger)
    assert len(logger.filters) == original_count + 1
    logger.filters = logger.filters[:-1]  # restore for other tests
