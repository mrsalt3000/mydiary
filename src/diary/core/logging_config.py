"""Shared logging configuration (Task 01 baseline, request-id from Task 19).

Both the web process (``diary.main``) and the worker CLI
(``diary.workers.runner.main``) call :func:`configure_logging` instead of
``logging.basicConfig`` so the log line format and the request-id handling are
identical everywhere.

Format contract
===============

Every line rendered by the root handler has the shape::

    %(asctime)s | %(levelname)s | %(name)s | request_id=%(request_id)s | %(message)s

``request_id`` is filled from the current request context by
:class:`~diary.core.request_context.RequestIdFilter`; records emitted outside
an HTTP request render ``request_id=-``. The ``defaults`` of the formatter
keep non-diary loggers (uvicorn, sqlalchemy, ...) from crashing on the missing
attribute.

Privacy (AGENTS.md §8): the format itself never carries diary content; the
``message`` side is produced by application loggers that follow the same
content-free contract. This module attaches the request-id filter to the root
logger and to every ``diary.*`` logger registered so far, so caplog records
and handler output alike expose the id.
"""

from __future__ import annotations

import logging

from diary.core.request_context import NO_REQUEST_ID, attach_request_id_filter

#: Log format with an always-present ``request_id`` field. ``%(request_id)s``
#: is provided by :class:`RequestIdFilter`; the formatter ``defaults`` cover
#: loggers that never went through the filter.
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | request_id=%(request_id)s | %(message)s"

#: Level of the root logger configured by :func:`configure_logging`.
DEFAULT_LEVEL = logging.INFO


def configure_logging(*, level: int = DEFAULT_LEVEL) -> None:
    """Configure the root logger for a diary process (idempotent).

    Installs one stream handler with the shared request-id format on the root
    logger when no handler is present yet (``logging.basicConfig`` semantics:
    a process keeps whatever it configured first), then makes sure the
    request-id filter is attached to the root logger and to every existing
    ``diary`` logger. Loggers created later (``diary.workers.runner``,
    ``diary.api``) attach the filter themselves at import time; the worker and
    the web process import their modules before processing anything.
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, defaults={"request_id": NO_REQUEST_ID}))
        root.addHandler(handler)
    root.setLevel(level)
    attach_request_id_filter(root)
    for name, maybe_logger in list(logging.Logger.manager.loggerDict.items()):
        if not isinstance(maybe_logger, logging.Logger):
            continue
        if name == "diary" or name.startswith("diary."):
            attach_request_id_filter(maybe_logger)


__all__ = ["DEFAULT_LEVEL", "LOG_FORMAT", "configure_logging"]
