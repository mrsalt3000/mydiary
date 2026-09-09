"""Unit tests for the worker runner CLI and helpers (Task 08).

Covers the argparse surface (``python -m diary.workers.runner``), the safe
error-summary builder and the module's importability/``--help``. No database is
touched here — helpers are tested in isolation; queue behaviour lives in the
integration tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from diary.workers.runner import (
    _default_worker_id,
    _safe_error_summary,
    build_argument_parser,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_safe_error_summary_is_class_and_short_message() -> None:
    exc = ValueError("очень важная деталь " * 60)  # longer than the cap
    summary = _safe_error_summary(exc)
    assert summary.startswith("ValueError: ")
    assert "Traceback" not in summary
    assert len(summary) == len("ValueError: ") + 300  # class name + capped message


def test_safe_error_summary_without_message_keeps_class_name() -> None:
    assert _safe_error_summary(RuntimeError("   \n  ")) == "RuntimeError"


def test_safe_error_summary_collapses_whitespace() -> None:
    exc = RuntimeError("multiple   spaces\nand a newline")
    assert _safe_error_summary(exc) == "RuntimeError: multiple spaces and a newline"


def test_default_worker_id_contains_hostname_and_pid() -> None:
    import os
    import socket

    worker_id = _default_worker_id()
    assert worker_id.startswith(f"{socket.gethostname()}-")
    assert worker_id.endswith(f"-{os.getpid()}")


def test_parser_exposes_all_runner_flags() -> None:
    parser = build_argument_parser()
    args = parser.parse_args(
        ["--once", "--poll-interval", "2.5", "--worker-id", "w1", "--reset-stale"]
    )
    assert args.once is True
    assert args.poll_interval == 2.5
    assert args.worker_id == "w1"
    assert args.reset_stale is True


def test_parser_defaults() -> None:
    args = build_argument_parser().parse_args([])
    assert args.once is False
    assert args.poll_interval == 5.0
    assert args.worker_id is None
    assert args.reset_stale is False


def test_parser_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_argument_parser().parse_args(["--help"])
    assert excinfo.value.code == 0


def test_module_help_via_python_m_is_runnable() -> None:
    """The required acceptance command must run from the repository root."""
    result = subprocess.run(
        [sys.executable, "-m", "diary.workers.runner", "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--once" in result.stdout
    assert "--poll-interval" in result.stdout
    assert "--reset-stale" in result.stdout
    assert "--worker-id" in result.stdout
