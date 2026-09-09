"""Unit tests for the job-queue retry policy (Task 08, ADR-003).

Covers the pure exponential-backoff helper ``retry_delay_seconds`` and the
queue-policy settings defaults/validation. Deterministic: the MVP backoff uses
no jitter, so exact delays are asserted.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from diary.core.settings import AppSettings
from diary.db.repositories import retry_delay_seconds


def _delay(
    attempts: int,
    *,
    base_s: float = 2.0,
    multiplier: float = 2.0,
    max_backoff_s: float = 300.0,
) -> float:
    return retry_delay_seconds(
        attempts=attempts,
        base_s=base_s,
        multiplier=multiplier,
        max_backoff_s=max_backoff_s,
    )


def test_backoff_grows_exponentially_from_the_base() -> None:
    # First failed attempt waits base; each next retry multiplies by 2.
    assert _delay(1) == 2.0
    assert _delay(2) == 4.0
    assert _delay(3) == 8.0
    assert _delay(4) == 16.0


def test_backoff_is_deterministic_without_jitter() -> None:
    kwargs = dict(attempts=3, base_s=1.5, multiplier=3.0, max_backoff_s=60.0)
    assert retry_delay_seconds(**kwargs) == retry_delay_seconds(**kwargs)


def test_backoff_respects_the_maximum_cap() -> None:
    assert _delay(9, base_s=2.0, multiplier=2.0, max_backoff_s=300.0) == 300.0
    assert _delay(20) == 300.0
    capped = retry_delay_seconds(attempts=1, base_s=50.0, multiplier=2.0, max_backoff_s=30.0)
    assert capped == 30.0


def test_backoff_honours_custom_base_and_multiplier() -> None:
    assert _delay(1, base_s=5.0, multiplier=3.0) == 5.0
    assert _delay(2, base_s=5.0, multiplier=3.0) == 15.0
    assert _delay(3, base_s=5.0, multiplier=3.0) == 45.0


def test_backoff_rejects_zero_or_negative_attempts() -> None:
    for attempts in (0, -1, -5):
        with pytest.raises(ValueError, match="attempts must be >= 1"):
            _delay(attempts)


def test_settings_expose_default_queue_policy() -> None:
    settings = AppSettings(_env_file=None)
    assert settings.job_max_attempts == 3
    assert settings.job_backoff_base_s == 2.0
    assert settings.job_backoff_multiplier == 2.0
    assert settings.job_max_backoff_s == 300.0
    assert settings.job_stale_after_s == 300.0


def test_settings_reject_invalid_queue_policy() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, job_max_attempts=0)
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, job_backoff_base_s=0)
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, job_max_backoff_s=-1)
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, job_stale_after_s=0)


def test_settings_queue_policy_reads_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_JOB_MAX_ATTEMPTS", "7")
    monkeypatch.setenv("APP_JOB_BACKOFF_BASE_S", "10")
    settings = AppSettings(_env_file=None)
    assert settings.job_max_attempts == 7
    assert settings.job_backoff_base_s == 10.0
