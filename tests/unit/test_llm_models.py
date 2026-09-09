"""Unit tests for the normalised LLM models and input digest (Task 11).

Synthetic only: no network calls and no real API keys. These tests pin the
shape of :class:`LLMRequest`/:class:`LLMResponse`/:class:`LLMCallProvenance`
and verify that provenance serialisation never carries request content.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from diary.domain.llm.models import (
    LLMCallOutcome,
    LLMCallProvenance,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    request_input_sha256,
    utc_now,
)


def _tz_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# LLMRequest / LLMMessage
# ---------------------------------------------------------------------------


def test_request_from_prompt_builds_single_user_message() -> None:
    request = LLMRequest.from_prompt("Текст для обработки", prompt_version="v1")
    assert len(request.messages) == 1
    assert request.messages[0].role == "user"
    assert request.messages[0].content == "Текст для обработки"
    assert request.prompt_version == "v1"
    assert request.model is None


def test_request_rejects_empty_messages() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(messages=[])


def test_message_rejects_blank_or_empty_content() -> None:
    with pytest.raises(ValidationError):
        LLMMessage(role="user", content="")
    with pytest.raises(ValidationError):
        LLMMessage(role="user", content="   ")


def test_message_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError):
        LLMMessage(role="admin", content="hello")  # type: ignore[arg-type]


def test_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(messages=[LLMMessage(role="user", content="x")], bogus=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# input digest
# ---------------------------------------------------------------------------


def test_input_sha256_is_deterministic_for_same_content() -> None:
    first = LLMRequest.from_prompt("Один и тот же текст")
    second = LLMRequest.from_prompt("Один и тот же текст")
    digest = request_input_sha256(first)
    assert digest == request_input_sha256(second)
    assert len(digest) == 64


def test_input_sha256_changes_with_content_and_never_contains_it() -> None:
    original = LLMRequest.from_prompt("Уникальный фрагмент дневника")
    digest = request_input_sha256(original)
    assert "Уникальный" not in digest
    changed = LLMRequest.from_prompt("Уникальный фрагмент дневника!")
    assert request_input_sha256(changed) != digest


def test_input_sha256_covers_role_and_content() -> None:
    system = LLMRequest(messages=[LLMMessage(role="system", content="Инструкция")])
    user = LLMRequest(messages=[LLMMessage(role="user", content="Инструкция")])
    assert request_input_sha256(system) != request_input_sha256(user)


# ---------------------------------------------------------------------------
# LLMResponse / usage
# ---------------------------------------------------------------------------


def test_response_holds_text_model_usage_and_created_at() -> None:
    created = _tz_now()
    response = LLMResponse(
        text="Ответ модели",
        model="local-test",
        usage=LLMUsage(input_tokens=10, output_tokens=3),
        created_at=created,
    )
    assert response.text == "Ответ модели"
    assert response.usage == LLMUsage(input_tokens=10, output_tokens=3)
    assert response.created_at == created


def test_response_accepts_missing_optional_metadata() -> None:
    response = LLMResponse(text="Ответ")
    assert response.model is None
    assert response.usage is None
    assert response.created_at is None


def test_response_rejects_naive_created_at() -> None:
    with pytest.raises(ValidationError):
        LLMResponse(text="Ответ", created_at=datetime(2026, 1, 1))


def test_usage_rejects_negative_tokens() -> None:
    with pytest.raises(ValidationError):
        LLMUsage(input_tokens=-1, output_tokens=0)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _provenance() -> LLMCallProvenance:
    now = _tz_now()
    return LLMCallProvenance(
        provider_type="openai_compatible",
        model="local-test",
        prompt_version="v1",
        status=LLMCallOutcome.success,
        usage=LLMUsage(input_tokens=4, output_tokens=2),
        input_sha256="ab" * 32,
        started_at=now,
        finished_at=now,
    )


def test_provenance_fields_roundtrip() -> None:
    provenance = _provenance()
    assert provenance.status is LLMCallOutcome.success
    assert provenance.error_class is None
    assert provenance.status_code is None
    assert provenance.usage == LLMUsage(input_tokens=4, output_tokens=2)


def test_provenance_serialization_has_no_content_fields() -> None:
    """Provenance carries only metadata — its schema has no content/text field."""
    provenance = _provenance()
    assert "content" not in LLMCallProvenance.model_fields
    assert "text" not in LLMCallProvenance.model_fields
    dumped = provenance.model_dump_json()
    assert "input_sha256" in dumped


def test_provenance_requires_tz_aware_timestamps() -> None:
    base = _provenance()
    with pytest.raises(ValidationError):
        LLMCallProvenance(
            **{
                **base.model_dump(),
                "started_at": datetime(2026, 1, 1),
            }
        )
    with pytest.raises(ValidationError):
        LLMCallProvenance(
            **{
                **base.model_dump(),
                "finished_at": datetime(2026, 1, 1),
            }
        )


def test_utc_now_is_timezone_aware() -> None:
    now = utc_now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timezone.utc.utcoffset(None)
