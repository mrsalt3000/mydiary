"""Integration tests for the T04 STT-correction layer writing through T08.

Runs against an isolated, migration-created test database (never the production
one) with a fake LLM provider, and covers the parts of the T04 definition of
done that need real persistence:

* a valid completion is validated and appended to ``artifacts`` under the
  ``raw_id`` anchor, with no weekly export and no rendered file;
* a repeated run appends ``v+1`` and leaves the previous version untouched
  (ADR-005);
* the raw source row (``messages``) is unchanged by the run — value by value —
  and the layer has no other write path;
* the stored payload is the validated contract, including ``uncertain``
  fragments that are verbatim substrings of the stored ``text``.

All content is synthetic (AGENTS.md §13); no endpoint, model or real
configuration file is used.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from diary.core.models_config import OpenAICompatibleProfile
from diary.db.models import Artifact, ArtifactLayer, Message, MessageKind, PipelineType, RunStatus
from diary.db.repositories import MessageRepository, ProcessingRunRepository
from diary.db.results import list_layer_results, trace_layer_result
from diary.domain.contracts import RawEntry
from diary.domain.llm.errors import LLMStructuredOutputError
from diary.domain.llm.models import LLMRequest, LLMResponse
from diary.domain.pipelines.stt_fix.service import correct_stt

RAW_TEXT = "сегодня утром я превет какдила ходил в парк"
CORRECTED_TEXT = "Сегодня утром я привет как дела, ходил в парк."


class _FakeProvider:
    """Deterministic provider: one fixed response, recorded requests."""

    def __init__(self, payload: object | str) -> None:
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        self.response = LLMResponse(text=text)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.response


def _profile(model: str = "diary-stt") -> OpenAICompatibleProfile:
    return OpenAICompatibleProfile(
        provider="openai_compatible",
        base_url="http://127.0.0.1:1/v1",  # never contacted by the fake
        model=model,
        api_key_env="SYNTHETIC_PHASE2_KEY",
        temperature=0.15,
        max_tokens=3072,
    )


def _message(session: Session, *, body: str = RAW_TEXT) -> Message:
    return MessageRepository(session).create(
        kind=MessageKind.text,
        body_text=body,
        created_at=datetime(2026, 9, 19, 9, 30, tzinfo=timezone.utc),
    )


#: Timezone-aware moment of the entry the layer corrects (synthetic).
_ENTRY_TS = datetime(2026, 9, 19, 9, 30, tzinfo=timezone.utc)


def _entry(message: Message) -> RawEntry:
    return RawEntry(id=message.id, ts=_ENTRY_TS, source="text", text=message.body_text)


def _run(session: Session):
    return ProcessingRunRepository(session).create(
        pipeline_type=PipelineType.stt_corrected, status=RunStatus.succeeded
    )


def _message_values(session: Session, message_id: uuid.UUID) -> dict[str, object]:
    """Every column value of the raw message row (value-by-value comparison)."""
    row = session.scalars(select(Message).where(Message.id == message_id)).one()
    return {column.name: getattr(row, column.name) for column in Message.__table__.columns}


# ---------------------------------------------------------------------------
# storing a validated correction
# ---------------------------------------------------------------------------


def test_valid_correction_is_appended_under_the_message_anchor(session: Session) -> None:
    message = _message(session)
    raw_before = _message_values(session, message.id)
    provider = _FakeProvider({"text": CORRECTED_TEXT, "uncertain": []})

    outcome = correct_stt(
        session,
        entry=_entry(message),
        profile=_profile(),
        provider=provider,
        processing_run_id=_run(session).id,
    )
    session.commit()

    stored = session.get(Artifact, outcome.artifact.id)
    assert stored is not None
    assert stored.layer is ArtifactLayer.stt_corrected
    assert stored.export_id is None  # second anchor family: no weekly export
    assert stored.raw_id == message.id
    assert stored.local_date is None and stored.week_id is None and stored.summary_id is None
    assert stored.version == 1
    assert stored.path is None and stored.sha256 is None  # stored inline (ADR-008 §1)
    assert stored.result_json == {
        "raw_id": str(message.id),
        "text": CORRECTED_TEXT,
        "uncertain": [],
        "model": "diary-stt",
        "prompt_ver": "v1",
    }

    # The raw source row is untouched, value by value.
    assert _message_values(session, message.id) == raw_before
    assert _message_values(session, message.id)["body_text"] == RAW_TEXT

    # The result is traceable back to the message that produced it.
    trace = trace_layer_result(session, stored.id)
    assert trace.anchor_field == "raw_id"
    assert trace.source_kind == "message"
    assert trace.source_message is not None
    assert trace.source_message.id == message.id


def test_uncertain_fragments_are_stored_verbatim_substrings(session: Session) -> None:
    fragment = "три тыщи восемьсот"
    # The source carries the fragment, so the correction stays grounded (the
    # runtime gate refuses an invented quantity, T17b).
    message = _message(session, body=f"вечером пазваню в банк {fragment}")
    text = f"Вечером позвоню в банк, {fragment}."
    provider = _FakeProvider({"text": text, "uncertain": [fragment]})

    outcome = correct_stt(session, entry=_entry(message), profile=_profile(), provider=provider)
    session.commit()

    stored = session.get(Artifact, outcome.artifact.id)
    assert stored is not None and stored.result_json is not None
    assert stored.result_json["text"] == text
    assert stored.result_json["uncertain"] == [fragment]
    assert fragment in stored.result_json["text"]
    # No marker was added to the text by the layer.
    assert "[?]" not in stored.result_json["text"]


def test_a_rerun_appends_a_new_version_and_keeps_the_previous_one(session: Session) -> None:
    message = _message(session)
    first_provider = _FakeProvider({"text": CORRECTED_TEXT, "uncertain": []})
    second_provider = _FakeProvider({"text": "Другой вариант правки.", "uncertain": []})

    first = correct_stt(session, entry=_entry(message), profile=_profile(), provider=first_provider)
    session.commit()
    second = correct_stt(
        session, entry=_entry(message), profile=_profile(), provider=second_provider
    )
    session.commit()

    assert first.artifact.version == 1
    assert second.artifact.version == 2
    lineage = list_layer_results(session, layer=ArtifactLayer.stt_corrected, raw_id=message.id)
    assert [row.version for row in lineage] == [1, 2]
    # Append-only: the first version still carries its own text (ADR-005).
    assert lineage[0].result_json is not None and lineage[0].result_json["text"] == CORRECTED_TEXT
    assert lineage[1].result_json is not None
    assert lineage[1].result_json["text"] == "Другой вариант правки."


# ---------------------------------------------------------------------------
# a failed run stores nothing
# ---------------------------------------------------------------------------


def test_invalid_completion_stores_nothing_and_leaves_the_raw_row_alone(
    session: Session,
) -> None:
    message = _message(session)
    session.commit()  # the raw source row is persisted before the run
    raw_before = _message_values(session, message.id)
    provider = _FakeProvider("Не JSON вовсе.")

    with pytest.raises(LLMStructuredOutputError):
        correct_stt(session, entry=_entry(message), profile=_profile(), provider=provider)
    session.rollback()

    assert list_layer_results(session, layer=ArtifactLayer.stt_corrected, raw_id=message.id) == []
    assert _message_values(session, message.id) == raw_before
