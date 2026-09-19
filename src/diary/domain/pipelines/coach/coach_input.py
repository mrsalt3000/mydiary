"""Read-only selection of exactly one weekly summary for the coaching note (task T07).

The coaching note (layer 5) is built from **one** upstream artifact: the newest
stored ``weekly_summary`` result of an ISO week (T07 decision 1 of the task
brief: «Вход — ровно одна недельная сводка (артефакт слоя ``weekly_summary``,
его новейшая версия). Ни raw, ни дневные сводки напрямую»). This module is
that selection concern, kept apart from the LLM call:

* :func:`week_summary_artifact` — the newest ``weekly_summary`` artifact of a
  week, read through the task-T08 store (``latest_layer_result``);
* :func:`coach_input_from_artifact` — validate a stored artifact into a
  :class:`CoachInput` (the ``WeeklySummary`` contract **plus the artifact row**
  whose id becomes the storage anchor);
* :func:`collect_week_summary` — the week's single coaching input, or a loud
  failure when the week has no weekly summary yet;
* :func:`ensure_input_matches_week` — the guard that the summary fed to a run
  belongs to the run's week id (makes the caller-supplied seam safe).

Two identities, one name (T00 / ADR-008 §5)
===========================================

The name ``summary_id`` appears twice in this layer and means two different
things:

* the **contract field** ``CoachNote.summary_id`` is a **string** and carries
  the ``week_id`` of the upstream weekly summary (``YYYY-Www``) — this is what
  T00 froze and what :func:`CoachNotePipeline.parse_result` fills from the
  backend;
* the **storage anchor** ``artifacts.summary_id`` is a **UUID**
  (``ForeignKey("artifacts.id")``) and carries the **artifact id** of that same
  upstream weekly summary — this is what :func:`save_layer_result` receives and
  what the schema invariant ``ck_artifacts_anchor_family`` checks.

Both values come from the backend, never from the model. The layer keeps them
strictly apart and records the relation explicitly:

    storage anchor (artifacts.summary_id)  = CoachInput.artifact.id
    contract field (CoachNote.summary_id)  = CoachInput.artifact.week_id
                                             = CoachInput.summary.week_id

Read-only by construction: the module reads ``artifacts`` rows of the
``weekly_summary`` layer and never writes anywhere; ``messages``,
``transcriptions``, ``daily_summary`` results and ``weekly_exports`` are never
touched. A stored payload that does not validate back into ``WeeklySummary``,
or whose ``week_id`` disagrees with its own anchor, raises
:class:`CoachInputError` instead of being silently accepted (a lying anchor is
a veto of this layer).

Privacy: the module returns validated contract objects; it never logs and never
puts source text into an error message (AGENTS.md §7-8).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from pydantic import ValidationError
from sqlalchemy.orm import Session

from diary.db.models import Artifact, ArtifactLayer
from diary.db.results import latest_layer_result
from diary.domain.contracts import WeeklySummary
from diary.domain.contracts.weekly_summary import validate_week_id
from diary.domain.pipelines.base import PipelinePermanentError

__all__ = [
    "CoachInput",
    "CoachInputError",
    "coach_input_from_artifact",
    "collect_week_summary",
    "ensure_input_matches_week",
    "week_summary_artifact",
]

#: Static, non-sensitive error summaries (AGENTS.md §8).
_BAD_WEEK_ID_MESSAGE = "week_id must be an existing ISO week (YYYY-Www, ISO year)."
_WRONG_LAYER_MESSAGE = "A coaching note is built only from a weekly_summary artifact."
_NO_SUMMARY_MESSAGE = "The ISO week has no weekly summary to build a coaching note from."
_CORRUPT_SUMMARY_MESSAGE = (
    "The stored weekly summary does not validate into the weekly_summary contract."
)
_MISPLACED_SUMMARY_MESSAGE = (
    "The stored weekly summary reports a different week id than its own anchor."
)
_FOREIGN_SUMMARY_MESSAGE = (
    "The weekly summary belongs to a different ISO week than the week_id of this run."
)


class CoachInputError(PipelinePermanentError):
    """The coaching note has no usable weekly-summary input.

    Raised for an unusable input — a malformed ``week_id``, a week without a
    stored weekly summary, an artifact of another layer, a stored payload that
    no longer validates, or a summary of another week. The message is a static,
    non-sensitive summary (AGENTS.md §8); a failed run is never retried.
    """


@dataclass(frozen=True)
class CoachInput:
    """One run's input: the validated week summary and the artifact it came from.

    ``artifact`` is the stored ``weekly_summary`` row: its ``id`` is the
    **storage anchor** of the note (``artifacts.summary_id``, ADR-008 §5) and
    its ``week_id`` is the value of the **contract field**
    ``CoachNote.summary_id``. Keeping the row (not only the contract) is what
    makes the anchor unavoidable: the layer cannot store a note without naming
    the exact upstream artifact it was built from.
    """

    #: The validated upstream summary (the layer's declared input contract).
    summary: WeeklySummary
    #: The upstream ``weekly_summary`` artifact row the summary was read from.
    artifact: Artifact

    @property
    def week_id(self) -> str:
        """The ISO week id of the input (the contract's ``summary_id`` value)."""
        return self.summary.week_id

    @property
    def artifact_id(self) -> uuid.UUID:
        """The storage anchor: the id of the upstream ``weekly_summary`` artifact."""
        return self.artifact.id


def _validate_week_id(week_id: str) -> None:
    """Fail loudly on a malformed or non-existent ISO week id."""
    try:
        validate_week_id(week_id)
    except ValueError as exc:
        raise CoachInputError(_BAD_WEEK_ID_MESSAGE) from exc


def week_summary_artifact(session: Session, *, week_id: str) -> Artifact | None:
    """The newest ``weekly_summary`` artifact of the ISO week ``week_id``.

    The week's lineage is read through the task-T08 store: a re-run of the week
    appends ``v+1``, and the newest version is the week's current summary (T07
    decision 1: «его новейшая версия»). Returns ``None`` when the week has no
    weekly summary yet. The ``week_id`` shape is validated by the store.
    """
    return latest_layer_result(session, layer=ArtifactLayer.weekly_summary, week_id=week_id)


def coach_input_from_artifact(artifact: Artifact) -> CoachInput:
    """Validate one stored artifact into a :class:`CoachInput`.

    The artifact must be a ``weekly_summary`` row, must carry the inline
    validated payload of that layer (``result_json`` — a summary stored only as
    a rendered file is not this layer's input) and the payload's ``week_id``
    must equal the anchor's ``week_id``. Every violation raises
    :class:`CoachInputError`; nothing is repaired or guessed.
    """
    if artifact.layer is not ArtifactLayer.weekly_summary or artifact.week_id is None:
        raise CoachInputError(_WRONG_LAYER_MESSAGE)
    payload = artifact.result_json
    if not isinstance(payload, dict):
        raise CoachInputError(_CORRUPT_SUMMARY_MESSAGE)
    try:
        summary = WeeklySummary.model_validate(payload)
    except ValidationError as exc:
        raise CoachInputError(_CORRUPT_SUMMARY_MESSAGE) from exc
    if summary.week_id != artifact.week_id:
        raise CoachInputError(_MISPLACED_SUMMARY_MESSAGE)
    return CoachInput(summary=summary, artifact=artifact)


def ensure_input_matches_week(coach_input: CoachInput, week_id: str) -> None:
    """Fail loudly unless ``week_id`` is valid and the input belongs to it.

    The run's identity is the ``week_id``; a summary of another week inside the
    run would make the recorded anchor a lie. The collected input satisfies
    this by construction — the check is what makes the caller-supplied
    ``coach_input`` seam (tests, the T09 orchestrator) safe.
    """
    _validate_week_id(week_id)
    if coach_input.summary.week_id != week_id:
        raise CoachInputError(_FOREIGN_SUMMARY_MESSAGE)


def collect_week_summary(session: Session, *, week_id: str) -> CoachInput:
    """The single coaching input of the ISO week ``week_id``.

    Reads exactly one artifact — the newest ``weekly_summary`` result of the
    week — and returns it as a validated :class:`CoachInput`. ``messages``,
    ``transcriptions``, the ``daily_summary`` layer and ``weekly_exports`` are
    never read: the input is the already grounded week summary, never raw
    material and never the day layer directly. A week without a weekly summary
    fails loudly instead of asking the model to invent the week.
    """
    _validate_week_id(week_id)
    artifact = week_summary_artifact(session, week_id=week_id)
    if artifact is None:
        raise CoachInputError(_NO_SUMMARY_MESSAGE)
    return coach_input_from_artifact(artifact)
