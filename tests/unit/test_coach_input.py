"""Unit tests for the T07 weekly-summary input of the coaching-note layer.

The selection concern of T07 is deliberately small, and its pure part — turning
a stored ``weekly_summary`` artifact row into a validated
:class:`~diary.domain.pipelines.coach.coach_input.CoachInput` — is testable
without a database:

* only a ``weekly_summary`` row is acceptable: an artifact of another layer (in
  particular layer 4 ``wellbeing``) or a row without a week anchor is rejected;
* the input needs the inline validated payload; a summary stored only as a
  rendered file is not this layer's input;
* a payload that no longer validates into :class:`WeeklySummary`, or whose
  ``week_id`` disagrees with the artifact's own anchor, fails loudly instead of
  being silently accepted;
* the run's ``week_id`` must be the id of the summary that is actually fed.

The database read itself (newest version of the week, missing week) is covered
by the integration tests, which own a migrated database. All values are
synthetic (AGENTS.md §13).
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest

from diary.db.models import Artifact, ArtifactLayer
from diary.domain.contracts import WeeklySummary
from diary.domain.pipelines.coach.coach_input import (
    CoachInput,
    CoachInputError,
    coach_input_from_artifact,
    ensure_input_matches_week,
)

WEEK_ID = "2026-W38"
OTHER_WEEK_ID = "2026-W39"
DAILY_IDS = ["2026-09-14", "2026-09-15"]
ROLLUP = "Синтетическая неделя вокруг одного проекта."


def _summary(week_id: str = WEEK_ID) -> WeeklySummary:
    return WeeklySummary(week_id=week_id, daily_ids=DAILY_IDS, rollup=ROLLUP)


#: Marks "the caller did not override the inline payload".
_DEFAULT_PAYLOAD = object()


def _artifact(
    *,
    layer: ArtifactLayer = ArtifactLayer.weekly_summary,
    week_id: str | None = WEEK_ID,
    payload: object = _DEFAULT_PAYLOAD,
    path: str | None = None,
) -> Artifact:
    """A detached artifact row; the layer only reads its columns."""
    inline = _summary().model_dump(mode="json") if payload is _DEFAULT_PAYLOAD else payload
    return Artifact(
        id=uuid.uuid4(),
        layer=layer,
        week_id=week_id,
        version=1,
        result_json=inline,
        path=path,
        sha256="a" * 64 if path is not None else None,
    )


# ---------------------------------------------------------------------------
# a valid stored summary becomes a CoachInput
# ---------------------------------------------------------------------------


def test_a_stored_week_summary_becomes_the_input() -> None:
    artifact = _artifact()

    coach_input = coach_input_from_artifact(artifact)

    assert isinstance(coach_input, CoachInput)
    assert coach_input.summary == _summary()
    assert coach_input.artifact is artifact
    # The two identities of the run, kept apart on purpose.
    assert coach_input.week_id == WEEK_ID  # contract field: a week id string
    assert isinstance(coach_input.week_id, str)
    assert coach_input.artifact_id == artifact.id  # storage anchor: an artifact id
    assert isinstance(coach_input.artifact_id, uuid.UUID)


# ---------------------------------------------------------------------------
# the stored row must be a weekly summary with an inline payload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layer",
    [ArtifactLayer.wellbeing, ArtifactLayer.daily_summary, ArtifactLayer.coach_note],
)
def test_an_artifact_of_another_layer_is_rejected(layer: ArtifactLayer) -> None:
    with pytest.raises(CoachInputError):
        coach_input_from_artifact(_artifact(layer=layer, week_id=None))


def test_an_artifact_without_a_week_anchor_is_rejected() -> None:
    with pytest.raises(CoachInputError):
        coach_input_from_artifact(_artifact(week_id=None))


def test_a_summary_stored_only_as_a_file_is_rejected() -> None:
    """A file-backed result carries no payload to read, so it is not an input."""
    artifact = _artifact(
        payload=None,
        path="exports/05_weekly_summary/2026-W38_weekly_summary_v1.md",
    )

    assert artifact.result_json is None
    with pytest.raises(CoachInputError):
        coach_input_from_artifact(artifact)


@pytest.mark.parametrize(
    "payload",
    [
        {},  # no fields at all
        {"week_id": WEEK_ID},  # no rollup / daily_ids
        {"week_id": "2026-W54", "daily_ids": DAILY_IDS, "rollup": ROLLUP},  # not a real ISO week
        {"week_id": WEEK_ID, "daily_ids": [], "rollup": ROLLUP},  # no day at all
        {"week_id": WEEK_ID, "daily_ids": DAILY_IDS, "rollup": ""},  # empty rollup
        {"week_id": WEEK_ID, "daily_ids": DAILY_IDS, "rollup": ROLLUP, "extra": 1},  # extra field
        "not-a-mapping",
    ],
)
def test_a_payload_that_does_not_validate_is_rejected(payload: object) -> None:
    with pytest.raises(CoachInputError):
        coach_input_from_artifact(_artifact(payload=payload))


def test_a_payload_of_another_week_than_the_anchor_is_rejected() -> None:
    """A lying anchor (payload says W39, row anchored to W38) fails loudly."""
    payload = _summary(OTHER_WEEK_ID).model_dump(mode="json")

    with pytest.raises(CoachInputError):
        coach_input_from_artifact(_artifact(week_id=WEEK_ID, payload=payload))


# ---------------------------------------------------------------------------
# the membership guard
# ---------------------------------------------------------------------------


def _coach_input(week_id: str = WEEK_ID) -> CoachInput:
    summary = _summary(week_id)
    return CoachInput(summary=summary, artifact=_artifact(week_id=week_id))


def test_the_matching_week_passes_the_guard() -> None:
    ensure_input_matches_week(_coach_input(), WEEK_ID)  # does not raise


def test_a_summary_of_another_week_fails_the_guard() -> None:
    with pytest.raises(CoachInputError):
        ensure_input_matches_week(_coach_input(OTHER_WEEK_ID), WEEK_ID)


@pytest.mark.parametrize("week_id", ["2026-W54", "2026-38", "not-a-week", ""])
def test_a_malformed_week_id_fails_the_guard(week_id: str) -> None:
    with pytest.raises(CoachInputError):
        ensure_input_matches_week(_coach_input(), week_id)


def test_the_guard_does_not_depend_on_a_database() -> None:
    """The guard is pure: it never touches a session or the store."""
    coach_input = _coach_input()
    ensure_input_matches_week(coach_input, WEEK_ID)
    assert coach_input.summary.daily_ids == DAILY_IDS
    assert date.fromisoformat(DAILY_IDS[0]).isoformat() == DAILY_IDS[0]
