"""Append-only result store of the phase-2 second anchor family (task T08).

The phase-2 layers (``stt_corrected``, ``daily_summary``, ``weekly_summary``,
``coach_note``) are anchored directly to their source instead of a weekly
export (ADR-007 §1; persistence shape: ADR-008). Their results live in the
existing ``artifacts`` table — no parallel table and no parallel anchor system
— and this module is the only write path to them:

* :func:`save_layer_result` — append one result for an (anchor, layer) pair.
  The version is **computed, never passed in**: a repeated run appends ``v+1``
  and the previous row keeps its content, path and hash (ADR-005,
  ``docs/data-lineage.md`` §3). Nothing in this module updates or deletes a
  row, and nothing here writes to a raw source table (``messages``,
  ``audio_files``, ``transcriptions``, ``weekly_exports``) — the schema
  constraints in :mod:`diary.db.models` enforce the anchor invariants as a
  second line of defence;
* :func:`list_layer_results` / :func:`latest_layer_result` — read a result
  lineage by its anchor (message, local day, ISO week, upstream summary);
* :func:`trace_layer_result` — the traceability answer of ADR-007 §7 and
  ``docs/data-lineage.md`` §7: which source the result came from, which run
  produced it, which version it is and what its hash is.

Errors are domain errors with static, non-sensitive messages: no diary
content, prompt text or LLM output is ever put into an exception message or a
log line (AGENTS.md §7-8).
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from diary.core.settings import get_settings
from diary.db.base import utc_now
from diary.db.models import (
    SECOND_FAMILY_LAYER_ANCHORS,
    Artifact,
    ArtifactLayer,
    Message,
    ProcessingRun,
)

__all__ = [
    "AnchorError",
    "AnchorSourceError",
    "LayerResultError",
    "LayerResultTrace",
    "ResultPayloadError",
    "UnsupportedLayerError",
    "WEEK_ID_PATTERN",
    "anchor_field_for",
    "latest_layer_result",
    "list_layer_results",
    "next_result_version",
    "save_layer_result",
    "trace_layer_result",
]

#: Shape of the ``weekly_summary`` anchor (ADR-007 §5: ISO year + ISO week).
WEEK_ID_PATTERN = r"^\d{4}-W\d{2}$"
_WEEK_ID_RE = re.compile(WEEK_ID_PATTERN)

#: Shape of a stored SHA-256 digest.
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Kind of source an anchor points at, for :func:`trace_layer_result`.
_SOURCE_KINDS: dict[str, str] = {
    "raw_id": "message",
    "local_date": "local_day",
    "week_id": "iso_week",
    "summary_id": "artifact",
}


# ---------------------------------------------------------------------------
# Domain errors (static messages; never diary content)
# ---------------------------------------------------------------------------


class LayerResultError(Exception):
    """Base class for second-family result errors (never diary content)."""


class UnsupportedLayerError(LayerResultError):
    """The layer is not one of the phase-2 (second anchor family) layers."""


class AnchorError(LayerResultError):
    """The anchor of a result is missing, unknown or ambiguous."""


class AnchorSourceError(LayerResultError):
    """The anchor points at a source that does not exist or has the wrong kind."""


class ResultPayloadError(LayerResultError):
    """The result stores neither a rendered file nor a JSON result payload."""


# ---------------------------------------------------------------------------
# Anchor helpers
# ---------------------------------------------------------------------------


def anchor_field_for(layer: ArtifactLayer) -> str:
    """The single anchor column of a second-family ``layer``.

    Raises :class:`UnsupportedLayerError` for a first-family layer: results of
    layers 1-4 are written by the export/pipeline services through
    :class:`~diary.db.repositories.ArtifactRepository`, never here.
    """
    try:
        return SECOND_FAMILY_LAYER_ANCHORS[layer]
    except KeyError as exc:
        raise UnsupportedLayerError(
            f"Layer {layer.value!r} belongs to the export anchor family; only the phase-2 "
            "layers can be stored as second-family results."
        ) from exc


def _resolve_anchor(
    layer: ArtifactLayer,
    *,
    raw_id: uuid.UUID | None,
    local_date: date | None,
    local_tz: str | None,
    week_id: str | None,
    summary_id: uuid.UUID | None,
) -> dict[str, Any]:
    """Validate the anchor arguments of ``layer`` and return the DB columns.

    Exactly the anchor column of the layer must be set and every other anchor
    column must stay ``None`` — the same rule the ``ck_artifacts_anchor_family``
    check imposes on the table. ``local_tz`` accompanies ``local_date`` (a local
    date without its timezone is ambiguous) and defaults to the configured
    application timezone, which is what the day boundary is computed in.
    """
    expected = anchor_field_for(layer)
    provided = {
        "raw_id": raw_id,
        "local_date": local_date,
        "week_id": week_id,
        "summary_id": summary_id,
    }
    values = {name: value for name, value in provided.items() if value is not None}
    unexpected = sorted(set(values) - {expected})
    if unexpected:
        raise AnchorError(
            f"Layer {layer.value!r} is anchored by {expected!r}; "
            f"anchor(s) {', '.join(unexpected)} must not be set."
        )
    if expected not in values:
        raise AnchorError(f"Layer {layer.value!r} requires the {expected!r} anchor.")
    if expected == "week_id":
        assert week_id is not None  # guaranteed by the anchor check above
        if not _WEEK_ID_RE.fullmatch(week_id):
            raise AnchorError(
                "week_id must look like YYYY-Www (ISO year and ISO week, e.g. 2026-W38)."
            )
        if local_tz is not None:
            raise AnchorError("'local_tz' belongs to the daily_summary anchor only.")
        return {"week_id": week_id}
    if expected == "local_date":
        assert local_date is not None  # guaranteed by the anchor check above
        return {"local_date": local_date, "local_tz": local_tz or get_settings().timezone}
    if local_tz is not None:
        raise AnchorError(
            f"Layer {layer.value!r} is anchored by {expected!r}; 'local_tz' belongs to the "
            "daily_summary anchor only."
        )
    return {expected: values[expected]}


def _resolve_anchor_source(
    session: Session,
    layer: ArtifactLayer,
    anchor: dict[str, Any],
    *,
    processing_run_id: uuid.UUID | None,
) -> None:
    """Verify that the anchor and the run point at existing rows.

    The database foreign keys cover this as well; checking here turns an
    ``IntegrityError`` into a clear domain error and keeps the invariant "a
    result is always traceable to its source and to its run" explicit.
    """
    if layer is ArtifactLayer.coach_note:
        upstream = session.get(Artifact, anchor["summary_id"])
        if upstream is None:
            raise AnchorSourceError("The weekly summary this note is anchored to does not exist.")
        if upstream.layer is not ArtifactLayer.weekly_summary:
            raise AnchorSourceError(
                "A coach note can only be built from a weekly_summary artifact; the anchor "
                f"points at a {upstream.layer.value!r} artifact."
            )
    if layer is ArtifactLayer.stt_corrected:
        if session.get(Message, anchor["raw_id"]) is None:
            raise AnchorSourceError("The message this correction is anchored to does not exist.")
    if processing_run_id is not None and session.get(ProcessingRun, processing_run_id) is None:
        raise AnchorSourceError("The processing run recorded for this result does not exist.")


# ---------------------------------------------------------------------------
# Result payload helpers
# ---------------------------------------------------------------------------


def normalize_result_payload(
    result: Mapping[str, Any] | BaseModel | None,
) -> dict[str, object] | None:
    """Normalize a layer result into the JSON stored in ``result_json``.

    A validated Pydantic contract of the phase-2 family is serialized with
    ``model_dump(mode="json")`` (AGENTS.md §15: only validated structured output
    reaches storage), a mapping is copied as-is, and anything else is rejected.
    The payload must be JSON-serializable — the column is a JSON column, and a
    silently lossy ``default=str`` is exactly what this check prevents.
    """
    if result is None:
        return None
    if isinstance(result, BaseModel):
        payload: dict[str, Any] = result.model_dump(mode="json")
    elif isinstance(result, Mapping):
        payload = dict(result)
    else:
        raise ResultPayloadError(
            "The layer result must be a validated Pydantic contract or a mapping."
        )
    try:
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ResultPayloadError("The layer result is not JSON-serializable.") from exc
    return payload


def _normalize_file(path: str | None, sha256: str | None) -> tuple[str | None, str | None]:
    """Validate the optional rendered-file pair (path + digest) of a result.

    The path is server-generated, so it must be relative to the data root with
    forward slashes and without traversal segments (AGENTS.md §7); the digest
    must be a SHA-256 and is stored together with the path or not at all.
    """
    if path is None:
        if sha256 is not None:
            raise ResultPayloadError("A sha256 without a rendered file path makes no sense.")
        return None, None
    if not path or path != path.strip():
        raise ResultPayloadError("The artifact path must be non-empty without surrounding space.")
    if path.startswith("/") or path.startswith("\\") or "\\" in path:
        raise ResultPayloadError("The artifact path must be data-root relative (forward slashes).")
    if ".." in path.split("/"):
        raise ResultPayloadError("The artifact path must not contain '..' segments.")
    if sha256 is None or not _SHA256_RE.fullmatch(sha256):
        raise ResultPayloadError(
            "A rendered artifact file must be stored together with its SHA-256 digest."
        )
    return path, sha256


# ---------------------------------------------------------------------------
# Read path (by anchor)
# ---------------------------------------------------------------------------


def list_layer_results(
    session: Session,
    *,
    layer: ArtifactLayer | None = None,
    raw_id: uuid.UUID | None = None,
    local_date: date | None = None,
    local_tz: str | None = None,
    week_id: str | None = None,
    summary_id: uuid.UUID | None = None,
) -> list[Artifact]:
    """Second-family results matching an anchor, in deterministic order.

    At most one anchor kind may be given (the two families are never mixed), and
    when both a layer and an anchor are given they must belong together. Rows
    are ordered by creation time and then by version, so the last element of a
    single (anchor, layer) lineage is its newest version.
    """
    statement = select(Artifact).where(Artifact.export_id.is_(None))
    if layer is not None:
        statement = statement.where(Artifact.layer == layer)
    filters: dict[str, Any] = {
        "raw_id": raw_id,
        "local_date": local_date,
        "week_id": week_id,
        "summary_id": summary_id,
    }
    given = {name: value for name, value in filters.items() if value is not None}
    if len(given) > 1:
        raise AnchorError(
            f"A result lineage has exactly one anchor; got {', '.join(sorted(given))} at once."
        )
    if given and layer is not None:
        expected = anchor_field_for(layer)
        if expected not in given:
            name = next(iter(given))
            raise AnchorError(
                f"Layer {layer.value!r} is anchored by {expected!r}, not by {name!r}."
            )
    if given:
        name, value = next(iter(given.items()))
        statement = statement.where(getattr(Artifact, name) == value)
        if name == "local_date" and local_tz is not None:
            statement = statement.where(Artifact.local_tz == local_tz)
    elif local_tz is not None:
        raise AnchorError("'local_tz' narrows the daily_summary anchor and needs 'local_date'.")
    statement = statement.order_by(
        Artifact.created_at.asc(), Artifact.version.asc().nullsfirst(), Artifact.id.asc()
    )
    return list(session.scalars(statement))


def latest_layer_result(
    session: Session,
    *,
    layer: ArtifactLayer,
    raw_id: uuid.UUID | None = None,
    local_date: date | None = None,
    local_tz: str | None = None,
    week_id: str | None = None,
    summary_id: uuid.UUID | None = None,
) -> Artifact | None:
    """The newest version of one (anchor, layer) pair, or ``None``.

    The anchor must belong to ``layer`` (the rule :func:`save_layer_result`
    applies), so a wrong-anchor lookup fails loudly instead of silently
    returning nothing.
    """
    anchor = _resolve_anchor(
        layer,
        raw_id=raw_id,
        local_date=local_date,
        local_tz=local_tz,
        week_id=week_id,
        summary_id=summary_id,
    )
    rows = list_layer_results(session, layer=layer, **anchor)
    if not rows:
        return None
    return max(rows, key=lambda row: row.version or 0)


def next_result_version(
    session: Session,
    *,
    layer: ArtifactLayer,
    raw_id: uuid.UUID | None = None,
    local_date: date | None = None,
    local_tz: str | None = None,
    week_id: str | None = None,
    summary_id: uuid.UUID | None = None,
) -> int:
    """Next ``vM`` of the (anchor, layer) pair: newest version + 1, ``v1`` first.

    The version belongs to the pair, not to the contract (ADR-007 §2): a
    repeated run of the same layer over the same input appends the next version
    and never rewrites the previous result.
    """
    anchor = _resolve_anchor(
        layer,
        raw_id=raw_id,
        local_date=local_date,
        local_tz=local_tz,
        week_id=week_id,
        summary_id=summary_id,
    )
    rows = list_layer_results(session, layer=layer, **anchor)
    if not rows:
        return 1
    return max(row.version or 0 for row in rows) + 1


# ---------------------------------------------------------------------------
# Write path (append-only)
# ---------------------------------------------------------------------------


def save_layer_result(
    session: Session,
    *,
    layer: ArtifactLayer,
    result: Mapping[str, Any] | BaseModel | None = None,
    path: str | None = None,
    sha256: str | None = None,
    processing_run_id: uuid.UUID | None = None,
    created_at: datetime | None = None,
    raw_id: uuid.UUID | None = None,
    local_date: date | None = None,
    local_tz: str | None = None,
    week_id: str | None = None,
    summary_id: uuid.UUID | None = None,
) -> Artifact:
    """Append one second-family layer result and return the stored row.

    The row is anchored to exactly one source (``raw_id`` for ``stt_corrected``,
    ``local_date``/``local_tz`` for ``daily_summary``, ``week_id`` for
    ``weekly_summary``, ``summary_id`` for ``coach_note``) and stores the
    validated result payload (``result`` → ``result_json``) and/or the rendered
    file (``path`` + ``sha256``). At least one of the two must be given — a
    result row that stores nothing is rejected.

    The version is computed from the existing lineage
    (:func:`next_result_version`): re-running the same layer over the same input
    appends ``v+1``; nothing is overwritten and there is no parameter that could
    ask for a rewrite. The caller owns the transaction (this function flushes,
    never commits), so the result row and the processing run it belongs to are
    committed atomically. A duplicate version of the same pair — only reachable
    through a lost update race — is rejected by the
    ``uq_artifacts_*_layer_version`` constraints.

    Raises :class:`UnsupportedLayerError` for a first-family layer,
    :class:`AnchorError` for a wrong/ambiguous anchor,
    :class:`AnchorSourceError` when the anchor or the run does not exist, and
    :class:`ResultPayloadError` for an unusable payload/file pair.
    """
    payload = normalize_result_payload(result)
    relative_path, digest = _normalize_file(path, sha256)
    if payload is None and relative_path is None:
        raise ResultPayloadError(
            "A layer result must store a rendered file and/or a JSON result payload."
        )
    anchor = _resolve_anchor(
        layer,
        raw_id=raw_id,
        local_date=local_date,
        local_tz=local_tz,
        week_id=week_id,
        summary_id=summary_id,
    )
    _resolve_anchor_source(session, layer, anchor, processing_run_id=processing_run_id)
    artifact = Artifact(
        layer=layer,
        export_id=None,
        version=next_result_version(session, layer=layer, **anchor),
        path=relative_path,
        sha256=digest,
        result_json=payload,
        processing_run_id=processing_run_id,
        created_at=created_at or utc_now(),
        **anchor,
    )
    session.add(artifact)
    session.flush()
    return artifact


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerResultTrace:
    """Traceability record of one second-family result (data-lineage §7).

    ``anchor_field``/``anchor_value`` name the source of the result,
    ``source_kind`` says what kind of source it is (``message``, ``local_day``,
    ``iso_week`` or ``artifact``), ``source_message``/``source_artifact`` carry
    the resolved source row when it lives in this database, and ``run`` is the
    processing run that produced the result (``None`` if none was recorded).
    """

    artifact: Artifact
    layer: ArtifactLayer
    anchor_field: str
    anchor_value: object
    source_kind: str
    source_message: Message | None
    source_artifact: Artifact | None
    run: ProcessingRun | None


def trace_layer_result(session: Session, artifact_id: uuid.UUID) -> LayerResultTrace:
    """Resolve «result → source → run» for one second-family result.

    Raises :class:`LayerResultError` for an unknown artifact id and
    :class:`UnsupportedLayerError` for a first-family artifact, whose lineage
    runs through the weekly export instead (``ArtifactRepository`` and
    ``processing_runs.input_export_id``).
    """
    artifact = session.get(Artifact, artifact_id)
    if artifact is None:
        raise LayerResultError("Artifact not found.")
    anchor_field = anchor_field_for(artifact.layer)
    anchor_value = getattr(artifact, anchor_field)
    if anchor_value is None:
        raise LayerResultError(
            "The artifact carries no second-family anchor; its lineage is not a phase-2 result."
        )
    source_message: Message | None = None
    source_artifact: Artifact | None = None
    if anchor_field == "raw_id":
        source_message = session.get(Message, anchor_value)
    elif anchor_field == "summary_id":
        source_artifact = session.get(Artifact, anchor_value)
    run = (
        session.get(ProcessingRun, artifact.processing_run_id)
        if artifact.processing_run_id is not None
        else None
    )
    return LayerResultTrace(
        artifact=artifact,
        layer=artifact.layer,
        anchor_field=anchor_field,
        anchor_value=anchor_value,
        source_kind=_SOURCE_KINDS[anchor_field],
        source_message=source_message,
        source_artifact=source_artifact,
        run=run,
    )
