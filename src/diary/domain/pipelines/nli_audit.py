"""Owner-facing NLI report over a **stored** day summary: an instrument, not a gate (T27).

**The gap this closes.** The NLI rules of T24a could only be pointed at the synthetic
golden set (``eval/run.py --nli``), so the owner of the diary had no way to ask the one
question the instrument exists for about *their own* data: does the day summary I have
stored contradict the day it was written from? This module answers exactly that over the
application database and prints a **content-free report** — coordinates (``facts#2``),
probabilities, verdicts and counters.

**It is not a gate.** Nothing it computes is stored, nothing it finds is fed back into a
layer, and its exit code says only whether the report could be produced — never that a
summary «passed». The only thing allowed to refuse a summary stays the runtime rule of
T24b (:mod:`diary.domain.pipelines.nli_guard`, ``ADR-014``). The rule and the scope are
shared *by construction* — the same ``claims_for`` / ``judge_claims`` of
:mod:`diary.domain.contracts.nli` and the same :data:`GUARDED_CONTRACT` — so the tool
cannot report on something the product does not stand behind.

**Scope: the day layer only.** The weekly rollup is outside the rule after T26
(``ADR-015``: the week's own formulation was measured and would have refused every one of
ten live weekly summaries), and the coaching note was never in scope. The tool therefore
does not offer a weekly check: offering one would imply a guarantee the product does not
make (``docs/nli-validation-analysis.md`` §10).

**The source is reconstructed the way the layer built it, never «something similar».**
A ``daily_summary`` artifact carries its own anchor — ``local_date`` and ``local_tz``,
guaranteed by the schema (``ck_artifacts_anchor_local_tz``) and reaching the database
since T30 — so the premise is rebuilt through
:func:`~diary.domain.pipelines.sum_daily.day_input.collect_day_entries`, the very
selection the layer itself used. When the source cannot be reconstructed (no structured
result stored, an incomplete anchor, a day that has no active entries any more, a voice
message without its audio row) the audit **refuses** with a clear message instead of
judging against a reconstruction: a report built on a guess would look like a
measurement while measuring nothing.

**Fail-closed when the instrument is missing.** A missing ``nli`` extra, missing weights
or a hash mismatch is a
:class:`~diary.domain.pipelines.nli_audit.NliAuditUnavailableError` (a non-zero exit),
never an empty report — the same policy as the runtime gate and the harness (the owner's
decision for T24: «если NLI недоступен — Fail»).

**Closed when there is nothing to check.** An empty result — no stored day summary at
all, or none for the requested ``--date`` — is a clear message naming the command that
produces one, again with a non-zero exit. A green report over nothing would read as «the
stored summaries were checked».

**Privacy.** The claim texts go to the judge — the same disclosure the layer already makes
to its own model — and nowhere else: the report carries coordinates, probabilities and
counts, never diary text, and this module logs nothing (AGENTS.md §7-8).

**Read-only.** :func:`read_only_session` opens the database with SQLite ``mode=ro``, so
the audit cannot write even by mistake and SQLite itself refuses a write. The report also
never touches the artifact rows it reads.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import Engine
from sqlalchemy import create_engine as _sa_create_engine
from sqlalchemy.orm import Session, sessionmaker

from diary.core.settings import AppSettings, get_settings
from diary.db.models import Artifact, ArtifactLayer
from diary.db.results import list_layer_results
from diary.db.session import default_database_path
from diary.domain.contracts.nli import (
    CONTRADICTION_THRESHOLD,
    NliError,
    NliJudge,
    NliMetrics,
    NliPremiseError,
    claims_for,
    judge_claims,
    metrics_for,
    premise_from_texts,
    premise_sentences,
)
from diary.domain.nli.onnx_judge import NliUnavailableError
from diary.domain.pipelines.base import PipelineError
from diary.domain.pipelines.nli_guard import GUARDED_CONTRACT, default_nli_judge
from diary.domain.pipelines.sum_daily.day_input import DayInputError, collect_day_entries

__all__ = [
    "ClaimAudit",
    "NliAuditError",
    "NliAuditNoArtifactError",
    "NliAuditReport",
    "NliAuditUnavailableError",
    "audit_stored_day",
    "day_source_texts",
    "format_report",
    "read_only_session",
    "select_daily_artifact",
]

#: Fixed, content-free failure when the instrument cannot be used (AGENTS.md §8).
_INSTRUMENT_UNAVAILABLE = (
    "the NLI instrument is unavailable, so the stored summary cannot be checked: "
    'install the optional extra (`pip install -e ".[nli]"`) and fetch the pinned weights '
    "with `scripts/fetch-nli-model.sh`"
)

#: Fixed, content-free failure when the stored summary has no reconstructible source.
_SOURCE_UNAVAILABLE = (
    "the source of the stored summary cannot be reconstructed from the database, so no "
    "report is produced (the audit never judges against a reconstruction)"
)

#: How one produces a stored day summary — named in every «nothing to check» message.
_RUN_A_DAY = "scripts/run-local.sh cli day --date YYYY-MM-DD"


class NliAuditError(PipelineError):
    """The stored-summary audit could not produce a report.

    Every message is static and carries no diary content (AGENTS.md §8); the caller
    (the CLI) turns it into a non-zero exit code.
    """


class NliAuditNoArtifactError(NliAuditError):
    """There is no stored day summary to check (or none for the requested day)."""


class NliAuditUnavailableError(NliAuditError):
    """The NLI instrument is requested but cannot be used — fail closed, never empty."""


@dataclass(frozen=True)
class ClaimAudit:
    """One claim's verdict, addressed by coordinate and never by text.

    ``p_contradiction`` is the claim-level aggregate the rule compares with
    :data:`~diary.domain.contracts.nli.CONTRADICTION_THRESHOLD` — reported by the same
    definition as the runtime gate and the harness, so the tool cannot drift from what it
    measures. For a claim every sentence of which the source supports it is ``0.0`` by
    construction (nothing of that claim is unsupported, so there is no contradiction
    evidence to weigh); that is the rule's own semantics, not a missing number.
    """

    #: ``<field>#<1-based index>`` — the address of the claim, never its text.
    coordinate: str
    #: Whether the contradiction rule may condemn this claim at all (T24a: label fields
    #: such as themes are judged but never condemned — a topic is not an assertion).
    propositional: bool
    #: Strongest contradiction among the claim's unsupported sentences.
    p_contradiction: float
    #: The rule's verdict label: ``entailment`` / ``contradiction`` / ``neutral``.
    verdict: str
    #: Whether the runtime rule would refuse this claim (propositional and over threshold).
    flagged: bool
    #: Raw ``p_contradiction >= CONTRADICTION_THRESHOLD``, regardless of scope. Kept
    #: apart from :attr:`flagged` because the two differ exactly where the measured
    #: exemptions live (a theme label read as a contradiction, T25 §8.4).
    above_threshold: bool
    #: 1-based index of the source sentence the verdict rests on (provenance, no text).
    governing_sentence: int


@dataclass(frozen=True)
class NliAuditReport:
    """The content-free report of one stored day summary (counts and coordinates only)."""

    #: ``artifacts.id`` of the audited day summary.
    artifact_id: uuid.UUID
    #: ``vM`` of the (day, layer) pair; ``None`` only for first-family rows.
    version: int | None
    #: The anchor taken from the artifact itself: the local day it summarizes.
    local_date: date
    #: The IANA timezone the day boundary was computed in, from the artifact.
    local_tz: str
    #: The processing run that produced the artifact, or ``None`` if it recorded none.
    processing_run_id: uuid.UUID | None
    #: Number of source entries the layer saw (the size of the reconstructed day).
    entries: int
    #: Number of sentences the premise was split into — what the claims were judged
    #: against one by one.
    premise_sentences: int
    #: One entry per claim of the summary, in the contract's own field order.
    claims: tuple[ClaimAudit, ...]
    #: The aggregate of the same judgements (counts, extremes, coordinates).
    metrics: NliMetrics

    @property
    def contradicting(self) -> int:
        """Claims the runtime rule would refuse (propositions over the threshold)."""
        return self.metrics.contradiction

    @property
    def above_threshold(self) -> int:
        """Claims whose measured contradiction reaches the threshold, scope aside."""
        return sum(1 for claim in self.claims if claim.above_threshold)

    @property
    def max_p_contradiction(self) -> float:
        """The highest claim-level contradiction probability of the summary."""
        return self.metrics.max_contradiction


@contextmanager
def read_only_session(settings: AppSettings | None = None) -> Iterator[Session]:
    """A session over the application database opened **read-only**.

    SQLite ``mode=ro`` (a URI filename) makes the read-only property the database's
    own, not a convention of this module: a write through this session fails with
    ``attempt to write a readonly database``. The engine of
    :mod:`diary.db.session` is deliberately *not* reused here — it applies
    ``PRAGMA journal_mode=WAL`` on connect, which is a write, and the audit must not
    need write access to the file at all.

    A missing database file is a clear error instead of a driver-level traceback.

    One honest detail for a reviewer comparing hashes: opening a WAL-mode database —
    even read-only — may materialise or touch SQLite's own side-car files (``-shm``
    and an empty ``-wal``) next to it. They are coordination artefacts of *opening*,
    not data: the database file itself is never written, and a WAL that carries a
    frame would mean a write happened (the test asserts the ``-wal`` stays empty).
    """
    resolved = settings if settings is not None else get_settings()
    path = default_database_path(resolved)
    if not path.is_file():
        raise NliAuditError(
            f"the application database {path} does not exist: create and migrate it "
            "with `alembic upgrade head` (under the same APP_DATA_ROOT) first"
        )
    engine = _open_read_only_engine(path)
    try:
        yield sessionmaker(bind=engine, class_=Session, expire_on_commit=False, future=True)()
    finally:
        engine.dispose()


def _open_read_only_engine(path: Path) -> Engine:
    """An engine over ``path`` with SQLite URI ``mode=ro`` (no pragmas, no writes)."""
    url = f"sqlite:///file:{path.as_posix()}?mode=ro&uri=true"
    return _sa_create_engine(url, future=True)


def select_daily_artifact(session: Session, *, local_date: date | None = None) -> Artifact:
    """The stored day summary to audit: the newest one, or the one of ``local_date``.

    Both branches take the newest artifact of the (day, layer) lineage in the store's
    own deterministic order (:func:`~diary.db.results.list_layer_results`: creation
    time, then version, then id) — «the newest saved artifact» of the task's definition
    of done.

    With ``local_date`` the lineage is selected by the **local day alone**: the timezone
    is deliberately *not* part of the lookup. A ``daily_summary`` artifact carries the
    ``local_tz`` of the run that wrote it (an explicit ``--tz`` of that day, or the
    ``APP_TIMEZONE`` in force at the time; T30), and a day recorded under another
    timezone is still that day's summary. Selecting on the pair
    ``(local_date, APP_TIMEZONE)`` would answer «no stored daily summary for <date>»
    about a day that *is* stored — the very error that looks like a normal state. When
    several timezones carry the same date, the newest artifact of the day wins (the
    store order above); the timezone of the artifact that was found is printed by the
    report itself (``tz=…``), so nothing about the day is hidden.

    Both empty cases raise :class:`NliAuditNoArtifactError` naming the day and the
    command that produces one — a report over nothing would read as «everything is
    fine».
    """
    if local_date is not None:
        rows = list_layer_results(session, layer=ArtifactLayer.daily_summary, local_date=local_date)
        if not rows:
            raise NliAuditNoArtifactError(
                f"no stored daily summary for {local_date.isoformat()}: run that day first "
                f"with `{_RUN_A_DAY}` (or drop --date to audit the newest stored day)"
            )
        return rows[-1]
    rows = list_layer_results(session, layer=ArtifactLayer.daily_summary)
    if not rows:
        raise NliAuditNoArtifactError(
            f"no stored daily summary to check: run a day first with `{_RUN_A_DAY}`"
        )
    return rows[-1]


def _stored_day_anchor(artifact: Artifact) -> tuple[date, str]:
    """The ``(local_date, local_tz)`` anchor of a stored day summary, validated.

    The schema guarantees both for a ``daily_summary`` row
    (``ck_artifacts_anchor_family`` + ``ck_artifacts_anchor_local_tz``); the check is
    here anyway, because a day without its timezone or date cannot be reconstructed and
    «cannot be reconstructed» must be a loud refusal rather than a wrong report.
    """
    if artifact.local_date is None or not artifact.local_tz:
        raise NliAuditError(
            f"{_SOURCE_UNAVAILABLE}: the stored anchor carries no local date/timezone"
        )
    return artifact.local_date, artifact.local_tz


def day_source_texts(session: Session, artifact: Artifact) -> list[str]:
    """The source texts of ``artifact``, rebuilt exactly as its layer selected them.

    The anchor comes from the artifact itself — ``local_date`` + ``local_tz`` — and the
    selection is :func:`~diary.domain.pipelines.sum_daily.day_input.collect_day_entries`,
    the one the layer uses. Nothing is invented: an artifact whose anchor or structured
    result cannot be read is a loud
    :class:`NliAuditError`, and a day whose messages are gone is one too (the
    reconstruction would otherwise be silently empty).
    """
    local_date, local_tz = _stored_day_anchor(artifact)
    try:
        timezone = ZoneInfo(local_tz)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise NliAuditError(
            f"{_SOURCE_UNAVAILABLE}: the stored timezone {local_tz!r} is not known"
        ) from exc
    try:
        entries = collect_day_entries(session, local_date=local_date, tz=timezone)
    except DayInputError as exc:
        # A static, non-sensitive reason (AGENTS.md §8) — the day cannot be mapped onto
        # raw_entry[] without inventing something (a voice message with no audio row).
        raise NliAuditError(f"{_SOURCE_UNAVAILABLE}: {exc}") from exc
    return [entry.text for entry in entries]


def audit_stored_day(
    session: Session,
    *,
    local_date: date | None = None,
    judge: NliJudge | None = None,
) -> NliAuditReport:
    """Judge the ``facts`` of a stored day summary against the day it was written from.

    The rule, the scope and the premise are the runtime ones: ``facts`` of the
    ``daily_summary`` contract against the texts of the day's entries
    (:data:`GUARDED_CONTRACT`, :func:`~diary.domain.pipelines.nli_guard.default_nli_judge`,
    ``premise_from_texts``). Label fields are judged and reported as well — the harness
    judges them the same way — but the rule never condemns them (a theme is not an
    assertion, :data:`~diary.domain.contracts.nli.PROPOSITIONAL_FIELDS`), and the report
    keeps the rule's verdict apart from the raw threshold comparison. The instrument is
    proved **before** any claim is judged, so a missing extra or weights fails the
    command instead of producing a partial report.

    ``judge`` overrides the instrument (tests install a scripted judge); ``None`` means
    the pinned ONNX cross-encoder of the settings, which is what the CLI passes.
    """
    artifact = select_daily_artifact(session, local_date=local_date)
    payload = artifact.result_json
    if payload is None:
        raise NliAuditError(
            f"{_SOURCE_UNAVAILABLE}: the artifact stores no structured result to judge"
        )
    day, local_tz = _stored_day_anchor(artifact)
    source_texts = day_source_texts(session, artifact)
    resolved = judge if judge is not None else default_nli_judge()
    try:
        load = getattr(resolved, "load", None)
        if callable(load):
            # Prove the instrument now, so «no weights» is a property of the command
            # rather than of the first claim that happens to be judged.
            load()
        premise = premise_from_texts(source_texts)
        claims = claims_for(GUARDED_CONTRACT, payload)
        judgements = judge_claims(premise, claims, resolved)
    except NliUnavailableError as exc:
        raise NliAuditUnavailableError(_INSTRUMENT_UNAVAILABLE) from exc
    except NliPremiseError as exc:
        raise NliAuditError(
            f"{_SOURCE_UNAVAILABLE}: the day has no source text left to compare against"
        ) from exc
    except NliError as exc:
        # A judge answer the rules cannot use is a broken instrument, not a verdict.
        raise NliAuditUnavailableError(_INSTRUMENT_UNAVAILABLE) from exc

    return NliAuditReport(
        artifact_id=artifact.id,
        version=artifact.version,
        local_date=day,
        local_tz=local_tz,
        processing_run_id=artifact.processing_run_id,
        entries=len(source_texts),
        premise_sentences=len(premise_sentences(premise)),
        claims=tuple(
            ClaimAudit(
                coordinate=judgement.claim.coordinate,
                propositional=judgement.claim.is_propositional,
                p_contradiction=judgement.contradiction_probability,
                verdict=judgement.label,
                flagged=judgement.contradicts,
                above_threshold=judgement.contradiction_probability >= CONTRADICTION_THRESHOLD,
                governing_sentence=judgement.governing_sentence,
            )
            for judgement in judgements
        ),
        metrics=metrics_for(judgements),
    )


def format_report(report: NliAuditReport) -> list[str]:
    """Render the report as `key=value` lines: identifiers, counts, no diary text.

    Built from the report's own numbers only, so it cannot leak a claim even by
    accident (AGENTS.md §7-8). One line per claim carries its **coordinate**
    (``facts#2``), the probability the rule compares and the verdict; the totals line
    closes the measurement. Probabilities are printed to three decimals — enough to see
    a claim approaching the threshold, and always a real number.
    """
    run = str(report.processing_run_id) if report.processing_run_id else "none"
    version = str(report.version) if report.version is not None else "none"
    lines = [
        "nli audit: stored daily summary, report only (nothing is stored, nothing is changed)",
        (
            f"artifact id={report.artifact_id} layer={GUARDED_CONTRACT} "
            f"date={report.local_date.isoformat()} tz={report.local_tz} version={version} "
            f"run={run}"
        ),
        f"source entries={report.entries} premise_sentences={report.premise_sentences}",
        (
            "rule no_contradiction_with_source "
            f"threshold_contradiction={CONTRADICTION_THRESHOLD:.3f} scope=facts"
        ),
    ]
    for claim in report.claims:
        lines.append(
            f"claim coordinate={claim.coordinate} "
            f"propositional={'yes' if claim.propositional else 'no'} "
            f"p_contradiction={claim.p_contradiction:.3f} verdict={claim.verdict} "
            f"flagged={'yes' if claim.flagged else 'no'} "
            f"above_threshold={'yes' if claim.above_threshold else 'no'} "
            f"source_sentence={claim.governing_sentence}"
        )
    lines.append(
        f"totals claims={len(report.claims)} "
        f"propositional={sum(1 for claim in report.claims if claim.propositional)} "
        f"contradicting={report.contradicting} above_threshold={report.above_threshold} "
        f"unsupported={len(report.metrics.unsupported_coordinates)} "
        f"max_p_contradiction={report.max_p_contradiction:.3f} "
        f"min_entailment={report.metrics.min_entailment:.3f}"
    )
    lines.append(
        "nli audit: the weekly layer is not covered (T26/ADR-015: the week is outside "
        "this rule); the artifact and the database are untouched"
    )
    return lines
