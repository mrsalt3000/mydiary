"""Integration tests for the owner-facing NLI audit of a stored day summary (T27).

The command is an instrument for the owner of the data, **not a gate**: it takes a stored
``daily_summary`` artifact, rebuilds its source the way the layer did, judges the claims
with the runtime rules and prints a report without text. These tests pin, on an isolated
migration-created database and with synthetic content only (AGENTS.md §13):

* the shape and the counters of the report, and that it carries **no** diary text;
* the artifact selection — the newest stored day by default, and ``--date`` for one day,
  selected by the local day **alone**: a day recorded under another timezone (an explicit
  ``--tz`` of that run, or a changed ``APP_TIMEZONE``) is still that day's summary, and
  with several timezones on one date the newest artifact of the day wins;
* the two refusal paths of the definition of done: no stored summary (with the hint that
  names `scripts/run-local.sh cli day --date …`) and an unavailable instrument
  (fail-closed, the policy of T24/T24b);
* that the source is reconstructed from the database and never replaced by «something
  similar» — a day whose entries are gone is a refusal, not a report over a guess, and so
  is a stored anchor that names an unknown timezone or carries no date/timezone at all;
* that the database is opened **read-only**: the whole data root is byte-identical after a
  CLI run, and a write through the read-only session is refused by SQLite itself.

No model, endpoint, GPU, configuration file or real diary content is touched: the judge is
a scripted double (the pattern of the NLI tests) and the weights are never loaded.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable, Sequence
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from diary.cli import main
from diary.core.settings import AppSettings, get_settings
from diary.db.models import Artifact, ArtifactLayer, MessageKind
from diary.db.repositories import MessageRepository
from diary.db.results import save_layer_result
from diary.db.session import create_engine, create_session_factory
from diary.domain.contracts import DailySummary
from diary.domain.nli.onnx_judge import NliUnavailableError
from diary.domain.pipelines import nli_audit
from diary.domain.pipelines.nli_audit import (
    NliAuditError,
    NliAuditNoArtifactError,
    NliAuditUnavailableError,
    audit_stored_day,
    day_source_texts,
    format_report,
    read_only_session,
    select_daily_artifact,
)

#: The synthetic local day (and a second one, for the selection tests).
DAY = date(2026, 9, 14)
OTHER_DAY = date(2026, 9, 15)
LOCAL_TZ = "Europe/Moscow"

#: Synthetic diary content. Every string here must be absent from any report.
MORNING_TEXT = "Утром сходил в поликлинику за справкой для бассейна."
EVENING_TEXT = "Вечером починил велосипед и посмотрел фильм."
FACT_FAITHFUL = "Сходил в поликлинику за справкой."
FACT_CONTRADICTING = "В поликлинику не ходил и справки не брал."
THEME_TEXT = "поликлиника"

#: Every synthetic string that must never reach the report or an error message.
SECRET_TEXT: tuple[str, ...] = (
    MORNING_TEXT,
    EVENING_TEXT,
    FACT_FAITHFUL,
    FACT_CONTRADICTING,
    THEME_TEXT,
)

#: Judge answers (entailment, contradiction, neutral) keyed by a marker in the claim.
_CONTRADICTION_RULE = (0.02, 0.95, 0.03)
_SUPPORT_RULE = (0.97, 0.01, 0.02)
#: A nominal label read as a contradiction: above the threshold, still not an assertion.
_LABEL_RULE = (0.04, 0.90, 0.06)
_DEFAULT_RULE = (0.97, 0.01, 0.02)


class _ScriptedJudge:
    """A judge answering by the claim it is asked about (first matching marker wins)."""

    def __init__(self, rules: Sequence[tuple[str, tuple[float, float, float]]]) -> None:
        self.rules = list(rules)
        self.pairs: list[tuple[str, str]] = []

    def load(self) -> None:
        """The audit proves the instrument before judging; the double needs nothing."""

    def __call__(self, pairs: Iterable[tuple[str, str]]) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for _premise, hypothesis in pairs:
            self.pairs.append((_premise, hypothesis))
            answer = next(
                (rule for marker, rule in self.rules if marker in hypothesis), _DEFAULT_RULE
            )
            rows.append(
                {
                    "entailment": answer[0],
                    "contradiction": answer[1],
                    "neutral": answer[2],
                }
            )
        return rows


def _judge() -> _ScriptedJudge:
    """The double of the audit tests: one contradiction, one support, one label."""
    return _ScriptedJudge(
        [
            ("не ходил", _CONTRADICTION_RULE),
            ("Сходил в поликлинику", _SUPPORT_RULE),
            (THEME_TEXT, _LABEL_RULE),
        ]
    )


class _Unavailable:
    """An instrument that cannot be built — the fail-closed path of T24/T24b."""

    def load(self) -> None:
        raise NliUnavailableError("/nowhere/model.onnx is missing")


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _add_message(session: Session, *, body: str, created_at: datetime) -> uuid.UUID:
    return (
        MessageRepository(session)
        .create(kind=MessageKind.text, body_text=body, created_at=created_at)
        .id
    )


def _store_summary(
    session: Session,
    *,
    local_date: date,
    created_at: datetime,
    facts: Sequence[str] | None = None,
    themes: Sequence[str] | None = None,
    local_tz: str = LOCAL_TZ,
) -> Artifact:
    """Append a day summary through the production store (no LLM, no provider)."""
    summary = DailySummary(
        date=local_date,
        facts=list(facts) if facts is not None else [FACT_FAITHFUL],
        themes=list(themes) if themes is not None else [],
        energy=None,
        open_questions=[],
    )
    return save_layer_result(
        session,
        layer=ArtifactLayer.daily_summary,
        local_date=local_date,
        local_tz=local_tz,
        result=summary,
        created_at=created_at,
    )


def _seed_day(session: Session) -> Artifact:
    """One synthetic day with two entries and one stored summary (two facts + a label)."""
    _add_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-14T04:10:00"))
    _add_message(session, body=EVENING_TEXT, created_at=_utc("2026-09-14T16:45:00"))
    return _store_summary(
        session,
        local_date=DAY,
        created_at=_utc("2026-09-14T18:00:00"),
        facts=[FACT_CONTRADICTING, FACT_FAITHFUL],
        themes=[THEME_TEXT],
    )


def _assert_no_text(rendered: str) -> None:
    """The report (or a message) carries no synthetic diary content at all."""
    for secret in SECRET_TEXT:
        assert secret not in rendered


# ---------------------------------------------------------------------------
# the report of a stored day summary
# ---------------------------------------------------------------------------


def test_the_report_carries_coordinates_counters_and_verdicts(session: Session) -> None:
    artifact = _seed_day(session)

    report = audit_stored_day(session, judge=_judge())
    rendered = "\n".join(format_report(report))

    assert report.artifact_id == artifact.id
    assert report.version == 1
    assert report.local_date == DAY
    assert report.local_tz == LOCAL_TZ
    assert report.entries == 2  # the day's two entries, not the summary's two facts
    assert report.premise_sentences == 2

    # One entry per claim of the contract, in field order: facts then themes.
    assert [claim.coordinate for claim in report.claims] == ["facts#1", "facts#2", "themes#1"]
    contradicted, supported, label = report.claims
    assert (contradicted.propositional, contradicted.verdict) == (True, "contradiction")
    assert contradicted.flagged and contradicted.above_threshold
    assert contradicted.p_contradiction == pytest.approx(0.95)
    assert (supported.propositional, supported.verdict) == (True, "entailment")
    assert not supported.flagged and not supported.above_threshold
    # A label is judged and may sit above the threshold, but it never condemns (T24a).
    assert (label.propositional, label.verdict) == (False, "neutral")
    assert not label.flagged and label.above_threshold

    assert report.contradicting == 1
    assert report.above_threshold == 2
    assert report.max_p_contradiction == pytest.approx(0.95)

    assert f"artifact id={artifact.id} layer=daily_summary date=2026-09-14" in rendered
    assert "tz=Europe/Moscow version=1" in rendered
    assert "source entries=2 premise_sentences=2" in rendered
    assert (
        "claim coordinate=facts#1 propositional=yes p_contradiction=0.950 "
        "verdict=contradiction flagged=yes above_threshold=yes" in rendered
    )
    assert (
        "claim coordinate=facts#2 propositional=yes p_contradiction=0.000 "
        "verdict=entailment flagged=no above_threshold=no" in rendered
    )
    assert (
        "claim coordinate=themes#1 propositional=no p_contradiction=0.900 "
        "verdict=neutral flagged=no above_threshold=yes" in rendered
    )
    assert (
        "totals claims=3 propositional=2 contradicting=1 above_threshold=2 unsupported=1 "
        "max_p_contradiction=0.950 min_entailment=0.020" in rendered
    )
    _assert_no_text(rendered)
    # The report says what it is: an instrument, and the week is out of its scope.
    assert "report only" in rendered
    assert "T26/ADR-015" in rendered


def test_the_report_prints_no_text_of_the_entries_or_the_summary(session: Session) -> None:
    """The one property that must hold for every future edit: no diary text, ever."""
    _seed_day(session)
    judge = _judge()

    report = audit_stored_day(session, judge=judge)
    rendered = "\n".join(format_report(report))

    _assert_no_text(rendered)
    # The claim texts did reach the judge (that is how an entailment is computed) — the
    # judge is the only place they may go.
    hypotheses = {hypothesis for _premise, hypothesis in judge.pairs}
    assert FACT_CONTRADICTING in hypotheses
    assert FACT_FAITHFUL in hypotheses


def test_the_judge_sees_the_whole_day_as_the_premise(session: Session) -> None:
    """The premise is every entry of the day, exactly as the layer's own source is."""
    _seed_day(session)
    judge = _judge()

    audit_stored_day(session, judge=judge)

    premises = {premise for premise, _hypothesis in judge.pairs}
    assert premises == {MORNING_TEXT, EVENING_TEXT}


def test_the_default_judge_is_resolved_through_the_module_hook(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an explicit judge the audit uses the runtime hook (patched in tests)."""
    _seed_day(session)
    judge = _judge()
    monkeypatch.setattr(nli_audit, "default_nli_judge", lambda settings=None: judge)

    report = audit_stored_day(session)

    assert report.contradicting == 1
    assert judge.pairs


# ---------------------------------------------------------------------------
# choosing the artifact
# ---------------------------------------------------------------------------


def test_the_newest_stored_day_is_audited_by_default(session: Session) -> None:
    _add_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-14T04:10:00"))
    _add_message(session, body=EVENING_TEXT, created_at=_utc("2026-09-15T04:10:00"))
    first = _store_summary(session, local_date=DAY, created_at=_utc("2026-09-14T18:00:00"))
    second = _store_summary(session, local_date=DAY, created_at=_utc("2026-09-14T19:00:00"))
    newest = _store_summary(session, local_date=OTHER_DAY, created_at=_utc("2026-09-15T18:00:00"))

    assert first.version == 1 and second.version == 2
    assert select_daily_artifact(session).id == newest.id
    # The day selector takes the newest version of that day's lineage, not the oldest.
    assert select_daily_artifact(session, local_date=DAY).id == second.id


def test_the_date_selector_ignores_the_timezone_the_day_was_recorded_in(
    session: Session,
) -> None:
    """``--date`` is the local day, not the pair (local day, ``APP_TIMEZONE``).

    A ``daily_summary`` row carries the timezone of the run that wrote it, so a day
    recorded under another ``--tz`` (or under a different ``APP_TIMEZONE``) is still that
    day's summary. Selecting on the pair ``(local_date, APP_TIMEZONE)`` would answer «no
    stored daily summary for <date>» about a day that *is* stored — an error that looks
    like a normal state. With several timezones on one date the newest artifact of the
    day's lineage wins, by the store's own order (``created_at``, ``version``, ``id``).
    """
    _add_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-14T04:10:00"))
    _add_message(session, body=EVENING_TEXT, created_at=_utc("2026-09-14T16:45:00"))
    # The same local day, recorded twice: once in the application timezone, once — later —
    # in another one. Both rows are legitimate summaries of 2026-09-14.
    in_app_tz = _store_summary(session, local_date=DAY, created_at=_utc("2026-09-14T17:00:00"))
    in_other_tz = _store_summary(
        session,
        local_date=DAY,
        local_tz="Europe/Berlin",
        created_at=_utc("2026-09-14T18:00:00"),
    )
    assert in_app_tz.local_tz == get_settings().timezone
    assert in_other_tz.local_tz != get_settings().timezone

    assert select_daily_artifact(session, local_date=DAY).id == in_other_tz.id

    report = audit_stored_day(session, local_date=DAY, judge=_judge())

    assert report.artifact_id == in_other_tz.id
    assert report.local_date == DAY
    assert report.local_tz == "Europe/Berlin"
    # The source is rebuilt in the timezone of the artifact, so the day is the same day.
    assert report.entries == 2


def test_a_day_without_a_stored_summary_is_a_clear_refusal(session: Session) -> None:
    _store_summary(session, local_date=DAY, created_at=_utc("2026-09-14T18:00:00"))

    with pytest.raises(NliAuditNoArtifactError) as error:
        select_daily_artifact(session, local_date=OTHER_DAY)

    message = str(error.value)
    assert OTHER_DAY.isoformat() in message
    assert "scripts/run-local.sh cli day --date" in message  # the hint of DoD 3
    _assert_no_text(message)


def test_an_empty_store_is_a_clear_refusal(session: Session) -> None:
    with pytest.raises(NliAuditNoArtifactError) as error:
        audit_stored_day(session, judge=_judge())

    message = str(error.value)
    assert "scripts/run-local.sh cli day --date" in message
    assert "no stored daily summary" in message
    _assert_no_text(message)


# ---------------------------------------------------------------------------
# refusals instead of guesses
# ---------------------------------------------------------------------------


def test_a_day_whose_entries_are_gone_is_refused_not_reconstructed(session: Session) -> None:
    """No entries left to compare against: the audit refuses instead of judging a guess."""
    artifact = _store_summary(session, local_date=DAY, created_at=_utc("2026-09-14T18:00:00"))

    assert day_source_texts(session, artifact) == []
    with pytest.raises(NliAuditError) as error:
        audit_stored_day(session, judge=_judge())

    assert "cannot be reconstructed" in str(error.value)
    _assert_no_text(str(error.value))


def _insert_day_summary_row(session: Session, *, local_date: str, local_tz: str | None) -> None:
    """Raw INSERT of a day-summary row the production write path cannot produce.

    ``save_layer_result`` always writes a non-empty ``local_tz`` for a day anchor (an
    empty one falls back to ``APP_TIMEZONE``), so a row with an empty timezone exists
    only if the database was written by something else. The schema's
    ``ck_artifacts_anchor_local_tz`` allows it (it pins date and timezone to be
    *together* NULL), which is exactly the row the guard defends against.

    ``created_at`` is bound as the naive-UTC text ``UTCDateTime`` stores (the pattern of
    ``test_layer_results.py``): a raw ``datetime`` reaches the driver unconverted.
    """
    session.execute(
        text(
            "INSERT INTO artifacts (id, layer, export_id, version, path, sha256, created_at,"
            " raw_id, local_date, local_tz, week_id, summary_id, result_json)"
            " VALUES (:id, 'daily_summary', NULL, 1, NULL, NULL, :created_at,"
            " NULL, :local_date, :local_tz, NULL, NULL, :result_json)"
        ),
        {
            "id": uuid.uuid4().hex,
            "created_at": "2026-09-14 18:00:00.000000",
            "local_date": local_date,
            "local_tz": local_tz,
            "result_json": json.dumps({"date": DAY.isoformat(), "facts": [FACT_FAITHFUL]}),
        },
    )
    session.flush()


def test_an_unknown_timezone_in_the_stored_anchor_is_a_loud_refusal(session: Session) -> None:
    """A day anchor naming a timezone nobody knows cannot be reconstructed — refuse.

    Nothing validates the timezone *name* at write time (the schema only requires a day
    anchor to have one), so this row is reachable through the production store. A silent
    fallback to UTC — or to ``APP_TIMEZONE`` — would produce a report over a day boundary
    the layer never used, i.e. a measurement of something else.
    """
    _add_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-14T04:10:00"))
    stored = _store_summary(
        session,
        local_date=DAY,
        local_tz="Mars/Phobos",
        created_at=_utc("2026-09-14T18:00:00"),
    )
    assert stored.local_tz == "Mars/Phobos"  # the store accepted the name

    with pytest.raises(NliAuditError) as error:
        audit_stored_day(session, judge=_judge())

    message = str(error.value)
    assert "Mars/Phobos" in message  # names what is wrong, not just that something is
    assert "is not known" in message
    assert "cannot be reconstructed" in message
    _assert_no_text(message)


def test_an_empty_timezone_in_a_stored_anchor_is_a_loud_refusal(session: Session) -> None:
    """A day anchor with no timezone refuses before the day is even looked at."""
    _add_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-14T04:10:00"))
    _insert_day_summary_row(session, local_date=DAY.isoformat(), local_tz="")

    with pytest.raises(NliAuditError) as error:
        audit_stored_day(session, judge=_judge())

    message = str(error.value)
    assert "anchor carries no local date/timezone" in message
    assert "cannot be reconstructed" in message
    _assert_no_text(message)


@pytest.mark.parametrize(
    ("local_date", "local_tz"),
    [
        (DAY, None),  # a date without its timezone
        (None, LOCAL_TZ),  # a timezone without its date
        (None, None),  # neither
    ],
    ids=["no timezone", "no local date", "neither"],
)
def test_a_stored_anchor_without_its_date_or_timezone_is_refused(
    session: Session, local_date: date | None, local_tz: str | None
) -> None:
    """The schema forbids these rows, so the guard is pinned on a detached one.

    ``ck_artifacts_anchor_family`` requires a ``daily_summary`` row to carry
    ``local_date`` and ``ck_artifacts_anchor_local_tz`` requires a date to come with its
    timezone, so no write path — not even a raw INSERT — can store these combinations.
    The branch is still exercised directly: «a day without its timezone cannot be
    reconstructed» must stay a loud refusal if a future schema or migration ever relaxed
    the constraint (and the check is deliberately reached before any database read).
    """
    artifact = Artifact(
        id=uuid.uuid4(),
        layer=ArtifactLayer.daily_summary,
        version=1,
        local_date=local_date,
        local_tz=local_tz,
        result_json={},
    )

    with pytest.raises(NliAuditError) as error:
        day_source_texts(session, artifact)

    message = str(error.value)
    assert "anchor carries no local date/timezone" in message
    assert "cannot be reconstructed" in message
    _assert_no_text(message)


def test_an_unavailable_instrument_is_fail_closed(session: Session) -> None:
    """The policy of T24/T24b: a requested check that cannot run is a loud failure."""
    _seed_day(session)

    with pytest.raises(NliAuditUnavailableError) as error:
        audit_stored_day(session, judge=_Unavailable())

    assert "instrument is unavailable" in str(error.value)
    _assert_no_text(str(error.value))


def test_the_instrument_is_proved_by_the_default_hook(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_day(session)
    monkeypatch.setattr(nli_audit, "default_nli_judge", lambda settings=None: _Unavailable())

    with pytest.raises(NliAuditUnavailableError):
        audit_stored_day(session)


# ---------------------------------------------------------------------------
# read-only by construction
# ---------------------------------------------------------------------------


def test_the_read_only_session_refuses_a_write(tmp_path: Path, run_upgrade) -> None:
    """The read-only property is SQLite's, not a convention of this module."""
    db_dir = tmp_path / "data" / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    run_upgrade(f"sqlite:///{(db_dir / 'diary.db').as_posix()}")
    settings = get_settings().model_copy(update={"db_dir": db_dir})

    with read_only_session(settings) as session:
        assert session.execute(text("SELECT 1")).scalar() == 1
        with pytest.raises(OperationalError):
            session.execute(text("CREATE TABLE t_probe (x INTEGER)"))
        session.rollback()


def test_a_missing_database_is_a_clear_refusal(tmp_path: Path) -> None:
    settings = get_settings().model_copy(update={"db_dir": tmp_path / "absent"})

    with pytest.raises(NliAuditError) as error, read_only_session(settings):
        pass  # the refusal happens before the session body runs

    assert "does not exist" in str(error.value)


# ---------------------------------------------------------------------------
# the CLI entry point: the command the owner runs
# ---------------------------------------------------------------------------


def _tree_digest(root: Path) -> dict[str, str]:
    """sha256 of every file under ``root`` — the database and anything beside it."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _assert_the_database_did_not_change(
    before: dict[str, str], after: dict[str, str], data_root: Path
) -> None:
    """The database and every file that existed before are byte-identical after the run.

    A read-only open of a WAL-mode database may leave SQLite's own side-car files behind
    (``-shm``/``-wal``) — coordination artefacts of *opening*, not data. They may appear
    only with the WAL empty: any frame in the WAL would mean a write happened, and the
    database file itself is compared byte by byte.
    """
    database = "db/diary.db"
    assert after[database] == before[database]
    assert all(after.get(name) == digest for name, digest in before.items())
    for name in set(after) - set(before):
        assert name.endswith(("-wal", "-shm")), name
        if name.endswith("-wal"):
            assert (data_root / name).stat().st_size == 0, "a WAL frame means a write"


@pytest.fixture
def data_root(tmp_path: Path, run_upgrade) -> AppSettings:
    """A private data root whose database exists only through the migrations."""
    data_dir = tmp_path / "data"
    db_dir = data_dir / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    run_upgrade(f"sqlite:///{(db_dir / 'diary.db').as_posix()}")
    return get_settings().model_copy(update={"data_root": data_dir, "db_dir": db_dir})


def _seed_through_the_store(settings: AppSettings) -> None:
    """Two synthetic days with their entries and their stored summaries (real writes)."""
    engine = create_engine(settings=settings)
    try:
        with create_session_factory(engine)() as session:
            _add_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-14T04:10:00"))
            _add_message(session, body=EVENING_TEXT, created_at=_utc("2026-09-15T04:10:00"))
            _store_summary(
                session,
                local_date=DAY,
                created_at=_utc("2026-09-14T18:00:00"),
                facts=[FACT_CONTRADICTING, FACT_FAITHFUL],
                themes=[THEME_TEXT],
            )
            _store_summary(
                session,
                local_date=OTHER_DAY,
                created_at=_utc("2026-09-15T18:00:00"),
                facts=[FACT_FAITHFUL],
            )
            session.commit()
    finally:
        engine.dispose()


@pytest.fixture
def cli_environment(data_root: AppSettings, monkeypatch: pytest.MonkeyPatch) -> AppSettings:
    """Point the CLI at the private data root and give it a scripted instrument."""
    monkeypatch.setattr(nli_audit, "get_settings", lambda: data_root)
    monkeypatch.setattr("diary.cli.default_nli_judge", _judge)
    return data_root


def _seed_one_day(settings: AppSettings, *, local_tz: str) -> None:
    """One synthetic day: its two entries and one stored summary recorded in ``local_tz``."""
    engine = create_engine(settings=settings)
    try:
        with create_session_factory(engine)() as session:
            _add_message(session, body=MORNING_TEXT, created_at=_utc("2026-09-14T04:10:00"))
            _add_message(session, body=EVENING_TEXT, created_at=_utc("2026-09-14T16:45:00"))
            _store_summary(
                session,
                local_date=DAY,
                local_tz=local_tz,
                created_at=_utc("2026-09-14T18:00:00"),
            )
            session.commit()
    finally:
        engine.dispose()


def test_the_command_reports_a_stored_day_without_touching_the_data_root(
    cli_environment: AppSettings, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_through_the_store(cli_environment)
    before = _tree_digest(cli_environment.data_root)

    exit_code = main(["nli"])

    captured = capsys.readouterr()
    assert exit_code == 0
    # The newest stored day is the one of OTHER_DAY (the later summary).
    assert f"date={OTHER_DAY.isoformat()}" in captured.out
    assert "totals claims=1" in captured.out
    assert "verdict=entailment" in captured.out
    _assert_no_text(captured.out)
    _assert_no_text(captured.err)

    # Read-only: the database (and every file that existed before) is byte-identical.
    _assert_the_database_did_not_change(
        before, _tree_digest(cli_environment.data_root), cli_environment.data_root
    )


def test_the_command_date_flag_audits_that_day(
    cli_environment: AppSettings, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_through_the_store(cli_environment)

    exit_code = main(["nli", "--date", DAY.isoformat()])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"date={DAY.isoformat()}" in captured.out
    assert f"date={OTHER_DAY.isoformat()}" not in captured.out
    assert "totals claims=3 propositional=2 contradicting=1" in captured.out
    _assert_no_text(captured.out)


def test_the_command_date_flag_finds_a_day_recorded_in_another_timezone(
    cli_environment: AppSettings, capsys: pytest.CaptureFixture[str]
) -> None:
    """A day run with an explicit ``--tz`` other than ``APP_TIMEZONE`` is not lost.

    The report must not answer «no stored daily summary» about a day that is stored: the
    selector is the local day, and the timezone of the artifact found is printed by the
    report itself (``tz=…``), so the owner sees which boundary the summary was built for.
    """
    assert get_settings().timezone == LOCAL_TZ
    _seed_one_day(cli_environment, local_tz="Europe/Berlin")

    exit_code = main(["nli", "--date", DAY.isoformat()])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"date={DAY.isoformat()}" in captured.out
    assert "tz=Europe/Berlin" in captured.out
    assert "source entries=2" in captured.out  # the day really was reconstructed
    assert "totals claims=1" in captured.out
    _assert_no_text(captured.out)


def test_the_command_date_flag_for_a_missing_day_keeps_the_hint(
    cli_environment: AppSettings, capsys: pytest.CaptureFixture[str]
) -> None:
    """A genuinely absent day is still «no stored daily summary … run that day first»."""
    _seed_through_the_store(cli_environment)  # DAY and OTHER_DAY exist, this one does not
    missing = date(2026, 9, 20)

    exit_code = main(["nli", "--date", missing.isoformat()])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""  # not an empty report
    assert f"no stored daily summary for {missing.isoformat()}" in captured.err
    assert "scripts/run-local.sh cli day --date" in captured.err


def test_the_command_without_a_stored_summary_exits_nonzero_with_a_hint(
    data_root: AppSettings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(nli_audit, "get_settings", lambda: data_root)
    monkeypatch.setattr("diary.cli.default_nli_judge", _judge)

    exit_code = main(["nli"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""  # not an empty report
    assert "no stored daily summary" in captured.err
    assert "scripts/run-local.sh cli day --date" in captured.err


def test_the_command_fails_closed_when_the_instrument_is_unavailable(
    data_root: AppSettings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_through_the_store(data_root)
    monkeypatch.setattr(nli_audit, "get_settings", lambda: data_root)
    monkeypatch.setattr("diary.cli.default_nli_judge", _Unavailable)

    exit_code = main(["nli"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "instrument is unavailable" in captured.err
    _assert_no_text(captured.err)


def test_the_command_rejects_a_malformed_date(
    data_root: AppSettings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(nli_audit, "get_settings", lambda: data_root)

    exit_code = main(["nli", "--date", "14.09.2026"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--date must be an ISO date" in captured.err


def test_the_command_never_needs_an_llm_configuration(
    cli_environment: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The audit is not a layer: it works with no model configuration at all.

    A day or week run resolves ``configs/models.yaml`` and fails without it; the report
    must not, because it never calls a model.
    """
    _seed_through_the_store(cli_environment)
    monkeypatch.setattr("diary.cli.load_models_config", _raise_models_config)

    assert main(["nli"]) == 0
    assert "totals claims=" in capsys.readouterr().out


def _raise_models_config(path: object = None) -> None:
    raise AssertionError("the audit must not read the model configuration")
