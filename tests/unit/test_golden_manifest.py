"""Unit tests for the golden eval set (T00).

The set is eleven synthetic Russian chunks in ``eval/golden``, loaded through
``diary.domain.contracts.golden`` (the single definition of the format). The
tests verify the mix and the manifest thresholds, the per-chunk budget
bookkeeping (``chunk_chars`` matches the serialized input and the context rule
holds), the self-consistency of every chunk (dates, ISO weeks, category →
contract) and the privacy invariant: the golden text is authored synthetic
content and does not overlap the private data directory. The privacy check
skips when ``data/`` is absent, so the suite stays hermetic in CI.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest

from diary.core.settings import PROJECT_ROOT
from diary.domain.contracts.golden import (
    ALLOWED_ORIGINS,
    DEFAULT_FORBIDDEN_LEXEMES,
    DEFAULT_NUM_CTX,
    GOLDEN_DIR,
    WEEKLY_SPLIT_NUM_CTX,
    GoldenChunk,
    canonical_input_json,
    chunk_path,
    estimate_tokens,
    input_char_count,
    input_payload,
    load_chunk,
    load_manifest,
    num_ctx_for,
)
from diary.domain.contracts.raw_entry import RawEntry
from diary.domain.contracts.weekly_summary import WeeklySummary

MANIFEST = load_manifest()
CHUNK_IDS = tuple(MANIFEST.chunks)

#: Suffixes searched for golden/private text overlap.
_TEXT_SUFFIXES = frozenset({".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".log"})
_PRIVATE_SUFFIXES = _TEXT_SUFFIXES | {".db"}

#: Only phrases of this length (with a space) are compared: identifiers, dates
#: and paths are not content and would only produce noise.
_MIN_PHRASE_LEN = 20

#: Files larger than this are not scanned (the private DB stays well below).
_MAX_SCAN_BYTES = 16 * 1024 * 1024


@pytest.fixture(scope="module")
def chunks() -> dict[str, GoldenChunk]:
    return {chunk_id: load_chunk(chunk_path(chunk_id)) for chunk_id in CHUNK_IDS}


def _string_leaves(payload: object) -> Iterator[str]:
    if isinstance(payload, str):
        yield payload
    elif isinstance(payload, dict):
        for value in payload.values():
            yield from _string_leaves(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _string_leaves(item)


def _golden_phrases(chunks: dict[str, GoldenChunk]) -> set[str]:
    phrases: set[str] = set()
    for chunk in chunks.values():
        for value in _string_leaves(input_payload(chunk.input)):
            if len(value) >= _MIN_PHRASE_LEN and " " in value:
                phrases.add(value)
    return phrases


def _private_files(data_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix not in _PRIVATE_SUFFIXES:
            continue
        try:
            if path.stat().st_size > _MAX_SCAN_BYTES:
                continue
        except OSError:  # pragma: no cover - unreadable file is not our topic
            continue
        files.append(path)
    return files


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def test_manifest_identity() -> None:
    assert MANIFEST.version == 1
    assert MANIFEST.lang == "ru"
    assert MANIFEST.synthetic is True
    # T35 added the long day chunk (the split fixture) without touching the
    # threshold: the number of chunks that must stay clean is a decision of T00,
    # not something a new fixture may raise or lower.
    assert MANIFEST.threshold.pass_count == 8
    assert MANIFEST.threshold.total == 11
    assert MANIFEST.budget_default.num_ctx == DEFAULT_NUM_CTX
    assert MANIFEST.latency_budget_s.stt_fix == 5
    assert MANIFEST.latency_budget_s.daily == 15
    # T21: the weekly rollup runs on the CPU twin, so it carries its own budget.
    assert MANIFEST.latency_budget_s.weekly == 180


def test_manifest_declares_eleven_unique_chunks_with_the_required_mix() -> None:
    assert len(CHUNK_IDS) == 11
    assert len(set(CHUNK_IDS)) == 11
    assert MANIFEST.counts.model_dump() == {
        "stt_error": 3,
        "day": 4,
        "week": 2,
        "wellbeing": 2,
    }
    assert GOLDEN_DIR.name == "golden"
    assert GOLDEN_DIR.parent.name == "eval"


def test_golden_files_are_deterministically_formatted() -> None:
    paths = [GOLDEN_DIR / "manifest.json", *(chunk_path(chunk_id) for chunk_id in CHUNK_IDS)]
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        assert raw.endswith("\n"), f"{path.name} must end with a newline"
        assert (
            raw == json.dumps(json.loads(raw), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ), f"{path.name} is not in the canonical golden format"


# ---------------------------------------------------------------------------
# chunk files and mix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_id", CHUNK_IDS)
def test_chunk_file_exists_and_matches_its_id(
    chunk_id: str, chunks: dict[str, GoldenChunk]
) -> None:
    path = chunk_path(chunk_id)
    assert path.is_file()
    assert path.parent.name == "chunks"
    assert chunks[chunk_id].id == chunk_id


def test_actual_categories_match_the_declared_counts(chunks: dict[str, GoldenChunk]) -> None:
    counted = Counter(chunk.category for chunk in chunks.values())
    assert counted == Counter(MANIFEST.counts.model_dump())


def test_contracts_are_the_expected_ones(chunks: dict[str, GoldenChunk]) -> None:
    counted = Counter(chunk.contract for chunk in chunks.values())
    expected = Counter(
        {"stt_corrected": 3, "daily_summary": 4, "weekly_summary": 2, "coach_note": 2}
    )
    assert counted == expected


@pytest.mark.parametrize("chunk_id", CHUNK_IDS)
def test_chunk_is_synthetic_russian_and_authored(
    chunk_id: str, chunks: dict[str, GoldenChunk]
) -> None:
    chunk = chunks[chunk_id]
    assert chunk.lang == "ru"
    assert chunk.synthetic is True
    assert chunk.origin in ALLOWED_ORIGINS
    assert chunk.invariants.kind == chunk.contract
    assert chunk.invariants.forbidden_lexemes == list(DEFAULT_FORBIDDEN_LEXEMES)


# ---------------------------------------------------------------------------
# context budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_id", CHUNK_IDS)
def test_chunk_budget_matches_the_measured_input(
    chunk_id: str, chunks: dict[str, GoldenChunk]
) -> None:
    chunk = chunks[chunk_id]
    measured_chars = input_char_count(chunk)
    assert chunk.budget.chunk_chars == measured_chars
    assert chunk.budget.chunk_tokens_est == estimate_tokens(measured_chars)
    assert chunk.budget.num_ctx == num_ctx_for(chunk.category, chunk.budget.chunks_planned)
    assert chunk.budget.num_ctx in {DEFAULT_NUM_CTX, WEEKLY_SPLIT_NUM_CTX}
    assert chunk.budget.chunk_tokens_est <= chunk.budget.num_ctx
    assert len(json.dumps(json.loads(canonical_input_json(chunk.input)), ensure_ascii=False)) > 0


def test_split_weekly_chunks_get_the_larger_context() -> None:
    assert num_ctx_for("week", 1) == DEFAULT_NUM_CTX
    assert num_ctx_for("week", 2) == WEEKLY_SPLIT_NUM_CTX
    assert num_ctx_for("stt_error", 3) == DEFAULT_NUM_CTX
    assert num_ctx_for("day", 2) == DEFAULT_NUM_CTX
    assert num_ctx_for("wellbeing", 2) == DEFAULT_NUM_CTX


def test_only_the_long_day_declares_a_split_and_it_keeps_the_default_window(
    chunks: dict[str, GoldenChunk],
) -> None:
    """T35: a day that does not fit is split by entries, not given a bigger window.

    ``num_ctx_for`` raises the window only for a **weekly** chunk fed in pieces; the
    day layer divides a long day into parts inside the same 8192 (``ADR-016``). So the
    one chunk that declares several pieces is the long day, and every chunk of the set
    still lives in the default window — a split is not a licence to widen it.
    """
    split = [chunk for chunk in chunks.values() if chunk.budget.chunks_planned > 1]
    assert [chunk.id for chunk in split] == ["day-04-long-wednesday"]
    assert split[0].category == "day"
    for chunk in chunks.values():
        assert chunk.budget.num_ctx == DEFAULT_NUM_CTX, chunk.id
        assert chunk.budget.chunk_tokens_est <= chunk.budget.num_ctx, chunk.id


# ---------------------------------------------------------------------------
# per-category self-consistency
# ---------------------------------------------------------------------------


def test_stt_chunks_carry_raw_unmarked_text(chunks: dict[str, GoldenChunk]) -> None:
    for chunk in chunks.values():
        if chunk.category != "stt_error":
            continue
        assert len(chunk.input) == 1
        entry = chunk.input[0]
        assert isinstance(entry, RawEntry)
        assert entry.source == "voice"
        assert entry.audio_path is not None
        assert entry.text.strip()
        # Raw STT: no sentence casing and no doubt marker — restoring
        # punctuation/casing and filling uncertain[] is the task itself.
        assert not any(character.isupper() for character in entry.text)
        assert entry.text[-1] not in ".!?"
        assert "[?]" not in entry.text


def test_day_chunks_match_their_declared_date(chunks: dict[str, GoldenChunk]) -> None:
    for chunk in chunks.values():
        if chunk.category != "day":
            continue
        expected = chunk.invariants.expected_date
        assert chunk.invariants.expected_energy in {"low", "medium", "high", None}
        for entry in chunk.input:
            assert isinstance(entry, RawEntry)
            assert entry.ts.date() == expected


def test_week_chunks_match_their_declared_iso_week(chunks: dict[str, GoldenChunk]) -> None:
    for chunk in chunks.values():
        if chunk.category != "week":
            continue
        expected = chunk.invariants.expected_week_id
        for item in chunk.input:
            iso = item.date.isocalendar()
            assert f"{iso.year}-W{iso.week:02d}" == expected


def test_coach_chunks_roll_up_a_declared_week(chunks: dict[str, GoldenChunk]) -> None:
    weeks = {
        chunk.invariants.expected_week_id: chunk
        for chunk in chunks.values()
        if chunk.category == "week"
    }
    for chunk in chunks.values():
        if chunk.category != "wellbeing":
            continue
        assert isinstance(chunk.input, WeeklySummary)
        assert chunk.input.week_id in weeks
        assert chunk.input.rollup
        assert chunk.input.daily_ids == [
            item.date.isoformat() for item in weeks[chunk.input.week_id].input
        ]


# ---------------------------------------------------------------------------
# authoring rules (blacklist, absence of a doubt marker)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_id", CHUNK_IDS)
def test_inputs_avoid_the_forbidden_lexemes(chunk_id: str, chunks: dict[str, GoldenChunk]) -> None:
    serialized = canonical_input_json(chunks[chunk_id].input).lower()
    for lexeme in DEFAULT_FORBIDDEN_LEXEMES:
        assert lexeme not in serialized


@pytest.mark.parametrize("chunk_id", CHUNK_IDS)
def test_inputs_use_no_question_mark_doubt_marker(
    chunk_id: str, chunks: dict[str, GoldenChunk]
) -> None:
    assert "[?]" not in canonical_input_json(chunks[chunk_id].input)


# ---------------------------------------------------------------------------
# privacy: the golden text is synthetic and disjoint from the private data
# ---------------------------------------------------------------------------


def test_golden_text_does_not_overlap_private_data(chunks: dict[str, GoldenChunk]) -> None:
    data_root = PROJECT_ROOT / "data"
    if not data_root.is_dir():
        pytest.skip("private data directory is absent; nothing to compare against")
    files = _private_files(data_root)
    if not files:
        pytest.skip("private data directory holds no scanned file")

    phrases = _golden_phrases(chunks)
    assert phrases, "golden chunks must contain authored phrases to compare"

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        overlap = [phrase for phrase in phrases if phrase in text]
        assert not overlap, (
            f"{path.relative_to(data_root)} reproduces {len(overlap)} authored "
            "golden phrase(s); the golden set must stay purely synthetic"
        )
        if path.suffix not in _TEXT_SUFFIXES:
            continue
        for line in text.splitlines():
            candidate = line.strip()
            if len(candidate) < _MIN_PHRASE_LEN or " " not in candidate:
                continue
            assert not any(candidate in phrase for phrase in phrases), (
                f"a private line of {path.relative_to(data_root)} appears inside "
                "a golden chunk; the golden set must stay purely synthetic"
            )
