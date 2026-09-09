"""Unit tests for the summarization document helpers (Task 14).

The entry splitter, the deterministic character chunker and the partial-summary
merge are pure functions: no database, no settings and no LLM. Synthetic data
only (AGENTS.md §13).
"""

from __future__ import annotations

import pytest

from diary.domain.pipelines.summarize.pipeline import (
    chunk_entries_by_chars,
    merge_partials,
    split_normalized_entries,
)
from diary.domain.pipelines.summarize.schema import (
    SummaryResult,
    WellbeingSection,
)

_DOCUMENT = (
    "# Дневник — нормализованный\n"
    "Дата воскресенья: 2026-09-13\n"
    "Номер выгрузки: 0001\n"
    "---\n"
    "\n"
    "[2026-09-07 09:00] TEXT\n"
    "Первая запись.\n"
    "\n"
    "[2026-09-07 20:00] TEXT\n"
    "Вторая запись\n"
    "с переносом строки.\n"
)


def _result(
    *,
    key_events: list[str] | None = None,
    wellbeing_explicit: bool = False,
    wellbeing_points: list[str] | None = None,
) -> SummaryResult:
    return SummaryResult(
        key_events=key_events or [],
        work_and_load=[],
        wellbeing=WellbeingSection(
            explicit_support=wellbeing_explicit,
            points=wellbeing_points if wellbeing_points is not None else [],
        ),
        supportive_factors=[],
        open_questions=[],
    )


# ---------------------------------------------------------------------------
# entry splitting
# ---------------------------------------------------------------------------


def test_split_keeps_entries_in_order_and_drops_the_header() -> None:
    entries = split_normalized_entries(_DOCUMENT)
    assert len(entries) == 2
    assert entries[0].startswith("[2026-09-07 09:00] TEXT\nПервая запись.")
    assert entries[1].startswith("[2026-09-07 20:00] TEXT\nВторая запись")
    assert "\nс переносом строки." in entries[1]


def test_split_rejoining_reproduces_the_source_layout() -> None:
    entries = split_normalized_entries(_DOCUMENT)
    rejoined = "\n\n".join(entries) + "\n"
    # Only the separator line and the blank after it are header artifacts.
    body = _DOCUMENT.split("---\n", 1)[1].lstrip("\n")
    assert rejoined == body


def test_split_rejects_document_without_entries() -> None:
    with pytest.raises(ValueError):
        split_normalized_entries("# Только заголовок\n---\nбез записей\n")


def test_split_rejects_document_without_separator() -> None:
    with pytest.raises(ValueError):
        split_normalized_entries("[2026-09-07 09:00] TEXT\nБез шапки.\n")


# ---------------------------------------------------------------------------
# deterministic character chunking
# ---------------------------------------------------------------------------


def test_chunking_never_splits_an_entry_and_preserves_order() -> None:
    entries = split_normalized_entries(_DOCUMENT)
    chunks = chunk_entries_by_chars(entries, chunk_size=10)
    # Two entries, both larger than the tiny chunk size → one entry each.
    assert len(chunks) == 2
    assert chunks[0] == [entries[0]]
    assert chunks[1] == [entries[1]]


def test_chunking_groups_entries_up_to_the_size_budget() -> None:
    entries = [f"Запись {index}: " + "x" * 40 for index in range(5)]
    chunks = chunk_entries_by_chars(entries, chunk_size=120)
    flattened = [entry for chunk in chunks for entry in chunk]
    assert flattened == entries  # order preserved, nothing dropped
    for chunk in chunks[:-1]:
        assert sum(len(entry) + 2 for entry in chunk) <= 120


def test_chunking_is_deterministic() -> None:
    entries = [f"Запись {index}: " + "y" * 30 for index in range(9)]
    first = chunk_entries_by_chars(entries, chunk_size=100)
    second = chunk_entries_by_chars(entries, chunk_size=100)
    assert first == second


# ---------------------------------------------------------------------------
# deterministic merge (reduce)
# ---------------------------------------------------------------------------


def test_merge_concatenates_sections_in_order_and_deduplicates() -> None:
    partial_a = _result(key_events=["Событие A.", "Событие B."])
    partial_b = _result(key_events=["Событие B.", "Событие C."])
    merged = merge_partials([partial_a, partial_b])
    assert merged.key_events == ["Событие A.", "Событие B.", "Событие C."]


def test_merge_wellbeing_requires_support_from_any_partial() -> None:
    without_support = _result(wellbeing_explicit=False)
    with_support = _result(wellbeing_explicit=True, wellbeing_points=["Усталость к концу недели."])
    merged = merge_partials([without_support, with_support])
    assert merged.wellbeing.explicit_support is True
    assert merged.wellbeing.points == ["Усталость к концу недели."]


def test_merge_wellbeing_without_any_support_stays_empty() -> None:
    merged = merge_partials([_result(wellbeing_explicit=False), _result(wellbeing_explicit=False)])
    assert merged.wellbeing.explicit_support is False
    assert merged.wellbeing.points == []
