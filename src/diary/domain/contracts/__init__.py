"""Structured contracts of the phase-2 derivational layers (task T00).

Public surface of the family introduced by the second backlog
(``docs/tasklist-phase2.md``, ADR-007):

- :class:`RawEntry` — one source diary message (input side);
- :class:`SttCorrection` — corrected message text with ``uncertain[]``;
- :class:`DailySummary` — grounded per-day rollup;
- :class:`WeeklySummary` — grounded per-ISO-week rollup;
- :class:`CoachNote` — cautious note over a weekly summary (new layer 5);
- :data:`COACH_DISCLAIMER` — the fixed limitation of every coaching note.

Every model forbids unknown fields and is validated before any artifact is
written (AGENTS.md §15). JSON Schema of each contract is generated into
``schemas/<contract_name>.json`` (``make contracts``); the golden eval set in
``eval/golden`` is loaded through :mod:`diary.domain.contracts.golden`.

The layer 4 (``wellbeing``) pipeline and its ``DISCLAIMER`` are untouched by
this package: the coaching note is an additional layer, not a replacement.
"""

from diary.domain.contracts.coach_note import COACH_DISCLAIMER, CoachNote
from diary.domain.contracts.daily_summary import DailySummary
from diary.domain.contracts.raw_entry import RawEntry
from diary.domain.contracts.stt_corrected import SttCorrection
from diary.domain.contracts.weekly_summary import WeeklySummary

__all__ = [
    "COACH_DISCLAIMER",
    "CoachNote",
    "DailySummary",
    "RawEntry",
    "SttCorrection",
    "WeeklySummary",
]
