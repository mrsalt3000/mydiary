"""Weekly original TXT exports (Task 10, architecture brief §7).

Public surface of the domain package:

- :mod:`diary.domain.exports.weeks` — pure calendar-week helpers (boundaries,
  Sunday-date derivation);
- :mod:`diary.domain.exports.service` — the export use cases: collecting the
  week's messages, deterministic TXT rendering, global no-gap sequence
  numbering, explicit re-export revisions, atomic file writes and the
  ``weekly_exports`` bookkeeping row.

Derived layers (normalize/summary/wellbeing) build on top of this package in
later tasks (Task 12–15); nothing here depends on them.
"""

from diary.domain.exports.weeks import (
    is_sunday,
    sunday_date_for,
    week_bounds,
    week_monday,
)

__all__ = [
    "is_sunday",
    "sunday_date_for",
    "week_bounds",
    "week_monday",
]
