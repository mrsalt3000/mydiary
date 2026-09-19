"""Weekly-summary layer — ``weekly_summary`` (phase 2, task T06).

The layer merges **the daily summaries of one ISO week** into a single grounded
rollup and stores the validated result under the week anchor. Its definition of
done (``docs/tasklist-phase2.md`` §6, T06) is a set of vetoes:

* **only a merge of the day summaries, never raw** — the input is the
  ``DailySummary`` results of the layer ``daily_summary`` (task T05), read from
  ``artifacts``; this package never touches ``messages``, ``transcriptions`` or
  ``weekly_exports``;
* **the anchor is the ISO week** — ``week_id`` (``YYYY-Www``, ISO year) is
  computed from the local day in ``APP_TIMEZONE`` (project weeks are
  Monday–Sunday and map onto the ISO week of their Monday, ADR-007 §5); the
  weekly export is **not** an anchor of this layer;
* **strictly the contract's fields** — ``rollup`` plus the backend-owned
  ``week_id`` / ``daily_ids``; an extra field (advice, evaluation) fails the run
  and stores nothing (``extra="forbid"``);
* **no advice, no diagnoses** — the prompt forbids advice, morality, evaluation
  and medical framing; the layer stores no ``facts[]`` of its own, so the
  machine check of the golden set (T10) is what pins that veto;
* **no day lost or duplicated** — the context-budget chunker partitions the
  days into ordered, non-overlapping pieces; if more than one piece is needed,
  the partial rollups are reconciled into one coherent ``rollup`` by a merge
  call, never concatenated.

The day summaries are **immutable input**: the selector only reads the stored
results, and the layer's single write path is the append-only result store of
task T08 (``diary.db.results.save_layer_result``), which appends a new version
for the (week, layer) pair on every rerun (ADR-005).

Modules
=======

* :mod:`.week_input` — ISO-week math (``week_id_of``/``week_days``), the loud
  week-membership guard and the read-only selection of the week's daily
  summaries;
* :mod:`.pipeline` — :class:`SumWeeklyPipeline`: the versioned prompt template,
  the rollup/merge requests, the context-budget chunker and the strict parse of
  the completion into the :class:`~diary.domain.contracts.WeeklySummary`
  contract (``week_id``/``daily_ids`` are supplied by the backend);
* :mod:`.service` — :func:`sum_weekly` / :func:`run_sum_weekly`: one weekly
  summary run, stored through ``save_layer_result``;
* :mod:`.registration` — guarded wiring of the layer into the pipeline registry,
  mirroring the daily-summary registration.

Privacy: the day summaries enter in-memory requests/contracts only. They are
never logged, and no error message, run metadata or stored payload contains
anything but the validated summary (AGENTS.md §7-8, §15).
"""

from diary.domain.pipelines.sum_weekly.pipeline import (
    MERGE_HEADER,
    NUM_CTX,
    PROFILE_NAME,
    PROMPT_PATH,
    PROMPT_VERSION,
    SUMMARIES_HEADER,
    SumWeeklyPipeline,
)
from diary.domain.pipelines.sum_weekly.registration import ensure_sum_weekly_pipeline_registered
from diary.domain.pipelines.sum_weekly.service import SumWeeklyOutcome, run_sum_weekly, sum_weekly
from diary.domain.pipelines.sum_weekly.week_input import (
    WeeklyInputError,
    collect_week_summaries,
    ensure_summaries_match_week,
    week_days,
    week_id_of,
    week_monday,
)

__all__ = [
    "MERGE_HEADER",
    "NUM_CTX",
    "PROFILE_NAME",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "SUMMARIES_HEADER",
    "SumWeeklyOutcome",
    "SumWeeklyPipeline",
    "WeeklyInputError",
    "collect_week_summaries",
    "ensure_summaries_match_week",
    "ensure_sum_weekly_pipeline_registered",
    "run_sum_weekly",
    "sum_weekly",
    "week_days",
    "week_id_of",
    "week_monday",
]
