"""Daily-summary layer — ``daily_summary`` (phase 2, task T05).

The layer turns **one local calendar day** of diary messages (``APP_TIMEZONE``,
ADR-007 §1) into a grounded summary and stores the validated result under the
day anchor. Its definition of done (``docs/tasklist-phase2.md`` §6, T05) is a
set of vetoes:

* **strictly the contract's content fields** — ``facts`` / ``themes`` /
  ``energy`` / ``open_questions``, plus the backend-owned ``date``: no extra
  field survives validation (``extra="forbid"``), and the prompt asks for
  nothing else;
* **no morality** — no advice, recommendations, evaluation, consolation or
  medical framing; the prompt forbids them and the generated text fields are
  covered by the golden ``forbidden_lexemes`` check in T10;
* **``energy`` is never invented** — the field is one of ``low``/``medium``/
  ``high`` only when the source states the level, and ``null`` otherwise
  (ADR-007 §4). A default level is a veto, not a convenience;
* **no facts beyond the input** — numbers, names, events and motives come from
  the day's messages only;
* **one day per run** — the input is the day's messages and nothing else; days
  are never aggregated "while we are at it" (that is the weekly layer, T06);
* **never more than the context window** (T23) — prompt + entries + the
  generation cap must fit ``num_ctx``, estimated with the eval contract's
  coefficient. A day that fits one window is summarized in one call exactly as
  before; a longer day is split by entries, each part is summarized with the same
  prompt, and the parts are folded back into one day by the merge template.

The day's messages are **immutable input**: the selector reads ``messages``,
``audio_files`` and ``transcriptions`` only, and the layer's single write path
is the append-only result store of task T08
(``diary.db.results.save_layer_result``), which appends a new version for the
(day, layer) pair on every rerun (ADR-005).

Modules
=======

* :mod:`.day_input` — the local-day boundary math and the read-only selection of
  the day's messages as ``raw_entry[]``;
* :mod:`.pipeline` — :class:`SumDailyPipeline`: the versioned prompt template, the
  request built from the day's entries, and the strict parse of the completion
  into the :class:`~diary.domain.contracts.DailySummary` contract (the ``date``
  anchor is supplied by the backend);
* :mod:`.service` — :func:`sum_daily` / :func:`run_sum_daily`: one day summary
  run, stored through ``save_layer_result``;
* :mod:`.registration` — guarded wiring of the layer into the pipeline registry,
  mirroring the STT-correction registration.

Privacy: the day's texts enter an in-memory request/contract only. They are
never logged, and no error message, run metadata or stored payload contains
anything but the validated summary (AGENTS.md §7-8, §15).
"""

from diary.domain.pipelines.sum_daily.day_input import (
    DayInputError,
    collect_day_entries,
    day_bounds,
    local_date_of,
)
from diary.domain.pipelines.sum_daily.pipeline import (
    ENTRIES_HEADER,
    MERGE_PROMPT_PATH,
    MERGE_PROMPT_VERSION,
    NUM_CTX,
    PARTS_HEADER,
    PROFILE_NAME,
    PROMPT_PATH,
    PROMPT_VERSION,
    SumDailyPipeline,
)
from diary.domain.pipelines.sum_daily.registration import ensure_sum_daily_pipeline_registered
from diary.domain.pipelines.sum_daily.service import SumDailyOutcome, run_sum_daily, sum_daily

__all__ = [
    "ENTRIES_HEADER",
    "MERGE_PROMPT_PATH",
    "MERGE_PROMPT_VERSION",
    "NUM_CTX",
    "PARTS_HEADER",
    "PROFILE_NAME",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "DayInputError",
    "SumDailyOutcome",
    "SumDailyPipeline",
    "collect_day_entries",
    "day_bounds",
    "ensure_sum_daily_pipeline_registered",
    "local_date_of",
    "run_sum_daily",
    "sum_daily",
]
