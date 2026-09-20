"""Coaching-note layer — ``coach_note`` (layer 5, phase 2, task T07).

The layer turns **one weekly summary** into a short, cautious note: exactly
three observations, exactly two hypotheses and one small experiment for the next
7 days, plus a fixed disclaimer. Its definition of done
(``docs/tasklist-phase2.md`` §6, T07) is a set of vetoes:

* **one weekly summary, nothing else** — the input is the newest stored
  ``WeeklySummary`` artifact of an ISO week (task T06); raw messages, the
  ``daily_summary`` layer and weekly exports are never read;
* **layer 5 is separate from layer 4** — the layer has its own pipeline type
  (``coach_note``), its own prompt (``prompts/coach_note/v2.md``), its own
  profile (``diary_coach``) and its own disclaimer constant
  (:data:`~diary.domain.contracts.COACH_DISCLAIMER`). It does **not** import,
  re-use or modify ``diary.domain.pipelines.wellbeing`` (its safety classifier,
  its ``DISCLAIMER``) or the ``PipelineType.wellbeing``/``ArtifactLayer.wellbeing``
  values (ADR-007 §8; T07 decision 2);
* **hypotheses are hypotheses** — the prompt requires tentative wording
  («Возможно,», «Похоже,»), forbids asserted causes («ты устал, потому что…»),
  diagnoses, treatment plans, new facts and prescriptive advice. The contract
  carries the shape (3/2/1, the digit ``7`` in ``experiment_7d``, disclaimer
  equality) and is never weakened;
* **launch is explicit only** — the layer registers no automatic trigger, has
  no job handler and is called from no route; it runs only when a caller
  invokes :func:`run_coach_note` (T07 decision 3).

The weekly summary is **immutable input**: the selector only reads stored
results, and the layer's single write path is the append-only result store of
task T08 (``diary.db.results.save_layer_result``), which appends a new version
for the (upstream artifact, layer) pair on every rerun (ADR-005).

Modules
=======

* :mod:`.coach_input` — the read-only selection of exactly one weekly summary,
  and the documented relation between the two values both named
  ``summary_id`` (the contract's ``week_id`` string vs the storage anchor's
  artifact id);
* :mod:`.pipeline` — :class:`CoachNotePipeline`: the versioned prompt template,
  the request and the strict parse into the
  :class:`~diary.domain.contracts.CoachNote` contract (``summary_id`` and
  ``disclaimer`` are supplied by the backend);
* :mod:`.service` — :func:`coach_note` / :func:`run_coach_note`: one coaching
  note, stored through ``save_layer_result``;
* :mod:`.registration` — guarded wiring of the layer into the pipeline registry
  on explicit request (never on import, never from the worker startup).

Privacy: the weekly summary enters in-memory requests/contracts only. It is
never logged, and no error message, run metadata or stored payload contains
anything but the validated note (AGENTS.md §7-8, §15).
"""

from diary.domain.pipelines.coach.coach_input import (
    CoachInput,
    CoachInputError,
    coach_input_from_artifact,
    collect_week_summary,
    ensure_input_matches_week,
    week_summary_artifact,
)
from diary.domain.pipelines.coach.pipeline import (
    PROFILE_NAME,
    PROMPT_PATH,
    PROMPT_VERSION,
    SUMMARY_HEADER,
    CoachNotePipeline,
)
from diary.domain.pipelines.coach.registration import ensure_coach_pipeline_registered
from diary.domain.pipelines.coach.service import CoachNoteOutcome, coach_note, run_coach_note

__all__ = [
    "PROFILE_NAME",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "SUMMARY_HEADER",
    "CoachInput",
    "CoachInputError",
    "CoachNoteOutcome",
    "CoachNotePipeline",
    "coach_input_from_artifact",
    "coach_note",
    "collect_week_summary",
    "ensure_coach_pipeline_registered",
    "ensure_input_matches_week",
    "run_coach_note",
    "week_summary_artifact",
]
