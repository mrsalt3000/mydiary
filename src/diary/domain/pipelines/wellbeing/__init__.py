"""Wellbeing reflection pipeline package (layer 4, Task 15).

A concrete derived-artifact pipeline that consumes a normalized document
(layer 2) and renders a cautious, fact-grounded, non-diagnostic Markdown
reflection into ``data/exports/04_wellbeing/``, gated by an explicit user
action with confirmation and guarded by a deterministic safety classifier
(ADR-006). See the module docstrings of ``safety.py``, ``schema.py``,
``pipeline.py`` and ``service.py`` for the full behavior.
"""

from diary.domain.pipelines.wellbeing.pipeline import (
    PROMPT_PATH,
    PROMPT_VERSION,
    WellbeingPipeline,
)
from diary.domain.pipelines.wellbeing.registration import (
    ensure_wellbeing_pipeline_registered,
)
from diary.domain.pipelines.wellbeing.safety import (
    DISCLAIMER,
    ESCALATION_LINES,
    ESCALATION_TITLE,
    SUPPORT_NOTE,
    SafetyLevel,
    classify,
    escalation_block,
)
from diary.domain.pipelines.wellbeing.schema import WellbeingResult
from diary.domain.pipelines.wellbeing.service import (
    WellbeingConfirmationRequiredError,
    WellbeingDisabledError,
    WellbeingEnqueueError,
    WellbeingInputUnavailableError,
    enqueue_wellbeing_run,
)

__all__ = [
    "DISCLAIMER",
    "ESCALATION_LINES",
    "ESCALATION_TITLE",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "SUPPORT_NOTE",
    "SafetyLevel",
    "WellbeingConfirmationRequiredError",
    "WellbeingDisabledError",
    "WellbeingEnqueueError",
    "WellbeingInputUnavailableError",
    "WellbeingPipeline",
    "WellbeingResult",
    "classify",
    "enqueue_wellbeing_run",
    "ensure_wellbeing_pipeline_registered",
    "escalation_block",
]
