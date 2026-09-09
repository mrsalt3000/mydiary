"""Summarization pipeline package (layer 3, Task 14).

A concrete derived-artifact pipeline that consumes a normalized document
(layer 2) and renders a concise, source-grounded weekly Markdown summary into
``data/exports/03_summaries/``. See the module docstrings of ``pipeline.py``,
``schema.py`` and ``batch.py`` for the full behavior.
"""

from diary.domain.pipelines.summarize.batch import (
    enqueue_summarize_batch,
    list_eligible_exports,
)
from diary.domain.pipelines.summarize.pipeline import (
    FINAL_MODE_LINE,
    PARTIAL_MODE_PREFIX,
    PROMPT_PATH,
    PROMPT_VERSION,
    SummarizePipeline,
    chunk_entries_by_chars,
    merge_partials,
    split_normalized_entries,
)
from diary.domain.pipelines.summarize.registration import (
    ensure_summarize_pipeline_registered,
)
from diary.domain.pipelines.summarize.schema import SummaryResult, WellbeingSection

__all__ = [
    "FINAL_MODE_LINE",
    "PARTIAL_MODE_PREFIX",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "SummaryResult",
    "SummarizePipeline",
    "WellbeingSection",
    "chunk_entries_by_chars",
    "ensure_summarize_pipeline_registered",
    "enqueue_summarize_batch",
    "list_eligible_exports",
    "merge_partials",
    "split_normalized_entries",
]
