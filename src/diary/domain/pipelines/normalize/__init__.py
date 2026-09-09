"""Normalization pipeline package (layer 2; Task 13).

Concrete derived-artifact pipeline over the Task-12 framework:

* :mod:`.parser` — deterministic parser of the layer-1 original TXT format
  into structured, order-preserving entries;
* :mod:`.schema` — the strict Pydantic result schema (fixed fields, no extra
  keys) validated before any file is written;
* :mod:`.pipeline` — :class:`~.pipeline.NormalizePipeline`: prompt building
  with deterministic entry-chunking, schema + completeness validation, and
  backend rendering of the layer-2 TXT with a lineage header;
* :mod:`.registration` — explicit wiring of the pipeline into the registry
  (called from the worker startup, never as an import side effect).

No diary content, prompts or LLM responses are ever logged or persisted
anywhere in this package (AGENTS.md §7–8).
"""

from diary.domain.pipelines.normalize.batch import (
    enqueue_normalize_batch,
    list_eligible_exports,
)
from diary.domain.pipelines.normalize.parser import (
    OriginalDocument,
    OriginalParseError,
    ParsedEntry,
    parse_original_text,
)
from diary.domain.pipelines.normalize.pipeline import (
    ENTRIES_HEADER,
    PROMPT_PATH,
    PROMPT_VERSION,
    NormalizePipeline,
    extract_json_object,
)
from diary.domain.pipelines.normalize.registration import (
    ensure_normalize_pipeline_registered,
)
from diary.domain.pipelines.normalize.schema import (
    NormalizedEntry,
    NormalizeResult,
)

__all__ = [
    "ENTRIES_HEADER",
    "NormalizePipeline",
    "NormalizeResult",
    "NormalizedEntry",
    "OriginalDocument",
    "OriginalParseError",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "ParsedEntry",
    "enqueue_normalize_batch",
    "ensure_normalize_pipeline_registered",
    "extract_json_object",
    "list_eligible_exports",
    "parse_original_text",
]
