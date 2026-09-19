"""Wellbeing reflection pipeline (layer 4; Task 15, architecture brief §10.3).

Responsibilities
================

* Consume **only** a normalized artifact (layer 2) resolved by database
  identity — the framework resolves the export's latest succeeded
  ``normalized`` artifact through ``input_layer``; the original export (layer
  1) is never read. An export without a normalized result fails permanently
  with a content-free "run the upstream pipeline first" error.
* **Safety classification runs before the model and over the FULL normalized
  document** (:func:`~diary.domain.pipelines.wellbeing.safety.classify`). The
  level is decided by deterministic code rules; the LLM output cannot set,
  downgrade or override it — the structured schema carries no level field at
  all. The same classifier drives the read-only wellbeing page, so the
  deterministic urgent guidance never depends on an LLM run having succeeded.
* Build **one single-pass request** over the whole normalized document
  (entries re-joined from the layer-2 file; the layer-2 header is not sent).
  The wellbeing reflection is a holistic, deliberate read of the week, so the
  MVP deliberately does not chunk or map-reduce it: a partial reflection over
  part of a week could give a false sense of a complete reading, and the
  tasklist (Task 15) does not require chunking. An oversized document fails
  safely through the provider's own limits (a deterministic, content-free run
  failure; no file is written).
* Validate the model response against the strict :class:`WellbeingResult`
  schema (extra keys rejected, observed facts must be date-anchored,
  recommendations 1-3) before anything is written; a malformed response fails
  the run permanently with no artifact.
* Render the final layer-4 Markdown **from the validated structured data plus
  the deterministic level** (never from raw model text). Every report carries
  the mandatory disclaimer (``safety.DISCLAIMER``) and its header identifies
  the exact normalized source file. When the deterministic level is
  ``urgent`` the report leads with the **fixed escalation block**
  (``safety.escalation_block()``); when it is ``needs_support`` a fixed
  supportive note is appended after the sections. None of that text comes from
  the model (ADR-006: free LLM text never governs the crisis scenario).

Privacy: diary content and the model output are in-memory request/structured
objects only; error messages are static (AGENTS.md §7-8, §15).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from diary.core.models_config import TaskType
from diary.core.settings import PROJECT_ROOT, get_settings
from diary.db.models import Artifact, ArtifactLayer, PipelineType, WeeklyExport
from diary.domain.exports.weeks import week_monday
from diary.domain.llm.models import LLMRequest, LLMResponse, utc_now
from diary.domain.pipelines.base import (
    Pipeline,
    PipelineInputError,
    PipelineOutputError,
    PipelinePermanentError,
)
from diary.domain.pipelines.normalize.pipeline import extract_json_object
from diary.domain.pipelines.summarize.pipeline import split_normalized_entries
from diary.domain.pipelines.wellbeing.safety import (
    DISCLAIMER,
    LEVEL_LABEL,
    SUPPORT_NOTE,
    SafetyLevel,
    classify,
    escalation_block,
)
from diary.domain.pipelines.wellbeing.schema import WellbeingResult

#: Versioned prompt template of this pipeline (recorded in run provenance).
PROMPT_VERSION = "v1"
#: Prompt template location: ``prompts/wellbeing/v1.md``.
PROMPT_PATH = PROJECT_ROOT / "prompts" / "wellbeing" / f"{PROMPT_VERSION}.md"

#: Line introducing the normalized document inside a request.
_TEXT_LABEL = "Ниже — полный нормализованный документ недели:"

#: Static, non-sensitive error summaries (AGENTS.md §8).
_INPUT_READ_MESSAGE = "The normalized source file cannot be read."
_INPUT_PARSE_MESSAGE = "The normalized source cannot be parsed into entries."
_INPUT_EMPTY_MESSAGE = "The normalized source contains no entries to reflect on."
_OUTPUT_INVALID_MESSAGE = "The pipeline output did not pass structured validation."
_PROMPT_MISSING_MESSAGE = "The wellbeing prompt template is unavailable."
_CONTEXT_MISSING_MESSAGE = "The wellbeing run context is missing."

#: Rendered marker for a section without content.
_NO_DATA_MARKER = "— нет данных —"


@dataclass(frozen=True)
class _RunContext:
    """Transient per-run state shared between request building, response
    parsing and rendering (single-threaded worker; see the concurrency note in
    the normalization pipeline module docstring, mirrored here)."""

    #: The exact normalized input artifact of this run (for the header link).
    input_artifact: Artifact
    #: The deterministic safety level of the FULL source text (never from LLM).
    level: SafetyLevel


class WellbeingPipeline(Pipeline[WellbeingResult]):
    """The real wellbeing reflection pipeline (registered in Task 15).

    Registered for :attr:`~diary.db.models.PipelineType.wellbeing` in the
    pipeline registry by :mod:`.registration`; the worker registers it at
    startup so real ``wellbeing`` jobs no longer fail with "not registered".
    """

    pipeline_type = PipelineType.wellbeing
    layer = ArtifactLayer.wellbeing
    output_ext = ".md"
    task_type = TaskType.wellbeing_analysis
    result_schema = WellbeingResult
    #: The layer-4 header embeds the export row and the derived version, so the
    #: framework calls :meth:`render_artifact` with that context.
    render_requires_context = True
    #: Layer 4 is built only from layer 2 (normalized), never the original.
    input_layer = ArtifactLayer.normalized

    def __init__(self) -> None:
        self._context: _RunContext | None = None

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def build_request(self, *, export: WeeklyExport, input_artifact: Artifact | None) -> LLMRequest:
        """Build the single single-pass reflection request of the run.

        The input must be a normalized artifact (layer 2) — the framework
        resolves it for this pipeline. The deterministic safety classifier
        runs first over the **full** normalized file text; then the layer-2
        entries (header excluded) are sent to the model in one request.
        """
        if input_artifact is None:
            raise PipelineInputError(
                "Wellbeing analysis always consumes a normalized artifact (layer 2)."
            )
        text = self._load_normalized_text(input_artifact)
        # The full source text is what the crisis classifier must see.
        level = classify(text)
        try:
            entries = split_normalized_entries(text)
        except ValueError as exc:
            raise PipelineInputError(_INPUT_PARSE_MESSAGE) from exc
        if not entries:
            raise PipelineInputError(_INPUT_EMPTY_MESSAGE)
        self._context = _RunContext(input_artifact=input_artifact, level=level)
        document = "\n\n".join(entries)
        prompt = self._load_prompt()
        content = f"{prompt}\n\n{_TEXT_LABEL}\n\n{document}"
        return LLMRequest.from_prompt(content, prompt_version=PROMPT_VERSION)

    def _load_normalized_text(self, input_artifact: Artifact) -> str:
        """Read the normalized input file (resolved by DB identity)."""
        assert input_artifact.path is not None  # a layer-2 artifact is file-backed
        path: Path = get_settings().data_root / input_artifact.path
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PipelineInputError(_INPUT_READ_MESSAGE) from exc

    def _load_prompt(self) -> str:
        """The versioned prompt template text (static, no diary content)."""
        try:
            return PROMPT_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            raise PipelinePermanentError(_PROMPT_MISSING_MESSAGE) from exc

    # ------------------------------------------------------------------
    # Structured-output parsing
    # ------------------------------------------------------------------

    def parse_result(self, response: LLMResponse) -> WellbeingResult:
        """Validate one response against the strict wellbeing schema.

        The schema carries no safety-level field: the deterministic level from
        :meth:`build_request` is what drives the render and the crisis flow.
        """
        try:
            payload = extract_json_object(response.text)
        except ValueError as exc:
            raise PipelineOutputError(_OUTPUT_INVALID_MESSAGE) from exc
        try:
            return WellbeingResult.model_validate(payload)
        except ValidationError as exc:
            raise PipelineOutputError(_OUTPUT_INVALID_MESSAGE) from exc

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, structured: WellbeingResult) -> str:
        """Plain-render guard (this pipeline always renders with run context).

        The framework calls :meth:`render_artifact` because
        ``render_requires_context`` is set; reaching this method means the
        framework contract changed and must fail loudly, not produce a
        header-less artifact.
        """
        raise PipelinePermanentError("The wellbeing pipeline must render with run context.")

    def render_artifact(
        self, structured: WellbeingResult, *, export: WeeklyExport, version: int
    ) -> str:
        """Render the layer-4 Markdown from validated data plus the export row.

        The header embeds the week lineage, the exact normalized source file
        name and the deterministic ``Уровень`` (``normal``/``needs_support``/
        ``urgent``). Every report carries the mandatory disclaimer. For
        ``urgent`` the report leads with the fixed escalation block; for
        ``needs_support`` a fixed supportive note follows the sections. All of
        this content comes from code constants, never from the model.
        """
        context = self._context
        if context is None:
            raise PipelineOutputError(_CONTEXT_MISSING_MESSAGE)
        settings = get_settings()
        created_local = utc_now().astimezone(settings.tzinfo)
        sunday = export.week_end_date
        assert context.input_artifact.path is not None  # a layer-2 artifact is file-backed
        source_filename = context.input_artifact.path.rsplit("/", 1)[-1]
        lines: list[str] = [
            "# Wellbeing-рефлексия",
            f"Неделя: {week_monday(sunday).isoformat()} — {sunday.isoformat()}",
            f"Дата воскресенья: {sunday.isoformat()}",
            f"Номер выгрузки: {export.seq_number:04d}",
            f"Ревизия: r{export.revision}",
            f"Версия: v{version}",
            f"Источник (нормализованный): {source_filename}",
            f"{LEVEL_LABEL}: {context.level.value}",
            f"Создано: {created_local.isoformat(timespec='seconds')}",
            f"Timezone: {settings.timezone}",
            "",
            "---",
            "",
        ]
        if context.level is SafetyLevel.urgent:
            lines.extend(escalation_block())
            lines.extend(["", "---", ""])
        lines.append(f"> {DISCLAIMER}")
        lines.append("")
        lines.extend(self._render_section("Наблюдаемые факты", structured.observed_facts))
        lines.extend(
            self._render_section("Осторожная интерпретация", structured.cautious_interpretation)
        )
        lines.extend(self._render_section("Рекомендации", structured.recommendations))
        lines.extend(
            self._render_section("Уверенность и ограничения", structured.confidence_and_limitations)
        )
        if context.level is SafetyLevel.needs_support:
            lines.extend(["", "---", "", SUPPORT_NOTE])
        return "\n".join(lines).rstrip("\n") + "\n"

    @staticmethod
    def _render_section(title: str, items: list[str]) -> list[str]:
        """One ``##`` section: a single bullet per item, one line each.

        Item text is whitespace-collapsed so diary text can never smuggle
        blank lines or markdown structure into the rendered bullet list.
        """
        lines = [f"## {title}", ""]
        if items:
            for item in items:
                collapsed = " ".join(item.split())
                if collapsed:
                    lines.append(f"- {collapsed}")
            if len(lines) == 2:
                lines.append(_NO_DATA_MARKER)
        else:
            lines.append(_NO_DATA_MARKER)
        return lines


__all__ = [
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "WellbeingPipeline",
]
