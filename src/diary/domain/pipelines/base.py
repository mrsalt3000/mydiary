"""Abstract pipeline interface and error taxonomy (Task 12).

A concrete pipeline (Tasks 13-15) implements four responsibilities over one
derived-artifact layer:

* class attributes identify it: ``pipeline_type`` (which durable job type
  drives it), ``layer`` (the artifact layer it produces), ``output_ext``
  (rendered file extension) and ``task_type`` (the Task-03 LLM profile it
  uses);
* :meth:`Pipeline.build_request` turns the resolved input (the source export
  and, for layers 3-4, the exact input artifact) into an :class:`LLMRequest`.
  The input content may enter the request — it is an in-memory object only and
  must never be logged;
* :meth:`Pipeline.parse_result` parses the provider response into the
  structured model (:attr:`Pipeline.result_schema`). Invalid output raises a
  :class:`PipelineOutputError` — the framework then records a failed run and
  never writes a file;
* :meth:`Pipeline.render` produces the final TXT/Markdown text *from the
  validated structured data* (never from raw LLM text; AGENTS.md §15).

Error taxonomy: :class:`PipelinePermanentError` and its subclasses are
deterministic failures that the worker must not retry (unregistered pipeline,
invalid input, invalid structured output, storage failure). Transient LLM
errors (network, timeouts, rate limits, HTTP 5xx) are raised unchanged by the
framework and follow the Task-08 retry policy. Every message is a static,
non-sensitive summary — no diary content, prompts, LLM output or secrets
(AGENTS.md §8).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

from diary.core.models_config import TaskType
from diary.db.models import Artifact, ArtifactLayer, PipelineType, WeeklyExport
from diary.domain.llm.models import LLMRequest, LLMResponse

#: Type of the structured, validated pipeline output (a Pydantic model).
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class PipelineError(Exception):
    """Base class for pipeline errors (never diary content or secrets)."""


class PipelinePermanentError(PipelineError):
    """A deterministic pipeline failure that must not be retried.

    The worker maps this to a permanent job failure immediately, regardless of
    the remaining ``max_attempts``. Messages are static and non-sensitive.
    """


class PipelineNotRegisteredError(PipelinePermanentError):
    """No implementation is registered for the requested pipeline type yet."""


class PipelineInputError(PipelinePermanentError):
    """The job input is unusable (missing export/artifact/file, bad payload)."""


class PipelineExportNotFoundError(PipelineInputError):
    """The export row referenced by the job no longer exists."""


class PipelineOriginalFileMissingError(PipelineInputError):
    """The export row exists but its original artifact file is gone."""


class PipelineOutputError(PipelinePermanentError):
    """The structured LLM output failed validation; no file may be written."""


class PipelineStorageError(PipelinePermanentError):
    """The derived artifact file could not be written atomically."""


class Pipeline(ABC, Generic[SchemaT]):
    """Abstract derived-artifact pipeline (normalize/summarize/wellbeing).

    Instances are stateless and registered in the pipeline registry under
    ``pipeline_type``; the framework resolves the LLM profile by
    ``task_type`` and drives ``build_request`` → provider → ``parse_result``
    → ``render`` for every run.
    """

    #: Pipeline/job kind that drives this pipeline (normalize/summarize/...).
    pipeline_type: ClassVar[PipelineType]
    #: Artifact layer this pipeline produces (normalized/summary/wellbeing).
    layer: ClassVar[ArtifactLayer]
    #: File extension of the rendered output (``.txt``/``.md``).
    output_ext: ClassVar[str]
    #: Task-03 LLM profile this pipeline uses (normalization/summarization/...).
    task_type: ClassVar[TaskType]
    #: Pydantic model of the structured output the pipeline validates.
    result_schema: ClassVar[type[SchemaT]]
    #: Derived artifact layer this pipeline consumes as its input, or ``None``
    #: when the input is the original export itself. Layers 3-4 declare
    #: ``ArtifactLayer.normalized`` here; the framework (Task 12) then resolves
    #: the export's latest succeeded artifact of that layer when a job payload
    #: carries no explicit ``input_artifact_id``, and the pipeline never reads
    #: the original export (AGENTS.md §6).
    input_layer: ClassVar[ArtifactLayer | None] = None

    @abstractmethod
    def build_request(self, *, export: WeeklyExport, input_artifact: Artifact | None) -> LLMRequest:
        """Build the LLM request for one run.

        ``export`` is the source original export of the lineage (its
        ``path``/``week_end_date``/``seq_number``/``revision`` anchor the run);
        ``input_artifact`` is the exact derived input artifact when the
        pipeline consumes a lower derived layer (normalized for layers 3-4),
        else ``None`` when the input is the original export itself. The input
        content may be embedded in the request but must never be logged or
        persisted outside the request object.
        """

    @abstractmethod
    def parse_result(self, response: LLMResponse) -> SchemaT:
        """Parse a provider response into the validated structured model.

        Raises :class:`PipelineOutputError` when the response is not a valid
        instance of :attr:`result_schema`; the framework then fails the run
        without writing any file.
        """

    @abstractmethod
    def render(self, structured: SchemaT) -> str:
        """Render the final TXT/Markdown from validated structured data."""

    # ------------------------------------------------------------------
    # Task-13 extensions (chunked runs and context-aware rendering)
    # ------------------------------------------------------------------
    # The concrete methods below keep the Task-12 single-request flow
    # byte-for-byte identical (the defaults build one request, expect one
    # response and call :meth:`render` with the structured output alone).
    # Pipelines that need more — normalization splits long exports into
    # deterministic entry-chunks and its artifact header embeds the export
    # lineage and the derived version — override them explicitly.

    #: Whether the framework should render through
    #: :meth:`render_artifact` (passing the export row and the computed
    #: derived version) instead of the plain single-argument :meth:`render`.
    #: Set by pipelines whose artifact header embeds ``seq/revision/version``
    #: lineage (normalization, Task 13). The Task-12 default stays ``False``.
    render_requires_context: ClassVar[bool] = False

    def build_requests(
        self, *, export: WeeklyExport, input_artifact: Artifact | None
    ) -> list[LLMRequest]:
        """Build the run's LLM request(s) — one per provider call.

        The default returns a single request from :meth:`build_request` (the
        Task-12 flow). Pipelines that split their input into chunks
        (normalization, Task 13) override this to return one request per
        chunk, in input order; the framework calls the provider once per
        request and hands all responses to :meth:`parse_responses`.
        """
        return [self.build_request(export=export, input_artifact=input_artifact)]

    def parse_responses(self, responses: list[LLMResponse]) -> SchemaT:
        """Validate the run's provider responses into the structured model.

        The default validates the single response of the Task-12 flow via
        :meth:`parse_result`. Chunked pipelines override this to parse every
        response and merge the results in chunk order while enforcing
        completeness against the chunk inputs.
        """
        if len(responses) != 1:
            raise PipelineOutputError(
                "This pipeline expected exactly one LLM response for the run."
            )
        return self.parse_result(responses[0])

    def render_artifact(self, structured: SchemaT, *, export: WeeklyExport, version: int) -> str:
        """Render the artifact with run context (used when
        ``render_requires_context`` is ``True``).

        Receives the source export row and the derived version ``vM`` so the
        rendered header can embed the lineage (week, sequence, revision,
        version and the original file link). Pipelines that leave
        ``render_requires_context`` unset keep the plain :meth:`render` flow
        and never hit this method.
        """
        raise NotImplementedError(
            "This pipeline does not support context-aware artifact rendering."
        )


__all__ = [
    "Pipeline",
    "PipelineError",
    "PipelineExportNotFoundError",
    "PipelineInputError",
    "PipelineNotRegisteredError",
    "PipelineOriginalFileMissingError",
    "PipelineOutputError",
    "PipelinePermanentError",
    "PipelineStorageError",
]
