"""Error taxonomy of the transcription provider layer (Task 09; retry policy T44).

Leaf module on purpose: it imports nothing but the standard library, so the
provider adapters, the factory and the worker boundary can all share one
taxonomy without importing each other (``providers`` -> ``gigaam`` -> ``errors``
is a DAG, not a cycle) — and so the sidecar worker script, which runs in a venv
that has neither pydantic nor sqlalchemy, can be reasoned about against the same
contract.

Retry policy lives **on the class**, exactly as in
:mod:`diary.domain.llm.errors`:

* ``retryable = False`` — a deterministic failure: the same configuration,
  the same file and the same machine produce it again, so retrying only burns
  the job's attempt budget. This is the **default**, so a new error class can
  never silently become transient: the T43 acceptance found precisely that
  defect (the factory's refusal was retried although no retry could help);
* ``retryable = True`` — a transient failure: the identical request may
  succeed later (decoder/model runtime failure, timeout).

The worker boundary (``diary.workers.handlers``) consults the attribute instead
of listing classes, so the mapping cannot drift out of date.

Privacy (AGENTS.md §7–8): every message is a static, non-sensitive summary. No
diary content, audio path, decoder payload or provider stderr ever appears in
``str()``/``repr()``/``args``. Where a third-party class name is useful for
diagnosis, only the *name* is echoed — never the original message.
"""

from __future__ import annotations

from typing import ClassVar


class TranscriptionProviderError(Exception):
    """Base class for transcription-provider errors.

    Messages are always static and never contain diary content or secrets;
    ``retryable`` (see the module docstring) defaults to ``False``.
    """

    #: Whether a later attempt with the same configuration may succeed.
    retryable: ClassVar[bool] = False


class ProviderConfigurationError(TranscriptionProviderError):
    """The configured backend cannot work at all; retrying cannot fix it.

    Raised for conditions the configuration (or the machine) already decides:
    the ``provider`` tag has no registered adapter, a profile switch is not
    supported by the chosen adapter (``unload_after: false`` for the GigaAM
    sidecar), or the profile names a mode the adapter cannot serve. It is the
    non-retryable half of the taxonomy, and
    :class:`ProviderUnavailableError` is the "runtime/weights are not here"
    specialisation of it.
    """


class ProviderUnavailableError(ProviderConfigurationError):
    """The configured provider runtime is not installed or not provisioned.

    Raised *before* any model is constructed, downloaded or loaded: for
    faster-whisper when the optional package is missing, for the GigaAM sidecar
    when the configured interpreter/worker script does not exist or its weights
    are not in the local cache. Retrying would never help without an explicit
    provisioning step.
    """


class TranscriptionExecutionError(TranscriptionProviderError):
    """The provider failed at runtime; the message is a sanitised summary.

    Only the exception *class name* is echoed — never the original message,
    which could carry audio/decoder internals. Transient by contract: the
    worker's retry policy applies.
    """

    retryable: ClassVar[bool] = True


__all__ = [
    "ProviderConfigurationError",
    "ProviderUnavailableError",
    "TranscriptionExecutionError",
    "TranscriptionProviderError",
]
