"""Explicit LLM unload for the sequential STT turn (task T11, ADR-010).

The phase-2 veto is «Whisper and the LLM are never resident in VRAM at the same
time» (``docs/tasklist-phase2.md`` §2). Since T03 the order of an STT turn is
enforced structurally by :func:`diary.domain.transcription.asr.run`, which
receives an **injected** ``unload_llm`` callable — the ASR module deliberately
knows nothing about Ollama or HTTP. This module is the production side of that
injection: it builds the callable from the model configuration.

What "unload" means for a local Ollama server
============================================

Ollama keeps a model resident until its ``keep_alive`` expires (or forever with
a ``keep_alive`` of ``-1``). The documented way to evict it immediately is the
``keep_alive: 0`` request, which is exactly what the ``ollama stop <model>``
command (and ``scripts/serve-ollama.sh unload <model>``) does server-side::

    POST <native-root>/api/generate   {"model": "<tag>", "keep_alive": 0}

The OpenAI-compatible ``/v1`` surface does **not** carry ``keep_alive``, so the
request goes to the server's native API. The native root is derived from the
profile's ``base_url`` by keeping only ``scheme://host:port``
(``http://127.0.0.1:11434/v1`` → ``http://127.0.0.1:11434``).

Which profiles are unloaded
===========================

Every configured ``openai_compatible`` profile whose endpoint is **loopback**
(``127.0.0.1``/``localhost``/``::1``) — the MVP profiles and the optional
phase-2 profiles alike. An external endpoint has nothing local to evict and is
skipped: the veto is about this machine's VRAM, and diary content never leaves
the machine in this step (no request body carries content).

Failure policy (fail-safe on "no server", fail-closed once it answers)
======================================================================

``asr.run`` treats a **raised** unload failure as fatal (it refuses to start
STT). Only a model that *provably* cannot be resident may be reported as "not
there":

* the connection cannot be **established** (``ConnectError`` /
  ``ConnectTimeout``: nothing is listening) → success, logged as
  ``result=unreachable`` — no process holds the model, so nothing occupies
  VRAM;
* the server answers ``4xx`` → success (an unknown model or an endpoint without
  the unload API: again nothing is resident under that name);
* a connection **was established** but the unload is not confirmed — a
  transport timeout *after* connecting (``ReadTimeout`` / ``WriteTimeout`` /
  ``PoolTimeout``), any other transport failure after connecting, or the server
  answers ``5xx`` or any other status → :class:`LlmUnloadFailedError`. A server
  that answers but does not confirm the unload must not be passed off as "no
  model": the behaviour matches the ``5xx`` rule, and a retry is safe because
  speech-to-text has not been started yet.

Privacy (AGENTS.md §7-8): only model names and endpoints are logged — both are
configuration values. No prompt, no completion and no request body ever leaves
this module.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from diary.core.models_config import (
    ModelsConfig,
    OpenAICompatibleProfile,
    load_models_config,
)

logger = logging.getLogger("diary.domain.llm.unload")

#: Hosts treated as "this machine" (the VRAM veto only concerns local models).
LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})

#: Transport timeout of one unload request (a server that is up answers in ms).
DEFAULT_UNLOAD_TIMEOUT_S = 10.0

#: ``keep_alive`` value that evicts a model immediately (== ``ollama stop``).
KEEP_ALIVE_EVICT = 0


class LlmUnloadFailedError(Exception):
    """A reachable local server did not confirm the model unload.

    The message names only the endpoint and the status class — never a payload
    (AGENTS.md §8). :func:`diary.domain.transcription.asr.run` turns it into a
    loud :class:`~diary.domain.transcription.asr.LlmUnloadError` and starts no
    speech-to-text, which is the safe direction for the VRAM veto.
    """


@dataclass(frozen=True)
class LocalLlmEndpoint:
    """One local model that can hold VRAM: its native root URL and model tag."""

    #: ``scheme://host:port`` of the server (native API, no ``/v1``).
    native_root: str
    #: Model tag/id as named by the profile (``diary-stt``, ``diary-sum``, ...).
    model: str


def is_loopback_base_url(base_url: str) -> bool:
    """Whether ``base_url`` points at this machine."""
    hostname = urlsplit(base_url).hostname
    return hostname is not None and hostname in LOOPBACK_HOSTS


def native_root_of(base_url: str) -> str:
    """``scheme://host[:port]`` of ``base_url`` (the server's native API root).

    ``http://127.0.0.1:11434/v1`` → ``http://127.0.0.1:11434``. Any path is
    dropped: ``keep_alive`` lives on ``/api/generate``, not below the
    OpenAI-compatible ``/v1`` prefix.
    """
    parts = urlsplit(base_url)
    netloc = parts.hostname or ""
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    return f"{parts.scheme}://{netloc}"


def _llm_profiles(models_config: ModelsConfig) -> list[OpenAICompatibleProfile]:
    """Every configured ``openai_compatible`` profile (MVP + phase 2)."""
    profiles: list[OpenAICompatibleProfile] = [
        profile
        for _task, profile in models_config.profiles_by_task()
        if isinstance(profile, OpenAICompatibleProfile)
    ]
    profiles.extend(profile for _name, profile in models_config.phase2_profiles())
    return profiles


def local_llm_endpoints(models_config: ModelsConfig) -> tuple[LocalLlmEndpoint, ...]:
    """The local models a manual STT turn has to evict, de-duplicated in order.

    Only loopback endpoints are returned (an external endpoint holds nothing in
    this machine's VRAM), and a model named by several profiles (``diary-sum``
    is both the summarization profile and ``diary_sum``) appears once.
    """
    endpoints: list[LocalLlmEndpoint] = []
    seen: set[tuple[str, str]] = set()
    for profile in _llm_profiles(models_config):
        if not is_loopback_base_url(profile.base_url):
            continue
        key = (native_root_of(profile.base_url), profile.model)
        if key in seen:
            continue
        seen.add(key)
        endpoints.append(LocalLlmEndpoint(native_root=key[0], model=key[1]))
    return tuple(endpoints)


def unload_model(
    endpoint: LocalLlmEndpoint,
    *,
    transport: httpx.BaseTransport | None = None,
    timeout_s: float = DEFAULT_UNLOAD_TIMEOUT_S,
) -> bool:
    """Evict one local model; return whether it is known to be gone.

    ``True`` means the server confirmed the unload request (``2xx``); ``False``
    means the model or the unload API is not there at all (``4xx``) or the
    connection could not be established (nothing is listening, so the model
    cannot be resident). A server that answers but does not confirm the unload
    — ``5xx``, a transport timeout after the connection was established, or any
    other failure after connecting — raises :class:`LlmUnloadFailedError`, so
    the caller never starts STT on an unconfirmed unload.

    ``transport`` is the test seam (``httpx.MockTransport``); production passes
    ``None`` and HTTPX uses its real network transport.
    """
    url = f"{endpoint.native_root}/api/generate"
    payload = {"model": endpoint.model, "keep_alive": KEEP_ALIVE_EVICT}
    try:
        with httpx.Client(transport=transport, timeout=httpx.Timeout(timeout_s)) as client:
            response = client.post(url, json=payload)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        # The connection was never established: nothing is listening on that
        # endpoint, so no process can hold the model.
        logger.info(
            "llm-unload endpoint=%s model=%s result=unreachable",
            endpoint.native_root,
            endpoint.model,
        )
        return False
    except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
        # The connection was established but the server did not confirm the
        # unload in time: a reachable server is not "no model" (fail-closed).
        raise LlmUnloadFailedError(
            f"The local LLM server at {endpoint.native_root} did not confirm the "
            f"unload of {endpoint.model!r} within the timeout."
        ) from exc
    except httpx.TransportError as exc:
        # Any other transport failure happened after the connection was
        # established (for example a broken protocol): the unload is still
        # unconfirmed, so it must not be reported as "no model".
        raise LlmUnloadFailedError(
            f"The local LLM server at {endpoint.native_root} did not confirm the "
            f"unload of {endpoint.model!r}: the connection failed after it was "
            "established."
        ) from exc
    if 200 <= response.status_code < 300:
        logger.info(
            "llm-unload endpoint=%s model=%s result=unloaded",
            endpoint.native_root,
            endpoint.model,
        )
        return True
    if 400 <= response.status_code < 500:
        # Unknown model, or a server whose API has no unload request: either
        # way no model under this name is resident here.
        logger.info(
            "llm-unload endpoint=%s model=%s result=absent status=%d",
            endpoint.native_root,
            endpoint.model,
            response.status_code,
        )
        return False
    raise LlmUnloadFailedError(
        f"The local LLM server at {endpoint.native_root} answered HTTP "
        f"{response.status_code} for the unload of {endpoint.model!r}, so the "
        "unload cannot be confirmed."
    )


def unload_local_llms(
    models_config: ModelsConfig | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    timeout_s: float = DEFAULT_UNLOAD_TIMEOUT_S,
) -> None:
    """Evict every local LLM the configuration names — the unload itself.

    With ``models_config=None`` the configuration is read here and now, so a
    long-lived worker always unloads what its current configuration describes.
    A configuration that cannot be loaded raises: an unload that cannot even be
    built must not be skipped silently (the veto would not hold). No local LLM
    profile (nothing can be resident in this machine's VRAM) is a logged no-op.
    """
    config = models_config if models_config is not None else load_models_config()
    endpoints = local_llm_endpoints(config)
    if not endpoints:
        logger.info("llm-unload no local LLM profile configured")
        return
    for endpoint in endpoints:
        unload_model(endpoint, transport=transport, timeout_s=timeout_s)


def build_llm_unloader(
    models_config: ModelsConfig | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    timeout_s: float = DEFAULT_UNLOAD_TIMEOUT_S,
) -> Callable[[], None]:
    """The ``unload_llm`` callable :func:`~diary.domain.transcription.asr.run` takes.

    Returns a synchronous callable that evicts every local LLM the configuration
    names (see :func:`unload_local_llms`); the configuration is resolved per
    call when ``models_config`` is ``None``.
    """

    def unload_llm() -> None:
        unload_local_llms(models_config, transport=transport, timeout_s=timeout_s)

    return unload_llm


__all__ = [
    "DEFAULT_UNLOAD_TIMEOUT_S",
    "KEEP_ALIVE_EVICT",
    "LOOPBACK_HOSTS",
    "LlmUnloadFailedError",
    "LocalLlmEndpoint",
    "build_llm_unloader",
    "is_loopback_base_url",
    "local_llm_endpoints",
    "native_root_of",
    "unload_local_llms",
    "unload_model",
]
