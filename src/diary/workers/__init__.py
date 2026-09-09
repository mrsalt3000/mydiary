"""Durable SQLite-backed worker (Task 08, ADR-003).

Submodules:

- ``handlers`` — the job-handler protocol and registry (a handler is a
  ``(session, job) -> None`` callable registered per ``JobType``);
- ``runner`` — the worker process: claims due ``jobs``, executes their
  handlers, applies the retry/backoff policy and writes safe status
  transitions. Run with ``python -m diary.workers.runner --help``.
"""
