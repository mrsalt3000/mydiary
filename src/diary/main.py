"""Local Diary application entrypoint.

Task 01: minimal FastAPI skeleton with a ``/health`` endpoint.

Logging note: HTTP request bodies are never logged. Uvicorn's access log
(the runner used by ``make dev``) logs only the request line and status
code; this module adds no middleware or handler that captures payloads.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title="Local Diary")


@app.get("/health")
def health() -> dict[str, str]:
    """Return a simple JSON health payload."""
    return {"status": "ok"}
