"""Local Diary application entrypoint.

Task 01: minimal FastAPI skeleton with a ``/health`` endpoint.
Task 05: the versioned messages API (``/api/v1``) is mounted; validation
errors (HTTP 422) are rendered without echoing submitted values so diary
content never leaks through error responses (AGENTS.md §8).
Task 06: the chat UI is served from ``src/diary/templates``; the page router
(``/`` and its fragments) is mounted at the root and local static assets
(``src/diary/static``) are served from ``/static``. No template or asset
references an external host (privacy rule).

Logging note: HTTP request bodies are never logged. Uvicorn's access log
(the runner used by ``make dev``) logs only the request line and status
code; this module adds no middleware or handler that captures payloads.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from diary.api.router import api_router
from diary.api.routes.pages import STATIC_DIR, pages_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title="Local Diary")
app.include_router(api_router)
app.include_router(pages_router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.exception_handler(RequestValidationError)
async def _request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return 422 details without echoing submitted values (privacy rule).

    Pydantic errors can carry the offending ``input``; dropping it (and any
    context) keeps diary content and large payloads out of error responses.
    """
    errors = [
        {
            "loc": list(error.get("loc", [])),
            "msg": error.get("msg", "Invalid request."),
            "type": error.get("type", "value_error"),
        }
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health() -> dict[str, str]:
    """Return a simple JSON health payload."""
    return {"status": "ok"}
