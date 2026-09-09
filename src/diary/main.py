"""Local Diary application entrypoint.

Task 01: minimal FastAPI skeleton with a ``/health`` endpoint.
Task 05: the versioned messages API (``/api/v1``) is mounted; validation
errors (HTTP 422) are rendered without echoing submitted values so diary
content never leaks through error responses (AGENTS.md §8).
Task 06: the chat UI is served from ``src/diary/templates``; the page router
(``/`` and its fragments) is mounted at the root and local static assets
(``src/diary/static``) are served from ``/static``. No template or asset
references an external host (privacy rule).
Task 19: ``GET /ready`` reports SQLite reachability; a request-id middleware
and the shared log format (``diary.core.logging_config``) tag every request.

Logging note: HTTP request bodies are never logged. The request-id middleware
logs only method/path/status/duration; the access log of the runner used by
``make dev`` logs only the request line and status code. No middleware or
handler captures payloads.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from diary.api.middleware import RequestContextMiddleware
from diary.api.router import api_router
from diary.api.routes.messages import get_session
from diary.api.routes.pages import STATIC_DIR, pages_router
from diary.core.logging_config import configure_logging

configure_logging()
logger = logging.getLogger("diary.main")

app = FastAPI(title="Local Diary")
app.add_middleware(RequestContextMiddleware)
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


@app.get("/ready")
def ready(session: Annotated[Session, Depends(get_session)]) -> JSONResponse:
    """Liveness/readiness probe: verify the SQLite database is reachable.

    Runs a lightweight ``SELECT 1`` on the application database. Returns
    ``200 {"status": "ready"}`` when the database answers and ``503
    {"status": "unavailable"}`` when it does not. The check covers database
    *reachability* only — it does not verify that migrations are applied or
    that every dependency (e.g. a configured STT/LLM provider) is installed;
    those states surface through the jobs and the privacy page instead.
    """
    try:
        session.execute(text("SELECT 1"))
    except SQLAlchemyError:
        logger.warning("ready check failed: the database is unreachable.")
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    return JSONResponse(content={"status": "ready"})
