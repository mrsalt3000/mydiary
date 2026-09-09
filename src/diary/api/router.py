"""Root router for the versioned HTTP API (``/api/v1``, Task 05).

New route groups are included here as they are implemented by later tasks.
"""

from __future__ import annotations

from fastapi import APIRouter

from diary.api.routes import audio, exports, jobs, messages, pipelines

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(messages.router)
api_router.include_router(audio.router)
api_router.include_router(jobs.router)
api_router.include_router(exports.router)
api_router.include_router(pipelines.router)
api_router.include_router(pipelines.runs_router)
