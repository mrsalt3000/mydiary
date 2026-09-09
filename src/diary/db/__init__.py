"""SQLite persistence layer for the Local Diary application (Task 04).

Public surface: the declarative base with its naming convention and UTC time
handling (``diary.db.base``), plus engine/session factories (``diary.db.session``).
ORM models live in ``diary.db.models`` and repositories in
``diary.db.repositories`` and are imported explicitly where needed.
"""

from diary.db.base import Base, UTCDateTime, utc_now
from diary.db.session import (
    DEFAULT_DB_FILENAME,
    create_engine,
    create_session_factory,
    database_url,
    default_database_path,
)

__all__ = [
    "Base",
    "DEFAULT_DB_FILENAME",
    "UTCDateTime",
    "create_engine",
    "create_session_factory",
    "database_url",
    "default_database_path",
    "utc_now",
]
