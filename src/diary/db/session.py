"""Engine and session factory for the SQLite database (Task 04).

The database file lives under ``AppSettings.db_dir`` (default
``<data_root>/db/diary.db``). For local-first single-user SQLite shared by the
web process and the worker process we enable:

- ``PRAGMA foreign_keys=ON`` — SQLite does not enforce foreign keys by default;
- ``PRAGMA journal_mode=WAL`` — concurrent readers/writer across processes;
- ``PRAGMA busy_timeout`` — bounded waiting instead of immediate "database is
  locked" errors;
- ``PRAGMA synchronous=NORMAL`` — the recommended WAL durability setting.

The database URL can be overridden explicitly (``url=...``) or by pointing the
settings at a different ``APP_DB_DIR`` — integration tests pass a temporary
SQLite file URL and never touch the production database.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, event
from sqlalchemy import create_engine as _sa_create_engine
from sqlalchemy.orm import Session, sessionmaker

from diary.core.settings import AppSettings, get_settings

#: File name of the application database inside ``AppSettings.db_dir``.
DEFAULT_DB_FILENAME = "diary.db"

#: Pragma statements applied to every new SQLite connection.
_SQLITE_PRAGMAS: tuple[tuple[str, str], ...] = (
    ("foreign_keys", "ON"),
    ("journal_mode", "WAL"),
    ("busy_timeout", "5000"),
    ("synchronous", "NORMAL"),
)


def default_database_path(settings: AppSettings | None = None) -> Path:
    """Absolute path of the default SQLite database file."""
    return (settings or get_settings()).db_dir / DEFAULT_DB_FILENAME


def database_url(settings: AppSettings | None = None) -> str:
    """SQLAlchemy URL for the default database, creating its directory if needed."""
    resolved = settings or get_settings()
    path = resolved.db_dir / DEFAULT_DB_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def create_engine(url: str | None = None, *, settings: AppSettings | None = None) -> Engine:
    """Create a SQLite engine with diary-specific connection settings.

    ``url`` wins when given; otherwise the URL is derived from ``settings``
    (or the cached application settings when ``settings`` is ``None``).
    """
    engine_url = url if url is not None else database_url(settings)
    engine = _sa_create_engine(
        engine_url,
        connect_args={"check_same_thread": False},
        future=True,
    )
    event.listen(engine, "connect", _configure_sqlite_connection)
    return engine


def _configure_sqlite_connection(dbapi_connection: object, connection_record: object) -> None:
    """Apply per-connection SQLite pragmas (see module docstring)."""
    # SQLAlchemy's pysqlite dialect exposes a PEP-249 cursor factory.
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    try:
        for pragma, value in _SQLITE_PRAGMAS:
            cursor.execute(f"PRAGMA {pragma}={value}")
    finally:
        cursor.close()


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a ``sessionmaker`` bound to ``engine`` for use by repositories."""
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False, future=True)


__all__ = [
    "DEFAULT_DB_FILENAME",
    "create_engine",
    "create_session_factory",
    "database_url",
    "default_database_path",
]
