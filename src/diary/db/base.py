"""Shared SQLAlchemy foundations for the Local Diary persistence layer (Task 04).

Defines the declarative base, the metadata-level naming convention for
constraints/indexes, and the UTC-aware datetime type used by every timestamp
column. The naming convention guarantees deterministic, stable constraint and
index names in migrations and in the live SQLite schema (see ``sqlite-data-model``
skill: stable IDs, consistent timezone-aware timestamps, foreign keys,
check-constrained statuses).

All timestamps are stored as UTC. SQLite has no timezone-aware datetime type,
so :class:`UTCDateTime` stores an ordinary ``DATETIME`` text value in UTC and
re-attaches ``timezone.utc`` on read.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, MetaData
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator

#: Metadata-level naming convention: constraints and indexes created through the
#: ORM receive deterministic names (``ix_<table>_<column>``, ``uq_...``,
#: ``ck_...``, ``fk_...``, ``pk_<table>``).
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for every diary table."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC :class:`datetime`."""
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware :class:`datetime` persisted as naive UTC text.

    ``process_bind_param`` converts any aware value to UTC and strips the
    ``tzinfo`` before storage; ``process_result_value`` re-attaches
    ``timezone.utc``. Naive values are rejected on bind so callers can never
    accidentally store a non-UTC timestamp.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"UTCDateTime expects datetime, got {type(value).__name__}")
        if value.tzinfo is None:
            raise ValueError(
                "UTCDateTime requires a timezone-aware datetime; use diary.db.base.utc_now"
            )
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.replace(tzinfo=timezone.utc)
        raise TypeError(
            f"UTCDateTime expected datetime from the driver, got {type(value).__name__}"
        )
