"""Alembic environment for Local Diary (Task 04).

The migration target is ``Base.metadata`` (``diary.db.base``) with every table
registered by importing ``diary.db.models``. The database URL resolution order:

1. ``sqlalchemy.url`` set on the Alembic ``Config`` (e.g. by ``alembic.ini`` or
   ``config.set_main_option(...)`` in tests);
2. the ``ALEMBIC_DATABASE_URL`` / ``TEST_DATABASE_URL`` environment variables;
3. ``AppSettings`` (``APP_DB_DIR``, default ``data/db/diary.db``).

Migrations contain no user data and no secrets (AGENTS.md §9).
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from diary.core.settings import get_settings
from diary.db import models  # noqa: F401  (importing registers tables on Base.metadata)
from diary.db.base import Base
from diary.db.session import database_url

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

#: Environment variables that override the database URL for special runs/tests.
_URL_OVERRIDE_VARS = ("ALEMBIC_DATABASE_URL", "TEST_DATABASE_URL")


def _database_url() -> str:
    """Resolve the SQLite URL following the precedence documented above."""
    configured = (config.get_main_option("sqlalchemy.url") or "").strip()
    if configured:
        return configured
    for variable in _URL_OVERRIDE_VARS:
        override = os.environ.get(variable, "").strip()
        if override:
            return override
    return database_url(get_settings())


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a DB connection)."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against the resolved database."""
    configuration = {"sqlalchemy.url": _database_url()}
    connectable = engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)
    try:
        with connectable.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
