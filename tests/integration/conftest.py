"""Shared fixtures for integration tests (Task 04).

Every integration test runs against its own isolated SQLite database file in a
pytest ``tmp_path`` — never against the production ``data/db/diary.db``. The
database is created purely by running the Alembic migrations, which is the only
supported way to create a fresh schema (AGENTS.md §9).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import Engine

from diary.db.session import create_engine, create_session_factory

#: Repository root: tests/integration/conftest.py -> parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = REPO_ROOT / "alembic.ini"
ALEMBIC_DIR = REPO_ROOT / "alembic"


def make_alembic_config(database_url: str) -> AlembicConfig:
    """An Alembic ``Config`` pointed at ``database_url`` and the real scripts.

    The config is built without a ``config_file_name`` so ``alembic/env.py``
    skips ``fileConfig(alembic.ini)``: in-process migrations must not
    reconfigure the root logger or disable pre-existing ``diary.*`` loggers
    (which would break caplog-based log assertions in the Task-19 tests).
    """
    config = AlembicConfig()
    config.set_main_option("script_location", str(ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def upgrade_database(database_url: str) -> None:
    """Upgrade ``database_url`` to head (in-process Alembic run)."""
    command.upgrade(make_alembic_config(database_url), "head")


@pytest.fixture
def run_upgrade():
    """Fixture returning :func:`upgrade_database` bound to this test run."""
    return upgrade_database


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A private SQLite file path for one test."""
    return tmp_path / "diary-test.db"


@pytest.fixture
def db_url(db_path: Path) -> str:
    """SQLAlchemy URL of the test database file."""
    return f"sqlite:///{db_path.as_posix()}"


@pytest.fixture
def migrated_db_url(db_url: str) -> str:
    """A fresh test database created solely through Alembic migrations."""
    upgrade_database(db_url)
    return db_url


@pytest.fixture
def engine(migrated_db_url: str) -> Iterator[Engine]:
    """Engine over the migrated test database (diary pragmas applied)."""
    test_engine = create_engine(migrated_db_url)
    try:
        yield test_engine
    finally:
        test_engine.dispose()


@pytest.fixture
def session_factory(engine: Engine):
    """Session factory bound to the migrated test database."""
    return create_session_factory(engine)


@pytest.fixture
def session(session_factory):
    """One session per test; the caller owns commits and rollbacks."""
    with session_factory() as current_session:
        yield current_session
