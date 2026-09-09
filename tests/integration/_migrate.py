"""Shared helper for backup/recovery tests: create a migrated SQLite file.

Not collected by pytest (leading underscore). The integration conftest is
not importable as ``tests.integration.conftest`` under pytest's import mode,
so the backup tests run Alembic directly against a per-test database file.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config as AlembicConfig

#: Repository root: tests/integration/_migrate.py -> parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]


def upgrade_database_file(db_path: Path) -> None:
    """Upgrade ``db_path`` to head (in-process Alembic run)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # No ``config_file_name`` on purpose: ``alembic/env.py`` must skip
    # ``fileConfig(alembic.ini)`` so in-process migrations never reconfigure
    # the root logger or disable ``diary.*`` loggers (caplog relies on them).
    config = AlembicConfig()
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path.as_posix()}")
    command.upgrade(config, "head")
