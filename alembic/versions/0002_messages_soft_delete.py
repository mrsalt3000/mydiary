"""Soft-delete support: nullable ``messages.deleted_at``

Revision ID: 0002_messages_soft_delete
Revises: 0001_initial
Create Date: 2026-09-09

Adds the nullable UTC ``deleted_at`` timestamp to ``messages`` (deletion
policy fixed in Task 05: soft delete only — the row stays in the database for
traceability, the body is never overwritten, and soft-deleted messages are
hidden from the history by default). No index is required: history queries
are served by the existing ``ix_messages_created_at`` index plus a cheap
``deleted_at IS NULL`` filter.

Migration policy (AGENTS.md §9): schema changes go through Alembic only;
migrations are repeatable on a fresh database and never contain user data or
secrets.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_messages_soft_delete"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the schema change."""
    op.add_column(
        "messages",
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    """Revert the schema change."""
    # SQLite supports ALTER TABLE DROP COLUMN only through table recreation,
    # which Alembic performs in batch mode.
    with op.batch_alter_table("messages") as batch_op:
        batch_op.drop_column("deleted_at")
