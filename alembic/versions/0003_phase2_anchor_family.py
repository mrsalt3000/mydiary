"""Second anchor family of the phase-2 derivational layers (task T08)

Revision ID: 0003_phase2_anchor_family
Revises: 0002_messages_soft_delete
Create Date: 2026-09-19

Implements variant B of ADR-007 (persistence shape: ADR-008) — the deltas
needed so the results of the phase-2 layers can be stored, in ``artifacts``,
without a weekly export:

* ``artifacts.export_id`` becomes **nullable**: a second-family row is anchored
  directly to its source instead of a weekly export (ADR-007 §10 explicitly
  deferred this to T08);
* four **anchor columns** are added — ``raw_id`` (message, layer
  ``stt_corrected``), ``local_date`` + ``local_tz`` (local calendar day, layer
  ``daily_summary``), ``week_id`` (ISO week ``YYYY-Www``, layer
  ``weekly_summary``) and ``summary_id`` (upstream weekly-summary artifact,
  layer ``coach_note``) — plus ``result_json``, which stores the validated
  structured result of the layer (the AI output);
* ``path``/``sha256`` become nullable: a second-family result may be stored
  inline (no rendered file);
* the schema enforces the invariants itself: ``ck_artifacts_anchor_family``
  (exactly one anchor family is set, the layer matches its anchor, and a
  second-family row always carries its ``vM`` version),
  ``ck_artifacts_anchor_local_tz`` (a day anchor always records its timezone)
  and ``ck_artifacts_anchor_payload`` (a first-family artifact keeps its
  previous shape — file plus hash, no inline result — while a second-family
  result stores a file and/or an inline result); five
  ``uq_artifacts_*_layer_version`` constraints make append-only versioning a
  schema guarantee (a repeated run appends ``v+1`` and can never overwrite a
  previous result);
* the value sets of the three enums grow: ``artifact_layer`` (phase-2 layers
  incl. the new layer 5 ``coach_note``), ``pipeline_type`` and ``job_type``.
  On SQLite these enums are ``VARCHAR`` columns with a ``CHECK`` constraint,
  and a ``CHECK`` cannot be altered in place — hence ``batch_alter_table``
  (table recreation) for all three tables.

Constraint names and the metadata naming convention
===================================================

``alembic/env.py`` sets ``target_metadata = Base.metadata``, whose naming
convention declares ``ck`` as ``ck_%(table_name)s_%(constraint_name)s``.
Alembic builds the constraint objects of ``create_check_constraint`` /
``drop_constraint`` through that convention
(``alembic.operations.schemaobj.SchemaObjects``): the argument is the
``%(constraint_name)s`` token, and the stored name becomes
``ck_<table>_<token>``. Passing an already prefixed name would store a doubled
``ck_artifacts_ck_artifacts_...`` name (verified empirically against a copy of
the real database), so this revision passes bare tokens — exactly what the
model declares (``CheckConstraint(..., name="anchor_family")``) and what
migration ``0001`` produced for the enum checks. The resulting names are the
ones asserted by ``tests/integration/test_migrations_phase2.py``.

Data safety: the migration only widens the schema. Every row written before it
still satisfies the new constraints (a first-family artifact has ``export_id``
set, no anchor, and keeps its ``path``/``sha256``), no column is dropped, no
value is rewritten and no user data or secret is contained here. Fresh
databases are still created by migrations alone (AGENTS.md §9).

The ``downgrade`` refuses to run while second-family rows exist: restoring
``export_id NOT NULL`` and the old layer ``CHECK`` would have to delete or
rewrite real results, which no migration may do silently.

Privacy: schema only — no diary content, no secrets (AGENTS.md §7-8).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_phase2_anchor_family"
down_revision: str | None = "0002_messages_soft_delete"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# ---------------------------------------------------------------------------
# Frozen SQL and names of the constraints touched by this revision. The SQL is
# duplicated from the model constants on purpose: a later model change must not
# silently rewrite an already applied revision.
# ---------------------------------------------------------------------------

#: Layer values of the first (export-centric) anchor family.
_EXPORT_LAYERS = "'original', 'normalized', 'summary', 'wellbeing'"
#: Layer values of the second (phase-2) anchor family.
_SECOND_FAMILY_LAYERS = "'stt_corrected', 'daily_summary', 'weekly_summary', 'coach_note'"

_ARTIFACT_LAYER_CHECK = f"layer IN ({_EXPORT_LAYERS}, {_SECOND_FAMILY_LAYERS})"

_ANCHOR_FAMILY_CHECK = (
    f"(export_id IS NOT NULL AND layer IN ({_EXPORT_LAYERS})"
    " AND raw_id IS NULL AND local_date IS NULL AND week_id IS NULL AND summary_id IS NULL)"
    " OR (export_id IS NULL AND version IS NOT NULL"
    " AND (CASE WHEN raw_id IS NOT NULL THEN 1 ELSE 0 END"
    " + CASE WHEN local_date IS NOT NULL THEN 1 ELSE 0 END"
    " + CASE WHEN week_id IS NOT NULL THEN 1 ELSE 0 END"
    " + CASE WHEN summary_id IS NOT NULL THEN 1 ELSE 0 END) = 1"
    " AND ((layer = 'stt_corrected' AND raw_id IS NOT NULL)"
    " OR (layer = 'daily_summary' AND local_date IS NOT NULL)"
    " OR (layer = 'weekly_summary' AND week_id IS NOT NULL)"
    " OR (layer = 'coach_note' AND summary_id IS NOT NULL)))"
)

_ANCHOR_LOCAL_TZ_CHECK = "(local_date IS NULL) = (local_tz IS NULL)"

_ANCHOR_PAYLOAD_CHECK = (
    "(export_id IS NOT NULL AND path IS NOT NULL AND sha256 IS NOT NULL AND result_json IS NULL)"
    " OR (export_id IS NULL AND (path IS NOT NULL OR result_json IS NOT NULL))"
)

_PIPELINE_TYPE_CHECK = (
    "pipeline_type IN ('normalize', 'summarize', 'wellbeing',"
    " 'stt_corrected', 'daily_summary', 'weekly_summary', 'coach_note')"
)

_JOB_TYPE_CHECK = (
    "job_type IN ('transcribe_audio', 'normalize', 'summarize', 'wellbeing',"
    " 'stt_corrected', 'daily_summary', 'weekly_summary', 'coach_note')"
)

#: Old (pre-migration) value sets, restored by ``downgrade``.
_OLD_ARTIFACT_LAYER_CHECK = f"layer IN ({_EXPORT_LAYERS})"
_OLD_PIPELINE_TYPE_CHECK = "pipeline_type IN ('normalize', 'summarize', 'wellbeing')"
_OLD_JOB_TYPE_CHECK = "job_type IN ('transcribe_audio', 'normalize', 'summarize', 'wellbeing')"

#: Constraint names as ``%(constraint_name)s`` tokens (see module docstring):
#: the naming convention turns each of them into ``ck_<table>_<token>``.
_ARTIFACT_LAYER_CHECK_NAME = "artifact_layer"
_ANCHOR_FAMILY_CHECK_NAME = "anchor_family"
_ANCHOR_LOCAL_TZ_CHECK_NAME = "anchor_local_tz"
_ANCHOR_PAYLOAD_CHECK_NAME = "anchor_payload"
_PIPELINE_TYPE_CHECK_NAME = "pipeline_type"
_JOB_TYPE_CHECK_NAME = "job_type"

#: Second-family anchor columns added by this revision.
_NEW_COLUMNS = ("raw_id", "local_date", "local_tz", "week_id", "summary_id", "result_json")

#: New per-anchor indexes (mirroring ``ix_artifacts_export_id_layer``).
_NEW_INDEXES = (
    ("ix_artifacts_raw_id_layer", ["raw_id", "layer"]),
    ("ix_artifacts_local_date_layer", ["local_date", "layer"]),
    ("ix_artifacts_week_id_layer", ["week_id", "layer"]),
    ("ix_artifacts_summary_id_layer", ["summary_id", "layer"]),
)

#: New per-anchor version constraints (append-only guarantee).
_NEW_UNIQUE_CONSTRAINTS = (
    ("uq_artifacts_export_id_layer_version", ["export_id", "layer", "version"]),
    ("uq_artifacts_raw_id_layer_version", ["raw_id", "layer", "version"]),
    ("uq_artifacts_local_date_layer_version", ["local_date", "local_tz", "layer", "version"]),
    ("uq_artifacts_week_id_layer_version", ["week_id", "layer", "version"]),
    ("uq_artifacts_summary_id_layer_version", ["summary_id", "layer", "version"]),
)


def _second_family_row_count() -> int:
    """Number of artifact rows that exist only because of this revision."""
    connection = op.get_bind()
    return int(
        connection.execute(
            sa.text("SELECT COUNT(*) FROM artifacts WHERE export_id IS NULL")
        ).scalar_one()
    )


def upgrade() -> None:
    """Apply the schema change."""
    with op.batch_alter_table("artifacts", recreate="always") as batch_op:
        # 1. Anchor columns + inline result payload.
        batch_op.add_column(sa.Column("raw_id", sa.Uuid(), nullable=True))
        batch_op.add_column(sa.Column("local_date", sa.Date(), nullable=True))
        batch_op.add_column(sa.Column("local_tz", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("week_id", sa.String(length=8), nullable=True))
        batch_op.add_column(sa.Column("summary_id", sa.Uuid(), nullable=True))
        batch_op.add_column(sa.Column("result_json", sa.JSON(), nullable=True))
        # 2. A second-family result needs no weekly export and no file.
        batch_op.alter_column("export_id", existing_type=sa.Uuid(), nullable=True)
        batch_op.alter_column("path", existing_type=sa.String(length=512), nullable=True)
        batch_op.alter_column("sha256", existing_type=sa.String(length=64), nullable=True)
        # 3. Widen the layer enum: a CHECK cannot be altered in place, so the
        #    old one is dropped and re-created with the phase-2 layer values.
        batch_op.drop_constraint(_ARTIFACT_LAYER_CHECK_NAME, type_="check")
        batch_op.create_check_constraint(_ARTIFACT_LAYER_CHECK_NAME, _ARTIFACT_LAYER_CHECK)
        # 4. Anchor foreign keys (the message / the upstream summary artifact).
        batch_op.create_foreign_key("fk_artifacts_raw_id_messages", "messages", ["raw_id"], ["id"])
        batch_op.create_foreign_key(
            "fk_artifacts_summary_id_artifacts", "artifacts", ["summary_id"], ["id"]
        )
        # 5. Schema-level invariants of the two anchor families.
        batch_op.create_check_constraint(_ANCHOR_FAMILY_CHECK_NAME, _ANCHOR_FAMILY_CHECK)
        batch_op.create_check_constraint(_ANCHOR_LOCAL_TZ_CHECK_NAME, _ANCHOR_LOCAL_TZ_CHECK)
        batch_op.create_check_constraint(_ANCHOR_PAYLOAD_CHECK_NAME, _ANCHOR_PAYLOAD_CHECK)
        # 6. Lookup indexes and the append-only version guard per anchor.
        for index_name, columns in _NEW_INDEXES:
            batch_op.create_index(index_name, columns, unique=False)
        for constraint_name, columns in _NEW_UNIQUE_CONSTRAINTS:
            batch_op.create_unique_constraint(constraint_name, columns)

    # 7. The phase-2 pipelines/jobs need their own enum values; same SQLite
    #    CHECK-constraint limitation as above.
    with op.batch_alter_table("processing_runs", recreate="always") as batch_op:
        batch_op.drop_constraint(_PIPELINE_TYPE_CHECK_NAME, type_="check")
        batch_op.create_check_constraint(_PIPELINE_TYPE_CHECK_NAME, _PIPELINE_TYPE_CHECK)

    with op.batch_alter_table("jobs", recreate="always") as batch_op:
        batch_op.drop_constraint(_JOB_TYPE_CHECK_NAME, type_="check")
        batch_op.create_check_constraint(_JOB_TYPE_CHECK_NAME, _JOB_TYPE_CHECK)


def downgrade() -> None:
    """Revert the schema change.

    Refuses to run while second-family rows exist: the previous schema cannot
    represent them (``export_id NOT NULL`` plus the four-layer ``CHECK``), and
    dropping or rewriting real results is not something a migration may do
    silently (AGENTS.md §7, §14).
    """
    remaining = _second_family_row_count()
    if remaining:
        raise RuntimeError(
            "Downgrade 0003 refused: "
            f"{remaining} artifact row(s) belong to the second anchor family and cannot be "
            "represented by the previous schema. Re-run after exporting/removing them "
            "explicitly."
        )

    with op.batch_alter_table("jobs", recreate="always") as batch_op:
        batch_op.drop_constraint(_JOB_TYPE_CHECK_NAME, type_="check")
        batch_op.create_check_constraint(_JOB_TYPE_CHECK_NAME, _OLD_JOB_TYPE_CHECK)

    with op.batch_alter_table("processing_runs", recreate="always") as batch_op:
        batch_op.drop_constraint(_PIPELINE_TYPE_CHECK_NAME, type_="check")
        batch_op.create_check_constraint(_PIPELINE_TYPE_CHECK_NAME, _OLD_PIPELINE_TYPE_CHECK)

    with op.batch_alter_table("artifacts", recreate="always") as batch_op:
        for constraint_name, _columns in _NEW_UNIQUE_CONSTRAINTS:
            batch_op.drop_constraint(constraint_name, type_="unique")
        for index_name, _columns in _NEW_INDEXES:
            batch_op.drop_index(index_name)
        batch_op.drop_constraint(_ANCHOR_PAYLOAD_CHECK_NAME, type_="check")
        batch_op.drop_constraint(_ANCHOR_LOCAL_TZ_CHECK_NAME, type_="check")
        batch_op.drop_constraint(_ANCHOR_FAMILY_CHECK_NAME, type_="check")
        batch_op.drop_constraint("fk_artifacts_summary_id_artifacts", type_="foreignkey")
        batch_op.drop_constraint("fk_artifacts_raw_id_messages", type_="foreignkey")
        batch_op.drop_constraint(_ARTIFACT_LAYER_CHECK_NAME, type_="check")
        batch_op.create_check_constraint(_ARTIFACT_LAYER_CHECK_NAME, _OLD_ARTIFACT_LAYER_CHECK)
        batch_op.alter_column("export_id", existing_type=sa.Uuid(), nullable=False)
        batch_op.alter_column("path", existing_type=sa.String(length=512), nullable=False)
        batch_op.alter_column("sha256", existing_type=sa.String(length=64), nullable=False)
        for column_name in reversed(_NEW_COLUMNS):
            batch_op.drop_column(column_name)
