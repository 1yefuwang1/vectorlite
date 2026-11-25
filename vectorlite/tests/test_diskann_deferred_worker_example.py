"""Regression test for the public worker example in doc/diskann.md.

The helper below mirrors the documented lifecycle step for step: bounded duplicate
steps until each phase is left, publication only after validation completes, and a
clean empty-table path. It is deliberately written against public SQL only, so a
change to any documented step, phase number or option name fails here.
"""
import json
import os
import sqlite3
import struct
import sys
from pathlib import Path

import pytest

TABLE = "embeddings"
SCHEMA = "main"
PHASE_CAPTURING, PHASE_IMPORT, PHASE_VALIDATING, PHASE_VALIDATED = 0, 2, 3, 4
MAX_RECORDS = 128
BUDGET = 256 * 1024 * 1024


def artifact():
    """Explicit VECTORLITE_RUST_EXTENSION override, else the deployed extension."""
    override = os.environ.get("VECTORLITE_RUST_EXTENSION")
    if override:
        return Path(override).resolve()
    import vectorlite_py

    suffix = "dll" if sys.platform == "win32" else "dylib" if sys.platform == "darwin" else "so"
    return Path(vectorlite_py.vectorlite_path()).with_suffix(f".{suffix}")


def status(worker):
    return json.loads(worker.execute(
        "SELECT vectorlite_index_status(?,?)", (SCHEMA, TABLE)).fetchone()[0])


def control(worker, operation, **options):
    worker.execute(f"INSERT INTO {TABLE}(operation, path) VALUES (?, ?)",
                   (operation, json.dumps(options, separators=(",", ":"))))


def maintain(database, extension):
    worker = sqlite3.connect(database, isolation_level=None)
    try:
        worker.enable_load_extension(True)
        worker.load_extension(extension)
        if status(worker)["canonical_count"] == 0:
            return status(worker)

        control(worker, "index_prepare")
        current = status(worker)["current_job"]
        job, fence = current["job_id"], current["fence"]

        while status(worker)["phase"] == PHASE_CAPTURING:
            control(worker, "index_snapshot_step",
                    job_id=job, fence=fence, max_records=MAX_RECORDS)

        control(worker, "index_claim_build", job_id=job, fence=fence)
        candidate = json.loads(worker.execute(
            "SELECT vectorlite_index_build(?,?,?)",
            (SCHEMA, TABLE, json.dumps(
                {"job_id": job, "fence": fence,
                 "memory_budget_bytes": BUDGET}, separators=(",", ":"))),
        ).fetchone()[0])
        token = candidate["candidate_token"]

        while status(worker)["phase"] == PHASE_IMPORT:
            control(worker, "index_import_step", job_id=job, fence=fence,
                    candidate_token=token, max_records=MAX_RECORDS)
        while status(worker)["phase"] == PHASE_VALIDATING:
            control(worker, "index_validate_step", job_id=job, fence=fence,
                    candidate_token=token, max_records=MAX_RECORDS)

        validated = status(worker)
        assert validated["phase"] == PHASE_VALIDATED
        cursors = validated["current_job"]
        control(worker, "index_publish", job_id=job, fence=fence,
                candidate_token=token)

        # Publication clears the active job and makes the captured population
        # searchable. Rows written after the cutoff stay pending, so the backlog is
        # reported rather than asserted.
        final = status(worker)
        assert final["active_job"] is None
        assert final["published_generation"] is not None
        control(worker, "index_cleanup_step", max_records=MAX_RECORDS)
        final["validated_cursors"] = cursors
        return final
    finally:
        worker.close()


def create_table(connection):
    connection.execute(
        f'CREATE VIRTUAL TABLE {TABLE} USING vectorlite('
        f'embedding float32[3] l2, diskann(degree=8, build_list_size=32, max_visits=8192))')


def blob(values):
    return struct.pack(f"<{len(values)}f", *values)


@pytest.mark.parametrize("population", [300, 1000])
def test_documented_worker_publishes_multi_chunk_population(tmp_path, population):
    database = tmp_path / f"worker-{population}.db"
    app = sqlite3.connect(database, isolation_level=None)
    app.enable_load_extension(True)
    app.load_extension(str(artifact()))
    create_table(app)
    app.execute("BEGIN")
    for rowid in range(population):
        app.execute(
            f"INSERT INTO {TABLE}(rowid, embedding) VALUES (?, ?)",
            (rowid, blob((rowid % 7, rowid % 11, rowid % 13))),
        )
    app.execute("COMMIT")
    before = json.loads(app.execute(
        "SELECT vectorlite_index_status(?,?)", (SCHEMA, TABLE)).fetchone()[0])
    assert before["canonical_count"] == population
    assert before["eligible_count"] == 0
    assert app.execute(
        f"SELECT count(*) FROM {TABLE} WHERE knn_search(embedding, knn_param(?, 5))",
        (blob((0, 0, 0)),)).fetchone()[0] == 0

    final = maintain(str(database), str(artifact()))
    assert final["canonical_count"] == population
    assert final["eligible_count"] == population
    # Quiescent bulk load: nothing arrived after the cutoff, so there is no backlog.
    assert final["pending_count"] == 0
    assert final["published_generation"] is not None
    # The published generation is searchable, and it took more than one bounded
    # step per repeated phase for this population.
    cursors = final["validated_cursors"]
    assert cursors["captured_count"] == population
    # Import and validation walk native graph nodes: the population plus the
    # explicitly retained frozen root.
    assert cursors["import_cursor"] == population + 1
    assert cursors["validation_cursor"] == population + 1
    rows = app.execute(
        f"SELECT rowid, distance FROM {TABLE} "
        "WHERE knn_search(embedding, knn_param(?, 5)) ORDER BY distance, rowid",
        (blob((0, 0, 0)),)).fetchall()
    assert rows and rows[0][0] == 0 and rows[0][1] == pytest.approx(0.0)
    app.close()


def test_documented_worker_handles_empty_table(tmp_path):
    database = tmp_path / "worker-empty.db"
    app = sqlite3.connect(database, isolation_level=None)
    app.enable_load_extension(True)
    app.load_extension(str(artifact()))
    create_table(app)
    empty = maintain(str(database), str(artifact()))
    assert empty["canonical_count"] == 0
    assert empty["pending_count"] == 0
    assert empty["eligible_count"] == 0
    assert empty["published_generation"] is None
    assert empty["active_job"] is None
    assert app.execute(
        f"SELECT count(*) FROM {TABLE} WHERE knn_search(embedding, knn_param(?, 5))",
        (blob((0, 0, 0)),)).fetchone()[0] == 0
    app.close()

def test_post_cutoff_write_publishes_with_a_valid_pending_backlog(tmp_path):
    """A writer committing after the cutoff must not invalidate publication."""
    database = tmp_path / "worker-post-cutoff.db"
    app = sqlite3.connect(database, isolation_level=None)
    app.enable_load_extension(True)
    app.load_extension(str(artifact()))
    create_table(app)
    for rowid in range(300):
        app.execute(
            f"INSERT INTO {TABLE}(rowid, embedding) VALUES (?, ?)",
            (rowid, blob((rowid % 7, rowid % 11, rowid % 13))),
        )

    worker = sqlite3.connect(database, isolation_level=None)
    try:
        worker.enable_load_extension(True)
        worker.load_extension(str(artifact()))
        control(worker, "index_prepare")
        current = status(worker)["current_job"]
        job, fence = current["job_id"], current["fence"]
        while status(worker)["phase"] == PHASE_CAPTURING:
            control(worker, "index_snapshot_step",
                    job_id=job, fence=fence, max_records=MAX_RECORDS)

        # Another connection commits after the cutoff was captured.
        late_rowid = 5000
        app.execute(
            f"INSERT INTO {TABLE}(rowid, embedding) VALUES (?, ?)",
            (late_rowid, blob((1, 1, 1))),
        )

        control(worker, "index_claim_build", job_id=job, fence=fence)
        candidate = json.loads(worker.execute(
            "SELECT vectorlite_index_build(?,?,?)",
            (SCHEMA, TABLE, json.dumps(
                {"job_id": job, "fence": fence,
                 "memory_budget_bytes": BUDGET}, separators=(",", ":"))),
        ).fetchone()[0])
        token = candidate["candidate_token"]
        while status(worker)["phase"] == PHASE_IMPORT:
            control(worker, "index_import_step", job_id=job, fence=fence,
                    candidate_token=token, max_records=MAX_RECORDS)
        while status(worker)["phase"] == PHASE_VALIDATING:
            control(worker, "index_validate_step", job_id=job, fence=fence,
                    candidate_token=token, max_records=MAX_RECORDS)
        control(worker, "index_publish", job_id=job, fence=fence,
                candidate_token=token)

        # Publication succeeds with a backlog, which is the documented behavior.
        final = status(worker)
        assert final["active_job"] is None
        assert final["published_generation"] is not None
        assert final["canonical_count"] == 301
        assert final["eligible_count"] == 300
        assert final["pending_count"] == 1
        control(worker, "index_cleanup_step", max_records=MAX_RECORDS)
    finally:
        worker.close()

    # The late row is canonical but not yet searchable; published rows still are.
    late = app.execute(
        f"SELECT embedding FROM {TABLE} WHERE rowid=?", (late_rowid,)).fetchone()
    assert late is not None and late[0] == blob((1, 1, 1))
    labels = {row[0] for row in app.execute(
        f"SELECT rowid FROM {TABLE} WHERE knn_search(embedding, knn_param(?, 5))",
        (blob((0, 0, 0)),))}
    assert late_rowid not in labels
    assert 0 in labels
    app.close()
