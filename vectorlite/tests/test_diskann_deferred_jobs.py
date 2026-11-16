"""Deferred job protocol via real public SQL operations and explicit private artifact.

No shadow write simulates a successful build. Fault injection is clearly labelled.
Driver options are the root integration contract; unwired operations FAIL, not skip.
"""
import hashlib
import json
import os
import sqlite3
import struct
import sys
from pathlib import Path

import pytest


def artifact():
    """Explicit VECTORLITE_RUST_EXTENSION override, else the deployed extension."""
    override = os.environ.get("VECTORLITE_RUST_EXTENSION")
    if override:
        return Path(override).resolve()
    import vectorlite_py
    suffix = "dll" if sys.platform == "win32" else "dylib" if sys.platform == "darwin" else "so"
    return Path(vectorlite_py.vectorlite_path()).with_suffix(f".{suffix}")


def blob(values):
    return struct.pack(f"<{len(values)}f", *values)


def connect(path=":memory:", readonly=False):
    artifact_path = artifact()
    assert artifact_path.is_file(), artifact_path
    uri = f"file:{path}?mode=ro" if readonly else str(path)
    db = sqlite3.connect(uri, isolation_level=None, uri=readonly)
    db.row_factory = sqlite3.Row
    db.enable_load_extension(True)
    db.load_extension(str(artifact_path))
    return db


def create(db, metric="l2", schema="main", name="v"):
    db.execute(f'CREATE VIRTUAL TABLE "{schema}"."{name}" USING vectorlite(embedding float32[2] {metric},diskann(degree=4,build_list_size=8))')


def control(db, op, schema="main", name="v", **opts):
    if "job_id" in opts and "fence" not in opts:
        opts["fence"] = job(db, schema=schema, name=name, job_id=opts["job_id"])["fence"]
    db.execute(f'INSERT INTO "{schema}"."{name}"(operation,path) VALUES (?,?)', (op, json.dumps(opts)))


def job(db, schema="main", name="v", job_id=None):
    if job_id is None:
        job_id = db.execute(f'SELECT active_job FROM "{schema}"."{name}_diskann_meta"').fetchone()[0]
    row = db.execute(f'SELECT * FROM "{schema}"."{name}_diskann_jobs" WHERE job_id=?', (job_id,)).fetchone()
    assert row is not None
    return dict(row)


def prepare(db, **kwargs):
    control(db, "index_prepare", **kwargs)
    return job(db, **kwargs)


def snapshot(db, j, records=1, **kwargs):
    for _ in range(1000):
        if j["phase"] == 1:
            return j
        control(db, "index_snapshot_step", job_id=j["job_id"], max_records=records, max_bytes=65536, **kwargs)
        j = job(db, job_id=j["job_id"], **kwargs)
    pytest.fail("snapshot did not finish within fixture's bounded history")


def candidate(db, j, schema="main", name="v"):
    control(db, "index_claim_build", schema=schema, name=name, job_id=j["job_id"])
    row = db.execute("SELECT vectorlite_index_build(?,?,?)", (schema, name, json.dumps({"job_id": j["job_id"], "fence": j["fence"], "memory_budget_bytes": 128 * 1024 * 1024, "workers": 1}))).fetchone()
    result = json.loads(row[0])
    assert isinstance(result["candidate_token"], str)
    return result


def finish(db, j, result, records=1, publish=True, **kwargs):
    for phase, operation, cursor in [(2, "index_import_step", "import_cursor"), (3, "index_validate_step", "validation_cursor")]:
        for _ in range(1000):
            current = job(db, job_id=j["job_id"], **kwargs)
            if current["phase"] != phase:
                break
            before = current[cursor]
            control(db, operation, job_id=j["job_id"], candidate_token=result["candidate_token"], max_records=records, max_bytes=65536, **kwargs)
            after = job(db, job_id=j["job_id"], **kwargs)
            assert 0 <= after[cursor] - before <= records
        else:
            pytest.fail(f"{operation} did not complete")
    assert job(db, job_id=j["job_id"], **kwargs)["phase"] == 4
    if publish:
        control(db, "index_publish", job_id=j["job_id"], candidate_token=result["candidate_token"], **kwargs)
        assert job(db, job_id=j["job_id"], **kwargs)["phase"] == 5


def source_chain(j, rows):
    header = b"vectorlite/deferred/source-chain/v1" + j["config_hash"] + struct.pack("<QQQ", j["cutoff_revision"], j["cutoff_entry_id"], j["expected_count"])
    digest = hashlib.sha256(header).digest()
    for ordinal, entry, rowid, vector in rows:
        digest = hashlib.sha256(b"vectorlite/deferred/source-row/v1" + digest + struct.pack("<QQQQ", ordinal, entry, rowid, len(vector)) + vector).digest()
    return digest


def test_source_chain_independent_known_digest():
    # Independently computed with Node crypto SHA256, not the Rust implementation.
    j = {"config_hash": bytes([7]) * 32, "cutoff_revision": 12,
         "cutoff_entry_id": 25, "expected_count": 2}
    rows = [(0, 3, 99, bytes([0, 1, 2, 3])), (1, 8, 99, bytes([4, 5, 6, 7]))]
    assert source_chain(j, rows).hex() == "a92e10d03af8451a7c5c57a6fef1d3dac432eb34e0c162a729e7986931c64d59"


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_cutoff_membership_original_rowids_and_resumable_chain(metric):
    db = connect()
    try:
        create(db, metric)
        # A retired gap is scanned but not emitted; N captured from canonical_count.
        db.execute("INSERT INTO v(rowid,embedding) VALUES(1,?)", (blob((1, 2)),))
        db.execute("DELETE FROM v WHERE rowid=1")
        for rowid in [0, 7, 9223372036854775807]:
            db.execute("INSERT INTO v(rowid,embedding) VALUES(?,?)", (rowid, blob((3, 4))))
        j = prepare(db)
        assert j["expected_count"] == 3 and j["cutoff_entry_id"] == 4
        assert j["source_hash"] is None
        control(db, "index_snapshot_step", job_id=j["job_id"], max_records=1, max_bytes=65536)
        first = job(db)
        assert (first["scan_entry_id"], first["captured_count"]) == (1, 0)
        # Versions retired after cutoff remain input even before they are captured.
        db.execute("UPDATE v SET embedding=? WHERE rowid=7", (blob((8, 9)),))
        db.execute("DELETE FROM v WHERE rowid=0")
        db.execute("INSERT INTO v(rowid,embedding) VALUES(0,?)", (blob((5, 6)),))
        db.execute("INSERT INTO v(rowid,embedding) VALUES(22,?)", (blob((7, 8)),))
        # GC must pin still-uncaptured cutoff versions.
        for _ in range(12):
            control(db, "index_cleanup_step", max_records=1, max_bytes=65536)
        j = snapshot(db, first)
        rows = db.execute("SELECT m.ordinal,m.entry_id,m.public_rowid,e.vector FROM v_diskann_snapshot_map m JOIN v_diskann_entries e USING(entry_id) WHERE m.job_id=? ORDER BY ordinal", (j["job_id"],)).fetchall()
        assert [tuple(r)[:3] for r in rows] == [(0, 2, 0), (1, 3, 7), (2, 4, 9223372036854775807)]
        assert j["source_hash"] == source_chain(j, rows) == j["snapshot_chain"]
        assert db.execute("SELECT source_hash FROM v_diskann_generations").fetchone()[0] == j["source_hash"]
    finally:
        db.close()


def test_capture_savepoint_rollback_and_strict_options():
    db = connect()
    try:
        create(db)
        for i in range(4):
            db.execute("INSERT INTO v(rowid,embedding) VALUES(?,?)", (i, blob((i, 1))))
        j = prepare(db)
        db.execute("BEGIN")
        db.execute("SAVEPOINT chunk")
        control(db, "index_snapshot_step", job_id=j["job_id"], max_records=2, max_bytes=65536)
        assert job(db)["captured_count"] == 2
        db.execute("ROLLBACK TO chunk")
        db.execute("RELEASE chunk")
        db.execute("COMMIT")
        assert job(db)["captured_count"] == 0
        assert db.execute("SELECT count(*) FROM v_diskann_snapshot_map").fetchone()[0] == 0
        for options in [{"max_records": 0}, {"max_records": 257}, {"max_records": True}, {"max_records": "1"}, {"max_bytes": 0}, {"unknown": 1}]:
            with pytest.raises(sqlite3.DatabaseError):
                control(db, "index_snapshot_step", job_id=j["job_id"], **options)
        assert job(db)["captured_count"] == 0
    finally:
        db.close()


@pytest.mark.parametrize("population", [0, 1, 6])
def test_real_native_candidate_import_atomic_publish_and_root(population):
    db = connect()
    try:
        create(db)
        for i in range(population):
            db.execute("INSERT INTO v(rowid,embedding) VALUES(?,?)", (i, blob((i, 1))))
        j = snapshot(db, prepare(db))
        result = candidate(db, j)
        finish(db, j, result, publish=False)
        g = dict(db.execute("SELECT * FROM v_diskann_generations WHERE generation_id=?", (j["generation_id"],)).fetchone())
        assert g["state"] == 0 and g["eligible_count"] == population
        nodes = db.execute("SELECT native_local_id,entry_id,state,frozen_vector,neighbors FROM v_diskann_graph_nodes WHERE generation_id=? ORDER BY native_local_id", (j["generation_id"],)).fetchall()
        assert len(nodes) == (population + 1 if population else 0)
        assert g["root_local_id"] == (population if population else None)
        if population:
            assert nodes[0][0] == 0 and nodes[0][2] == 0
            assert tuple(nodes[-1])[:3] == (population, None, 2)
            assert len(nodes[-1][3]) == 8
        db.execute("BEGIN")
        control(db, "index_publish", job_id=j["job_id"], candidate_token=result["candidate_token"])
        assert db.in_transaction
        assert db.execute("SELECT published_generation FROM v_diskann_meta").fetchone()[0] == j["generation_id"]
        db.execute("ROLLBACK")
        assert db.execute("SELECT published_generation FROM v_diskann_meta").fetchone()[0] is None
        assert job(db)["phase"] == 4
        # A rolled-back publication may have released RAM; explicit restart, not
        # a fake candidate inserted into SQL, is the allowed recovery.
        old = job(db)
        control(db, "index_restart", job_id=old["job_id"], fence=old["fence"], force=True)
        new = snapshot(db, job(db))
        assert new["generation_id"] > old["generation_id"] and new["fence"] > old["fence"]
        finish(db, new, candidate(db, new))
    finally:
        db.close()


def test_retirement_before_after_import_and_no_publication_reconcile():
    db = connect()
    try:
        create(db)
        for i in range(4):
            db.execute("INSERT INTO v(rowid,embedding) VALUES(?,?)", (i, blob((i, 1))))
        j = snapshot(db, prepare(db))
        result = candidate(db, j)
        db.execute("DELETE FROM v WHERE rowid=0")  # before import
        control(db, "index_import_step", job_id=j["job_id"], candidate_token=result["candidate_token"], max_records=2, max_bytes=65536)
        assert db.execute("SELECT eligible_count FROM v_diskann_generations").fetchone()[0] == 1
        db.execute("UPDATE v SET embedding=? WHERE rowid=1", (blob((30, 40)),))  # after import
        assert db.execute("SELECT eligible_count FROM v_diskann_generations").fetchone()[0] == 0
        finish(db, j, result, publish=False)
        db.execute("DELETE FROM v WHERE rowid=2")  # after validation, before publish
        assert db.execute("SELECT eligible_count FROM v_diskann_generations").fetchone()[0] == 1
        control(db, "index_publish", job_id=j["job_id"], candidate_token=result["candidate_token"])
        assert db.execute("SELECT eligible_count,total_count FROM v_diskann_generations").fetchone()[:] == (1, 4)
    finally:
        db.close()


def test_late_snapshot_sql_error_rolls_back_map_and_cursor():
    db = connect()
    try:
        create(db)
        db.execute("INSERT INTO v(rowid,embedding) VALUES(7,?)", (blob((1, 2)),))
        j = prepare(db)
        denied = []

        def authorizer(action, table, column, schema, source):
            if action == sqlite3.SQLITE_UPDATE and table == "v_diskann_jobs":
                denied.append(column)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        db.set_authorizer(authorizer)
        try:
            with pytest.raises(sqlite3.DatabaseError) as error:
                control(db, "index_snapshot_step", job_id=j["job_id"], max_records=1, max_bytes=65536)
            assert error.value.sqlite_errorcode == sqlite3.SQLITE_AUTH
            assert denied, "must reach a real late durable progress SQL failure"
        finally:
            db.set_authorizer(None)
        assert job(db)["captured_count"] == 0
        assert db.execute("SELECT count(*) FROM v_diskann_snapshot_map").fetchone()[0] == 0
        snapshot(db, job(db))
    finally:
        db.close()


def test_claim_fence_and_build_transaction_guard():
    db = connect()
    try:
        create(db)
        db.execute("INSERT INTO v(rowid,embedding) VALUES(0,?)", (blob((1, 2)),))
        db.execute("INSERT INTO v(rowid,embedding) VALUES(1,?)", (blob((2, 3)),))
        j = snapshot(db, prepare(db))
        with pytest.raises(sqlite3.DatabaseError):
            control(db, "index_claim_build", job_id=j["job_id"], fence=j["fence"] + 1)
        assert job(db)["phase"] == 1
        control(db, "index_claim_build", job_id=j["job_id"])
        options = json.dumps({"job_id": j["job_id"], "fence": j["fence"], "memory_budget_bytes": 128 * 1024 * 1024})
        db.execute("BEGIN")
        try:
            with pytest.raises(sqlite3.Error, match="autocommit|BEGIN|transaction") as blocked:
                db.execute("SELECT vectorlite_index_build('main','v',?)", (options,)).fetchall()
            assert blocked.value.sqlite_errorcode == sqlite3.SQLITE_MISUSE
        finally:
            db.execute("ROLLBACK")
        assert job(db)["phase"] == 2
        cursor = db.execute("SELECT * FROM v_diskann_entries")
        assert cursor.fetchone() is not None
        try:
            with pytest.raises(sqlite3.Error, match="busy|cursor|statement|transaction") as blocked:
                db.execute("SELECT vectorlite_index_build('main','v',?)", (options,)).fetchall()
            assert blocked.value.sqlite_errorcode == sqlite3.SQLITE_MISUSE
        finally:
            cursor.close()
        assert job(db)["candidate_hash"] is None
    finally:
        db.close()


def test_late_import_sql_error_rolls_back_nodes_eligibility_and_cursor():
    db = connect()
    try:
        create(db)
        for i in range(3):
            db.execute("INSERT INTO v(rowid,embedding) VALUES(?,?)", (i, blob((i, 1))))
        j = snapshot(db, prepare(db))
        result = candidate(db, j)
        denied = []

        def authorizer(action, table, column, schema, source):
            if action == sqlite3.SQLITE_UPDATE and table == "v_diskann_jobs":
                denied.append(column)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        db.set_authorizer(authorizer)
        try:
            with pytest.raises(sqlite3.Error) as error:
                control(db, "index_import_step", job_id=j["job_id"], candidate_token=result["candidate_token"], max_records=2, max_bytes=65536)
            assert error.value.sqlite_errorcode == sqlite3.SQLITE_AUTH
            assert denied
        finally:
            db.set_authorizer(None)
        assert job(db)["import_cursor"] == 0
        assert db.execute("SELECT count(*) FROM v_diskann_graph_nodes").fetchone()[0] == 0
        assert db.execute("SELECT eligible_count FROM v_diskann_generations").fetchone()[0] == 0
        finish(db, j, result)
    finally:
        db.close()


@pytest.mark.parametrize("fault", ["logical_source", "original_rowid", "nonfinite_vector"])
def test_materializer_rejects_corrupt_sealed_input_without_sql_progress(fault):
    db = connect()
    try:
        create(db)
        for i in range(3):
            db.execute("INSERT INTO v(rowid,embedding) VALUES(?,?)", (i, blob((i, 1))))
        j = snapshot(db, prepare(db))
        control(db, "index_claim_build", job_id=j["job_id"])
        # Explicit unsupported shadow corruption for fault injection only.
        if fault == "logical_source":
            db.execute("UPDATE v_diskann_entries SET vector=? WHERE entry_id=1", (blob((10, 20)),))
        elif fault == "original_rowid":
            db.execute("UPDATE v_diskann_snapshot_map SET public_rowid=55 WHERE ordinal=0")
        else:
            db.execute("UPDATE v_diskann_entries SET vector=? WHERE entry_id=1", (blob((float("nan"), 1)),))
        options = json.dumps({"job_id": j["job_id"], "fence": j["fence"], "memory_budget_bytes": 128 * 1024 * 1024})
        with pytest.raises(sqlite3.Error) as error:
            db.execute("SELECT vectorlite_index_build('main','v',?)", (options,)).fetchall()
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
        current = job(db)
        assert current["phase"] == 2 and current["import_cursor"] == 0 and current["candidate_hash"] is None
        assert db.execute("SELECT count(*) FROM v_diskann_graph_nodes").fetchone()[0] == 0
    finally:
        db.close()


def test_progressive_validation_compares_ordered_native_adjacency():
    db = connect()
    try:
        create(db)
        for i in range(6):
            db.execute("INSERT INTO v(rowid,embedding) VALUES(?,?)", (i, blob((i, 1))))
        j = snapshot(db, prepare(db))
        result = candidate(db, j)
        control(db, "index_import_step", job_id=j["job_id"], candidate_token=result["candidate_token"], max_records=256, max_bytes=65536)
        assert job(db)["phase"] == 3 and job(db)["validation_cursor"] == 0
        original = db.execute("SELECT neighbors FROM v_diskann_graph_nodes WHERE native_local_id=0").fetchone()[0]
        assert len(original) >= 16, "fixture must exercise native adjacency ordering"
        ids = struct.unpack(f"<{len(original) // 8}Q", original)
        changed = struct.pack(f"<{len(ids)}Q", *reversed(ids))
        assert changed != original
        # Same valid degree/range/IDs but different native order: fault injection.
        db.execute("UPDATE v_diskann_graph_nodes SET neighbors=? WHERE native_local_id=0", (changed,))
        with pytest.raises(sqlite3.Error) as error:
            control(db, "index_validate_step", job_id=j["job_id"], candidate_token=result["candidate_token"], max_records=1, max_bytes=65536)
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
        assert job(db)["validation_cursor"] == 0 and job(db)["phase"] == 3
        db.execute("UPDATE v_diskann_graph_nodes SET neighbors=? WHERE native_local_id=0", (original,))
        finish(db, j, result)
    finally:
        db.close()


def test_gc_bounded_data_rows_actual_retired_count_and_published_navigation():
    db = connect()
    try:
        create(db)
        for i in range(6):
            db.execute("INSERT INTO v(rowid,embedding) VALUES(?,?)", (i, blob((i, 1))))
        old = snapshot(db, prepare(db))
        finish(db, old, candidate(db, old))
        db.execute("DELETE FROM v WHERE rowid=0")
        # Retired entry remains needed by the raw published navigation graph.
        for _ in range(10):
            control(db, "index_cleanup_step", max_records=1, max_bytes=65536)
        assert db.execute("SELECT count(*) FROM v_diskann_entries WHERE entry_id=1").fetchone()[0] == 1
        assert db.execute("SELECT retired_count FROM v_diskann_meta").fetchone()[0] == 1
        new = snapshot(db, prepare(db))
        finish(db, new, candidate(db, new))

        def counts():
            return tuple(db.execute(f"SELECT count(*) FROM v_diskann_{table}").fetchone()[0] for table in ["graph_nodes", "snapshot_map", "entries"])

        for _ in range(60):
            before = counts()
            control(db, "index_cleanup_step", max_records=1, max_bytes=65536)
            after = counts()
            assert sum(a - b for a, b in zip(before, after)) <= 1
            assert db.execute("SELECT retired_count FROM v_diskann_meta").fetchone()[0] == db.execute("SELECT count(*) FROM v_diskann_entries WHERE retired_revision IS NOT NULL").fetchone()[0]
        assert db.execute("SELECT count(*) FROM v_diskann_generations WHERE state=2").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM v_diskann_entries WHERE entry_id=1").fetchone()[0] == 0
        assert db.execute("SELECT retired_count FROM v_diskann_meta").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM v_diskann_graph_nodes WHERE generation_id=?", (new["generation_id"],)).fetchone()[0] == 6
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        db.close()


def test_readonly_reopen_status_and_explicit_fresh_takeover(tmp_path):
    path = tmp_path / "jobs.db"
    db = connect(path)
    try:
        create(db)
        db.execute("INSERT INTO v(rowid,embedding) VALUES(7,?)", (blob((1, 2)),))
        old = snapshot(db, prepare(db))
        candidate(db, old)
        old = job(db)
    finally:
        db.close()  # candidate RAM lost; durable snapshot retained
    ro = connect(path, readonly=True)
    try:
        before = tuple(ro.execute("SELECT * FROM v_diskann_jobs").fetchone())
        status = json.loads(ro.execute("SELECT vectorlite_index_status('main','v')").fetchone()[0])
        assert isinstance(status, dict)
        assert tuple(ro.execute("SELECT * FROM v_diskann_jobs").fetchone()) == before
    finally:
        ro.close()
    db = connect(path)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            control(db, "index_claim_build", job_id=old["job_id"])
        control(db, "index_restart", job_id=old["job_id"], fence=old["fence"], force=True)
        new = snapshot(db, job(db))
        assert new["job_id"] > old["job_id"] and new["generation_id"] > old["generation_id"] and new["fence"] > old["fence"]
        finish(db, new, candidate(db, new))
        for _ in range(30):
            control(db, "index_cleanup_step", max_records=1, max_bytes=65536)
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("SELECT count(*) FROM v_diskann_generations WHERE state=2").fetchone()[0] == 0
    finally:
        db.close()
