"""Format 5 deferred DiskANN SQL/storage regressions against the selected extension.

Small deterministic graphs test semantics, not beyond-RAM performance or RSS.
Shadow-table writes below are deliberate fault injection, not supported user API.
"""

import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys

import pytest
import vectorlite_py


@pytest.fixture
def extension_path():
    suffix = "dll" if sys.platform == "win32" else "dylib" if sys.platform == "darwin" else "so"
    # Use the same deployed/installed artifact without importing another test
    # module: the root and installed-wheel runners use --import-mode=importlib.
    default = Path(vectorlite_py.vectorlite_path()).with_suffix(f".{suffix}")
    library = Path(os.environ.get("VECTORLITE_RUST_EXTENSION", default))
    if not library.is_file():
        pytest.fail(f"Vectorlite extension is not built or installed: {library}")
    return library.resolve()


SUFFIXES = tuple("_diskann_" + suffix for suffix in (
    "meta", "entries", "current", "generations", "graph_nodes", "generation_retired", "jobs", "snapshot_map", "txn",
))
BUILD_BUDGET = 128 * 1024 * 1024


BASE_ROWS = [(1, [1.0, 0.0]), (7, [0.0, 2.0]), (19, [3.0, 4.0])]


def _q(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def _table(name="v", schema="main"):
    return f"{_q(schema)}.{_q(name)}"


def _blob(values):
    return struct.pack(f"<{len(values)}f", *values)


def _vector(blob):
    return struct.unpack(f"<{len(blob) // 4}f", blob)


def _open(extension_path, path=":memory:", *, readonly=False):
    database = f"{path.as_uri()}?mode=ro" if readonly else str(path)
    connection = sqlite3.connect(database, isolation_level=None, uri=readonly, timeout=1)
    try:
        connection.enable_load_extension(True)
        connection.load_extension(str(extension_path))
    except BaseException:
        connection.close()
        raise
    return connection


def _create(connection, name="v", schema="main", metric="l2", options="diskann()"):
    connection.execute(
        f"CREATE VIRTUAL TABLE {_table(name, schema)} USING "
        f"vectorlite(embedding float32[2] {metric}, {options})"
    )


def _insert(connection, rowid, vector, name="v", schema="main"):
    connection.execute(
        f"INSERT INTO {_table(name, schema)}(rowid,embedding) VALUES(?,?)",
        (rowid, _blob(vector)),
    ).fetchall()


def _seed(connection, name="v", schema="main"):
    """Seed and explicitly publish the standard searchable fixture, outside BEGIN."""
    for rowid, vector in BASE_ROWS:
        _insert(connection, rowid, vector, name, schema)
    _maintenance(connection, name=name, schema=schema)


def _get(connection, rowid, name="v", schema="main"):
    row = connection.execute(
        f"SELECT embedding FROM {_table(name, schema)} WHERE rowid=?", (rowid,)
    ).fetchone()
    return None if row is None else _vector(row[0])


def _knn(connection, query=(1.0, 0.0), k=10, *, name="v", schema="main", options=None):
    args = [_blob(query), k]
    parameter = "knn_param(?,?)"
    if options is not None:
        parameter = "knn_param(?,?,?)"
        args.append(json.dumps(options) if isinstance(options, dict) else options)
    return connection.execute(
        f"SELECT rowid,distance FROM {_table(name, schema)} "
        f"WHERE knn_search(embedding,{parameter})",
        args,
    ).fetchall()


def _status(connection, name="v", schema="main"):
    return json.loads(connection.execute(
        "SELECT vectorlite_index_status(?,?)", (schema, name),
    ).fetchone()[0])


def _control(connection, operation, options=None, name="v", schema="main"):
    connection.execute(
        f"INSERT INTO {_table(name, schema)}(operation,path) VALUES(?,?)",
        (operation, json.dumps({} if options is None else options)),
    ).fetchall()


def _jf(job):
    return {"job_id": job["job_id"], "fence": job["fence"]}


def _capture(connection, name="v", schema="main", records=256):
    for _ in range(10000):
        job = _status(connection, name, schema)["current_job"]
        if job["phase"] == 1:
            return job
        assert job["phase"] == 0
        _control(connection, "index_snapshot_step",
                 {**_jf(job), "max_records": records, "max_bytes": 65536}, name, schema)
    pytest.fail("capture failed to make bounded progress")


def _build(connection, job, name="v", schema="main", budget=BUILD_BUDGET):
    return json.loads(connection.execute(
        "SELECT vectorlite_index_build(?,?,?)",
        (schema, name, json.dumps({**_jf(job), "memory_budget_bytes": budget, "workers": 1})),
    ).fetchone()[0])


def _candidate(connection, name="v", schema="main", records=256):
    _control(connection, "index_prepare", name=name, schema=schema)
    job = _capture(connection, name, schema, records)
    _control(connection, "index_claim_build", _jf(job), name, schema)
    return job, _build(connection, job, name, schema)


def _finish(connection, job, candidate, name="v", schema="main", records=256, publish=True):
    opts = {**_jf(job), "candidate_token": candidate["candidate_token"]}
    for phase, operation, cursor in [(2, "index_import_step", "import_cursor"),
                                     (3, "index_validate_step", "validation_cursor")]:
        for _ in range(10000):
            current = _status(connection, name, schema)["current_job"]
            if current["phase"] != phase:
                break
            _control(connection, operation,
                     {**opts, "max_records": records, "max_bytes": 65536}, name, schema)
            after = _status(connection, name, schema)["current_job"]
            assert 0 < after[cursor] - current[cursor] <= records or candidate["node_count"] == 0
        else:
            pytest.fail(f"{operation} did not progress")
    assert _status(connection, name, schema)["current_job"]["phase"] == 4
    if publish:
        _control(connection, "index_publish", opts, name, schema)
        assert _status(connection, name, schema)["current_job"] is None


def _maintenance(connection, name="v", schema="main", records=256):
    # Explicit application-owned maintenance, never an implicit query/ingest action.
    assert not connection.in_transaction
    job, candidate = _candidate(connection, name, schema, records)
    _finish(connection, job, candidate, name, schema, records)
    _control(connection, "index_cleanup_step", name=name, schema=schema)
    return job


def _snapshot(connection, name="v", schema="main"):
    # Capture all nine Format 5 shadows, including compound generation/job keys.
    return tuple(
        tuple(connection.execute(
            f"SELECT * FROM {_table(name + suffix, schema)} ORDER BY 1,2"
        ).fetchall())
        for suffix in SUFFIXES
    )


def _schema_names(connection, schema="main"):
    return {row[0] for row in connection.execute(
        f"SELECT name FROM {_q(schema)}.sqlite_schema WHERE type='table'"
    )}


def _live_id(connection, rowid):
    return connection.execute(
        "SELECT entry_id FROM v_diskann_current WHERE public_rowid=?", (rowid,)
    ).fetchone()[0]


def _slot_payload(connection, generation_id, ids):
    """Format-5 16-byte (native_local_id, entry_id) neighbour slots.

    entry_id is zero only for the frozen root (native id total_count); every
    other slot carries that node's own entry identity.
    """
    rows = dict(connection.execute(
        "SELECT native_local_id,entry_id FROM v_diskann_graph_nodes "
        "WHERE generation_id=? AND state=0",
        (generation_id,),
    ).fetchall())
    root = connection.execute(
        "SELECT root_local_id FROM v_diskann_generations WHERE generation_id=?",
        (generation_id,),
    ).fetchone()[0]
    payload = bytearray()
    for native_id in ids:
        payload += struct.pack(
            "<QQ", native_id, 0 if native_id == root else rows[native_id]
        )
    return bytes(payload)


def _observe_allocations(connection):
    observed = []

    def record(rowid):
        observed.append(rowid)
        return 0

    connection.create_function("record_diskann_allocation", 1, record)
    connection.execute(
        "CREATE TRIGGER record_alloc AFTER INSERT ON v_diskann_entries "
        "BEGIN "
        "SELECT record_diskann_allocation(NEW.public_rowid); END"
    )
    return observed


@pytest.fixture
def conn(extension_path):
    connection = _open(extension_path)
    try:
        _create(connection)
        yield connection
    finally:
        connection.close()


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_diskann_defaults_readback_distances_and_named_search(extension_path, metric):
    connection = _open(extension_path)
    try:
        _create(connection, metric=metric)
        _seed(connection)
        for rowid, original in BASE_ROWS:
            expected = original
            if metric == "cosine":
                norm = math.sqrt(sum(value * value for value in original))
                expected = [value / norm for value in original]
            assert _get(connection, rowid) == pytest.approx(expected, abs=1e-6)
        default = _knn(connection)
        tuned = _knn(connection, options={"search_list_size": 100})
        assert [row[0] for row in tuned] == [row[0] for row in default]
        assert [row[1] for row in tuned] == pytest.approx([row[1] for row in default])
        assert {row[0] for row in tuned} == {1, 7, 19}
        for rowid, distance in tuned:
            vector = _get(connection, rowid)
            expected = sum((a - b) ** 2 for a, b in zip(vector, [1.0, 0.0]))
            if metric == "cosine":
                expected = 1.0 - vector[0]
            assert distance == pytest.approx(expected, abs=1e-5)
        assert [row[1] for row in tuned] == sorted(row[1] for row in tuned)
        assert 0 not in {row[0] for row in tuned}  # Frozen node is not a SQL row.
    finally:
        connection.close()


@pytest.mark.parametrize("declaration", ["float16[2] l2", "bfloat16[2] cosine", "float32[2] ip"])
def test_diskann_rejects_unsupported_space_without_shadow_leaks(conn, declaration):
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute(f"CREATE VIRTUAL TABLE bad USING vectorlite(embedding {declaration},diskann())")
    assert not {"bad", "bad_diskann_meta", "bad_diskann_entries"} & _schema_names(conn)
    _insert(conn, 1, [1, 0])
    assert _get(conn, 1) == (1.0, 0.0)


@pytest.mark.parametrize("options", [
    "degree=1", "degree=10001", "degree=4,degree=5", "build_list_size=0",
    "search_list_size=0", "cache_bytes=4095", "max_visits=0",
    "max_visits=4294967296", "alpha=0.9", "alpha=nan", "alpha=inf",
    "build_list_size=101,max_visits=100", "search_list_size=101,max_visits=100",
    "cache_bytes=9999999999999999999999999", "unknown=1",
])
def test_diskann_invalid_budgets_and_options_fail_before_creation(conn, options):
    with pytest.raises(sqlite3.DatabaseError):
        _create(conn, name="bad", options=f"diskann({options})")
    assert not {"bad", "bad_diskann_meta", "bad_diskann_entries"} & _schema_names(conn)


def test_diskann_small_cache_and_bounded_named_search(extension_path):
    connection = _open(extension_path)
    try:
        _create(connection, options="diskann(degree=4,build_list_size=8,search_list_size=8,alpha=1,cache_bytes=1048576,max_visits=128)")
        _seed(connection)
        assert {row[0] for row in _knn(connection, k=3)} == {1, 7, 19}
        before = _snapshot(connection)
        with pytest.raises(sqlite3.DatabaseError):
            _knn(connection, k=1, options={"search_list_size": 2**63 - 1})
        assert _snapshot(connection) == before
    finally:
        connection.close()


@pytest.mark.parametrize("options", [
    10, "{}", "[]", "not-json", '{"ef":10}', '{"search_list_size":0}',
    '{"search_list_size":-1}', '{"search_list_size":1.5}',
    '{"search_list_size":true}', '{"search_list_size":null}',
    '{"search_list_size":10,"extra":1}',
])
def test_diskann_search_options_do_not_reinterpret_hnsw_ef(conn, options):
    _seed(conn)
    before = _snapshot(conn)
    with pytest.raises(sqlite3.DatabaseError):
        _knn(conn, options=options)
    assert _snapshot(conn) == before
    assert _knn(conn, k=1)[0][0] == 1


def test_diskann_named_options_are_rejected_by_hnsw(conn):
    conn.execute("CREATE VIRTUAL TABLE h USING vectorlite(embedding float32[2],hnsw(max_elements=8))")
    _insert(conn, 1, [1, 0], name="h")
    with pytest.raises(sqlite3.DatabaseError):
        _knn(conn, name="h", options={"search_list_size": 10})
    assert _knn(conn, name="h", options=20)[0][0] == 1


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_diskann_zero_vector_is_finite_and_roundtrips(extension_path, metric):
    connection = _open(extension_path)
    try:
        _create(connection, metric=metric)
        _insert(connection, 0, [0, 0])
        assert _get(connection, 0) == (0.0, 0.0)
        _maintenance(connection)
        rows = _knn(connection, query=[0, 0])
        assert rows and all(math.isfinite(distance) for _, distance in rows)
    finally:
        connection.close()


@pytest.mark.parametrize("bad", [[math.nan, 0], [math.inf, 0], [-math.inf, 1]])
def test_diskann_nonfinite_insert_and_query_are_failure_atomic(conn, bad):
    _seed(conn)
    before = _snapshot(conn)
    with pytest.raises(sqlite3.DatabaseError):
        _insert(conn, 100, bad)
    with pytest.raises(sqlite3.DatabaseError):
        _knn(conn, query=bad)
    assert _snapshot(conn) == before
    assert _knn(conn, k=1)[0][0] == 1


@pytest.mark.parametrize("metric,bad,valid", [
    ("l2", [1e19, 0], [1e18, 0]),
    ("cosine", [2e19, 0], [1e19, 0]),
])
def test_diskann_numeric_headroom_and_extreme_normalization(extension_path, metric, bad, valid):
    connection = _open(extension_path)
    try:
        _create(connection, metric=metric)
        before = _snapshot(connection)
        with pytest.raises(sqlite3.DatabaseError):
            _insert(connection, 1, bad)
        with pytest.raises(sqlite3.DatabaseError):
            _knn(connection, query=bad)
        assert _snapshot(connection) == before
        _insert(connection, 1, valid)
        assert all(math.isfinite(value) for value in _get(connection, 1))
        if metric == "cosine":
            assert _get(connection, 1) == pytest.approx([1, 0], abs=1e-6)
        _maintenance(connection)
        assert all(math.isfinite(distance) for _, distance in _knn(connection, query=valid))
    finally:
        connection.close()


def test_diskann_sparse_rowids_filter_and_internal_ids(conn):
    labels = [0, 2**40 + 7, 2**63 - 1]
    for i, label in enumerate(labels):
        _insert(conn, label, [i + 1, 0])
    assert _get(conn, labels[-1]) == (3.0, 0.0)
    _maintenance(conn)
    assert {row[0] for row in _knn(conn, k=10)} == set(labels)
    rows = conn.execute(
        "SELECT rowid,distance FROM v WHERE knn_search(embedding,knn_param(?,1)) "
        "AND rowid IN (?,?)", (_blob([1, 0]), labels[1], labels[2])
    ).fetchall()
    assert rows == [(labels[1], 1.0)]
    ids = [row[0] for row in conn.execute("SELECT entry_id FROM v_diskann_current")]
    assert all(1 <= entry_id <= len(labels) for entry_id in ids)
    with pytest.raises(sqlite3.DatabaseError):
        _insert(conn, -1, [1, 0])
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE v SET rowid=? WHERE rowid=?", (5, labels[0]))


def test_diskann_empty_table_and_negative_query_labels(conn):
    assert _knn(conn) == []
    _seed(conn)
    assert _get(conn, -1) is None
    assert conn.execute("SELECT rowid FROM v WHERE rowid IN (-1,-7)").fetchall() == []
    assert conn.execute("SELECT rowid FROM v WHERE rowid IN (-1,1) ORDER BY rowid").fetchall() == [(1,)]
    assert conn.execute(
        "SELECT rowid FROM v WHERE knn_search(embedding,knn_param(?,3)) AND rowid=-1",
        (_blob([1, 0]),),
    ).fetchall() == []


def test_diskann_recreate_same_name_does_not_reuse_identity_or_cached_rows(conn):
    _seed(conn)
    old_identity = conn.execute("SELECT instance_id FROM v_diskann_meta").fetchone()[0]
    assert _knn(conn, k=1)[0][0] == 1
    conn.execute("DROP TABLE v")
    _create(conn)
    new_identity = conn.execute("SELECT instance_id FROM v_diskann_meta").fetchone()[0]
    assert new_identity != old_identity
    assert _get(conn, 1) is None
    assert _knn(conn) == []
    _insert(conn, 1, [8, 1])
    assert _get(conn, 1) == (8.0, 1.0)
    assert _knn(conn) == []
    _maintenance(conn)
    assert _knn(conn, query=[8, 1], k=1) == [(1, 0.0)]


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_diskann_repeated_updates_preserve_live_readback_through_rebuild(extension_path, metric):
    import random

    connection = _open(extension_path)
    try:
        _create(connection, metric=metric)
        rng = random.Random(20261007)
        data = {rowid: [rng.uniform(-1, 1), rng.uniform(-1, 1)]
                for rowid in range(1, 65)}
        for rowid, values in data.items():
            _insert(connection, rowid, values)
        for iteration in range(96):
            rowid = 1 + iteration % 32
            data[rowid] = [rng.uniform(-1, 1), rng.uniform(-1, 1)]
            connection.execute("UPDATE v SET embedding=? WHERE rowid=?",
                               (_blob(data[rowid]), rowid))
        for rowid in range(49, 65):
            connection.execute("DELETE FROM v WHERE rowid=?", (rowid,))
            del data[rowid]
        _maintenance(connection)
        ids_before = {rowid: _live_id(connection, rowid) for rowid in data}
        # High L on this bounded dataset exercises graph traversal, not the
        # k>=population exact path, even with many retained retired vectors.
        for rowid in (1, 16, 32, 48):
            found = _knn(connection, query=data[rowid], k=1,
                         options={"search_list_size": 512})
            assert found and found[0][0] == rowid
        expected = {rowid: connection.execute("SELECT embedding FROM v WHERE rowid=?",
                                              (rowid,)).fetchone()[0] for rowid in data}
        _maintenance(connection)
        assert _status(connection)["current_job"] is None
        for rowid, blob in expected.items():
            assert connection.execute("SELECT embedding FROM v WHERE rowid=?",
                                      (rowid,)).fetchone()[0] == blob
            assert _live_id(connection, rowid) == ids_before[rowid]
        for rowid in (1, 16, 32, 48):
            assert _knn(connection, query=data[rowid], k=1,
                        options={"search_list_size": 512})[0][0] == rowid
    finally:
        connection.close()


def test_diskann_updates_allocate_new_ids_and_delete_is_logical(conn):
    _seed(conn)
    old = _live_id(conn, 1)
    conn.execute("UPDATE v SET embedding=? WHERE rowid=1", (_blob([8, 1]),))
    new = _live_id(conn, 1)
    assert new > old
    assert conn.execute("SELECT public_rowid,retired_revision IS NOT NULL FROM v_diskann_entries WHERE entry_id=?", (old,)).fetchone() == (1, 1)
    assert _get(conn, 1) == (8.0, 1.0)
    conn.execute("DELETE FROM v WHERE rowid=1")
    assert _get(conn, 1) is None
    assert 1 not in {row[0] for row in _knn(conn)}
    assert conn.execute("SELECT public_rowid,retired_revision IS NOT NULL FROM v_diskann_entries WHERE entry_id=?", (new,)).fetchone() == (1, 1)
    _insert(conn, 1, [9, 1])
    assert _live_id(conn, 1) > new


def test_diskann_metadata_and_graph_share_commit_and_rollback(conn):
    _seed(conn)
    conn.execute("CREATE TABLE metadata(rowid INTEGER PRIMARY KEY,value)")
    before = _snapshot(conn)
    conn.execute("BEGIN")
    conn.execute("INSERT INTO metadata VALUES(42,'pending')")
    _insert(conn, 42, [4, 2])
    assert _get(conn, 42) == (4.0, 2.0)
    conn.execute("ROLLBACK")
    assert _snapshot(conn) == before
    assert conn.execute("SELECT * FROM metadata").fetchall() == []
    conn.execute("BEGIN")
    conn.execute("INSERT INTO metadata VALUES(42,'committed')")
    _insert(conn, 42, [4, 2])
    conn.execute("COMMIT")
    assert _get(conn, 42) == (4.0, 2.0)
    assert conn.execute("SELECT * FROM metadata").fetchall() == [(42, "committed")]


@pytest.mark.parametrize("conflict", ["", "OR IGNORE", "OR FAIL", "OR REPLACE"])
def test_diskann_failed_multirow_duplicate_aborts_statement_not_outer_transaction(conn, conflict):
    _seed(conn)
    conn.execute("CREATE TABLE metadata(value)")
    observed = _observe_allocations(conn)
    conn.execute("BEGIN")
    conn.execute("INSERT INTO metadata VALUES('prior statement')")
    _insert(conn, 55, [5, 1])
    observed.clear()
    before = _snapshot(conn)
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute(
            f"INSERT {conflict} INTO v(rowid,embedding) VALUES(100,?),(7,?)",
            (_blob([8, 1]), _blob([9, 1])),
        )
    assert observed == [100]  # Prove row1 allocated before row2 failed.
    assert conn.in_transaction
    assert _snapshot(conn) == before
    assert conn.execute("SELECT * FROM metadata").fetchall() == [("prior statement",)]
    assert _get(conn, 55) == (5.0, 1.0)
    assert _get(conn, 100) is None
    conn.execute("COMMIT")


def test_diskann_failed_multirow_vector_validation_restores_first_allocation(conn):
    _seed(conn)
    observed = _observe_allocations(conn)
    before = _snapshot(conn)
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("INSERT INTO v(rowid,embedding) VALUES(100,?),(101,?)", (_blob([8, 1]), _blob([9])))
    assert observed == [100]
    assert _snapshot(conn) == before


def test_diskann_failed_multirow_update_restores_vectors_edges_and_counters(conn):
    _seed(conn)
    observed = _observe_allocations(conn)
    conn.execute("CREATE TABLE metadata(value)")
    conn.execute("BEGIN")
    conn.execute("INSERT INTO metadata VALUES('prior')")
    before = _snapshot(conn)
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute(
            "UPDATE v SET embedding=CASE rowid WHEN 1 THEN ? ELSE ? END "
            "WHERE knn_search(embedding,knn_param(?,2))",
            (_blob([8, 1]), _blob([9]), _blob([1, 0])),
        )
    assert observed == [1]
    assert conn.in_transaction
    assert _snapshot(conn) == before
    assert conn.execute("SELECT * FROM metadata").fetchall() == [("prior",)]
    conn.execute("COMMIT")


def test_diskann_neighbor_write_failure_after_allocation_is_atomic(conn):
    _seed(conn)
    observed = _observe_allocations(conn)
    conn.execute(
        "CREATE TRIGGER fail_neighbors BEFORE INSERT ON v_diskann_current "
        "BEGIN SELECT RAISE(ABORT,'injected neighbor failure'); END"
    )
    conn.execute("BEGIN")
    before = _snapshot(conn)
    with pytest.raises(sqlite3.DatabaseError, match="injected neighbor failure"):
        _insert(conn, 100, [2, 1])
    assert observed == [100]
    assert conn.in_transaction
    assert _snapshot(conn) == before
    conn.execute("ROLLBACK")
    conn.execute("DROP TRIGGER fail_neighbors")
    _insert(conn, 100, [2, 1])
    assert _get(conn, 100) == (2.0, 1.0)


def test_diskann_nested_and_late_savepoint_enrollment(conn):
    _seed(conn)
    conn.execute("CREATE TABLE metadata(value)")
    baseline = _snapshot(conn)
    conn.execute("BEGIN")
    conn.execute("INSERT INTO metadata VALUES('survives')")
    conn.execute("SAVEPOINT early")  # Backend has not written in this transaction.
    _insert(conn, 42, [4, 2])
    outer = _snapshot(conn)
    conn.execute("SAVEPOINT inner")
    conn.execute("UPDATE v SET embedding=? WHERE rowid=1", (_blob([8, 1]),))
    conn.execute("DELETE FROM v WHERE rowid=7")
    conn.execute("ROLLBACK TO inner")
    assert _snapshot(conn) == outer
    conn.execute("RELEASE inner")
    conn.execute("ROLLBACK TO early")
    assert _snapshot(conn) == baseline
    _insert(conn, 43, [5, 2])  # Reuse rolled-back revisions safely.
    assert _get(conn, 43) == (5.0, 2.0)
    assert _get(conn, 42) is None
    conn.execute("RELEASE early")
    conn.execute("COMMIT")
    assert conn.execute("SELECT * FROM metadata").fetchall() == [("survives",)]


def test_diskann_savepoint_transaction_and_release_then_outer_rollback(conn):
    _seed(conn)
    before = _snapshot(conn)
    conn.execute("SAVEPOINT outermost")
    _insert(conn, 42, [4, 2])
    conn.execute("SAVEPOINT inner")
    conn.execute("DELETE FROM v WHERE rowid=1")
    conn.execute("RELEASE inner")
    conn.execute("ROLLBACK TO outermost")
    assert _snapshot(conn) == before
    conn.execute("RELEASE outermost")
    assert not conn.in_transaction


def test_diskann_progress_interrupt_restores_statement_state(conn):
    _seed(conn)
    conn.execute("CREATE TABLE metadata(value)")
    conn.execute("BEGIN")
    conn.execute("INSERT INTO metadata VALUES('prior')")
    before = _snapshot(conn)
    observed = []
    armed = False

    def record(rowid):
        nonlocal armed
        observed.append(rowid)
        armed = True
        return 0

    conn.create_function("arm_diskann_interrupt", 1, record)
    conn.execute(
        "CREATE TRIGGER arm_interrupt AFTER INSERT ON v_diskann_entries "
        "BEGIN SELECT arm_diskann_interrupt(NEW.public_rowid); END"
    )
    conn.set_progress_handler(lambda: int(armed), 1)
    try:
        with pytest.raises(sqlite3.DatabaseError) as error:
            _insert(conn, 100, [2, 1])
        assert error.value.sqlite_errorcode & 0xff == sqlite3.SQLITE_INTERRUPT
    finally:
        conn.set_progress_handler(None, 0)  # Never interrupt validation queries.
    assert observed == [100]
    assert _snapshot(conn) == before
    # SQLITE_INTERRUPT may roll back the entire explicit transaction, per SQLite.
    if conn.in_transaction:
        assert conn.execute("SELECT * FROM metadata").fetchall() == [("prior",)]
        conn.execute("ROLLBACK")
    else:
        assert conn.execute("SELECT * FROM metadata").fetchall() == []


def test_diskann_create_drop_and_rename_rollbacks_restore_durable_identity(conn):
    _seed(conn)
    before = _snapshot(conn)
    conn.execute("BEGIN")
    _create(conn, name="fresh")
    _insert(conn, 42, [4, 2], name="fresh")
    conn.execute("ROLLBACK")
    assert not {"fresh", "fresh_diskann_meta", "fresh_diskann_entries"} & _schema_names(conn)
    assert _snapshot(conn) == before
    conn.execute("BEGIN")
    conn.execute("DROP TABLE v")
    conn.execute("ROLLBACK")
    assert _snapshot(conn) == before
    assert _get(conn, 1) == (1.0, 0.0)
    conn.execute("BEGIN")
    conn.execute("ALTER TABLE v RENAME TO renamed")
    assert _get(conn, 1, name="renamed") == (1.0, 0.0)
    assert _snapshot(conn, name="renamed") == before
    conn.execute("ROLLBACK")
    assert "renamed_diskann_meta" not in _schema_names(conn)
    assert _snapshot(conn) == before
    assert _knn(conn, k=1)[0][0] == 1


@pytest.mark.parametrize("suffix", [*SUFFIXES])
def test_diskann_create_shadow_collision_does_not_replace_ordinary_table(conn, suffix):
    collision = "bad" + suffix
    conn.execute(f"CREATE TABLE {_q(collision)}(value)")
    conn.execute(f"INSERT INTO {_q(collision)} VALUES('keep')")
    with pytest.raises(sqlite3.DatabaseError):
        _create(conn, name="bad")
    assert conn.execute(f"SELECT * FROM {_q(collision)}").fetchall() == [("keep",)]
    assert {name for name in _schema_names(conn) if name.startswith("bad")} == {collision}


@pytest.mark.parametrize("suffix", [*SUFFIXES[1:]])
def test_diskann_partial_rename_failure_restores_all_shadows(conn, suffix):
    _seed(conn)
    collision = "renamed" + suffix
    conn.execute(f"CREATE TABLE {_q(collision)}(value)")
    conn.execute(f"INSERT INTO {_q(collision)} VALUES('collision')")
    conn.execute("CREATE TABLE metadata(value)")
    conn.execute("BEGIN")
    conn.execute("INSERT INTO metadata VALUES('prior')")
    before = _snapshot(conn)
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("ALTER TABLE v RENAME TO renamed")
    assert _snapshot(conn) == before
    assert "renamed_diskann_meta" not in _schema_names(conn)
    assert conn.execute(f"SELECT * FROM {_q(collision)}").fetchall() == [("collision",)]
    assert conn.execute("SELECT * FROM metadata").fetchall() == [("prior",)]
    conn.execute("COMMIT")
    assert _get(conn, 1) == (1.0, 0.0)


def test_diskann_quoted_names_attached_schema_and_vacuum(conn, tmp_path):
    schema = 'attached " schema'
    name = 'vectors " odd'
    conn.execute(f"ATTACH DATABASE ? AS {_q(schema)}", (str(tmp_path / "attached.db"),))
    _create(conn, name=name, schema=schema)
    _seed(conn, name=name, schema=schema)
    before = _snapshot(conn, name=name, schema=schema)
    conn.execute("VACUUM")
    conn.execute(f"VACUUM {_q(schema)}")
    assert _snapshot(conn, name=name, schema=schema) == before
    renamed = 'renamed " table'
    conn.execute(f"ALTER TABLE {_table(name, schema)} RENAME TO {_q(renamed)}")
    assert _get(conn, 7, name=renamed, schema=schema) == (0.0, 2.0)
    assert _snapshot(conn, name=renamed, schema=schema) == before
    _create(conn, name="local", schema="temp")
    _insert(conn, 42, [4, 2], name="local", schema="temp")
    assert _get(conn, 42, name="local", schema="temp") == (4.0, 2.0)


def test_diskann_reopen_readonly_and_sqlite_backup(extension_path, tmp_path):
    path = tmp_path / "vectors.db"
    connection = _open(extension_path, path)
    try:
        _create(connection)
        _seed(connection)
        before = _snapshot(connection)
        query = _knn(connection)
        target = _open(extension_path, tmp_path / "backup.db")
        try:
            connection.backup(target)
            assert _snapshot(target) == before
            assert _knn(target) == query
        finally:
            target.close()
    finally:
        connection.close()
    reopened = _open(extension_path, path)
    try:
        assert _snapshot(reopened) == before
        assert _knn(reopened) == query
    finally:
        reopened.close()
    readonly = _open(extension_path, path, readonly=True)
    try:
        assert _knn(readonly) == query
        with pytest.raises(sqlite3.DatabaseError) as error:
            _insert(readonly, 42, [4, 2])
        assert error.value.sqlite_errorcode & 0xff == sqlite3.SQLITE_READONLY
        assert _snapshot(readonly) == before
    finally:
        readonly.close()


def test_diskann_process_crash_keeps_commit_and_rolls_back_pending_graph(extension_path, tmp_path):
    path = tmp_path / "crash.db"
    connection = _open(extension_path, path)
    _create(connection)
    _seed(connection)
    connection.close()
    script = r'''
import os, sqlite3, struct, sys
connection = sqlite3.connect(sys.argv[2], isolation_level=None)
connection.enable_load_extension(True)
connection.load_extension(sys.argv[1])
connection.execute("BEGIN IMMEDIATE")
connection.execute("INSERT INTO v(rowid,embedding) VALUES(100,?)", (struct.pack('<2f', 2, 1),))
connection.execute("COMMIT")
connection.execute("BEGIN IMMEDIATE")
connection.execute("INSERT INTO v(rowid,embedding) VALUES(101,?)", (struct.pack('<2f', 8, 1),))
os._exit(73)
'''
    process = subprocess.run([sys.executable, "-c", script, str(extension_path), str(path)], capture_output=True, timeout=30)
    assert process.returncode == 73, process.stderr.decode(errors="replace")
    reopened = _open(extension_path, path)
    try:
        assert _get(reopened, 100) == (2.0, 1.0)
        assert _get(reopened, 101) is None
        assert {row[0] for row in _knn(reopened)} == {1, 7, 19}
        assert _status(reopened)["pending_count"] == 1
        _maintenance(reopened)
        assert {row[0] for row in _knn(reopened)} == {1, 7, 19, 100}
        assert reopened.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        reopened.close()


def test_diskann_wal_snapshot_old_reader_new_reader_and_cache_refresh(extension_path, tmp_path):
    path = tmp_path / "wal.db"
    reader = _open(extension_path, path)
    writer = _open(extension_path, path)
    try:
        assert reader.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        _create(reader)
        _seed(reader)
        reader.execute("BEGIN")
        before = _snapshot(reader)
        old = _knn(reader)
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("UPDATE v SET embedding=? WHERE rowid=1", (_blob([8, 1]),))
        writer.execute("DELETE FROM v WHERE rowid=7")
        _insert(writer, 42, [4, 2])
        writer.execute("COMMIT")
        assert {row[0] for row in _knn(writer)} == {19}
        _maintenance(writer)
        assert _snapshot(reader) == before
        assert _knn(reader) == old
        assert _get(reader, 1) == (1.0, 0.0)
        fresh = _open(extension_path, path)
        try:
            assert _get(fresh, 1) == (8.0, 1.0)
            assert {row[0] for row in _knn(fresh)} == {1, 19, 42}
        finally:
            fresh.close()
        reader.execute("COMMIT")
        assert _get(reader, 1) == (8.0, 1.0)
        assert {row[0] for row in _knn(reader)} == {1, 19, 42}
    finally:
        writer.close()
        reader.close()


def test_diskann_wal_busy_snapshot_error_is_not_lossy(extension_path, tmp_path):
    path = tmp_path / "busy.db"
    reader = _open(extension_path, path)
    writer = _open(extension_path, path)
    try:
        reader.execute("PRAGMA journal_mode=WAL")
        _create(reader)
        _seed(reader)
        reader.execute("BEGIN")
        assert _get(reader, 1) == (1.0, 0.0)
        _insert(writer, 42, [4, 2])
        with pytest.raises(sqlite3.DatabaseError) as error:
            _insert(reader, 43, [5, 2])
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_BUSY_SNAPSHOT
        reader.execute("ROLLBACK")
        assert _get(reader, 42) == (4.0, 2.0)
        assert _get(reader, 43) is None
    finally:
        writer.close()
        reader.close()


def test_diskann_projected_cursor_keeps_vector_and_distance_together(conn):
    _seed(conn)
    cursor = conn.execute("SELECT rowid,embedding,distance FROM v WHERE knn_search(embedding,knn_param(?,3))", (_blob([1, 0]),))
    first = cursor.fetchone()
    assert first[0] == 1
    conn.execute("UPDATE v SET embedding=? WHERE rowid=7", (_blob([9, 1]),))
    remaining = cursor.fetchall()
    assert next(_vector(blob) for rowid, blob, _ in remaining if rowid == 7) == (0.0, 2.0)
    for _, blob, distance in [first, *remaining]:
        vector = _vector(blob)
        assert distance == pytest.approx(sum((a - b) ** 2 for a, b in zip(vector, [1, 0])))
    assert _get(conn, 7) == (9.0, 1.0)


def test_diskann_foreign_schema_reparse_preserves_graph(extension_path, tmp_path):
    path = tmp_path / "ddl.db"
    connection = _open(extension_path, path)
    foreign = _open(extension_path, path)
    try:
        _create(connection)
        _seed(connection)
        before = _snapshot(connection)
        foreign.execute("CREATE TABLE ordinary(value)")
        foreign.execute("ALTER TABLE ordinary ADD COLUMN extra")
        assert _snapshot(connection) == before
        assert _knn(connection, k=1)[0][0] == 1
    finally:
        foreign.close()
        connection.close()


@pytest.mark.parametrize("operation", ["save", "load"])
def test_diskann_has_no_external_file_command(conn, tmp_path, operation):
    _seed(conn)
    path = tmp_path / "not-an-index.bin"
    before = _snapshot(conn)
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("INSERT INTO v(operation,path) VALUES(?,?)", (operation, str(path)))
    assert not path.exists()
    assert _snapshot(conn) == before


@pytest.mark.parametrize("kind", ["short-vector", "nan-vector", "short-neighbors", "oversized-neighbors", "missing-reference", "missing-frozen"])
def test_diskann_corrupt_shadow_nodes_raise_errors_not_false_absence(extension_path, tmp_path, kind):
    """Format 5 payload damage must fail closed, never EOF/false absence.

    Payload authority is split (D1/D10): the graph_nodes row stays authoritative
    for public_rowid/state/adjacency (frozen-root adjacency cases), while live
    vectors are canonical in v_diskann_entries, so the vector cases inject the
    scored canonical row (query-visible again under the lean v11 layout). A
    missing root row and a dangling adjacency reference stay corruption.
    """
    path = tmp_path / f"corrupt-{kind}.db"
    connection = _open(extension_path, path)
    _create(connection)
    _seed(connection)
    live = connection.execute(
        "SELECT native_local_id FROM v_diskann_graph_nodes WHERE entry_id=? AND state=0",
        (_live_id(connection, 1),),
    ).fetchone()[0]
    entry = _live_id(connection, 1)
    connection.close()
    raw = sqlite3.connect(path, isolation_level=None)
    try:
        raw.execute("PRAGMA ignore_check_constraints=ON")
        if kind in {"short-vector", "nan-vector"}:
            # Lean v11: the scored live vector is the canonical entries payload.
            raw.execute("UPDATE v_diskann_entries SET vector=? WHERE entry_id=?",
                        (b"\0" if kind == "short-vector" else _blob([math.nan, 0]), entry))
        elif kind == "missing-frozen":
            raw.execute("DELETE FROM v_diskann_graph_nodes WHERE state=2")
        else:
            payload = {
                "short-neighbors": b"\0",
                "oversized-neighbors": struct.pack("<512Q", *([0] * 512)),
                "missing-reference": struct.pack("<QQ", 999999, entry),
            }[kind]
            raw.execute("UPDATE v_diskann_graph_nodes SET neighbors=? WHERE state=2", (payload,))
    finally:
        raw.close()
    reopened = _open(extension_path, path)
    try:
        with pytest.raises(sqlite3.DatabaseError) as error:
            _knn(reopened, k=1)
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
    finally:
        reopened.close()


@pytest.mark.parametrize("metric", ["l2", "cosine"])
@pytest.mark.parametrize("access", ["point", "search", "filtered", "update", "snapshot"])
@pytest.mark.parametrize("payload", [
    pytest.param(b"\0", id="misaligned-length"),
    pytest.param(_blob([1.0]), id="short-dimension"),
    pytest.param(_blob([1.0, 0.0, 0.0]), id="long-dimension"),
    pytest.param("not a blob", id="text"),
    pytest.param(1, id="integer"),
    pytest.param(_blob([math.nan, 0.0]), id="nan"),
    pytest.param(_blob([0.0, math.inf]), id="positive-infinity"),
    pytest.param(_blob([-math.inf, 0.0]), id="negative-infinity"),
])
def test_diskann_stored_vector_validation_keeps_error_codes_and_atomicity(
    extension_path, metric, access, payload,
):
    connection = _open(extension_path)
    try:
        _create(connection, metric=metric)
        _seed(connection)
        if access == "snapshot":
            _control(connection, "index_prepare")
        live = _live_id(connection, 1)
        # Lean v11 payload authority (D10 row 2): live vectors are canonical in
        # v_diskann_entries on every access class — the graph route scores the
        # canonical row, and point reads, capture and mutation validation read the
        # same payload. Corrupt the table every access class actually reads.
        target = "v_diskann_entries"
        key = ("entry_id", live)
        original = connection.execute(
            f"SELECT vector FROM {target} WHERE {key[0]}=?", (key[1],),
        ).fetchone()[0]
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute(
            f"UPDATE {target} SET vector=? WHERE {key[0]}=?", (payload, key[1]),
        )
        assert connection.execute(
            f"SELECT vector FROM {target} WHERE {key[0]}=?", (key[1],),
        ).fetchone()[0] == payload, "the corruption injection must actually persist"
        before = _snapshot(connection)
        with pytest.raises(sqlite3.DatabaseError) as error:
            if access == "point":
                _get(connection, 1)
            elif access == "search":
                _knn(connection)
            elif access == "filtered":
                connection.execute(
                    "SELECT rowid,distance FROM v WHERE "
                    "knn_search(embedding,knn_param(?,2)) AND rowid IN (1,19)",
                    (_blob([1.0, 0.0]),),
                ).fetchall()
            elif access == "update":
                connection.execute("UPDATE v SET embedding=? WHERE rowid=1", (_blob([2, 1]),))
            else:
                _control(connection, "index_snapshot_step", _jf(_status(connection)["current_job"]))
        # Lean Format 5 keeps the same validation order: the reached canonical
        # entries payload read (the only live-vector authority, D10 row 2) fails as
        # corruption before any write is committed.
        expected_primary = expected_extended = sqlite3.SQLITE_CORRUPT
        assert error.value.sqlite_errorcode & 0xff == expected_primary
        assert error.value.sqlite_errorcode == expected_extended
        assert _snapshot(connection) == before
        connection.execute(
            f"UPDATE {target} SET vector=? WHERE {key[0]}=?", (original, key[1]),
        )
        _insert(connection, 99, [2.0, 1.0])
        assert _knn(connection, k=1)[0][0] == 1
    finally:
        connection.close()


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_diskann_stored_vector_not_null_guard_rejects_injection(extension_path, metric):
    connection = _open(extension_path)
    try:
        _create(connection, metric=metric)
        _seed(connection)
        live = _live_id(connection, 1)
        connection.execute("PRAGMA ignore_check_constraints=ON")
        before = _snapshot(connection)
        with pytest.raises(sqlite3.IntegrityError) as error:
            connection.execute(
                "UPDATE v_diskann_entries SET vector=NULL WHERE entry_id=?", (live,),
            )
        # Unlike CHECK, NOT NULL cannot be bypassed by the injection PRAGMA.
        # This is schema-level rejection, not a successfully injected bad vector.
        assert error.value.sqlite_errorcode & 0xff == sqlite3.SQLITE_CONSTRAINT
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_NOTNULL
        assert _snapshot(connection) == before
        _insert(connection, 99, [2.0, 1.0])
        assert _knn(connection, k=1)[0][0] == 1
    finally:
        connection.close()


@pytest.mark.parametrize("column,value", [
    ("revision", -1), ("next_entry_id", 0), ("canonical_count", -1),
    ("retired_count", -1), ("next_fence", 0), ("cleanup_entry_id", 2**63 - 1),
])
def test_diskann_corrupt_meta_counters_and_entrypoints_raise_errors(extension_path, tmp_path, column, value):
    path = tmp_path / f"meta-{column}.db"
    connection = _open(extension_path, path)
    try:
        _create(connection)
        _seed(connection)
    finally:
        connection.close()
    raw = sqlite3.connect(path, isolation_level=None)
    try:
        raw.execute("PRAGMA ignore_check_constraints=ON")
        raw.execute(f"UPDATE v_diskann_meta SET {_q(column)}=?", (value,))
    finally:
        raw.close()
    reopened = _open(extension_path, path)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            _knn(reopened)
    finally:
        reopened.close()


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_diskann_missing_shadow_table_is_not_recreated_on_connect(extension_path, tmp_path, suffix):
    path = tmp_path / "missing-shadow.db"
    connection = _open(extension_path, path)
    try:
        _create(connection)
        _seed(connection)
    finally:
        connection.close()
    raw = sqlite3.connect(path, isolation_level=None)
    try:
        raw.execute(f"DROP TABLE {_q('v' + suffix)}")
    finally:
        raw.close()
    reopened = _open(extension_path, path)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            _knn(reopened)
        assert "v" + suffix not in _schema_names(reopened)
    finally:
        reopened.close()


def test_diskann_corrupt_descriptor_cannot_be_silently_recreated(extension_path, tmp_path):
    path = tmp_path / "descriptor.db"
    connection = _open(extension_path, path)
    _create(connection)
    _seed(connection)
    connection.close()
    raw = sqlite3.connect(path, isolation_level=None)
    try:
        raw.execute("PRAGMA ignore_check_constraints=ON")
        raw.execute("UPDATE v_diskann_meta SET descriptor=?", (b"unsupported-descriptor",))
        before = _snapshot(raw)
    finally:
        raw.close()
    reopened = _open(extension_path, path)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            _knn(reopened)
        assert _snapshot(reopened) == before
    finally:
        reopened.close()


@pytest.mark.parametrize("outer_policy", ["", "OR IGNORE", "OR FAIL", "OR REPLACE"])
@pytest.mark.parametrize("raise_mode", ["ABORT", "FAIL"])
def test_diskann_single_callback_late_errors_are_atomic(conn, outer_policy, raise_mode):
    _seed(conn)
    conn.execute("CREATE TABLE metadata(value)")
    conn.execute(
        "CREATE TRIGGER late_error BEFORE INSERT ON v_diskann_current "
        f"BEGIN SELECT RAISE({raise_mode},'late callback failure'); END"
    )
    conn.execute("BEGIN")
    conn.execute("INSERT INTO metadata VALUES('prior')")
    before = _snapshot(conn)
    with pytest.raises(sqlite3.DatabaseError, match="late callback failure"):
        conn.execute(
            f"INSERT {outer_policy} INTO v(rowid,embedding) VALUES(100,?)",
            (_blob([2, 1]),),
        )
    assert conn.in_transaction
    assert _snapshot(conn) == before
    assert conn.execute("SELECT * FROM metadata").fetchall() == [("prior",)]
    conn.execute("ROLLBACK")


@pytest.mark.parametrize("target", ["entries", "current"])
def test_diskann_ignored_allocation_cannot_commit_partial_graph(conn, target):
    _seed(conn)
    conn.execute(f"CREATE TRIGGER ignore_allocation BEFORE INSERT ON v_diskann_{target} "
                 "BEGIN SELECT RAISE(IGNORE); END")
    conn.execute("BEGIN")
    before = _snapshot(conn)
    with pytest.raises(sqlite3.DatabaseError) as error:
        _insert(conn, 100, [2, 1])
    assert error.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
    assert _snapshot(conn) == before
    conn.execute("ROLLBACK")


@pytest.mark.parametrize("temporary", [False, True])
@pytest.mark.parametrize("raise_mode", ["FAIL", "IGNORE"])
def test_diskann_rejects_unexpected_carrier_triggers_before_graph_write(conn, temporary, raise_mode):
    _seed(conn)
    qualifier = "TEMP " if temporary else ""
    message = ", 'unexpected carrier trigger'" if raise_mode == "FAIL" else ""
    conn.execute(
        f"CREATE {qualifier}TRIGGER unexpected_guard AFTER UPDATE ON v_diskann_txn "
        f"BEGIN SELECT RAISE({raise_mode}{message}); END"
    )
    before = _snapshot(conn)
    with pytest.raises(sqlite3.DatabaseError):
        _insert(conn, 100, [2, 1])
    assert _snapshot(conn) == before


@pytest.mark.parametrize("damage", ["missing-row", "bad-value"])
def test_diskann_rejects_corrupt_carrier_before_allocation(conn, damage):
    _seed(conn)
    conn.execute("PRAGMA ignore_check_constraints=ON")
    if damage == "missing-row":
        conn.execute("DELETE FROM v_diskann_txn WHERE singleton=2")
    else:
        conn.execute("UPDATE v_diskann_txn SET value=1 WHERE singleton=2")
    before = _snapshot(conn)
    with pytest.raises(sqlite3.DatabaseError):
        _insert(conn, 100, [2, 1])
    assert _snapshot(conn) == before


def test_diskann_carrier_damage_during_graph_action_rolls_back(conn):
    _seed(conn)
    conn.execute(
        "CREATE TRIGGER damage_guard AFTER INSERT ON v_diskann_entries "
        "BEGIN DELETE FROM v_diskann_txn WHERE singleton=2; END"
    )
    conn.execute("BEGIN")
    before = _snapshot(conn)
    with pytest.raises(sqlite3.DatabaseError):
        _insert(conn, 100, [2, 1])
    assert _snapshot(conn) == before
    conn.execute("ROLLBACK")


def test_diskann_atomic_write_does_not_require_enabled_triggers(conn):
    # The carrier is a genuine multirow UPDATE, not a trigger-dependent wrapper.
    _seed(conn)
    conn.setconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_TRIGGER, False)
    conn.execute("BEGIN")
    before = _snapshot(conn)
    _insert(conn, 100, [2, 1])
    assert _get(conn, 100) == (2.0, 1.0)
    conn.execute("ROLLBACK")
    assert _snapshot(conn) == before
    conn.setconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_TRIGGER, True)


def test_diskann_count_changes_rows_do_not_break_atomic_writes(conn):
    conn.execute("PRAGMA count_changes=ON")
    _seed(conn)
    assert _get(conn, 1) == (1.0, 0.0)
    conn.execute(
        "CREATE TRIGGER late_count_error BEFORE INSERT ON v_diskann_current "
        "BEGIN SELECT RAISE(ABORT,'count changes failure'); END"
    )
    conn.execute("BEGIN")
    before = _snapshot(conn)
    with pytest.raises(sqlite3.DatabaseError, match="count changes failure"):
        _insert(conn, 100, [2, 1])
    assert _snapshot(conn) == before
    conn.execute("ROLLBACK")
    conn.execute("PRAGMA count_changes=OFF")


def test_diskann_journal_off_rejects_mutation_without_side_effects(extension_path, tmp_path):
    path = tmp_path / "journal-off.db"
    connection = _open(extension_path, path)
    try:
        _create(connection)
        _seed(connection)
        before = _snapshot(connection)
        assert connection.execute("PRAGMA journal_mode=OFF").fetchone()[0] == "off"
        with pytest.raises(sqlite3.DatabaseError, match="journal"):
            _insert(connection, 100, [2, 1])
        assert _snapshot(connection) == before
        assert _get(connection, 1) == (1.0, 0.0)
        connection.execute("PRAGMA journal_mode=DELETE")
        _insert(connection, 100, [2, 1])
        assert _get(connection, 100) == (2.0, 1.0)
    finally:
        connection.close()


@pytest.mark.parametrize("argument", [None, 1, 4096, b"not a tagged pointer", "0x1234"])
def test_diskann_batch_rejects_sql_values_as_native_pointers(conn, argument):
    before = _snapshot(conn)
    with pytest.raises(sqlite3.Error) as error:
        conn.execute("INSERT INTO v(operation,embedding) VALUES('insert_batch',?)", (argument,))
    assert error.value.sqlite_errorcode == sqlite3.SQLITE_MISUSE
    assert _snapshot(conn) == before


def test_diskann_private_atomic_function_cannot_be_called_from_sql(conn):
    before = _snapshot(conn)
    for argument in (None, 0, b"not a live pointer"):
        with pytest.raises(sqlite3.Error) as error:
            conn.execute("SELECT vectorlite_atomic_write(?,1)", (argument,)).fetchall()
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_MISUSE
        assert _snapshot(conn) == before
    conn.execute("CREATE VIEW forbidden_atomic AS SELECT vectorlite_atomic_write(NULL,1)")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("SELECT * FROM forbidden_atomic").fetchall()
    assert _snapshot(conn) == before


def test_diskann_batch_read_detects_missing_dangling_neighbor_fail_closed(extension_path):
    connection = _open(extension_path)
    try:
        _create(connection)
        for rowid in range(1, 25):
            _insert(connection, rowid, [float(rowid), 0.5])
        _maintenance(connection)
        generation = connection.execute(
            "SELECT published_generation FROM v_diskann_meta"
        ).fetchone()[0]
        connection.execute("UPDATE v_diskann_graph_nodes SET neighbors=? WHERE state=2",
                           (_slot_payload(connection, generation, (0, 4, 23)),))
        connection.execute("DELETE FROM v_diskann_graph_nodes WHERE native_local_id=4")
        with pytest.raises(sqlite3.DatabaseError) as error:
            _knn(connection, k=3, query=[5.0, 0.5])
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
    finally:
        connection.close()


@pytest.mark.parametrize("mode", ["cross-table", "attached-aux", "dynamic-ddl", "incremental-blob"])
def test_diskann_reentrant_trace_callbacks_and_cross_table_triggers_invalidate_caches(extension_path, mode):
    connection = _open(extension_path)
    try:
        _create(connection)
        for rowid in range(1, 9):
            _insert(connection, rowid, [float(rowid), 0.0])
        _maintenance(connection)
        _control(connection, "index_prepare")
        if mode in ("cross-table", "attached-aux"):
            if mode == "attached-aux":
                connection.execute("ATTACH ':memory:' AS aux")
            target = "aux.app" if mode == "attached-aux" else "app"
            connection.execute(f"CREATE TABLE {target}(event TEXT)")
            connection.execute(f"CREATE TEMP TRIGGER app_poison AFTER INSERT ON {target} "
                               "BEGIN UPDATE v_diskann_entries SET vector=X'0000c07f00000000' WHERE entry_id=2; END")
        fired = False
        unchanged_total_changes = False
        def trace_hook(sql):
            nonlocal fired, unchanged_total_changes
            if fired or 'INSERT INTO "main"."v_diskann_snapshot_map"' not in sql:
                return
            fired = True
            if mode == "dynamic-ddl":
                connection.execute("CREATE TRIGGER dynamic_poison AFTER INSERT ON v_diskann_snapshot_map "
                                   "BEGIN UPDATE v_diskann_entries SET vector=X'0000c07f00000000' WHERE entry_id=2; END")
            elif mode in ("cross-table", "attached-aux"):
                connection.execute(f"INSERT INTO {target} VALUES('capture')")
            else:
                before_changes = connection.total_changes
                with connection.blobopen("v_diskann_entries", "vector", 2) as blob:
                    blob.write(bytes.fromhex("0000c07f00000000"))
                unchanged_total_changes = connection.total_changes == before_changes
        connection.execute("BEGIN")
        before = _snapshot(connection)
        connection.set_trace_callback(trace_hook)
        try:
            with pytest.raises(sqlite3.DatabaseError, match="non-finite") as error:
                _control(connection, "index_snapshot_step", _jf(_status(connection)["current_job"]))
        finally:
            connection.set_trace_callback(None)
        assert fired
        if mode == "incremental-blob":
            assert unchanged_total_changes
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
        assert _snapshot(connection) == before
        connection.execute("ROLLBACK")
    finally:
        connection.close()


def test_diskann_large_k_effective_beam_preserves_scratch_headroom(extension_path):
    connection = _open(extension_path)
    try:
        _create(
            connection,
            options="diskann(degree=4,build_list_size=8,search_list_size=8,cache_bytes=8400000,max_visits=32768)",
        )
        connection.execute("BEGIN")
        for i in range(1, 2401):
            _insert(connection, i, [i / 2400.0, (i % 31) / 31.0])
        connection.execute("COMMIT")
        _maintenance(connection)
        # Format 5 additionally charges current-version and entry/graph reads.
        # Double the old 4.2 MB bound for this larger work allowance.
        # Both omitted override and explicit override below `k` expand to
        # effective beam `l = 2000` in core search and must not fail `TOOBIG`
        # due to optional cache carving.
        omitted = _knn(connection, query=(0.5, 0.5), k=2000)
        assert len(omitted) == 2000
        below_k = _knn(connection, query=(0.5, 0.5), k=2000, options={"search_list_size": 4})
        assert below_k == omitted
    finally:
        connection.close()


@pytest.mark.parametrize("journal", ["WAL", "DELETE"])
def test_diskann_public_deferred_lifecycle_and_current_version_visibility(extension_path, tmp_path, journal):
    path = tmp_path / f"lifecycle-{journal}.db"
    application = _open(extension_path, path)
    worker = _open(extension_path, path)
    try:
        assert application.execute(f"PRAGMA journal_mode={journal}").fetchone()[0] == journal.lower()
        _create(application)
        for rowid, vector in BASE_ROWS:
            _insert(application, rowid, vector)
        s = _status(application)
        assert (s["canonical_count"], s["eligible_count"], s["pending_count"]) == (3, 0, 3)
        assert _knn(application) == []
        assert application.execute("SELECT rowid FROM v WHERE knn_search(embedding,knn_param(?,99)) AND rowid IN (1,7)", (_blob([1, 0]),)).fetchall() == []
        job, candidate = _candidate(worker, records=1)
        assert _status(application)["current_job"]["phase"] == 2
        assert _knn(application) == []
        _finish(worker, job, candidate, records=1, publish=False)
        assert _status(application)["current_job"]["phase"] == 4
        assert _knn(application) == []
        _control(worker, "index_publish", {**_jf(job), "candidate_token": candidate["candidate_token"]})
        assert {r[0] for r in _knn(application)} == {1, 7, 19}
        assert _status(application)["pending_count"] == 0
        application.execute("UPDATE v SET embedding=? WHERE rowid=1", (_blob([8, 1]),))
        assert _get(application, 1) == (8.0, 1.0)
        assert {r[0] for r in _knn(application)} == {7, 19}
        assert application.execute("SELECT rowid FROM v WHERE knn_search(embedding,knn_param(?,99)) AND rowid=1", (_blob([8, 1]),)).fetchall() == []
        application.execute("DELETE FROM v WHERE rowid=7")
        assert _get(application, 7) is None
        assert {r[0] for r in _knn(application)} == {19}
        assert _status(application)["pending_count"] == 1
        _maintenance(worker, records=1)
        assert {r[0] for r in _knn(application)} == {1, 19}
        assert _knn(application, query=[8, 1], k=1) == [(1, 0.0)]
        assert _status(application)["pending_count"] == 0
        assert application.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        worker.close()
        application.close()


@pytest.mark.parametrize("journal", ["WAL", "DELETE"])
def test_diskann_build_limit_keeps_published_generation_and_pending_rows(extension_path, tmp_path, journal):
    path = tmp_path / f"limit-{journal}.db"
    application = _open(extension_path, path)
    worker = _open(extension_path, path)
    try:
        application.execute(f"PRAGMA journal_mode={journal}")
        _create(application)
        _seed(application)
        old = _status(application)["published_generation"]
        _insert(application, 42, [4, 2])
        _control(worker, "index_prepare")
        job = _capture(worker, records=1)
        _control(worker, "index_claim_build", _jf(job))
        before = _snapshot(worker)
        with pytest.raises(sqlite3.Error) as error:
            _build(worker, job, budget=1)
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_TOOBIG
        assert _snapshot(worker) == before
        s = _status(application)
        assert s["published_generation"] == old and s["pending_count"] == 1
        assert s["current_job"]["import_cursor"] == 0
        assert {r[0] for r in _knn(application)} == {1, 7, 19}
        assert _get(application, 42) == (4.0, 2.0)
    finally:
        worker.close()
        application.close()
    # A fixed connection-total admission budget is not enlarged in place.
    worker = _open(extension_path, path)
    try:
        _control(worker, "index_restart", {**_jf(job), "force": True})
        new = _capture(worker)
        assert new["fence"] > job["fence"]
        _control(worker, "index_claim_build", _jf(new))
        _finish(worker, new, _build(worker, new))
        assert _status(worker)["pending_count"] == 0
    finally:
        worker.close()


@pytest.mark.parametrize("recovery", ["cancel", "restart"])
def test_diskann_lost_ram_competing_workers_and_stale_fence(extension_path, tmp_path, recovery):
    path = tmp_path / f"fence-{recovery}.db"
    first = _open(extension_path, path)
    second = _open(extension_path, path)
    try:
        _create(first)
        _seed(first)
        _insert(first, 42, [4, 2])
        old, candidate = _candidate(first)
        _control(first, "index_import_step", {**_jf(old), "candidate_token": candidate["candidate_token"], "max_records": 1})
        before = _snapshot(first)
        for operation in ("index_claim_build", "index_restart"):
            with pytest.raises(sqlite3.Error) as error:
                _control(second, operation, _jf(old))
            assert error.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
        with pytest.raises(sqlite3.Error) as error:
            _build(second, old)
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
        assert _snapshot(first) == before
        first.close()  # SQL survives; the partial native candidate does not.
        with pytest.raises(sqlite3.Error) as error:
            _control(second, "index_import_step", {**_jf(old), "candidate_token": candidate["candidate_token"]})
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
        _control(second, "index_" + recovery, {**_jf(old), "force": True})
        if recovery == "cancel":
            assert _status(second)["current_job"] is None
            _control(second, "index_prepare")
        new = _capture(second)
        assert new["job_id"] > old["job_id"] and new["fence"] > old["fence"]
        assert new["generation_id"] > old["generation_id"]
        with pytest.raises(sqlite3.Error) as error:
            _control(second, "index_claim_build", _jf(old))
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
        _control(second, "index_claim_build", _jf(new))
        _finish(second, new, _build(second, new), records=1)
        assert {r[0] for r in _knn(second)} == {1, 7, 19, 42}
    finally:
        first.close()
        second.close()


def test_diskann_readonly_cannot_build_and_repeated_load_preserves_ready_candidate(extension_path, tmp_path):
    path = tmp_path / "readonly-build.db"
    worker = _open(extension_path, path)
    try:
        _create(worker)
        _seed(worker)
        _insert(worker, 42, [4, 2])
        job, candidate = _candidate(worker)
        before = _snapshot(worker)
        with pytest.raises(sqlite3.Error, match="load.once|fresh.connection") as error:
            worker.load_extension(str(extension_path))
        # sqlite3.load_extension raises a wrapper OperationalError without code attributes.
        assert isinstance(error.value, sqlite3.OperationalError)
        assert _snapshot(worker) == before
        assert _build(worker, job)["candidate_token"] == candidate["candidate_token"]
        readonly = _open(extension_path, path, readonly=True)
        try:
            assert _status(readonly)["current_job"]["phase"] == 2
            with pytest.raises(sqlite3.Error) as error:
                _build(readonly, job)
            assert error.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
            assert _snapshot(readonly) == before
        finally:
            readonly.close()
        _finish(worker, job, candidate)
    finally:
        worker.close()


@pytest.mark.parametrize("stage", ["snapshot", "import", "validate", "publish"])
def test_diskann_maintenance_savepoint_release_and_outer_rollback(conn, stage):
    _seed(conn)
    _insert(conn, 42, [4, 2])
    if stage == "snapshot":
        _control(conn, "index_prepare")
        job = _status(conn)["current_job"]
        operation, opts = "index_snapshot_step", _jf(job)
    else:
        job, candidate = _candidate(conn)
        opts = {**_jf(job), "candidate_token": candidate["candidate_token"]}
        operation = "index_" + ("validate_step" if stage == "validate" else "import_step" if stage == "import" else "publish")
        if stage == "validate":
            _control(conn, "index_import_step", opts)
        elif stage == "publish":
            _finish(conn, job, candidate, publish=False)
    before = _snapshot(conn)
    old_results = _knn(conn)
    conn.execute("BEGIN")
    conn.execute("SAVEPOINT maintenance")
    _control(conn, operation, opts)
    assert _snapshot(conn) != before
    conn.execute("RELEASE maintenance")
    conn.execute("ROLLBACK")
    assert _snapshot(conn) == before and _knn(conn) == old_results
    _control(conn, operation, opts)  # RAM is immutable; rolled-back SQL cursor is authoritative.


@pytest.mark.parametrize("metric", ["l2", "cosine"])
@pytest.mark.parametrize("fault", ["entry", "counter"])
def test_diskann_canonical_checkpoint_corruption_after_write_rolls_back(extension_path, metric, fault):
    connection = _open(extension_path)
    try:
        _create(connection, metric=metric)
        _seed(connection)
        observed = _observe_allocations(connection)
        connection.execute("PRAGMA ignore_check_constraints=ON")
        when = "AFTER INSERT ON v_diskann_entries" if fault == "entry" else "AFTER UPDATE OF canonical_count ON v_diskann_meta"
        connection.execute(f"CREATE TRIGGER damage_checkpoint {when} "
                           "BEGIN UPDATE v_diskann_meta SET next_fence=0; END")
        connection.execute("CREATE TABLE metadata(value)")
        connection.execute("BEGIN")
        connection.execute("INSERT INTO metadata VALUES('prior')")
        before = _snapshot(connection)
        with pytest.raises(sqlite3.Error) as error:
            _insert(connection, 100, [2, 1])
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
        assert observed == [100]
        assert _snapshot(connection) == before
        assert connection.execute("SELECT * FROM metadata").fetchall() == [("prior",)]
        connection.execute("ROLLBACK")
    finally:
        connection.close()


@pytest.mark.parametrize("fault", ["map-rowid", "map-entry", "map-missing", "chain"])
def test_diskann_snapshot_map_corruption_fails_before_native_or_import(conn, fault):
    _seed(conn)
    _control(conn, "index_prepare")
    job = _capture(conn)
    _control(conn, "index_claim_build", _jf(job))
    if fault == "map-rowid":
        conn.execute("UPDATE v_diskann_snapshot_map SET public_rowid=99 WHERE ordinal=0")
    elif fault == "map-entry":
        conn.execute("UPDATE v_diskann_snapshot_map SET entry_id=99 WHERE ordinal=0")
    elif fault == "map-missing":
        conn.execute("DELETE FROM v_diskann_snapshot_map WHERE ordinal=0")
    else:
        conn.execute("UPDATE v_diskann_jobs SET snapshot_chain=zeroblob(32)")
    before = _snapshot(conn)
    with pytest.raises(sqlite3.Error) as error:
        _build(conn, job)
    assert error.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
    assert _snapshot(conn) == before
    assert _status(conn)["current_job"]["import_cursor"] == 0


@pytest.mark.parametrize("column,value", [("state", 99), ("root_local_id", 99999),
    ("eligible_count", -1), ("total_count", 1)])
def test_diskann_generation_state_and_entrypoint_corruption_fail_closed(conn, column, value):
    _seed(conn)
    conn.execute("PRAGMA ignore_check_constraints=ON")
    conn.execute(f"UPDATE v_diskann_generations SET {_q(column)}=?", (value,))
    before = _snapshot(conn)
    with pytest.raises(sqlite3.Error) as error:
        _knn(conn, k=1)
    assert error.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
    assert _snapshot(conn) == before


@pytest.mark.parametrize("raise_mode", ["ABORT", "FAIL", "IGNORE"])
def test_diskann_import_failure_keeps_old_generation_and_chunk_atomic(conn, raise_mode):
    _seed(conn)
    old_results = _knn(conn)
    _insert(conn, 42, [4, 2])
    job, candidate = _candidate(conn)
    observed = []
    conn.create_function("observe_import", 0, lambda: observed.append(True) or 0)
    conn.execute("CREATE TRIGGER observe_graph AFTER INSERT ON v_diskann_graph_nodes "
                 "BEGIN SELECT observe_import(); END")
    message = "" if raise_mode == "IGNORE" else ",'late import failure'"
    conn.execute("CREATE TRIGGER fail_import BEFORE INSERT ON v_diskann_graph_nodes "
                 f"WHEN NEW.native_local_id=1 BEGIN SELECT RAISE({raise_mode}{message}); END")
    conn.execute("BEGIN")
    before = _snapshot(conn)
    with pytest.raises(sqlite3.Error) as error:
        _control(conn, "index_import_step", {**_jf(job), "candidate_token": candidate["candidate_token"]})
    assert error.value.sqlite_errorcode & 0xff == (sqlite3.SQLITE_BUSY if raise_mode == "IGNORE" else sqlite3.SQLITE_CONSTRAINT)
    assert observed
    assert _snapshot(conn) == before and _knn(conn) == old_results
    conn.execute("ROLLBACK")
    conn.execute("DROP TRIGGER fail_import")
    _finish(conn, job, candidate)


@pytest.mark.parametrize("qualifier", ["", "TEMP "])
def test_diskann_capture_trigger_payload_poison_is_not_cached(conn, qualifier):
    _seed(conn)
    _control(conn, "index_prepare")
    poisoned = []
    conn.create_function("record_poison", 0, lambda: poisoned.append(True) or 0)
    conn.execute(f"CREATE {qualifier}TRIGGER poison_capture AFTER INSERT ON v_diskann_snapshot_map "
                 "BEGIN SELECT record_poison(); UPDATE v_diskann_entries SET vector=X'0000c07f00000000' WHERE entry_id=2; END")
    conn.execute("BEGIN")
    before = _snapshot(conn)
    with pytest.raises(sqlite3.Error, match="non-finite") as error:
        _control(conn, "index_snapshot_step", _jf(_status(conn)["current_job"]))
    assert poisoned and error.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
    assert _snapshot(conn) == before
    conn.execute("ROLLBACK")


@pytest.mark.parametrize("journal", ["WAL", "DELETE"])
@pytest.mark.parametrize("stage", ["capture", "compute", "mid-import", "validated", "publish", "gc"])
def test_diskann_process_exit_maintenance_recovery(extension_path, tmp_path, journal, stage):
    path = tmp_path / f"maintenance-crash-{journal}-{stage}.db"
    connection = _open(extension_path, path)
    connection.execute(f"PRAGMA journal_mode={journal}")
    _create(connection)
    _seed(connection)
    old_generation = _status(connection)["published_generation"]
    _insert(connection, 100, [2, 1])
    connection.close()
    script = r'''
import os, runpy, sys
h = runpy.run_path(sys.argv[3])
db = h["_open"](sys.argv[1], sys.argv[2])
control, status, jf = h["_control"], h["_status"], h["_jf"]
stage = sys.argv[4]
control(db, "index_prepare")
job = status(db)["current_job"]
if stage == "capture":
    control(db, "index_snapshot_step", {**jf(job), "max_records": 1})
    db.execute("BEGIN IMMEDIATE")
    control(db, "index_snapshot_step", {**jf(job), "max_records": 1})
    os._exit(73)
job = h["_capture"](db, records=1)
control(db, "index_claim_build", jf(job))
candidate = h["_build"](db, job)
opts = {**jf(job), "candidate_token": candidate["candidate_token"]}
if stage == "compute":
    os._exit(73)
if stage == "mid-import":
    control(db, "index_import_step", {**opts, "max_records": 1})
    db.execute("BEGIN IMMEDIATE")
    control(db, "index_import_step", {**opts, "max_records": 1})
    os._exit(73)
h["_finish"](db, job, candidate, records=1, publish=False)
if stage == "validated":
    os._exit(73)
if stage == "publish":
    db.execute("BEGIN IMMEDIATE")
    control(db, "index_publish", opts)
    os._exit(73)
control(db, "index_publish", opts)
control(db, "index_cleanup_step", {"max_records": 1})
db.execute("BEGIN IMMEDIATE")
control(db, "index_cleanup_step", {"max_records": 1})
os._exit(73)
'''
    process = subprocess.run([sys.executable, "-c", script, str(extension_path), str(path),
                              str(Path(__file__).resolve()), stage], capture_output=True, timeout=30)
    assert process.returncode == 73, process.stderr.decode(errors="replace")
    readonly = _open(extension_path, path, readonly=True)
    try:
        assert readonly.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        s = _status(readonly)
        assert s["canonical_count"] == 4
        if stage == "gc":
            assert s["current_job"] is None and s["pending_count"] == 0
            assert s["published_generation"] > old_generation
            assert {r[0] for r in _knn(readonly)} == {1, 7, 19, 100}
        else:
            assert s["published_generation"] == old_generation and s["pending_count"] == 1
            assert {r[0] for r in _knn(readonly)} == {1, 7, 19}
            job = s["current_job"]
            assert job["phase"] == {"capture": 0, "compute": 2, "mid-import": 2,
                                     "validated": 4, "publish": 4}[stage]
            if stage == "capture":
                assert job["captured_count"] == 1
            if stage == "mid-import":
                assert job["import_cursor"] == 1
    finally:
        readonly.close()
    reopened = _open(extension_path, path)
    try:
        if stage != "gc":
            if stage == "capture":
                new = _capture(reopened, records=1)  # Unclaimed durable capture is directly resumable.
            else:
                with pytest.raises(sqlite3.Error):
                    _build(reopened, job)  # Never stitch fresh RAM into partially imported SQL.
                _control(reopened, "index_restart", {**_jf(job), "force": True})
                new = _capture(reopened, records=1)
                assert new["fence"] > job["fence"] and new["generation_id"] > job["generation_id"]
            _control(reopened, "index_claim_build", _jf(new))
            _finish(reopened, new, _build(reopened, new), records=1)
        for _ in range(30):
            _control(reopened, "index_cleanup_step", {"max_records": 1})
        assert _status(reopened)["pending_count"] == 0
        assert {r[0] for r in _knn(reopened)} == {1, 7, 19, 100}
        assert reopened.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        reopened.close()


def test_diskann_format5_round_trip_reopen_query_and_literal(extension_path, tmp_path):
    """T1 defect class: a Format-5 database reopens and queries successfully.

    The stored meta literal is exactly 5 and the status JSON reports format 5.
    A close/reopen of the same file must answer the same KNN query, prove a clean
    integrity check, and reject a literal that no longer matches the compiled
    Format-5 layouts instead of silently misreading it. A literal-mismatch
    regression therefore fails this test rather than passing through.
    """
    path = tmp_path / "format5-round-trip.db"
    connection = _open(extension_path, path)
    try:
        _create(connection)
        _seed(connection)
        _insert(connection, 42, [4, 2])
        _maintenance(connection)
    finally:
        connection.close()
    reopened = _open(extension_path, path, readonly=True)
    try:
        assert reopened.execute("SELECT format_version FROM v_diskann_meta").fetchone() == (5,)
        assert _status(reopened)["format"] == 5
        assert _knn(reopened, query=[4.0, 2.0], k=1) == [(42, 0.0)]
        assert reopened.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        reopened.close()
    raw = sqlite3.connect(path, isolation_level=None)
    try:
        raw.execute("PRAGMA ignore_check_constraints=ON")
        raw.execute("UPDATE v_diskann_meta SET format_version=4")
    finally:
        raw.close()
    mismatched = _open(extension_path, path)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="unsupported"):
            _get(mismatched, 42)
    finally:
        mismatched.close()


@pytest.mark.parametrize("cache_bytes", [4096, 24576])
def test_diskann_canonical_workspace_rejects_before_shadow_ddl(extension_path, cache_bytes):
    connection = _open(extension_path)
    try:
        attempted = []
        def authorize(operation, name, _column, _schema, _source):
            if operation == sqlite3.SQLITE_CREATE_TABLE and name in {"v" + s for s in SUFFIXES}:
                attempted.append(name)
            return sqlite3.SQLITE_OK
        connection.set_authorizer(authorize)
        with pytest.raises(sqlite3.Error) as error:
            _create(connection, options=f"diskann(degree=4,build_list_size=8,cache_bytes={cache_bytes},max_visits=128)")
        connection.set_authorizer(None)
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_TOOBIG
        assert attempted == []
        assert not ({"v"} | {"v" + s for s in SUFFIXES}) & _schema_names(connection)
        # D6 pins the canonical floor at 26,824 bytes for table "v" (dim=2), so
        # cache_bytes=24576 still rejects and cache_bytes=27000 still accepts;
        # the ninth layout is excluded from the canonical floor name fold.
        _create(connection, options="diskann(degree=4,build_list_size=8,cache_bytes=27000,max_visits=128)")
        _insert(connection, 1, [1, 0])
        assert _get(connection, 1) == (1.0, 0.0)
        before = _snapshot(connection)
        with pytest.raises(sqlite3.Error, match="query workspace") as error:
            _knn(connection)
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_TOOBIG
        assert _snapshot(connection) == before
        assert connection.execute("SELECT count(*) FROM v_diskann_graph_nodes").fetchone() == (0,)
    finally:
        connection.close()


def test_diskann_retired_navigation_bridge_is_immutable_and_gc_pinned(conn):
    for rowid in range(32):
        _insert(conn, rowid, [float(rowid), 0])
    _maintenance(conn)
    # Deliberately controlled valid directed topology, not a user import API.
    # Native frozen root N has a one-way incoming edge to soon-retired node0.
    generation = conn.execute("SELECT published_generation FROM v_diskann_meta").fetchone()[0]
    for local in range(33):
        neighbors = [0] if local == 32 else [1] if local == 0 else [local - 1, local + 1] if local < 31 else [30]
        conn.execute("UPDATE v_diskann_graph_nodes SET neighbors=? WHERE native_local_id=?",
                     (_slot_payload(conn, generation, neighbors), local))
    graph_before = conn.execute("SELECT * FROM v_diskann_graph_nodes ORDER BY native_local_id").fetchall()
    retired_entry = _live_id(conn, 0)
    conn.execute("DELETE FROM v WHERE rowid=0")
    assert conn.execute("SELECT * FROM v_diskann_graph_nodes ORDER BY native_local_id").fetchall() == graph_before
    assert _knn(conn, query=[1, 0], k=1, options={"search_list_size": 64}) == [(1, 0.0)]
    for _ in range(10):
        _control(conn, "index_cleanup_step", {"max_records": 1})
    assert conn.execute("SELECT count(*) FROM v_diskann_entries WHERE entry_id=?", (retired_entry,)).fetchone() == (1,)
    _maintenance(conn)
    for _ in range(200):
        _control(conn, "index_cleanup_step", {"max_records": 1})
    assert conn.execute("SELECT count(*) FROM v_diskann_entries WHERE entry_id=?", (retired_entry,)).fetchone() == (0,)
    assert _knn(conn, query=[1, 0], k=1) == [(1, 0.0)]


def test_diskann_read_trace_incremental_blob_poison_is_not_masked_by_store_cache(conn):
    _seed(conn)
    _control(conn, "index_prepare")
    fired = False
    unchanged_total_changes = False
    def trace_hook(sql):
        nonlocal fired, unchanged_total_changes
        if not fired and 'SELECT public_rowid,created_revision,retired_revision,' in sql:
            fired = True
            before_changes = conn.total_changes
            with conn.blobopen("v_diskann_entries", "vector", 1) as blob:
                blob.write(bytes.fromhex("0000c07f00000000"))
            unchanged_total_changes = conn.total_changes == before_changes
    conn.execute("BEGIN")
    before = _snapshot(conn)
    job = _status(conn)["current_job"]
    conn.set_trace_callback(trace_hook)
    try:
        with pytest.raises(sqlite3.Error, match="non-finite") as error:
            _control(conn, "index_snapshot_step", _jf(job))
    finally:
        conn.set_trace_callback(None)
    assert fired and unchanged_total_changes
    assert error.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
    assert _snapshot(conn) == before
    conn.execute("ROLLBACK")


@pytest.mark.parametrize("metric", ["l2", "cosine"])
@pytest.mark.parametrize("fault", ["job-counter", "generation-state", "carrier"])
def test_diskann_capture_checkpoint_corruption_rolls_back_map_and_progress(extension_path, metric, fault):
    connection = _open(extension_path)
    try:
        _create(connection, metric=metric)
        _seed(connection)
        _control(connection, "index_prepare")
        damage = {"job-counter": "UPDATE v_diskann_jobs SET captured_count=-1",
                  "generation-state": "UPDATE v_diskann_generations SET state=99 WHERE state=0",
                  "carrier": "DELETE FROM v_diskann_txn WHERE singleton=2"}[fault]
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute("CREATE TRIGGER damage_capture AFTER INSERT ON v_diskann_snapshot_map "
                           f"BEGIN {damage}; END")
        connection.execute("CREATE TABLE metadata(value)")
        connection.execute("BEGIN")
        connection.execute("INSERT INTO metadata VALUES('prior')")
        before = _snapshot(connection)
        job = _status(connection)["current_job"]
        with pytest.raises(sqlite3.Error) as error:
            _control(connection, "index_snapshot_step", {**_jf(job), "max_records": 1})
        expected = sqlite3.SQLITE_BUSY if fault == "job-counter" else sqlite3.SQLITE_CORRUPT
        assert error.value.sqlite_errorcode == expected
        assert _snapshot(connection) == before
        assert connection.execute("SELECT * FROM metadata").fetchall() == [("prior",)]
        connection.execute("ROLLBACK")
    finally:
        connection.close()


def test_diskann_validation_rejects_imported_order_corruption_without_progress(conn):
    _seed(conn)
    _insert(conn, 42, [4, 2])
    job, candidate = _candidate(conn)
    opts = {**_jf(job), "candidate_token": candidate["candidate_token"]}
    _control(conn, "index_import_step", opts)
    generation = job["generation_id"]
    rows = conn.execute("SELECT native_local_id,neighbors FROM v_diskann_graph_nodes WHERE generation_id=? ORDER BY native_local_id", (generation,)).fetchall()
    local, original = next((local, neighbors) for local, neighbors in rows if len(neighbors) >= 32)
    slots = [original[offset : offset + 16] for offset in range(0, len(original), 16)]
    changed = b"".join(reversed(slots))
    assert changed != original
    conn.execute("UPDATE v_diskann_graph_nodes SET neighbors=? WHERE generation_id=? AND native_local_id=?", (changed, generation, local))
    before = _snapshot(conn)
    with pytest.raises(sqlite3.Error) as error:
        _control(conn, "index_validate_step", opts)
    assert error.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
    assert _snapshot(conn) == before
    assert _status(conn)["current_job"]["validation_cursor"] == 0
    assert {r[0] for r in _knn(conn)} == {1, 7, 19}
    conn.execute("UPDATE v_diskann_graph_nodes SET neighbors=? WHERE generation_id=? AND native_local_id=?", (original, generation, local))
    _finish(conn, job, candidate)
