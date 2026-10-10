"""DiskANN SQL/storage regressions against the shared freshly built extension.

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
    )


def _seed(connection, name="v", schema="main"):
    for rowid, vector in BASE_ROWS:
        _insert(connection, rowid, vector, name, schema)


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


def _snapshot(connection, name="v", schema="main"):
    # Cover every descriptor/counter/entrypoint/edge, independent of meta names.
    return tuple(
        tuple(connection.execute(
            f"SELECT * FROM {_table(name + suffix, schema)} ORDER BY 1"
        ).fetchall())
        for suffix in ("_diskann_meta", "_diskann_nodes", "_diskann_txn", "_diskann_rebuild")
    )


def _schema_names(connection, schema="main"):
    return {row[0] for row in connection.execute(
        f"SELECT name FROM {_q(schema)}.sqlite_schema WHERE type='table'"
    )}


def _live_id(connection, rowid):
    return connection.execute(
        "SELECT node_id FROM v_diskann_nodes WHERE public_rowid=? AND state=0", (rowid,)
    ).fetchone()[0]


def _observe_allocations(connection):
    observed = []

    def record(rowid):
        observed.append(rowid)
        return 0

    connection.create_function("record_diskann_allocation", 1, record)
    connection.execute(
        "CREATE TRIGGER record_alloc AFTER INSERT ON v_diskann_nodes "
        "WHEN NEW.state=0 AND NEW.public_rowid IS NOT NULL BEGIN "
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
    assert not {"bad", "bad_diskann_meta", "bad_diskann_nodes"} & _schema_names(conn)
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
    assert not {"bad", "bad_diskann_meta", "bad_diskann_nodes"} & _schema_names(conn)


def test_diskann_tiny_cache_rejects_before_frozen_or_user_allocation(extension_path):
    connection = _open(extension_path)
    try:
        # The syntactic minimum cannot admit even the fixed SQL plan workspace.
        with pytest.raises(sqlite3.DatabaseError) as error:
            _create(connection, options="diskann(degree=4,build_list_size=8,search_list_size=8,alpha=1,cache_bytes=4096,max_visits=128)")
        assert error.value.sqlite_errorcode & 0xff == sqlite3.SQLITE_TOOBIG
        assert not {"v", "v_diskann_meta", "v_diskann_nodes", "v_diskann_txn", "v_diskann_rebuild"} & _schema_names(connection)
        # Enough for the fixed plans, but not the admitted algorithm scratch.
        _create(connection, options="diskann(degree=4,build_list_size=8,search_list_size=8,alpha=1,cache_bytes=32768,max_visits=128)")
        before = _snapshot(connection)
        observed = _observe_allocations(connection)
        with pytest.raises(sqlite3.DatabaseError) as error:
            _insert(connection, 1, [1, 0])
        assert error.value.sqlite_errorcode & 0xff == sqlite3.SQLITE_TOOBIG
        assert observed == []
        assert _snapshot(connection) == before
        assert connection.execute("SELECT count(*) FROM v_diskann_nodes").fetchone() == (0,)
    finally:
        connection.close()


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
        assert all(math.isfinite(distance) for _, distance in _knn(connection, query=valid))
    finally:
        connection.close()


def test_diskann_sparse_rowids_filter_and_internal_ids(conn):
    labels = [0, 2**40 + 7, 2**63 - 1]
    for i, label in enumerate(labels):
        _insert(conn, label, [i + 1, 0])
    assert _get(conn, labels[-1]) == (3.0, 0.0)
    assert {row[0] for row in _knn(conn, k=10)} == set(labels)
    rows = conn.execute(
        "SELECT rowid,distance FROM v WHERE knn_search(embedding,knn_param(?,1)) "
        "AND rowid IN (?,?)", (_blob([1, 0]), labels[1], labels[2])
    ).fetchall()
    assert rows == [(labels[1], 1.0)]
    ids = [row[0] for row in conn.execute("SELECT node_id FROM v_diskann_nodes WHERE state=0")]
    assert all(1 <= node_id <= len(labels) for node_id in ids)
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
    assert _knn(conn, query=[8, 1], k=1) == [(1, 0.0)]


def test_diskann_retired_one_way_bridge_keeps_live_results_reachable(conn):
    _seed(conn)
    bridge = _live_id(conn, 1)
    target = _live_id(conn, 7)
    spare = _live_id(conn, 19)
    # Controlled legal directed topology. The bridge has an incoming frozen
    # edge but no reverse edge to the frozen node, the case OneHop can miss.
    for node, neighbors in [(0, [bridge]), (bridge, [target]),
                            (target, [bridge, spare]), (spare, [target])]:
        conn.execute("UPDATE v_diskann_nodes SET neighbors=? WHERE node_id=?",
                     (struct.pack(f"<{len(neighbors)}Q", *neighbors), node))
    conn.execute("DELETE FROM v WHERE rowid=1")
    assert conn.execute("SELECT neighbors FROM v_diskann_nodes WHERE node_id=?",
                        (bridge,)).fetchone()[0] == struct.pack("<Q", target)
    assert _knn(conn, query=[0, 2], k=1) == [(7, 0.0)]
    conn.execute("INSERT INTO v(operation) VALUES('consolidate')")
    assert _knn(conn, query=[0, 2], k=1) == [(7, 0.0)]
    assert conn.execute("SELECT count(*) FROM v_diskann_rebuild").fetchone() == (0,)
    assert conn.execute("SELECT count(*) FROM v_diskann_nodes WHERE state=1").fetchone() == (0,)
    assert _live_id(conn, 7) > spare


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
        ids_before = {rowid: _live_id(connection, rowid) for rowid in data}
        # High L on this bounded dataset exercises graph traversal, not the
        # k>=population exact path, even with many retained retired vectors.
        for rowid in (1, 16, 32, 48):
            found = _knn(connection, query=data[rowid], k=1,
                         options={"search_list_size": 512})
            assert found and found[0][0] == rowid
        expected = {rowid: connection.execute("SELECT embedding FROM v WHERE rowid=?",
                                              (rowid,)).fetchone()[0] for rowid in data}
        connection.execute("INSERT INTO v(operation) VALUES('consolidate')")
        assert connection.execute("SELECT count(*) FROM v_diskann_rebuild").fetchone() == (0,)
        for rowid, blob in expected.items():
            assert connection.execute("SELECT embedding FROM v WHERE rowid=?",
                                      (rowid,)).fetchone()[0] == blob
            assert _live_id(connection, rowid) > max(ids_before.values())
        for rowid in (1, 16, 32, 48):
            assert _knn(connection, query=data[rowid], k=1,
                        options={"search_list_size": 512})[0][0] == rowid
    finally:
        connection.close()


def test_diskann_failed_rebuild_restores_entire_old_graph_and_empty_staging(conn):
    _seed(conn)
    conn.execute("DELETE FROM v WHERE rowid=7")
    conn.execute(
        "CREATE TRIGGER fail_rebuild BEFORE INSERT ON v_diskann_nodes "
        "WHEN NEW.state=0 BEGIN SELECT RAISE(ABORT,'rebuild insertion failure'); END"
    )
    conn.execute("BEGIN")
    before = _snapshot(conn)
    with pytest.raises(sqlite3.DatabaseError, match="rebuild insertion failure"):
        conn.execute("INSERT INTO v(operation) VALUES('consolidate')")
    assert _snapshot(conn) == before
    assert connection_is_staging_empty(conn)
    conn.execute("ROLLBACK")
    conn.execute("DROP TRIGGER fail_rebuild")
    assert _knn(conn, k=1)[0][0] == 1


def connection_is_staging_empty(connection):
    return connection.execute("SELECT count(*) FROM v_diskann_rebuild").fetchone() == (0,)


def test_diskann_updates_allocate_new_ids_and_delete_is_logical(conn):
    _seed(conn)
    old = _live_id(conn, 1)
    conn.execute("UPDATE v SET embedding=? WHERE rowid=1", (_blob([8, 1]),))
    new = _live_id(conn, 1)
    assert new > old
    assert conn.execute("SELECT public_rowid,state FROM v_diskann_nodes WHERE node_id=?", (old,)).fetchone() == (None, 1)
    assert _get(conn, 1) == (8.0, 1.0)
    conn.execute("DELETE FROM v WHERE rowid=1")
    assert _get(conn, 1) is None
    assert 1 not in {row[0] for row in _knn(conn)}
    assert conn.execute("SELECT public_rowid,state FROM v_diskann_nodes WHERE node_id=?", (new,)).fetchone() == (None, 1)
    _insert(conn, 1, [9, 1])
    assert _live_id(conn, 1) > new


def test_diskann_consolidation_is_transactional_and_removes_tombstone_references(conn):
    _seed(conn)
    conn.execute("DELETE FROM v WHERE rowid=7")
    conn.execute("UPDATE v SET embedding=? WHERE rowid=1", (_blob([2, 1]),))
    before = _snapshot(conn)
    conn.execute("BEGIN")
    conn.execute("INSERT INTO v(operation) VALUES('consolidate')")
    assert conn.execute("SELECT count(*) FROM v_diskann_nodes WHERE state=1").fetchone() == (0,)
    conn.execute("ROLLBACK")
    assert _snapshot(conn) == before
    conn.execute("INSERT INTO v(operation) VALUES('consolidate')")
    assert conn.execute("SELECT count(*) FROM v_diskann_nodes WHERE state=1").fetchone() == (0,)
    nodes = conn.execute("SELECT node_id,neighbors FROM v_diskann_nodes").fetchall()
    present = {node_id for node_id, _ in nodes}
    for _, neighbors in nodes:
        assert len(neighbors) % 8 == 0
        assert set(struct.unpack(f"<{len(neighbors) // 8}Q", neighbors)) <= present
    assert {row[0] for row in _knn(conn)} == {1, 19}


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
        "CREATE TRIGGER fail_neighbors BEFORE UPDATE OF neighbors ON v_diskann_nodes "
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
        "CREATE TRIGGER arm_interrupt AFTER INSERT ON v_diskann_nodes "
        "WHEN NEW.state=0 BEGIN SELECT arm_diskann_interrupt(NEW.public_rowid); END"
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
    assert not {"fresh", "fresh_diskann_meta", "fresh_diskann_nodes"} & _schema_names(conn)
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


@pytest.mark.parametrize("suffix", ["_diskann_meta", "_diskann_nodes", "_diskann_txn", "_diskann_rebuild"])
def test_diskann_create_shadow_collision_does_not_replace_ordinary_table(conn, suffix):
    collision = "bad" + suffix
    conn.execute(f"CREATE TABLE {_q(collision)}(value)")
    conn.execute(f"INSERT INTO {_q(collision)} VALUES('keep')")
    with pytest.raises(sqlite3.DatabaseError):
        _create(conn, name="bad")
    assert conn.execute(f"SELECT * FROM {_q(collision)}").fetchall() == [("keep",)]
    assert {name for name in _schema_names(conn) if name.startswith("bad")} == {collision}


@pytest.mark.parametrize("suffix", ["_diskann_nodes", "_diskann_txn", "_diskann_rebuild"])
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
    path = tmp_path / f"corrupt-{kind}.db"
    connection = _open(extension_path, path)
    _create(connection)
    _seed(connection)
    live = _live_id(connection, 1)
    connection.close()
    # Corrupt with an ordinary SQLite connection, then force a fresh xConnect.
    raw = sqlite3.connect(path, isolation_level=None)
    try:
        raw.execute("PRAGMA ignore_check_constraints=ON")
        if kind == "short-vector":
            raw.execute("UPDATE v_diskann_nodes SET vector=? WHERE node_id=?", (b"\0", live))
        elif kind == "nan-vector":
            raw.execute("UPDATE v_diskann_nodes SET vector=? WHERE node_id=?", (_blob([math.nan, 0]), live))
        elif kind == "short-neighbors":
            raw.execute("UPDATE v_diskann_nodes SET neighbors=? WHERE node_id=0", (b"\0",))
        elif kind == "oversized-neighbors":
            raw.execute("UPDATE v_diskann_nodes SET neighbors=? WHERE node_id=0", (struct.pack("<512Q", *([live] * 512)),))
        elif kind == "missing-reference":
            raw.execute("UPDATE v_diskann_nodes SET neighbors=? WHERE node_id=0", (struct.pack("<Q", 999999),))
        else:
            raw.execute("DELETE FROM v_diskann_nodes WHERE node_id=0")
    finally:
        raw.close()
    reopened = _open(extension_path, path)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            if kind in {"short-vector", "nan-vector"}:
                _get(reopened, 1)
            else:
                _knn(reopened)
    finally:
        reopened.close()


@pytest.mark.parametrize("column,value", [
    ("revision", -1), ("next_node_id", 0), ("live_count", -1),
    ("deleted_count", -1), ("entrypoints", b"\x01"),
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


def test_diskann_missing_shadow_table_is_not_recreated_on_connect(extension_path, tmp_path):
    path = tmp_path / "missing-shadow.db"
    connection = _open(extension_path, path)
    try:
        _create(connection)
        _seed(connection)
    finally:
        connection.close()
    raw = sqlite3.connect(path, isolation_level=None)
    try:
        raw.execute("DROP TABLE v_diskann_nodes")
    finally:
        raw.close()
    reopened = _open(extension_path, path)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            _knn(reopened)
        assert "v_diskann_nodes" not in _schema_names(reopened)
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
        "CREATE TRIGGER late_error BEFORE UPDATE OF neighbors ON v_diskann_nodes "
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


@pytest.mark.parametrize("target_state", [0, 2])
def test_diskann_ignored_allocation_cannot_commit_partial_graph(conn, target_state):
    if target_state == 0:
        _seed(conn)
    conn.execute(
        "CREATE TRIGGER ignore_allocation BEFORE INSERT ON v_diskann_nodes "
        f"WHEN NEW.state={target_state} BEGIN SELECT RAISE(IGNORE); END"
    )
    conn.execute("BEGIN")
    before = _snapshot(conn)
    with pytest.raises(sqlite3.DatabaseError):
        _insert(conn, 100, [2, 1])
    assert _snapshot(conn) == before
    # SQLite may mark the transaction corruption-readonly after CORRUPT; end it
    # rather than demanding recovery writes inside that same failed transaction.
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
        "CREATE TRIGGER damage_guard AFTER INSERT ON v_diskann_nodes "
        "WHEN NEW.state=0 BEGIN DELETE FROM v_diskann_txn WHERE singleton=2; END"
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
        "CREATE TRIGGER late_count_error BEFORE UPDATE OF neighbors ON v_diskann_nodes "
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
