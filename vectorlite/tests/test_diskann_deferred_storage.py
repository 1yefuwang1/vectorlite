"""Format5 canonical deferred storage against an explicitly fresh library.

Direct shadow edits below are corruption/import-protocol fault injection only.
No test claims completed build/publication or weakens immediate-index suites.
"""
import os
import sqlite3
import struct
import sys
from pathlib import Path

import pytest

SUFFIXES = ("meta", "entries", "current", "generations", "graph_nodes", "generation_retired", "jobs", "snapshot_map", "txn")


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


def open_db(path=":memory:", readonly=False):
    library = artifact()
    assert library.is_file(), f"build or install the extension first: {library}"
    db = sqlite3.connect(f"file:{path}?mode=ro" if readonly else str(path), isolation_level=None, uri=readonly)
    db.enable_load_extension(True)
    db.load_extension(str(library))
    return db


def create(db, name="v", schema="main", metric="l2"):
    db.execute(f'CREATE VIRTUAL TABLE "{schema}"."{name}" USING vectorlite(embedding float32[2] {metric},diskann())')


def insert(db, rowid, values=(1, 2), name="v", schema="main"):
    db.execute(f'INSERT INTO "{schema}"."{name}"(rowid,embedding) VALUES(?,?)', (rowid, blob(values)))


def get(db, rowid, name="v", schema="main"):
    row = db.execute(f'SELECT embedding FROM "{schema}"."{name}" WHERE rowid=?', (rowid,)).fetchone()
    return None if row is None else row[0]


def knn(db, filter_sql="", name="v", schema="main", k=10):
    return db.execute(f'SELECT rowid,embedding,distance FROM "{schema}"."{name}" WHERE knn_search(embedding,knn_param(?,?)) {filter_sql}', (blob((1, 2)), k)).fetchall()


def snapshot(db):
    return tuple(tuple(db.execute(f'SELECT * FROM v_diskann_{s} ORDER BY 1,2')) for s in SUFFIXES)


@pytest.fixture
def db():
    db = open_db()
    create(db)
    yield db
    db.close()


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_pending_point_projection_empty_knn(metric):
    db = open_db()
    create(db, metric=metric)
    insert(db, 0, (3, 4))
    insert(db, 9223372036854775807, (0, 0))
    actual = struct.unpack("<2f", get(db, 0))
    assert actual == pytest.approx((3, 4) if metric == "l2" else (.6, .8))
    assert get(db, 9223372036854775807) == blob((0, 0))
    for filter_sql in ("", "AND rowid=0", "AND rowid IN (0,9223372036854775807)"):
        assert knn(db, filter_sql) == []
        # Existing SQL knn_param contract rejects k=0 before backend routing.
        with pytest.raises(sqlite3.DatabaseError, match="k should be greater than 0"):
            knn(db, filter_sql, k=0)
    assert db.execute("SELECT format_version,canonical_count,retired_count,published_generation,candidate_generation FROM v_diskann_meta").fetchone() == (5, 2, 0, None, None)
    assert db.execute("SELECT count(*) FROM v_diskann_graph_nodes").fetchone() == (0,)
    db.close()


def test_format5_seven_column_node_rows_and_ninth_layout(db):
    """Format 5 lean row: seven columns, root-only payload, and the ninth layout.

    The published node row order is fixed as (generation_id, native_local_id,
    entry_id, public_rowid, state, neighbors, vector). Live rows (state=0) store
    SQL NULL for the vector, whose canonical payload stays in diskann_entries;
    only the frozen root (state=2) carries its 8-byte vector blob. The retired-ids
    layout is registered as the ninth shadow with its compound WITHOUT ROWID
    primary key. The meta literal is exactly 5.
    """
    assert db.execute("SELECT format_version FROM v_diskann_meta").fetchone() == (5,)
    columns = [row[1] for row in db.execute("PRAGMA table_info(v_diskann_graph_nodes)")]
    assert columns == [
        "generation_id",
        "native_local_id",
        "entry_id",
        "public_rowid",
        "state",
        "neighbors",
        "vector",
    ]
    retired_columns = [row[1] for row in db.execute("PRAGMA table_info(v_diskann_generation_retired)")]
    assert retired_columns == ["generation_id", "entry_id"]
    assert db.execute(
        "SELECT count(*) FROM main.sqlite_schema WHERE type='table' AND name='v_diskann_generation_retired'"
    ).fetchone() == (1,)
    insert(db, 7, (1, 2))
    import_fixture(db, state=1)
    live = db.execute(
        "SELECT entry_id,public_rowid,state,neighbors,vector FROM v_diskann_graph_nodes WHERE state=0"
    ).fetchone()
    assert live[0] is not None and live[1] == 7 and live[2] == 0
    assert live[3] == b""
    # Lean v11 payload authority: the live row stores SQL NULL; the canonical
    # vector lives in diskann_entries (D1/D10).
    assert live[4] is None
    root = db.execute(
        "SELECT entry_id,public_rowid,state,length(neighbors),length(vector) FROM v_diskann_graph_nodes WHERE state=2"
    ).fetchone()
    assert root == (None, None, 2, 0, 8)
    assert db.execute(
        "SELECT vector FROM v_diskann_entries WHERE entry_id=?", (live[0],)
    ).fetchone() == (blob((1, 2)),)
    # The root-only payload rule is schema-enforced: a live row with a vector and a
    # frozen root without one are both rejected by the v5 CHECK.
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO v_diskann_graph_nodes VALUES(1,10,1,7,0,X'',?)", (blob((1, 2)),)
        )
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO v_diskann_graph_nodes VALUES(1,11,NULL,NULL,2,X'',NULL)")
    assert db.execute("SELECT count(*) FROM v_diskann_generation_retired").fetchone() == (0,)


def test_history_original_ids_immutable_versions_and_intervals(db):
    insert(db, 7, (1, 2))
    original = db.execute("SELECT entry_id,public_rowid,created_revision,vector FROM v_diskann_entries").fetchone()
    db.execute("UPDATE v SET embedding=? WHERE rowid=7", (blob((3, 4)),))
    db.execute("DELETE FROM v WHERE rowid=7")
    insert(db, 7, (5, 6))
    rows = db.execute("SELECT entry_id,public_rowid,created_revision,retired_revision,vector FROM v_diskann_entries ORDER BY entry_id").fetchall()
    assert rows == [(1, 7, 1, 2, blob((1, 2))), (2, 7, 2, 3, blob((3, 4))), (3, 7, 4, None, blob((5, 6)))]
    assert original == rows[0][:3] + (rows[0][4],)
    assert db.execute("SELECT * FROM v_diskann_current").fetchall() == [(7, 3)]
    assert db.execute("SELECT revision,next_entry_id,canonical_count,retired_count FROM v_diskann_meta").fetchone() == (4, 4, 1, 2)
    assert knn(db) == []


@pytest.mark.parametrize("action", ["insert", "update", "delete"])
def test_transaction_savepoint_and_outer_rollback(db, action):
    insert(db, 7)
    before = snapshot(db)
    db.execute("BEGIN")
    db.execute("SAVEPOINT s")
    if action == "insert":
        insert(db, 8)
    elif action == "update":
        db.execute("UPDATE v SET embedding=? WHERE rowid=7", (blob((4, 5)),))
    else:
        db.execute("DELETE FROM v WHERE rowid=7")
    db.execute("ROLLBACK TO s")
    assert snapshot(db) == before
    insert(db, 9)
    db.execute("RELEASE s")
    db.execute("ROLLBACK")
    assert snapshot(db) == before


@pytest.mark.parametrize("action", ["insert", "update", "delete"])
@pytest.mark.parametrize("raise_kind", ["ABORT", "FAIL", "IGNORE"])
def test_late_nested_failure_rolls_whole_carrier_back(db, action, raise_kind):
    insert(db, 7)
    before = snapshot(db)
    db.execute(f"CREATE TRIGGER late BEFORE UPDATE ON v_diskann_meta BEGIN SELECT RAISE({raise_kind}{'' if raise_kind == 'IGNORE' else ", 'original late SQL error'"}); END")
    with pytest.raises(sqlite3.DatabaseError) as error:
        if action == "insert":
            insert(db, 8)
        elif action == "update":
            db.execute("UPDATE v SET embedding=? WHERE rowid=7", (blob((4, 5)),))
        else:
            db.execute("DELETE FROM v WHERE rowid=7")
    if raise_kind != "IGNORE":
        assert "original late SQL error" in str(error.value)
    assert snapshot(db) == before


def test_whole_multirow_insert_duplicate_and_numeric_failure(db):
    before = snapshot(db)
    for vectors in ((blob((1, 2)), blob((float("nan"), 3))), (blob((1, 2)), blob((1e19, 0)))):
        with pytest.raises(sqlite3.DatabaseError):
            db.execute("INSERT INTO v(rowid,embedding) VALUES(1,?),(2,?)", vectors)
        assert snapshot(db) == before
    with pytest.raises(sqlite3.DatabaseError):
        db.execute("INSERT INTO v(rowid,embedding) VALUES(1,?),(1,?)", (blob((1, 2)), blob((3, 4))))
    assert snapshot(db) == before


def test_payload_header_checks_and_unsupported_format_before_writes(db):
    insert(db, 7)
    db.execute("PRAGMA ignore_check_constraints=ON")
    db.execute("UPDATE v_diskann_entries SET vector=zeroblob(1048576)")
    with pytest.raises(sqlite3.DatabaseError, match="byte length"):
        get(db, 7)
    # Pending-only KNN never opens any canonical vector payload.
    assert knn(db) == []
    db.execute("UPDATE v_diskann_meta SET format_version=3")
    before = snapshot(db)
    with pytest.raises(sqlite3.DatabaseError, match="unsupported"):
        insert(db, 8)
    assert snapshot(db) == before


def test_old_format_reopen_clear_readonly_rejection(tmp_path):
    path = tmp_path / "old.db"
    db = open_db(path)
    create(db)
    db.execute("PRAGMA ignore_check_constraints=ON")
    db.execute("UPDATE v_diskann_meta SET format_version=4")
    db.close()
    db = open_db(path, readonly=True)
    # The connect-time format check is fail-closed before any mutation; the
    # message names the unsupported stored format version.
    with pytest.raises(sqlite3.DatabaseError, match="unsupported"):
        get(db, 7)
    db.close()


def test_readonly_reopen_attachment_backup_and_no_changes(tmp_path):
    path = tmp_path / "store.db"
    db = open_db(path)
    create(db)
    insert(db, 7)
    db.close()
    db = open_db(path, readonly=True)
    assert get(db, 7) == blob((1, 2))
    assert knn(db) == []
    assert db.total_changes == 0
    db.close()
    db = open_db()
    db.execute("ATTACH ? AS aux", (str(path),))
    assert get(db, 7, schema="aux") == blob((1, 2))
    create(db, "tempv", "temp")
    insert(db, 9, name="tempv", schema="temp")
    assert get(db, 9, name="tempv", schema="temp") == blob((1, 2))
    db.close()


def test_journal_off_rejected_without_pragma_changes(db):
    before = snapshot(db)
    db.execute("PRAGMA journal_mode=OFF")
    with pytest.raises(sqlite3.DatabaseError, match="journal_mode=OFF"):
        insert(db, 8)
    assert db.execute("PRAGMA journal_mode").fetchone() == ("off",)
    assert snapshot(db) == before


def test_guard_schema_and_post_action_damage_rollback(db):
    before = snapshot(db)
    db.execute("CREATE TRIGGER damage AFTER INSERT ON v_diskann_entries BEGIN DELETE FROM v_diskann_txn WHERE singleton=2; END")
    with pytest.raises(sqlite3.DatabaseError, match="two unchanged rows"):
        insert(db, 8)
    assert snapshot(db) == before
    db.execute("DROP TRIGGER damage")
    db.execute("CREATE TEMP TRIGGER bad BEFORE UPDATE ON v_diskann_txn BEGIN SELECT RAISE(IGNORE); END")
    with pytest.raises(sqlite3.DatabaseError, match="triggers"):
        insert(db, 8)
    assert snapshot(db) == before


def import_fixture(db, state=0, generation=1, entry=1):
    # Minimal schema fixture for the reverse-map contract, not real publication.
    db.execute("INSERT INTO v_diskann_generations VALUES(?,1,?,1,zeroblob(32),zeroblob(32),1,1,1)", (generation, state))
    # Lean v11 node row: the live row stores SQL NULL; the vector stays canonical.
    db.execute("INSERT INTO v_diskann_graph_nodes SELECT ?,0,entry_id,public_rowid,0,X'',NULL FROM v_diskann_entries WHERE entry_id=?", (generation, entry))
    db.execute("INSERT INTO v_diskann_graph_nodes VALUES(?,1,NULL,NULL,2,X'',?)", (generation, blob((1, 2))))
    db.execute(
        "INSERT INTO v_diskann_jobs(job_id,generation_id,parent_generation,cutoff_revision,"
        "fence,owner_nonce,phase,expected_count,scan_entry_id,captured_count,import_cursor,"
        "validation_cursor,cleanup_cursor,source_hash,config_hash,candidate_hash,failure_code,"
        "cutoff_entry_id,snapshot_chain) "
        "VALUES(1,?,NULL,1,1,NULL,3,1,1,1,1,0,0,zeroblob(32),zeroblob(32),NULL,NULL,1,zeroblob(32))",
        (generation,),
    )
    if state == 0:
        db.execute("UPDATE v_diskann_meta SET next_job_id=2,next_generation_id=?,candidate_generation=?,active_job=1,next_fence=2", (generation + 1, generation))
    else:
        db.execute("UPDATE v_diskann_meta SET next_job_id=2,next_generation_id=?,published_generation=?,next_fence=2", (generation + 1, generation))


@pytest.mark.parametrize("state", [0, 1])
@pytest.mark.parametrize("action", ["update", "delete"])
def test_imported_candidate_and_published_eligibility_retirement(db, state, action):
    insert(db, 7)
    import_fixture(db, state)
    db.execute("BEGIN")
    if action == "update":
        db.execute("UPDATE v SET embedding=? WHERE rowid=7", (blob((3, 4)),))
    else:
        db.execute("DELETE FROM v WHERE rowid=7")
    assert db.execute("SELECT eligible_count FROM v_diskann_generations").fetchone() == (0,)
    db.execute("ROLLBACK")
    assert db.execute("SELECT eligible_count FROM v_diskann_generations").fetchone() == (1,)
    if state == 1:
        # A published generation is searchable through the T6 adapter; the fixture's
        # single live node is the current version of rowid 7 with vector (1,2).
        rows = knn(db)
        assert [row[0] for row in rows] == [7]
        assert rows[0][1] == blob((1, 2))
        assert rows[0][2] == pytest.approx(0.0)
    else:
        # A staged candidate is never searchable, even with an eligible node.
        assert knn(db) == []


def test_compound_native_zero_live_root_n_and_active_job_ddl_policy(db):
    insert(db, 7)
    import_fixture(db)
    assert db.execute("SELECT native_local_id,state FROM v_diskann_graph_nodes ORDER BY native_local_id").fetchall() == [(0, 0), (1, 2)]
    before = snapshot(db)
    for sql in ("ALTER TABLE v RENAME TO vv", "DROP TABLE v"):
        with pytest.raises(sqlite3.DatabaseError) as error:
            db.execute(sql)
        # SQLite may replace xRename SQLITE_BUSY text with its own message.
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
        assert snapshot(db) == before


def test_rename_rollback_and_drop_all_suffixes(db):
    insert(db, 7)
    db.execute("BEGIN")
    db.execute("ALTER TABLE v RENAME TO vv")
    assert get(db, 7, name="vv") == blob((1, 2))
    db.execute("ROLLBACK")
    assert get(db, 7) == blob((1, 2))
    db.execute("ALTER TABLE v RENAME TO vv")
    assert get(db, 7, name="vv") == blob((1, 2))
    db.execute("DROP TABLE vv")
    assert not db.execute("SELECT name FROM sqlite_schema WHERE name LIKE 'vv%'").fetchall()


def native_sqlite_library():
    """Path to a shared libsqlite3 with extension loading, or None if unavailable.

    CPython's driver and this host probe are separate SQLite instances; the probe
    needs a shared library that exports sqlite3_enable_load_extension. macOS ships a
    system copy that omits it, so an explicit VECTORLITE_TEST_SQLITE_LIBRARY (for
    example a libsqlite3 built from the amalgamation) is required there.
    """
    import ctypes
    import ctypes.util

    # Auto-discovery is POSIX-only: on Windows ctypes.util.find_library can return
    # an unrelated sqlite3.dll whose ABI does not match this process, which crashed
    # the probe. Windows therefore requires the explicit override.
    if sys.platform == "win32":
        library = os.environ.get("VECTORLITE_TEST_SQLITE_LIBRARY")
    else:
        library = os.environ.get("VECTORLITE_TEST_SQLITE_LIBRARY") or ctypes.util.find_library("sqlite3")
    if not library:
        return None
    try:
        api = ctypes.CDLL(library)
    except OSError:
        return None
    return library if hasattr(api, "sqlite3_enable_load_extension") else None


@pytest.mark.parametrize("chunk_size", [1, 8, 32])
@pytest.mark.parametrize("failure", [None, "late_duplicate", "late_numeric", "late_sql"])
def test_native_tagged_batch_canonical_whole_input_rollback(chunk_size, failure):
    if native_sqlite_library() is None:
        pytest.skip(
            "no shared libsqlite3 with sqlite3_enable_load_extension; "
            "set VECTORLITE_TEST_SQLITE_LIBRARY to run the native tagged-batch cases"
        )
    # The extension's host API table is process-local. Never mix native engines
    # and the Python driver's different SQLite table in a single process.
    import subprocess
    import sys
    result = subprocess.run([sys.executable, __file__, "--native", str(chunk_size), failure or "none"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def native_batch_qualification(chunk_size, failure):
    # Open a separately selected native host through its public C API; do not
    # extract or share Python Connection pointers. Native allocations and the
    # tagged descriptor stay alive and immutable through step/finalize.
    import ctypes as c
    import ctypes.util

    library = native_sqlite_library()
    assert library, "native tagged-batch host requires a shared libsqlite3 with extension loading"
    api = c.CDLL(library)
    api.sqlite3_libversion.restype = c.c_char_p
    version = tuple(map(int, api.sqlite3_libversion().decode().split(".")))
    assert version >= (3, 38, 0), version
    ptr = c.c_void_p
    for name, args, result in (
        ("sqlite3_open", [c.c_char_p, c.POINTER(ptr)], c.c_int),
        ("sqlite3_close", [ptr], c.c_int),
        ("sqlite3_enable_load_extension", [ptr, c.c_int], c.c_int),
        ("sqlite3_load_extension", [ptr, c.c_char_p, c.c_char_p, c.POINTER(ptr)], c.c_int),
        ("sqlite3_exec", [ptr, c.c_char_p, ptr, ptr, c.POINTER(ptr)], c.c_int),
        ("sqlite3_prepare_v2", [ptr, c.c_char_p, c.c_int, c.POINTER(ptr), ptr], c.c_int),
        ("sqlite3_step", [ptr], c.c_int),
        ("sqlite3_finalize", [ptr], c.c_int),
        ("sqlite3_bind_pointer", [ptr, c.c_int, ptr, c.c_char_p, ptr], c.c_int),
        ("sqlite3_column_int64", [ptr, c.c_int], c.c_int64),
    ):
        fn = getattr(api, name)
        fn.argtypes, fn.restype = args, result

    class Batch(c.Structure):
        _fields_ = [("abi_version", c.c_uint32), ("struct_size", c.c_uint32),
                    ("count", c.c_uint64), ("dimension", c.c_uint64),
                    ("vectors", c.POINTER(c.c_float)), ("rowids", c.POINTER(c.c_int64))]

    handle = ptr()
    assert api.sqlite3_open(b":memory:", c.byref(handle)) == 0

    def execute(sql):
        assert api.sqlite3_exec(handle, sql.encode(), None, None, None) == 0, sql

    def integer(sql):
        stmt = ptr()
        assert api.sqlite3_prepare_v2(handle, sql.encode(), -1, c.byref(stmt), None) == 0
        try:
            assert api.sqlite3_step(stmt) == 100
            return api.sqlite3_column_int64(stmt, 0)
        finally:
            assert api.sqlite3_finalize(stmt) == 0

    try:
        assert api.sqlite3_enable_load_extension(handle, 1) == 0
        assert api.sqlite3_load_extension(handle, str(artifact()).encode(), None, None) == 0
        execute("CREATE VIRTUAL TABLE v USING vectorlite(embedding float32[2] l2,diskann())")
        if failure == "late_sql":
            execute("CREATE TRIGGER late BEFORE UPDATE ON v_diskann_meta BEGIN SELECT RAISE(FAIL,'native late failure'); END")
        rows = list(range(65))
        values = [float(i % 17) for i in range(130)]
        if failure == "late_duplicate":
            rows[-1] = rows[0]
        if failure == "late_numeric":
            values[-1] = float("nan")
        rowids = (c.c_int64 * 65)(*rows)
        vectors = (c.c_float * 130)(*values)
        batch = Batch(1, c.sizeof(Batch), 65, 2, vectors, rowids)
        stmt = ptr()
        sql = f"""INSERT INTO v(operation,embedding,path) VALUES('insert_batch',?1,'{{"batch_size":{chunk_size}}}')"""
        assert api.sqlite3_prepare_v2(handle, sql.encode(), -1, c.byref(stmt), None) == 0
        try:
            tag = b"vectorlite.batch.f32.v1"
            assert api.sqlite3_bind_pointer(stmt, 1, c.cast(c.pointer(batch), ptr), tag, None) == 0
            rc = api.sqlite3_step(stmt)
            assert (rc == 101) if failure is None else (rc not in (100, 101))
        finally:
            finish = api.sqlite3_finalize(stmt)
            assert (finish == 0) if failure is None else (finish != 0)
        expected = 65 if failure is None else 0
        assert integer("SELECT canonical_count FROM v_diskann_meta") == expected
        assert integer("SELECT count(*) FROM v_diskann_entries") == expected
        assert integer("SELECT count(*) FROM v_diskann_current") == expected
        assert integer("SELECT revision FROM v_diskann_meta") == (1 if failure is None else 0)
        assert integer("SELECT next_entry_id FROM v_diskann_meta") == expected + 1
        assert integer("SELECT count(*) FROM v_diskann_graph_nodes") == 0
        assert integer("SELECT count(*) FROM v_diskann_txn WHERE value=0") == 2
    finally:
        assert api.sqlite3_close(handle) == 0


def test_retirement_updates_published_and_staged_counts_without_history_scan(db):
    insert(db, 7)
    import_fixture(db, state=1)
    db.execute("INSERT INTO v_diskann_generations VALUES(2,2,0,1,zeroblob(32),zeroblob(32),1,1,1)")
    # Lean v11 node row: live rows store SQL NULL; the vector stays canonical (D1).
    db.execute("INSERT INTO v_diskann_graph_nodes SELECT 2,0,entry_id,public_rowid,0,X'',NULL FROM v_diskann_entries WHERE entry_id=1")
    db.execute(
        "INSERT INTO v_diskann_jobs(job_id,generation_id,parent_generation,cutoff_revision,"
        "fence,owner_nonce,phase,expected_count,scan_entry_id,captured_count,import_cursor,"
        "validation_cursor,cleanup_cursor,source_hash,config_hash,candidate_hash,failure_code,"
        "cutoff_entry_id,snapshot_chain) "
        "VALUES(2,2,1,1,1,NULL,3,1,1,1,1,0,0,zeroblob(32),zeroblob(32),NULL,NULL,1,zeroblob(32))"
    )
    db.execute("UPDATE v_diskann_meta SET next_job_id=3,next_generation_id=3,candidate_generation=2,active_job=2")
    plan = db.execute("EXPLAIN QUERY PLAN SELECT native_local_id,state FROM v_diskann_graph_nodes WHERE generation_id=2 AND entry_id=1").fetchall()
    assert any("SEARCH" in row[3] and "entry_id=?" in row[3] for row in plan)
    db.execute("UPDATE v SET embedding=? WHERE rowid=7", (blob((3, 4)),))
    assert db.execute("SELECT eligible_count FROM v_diskann_generations ORDER BY generation_id").fetchall() == [(0,), (0,)]
    assert db.execute("SELECT entry_id,public_rowid FROM v_diskann_entries ORDER BY entry_id").fetchall() == [(1, 7), (2, 7)]


def test_retirement_before_candidate_node_import_preserves_counter(db):
    insert(db, 7)
    import_fixture(db)
    db.execute("DELETE FROM v_diskann_graph_nodes WHERE state=0")
    db.execute("UPDATE v_diskann_generations SET eligible_count=0")
    db.execute("DELETE FROM v WHERE rowid=7")
    assert db.execute("SELECT eligible_count FROM v_diskann_generations").fetchone() == (0,)
    assert db.execute("SELECT public_rowid,retired_revision FROM v_diskann_entries").fetchone() == (7, 2)


@pytest.mark.parametrize("damage", ["missing_root", "nonfinite_root", "bad_adjacency", "job_count"])
def test_bounded_open_validates_published_root_and_active_job(tmp_path, damage):
    path = tmp_path / "damage.db"
    db = open_db(path)
    create(db)
    insert(db, 7)
    import_fixture(db, state=0 if damage == "job_count" else 1)
    db.execute("PRAGMA ignore_check_constraints=ON")
    if damage == "missing_root":
        db.execute("DELETE FROM v_diskann_graph_nodes WHERE state=2")
    elif damage == "nonfinite_root":
        db.execute("UPDATE v_diskann_graph_nodes SET vector=? WHERE state=2", (blob((float("nan"), 0)),))
    elif damage == "bad_adjacency":
        db.execute("UPDATE v_diskann_graph_nodes SET neighbors=? WHERE state=2", (struct.pack("<Q", 2),))
    else:
        db.execute("UPDATE v_diskann_jobs SET captured_count=2")
    db.close()
    db = open_db(path, readonly=True)
    with pytest.raises(sqlite3.DatabaseError) as error:
        get(db, 7)
    assert error.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
    assert db.total_changes == 0
    db.close()


def test_empty_published_generation_allows_pending_without_root(db):
    insert(db, 7)
    db.execute("INSERT INTO v_diskann_generations VALUES(1,1,1,0,zeroblob(32),zeroblob(32),0,0,NULL)")
    db.execute("UPDATE v_diskann_meta SET published_generation=1,next_generation_id=2,next_job_id=2")
    assert knn(db) == []
    assert get(db, 7) == blob((1, 2))


def test_staged_generation_source_digest_is_unknown_until_sealed(db):
    insert(db, 7)
    import_fixture(db)
    db.execute("UPDATE v_diskann_generations SET source_hash=NULL")
    db.execute("UPDATE v_diskann_jobs SET source_hash=NULL,phase=0")
    insert(db, 8)
    assert get(db, 8) == blob((1, 2))
    assert knn(db) == []
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE v_diskann_generations SET state=1")


def test_late_rename_collision_rolls_back_all_shadow_names(db):
    insert(db, 7)
    before = snapshot(db)
    db.execute("CREATE TABLE vv_diskann_jobs(x)")
    with pytest.raises(sqlite3.DatabaseError):
        db.execute("ALTER TABLE v RENAME TO vv")
    assert snapshot(db) == before
    assert get(db, 7) == blob((1, 2))
    assert not db.execute("SELECT name FROM sqlite_schema WHERE name LIKE 'vv%' AND name!='vv_diskann_jobs'").fetchall()


@pytest.mark.parametrize("field", ["revision", "next_entry_id"])
def test_counter_exhaustion_rejects_before_canonical_mutations(db, field):
    insert(db, 7)
    db.execute(f"UPDATE v_diskann_meta SET {field}=9223372036854775807")
    before = snapshot(db)
    with pytest.raises(sqlite3.DatabaseError, match="exhausted"):
        insert(db, 8)
    assert snapshot(db) == before


# --- task-27 persistent-fence/highwater/chain refinements ------------------
# The refined meta tail is next_fence (positive allocator, initial 1) and
# cleanup_entry_id (bounded GC cursor, initial 0); the refined job tail is the
# fixed cutoff_entry_id highwater and the opaque 32-byte snapshot_chain. These
# readers fail closed even when a CHECK constraint was bypassed.

def job_fixture(db, *, fence=1, cutoff=1, expected=1, scan=0, chain=b"\x00" * 32):
    db.execute("INSERT INTO v_diskann_generations VALUES(1,1,0,0,NULL,zeroblob(32),1,1,1)")
    db.execute(
        "INSERT INTO v_diskann_jobs(job_id,generation_id,parent_generation,cutoff_revision,"
        "fence,owner_nonce,phase,expected_count,scan_entry_id,captured_count,import_cursor,"
        "validation_cursor,cleanup_cursor,config_hash,cutoff_entry_id,snapshot_chain) "
        "VALUES(1,1,NULL,1,?,NULL,0,?,?,0,0,0,0,zeroblob(32),?,?)",
        (fence, expected, scan, cutoff, chain),
    )
    db.execute(
        "UPDATE v_diskann_meta SET next_job_id=2,next_generation_id=2,active_job=1,"
        "candidate_generation=1,next_fence=?",
        (fence + 1,),
    )


def assert_corrupt(fn):
    with pytest.raises(sqlite3.DatabaseError) as error:
        fn()
    assert error.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT


def status_row(db):
    return db.execute("SELECT vectorlite_index_status('main','v')").fetchone()


def test_refined_meta_defaults_are_next_fence_one_and_cleanup_zero(db):
    assert db.execute("SELECT next_fence,cleanup_entry_id FROM v_diskann_meta").fetchone() == (1, 0)


@pytest.mark.parametrize("value", [0, -1])
def test_refined_meta_next_fence_must_be_positive(db, value):
    insert(db, 7)
    db.execute("PRAGMA ignore_check_constraints=ON")
    db.execute("UPDATE v_diskann_meta SET next_fence=?", (value,))
    assert_corrupt(lambda: get(db, 7))


@pytest.mark.parametrize("where", ["at_allocator", "negative"])
def test_refined_meta_cleanup_cursor_bounded_by_allocator(db, where):
    insert(db, 7)
    insert(db, 8)
    next_entry = db.execute("SELECT next_entry_id FROM v_diskann_meta").fetchone()[0]
    value = next_entry if where == "at_allocator" else -1
    db.execute("PRAGMA ignore_check_constraints=ON")
    db.execute("UPDATE v_diskann_meta SET cleanup_entry_id=?", (value,))
    assert_corrupt(lambda: get(db, 7))


def test_refined_active_job_fence_must_precede_allocator(db):
    insert(db, 7)
    job_fixture(db, fence=1)
    db.execute("UPDATE v_diskann_meta SET next_fence=1")
    assert_corrupt(lambda: status_row(db))


@pytest.mark.parametrize("cutoff", [2, 9223372036854775807])
def test_refined_job_cutoff_below_allocator(db, cutoff):
    insert(db, 7)
    job_fixture(db, cutoff=cutoff)
    assert_corrupt(lambda: status_row(db))


@pytest.mark.parametrize("field", ["scan_entry_id", "expected_count"])
def test_refined_job_scan_and_expected_bounded_by_cutoff(db, field):
    insert(db, 7)
    job_fixture(db, cutoff=1, expected=1)
    db.execute("PRAGMA ignore_check_constraints=ON")
    db.execute(f"UPDATE v_diskann_jobs SET {field}=2")
    assert_corrupt(lambda: status_row(db))


@pytest.mark.parametrize("length", [31, 33])
def test_refined_job_snapshot_chain_is_exactly_32_bytes(db, length):
    insert(db, 7)
    job_fixture(db)
    db.execute("PRAGMA ignore_check_constraints=ON")
    db.execute("UPDATE v_diskann_jobs SET snapshot_chain=zeroblob(?)", (length,))
    assert_corrupt(lambda: status_row(db))


def test_refined_job_snapshot_chain_cannot_be_null(db):
    insert(db, 7)
    job_fixture(db)
    db.execute("PRAGMA ignore_check_constraints=ON")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE v_diskann_jobs SET snapshot_chain=NULL")


def test_refined_canonical_dml_leaves_fence_and_cleanup_cursor_unchanged(db):
    insert(db, 7)
    job_fixture(db, cutoff=1)
    db.execute("UPDATE v_diskann_meta SET next_fence=4,cleanup_entry_id=1")
    before = db.execute("SELECT next_fence,cleanup_entry_id,next_job_id FROM v_diskann_meta").fetchone()
    db.execute("UPDATE v SET embedding=? WHERE rowid=7", (blob((3, 4)),))
    db.execute("DELETE FROM v WHERE rowid=7")
    insert(db, 9)
    assert db.execute("SELECT next_fence,cleanup_entry_id,next_job_id FROM v_diskann_meta").fetchone() == before
    assert db.execute("SELECT cutoff_entry_id FROM v_diskann_jobs").fetchone() == (1,)


def test_refined_job_highwater_is_frozen_across_later_inserts(db):
    insert(db, 7)
    job_fixture(db, cutoff=1, expected=1)
    next_before = db.execute("SELECT next_entry_id FROM v_diskann_meta").fetchone()[0]
    for rowid in range(20, 25):
        insert(db, rowid)
    assert db.execute("SELECT next_entry_id FROM v_diskann_meta").fetchone()[0] > next_before
    assert db.execute("SELECT cutoff_entry_id,expected_count FROM v_diskann_jobs").fetchone() == (1, 1)
    assert db.execute("SELECT count(*) FROM v_diskann_snapshot_map").fetchone() == (0,)


def test_refined_corrupt_fence_fields_fail_closed_on_readonly_reopen(tmp_path):
    path = tmp_path / "refined.db"
    db = open_db(path)
    create(db)
    insert(db, 7)
    job_fixture(db, cutoff=1)
    db.execute("PRAGMA ignore_check_constraints=ON")
    db.execute("UPDATE v_diskann_meta SET next_fence=0")
    db.close()
    db = open_db(path, readonly=True)
    try:
        assert_corrupt(lambda: get(db, 7))
        assert db.total_changes == 0
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 4 and sys.argv[1] == "--native"
    native_batch_qualification(int(sys.argv[2]), None if sys.argv[3] == "none" else sys.argv[3])