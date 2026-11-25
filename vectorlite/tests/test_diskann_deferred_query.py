"""Published Format5 query qualification against an explicit or deployed artifact.

The CSR injector is a trusted TEST fixture, not a public maintenance API. It
writes staged/published shadow rows solely to isolate T6 from T3/T5 integration.
Production callers must use the fenced maintenance operations. Native reference
cases additionally require VECTORLITE_T6_NATIVE_REFERENCE; absent reference
execution is explicitly skipped and must not be counted as native parity proof.
"""
import hashlib
import json
import os
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
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
    return np.asarray(values, dtype="<f4").tobytes()


def open_db(path=":memory:", readonly=False):
    library = artifact()
    assert library.is_file(), f"build or install the extension first: {library}"
    db = sqlite3.connect(f"file:{path}?mode=ro" if readonly else str(path), isolation_level=None, uri=readonly)
    db.enable_load_extension(True)
    db.load_extension(str(library))
    return db


def create(db, dim=2, metric="l2", options="degree=4,build_list_size=16,search_list_size=4,max_visits=4096", schema="main"):
    db.execute(f'CREATE VIRTUAL TABLE "{schema}".v USING vectorlite(embedding float32[{dim}] {metric},diskann({options}))')


def insert(db, rowid, vector, schema="main"):
    db.execute(f'INSERT INTO "{schema}".v(rowid,embedding) VALUES(?,?)', (rowid, blob(vector)))


def query(db, vector=(0, 0), k=10, filter_sql="", project=True, l=None, schema="main"):
    parameter = "knn_param(?,?)" if l is None else "knn_param(?,?,?)"
    parameters = (blob(vector), k) if l is None else (blob(vector), k, json.dumps({"search_list_size": l}))
    columns = "rowid,embedding,distance" if project else "rowid,distance"
    return db.execute(f'SELECT {columns} FROM "{schema}".v WHERE knn_search(embedding,{parameter}) {filter_sql}', parameters).fetchall()


def slot(native_local_id, entry_id):
    """One frozen Format-5 neighbour slot: two little-endian u64 values."""
    return struct.pack("<QQ", native_local_id, entry_id)


def slots(entries, ids):
    """Encodes one adjacency list as 16-byte (native_local_id, entry_id) slots.

    entry_id is zero only for the frozen root (native id len(entries), no
    snapshot-map row); every other slot carries that node's own entry id.
    """
    entry_ids = [entry for entry, _, _ in entries]
    root = len(entries)
    return b"".join(slot(target, 0 if target == root else entry_ids[target]) for target in ids)


def retired_entries(db, entries, schema="main"):
    """Snapshot entries whose entry id is not the live row of its public rowid."""
    prefix = f'"{schema}".v_diskann_'
    return [
        entry
        for entry, rowid, _ in entries
        if db.execute(f"SELECT entry_id FROM {prefix}current WHERE public_rowid=?", (rowid,)).fetchone() != (entry,)
    ]


def inject_csr(db, entries=None, adjacency=None, root=None, generation=1, published=True, schema="main", source_revision=None):
    """Trusted Format-5 SQL fixture retaining native dense IDs and exact adjacency order.

    Every synthetic generation/root builder in this file emits the seven-column
    node row (generation_id, native_local_id, entry_id, public_rowid, state,
    neighbors, vector) in the lean v11 shape: live rows (state=0) store SQL NULL
    for the vector, whose canonical payload stays in diskann_entries, and only the
    frozen root (state=2) carries its vector blob. Neighbour slots are 16-byte
    (native_local_id, entry_id) pairs whose entry id is zero only for the frozen
    root. generation_retired rows are inserted for every snapshot entry that is
    not live in current, so the D3 invariant
    COUNT == total_count - eligible_count holds immediately.
    """
    prefix = f'"{schema}".v_diskann_'
    if entries is None:
        entries = db.execute(f'SELECT c.entry_id,e.public_rowid,e.vector FROM {prefix}current c JOIN {prefix}entries e ON e.entry_id=c.entry_id ORDER BY c.entry_id').fetchall()
    n = len(entries)
    if adjacency is None:
        adjacency = [[j for j in range(n + 1) if j != i] for i in range(n + 1)] if n <= 4 else [[i + 1] if i < n else [0] for i in range(n + 1)]
    if root is None and n:
        root = entries[0][2]
    revision = db.execute(f'SELECT revision FROM {prefix}meta').fetchone()[0] if source_revision is None else source_revision
    retired = retired_entries(db, entries, schema)
    eligible = n - len(retired)
    payloads = [slots(entries, ids) for ids in adjacency]
    # Opaque nonzero fixture hashes, not production digest validation evidence.
    digest = bytes(range(32))
    db.execute(f'INSERT INTO {prefix}generations(generation_id,job_id,state,source_revision,source_hash,config_hash,total_count,eligible_count,root_local_id) VALUES(?,?,?,?,?,?,?,?,?)', (generation, generation, 1 if published else 0, revision, digest, digest[::-1], n, eligible, n if n else None))
    for i, (entry, rowid, _vector) in enumerate(entries):
        # Lean v11 row shape: the live node stores SQL NULL and keeps its
        # canonical payload in diskann_entries (D1/D10 payload authority).
        db.execute(f'INSERT INTO {prefix}graph_nodes VALUES(?,?,?,?,0,?,NULL)', (generation, i, entry, rowid, payloads[i]))
    if n:
        db.execute(f'INSERT INTO {prefix}graph_nodes VALUES(?,?,NULL,NULL,2,?,?)', (generation, n, payloads[n], root))
    for entry in retired:
        db.execute(f'INSERT INTO {prefix}generation_retired VALUES(?,?)', (generation, entry))
    db.execute(f'UPDATE {prefix}meta SET next_generation_id=max(next_generation_id,?),next_job_id=max(next_job_id,?)', (generation + 1, generation + 1))
    if published:
        db.execute(f'UPDATE {prefix}generations SET state=2 WHERE state=1 AND generation_id!=?', (generation,))
        db.execute(f'UPDATE {prefix}meta SET published_generation=?', (generation,))
    else:
        cutoff_entry = db.execute(f'SELECT next_entry_id-1 FROM {prefix}meta').fetchone()[0]
        parent = db.execute(f'SELECT published_generation FROM {prefix}meta').fetchone()[0]
        db.execute(f'INSERT INTO {prefix}jobs(job_id,generation_id,parent_generation,cutoff_revision,fence,owner_nonce,phase,expected_count,scan_entry_id,captured_count,import_cursor,validation_cursor,cleanup_cursor,source_hash,config_hash,candidate_hash,failure_code,cutoff_entry_id,snapshot_chain) VALUES(?,?,?,?,?,NULL,3,?,?,?,?,0,0,?,?,NULL,NULL,?,?)', (generation, generation, parent, revision, generation, n, cutoff_entry, n, n + (1 if n else 0), digest, digest[::-1], cutoff_entry, digest))
        db.execute(f'UPDATE {prefix}meta SET candidate_generation=?,active_job=?,next_fence=max(next_fence,?)', (generation, generation, generation + 1))
    return entries


def assert_retired_invariant(db, generation=None):
    """Asserts COUNT(generation_retired) == total_count - eligible_count.

    The D3 invariant holds for every staged/published generation that has not
    begun reclamation; it is asserted by the import validation pass and the
    publication transition. This helper pins it independently of those paths.
    """
    query_sql = "SELECT generation_id,total_count,eligible_count FROM v_diskann_generations"
    if generation is not None:
        query_sql += " WHERE generation_id=?"
    rows = db.execute(query_sql, () if generation is None else (generation,)).fetchall()
    assert rows, "expected at least one generation"
    for generation_id, total, eligible in rows:
        count = db.execute(
            "SELECT count(*) FROM v_diskann_generation_retired WHERE generation_id=?",
            (generation_id,),
        ).fetchone()[0]
        assert count == total - eligible, (generation_id, count, total, eligible)


def canonical_queries(db, queries, metric):
    if metric == "l2":
        return queries.copy()
    dimension = queries.shape[1]
    db.execute(f'CREATE VIRTUAL TABLE temp.t6_query_input USING vectorlite(embedding float32[{dimension}] cosine,diskann())')
    normalized = []
    for i, vector in enumerate(queries):
        db.execute('INSERT INTO temp.t6_query_input(rowid,embedding) VALUES(?,?)', (i, blob(vector)))
        normalized.append(np.frombuffer(db.execute('SELECT embedding FROM temp.t6_query_input WHERE rowid=?', (i,)).fetchone()[0], dtype="<f4"))
    return np.vstack(normalized)


@pytest.fixture
def db():
    db = open_db()
    create(db)
    yield db
    db.close()


@pytest.mark.parametrize("filter_sql", ["", "AND rowid=0", "AND rowid IN (0,7)"])
def test_pending_and_empty_generation_remain_empty(db, filter_sql):
    insert(db, 0, [0, 0])
    insert(db, 7, [1, 1])
    assert query(db, filter_sql=filter_sql) == []
    inject_csr(db, entries=[], adjacency=[])
    assert query(db, filter_sql=filter_sql) == []
    assert db.execute("SELECT count(*) FROM v_diskann_graph_nodes").fetchone() == (0,)
    with pytest.raises(sqlite3.DatabaseError, match="k should be greater than 0"):
        query(db, k=0)


@pytest.mark.parametrize("metric", ["l2", "cosine"])
@pytest.mark.parametrize("filter_sql,expected", [("", [0, 7, 9223372036854775807]), ("AND rowid=0", [0]), ("AND rowid IN (0,7,99,9223372036854775807)", [0, 7, 9223372036854775807])])
def test_exact_small_largek_native0_rootN_ties_zero(metric, filter_sql, expected):
    db = open_db()
    create(db, metric=metric)
    for rowid, vector in [(0, [0, 0]), (7, [0, 0]), (9223372036854775807, [1, 0])]:
        insert(db, rowid, vector)
    inject_csr(db)
    rows = query(db, k=1_000_000, filter_sql=filter_sql)
    assert [r[0] for r in rows] == expected
    assert all(r[2] == (sum(np.frombuffer(r[1], dtype="<f4") ** 2) if metric == "l2" else 1.0) for r in rows)
    assert db.execute("SELECT native_local_id,state FROM v_diskann_graph_nodes ORDER BY native_local_id").fetchall() == [(0, 0), (1, 0), (2, 0), (3, 2)]
    db.close()


@pytest.mark.parametrize("during_import", [False, True])
@pytest.mark.parametrize("action", ["update", "delete", "reinsert"])
@pytest.mark.parametrize("filter_sql", ["", "AND rowid=0", "AND rowid IN (0,1,2,100)"])
def test_version_retirement_import_and_after_publication(db, during_import, action, filter_sql):
    for i in range(3):
        insert(db, i, [i, 0])
    entries = inject_csr(db, published=not during_import)
    if during_import:
        assert query(db) == []
    if action == "update":
        db.execute("UPDATE v SET embedding=? WHERE rowid=0", (blob([30, 0]),))
    else:
        db.execute("DELETE FROM v WHERE rowid=0")
        if action == "reinsert":
            insert(db, 0, [30, 0])
    insert(db, 100, [0, 0])
    if during_import:
        assert db.execute("SELECT eligible_count FROM v_diskann_generations WHERE generation_id=1").fetchone() == (2,)
        db.execute("UPDATE v_diskann_generations SET state=1 WHERE generation_id=1")
        db.execute("UPDATE v_diskann_meta SET published_generation=1,candidate_generation=NULL,active_job=NULL")
    rows = query(db, k=1000, filter_sql=filter_sql)
    assert all(r[0] not in (0, 100) for r in rows)
    assert [r[0] for r in rows] == ([] if "rowid=0" in filter_sql else [1, 2])
    assert db.execute("SELECT vector FROM v_diskann_entries WHERE entry_id=?", (entries[0][0],)).fetchone() == (entries[0][2],)
    assert db.execute("SELECT embedding FROM v WHERE rowid=100").fetchone() == (blob([0, 0]),)
    if action != "delete":
        assert db.execute("SELECT embedding FROM v WHERE rowid=0").fetchone() == (blob([30, 0]),)


def test_ann_native_zero_is_live_and_query_is_read_only(db):
    for i in range(16):
        insert(db, i, [i, 0])
    inject_csr(db)
    before = db.total_changes
    for _ in range(3):
        rows = query(db, vector=[0, 0], k=2, l=4)
        assert [r[0] for r in rows] == [0, 1]
        assert rows[0] == (0, blob([0, 0]), 0.0)
    assert db.total_changes == before


def test_generation_cannot_declare_live_zero_as_frozen_root(db):
    for i in range(16):
        insert(db, i, [i, 0])
    inject_csr(db)
    assert db.execute("SELECT state FROM v_diskann_graph_nodes WHERE native_local_id=0").fetchone() == (0,)
    db.execute("PRAGMA ignore_check_constraints=ON")
    db.execute("UPDATE v_diskann_generations SET root_local_id=0")
    with pytest.raises(sqlite3.DatabaseError, match="root/counts") as error:
        query(db, k=2, l=4)
    assert error.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT


@pytest.mark.parametrize("schema", ["main", "temp", "aux"])
def test_schema_qualified_query_and_canonical_point_reads(schema):
    db = open_db()
    if schema == "aux":
        db.execute("ATTACH ':memory:' AS aux")
    create(db, schema=schema)
    insert(db, 0, [1, 0], schema=schema)
    inject_csr(db, schema=schema)
    db.execute(f'UPDATE "{schema}".v SET embedding=? WHERE rowid=0', (blob([2, 0]),))
    insert(db, 8, [0, 0], schema=schema)
    assert query(db, schema=schema) == []
    assert db.execute(f'SELECT rowid,embedding FROM "{schema}".v WHERE rowid IN (0,8) ORDER BY rowid').fetchall() == [(0, blob([2, 0])), (8, blob([0, 0]))]
    db.close()


@pytest.mark.parametrize("project", [False, True])
def test_ann_retired_navigation_bridge_and_pending_exclusion(db, project):
    for i in range(128):
        insert(db, i, [i, 0])
    inject_csr(db)
    for i in range(45):
        db.execute("UPDATE v SET embedding=? WHERE rowid=?", (blob([1000 + i, 0]), i))
    insert(db, 500, [50, 0])
    rows = query(db, vector=[50, 0], k=4, project=project, l=4)
    assert len(rows) == 4
    assert {r[0] for r in rows} == {49, 50, 51, 52} or {r[0] for r in rows} == {48, 49, 50, 51}
    assert all(45 <= r[0] < 128 and np.isfinite(r[-1]) for r in rows)
    if project:
        for rowid, vector, distance in rows:
            assert vector == blob([rowid, 0])
            assert distance == (rowid - 50) ** 2


def test_underfilled_filtered_ann_fallback_eligible_only(db):
    for i in range(128):
        insert(db, i, [i, 0])
    # Root reaches first16 only; filtered results are deliberately unreachable.
    adjacency = [[i + 1] if i < 15 else [] for i in range(128)] + [[0]]
    inject_csr(db, adjacency=adjacency)
    insert(db, 1000, [100, 0])
    db.execute("UPDATE v SET embedding=? WHERE rowid=100", (blob([1000, 0]),))
    rows = query(db, vector=[100, 0], k=3, l=4, filter_sql="AND rowid IN (100,101,102,103,104,105,1000)")
    assert [r[0] for r in rows] == [101, 102, 103]
    assert [r[1] for r in rows] == [blob([i, 0]) for i in (101, 102, 103)]


@pytest.mark.parametrize(
    "damage",
    [
        "scored-vector-size",
        "scored-vector-text",
        "scored-vector-nan",
        "data-frozen-payload",
        "root-role",
        "root-vector",
        "root-count",
        "neighbor-dangling",
        "neighbor-self",
        "neighbor-duplicate",
        "neighbor-range",
    ],
)
def test_corruption_rejected_not_empty_or_missing(db, damage):
    """D10 row 2 (lean v11): every query-visible payload class fails closed.

    Payload authority splits the classes: live vectors are canonical in
    diskann_entries, so the vector probes target the scored canonical entries row
    and remain query-visible exactly as in Format 4; the node-row public_rowid/
    state/adjacency probes stay on v_diskann_graph_nodes, which is still
    authoritative for those fields (D1/D10). The frozen root is the one node whose
    vector stays inline, so its payload probes stay on the node row. Every class
    must surface SQLITE_CORRUPT, never an empty answer or a silently missing row.
    The fallback route's live-node role rule is pinned separately in
    test_retired_shape_fallback_keeps_lean_role_rule.
    """
    for i in range(16):
        insert(db, i, [i, 0])
    inject_csr(db)
    db.execute("PRAGMA ignore_check_constraints=ON")
    entries = db.execute("SELECT entry_id,public_rowid FROM v_diskann_entries ORDER BY entry_id").fetchall()
    root = len(entries)
    # The canonical entry the query will score first (rowid 0 is the nearest row).
    scored = db.execute(
        "SELECT e.entry_id FROM v_diskann_current c JOIN v_diskann_entries e "
        "ON e.entry_id=c.entry_id WHERE c.public_rowid=0"
    ).fetchone()[0]
    if damage == "scored-vector-size":
        db.execute("UPDATE v_diskann_entries SET vector=zeroblob(1048576) WHERE entry_id=?", (scored,))
    elif damage == "scored-vector-text":
        db.execute("UPDATE v_diskann_entries SET vector='bad' WHERE entry_id=?", (scored,))
    elif damage == "scored-vector-nan":
        db.execute("UPDATE v_diskann_entries SET vector=? WHERE entry_id=?", (blob([np.nan, 0]), scored))
    elif damage == "data-frozen-payload":
        db.execute("UPDATE v_diskann_graph_nodes SET state=2 WHERE native_local_id=0")
    elif damage == "root-role":
        db.execute("UPDATE v_diskann_graph_nodes SET state=0 WHERE native_local_id=?", (root,))
    elif damage == "root-vector":
        db.execute("UPDATE v_diskann_graph_nodes SET vector=zeroblob(1048576) WHERE native_local_id=?", (root,))
    elif damage == "root-count":
        db.execute("UPDATE v_diskann_generations SET root_local_id=0")
    elif damage == "neighbor-dangling":
        db.execute("DELETE FROM v_diskann_graph_nodes WHERE native_local_id=0")
    else:
        if damage == "neighbor-self":
            payload = slot(root, 0)
        elif damage == "neighbor-duplicate":
            payload = slot(0, entries[0][0]) * 2
        else:
            payload = slot(1000, entries[0][0])
        db.execute("UPDATE v_diskann_graph_nodes SET neighbors=? WHERE native_local_id=?", (payload, root))
    with pytest.raises(sqlite3.DatabaseError) as error:
        query(db, k=2, l=4)
    assert error.value.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT
    # Error tears down the query; ordinary SQL still works.
    assert db.execute("SELECT 1").fetchone() == (1,)


def test_entries_side_corruption_is_query_visible_and_publish_time_caught():
    """D10 row 2 (lean v11): canonical entries corruption is query-visible again.

    Lean Format 5 no longer inlines live vectors, so the graph route scores every
    candidate from its canonical diskann_entries row: entries.vector damage on a
    scored row is query-visible SQLITE_CORRUPT, exactly like Format 4 and the
    baseline's per-read join. The next import/build validation still catches every
    damaged version it materializes before any graph work is committed, and the
    snapshot capture fails closed before any snapshot row is written.
    """
    db = open_db()
    try:
        create(db, options="degree=8,build_list_size=16,search_list_size=4,max_visits=4096")
        for i in range(16):
            insert(db, i, [i, 0])
        inject_csr(db)
        baseline = query(db, vector=[0, 0], k=4, l=4, project=True)
        assert len(baseline) == 4
        target = db.execute("SELECT entry_id FROM v_diskann_current WHERE public_rowid=0").fetchone()[0]
        original = db.execute(
            "SELECT vector FROM v_diskann_entries WHERE entry_id=?", (target,)
        ).fetchone()[0]
        db.execute("PRAGMA ignore_check_constraints=ON")
        db.execute("UPDATE v_diskann_entries SET vector=? WHERE entry_id=?", (blob([np.nan, 0]), target))
        # Query-visible scored canonical read (D10 row 2): the damaged entry is
        # scored by the graph route for both the projected and unprojected shapes.
        for project in (False, True):
            outcome, value = _capture(
                lambda: query(db, vector=[0, 0], k=4, l=4, project=project)
            )
            assert outcome == "err", (project, value)
            assert value.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, (project, value)
        # The next build materializes the damaged canonical version and fails
        # closed before any node row or candidate is committed.
        db.execute(
            "UPDATE v_diskann_entries SET vector=? WHERE entry_id=?", (original, target)
        )
        node_rows = db.execute("SELECT count(*) FROM v_diskann_graph_nodes").fetchone()[0]
        control_keys = _status(db)
        job = _prepare_and_capture(db)
        db.execute(
            "UPDATE v_diskann_entries SET vector=? WHERE entry_id=?",
            (blob([np.nan, 0]), target),
        )
        _control(db, "index_claim_build", {"job_id": job["job_id"], "fence": job["fence"]})
        options = {"job_id": job["job_id"], "fence": job["fence"], "memory_budget_bytes": 128 * 1024 * 1024, "workers": 1}
        with pytest.raises(sqlite3.DatabaseError) as error:
            db.execute("SELECT vectorlite_index_build('main','v',?)", (json.dumps(options),)).fetchall()
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
        current = _status(db)["current_job"]
        assert current["phase"] == 2 and current["import_cursor"] == 0
        assert db.execute("SELECT count(*) FROM v_diskann_graph_nodes").fetchone()[0] == node_rows
        assert _status(db)["canonical_count"] == control_keys["canonical_count"]
        # The same canonical damage is also detected by the snapshot capture that
        # precedes every build: a fresh restarted job fails before any snapshot
        # row or candidate is committed.
        _control(
            db,
            "index_restart",
            {"job_id": current["job_id"], "fence": current["fence"], "force": True},
        )
        restarted = _status(db)["current_job"]
        assert restarted["phase"] == 0
        with pytest.raises(sqlite3.DatabaseError) as capture_error:
            _control(
                db,
                "index_snapshot_step",
                {
                    "job_id": restarted["job_id"],
                    "fence": restarted["fence"],
                    "max_records": 1,
                    "max_bytes": 65536,
                },
            )
        assert capture_error.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
        after = _status(db)["current_job"]
        assert after["captured_count"] == 0
        assert db.execute(
            "SELECT count(*) FROM v_diskann_snapshot_map WHERE job_id=?",
            (restarted["job_id"],),
        ).fetchone()[0] == 0
    finally:
        db.close()


@pytest.mark.parametrize("project", [False, True])
def test_exact_and_ann_share_owner_visit_limit(project):
    db = open_db()
    create(db, options="degree=2,build_list_size=4,search_list_size=4,max_visits=8")
    for i in range(16):
        insert(db, i, [i, 0])
    inject_csr(db)
    with pytest.raises(sqlite3.DatabaseError, match="visit|max_visits"):
        query(db, k=16, project=project)
    with pytest.raises(sqlite3.DatabaseError, match="visit|max_visits"):
        query(db, vector=[10, 0], k=2, project=project, l=4)
    db.close()


def test_largek_budget_clamp_uses_eligible_not_pending_canonical():
    db = open_db()
    create(db, options="degree=2,build_list_size=4,search_list_size=4,max_visits=8,cache_bytes=131072")
    insert(db, 0, [2, 0])
    inject_csr(db)
    for i in range(1, 1001):
        insert(db, i, [0, 0])
    assert query(db, k=1_000_000) == [(0, blob([2, 0]), 4.0)]
    assert query(db, k=1_000_000, filter_sql="AND rowid IN (0,1,2)") == [(0, blob([2, 0]), 4.0)]
    db.close()


def test_query_cache_budget_rejects_exact_results():
    db = open_db()
    create(db, options="degree=2,build_list_size=4,search_list_size=4,max_visits=4096,cache_bytes=131072")
    for i in range(2000):
        insert(db, i, [i, 0])
    inject_csr(db)
    with pytest.raises(sqlite3.DatabaseError, match="workspace|cache_bytes"):
        query(db, k=2000, project=True)
    assert len(query(db, k=1, filter_sql="AND rowid=0")) == 1
    db.close()


def test_old_cursor_new_generation_and_readonly_snapshot(tmp_path):
    path = tmp_path / "query.db"
    db = open_db(path)
    db.execute("PRAGMA journal_mode=WAL")
    create(db)
    for i in range(3):
        insert(db, i, [i, 0])
    inject_csr(db)
    reader = open_db(path, readonly=True)
    old = reader.execute("SELECT rowid,embedding,distance FROM v WHERE knn_search(embedding,knn_param(?,3))", (blob([0, 0]),))
    assert old.fetchone() == (0, blob([0, 0]), 0.0)
    db.execute("UPDATE v SET embedding=? WHERE rowid=1", (blob([10, 0]),))
    insert(db, 8, [1, 0])
    inject_csr(db, generation=2)
    assert old.fetchall() == [(1, blob([1, 0]), 1.0), (2, blob([2, 0]), 4.0)]
    assert [r[0] for r in query(reader, k=10)] == [0, 8, 2, 1]
    assert reader.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    reader.close()
    db.close()


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_same_native_csr_provider_recall_and_scores(metric):
    executable = os.environ.get("VECTORLITE_T6_NATIVE_REFERENCE")
    if not executable:
        pytest.skip("requires separately compiled official provider reference; no native parity claim")
    db = open_db()
    create(db, dim=128, metric=metric, options="degree=32,build_list_size=100,search_list_size=10,max_visits=65536")
    rng = np.random.default_rng(42)
    data = rng.normal(size=(256, 128)).astype(np.float32)
    for i, vector in enumerate(data):
        insert(db, i, vector)
    entries = db.execute("SELECT entry_id,public_rowid,vector FROM v_diskann_entries ORDER BY entry_id").fetchall()
    canonical = np.vstack([np.frombuffer(e[2], dtype="<f4") for e in entries])
    queries = canonical[[0, 13, 127, 200]].copy()
    normalized_queries = canonical_queries(db, queries, metric)
    native = subprocess.run([executable], input=json.dumps({"data": canonical.tolist(), "queries": normalized_queries.tolist(), "metric": metric, "degree": 32, "build_l": 100, "alpha": 1.2, "k": 10, "l": 50}), text=True, capture_output=True, check=True)
    reference = json.loads(native.stdout)
    assert reference["root"] == len(entries)
    inject_csr(db, entries=entries, adjacency=reference["adjacency"], root=blob(reference["root_vector"]))
    audit = []
    for vector, normalized_query, expected in zip(queries, normalized_queries, reference["results"]):
        # Canonical cosine query is normalized once by the extension. Native uses
        # that exact shared-codec output from the separate canonical input table.
        rows = query(db, vector=vector, k=10, l=50)
        labels = {r[0] for r in rows}
        native_labels = {r[0] for r in expected}
        if metric == "l2":
            truth = set(np.argsort(np.sum((canonical.astype(np.float64) - vector.astype(np.float64)) ** 2, axis=1))[:10])
        else:
            truth = set(np.argsort(1.0 - canonical.astype(np.float64) @ normalized_query.astype(np.float64))[:10])
        assert len(labels & truth) >= len(native_labels & truth), (labels, native_labels, truth)
        expected_scores = dict(expected)
        differences = [(distance, expected_scores[rowid]) for rowid, _, distance in rows if rowid in expected_scores]
        audit.append({"sql_labels": sorted(labels), "native_labels": sorted(native_labels), "sql_recall": len(labels & truth) / 10, "native_recall": len(native_labels & truth) / 10, "same_label_score_bit_differences": sum(struct.pack("<f", a) != struct.pack("<f", b) for a, b in differences), "maximum_score_absolute_difference": max((abs(a - b) for a, b in differences), default=0)})
        for rowid, projected, distance in rows:
            assert projected == entries[rowid][2]
            assert np.isfinite(distance)
            if rowid in expected_scores:
                assert distance == pytest.approx(expected_scores[rowid], abs=3e-5, rel=3e-5)
    print("T6_PARITY " + json.dumps({"metric": metric, "population": len(entries), "dimension": 128, "degree": 32, "build_l": 100, "query_l": 50, "seed": 42, "canonical_sha256": hashlib.sha256(canonical.tobytes()).hexdigest(), "csr_sha256": hashlib.sha256(json.dumps(reference["adjacency"]).encode()).hexdigest(), "queries": audit}, sort_keys=True))
    db.close()



# ---------------------------------------------------------------------------
# T4: C1 contract cases. Success-path differential, fail-closed single-fault
# classes, coincident-fault recording, same-connection build guard, and
# reentrancy/staleness. These reuse the qualification helpers above.
# ---------------------------------------------------------------------------


def test_repeated_queries_same_and_second_connection_identical(tmp_path):
    """C1 success path: repeats and a second connection are bit-identical.

    Same artifact, same committed published generation: every repeat must
    return the same ordered rowids and the same distance bits, on the original
    connection and on a second connection. Projected bytes must equal the
    committed vector for that rowid.
    """
    path = tmp_path / "equivalence.db"
    db = open_db(path)
    db.execute("PRAGMA journal_mode=WAL")
    create(db, dim=8, options="degree=128,build_list_size=130,search_list_size=4,max_visits=4096")
    data = np.random.default_rng(42).normal(size=(128, 8)).astype(np.float32)
    for rowid, vector in enumerate(data):
        insert(db, rowid, vector)
    adjacency = [[j for j in range(128) if j != i] for i in range(128)] + [list(range(128))]
    inject_csr(db, adjacency=adjacency)
    second = open_db(path)

    def signature(rows, project):
        if project:
            return [(int(row[0]), struct.pack("<f", row[2]), bytes(row[1])) for row in rows]
        return [(int(row[0]), struct.pack("<f", row[1]), b"") for row in rows]

    queries = np.random.default_rng(43).normal(size=(4, 8)).astype(np.float32)
    cases = [
        (queries[0], 10, 10, "", True),
        (queries[1], 10, 100, "", False),
        (queries[2], 33, 100, "", True),
        (queries[0], 10, 50, "AND rowid IN (0,7,9999)", True),
    ]
    try:
        for vector, k, l, filter_sql, project in cases:
            first = query(db, vector=vector, k=k, l=l, filter_sql=filter_sql, project=project)
            assert first, (k, l, filter_sql, project)
            expected = signature(first, project)
            for _ in range(2):
                repeat = query(db, vector=vector, k=k, l=l, filter_sql=filter_sql, project=project)
                assert signature(repeat, project) == expected
            mirrored = query(second, vector=vector, k=k, l=l, filter_sql=filter_sql, project=project)
            assert signature(mirrored, project) == expected
            if project:
                for row in first:
                    assert row[1] == blob(data[int(row[0])])
                    assert np.isfinite(row[2])
    finally:
        second.close()
        db.close()


def test_filter_projection_k_and_list_size_internal_consistency():
    """C1: filters, projection, k=33/k=100 and L=10/L=100 stay deterministic.

    Every returned projection is the committed vector for its rowid, result
    distances are nondecreasing, repeated runs are bit-identical, and filters
    (rowid= and IN with hits and misses) return exactly their hit set.
    """
    db = open_db()
    create(db, dim=8, options="degree=128,build_list_size=130,search_list_size=4,max_visits=4096")
    data = np.random.default_rng(42).normal(size=(128, 8)).astype(np.float32)
    for rowid, vector in enumerate(data):
        insert(db, rowid, vector)
    adjacency = [[j for j in range(128) if j != i] for i in range(128)] + [list(range(128))]
    inject_csr(db, adjacency=adjacency)
    vector = np.random.default_rng(43).normal(size=8).astype(np.float32)

    def signature(rows, project):
        if project:
            return [(int(row[0]), struct.pack("<f", row[2]), bytes(row[1])) for row in rows]
        return [(int(row[0]), struct.pack("<f", row[1]), b"") for row in rows]

    def assert_projected(rows):
        for row in rows:
            assert row[1] == blob(data[int(row[0])])
            assert np.isfinite(row[2])
        distances = [row[2] for row in rows]
        assert distances == sorted(distances)

    try:
        for k in (33, 100):
            projected = query(db, vector=vector, k=k, l=100, project=True)
            assert len(projected) == k
            assert len({row[0] for row in projected}) == k
            assert_projected(projected)
            expected = signature(projected, True)
            for _ in range(2):
                assert signature(query(db, vector=vector, k=k, l=100, project=True), True) == expected
            plain = query(db, vector=vector, k=k, l=100, project=False)
            assert [(int(row[0]), struct.pack("<f", row[1])) for row in plain] == [
                (row[0], row[1]) for row in expected
            ]
        for l in (10, 100):
            rows = query(db, vector=vector, k=10, l=l, project=True)
            assert len(rows) == 10
            assert len({row[0] for row in rows}) == 10
            assert_projected(rows)
            expected = signature(rows, True)
            for _ in range(2):
                assert signature(query(db, vector=vector, k=10, l=l, project=True), True) == expected
        assert [row[0] for row in query(db, vector=vector, filter_sql="AND rowid=0", project=False)] == [0]
        hits = query(db, vector=vector, filter_sql="AND rowid IN (0,3,9999)", project=False)
        assert {row[0] for row in hits} == {0, 3}
        assert [row[1] for row in hits] == sorted(row[1] for row in hits)
        assert query(db, vector=vector, filter_sql="AND rowid IN (555,999)", project=False) == []
        projected_hits = query(db, vector=vector, filter_sql="AND rowid IN (0,3,7,9999)", project=True)
        assert {row[0] for row in projected_hits} == {0, 3, 7}
        assert_projected(projected_hits)
        filtered_expected = signature(projected_hits, True)
        for _ in range(2):
            repeat = query(db, vector=vector, filter_sql="AND rowid IN (0,3,7,9999)", project=True)
            assert signature(repeat, True) == filtered_expected
    finally:
        db.close()


@pytest.mark.parametrize(
    "damage",
    ["node-missing", "neighbor-dangling", "entry-vector-nan", "entry-vector-size"],
)
def test_single_fault_corruption_is_corrupt_without_rows(damage):
    """C1 single-fault classes: injected corruption fails SQLITE_CORRUPT, no rows.

    Lean Format 5 keeps the node row authoritative for public_rowid/state/
    adjacency (missing identity row, dangling adjacency) while live vectors are
    read from canonical diskann_entries, so the scored-payload single faults
    target the canonical entries row by its current entry id. Every class stays a
    corruption-class error and never serves a row first (D10 row 2).
    """
    db = open_db()
    create(db, options="degree=16,build_list_size=20,search_list_size=4,max_visits=4096")
    for i in range(16):
        insert(db, i, [i, 0])
    inject_csr(db)
    db.execute("PRAGMA ignore_check_constraints=ON")
    scored = db.execute(
        "SELECT e.entry_id FROM v_diskann_current c JOIN v_diskann_entries e "
        "ON e.entry_id=c.entry_id WHERE c.public_rowid=0"
    ).fetchone()[0]
    if damage == "node-missing":
        db.execute("DELETE FROM v_diskann_graph_nodes WHERE state=2")
    elif damage == "neighbor-dangling":
        db.execute("DELETE FROM v_diskann_graph_nodes WHERE native_local_id=0")
    elif damage == "entry-vector-nan":
        db.execute("UPDATE v_diskann_entries SET vector=? WHERE entry_id=?", (blob([np.nan, 0]), scored))
    else:
        db.execute("UPDATE v_diskann_entries SET vector=zeroblob(1048576) WHERE entry_id=?", (scored,))

    def drain():
        served = []
        error = None
        try:
            cursor = db.execute(
                "SELECT rowid,embedding,distance FROM v WHERE knn_search(embedding,knn_param(?,?))",
                (blob([0, 0]), 2),
            )
            try:
                while True:
                    row = cursor.fetchone()
                    if row is None:
                        break
                    served.append(row)
            finally:
                cursor.close()
        except sqlite3.DatabaseError as caught:
            error = caught
        return served, error

    served, error = drain()
    assert error is not None, damage
    assert error.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, (damage, error)
    assert served == []
    assert db.execute("SELECT 1").fetchone() == (1,)
    db.close()


@pytest.mark.parametrize("project", [False, True])
def test_single_fault_budget_exhaustion_is_toobig_without_rows(project):
    """C1 single-fault classes: budget exhaustion fails SQLITE_TOOBIG, no rows."""
    db = open_db()
    create(db, options="degree=2,build_list_size=4,search_list_size=4,max_visits=8")
    for i in range(16):
        insert(db, i, [i, 0])
    inject_csr(db)
    served = []
    error = None
    try:
        cursor = db.execute(
            "SELECT rowid,embedding,distance FROM v WHERE knn_search(embedding,knn_param(?,?))",
            (blob([0, 0]), 16),
        ) if project else db.execute(
            "SELECT rowid,distance FROM v WHERE knn_search(embedding,knn_param(?,?))",
            (blob([0, 0]), 16),
        )
        try:
            while True:
                row = cursor.fetchone()
                if row is None:
                    break
                served.append(row)
        finally:
            cursor.close()
    except sqlite3.DatabaseError as caught:
        error = caught
    assert error is not None, project
    assert error.sqlite_errorcode & 255 == sqlite3.SQLITE_TOOBIG, error
    assert served == []
    db.close()


@pytest.mark.parametrize("damage", ["root-missing", "neighbor-dangling", "entry-vector-nan"])
def test_coincident_corruption_and_visit_budget_records_error_without_rows(damage):
    """C1 coincident faults: error and no rows; which error fires is recorded.

    Under the approved relaxed contract a corruption that coincides with a
    tight max_visits may surface either SQLITE_CORRUPT or SQLITE_TOOBIG. The
    test records the observed class instead of requiring one, and requires
    fail-closed behaviour (an error with no served rows). Lean v11 faults keep
    node-row public_rowid/state/adjacency corruption on the published node (root
    identity and dangling adjacency) and use the canonical scored entry for the
    vector-payload class (D10 row 2).
    """
    db = open_db()
    create(db, options="degree=2,build_list_size=4,search_list_size=4,max_visits=4")
    for i in range(16):
        insert(db, i, [i, 0])
    inject_csr(db)
    db.execute("PRAGMA ignore_check_constraints=ON")
    if damage == "root-missing":
        db.execute("DELETE FROM v_diskann_graph_nodes WHERE state=2")
    elif damage == "neighbor-dangling":
        db.execute("DELETE FROM v_diskann_graph_nodes WHERE native_local_id=0")
    else:
        scored = db.execute(
            "SELECT e.entry_id FROM v_diskann_current c JOIN v_diskann_entries e "
            "ON e.entry_id=c.entry_id WHERE c.public_rowid=0"
        ).fetchone()[0]
        db.execute("UPDATE v_diskann_entries SET vector=? WHERE entry_id=?", (blob([np.nan, 0]), scored))
    served = []
    error = None
    try:
        cursor = db.execute(
            "SELECT rowid,embedding,distance FROM v WHERE knn_search(embedding,knn_param(?,?))",
            (blob([0, 0]), 16),
        )
        try:
            while True:
                row = cursor.fetchone()
                if row is None:
                    break
                served.append(row)
        finally:
            cursor.close()
    except sqlite3.DatabaseError as caught:
        error = caught
    assert error is not None, damage
    assert served == [], (damage, served)
    code = error.sqlite_errorcode
    assert code & 255 in (sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_TOOBIG), (damage, error)
    print(
        "T4_COINCIDENT "
        + json.dumps(
            {
                "damage": damage,
                "max_visits": 4,
                "sqlite_errorcode": code,
                "error_class": type(error).__name__,
                "message": str(error),
            }
        )
    )
    db.close()


def test_same_connection_build_guard_with_open_knn_cursor(tmp_path):
    """Build guard: an open KNN cursor keeps the same-connection build rejected.

    doc/diskann.md documents that vectorlite_index_build is rejected before any
    native construction when the calling connection holds another cursor.
    Observed rejection: SQLITE_MISUSE (InterfaceError, "DiskANN build requires
    transaction state NONE in every schema"); after the cursor is closed the
    same call succeeds with a candidate token and the lifecycle completes.
    """
    path = tmp_path / "build-guard.db"
    db = open_db(path)
    db.execute("PRAGMA journal_mode=WAL")
    create(db)
    for i in range(8):
        insert(db, i, [i, 0])

    def status():
        return json.loads(db.execute("SELECT vectorlite_index_status('main','v')").fetchone()[0])

    def control(operation, options=None):
        db.execute(
            "INSERT INTO v(operation,path) VALUES(?,?)",
            (operation, json.dumps({} if options is None else options)),
        )

    def job_fields(job):
        return {"job_id": job["job_id"], "fence": job["fence"]}

    def capture():
        for _ in range(10000):
            job = status()["current_job"]
            if job["phase"] == 1:
                return job
            control("index_snapshot_step", job_fields(job))
        pytest.fail("snapshot capture did not finish")

    def build(job):
        payload = {**job_fields(job), "memory_budget_bytes": 128 * 1024 * 1024, "workers": 1}
        raw = db.execute("SELECT vectorlite_index_build('main','v',?)", (json.dumps(payload),)).fetchone()[0]
        return json.loads(raw)

    def finish(job, candidate):
        payload = {**job_fields(job), "candidate_token": candidate["candidate_token"]}
        for phase, operation in ((2, "index_import_step"), (3, "index_validate_step")):
            for _ in range(10000):
                current = status()["current_job"]
                if current["phase"] != phase:
                    break
                control(operation, payload)
            else:
                pytest.fail(f"{operation} did not finish")
        assert status()["current_job"]["phase"] == 4
        control("index_publish", payload)
        control("index_cleanup_step")
        assert status()["current_job"] is None

    try:
        control("index_prepare")
        first_job = capture()
        control("index_claim_build", job_fields(first_job))
        finish(first_job, build(first_job))
        control("index_prepare")
        job = capture()
        control("index_claim_build", job_fields(job))
        payload = json.dumps({**job_fields(job), "memory_budget_bytes": 128 * 1024 * 1024, "workers": 1})
        cursor = db.execute(
            "SELECT rowid,embedding,distance FROM v WHERE knn_search(embedding,knn_param(?,?,?))",
            (blob([0, 0]), 3, json.dumps({"search_list_size": 4})),
        )
        try:
            assert cursor.fetchone() is not None
            with pytest.raises(sqlite3.Error, match="busy|cursor|statement|transaction|NONE") as blocked:
                db.execute("SELECT vectorlite_index_build('main','v',?)", (payload,)).fetchall()
            assert blocked.value.sqlite_errorcode == sqlite3.SQLITE_MISUSE
        finally:
            cursor.close()
        assert status()["current_job"]["phase"] == 2
        candidate = build(job)
        assert candidate.get("candidate_token")
        finish(job, candidate)
        rows = query(db, vector=[0, 0], k=2, l=4)
        assert len(rows) == 2
    finally:
        db.close()


def _reentrant_dataset(tmp_path, name):
    """One 128-row complete-graph published fixture in WAL mode."""
    path = tmp_path / name
    db = open_db(path)
    db.execute("PRAGMA journal_mode=WAL")
    create(db, dim=2, options="degree=128,build_list_size=130,search_list_size=4,max_visits=4096")
    for i in range(128):
        insert(db, i, [i, 0])
    inject_csr(db, adjacency=[[j for j in range(128) if j != i] for i in range(128)] + [list(range(128))])
    return db


def _drain_knn_rows(db):
    """Drains one k=10/L=100 projected KNN statement into (rows, error)."""
    served = []
    error = None
    try:
        cursor = db.execute(
            "SELECT rowid,embedding,distance FROM v WHERE knn_search(embedding,knn_param(?,?,?))",
            (blob([0, 0]), 10, json.dumps({"search_list_size": 100})),
        )
        try:
            while True:
                row = cursor.fetchone()
                if row is None:
                    break
                served.append(row)
        finally:
            cursor.close()
    except sqlite3.DatabaseError as caught:
        error = caught
    return served, error


def _projected_signature(rows):
    return [(int(row[0]), struct.pack("<f", row[2]), bytes(row[1])) for row in rows]


def test_reentrant_mid_query_payload_corruption_fails_closed(tmp_path):
    """D10 row 2 / last row: mid-query payload damage is never served.

    Lean Format 5 splits payload authority: live vectors are canonical in
    diskann_entries (so the publish-time vector guards apply to the scored
    canonical read and stay query-visible, D10 row 2) while the node row remains
    authoritative for public_rowid/state/adjacency. A progress handler commits
    one mutation while a k=10/L=100 projected KNN statement runs:
      * a second connection replaces a scored canonical vector with NaN;
      * a second connection truncates a scored canonical vector blob;
      * the querying connection itself replaces a scored canonical vector (same-
        connection reentrant DML fails closed through the D3 detector or the
        payload guard, never with stale rows);
      * a second connection performs an incremental sqlite3_blob_write on the
        scored canonical entries vector (query-visible again, D10 row 2);
      * a second connection truncates the frozen root's adjacency payload, the
        node-row class that keeps D10's re-read guard semantics (the v5 node
        table is WITHOUT ROWID, so sqlite3_blob_write cannot open it).

    A second connection's commit cannot invalidate this connection's read
    snapshot, so those branches return exactly the pre-mutation rows. A fresh
    query must observe the committed damage and fail closed, proving no reader or
    payload cache survives one query.
    """
    scenarios = (
        "second-connection-scored-nan",
        "second-connection-scored-short",
        "same-connection-scored-nan",
        "second-connection-scored-blob",
        "second-connection-root-adjacency",
    )
    for scenario in scenarios:
        db = _reentrant_dataset(tmp_path, f"reentrant-payload-{scenario}.db")
        mutator = None
        blob_handle = None
        progress = {"fired": 0, "done": False, "mutation": None}
        try:
            base = query(db, vector=[0, 0], k=10, l=100, project=True)
            assert base
            entry = db.execute("SELECT entry_id FROM v_diskann_current WHERE public_rowid=1").fetchone()[0]
            if scenario != "same-connection-scored-nan":
                mutator = open_db(tmp_path / f"reentrant-payload-{scenario}.db")
                # The truncated-blob variants are deliberate unsupported shadow
                # corruption injections; the CHECK guard is bypassed for them.
                mutator.execute("PRAGMA ignore_check_constraints=ON")
            if scenario == "second-connection-scored-blob":
                blob_handle = mutator.blobopen("v_diskann_entries", "vector", entry)

            def handler():
                progress["fired"] += 1
                if progress["done"] or progress["fired"] < 3:
                    return 0
                progress["done"] = True
                try:
                    if scenario == "second-connection-scored-blob":
                        blob_handle.write(blob([np.nan, 0]))
                        blob_handle.close()
                    elif scenario == "same-connection-scored-nan":
                        db.execute(
                            "UPDATE v_diskann_entries SET vector=? WHERE entry_id=?",
                            (blob([np.nan, 0]), entry),
                        )
                    elif scenario == "second-connection-scored-short":
                        mutator.execute(
                            "UPDATE v_diskann_entries SET vector=zeroblob(1) WHERE entry_id=?",
                            (entry,),
                        )
                    elif scenario == "second-connection-root-adjacency":
                        mutator.execute(
                            "UPDATE v_diskann_graph_nodes SET neighbors=zeroblob(8) WHERE state=2"
                        )
                    else:
                        mutator.execute(
                            "UPDATE v_diskann_entries SET vector=? WHERE entry_id=?",
                            (blob([np.nan, 0]), entry),
                        )
                    progress["mutation"] = "committed"
                except sqlite3.Error as caught:
                    progress["mutation"] = f"{type(caught).__name__}: {caught}"
                return 0

            db.set_progress_handler(handler, 50)
            try:
                served, error = _drain_knn_rows(db)
            finally:
                db.set_progress_handler(None, 0)
            assert progress["fired"] > 0, scenario
            assert progress["mutation"] == "committed", (scenario, progress, error)
            if scenario == "same-connection-scored-nan":
                # The reentrant write itself is the mutation: the operation must
                # fail closed and never serve a stale row.
                assert served == [], (scenario, served)
                assert error is not None, scenario
                assert error.sqlite_errorcode & 255 in (
                    sqlite3.SQLITE_BUSY,
                    sqlite3.SQLITE_CORRUPT,
                ), (scenario, error)
                fresh_served, fresh_error = _drain_knn_rows(db)
                assert fresh_served == [], (scenario, fresh_served)
                assert fresh_error is not None
                assert fresh_error.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT
            else:
                # A second connection's commit cannot move this connection's read
                # snapshot, so the in-flight statement returns exactly the
                # pre-mutation rows; a fresh statement observes the committed
                # damage and fails closed.
                assert error is None, (scenario, error)
                assert _projected_signature(served) == _projected_signature(base), scenario
                fresh_served, fresh_error = _drain_knn_rows(db)
                assert fresh_served == [], (scenario, fresh_served)
                assert fresh_error is not None, scenario
                assert fresh_error.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, (scenario, fresh_error)
            print(
                "T6_REENTRANT_PAYLOAD "
                + json.dumps(
                    {
                        "scenario": scenario,
                        "progress_callbacks": progress["fired"],
                        "mutation": progress["mutation"],
                        "in_flight_outcome": "error" if error is not None else "rows",
                        "in_flight_error": None if error is None else str(error),
                    },
                    sort_keys=True,
                )
            )
        finally:
            if blob_handle is not None:
                try:
                    blob_handle.close()
                except sqlite3.Error:
                    pass
            if mutator is not None:
                mutator.close()
            db.close()


def test_reentrant_mid_query_finite_published_mutation_never_serves_stale_rows(tmp_path):
    """Reentrancy/staleness: a committed mid-query canonical payload is never stale.

    Lean Format 5 scores live candidates from canonical diskann_entries, so the
    published payload a graph-route query reads is the canonical vector. A second
    connection replaces the canonical vector of row 1 with [50, 0] while a
    k=10/L=100 query runs. The in-flight statement either fails closed or returns
    only self-consistent pre-mutation rows (its snapshot). The next query must
    return exactly the post-mutation canonical result (row 1 leaves the top 10),
    which is impossible if any reader or payload cache retained the old bytes.
    """
    db = _reentrant_dataset(tmp_path, "finite-stale-format5.db")
    target_entry = db.execute(
        "SELECT entry_id FROM v_diskann_current WHERE public_rowid=1"
    ).fetchone()[0]
    mutator = open_db(tmp_path / "finite-stale-format5.db")
    progress = {"fired": 0, "done": False, "mutation": None}

    def handler():
        progress["fired"] += 1
        if progress["done"] or progress["fired"] < 3:
            return 0
        progress["done"] = True
        try:
            mutator.execute(
                "UPDATE v_diskann_entries SET vector=? WHERE entry_id=?",
                (blob([50, 0]), target_entry),
            )
            progress["mutation"] = "committed"
        except sqlite3.Error as caught:
            progress["mutation"] = f"{type(caught).__name__}: {caught}"
        return 0

    expected_post = [(rowid, blob([rowid, 0]), float(rowid * rowid)) for rowid in (0, 2, 3, 4, 5, 6, 7, 8, 9, 10)]
    try:
        base = query(db, vector=[0, 0], k=10, l=100, project=True)
        assert [row[0] for row in base] == list(range(10))
        db.set_progress_handler(handler, 50)
        try:
            served, error = _drain_knn_rows(db)
        finally:
            db.set_progress_handler(None, 0)
        assert progress["fired"] > 0
        assert progress["mutation"] == "committed", (progress, error)
        if error is not None:
            assert served == []
            assert error.sqlite_errorcode & 255 in (sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_BUSY), error
        else:
            assert len(served) == 10
            assert len({row[0] for row in served}) == 10
            distances = [row[2] for row in served]
            assert distances == sorted(distances)
            for rowid, embedding, distance in served:
                expected_embedding = blob([rowid, 0])
                expected_distance = float(rowid * rowid)
                assert embedding == expected_embedding, (rowid, embedding)
                assert struct.pack("<f", distance) == struct.pack("<f", expected_distance), (rowid, distance)
        assert db.execute(
            "SELECT vector FROM v_diskann_entries WHERE entry_id=?", (target_entry,)
        ).fetchone() == (blob([50, 0]),)
        fresh = query(db, vector=[0, 0], k=10, l=100, project=True)
        assert fresh == expected_post, fresh
        print(
            "T6_STALE_FINITE "
            + json.dumps(
                {
                    "progress_callbacks": progress["fired"],
                    "mutation": progress["mutation"],
                    "in_flight_outcome": "error" if error is not None else "rows",
                    "in_flight_rowids": [int(row[0]) for row in served],
                    "fresh_rowids": [int(row[0]) for row in fresh],
                },
                sort_keys=True,
            )
        )
    finally:
        mutator.close()
        db.close()


def test_reentrant_mid_query_dml_and_relocating_ddl_fail_busy(tmp_path):
    """D10 row 6: reentrant DML and shadow-relocating DDL fail closed with BUSY.

    Both detector stamps are exercised deterministically while one k=10/L=100
    KNN statement runs, with the mutation committed from a progress handler on
    the querying connection:
      * "dml": a same-connection shadow UPDATE moves sqlite3_total_changes64;
      * "rename": ALTER TABLE v_diskann_jobs RENAME TO v_diskann_jobs_old
        removes one shadow name from the nine-name count (the table is not read
        by the query, so the statement reaches stamp verification);
      * "ctas": the same rename followed by a CTAS recreation restores the name
        count while the rootpage sum moves (the renamed original keeps its
        pages; OP_Destroy cannot free them mid-operation).

    Contract: the in-flight statement raises SQLITE_BUSY and serves no rows
    (fail closed before any row is returned). The landed schema stays usable:
    a fresh query succeeds with the same answer, and the renamed table keeps its
    rows.
    """
    for scenario in ("dml", "rename", "ctas"):
        db = _reentrant_dataset(tmp_path, f"reentrant-detector-{scenario}.db")
        try:
            base = query(db, vector=[0, 0], k=10, l=100, project=True)
            assert base
            baseline = _projected_signature(base)
            jobs_rows = db.execute("SELECT count(*) FROM v_diskann_jobs").fetchone()[0]
            progress = {"fired": 0, "readers_traced": 0, "done": False, "mutation": None}

            def tracer(sql):
                if _NODE_READER_MARKER in sql:
                    progress["readers_traced"] += 1

            def handler():
                progress["fired"] += 1
                if progress["done"] or not progress["readers_traced"]:
                    return 0
                progress["done"] = True
                try:
                    if scenario == "dml":
                        db.execute("UPDATE v_diskann_meta SET retired_count=retired_count")
                    elif scenario == "rename":
                        db.execute("ALTER TABLE v_diskann_jobs RENAME TO v_diskann_jobs_old")
                    else:
                        db.execute("ALTER TABLE v_diskann_jobs RENAME TO v_diskann_jobs_old")
                        db.execute("CREATE TABLE v_diskann_jobs AS SELECT * FROM v_diskann_jobs_old")
                    progress["mutation"] = "committed"
                except sqlite3.Error as caught:
                    progress["mutation"] = f"{type(caught).__name__}: {caught}"
                return 0

            db.set_trace_callback(tracer)
            db.set_progress_handler(handler, 50)
            try:
                served, error = _drain_knn_rows(db)
            finally:
                db.set_progress_handler(None, 0)
                db.set_trace_callback(None)
            assert progress["fired"] > 0, scenario
            assert progress["readers_traced"] > 0, scenario
            assert progress["mutation"] == "committed", (scenario, progress)
            assert served == [], (scenario, served)
            assert error is not None, scenario
            assert error.sqlite_errorcode & 255 == sqlite3.SQLITE_BUSY, (scenario, error)
            if scenario == "dml":
                assert _projected_signature(
                    query(db, vector=[0, 0], k=10, l=100, project=True)
                ) == baseline
            else:
                # The landed hostile schema is not repaired: the renamed shadow
                # table keeps its rows, and the next query fails closed rather
                # than silently serving from a changed schema.
                assert db.execute("SELECT count(*) FROM v_diskann_jobs_old").fetchone()[0] == jobs_rows
                fresh = []
                fresh_error = None
                try:
                    fresh = query(db, vector=[0, 0], k=10, l=100, project=True)
                except sqlite3.DatabaseError as caught:
                    fresh_error = caught
                assert fresh == [], (scenario, fresh)
                assert fresh_error is not None, scenario
            print(
                "T6_DETECTOR "
                + json.dumps(
                    {
                        "scenario": scenario,
                        "progress_callbacks": progress["fired"],
                        "mutation": progress["mutation"],
                        "error_code": error.sqlite_errorcode,
                        "error_name": error.sqlite_errorname,
                        "rows_returned": len(served),
                    },
                    sort_keys=True,
                )
            )
        finally:
            db.close()


# ---------------------------------------------------------------------------
# T4(f) contract additions: transparent reprepare under a mid-query schema
# change, terminal reprepare failure after a reader was used, and small-budget
# scalar-fallback equivalence. Successful-path results stay strict; failure
# paths must stay fail-closed under C1.
# ---------------------------------------------------------------------------


# Format-5 node-reader SQL as prepared by the query adapter. Every graph-route
# node shape (NodeExpand, NodeScore and NodeIdentity) keys on
# generation_id/native_local_id; the D3 fallback join keys the same way. SQLite's
# statement trace reports the exact prepared SQL, so observing this marker proves
# a reader statement started executing before the handler below commits its DDL.
_NODE_READER_MARKER = "native_local_id=?2"


def _debug_library():
    """Debug instrumentation handle for the artifact, or None when unavailable.

    Debug extension builds export the plan D8 counters; release artifacts export
    no vectorlite_test_ symbol. Loading the same artifact path again reuses the
    process image SQLite already mapped, so the counters observe this
    extension's own work. Release builds return None and callers only record.
    """
    try:
        import ctypes

        library = getattr(_debug_library, "library", None)
        if library is None:
            library = ctypes.CDLL(str(artifact()))
            _debug_library.library = library
        return library
    except (OSError, TypeError):
        return None


def _debug_counter(name):
    """One monotonic Debug-only u64 counter, or None when unavailable.

    Release artifacts export no vectorlite_test_ symbol, so both the function
    lookup and the call are treated as "instrumentation unavailable"; callers
    only assert when a counter was verified live.
    """
    library = _debug_library()
    try:
        import ctypes

        function = getattr(library, name)
        function.restype = ctypes.c_uint64
        function.argtypes = []
        return int(function())
    except (AttributeError, TypeError):
        return None


def _debug_counters(*names):
    """Reads several Debug-only counters, all None when instrumentation is absent."""
    values = tuple(_debug_counter(name) for name in names)
    if any(value is None for value in values):
        return tuple(None for _ in values)
    return values


def _debug_hook(name):
    """Invokes one one-shot Debug-only hook; False when instrumentation is absent."""
    library = _debug_library()
    try:
        function = getattr(library, name)
        function.restype = None
        function.argtypes = []
        function()
        return True
    except (AttributeError, TypeError):
        return False


def _debug_reader_executions():
    """Retained-reader executions since process start, or None when unavailable."""
    library = _debug_library()
    try:
        return int(library.vectorlite_test_reader_executions())
    except (AttributeError, TypeError):
        return None


def _debug_statement_counters():
    """(successful prepares, finalizes) since the last reset, or None."""
    library = _debug_library()
    try:
        import ctypes

        counters = (ctypes.c_uint64 * 2)()
        library.vectorlite_test_read_counters(counters)
        return (int(counters[0]), int(counters[1]))
    except (AttributeError, TypeError):
        return None


def _debug_reset_statement_counters():
    """Resets the Debug prepare/finalize counters; False when unavailable."""
    library = _debug_library()
    try:
        library.vectorlite_test_reset_counters()
        return True
    except (AttributeError, TypeError):
        return False


def _rows_signature(rows, project):
    """Exact comparable signature: rowids, order, projection bytes, distance bits."""
    if project:
        return [(int(row[0]), bytes(row[1]), struct.pack("<f", row[2])) for row in rows]
    return [(int(row[0]), struct.pack("<f", row[1])) for row in rows]


def _status(db, name="v", schema="main"):
    """Public status JSON for one DiskANN table."""
    return json.loads(
        db.execute("SELECT vectorlite_index_status(?,?)", (schema, name)).fetchone()[0]
    )


def _control(db, operation, options=None, name="v", schema="main"):
    """Drives one public control operation through the virtual table."""
    db.execute(
        f'INSERT INTO "{schema}"."{name}"(operation,path) VALUES(?,?)',
        (operation, json.dumps({} if options is None else options)),
    )


def _prepare_and_capture(db, name="v", schema="main"):
    """Starts a public deferred job and drains its bounded snapshot capture."""
    _control(db, "index_prepare", name=name, schema=schema)
    for _ in range(10000):
        job = _status(db, name, schema)["current_job"]
        if job["phase"] == 1:
            return job
        assert job["phase"] == 0, job
        _control(
            db,
            "index_snapshot_step",
            {
                "job_id": job["job_id"],
                "fence": job["fence"],
                "max_records": 256,
                "max_bytes": 65536,
            },
            name,
            schema,
        )
    pytest.fail("snapshot capture did not finish")


def _seed_published(db, count, options, adjacency=None):
    """Seeds canonical rows and installs one published generation with the helpers."""
    create(db, dim=2, options=options)
    for i in range(count):
        insert(db, i, [i, 0])
    if adjacency is None:
        adjacency = [[(i + 1) % count] for i in range(count)] + [[0]]
    inject_csr(db, adjacency=adjacency)
    return adjacency


def _drain_knn(db, k, l, project=True):
    """Drains one KNN cursor, returning the served rows and any fail-closed error."""
    columns = "rowid,embedding,distance" if project else "rowid,distance"
    served = []
    error = None
    try:
        cursor = db.execute(
            f"SELECT {columns} FROM v WHERE knn_search(embedding,knn_param(?,?,?))",
            (blob([0, 0]), k, json.dumps({"search_list_size": l})),
        )
        try:
            while True:
                row = cursor.fetchone()
                if row is None:
                    break
                served.append(row)
        finally:
            cursor.close()
    except sqlite3.DatabaseError as caught:
        error = caught
    return served, error


def test_reprepare_mid_query_schema_change_is_transparent_or_fail_closed():
    """T4(f): a mid-query schema change reprepares transparently or fails closed.

    A maintained table (canonical inserts through the virtual table plus a
    published Format 5 generation installed by this file's trusted inject_csr
    fixture; the reader SQL and routing are the maintained index's) and an
    unrelated ordinary table are created. One k=10/L=100 projected KNN statement
    runs while a progress handler commits exactly ONE benign schema change on
    the same connection - CREATE TABLE t4_reprepare_probe(x) - and then removes
    itself.

    The handler is gated on the traced identity-reader SQL and runs at the first
    progress callback (fixed 50-VM-step quantum) after that reader started
    executing, so the DDL lands after a reader was used and before the remaining
    reads. SQLite then expires the retained per-operation readers; each must
    reprepare transparently on its next step.

    Contract: the query either returns exactly the baseline rows (same rowids,
    order, projection bytes and distance bits: reprepare succeeded
    transparently) or raises sqlite3.Error with no rows (fail closed). Wrong or
    stale rows are never accepted. Debug builds additionally record reader
    executions before and after the schema change and prepare/finalize balance,
    and assert them when the counter is verified live.

    Limitations recorded here: a same-connection DROP TABLE (of the unrelated
    table or of a shadow table) cannot be driven mid-reader at all. SQLite
    refuses every OP_Destroy while another reader VM is active on the connection
    and raises SQLITE_LOCKED "database table is locked", so CREATE TABLE is the
    deterministic benign DDL that can commit mid-query. The exact VM step at
    which SQLite first reports the reader trace is not a portable constant, so
    the handler waits for the traced identity-reader SQL instead of a hard-coded
    callback index; the 50-step quantum is fixed.
    """
    db = open_db()
    try:
        count = 128
        full = [[j for j in range(count) if j != i] for i in range(count)] + [list(range(count))]
        _seed_published(
            db,
            count,
            "degree=128,build_list_size=130,search_list_size=10,max_visits=65536",
            adjacency=full,
        )
        db.execute("CREATE TABLE unrelated(value)")
        db.execute("INSERT INTO unrelated VALUES(1)")
        baseline = query(db, vector=[0, 0], k=10, l=100, project=True)
        assert len(baseline) == 10, baseline
        expected = _rows_signature(baseline, True)
        probe = "t4_reprepare_probe"

        # Probe both optional Debug instrumentation channels with one control
        # query. Counters that do not move are treated as unavailable (a release
        # artifact exports no symbols; a second image would start from zero), so
        # no assertion below depends on a possibly unshared counter.
        counters_ready = _debug_reset_statement_counters()
        before_control = _debug_reader_executions()
        control = query(db, vector=[0, 0], k=10, l=100, project=True)
        after_control = _debug_reader_executions()
        control_counters = _debug_statement_counters() if counters_ready else None
        live_readers = (
            before_control is not None
            and after_control is not None
            and after_control > before_control
        )
        live_counters = control_counters is not None and control_counters[0] > 0
        assert _rows_signature(control, True) == expected

        state = {
            "fired": 0,
            "readers_traced": 0,
            "readers_at_ddl": None,
            "ddl": None,
            "ddl_error": None,
            "schema_before": None,
            "schema_after": None,
            "reader_executions_at_ddl": None,
            "done": False,
            "in_ddl": False,
        }

        def tracer(sql):
            if _NODE_READER_MARKER in sql:
                state["readers_traced"] += 1

        def handler():
            state["fired"] += 1
            if state["done"] or state["in_ddl"] or not state["readers_traced"]:
                return 0
            state["in_ddl"] = True
            try:
                state["done"] = True
                state["readers_at_ddl"] = state["readers_traced"]
                state["reader_executions_at_ddl"] = _debug_reader_executions()
                state["schema_before"] = db.execute("PRAGMA schema_version").fetchone()[0]
                try:
                    db.execute(f"CREATE TABLE {probe}(x)")
                    state["schema_after"] = db.execute("PRAGMA schema_version").fetchone()[0]
                    state["ddl"] = "created"
                except sqlite3.Error as caught:
                    state["ddl_error"] = f"{type(caught).__name__}: {caught}"
                finally:
                    db.set_progress_handler(None, 0)
            finally:
                state["in_ddl"] = False
            return 0

        _debug_reset_statement_counters()
        before_test = _debug_reader_executions()
        db.set_trace_callback(tracer)
        db.set_progress_handler(handler, 50)
        try:
            served, error = _drain_knn(db, 10, 100)
        finally:
            db.set_progress_handler(None, 0)
            db.set_trace_callback(None)
        after_test = _debug_reader_executions()
        measured_counters = _debug_statement_counters()

        assert state["fired"] > 0, state
        assert state["readers_traced"] > 0, state
        assert state["ddl"] == "created", state
        assert state["schema_after"] > state["schema_before"], state
        assert db.execute(
            "SELECT count(*) FROM sqlite_master WHERE name=?", (probe,)
        ).fetchone() == (1,)
        assert db.execute("SELECT count(*) FROM unrelated").fetchone() == (1,)
        if error is None:
            branch = "baseline-equal"
            assert _rows_signature(served, True) == expected, served
            assert state["readers_traced"] > state["readers_at_ddl"], state
        else:
            branch = "fail-closed"
            assert isinstance(error, sqlite3.Error)
            assert served == [], served
        if live_readers:
            assert state["reader_executions_at_ddl"] is not None, state
            assert state["reader_executions_at_ddl"] - before_test >= 1, state
            assert after_test - state["reader_executions_at_ddl"] >= 1, state
        if live_counters:
            assert measured_counters is not None, state
            assert measured_counters[0] >= 1, state
            assert measured_counters[0] == measured_counters[1], state
        assert _rows_signature(
            query(db, vector=[0, 0], k=10, l=100, project=True), True
        ) == expected
        print(
            "T4_REPREPARE "
            + json.dumps(
                {
                    "branch": branch,
                    "ddl": state["ddl"],
                    "schema_version_before": state["schema_before"],
                    "schema_version_after": state["schema_after"],
                    "progress_callbacks": state["fired"],
                    "reader_statements_before_ddl": state["readers_at_ddl"],
                    "reader_statements_after_ddl": state["readers_traced"]
                    - state["readers_at_ddl"],
                    "reader_executions_before_ddl": None
                    if not live_readers
                    else state["reader_executions_at_ddl"] - before_test,
                    "reader_executions_after_ddl": None
                    if not live_readers
                    else after_test - state["reader_executions_at_ddl"],
                    "prepare_finalize_balance": None
                    if not live_counters
                    else {
                        "prepares": measured_counters[0],
                        "finalizes": measured_counters[1],
                    },
                    "instrumentation_live": {
                        "reader_executions": live_readers,
                        "statement_counters": live_counters,
                    },
                    "rows_returned": len(served),
                    "error": None
                    if error is None
                    else {
                        "code": error.sqlite_errorcode,
                        "name": error.sqlite_errorname,
                        "message": str(error),
                    },
                },
                sort_keys=True,
            )
        )
    finally:
        db.close()


def test_terminal_reprepare_failure_after_reader_use_is_fail_closed(tmp_path):
    """T4(f) failure branch: terminal reprepare failure serves no rows.

    A scratch database copy is taken with the backup API from an in-memory
    source (so the damage below cannot leak), and one published generation is
    queried. While the k=2/L=4 projected KNN statement runs, a progress handler
    gated on the traced node-reader SQL (Format-5 readers key on
    generation_id/native_local_id) commits ONE schema change that invalidates
    the readers' SQL on the same connection: ALTER TABLE v_diskann_graph_nodes
    RENAME TO v_diskann_graph_nodes_renamed. Retained readers are expired and
    the next reader step must reprepare against a schema where
    v_diskann_graph_nodes no longer exists, failing terminally after a reader
    was used.

    Contract: the statement raises sqlite3.Error and serves no rows, the
    connection stays usable (simple SELECT works, the renamed table still holds
    every published node row), and a fresh KNN query does not serve stale rows
    (it fails closed again). The source database is unaffected. Debug builds
    also assert and record prepare/finalize balance after the failure.

    Limitation recorded: the requested DROP TABLE variant cannot be driven
    deterministically mid-query. SQLite refuses any DROP TABLE (OP_Destroy)
    while another reader VM is active on the connection - observed as
    SQLITE_LOCKED "database table is locked" - and a DROP executed after the
    readers finish cannot invalidate them. ALTER TABLE RENAME is the strongest
    deterministic DDL that removes the needed shadow table name from the schema
    while readers are live, and it produces the same terminal step/reprepare
    failure class (primary SQL error preserved, no rows).
    """
    source = open_db()
    target = None
    try:
        options = "degree=4,build_list_size=4,search_list_size=4,max_visits=4096"
        _seed_published(source, 16, options)
        baseline = query(source, vector=[0, 0], k=2, l=4, project=True)
        assert len(baseline) == 2, baseline
        expected = _rows_signature(baseline, True)
        node_rows = source.execute("SELECT count(*) FROM v_diskann_graph_nodes").fetchone()
        target = open_db(tmp_path / "terminal-reprepare.db")
        source.backup(target)
        assert _rows_signature(
            query(target, vector=[0, 0], k=2, l=4, project=True), True
        ) == expected

        counters_ready = _debug_reset_statement_counters()
        control = query(target, vector=[0, 0], k=2, l=4, project=True)
        control_counters = _debug_statement_counters() if counters_ready else None
        live_counters = control_counters is not None and control_counters[0] > 0
        assert _rows_signature(control, True) == expected

        state = {
            "fired": 0,
            "readers_traced": 0,
            "readers_at_ddl": None,
            "ddl": None,
            "ddl_error": None,
            "reader_executions_at_ddl": None,
            "done": False,
            "in_ddl": False,
        }

        def tracer(sql):
            if _NODE_READER_MARKER in sql:
                state["readers_traced"] += 1

        def handler():
            state["fired"] += 1
            if state["done"] or state["in_ddl"] or not state["readers_traced"]:
                return 0
            state["in_ddl"] = True
            try:
                state["done"] = True
                state["readers_at_ddl"] = state["readers_traced"]
                state["reader_executions_at_ddl"] = _debug_reader_executions()
                try:
                    target.execute(
                        "ALTER TABLE v_diskann_graph_nodes RENAME TO v_diskann_graph_nodes_renamed"
                    )
                    state["ddl"] = "renamed"
                except sqlite3.Error as caught:
                    state["ddl_error"] = f"{type(caught).__name__}: {caught}"
                finally:
                    target.set_progress_handler(None, 0)
            finally:
                state["in_ddl"] = False
            return 0

        _debug_reset_statement_counters()
        target.set_trace_callback(tracer)
        target.set_progress_handler(handler, 50)
        try:
            served, error = _drain_knn(target, 2, 4)
        finally:
            target.set_progress_handler(None, 0)
            target.set_trace_callback(None)
        measured_counters = _debug_statement_counters()

        assert state["fired"] > 0, state
        assert state["readers_traced"] > 0, state
        assert state["ddl"] == "renamed", state
        assert error is not None
        assert isinstance(error, sqlite3.Error)
        assert served == [], served
        assert target.execute("SELECT 1").fetchone() == (1,)
        assert target.execute(
            "SELECT count(*) FROM v_diskann_graph_nodes_renamed"
        ).fetchone() == node_rows
        fresh_served, fresh_error = _drain_knn(target, 2, 4)
        assert fresh_error is not None, fresh_served
        assert isinstance(fresh_error, sqlite3.Error)
        assert fresh_served == [], fresh_served
        if live_counters:
            assert measured_counters is not None, state
            assert measured_counters[0] >= 1, state
            assert measured_counters[0] == measured_counters[1], state
        assert _rows_signature(
            query(source, vector=[0, 0], k=2, l=4, project=True), True
        ) == expected
        assert source.execute(
            "SELECT count(*) FROM sqlite_master WHERE name='v_diskann_graph_nodes'"
        ).fetchone() == (1,)
        print(
            "T4_TERMINAL_REPREPARE "
            + json.dumps(
                {
                    "ddl": state["ddl"],
                    "progress_callbacks": state["fired"],
                    "reader_statements_before_ddl": state["readers_at_ddl"],
                    "reader_executions_at_ddl": state["reader_executions_at_ddl"],
                    "prepare_finalize_balance": None
                    if not live_counters
                    else {
                        "prepares": measured_counters[0],
                        "finalizes": measured_counters[1],
                    },
                    "instrumentation_live": live_counters,
                    "rows_returned": len(served),
                    "error": {
                        "code": error.sqlite_errorcode,
                        "name": error.sqlite_errorname,
                        "message": str(error),
                    },
                    "fresh_query_error": {
                        "code": fresh_error.sqlite_errorcode,
                        "name": fresh_error.sqlite_errorname,
                        "message": str(fresh_error),
                    },
                    "connection_usable": True,
                    "source_unchanged": True,
                },
                sort_keys=True,
            )
        )
    finally:
        if target is not None:
            target.close()
        source.close()


_SMALL_BUDGET_OPTIONS = "degree=4,build_list_size=4,search_list_size=4,max_visits=4096"
# T0 fixed baseline-derived minimum (16-row small-budget fixture; binding shape
# "filter"). D6 freezes it: tests probe one byte below/at/above this constant
# instead of re-deriving a per-artifact threshold.
_SMALL_BUDGET_MIN_CACHE_BYTES = 711_832
_SMALL_BUDGET_CASES = [
    ("plain", dict(vector=[0, 0], k=2, l=4, project=False)),
    (
        "filter",
        dict(
            vector=[0, 0],
            k=2,
            l=4,
            project=False,
            filter_sql="AND rowid IN (0,1,2,3,4,5,6,7)",
        ),
    ),
    ("project", dict(vector=[3, 0], k=3, l=4, project=True)),
]


def _small_budget_db(cache_bytes):
    """Builds one seeded, published database at a cache_bytes budget (None = default)."""
    db = open_db()
    try:
        options = _SMALL_BUDGET_OPTIONS
        if cache_bytes is not None:
            options += f",cache_bytes={cache_bytes}"
        create(db, dim=2, options=options)
        for i in range(16):
            insert(db, i, [i, 0])
        inject_csr(db, adjacency=[[(i + 1) % 16] for i in range(16)] + [[0]])
        return db
    except BaseException:
        db.close()
        raise


def _small_budget_outcomes(cache_bytes, cases):
    """Returns ({label: rows}, None) when every case is admitted, else (None, error)."""
    try:
        db = _small_budget_db(cache_bytes)
    except sqlite3.Error as error:
        return None, error
    try:
        results = {}
        for label, case in cases:
            try:
                results[label] = query(db, **case)
            except sqlite3.Error as error:
                return None, error
        return results, None
    finally:
        db.close()


def test_small_budget_pins_the_t0_minimum_and_carry_is_disabled():
    """D6/T0: the fixed minimum admitted cache_bytes is 711,832, not re-derived.

    T0 derived min_admitted_cache_bytes = 711,832 for this fixture (16 rows,
    degree=4/build_list_size=4/search_list_size=4/max_visits=4096; binding shape
    "filter"). One byte lower must reject every plain/filter/projected query with
    HEAD's SQLITE_TOOBIG scratch message; 711,832, 711,833 and 711,834 must admit
    all of them with identical rowids, order, projection bytes and distance bits,
    zero retained-reader executions and the D6 carry disabled.
    """
    cases = _SMALL_BUDGET_CASES
    results, error = _small_budget_outcomes(_SMALL_BUDGET_MIN_CACHE_BYTES - 1, cases)
    assert results is None and error is not None
    assert error.sqlite_errorcode == sqlite3.SQLITE_TOOBIG, error
    for budget in range(_SMALL_BUDGET_MIN_CACHE_BYTES, _SMALL_BUDGET_MIN_CACHE_BYTES + 3):
        admitted, admitted_error = _small_budget_outcomes(budget, cases)
        assert admitted_error is None, (budget, admitted_error)
        assert admitted is not None and all(admitted.values()), budget

    default = _small_budget_db(None)
    small = _small_budget_db(_SMALL_BUDGET_MIN_CACHE_BYTES)
    records = []
    try:
        for label, case in cases:
            default_before = _debug_counters(
                "vectorlite_test_reader_executions",
                "vectorlite_test_carry_hits",
            )
            default_rows = query(default, **case)
            default_after = _debug_counters(
                "vectorlite_test_reader_executions",
                "vectorlite_test_carry_hits",
            )
            small_before = _debug_counters(
                "vectorlite_test_reader_executions",
                "vectorlite_test_carry_admissions",
                "vectorlite_test_carry_hits",
                "vectorlite_test_pair_cache_disabled_operations",
                "vectorlite_test_pair_cache_hits",
            )
            small_rows = query(small, **case)
            small_after = _debug_counters(
                "vectorlite_test_reader_executions",
                "vectorlite_test_carry_admissions",
                "vectorlite_test_carry_hits",
                "vectorlite_test_pair_cache_disabled_operations",
                "vectorlite_test_pair_cache_hits",
            )
            assert small_rows, label
            assert default_rows, label
            assert _rows_signature(default_rows, case["project"]) == _rows_signature(
                small_rows, case["project"]
            ), label
            default_readers = (
                None
                if default_before[0] is None or default_after[0] is None
                else default_after[0] - default_before[0]
            )
            small_readers = (
                None
                if small_before[0] is None or small_after[0] is None
                else small_after[0] - small_before[0]
            )
            if small_before[1] is not None:
                # Zero reader allowance: no retained reader executes and neither
                # the carry nor the pair cache is admitted (D6 tight-budget path).
                assert small_readers == 0, (label, small_readers)
                assert small_after[1] - small_before[1] == 0, label
                assert small_after[2] - small_before[2] == 0, label
                # The graph route offers the pair cache last and this budget
                # declines it: the operation is marked disabled and every scored
                # candidate takes the NodeIdentity miss path instead (D6).
                assert small_after[3] - small_before[3] == 1, label
                assert small_after[4] - small_before[4] == 0, label
            if default_readers:
                assert small_readers == 0, (label, small_readers)
            records.append(
                {
                    "case": label,
                    "default_reader_executions": default_readers,
                    "small_reader_executions": small_readers,
                    "pair_cache_disabled": None
                    if small_before[3] is None
                    else small_after[3] - small_before[3],
                    "rows": len(small_rows),
                    "identical_rowids_order_distance_bits": True,
                }
            )
        print(
            "T6_SMALL_BUDGET "
            + json.dumps(
                {
                    "min_admitted_cache_bytes": _SMALL_BUDGET_MIN_CACHE_BYTES,
                    "rejected_cache_bytes": _SMALL_BUDGET_MIN_CACHE_BYTES - 1,
                    "admitted_cache_bytes": [
                        _SMALL_BUDGET_MIN_CACHE_BYTES,
                        _SMALL_BUDGET_MIN_CACHE_BYTES + 1,
                        _SMALL_BUDGET_MIN_CACHE_BYTES + 2,
                    ],
                    "default_cache_bytes": "default",
                    "cases": records,
                },
                sort_keys=True,
            )
        )
    finally:
        small.close()
        default.close()


_CARRY_FIXTURE_OPTIONS = "degree=16,build_list_size=20,search_list_size=4,max_visits=4096"


def _complete_fixture(rows=16, base=0, options=None):
    """Seeded published fixture with a complete digraph and a frozen root."""
    db = open_db()
    create(db, dim=2, options=options or _CARRY_FIXTURE_OPTIONS)
    for i in range(rows):
        insert(db, base + i, [i, 0])
    complete = [[j for j in range(rows + 1) if j != i] for i in range(rows + 1)]
    inject_csr(db, adjacency=complete)
    return db


def test_carry_on_off_equivalence_and_charge_preserving_final_loop():
    """D5/D6: the Debug force-carry-off hook changes no result or visit charge.

    The carry-on operation admits the ring, resolves the accepted top-k in the
    final loop with zero SQL (k carry hits) and charges exactly the same owner
    visits as HEAD. With the carry disabled the final loop performs k real
    charged NodeIdentity reads instead: rows, order, projection bytes and
    distance bits are identical, owner visits are identical, and exactly k extra
    carry misses / NodeIdentity executions appear.
    """
    db = _complete_fixture()
    try:
        vector = [3, 0]
        counters = (
            "vectorlite_test_carry_admissions",
            "vectorlite_test_carry_hits",
            "vectorlite_test_carry_misses",
            "vectorlite_test_node_identity_executions",
        )
        on_before = _debug_counters(*counters)
        on_rows = query(db, vector=vector, k=4, l=8, project=True)
        on_after = _debug_counters(*counters)
        on_visits = _debug_counter("vectorlite_test_last_query_visits")
        # The Debug-only hook is absent from release artifacts; the result and
        # visit-parity assertions below hold either way.
        _debug_hook("vectorlite_test_force_carry_off")
        off_before = _debug_counters(*counters)
        off_rows = query(db, vector=vector, k=4, l=8, project=True)
        off_after = _debug_counters(*counters)
        off_visits = _debug_counter("vectorlite_test_last_query_visits")
        assert _rows_signature(on_rows, True) == _rows_signature(off_rows, True)
        assert len(on_rows) == 4
        if on_before[0] is not None:
            assert on_after[0] - on_before[0] == 1
            assert off_after[0] - off_before[0] == 0
            assert on_after[1] - on_before[1] >= 4
            assert off_after[1] - off_before[1] == 0
            assert (off_after[2] - off_before[2]) - (on_after[2] - on_before[2]) == 4
            assert (off_after[3] - off_before[3]) - (on_after[3] - on_before[3]) == 4
        if on_visits is not None:
            assert off_visits == on_visits
        print(
            "T6_CARRY "
            + json.dumps(
                {
                    "rows_identical": True,
                    "carry_on_visits": on_visits,
                    "carry_off_visits": off_visits,
                    "carry_on_hits": None if on_before[1] is None else on_after[1] - on_before[1],
                    "carry_off_hits": None if off_before[1] is None else off_after[1] - off_before[1],
                    "carry_off_extra_misses": None
                    if on_before[2] is None
                    else (off_after[2] - off_before[2]) - (on_after[2] - on_before[2]),
                },
                sort_keys=True,
            )
        )
    finally:
        db.close()


def _fallback_route_db(rows=16, deleted=15):
    """Churned fixture at the pinned minimum budget: the retired shape is declined.

    With retired entries present, the D3 load cannot be admitted inside the
    zero reader allowance that the 711,832-byte budget leaves, so the whole
    query keeps HEAD's three-table fallback reader. The ring adjacency makes the
    deleted row's retired entry reachable (and therefore scored) within a small
    effective L, which the retirement-tamper pins below rely on.
    """
    db = open_db()
    try:
        create(
            db,
            dim=2,
            options=(
                "degree=4,build_list_size=4,search_list_size=4,max_visits=4096"
                + ",cache_bytes="
                + str(_SMALL_BUDGET_MIN_CACHE_BYTES)
            ),
        )
        for i in range(rows):
            insert(db, i, [i, 0])
        inject_csr(db, adjacency=[[(i + 1) % rows] for i in range(rows)] + [[0]])
        db.execute("DELETE FROM v WHERE rowid=?", (deleted,))
        return db
    except BaseException:
        db.close()
        raise


def _fallback_delta(db, action):
    """Runs one query and, on Debug artifacts, asserts the D3 fallback route.

    Release artifacts export no instrumentation, so the route counter is
    asserted only where it is verifiably live; the caller's result assertions
    stay unconditional.
    """
    before = _debug_counter("vectorlite_test_retired_fallbacks")
    rows = action()
    after = _debug_counter("vectorlite_test_retired_fallbacks")
    if before is not None and after is not None:
        assert after - before == 1, (before, after)
    return rows


def _capture(action):
    """Runs action, returning ("ok", value) or ("err", sqlite3 error)."""
    try:
        return "ok", action()
    except sqlite3.DatabaseError as caught:
        return "err", caught


def _canonical_topk(db, vector, k):
    """Brute-force top-k over canonical live rows: (rowid, vector, f32 distance)."""
    rows = db.execute(
        "SELECT c.public_rowid,e.vector FROM v_diskann_current c "
        "JOIN v_diskann_entries e ON e.entry_id=c.entry_id"
    ).fetchall()
    query_vector = np.asarray(vector, dtype=np.float64)
    scored = []
    for rowid, payload in rows:
        stored = np.frombuffer(payload, dtype="<f4").astype(np.float64)
        distance = float(np.sum((stored - query_vector) ** 2))
        scored.append((distance, rowid, payload))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [
        (rowid, payload, struct.pack("<f", float(np.float32(distance))))
        for distance, rowid, payload in scored[:k]
    ]


def test_three_phase_supported_churn_matches_canonical_truth():
    """D10 row 1: supported churn in all three phases matches canonical truth.

    Pre-import churn is applied before the generation is installed, so the
    trusted fixture writes every snapshot version and flags the ineligible ones
    exactly as the import path does. Mid-import churn runs against a staged
    generation (retire() maintains the candidate), and post-publish churn runs
    against the published generation. Every phase must answer the same ordered
    rowids, projection bytes and distance bits as a brute-force scan of the
    canonical live rows, and the D3 retired-count invariant must hold.
    """
    options = _CARRY_FIXTURE_OPTIONS
    vector = [5.5, 0]
    k = 6
    l = 8
    baseline = None
    for phase in ("pre-import", "mid-import", "post-publish"):
        db = open_db()
        try:
            create(db, dim=2, options=options)
            for i in range(16):
                insert(db, i, [i, 0])
            if phase == "pre-import":
                db.execute("DELETE FROM v WHERE rowid=7")
                db.execute("UPDATE v SET embedding=? WHERE rowid=3", (blob([300, 0]),))
                all_entries = db.execute(
                    "SELECT entry_id,public_rowid,vector FROM v_diskann_entries ORDER BY entry_id"
                ).fetchall()
                n = len(all_entries)
                complete = [[j for j in range(n + 1) if j != i] for i in range(n + 1)]
                inject_csr(db, entries=all_entries, adjacency=complete)
            elif phase == "mid-import":
                inject_csr(db, published=False)
                db.execute("DELETE FROM v WHERE rowid=7")
                db.execute("UPDATE v SET embedding=? WHERE rowid=3", (blob([300, 0]),))
                db.execute("UPDATE v_diskann_generations SET state=1 WHERE generation_id=1")
                db.execute(
                    "UPDATE v_diskann_meta SET published_generation=1,"
                    "candidate_generation=NULL,active_job=NULL"
                )
            else:
                inject_csr(db)
                db.execute("DELETE FROM v WHERE rowid=7")
                db.execute("UPDATE v SET embedding=? WHERE rowid=3", (blob([300, 0]),))
            assert_retired_invariant(db)
            expected = _canonical_topk(db, vector, k)
            rows = query(db, vector=vector, k=k, l=l, project=True)
            assert _rows_signature(rows, True) == expected, phase
            if baseline is None:
                baseline = expected
            assert expected == baseline, phase
            print(
                "T6_CHURN_PHASE "
                + json.dumps({"phase": phase, "rowids": [row[0] for row in expected]}, sort_keys=True)
            )
        finally:
            db.close()


@pytest.mark.parametrize("variant", ["delete", "repoint"])
def test_current_shadow_edit_divergence_is_graph_route_only(variant):
    """D10 row 3: a current-row shadow edit is observed by every route but graph.

    The published generation keeps serving its own state on the traversal route
    (Format 5 identity reads never join current), while the exact two-table route
    and the three-table fallback still observe the edit through their current
    joins. Accepted-row projection is itself an exact two-table read (every
    public knn query references the embedding column, v9 C3 note), so a graph
    route that accepts the affected row fails closed at projection instead of
    serving it. Each route's documented outcome is pinned below.
    """
    db = _complete_fixture()
    try:
        baseline = query(db, vector=[0, 0], k=1, l=2, project=False)
        assert [row[0] for row in baseline] == [0]
        entry = db.execute("SELECT entry_id FROM v_diskann_current WHERE public_rowid=0").fetchone()[0]
        db.execute("DELETE FROM v_diskann_current WHERE public_rowid=0")
        if variant == "repoint":
            db.execute("INSERT INTO v_diskann_current VALUES(99,?)", (entry,))
        outcome, value = _capture(lambda: query(db, vector=[0, 0], k=1, l=2, project=False))
        assert outcome == "err", (variant, value)
        assert value.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, (variant, value)
        assert "projection" in str(value).lower(), (variant, value)
        assert query(db, vector=[0, 0], k=1, l=2, project=False, filter_sql="AND rowid=0") == []
        small_in = query(
            db,
            vector=[0, 0],
            k=2,
            l=4,
            project=False,
            filter_sql="AND rowid IN (0,1,2)",
        )
        assert small_in and 0 not in {row[0] for row in small_in}, (variant, small_in)
        unfiltered = _capture(lambda: query(db, vector=[0, 0], k=4, l=100, project=False))
        assert unfiltered[0] == "err", (variant, unfiltered)
        assert unfiltered[1].sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT
        # Both are the exact route's current-join guards: the delete variant
        # trips the population count, the repoint variant the rowid equality.
        assert (
            "count" in str(unfiltered[1]).lower()
            or "identity" in str(unfiltered[1]).lower()
        ), (variant, unfiltered)
        print(
            "T6_CURRENT_EDIT "
            + json.dumps(
                {
                    "variant": variant,
                    "graph_route": "accepted published row then fail-closed at projection",
                    "exact_point": "observed deleted/repointed current row",
                    "small_in": "observed deleted/repointed current row",
                    "unfiltered_exact": "population-equality check (CORRUPT)",
                    "fallback": "observed current join",
                },
                sort_keys=True,
            )
        )
    finally:
        db.close()
    fallback = _fallback_route_db()
    try:
        fallback.execute("DELETE FROM v_diskann_current WHERE public_rowid=0")
        rows = _fallback_delta(
            fallback, lambda: query(fallback, vector=[0, 0], k=2, l=4, project=False)
        )
        assert rows and 0 not in {row[0] for row in rows}, rows
    finally:
        fallback.close()


@pytest.mark.parametrize(
    "tamper",
    ["retired-extra", "retired-missing", "eligible-plus", "eligible-minus", "total-plus"],
)
def test_retired_and_counter_tamper_fires_operation_start_invariant(tamper):
    """D10 row 5: count-changing retired/counter tamper is SQLITE_CORRUPT at start.

    The graph route's operation-start invariant
    COUNT(generation_retired) == total_count - eligible_count observes every
    count-changing tamper with either side. The query fails closed before any
    row is served, and the legacy exact route keeps its own documented
    behaviour (it never loads the set, D3).
    """
    db = _complete_fixture()
    try:
        db.execute("DELETE FROM v WHERE rowid=15")
        baseline = query(db, vector=[15, 0], k=2, l=4, project=False)
        assert [row[0] for row in baseline] == [14, 13]
        generation = db.execute("SELECT generation_id FROM v_diskann_generations").fetchone()[0]
        retired_entry = db.execute(
            "SELECT entry_id FROM v_diskann_entries WHERE public_rowid=15 AND retired_revision IS NOT NULL"
        ).fetchone()[0]
        db.execute("PRAGMA ignore_check_constraints=ON")
        if tamper == "retired-extra":
            db.execute("INSERT INTO v_diskann_generation_retired VALUES(?,?)", (generation, 9999))
        elif tamper == "retired-missing":
            db.execute(
                "DELETE FROM v_diskann_generation_retired WHERE generation_id=? AND entry_id=?",
                (generation, retired_entry),
            )
        elif tamper == "eligible-plus":
            db.execute("UPDATE v_diskann_generations SET eligible_count=eligible_count+1")
        elif tamper == "eligible-minus":
            db.execute("UPDATE v_diskann_generations SET eligible_count=eligible_count-1")
        else:
            db.execute("UPDATE v_diskann_generations SET total_count=total_count+1")
        outcome, value = _capture(lambda: query(db, vector=[15, 0], k=2, l=4, project=False))
        assert outcome == "err", (tamper, value)
        assert value.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, (tamper, value)
        print(
            "T6_TAMPER "
            + json.dumps({"tamper": tamper, "error": str(value)}, sort_keys=True)
        )
    finally:
        db.close()


def test_balanced_retired_substitution_is_the_enumerated_undetected_limitation():
    """D10 row 5 limitation: a balanced retired-id substitution keeps both sides.

    Swapping one retired id for another preserves COUNT and therefore the D3
    invariant; the count deliberately cannot see it. The pinned outcome is that
    the invariant passes and the retired row is treated as live: eligibility
    follows the substituted set, not the canonical entries. This is an accepted,
    enumerated limitation of the count-only invariant.
    """
    db = _complete_fixture()
    try:
        db.execute("DELETE FROM v WHERE rowid=15")
        baseline = query(db, vector=[15, 0], k=2, l=4, project=False)
        assert [row[0] for row in baseline] == [14, 13]
        generation = db.execute("SELECT generation_id FROM v_diskann_generations").fetchone()[0]
        retired_entry = db.execute(
            "SELECT entry_id FROM v_diskann_entries WHERE public_rowid=15 AND retired_revision IS NOT NULL"
        ).fetchone()[0]
        db.execute(
            "UPDATE v_diskann_generation_retired SET entry_id=9999 WHERE generation_id=? AND entry_id=?",
            (generation, retired_entry),
        )
        assert_retired_invariant(db, generation)
        # The count passes, so the retired node is no longer excluded: the graph
        # route accepts it. Every public knn query also performs its accepted-row
        # projection (an exact two-table read), which still binds the node's
        # current entry and therefore fails closed instead of serving stale bytes.
        outcome, value = _capture(lambda: query(db, vector=[15, 0], k=2, l=4, project=False))
        assert outcome == "err", value
        assert value.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, value
        # Lean fast route: the substituted set member's parent entry is no longer
        # booked, so the retirement-consistency rule fires at scoring, before the
        # accepted-row projection the pre-lean pin expected.
        assert "bookkeeping" in str(value).lower() or "retirement" in str(value).lower(), value
        print(
            "T6_BALANCED_SUBSTITUTION "
            + json.dumps(
                {
                    "baseline_rowids": [14, 13],
                    "invariant_passes": True,
                    "outcome": str(value),
                },
                sort_keys=True,
            )
        )
    finally:
        db.close()


def _substitute_generated_column(db, column, column_type, expression):
    """Root-preserving generated-column substitution on a node-row column.

    RENAME COLUMN plus ADD COLUMN ... GENERATED ... VIRTUAL are metadata-only, so
    the nine-object shadow count and the aggregate rootpage sum are unchanged and
    the two-stamp detector cannot see the change (D10 row 6, enumerated).
    """
    db.execute(f"ALTER TABLE v_diskann_graph_nodes RENAME COLUMN {column} TO {column}_original")
    db.execute(
        f"ALTER TABLE v_diskann_graph_nodes ADD COLUMN {column} {column_type} "
        f"GENERATED ALWAYS AS ({expression}) VIRTUAL"
    )


@pytest.mark.parametrize(
    "column,project",
    [
        ("vector", False),
        ("vector", True),
        ("public_rowid", False),
        ("public_rowid", True),
    ],
)
def test_root_preserving_generated_column_substitution_serves_altered_values(column, project):
    """D10 row 6 enumerated divergence: root-preserving ALTER column substitution.

    A mid-operation RENAME COLUMN plus ADD COLUMN ... GENERATED ... VIRTUAL
    substitution preserves the shadow-object count and aggregate rootpage sum, so
    neither detector stamp moves and the in-flight statement is not failed closed;
    the fresh statement afterwards fails closed through collateral schema
    validation (SQLITE_CORRUPT).

    Lean v11 changed which node-row columns the graph route reads, and the pins
    below record that per column:
      * "vector": live vectors are canonical in diskann_entries, so the renamed
        live-node column is never read: the in-flight query returns exactly the
        baseline rows (the pre-lean probe served substituted live vectors). This
        is the D10 row 6 disclosure applied to the lean payload — the alteration
        is unobservable through served rows, which is why the vector payload moved.
      * "public_rowid": already authoritative on the node row (D1/D10), so the
        graph route resolves the altered labels (baseline rowids +1000). Public
        queries still fail closed at the accepted-row point read because no current
        row carries the substituted rowid, and the adapter always materializes the
        accepted rows (every knn_search(embedding, ...) references the vector
        column, C3), so the altered labels are never served: this is D10 row 6's
        fail-closed-through-collateral-checks outcome for the probed column.
    """
    db = _complete_fixture(rows=16)
    try:
        vector = [7.5, 0]
        baseline = query(db, vector=vector, k=4, l=4, project=False)
        assert [row[0] for row in baseline] == [7, 8, 6, 9]
        if column == "vector":
            column_type = "BLOB"
            expression = "CASE WHEN state=0 THEN zeroblob(8) ELSE vector_original END"
        else:
            column_type = "INTEGER"
            expression = "CASE WHEN state=0 THEN public_rowid_original+1000 ELSE NULL END"
        baseline = query(db, vector=vector, k=4, l=4, project=project)
        state = {"altered": False}

        def alter_for_in_flight(sql):
            if state["altered"]:
                return
            if _NODE_READER_MARKER in sql:
                state["altered"] = True
                _substitute_generated_column(db, column, column_type, expression)

        db.set_trace_callback(alter_for_in_flight)
        try:
            outcome, value = _capture(
                lambda: query(db, vector=vector, k=4, l=4, project=project)
            )
        finally:
            db.set_trace_callback(None)
        assert state["altered"], "the substitution must land mid-operation"
        if column == "vector":
            assert outcome == "ok", value
            assert _rows_signature(value, project) == _rows_signature(baseline, project), value
        else:
            # The substituted rowid has no current row, and the adapter always
            # materializes the accepted rows (every public knn_search references
            # the vector column, C3), so the mandatory accepted-row projection fails
            # closed instead of serving the altered labels or mismatched bytes.
            assert outcome == "err", value
            assert value.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, value
            assert "projection" in str(value).lower(), value
        fresh_outcome, fresh = _capture(lambda: query(db, vector=vector, k=4, l=4, project=False))
        assert fresh_outcome == "err", fresh
        assert fresh.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, fresh
        print(
            "T6_GENERATED_SUBSTITUTION "
            + json.dumps(
                {
                    "column": column,
                    "project": project,
                    "in_flight_outcome": outcome,
                    "served": None
                    if outcome != "ok"
                    else [
                        [int(row[0]), None if not project else bytes(row[1]).hex()]
                        for row in value
                    ],
                    "fresh_error": str(fresh),
                },
                sort_keys=True,
            )
        )
    finally:
        db.close()


def _swap_shadow_names(db):
    """Name-only entries/current swap preserving the aggregate rootpage sum."""
    db.execute("ALTER TABLE v_diskann_entries RENAME TO v_diskann_swap_tmp")
    db.execute("ALTER TABLE v_diskann_current RENAME TO v_diskann_entries")
    db.execute("ALTER TABLE v_diskann_swap_tmp RENAME TO v_diskann_current")


def _in_flight_swap(db, action, marker):
    """Runs action while committing the name swap at the first traced marker."""
    state = {"seen": 0, "swapped": False}

    def tracer(sql):
        if state["swapped"] or marker not in sql:
            return
        state["seen"] += 1
        if state["seen"] == 1:
            state["swapped"] = True
            _swap_shadow_names(db)

    db.set_trace_callback(tracer)
    try:
        outcome = _capture(action)
    finally:
        db.set_trace_callback(None)
    assert state["swapped"], "the name swap must land mid-operation"
    return outcome


@pytest.mark.parametrize("churn", ["delete", "update"])
def test_name_swap_pins_route_outcomes(churn):
    """D10 row 6: count-preserving entries/current swaps across route shapes.

    The swap keeps the shadow-object count and the aggregate rootpage sum, so the
    two stamps never move; the in-flight readers then group by name instead of
    identity. Lean v11 changed the traced outcomes here: every exact, fallback and
    projection shape now joins canonical entries for its vector payload (the plan's
    "column-sharing swaps compile ... stale/retired published rows" sentence
    described the inline-vector payload, where those plans had no entries join).
    The pinned outcomes are:
      * after the swap the entries name resolves to the old current table, which
        has no vector/created_revision column, so the exact-point, small-IN,
        unfiltered-exact and three-table fallback shapes all fail closed at the
        collateral reprepare with an ordinary "no such column: e...." SQL error and
        serve no rows. This is D10 row 6's fail-closed-through-collateral-checks
        class; no stale/retired published row is served.
      * a mixed-case shadow name still resolves to the same table (COLLATE NOCASE
        stamp and case-insensitive catalog validation), so a case-variant rename of
        an unread shadow table stays transparent.
    """
    target = 7 if churn == "delete" else 3
    db = _complete_fixture(rows=16)
    try:
        if churn == "delete":
            db.execute("DELETE FROM v WHERE rowid=?", (target,))
        else:
            db.execute("UPDATE v SET embedding=? WHERE rowid=?", (blob([30, 0]), target))
        assert query(
            db, vector=[0, 0], k=1, l=4, project=False, filter_sql="AND rowid=" + str(target)
        ) == []
        point_outcome = _in_flight_swap(
            db,
            lambda: db.execute(
                "SELECT rowid,distance FROM v WHERE knn_search(embedding,knn_param(?,?,?)) AND rowid=?",
                (blob([0, 0]), 1, json.dumps({"search_list_size": 4}), target),
            ).fetchall(),
            "c.public_rowid=?2",
        )
        assert point_outcome[0] == "err", point_outcome
        assert "no such column" in str(point_outcome[1]).lower(), (churn, point_outcome)
        print(
            "T6_SWAP "
            + json.dumps(
                {"churn": churn, "exact_point_outcome": str(point_outcome[1])},
                sort_keys=True,
            )
        )
    finally:
        db.close()

    db = _complete_fixture(rows=16)
    try:
        if churn == "delete":
            db.execute("DELETE FROM v WHERE rowid=?", (target,))
        else:
            db.execute("UPDATE v SET embedding=? WHERE rowid=?", (blob([30, 0]), target))
        in_outcome = _in_flight_swap(
            db,
            lambda: db.execute(
                "SELECT rowid,distance FROM v WHERE knn_search(embedding,knn_param(?,?,?)) AND rowid IN (1,?)",
                (blob([0, 0]), 2, json.dumps({"search_list_size": 4}), target),
            ).fetchall(),
            "c.public_rowid=?2",
        )
        assert in_outcome[0] == "err", in_outcome
        assert "no such column" in str(in_outcome[1]).lower(), (churn, in_outcome)
    finally:
        db.close()

    db = _complete_fixture(rows=16)
    try:
        if churn == "delete":
            db.execute("DELETE FROM v WHERE rowid=?", (target,))
        else:
            db.execute("UPDATE v SET embedding=? WHERE rowid=?", (blob([30, 0]), target))
        scan_outcome = _in_flight_swap(
            db,
            lambda: query(db, vector=[0, 0], k=4, l=100, project=False),
            "e.created_revision",
        )
        assert scan_outcome[0] == "err", scan_outcome
        assert "no such column" in str(scan_outcome[1]).lower(), scan_outcome
    finally:
        db.close()

    fallback = _fallback_route_db()
    try:
        fallback_outcome = _in_flight_swap(
            fallback,
            lambda: query(fallback, vector=[0, 0], k=2, l=4, project=False),
            "LEFT JOIN",
        )
        assert fallback_outcome[0] == "err", fallback_outcome
        assert "no such column" in str(fallback_outcome[1]).lower(), fallback_outcome
        print(
            "T6_SWAP_FALLBACK "
            + json.dumps({"error": str(fallback_outcome[1])}, sort_keys=True)
        )
    finally:
        fallback.close()

    db = _complete_fixture(rows=16)
    try:
        baseline = _rows_signature(query(db, vector=[0, 0], k=2, l=4, project=False), False)
        db.execute("ALTER TABLE v_diskann_jobs RENAME TO jobs_stage")
        db.execute("ALTER TABLE jobs_stage RENAME TO V_DISKANN_JOBS")
        assert _rows_signature(
            query(db, vector=[0, 0], k=2, l=4, project=False), False
        ) == baseline
        assert db.execute("SELECT count(*) FROM V_DISKANN_JOBS").fetchone()[0] >= 0
    finally:
        db.close()


def test_pre_capture_window_write_is_absorbed_into_the_snapshot():
    """D10 row 6 (optional probe): a pre-capture write is absorbed, not BUSY.

    The detector's guarantee boundary is the completed captures. A shadow write
    committed while the metadata anchor statement starts lands before both stamps
    are captured, so the operation is internally consistent on the post-write
    state instead of failing with SQLITE_BUSY. The pinned fixture's outcome is
    the post-write state: the graph route accepts the published row and the
    accepted-row projection (an exact two-table read) observes the shadow
    current-row deletion and fails closed. Other collateral outcomes, including
    ordinary SQL errors, remain possible in this window by contract.
    """
    db = _complete_fixture(rows=16)
    try:
        baseline = query(db, vector=[0, 0], k=2, l=4, project=False)
        assert [row[0] for row in baseline] == [0, 1]
        state = {"done": False, "mutation": None}

        def tracer(sql):
            if state["done"]:
                return
            if "SELECT format_version" in sql:
                state["done"] = True
                try:
                    db.execute("DELETE FROM v_diskann_current WHERE public_rowid=0")
                    state["mutation"] = "shadow-deleted"
                except sqlite3.Error as caught:
                    state["mutation"] = f"{type(caught).__name__}: {caught}"

        db.set_trace_callback(tracer)
        try:
            outcome, value = _capture(
                lambda: query(db, vector=[0, 0], k=2, l=4, project=False)
            )
        finally:
            db.set_trace_callback(None)
        assert state["mutation"] == "shadow-deleted", state
        assert outcome == "err", value
        assert value.sqlite_errorcode & 255 != sqlite3.SQLITE_BUSY, value
        assert value.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, value
        print(
            "T6_PRE_CAPTURE "
            + json.dumps({"mutation": state["mutation"], "outcome": str(value)}, sort_keys=True)
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# T6' (v12/v16 lean reconciliation): retirement-consistency pins on both routes,
# the fallback route's lean role rule, and the forced pair-cache-off differential.
# The fast route classifies a scored canonical entries row from the
# retire-maintained set plus the entries interval columns; the fallback route
# keeps HEAD's current-based classify asymmetry with the restored interval checks
# (plan D4/D10 row 4, C1(ii)). A top-k-only fixture cannot close these holes,
# because the tampered row is scored but not returned.
# ---------------------------------------------------------------------------


def _generation_row(db):
    """(generation_id, source_revision, total_count, eligible_count) of the published row."""
    return db.execute(
        "SELECT generation_id,source_revision,total_count,eligible_count "
        "FROM v_diskann_generations WHERE state=1"
    ).fetchone()


def _entry_of(db, rowid):
    """Canonical entry id current for one public rowid (None after a supported delete)."""
    return db.execute(
        "SELECT entry_id FROM v_diskann_current WHERE public_rowid=?", (rowid,)
    ).fetchone()[0]


def _retired_entry(db, rowid):
    """The booked-retired entry id of a supported rowid deletion."""
    row = db.execute(
        "SELECT entry_id FROM v_diskann_entries WHERE public_rowid=? AND retired_revision IS NOT NULL",
        (rowid,),
    ).fetchone()
    assert row is not None, "the supported delete must book a retired entry"
    return row[0]


def test_retired_shape_fallback_keeps_lean_role_rule():
    """D4/D10 row 2: the fallback route still enforces the lean node role rule.

    The three-table fallback projects the root-only inline payload and validates
    that a live node's vector column is SQL NULL (HEAD's original role rule, now
    consistent with the lean row). A non-NULL live-node payload is query-visible
    corruption on the fallback route, which is the only route that reads the node
    payload column for all roles.
    """
    db = _fallback_route_db()
    try:
        live_node = db.execute(
            "SELECT native_local_id FROM v_diskann_graph_nodes WHERE state=0 ORDER BY native_local_id"
        ).fetchone()[0]
        db.execute("PRAGMA ignore_check_constraints=ON")
        db.execute(
            "UPDATE v_diskann_graph_nodes SET vector=? WHERE native_local_id=?",
            (blob([0, 0]), live_node),
        )
        before = _debug_counter("vectorlite_test_retired_fallbacks")
        outcome, value = _capture(
            lambda: query(db, vector=[0, 0], k=2, l=4, project=False)
        )
        after = _debug_counter("vectorlite_test_retired_fallbacks")
        if before is not None and after is not None:
            assert after - before == 1, (before, after)
        assert outcome == "err", value
        assert value.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, value
        assert "role" in str(value).lower(), value
    finally:
        db.close()


@pytest.mark.parametrize("revision_kind", ["source_revision", "above_source", "beyond_metadata"])
def test_scored_unbooked_retirement_revision_fails_closed_on_both_routes(revision_kind):
    """D10 row 4 retire-direction: any unbooked retirement revision is corruption.

    A live, still-current entry is shadow-given a retirement revision without the
    retire() bookkeeping that would add it to the retire-maintained set. Both
    routes must fail closed on the scored row (it is scored through the root's
    adjacency and never returned):
      * the fast route's single rule is "a retirement revision present while the
        entry is not in the retire-maintained set", subsuming the interval values
        below and never serving the row;
      * the fallback route keeps HEAD's rule "a still-current entry carrying ANY
        retirement revision is SQLITE_CORRUPT".
    The probed values cover at-the-snapshot, above-source-but-within-metadata and
    beyond-the-anchored-metadata revisions (the latter also fails the interval
    bound).
    """
    db = _complete_fixture(rows=16)
    try:
        db.execute("DELETE FROM v WHERE rowid=15")
        generation, source_revision, _total, _eligible = _generation_row(db)
        metadata_revision = db.execute("SELECT revision FROM v_diskann_meta").fetchone()[0]
        assert source_revision >= 1
        value = {
            "source_revision": source_revision,
            "above_source": source_revision + 1,
            "beyond_metadata": metadata_revision + 1,
        }[revision_kind]
        if revision_kind == "above_source":
            assert value <= metadata_revision, (value, metadata_revision)
        entry = _entry_of(db, 1)
        db.execute("PRAGMA ignore_check_constraints=ON")
        db.execute(
            "UPDATE v_diskann_entries SET retired_revision=? WHERE entry_id=?",
            (value, entry),
        )
        # Graph route (eligible 15 > l 8); the complete root adjacency scores row 1.
        outcome, error = _capture(lambda: query(db, vector=[1, 0], k=1, l=8, project=False))
        if revision_kind == "source_revision":
            # Exact shapes restore HEAD's eligible-interval rule: a still-current
            # entry carrying any retirement revision is corruption (D10 row 4).
            exact_outcome, exact_error = _capture(
                lambda: query(db, vector=[1, 0], k=1, l=4, project=False, filter_sql="AND rowid=1")
            )
    finally:
        db.close()
    assert outcome == "err", error
    assert error.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, error
    if revision_kind == "source_revision":
        assert exact_outcome == "err", exact_error
        assert exact_error.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, exact_error
        assert "identity" in str(exact_error).lower(), exact_error

    fallback = _fallback_route_db(rows=16, deleted=1)
    try:
        generation, source_revision, _total, _eligible = _generation_row(fallback)
        metadata_revision = fallback.execute("SELECT revision FROM v_diskann_meta").fetchone()[0]
        value = {
            "source_revision": source_revision,
            "above_source": source_revision + 1,
            "beyond_metadata": metadata_revision + 1,
        }[revision_kind]
        if revision_kind == "above_source":
            assert value <= metadata_revision, (value, metadata_revision)
        entry = _entry_of(fallback, 2)
        fallback.execute("PRAGMA ignore_check_constraints=ON")
        fallback.execute(
            "UPDATE v_diskann_entries SET retired_revision=? WHERE entry_id=?",
            (value, entry),
        )
        outcome, error = _fallback_delta(
            fallback,
            lambda: _capture(lambda: query(fallback, vector=[1, 0], k=1, l=4, project=False)),
        )
    finally:
        fallback.close()
    assert outcome == "err", error
    assert error.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, error


@pytest.mark.parametrize("variant", ["unretire", "unretire_with_current"])
def test_scored_un_retire_shadow_edit_diverges_by_route(variant):
    """D10 row 4 un-retire direction: fast route fails closed, fallback is HEAD-permissive.

    A booked-retired entry (in the retire-maintained set, carrying its retirement
    revision, not current) is shadow-edited so its entries.retired_revision
    becomes NULL, with and without a restored current row:
      * fast route: the symmetric un-retire rule ("a set member whose revision is
        NULL is corruption") fires on the scored-but-not-returned row;
      * fallback route: HEAD's classify asymmetry is reproduced — the non-current
        entry is excluded as Deleted (variant A), while the restored current row
        makes it Live and it is served (variant B). The baseline's per-read join
        does the same, so this is a route-divergent-but-HEAD-equivalent outcome,
        never a wrong-row answer for the fast route.
    """
    # Fast route: complete published fixture, supported delete of rowid 15.
    db = _complete_fixture(rows=16)
    try:
        db.execute("DELETE FROM v WHERE rowid=15")
        retired = _retired_entry(db, 15)
        db.execute("UPDATE v_diskann_entries SET retired_revision=NULL WHERE entry_id=?", (retired,))
        if variant == "unretire_with_current":
            db.execute("INSERT INTO v_diskann_current VALUES(15,?)", (retired,))
        # Complete root adjacency scores entry 16; k=1 never returns it.
        outcome, error = _capture(lambda: query(db, vector=[1, 0], k=1, l=8, project=False))
    finally:
        db.close()
    assert outcome == "err", error
    assert error.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, error

    # Fallback route: ring fixture at the pinned minimum budget; rowid 1's retired
    # entry is a bridge scored as neighbour of node 0.
    fallback = _fallback_route_db(rows=16, deleted=1)
    try:
        retired = _retired_entry(fallback, 1)
        fallback.execute(
            "UPDATE v_diskann_entries SET retired_revision=NULL WHERE entry_id=?", (retired,)
        )
        if variant == "unretire_with_current":
            fallback.execute("INSERT INTO v_diskann_current VALUES(1,?)", (retired,))
        rows = _fallback_delta(
            fallback,
            lambda: query(fallback, vector=[1, 0], k=1, l=4, project=False),
        )
    finally:
        fallback.close()
    if variant == "unretire":
        assert rows, rows
        assert 1 not in {row[0] for row in rows}, rows
    else:
        # The restored current row makes the retired published row Live on the
        # fallback route, exactly as HEAD's per-read join would.
        assert [row[0] for row in rows] == [1], rows


def test_un_retire_with_current_pins_exact_shape_outcomes():
    """D10 row 4 / T6': the exact shapes reproduce HEAD per shape on variant B.

    On a complete published fixture with rowid 15 deleted through the supported
    API, the shadow un-retire (clear the booked revision and restore the current
    row) leaves the published node row in place. HEAD's exact plans then behave
    per shape: the point shape serves the stale row (interval checks pass), the
    unfiltered exact scan fails its population/eligible_count guard because
    current now yields 16 rows while eligible_count is 15, and the small-IN shape
    serves both the live and the restored row.
    """
    db = _complete_fixture(rows=16)
    try:
        db.execute("DELETE FROM v WHERE rowid=15")
        retired = _retired_entry(db, 15)
        db.execute("UPDATE v_diskann_entries SET retired_revision=NULL WHERE entry_id=?", (retired,))
        db.execute("INSERT INTO v_diskann_current VALUES(15,?)", (retired,))
        # ExactPoint: serves the stale/retired published row with its payload.
        point = query(db, vector=[0, 0], k=1, l=4, filter_sql="AND rowid=15")
        assert [row[0] for row in point] == [15], point
        assert point[0][1] == blob([15, 0]), point
        assert point[0][2] == pytest.approx(225.0), point
        # Small IN: both the live row 1 and the restored row 15 are served.
        small_in = query(db, vector=[0, 0], k=3, l=4, filter_sql="AND rowid IN (1,15)")
        assert {row[0] for row in small_in} == {1, 15}, small_in
        # ExactUnfiltered: 16 current rows against eligible_count 15 is corruption.
        outcome, error = _capture(lambda: query(db, vector=[0, 0], k=16, l=100, project=False))
        assert outcome == "err", error
        assert error.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, error
        assert "count" in str(error).lower(), error
        print(
            "T6_UNRETIRE_EXACT "
            + json.dumps(
                {
                    "exact_point": [int(row[0]) for row in point],
                    "small_in": sorted(int(row[0]) for row in small_in),
                    "unfiltered_error": str(error),
                },
                sort_keys=True,
            )
        )
    finally:
        db.close()


@pytest.mark.parametrize("interval", ["at_creation", "beyond_metadata"])
def test_malformed_retirement_intervals_on_set_members_fail_closed(interval):
    """D10 row 4: interval ordering is validated on both routes for set members.

    A correctly booked retired (non-current) entry is shadow-given an invalid
    interval: retired_revision == created_revision (retirement must follow
    creation) or retired_revision > the anchored metadata revision. The interval
    columns are validated before membership on the fast route and whenever a
    revision is present on the fallback route, so both routes fail closed on the
    scored-but-not-returned bridge.
    """
    def tamper(target, retired, metadata_revision, created):
        target.execute("PRAGMA ignore_check_constraints=ON")
        target.execute(
            "UPDATE v_diskann_entries SET retired_revision=? WHERE entry_id=?",
            (created if interval == "at_creation" else metadata_revision + 1, retired),
        )

    db = _complete_fixture(rows=16)
    try:
        db.execute("DELETE FROM v WHERE rowid=15")
        retired = _retired_entry(db, 15)
        created = db.execute(
            "SELECT created_revision FROM v_diskann_entries WHERE entry_id=?", (retired,)
        ).fetchone()[0]
        metadata_revision = db.execute("SELECT revision FROM v_diskann_meta").fetchone()[0]
        tamper(db, retired, metadata_revision, created)
        outcome, error = _capture(lambda: query(db, vector=[1, 0], k=1, l=8, project=False))
    finally:
        db.close()
    assert outcome == "err", error
    assert error.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, error

    fallback = _fallback_route_db(rows=16, deleted=1)
    try:
        retired = _retired_entry(fallback, 1)
        created = fallback.execute(
            "SELECT created_revision FROM v_diskann_entries WHERE entry_id=?", (retired,)
        ).fetchone()[0]
        metadata_revision = fallback.execute("SELECT revision FROM v_diskann_meta").fetchone()[0]
        tamper(fallback, retired, metadata_revision, created)
        outcome, error = _fallback_delta(
            fallback,
            lambda: _capture(lambda: query(fallback, vector=[1, 0], k=1, l=4, project=False)),
        )
    finally:
        fallback.close()
    assert outcome == "err", error
    assert error.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, error


def test_booked_retirement_bridge_stays_navigable_and_never_returned():
    """D10 row 4 control: a correctly booked deleted bridge is navigable, never returned.

    The supported delete books the retire-maintained entry, sets the retirement
    revision and removes the current row; neighbors() must keep returning the
    retired bridge so traversal stays identical, while eligibility excludes it on
    both routes. The D3 count invariant is asserted alongside.
    """
    db = _complete_fixture(rows=16)
    try:
        db.execute("DELETE FROM v WHERE rowid=15")
        assert_retired_invariant(db)
        rows = query(db, vector=[15, 0], k=3, l=8, project=False)
        assert [row[0] for row in rows] == [14, 13, 12], rows
        print(
            "T6_BRIDGE_CONTROL "
            + json.dumps({"graph_rowids": [int(row[0]) for row in rows]}, sort_keys=True)
        )
    finally:
        db.close()
    fallback = _fallback_route_db(rows=16, deleted=15)
    try:
        assert_retired_invariant(fallback)
        rows = _fallback_delta(
            fallback,
            lambda: query(fallback, vector=[15, 0], k=1, l=4, project=False),
        )
        assert rows, rows
        assert 15 not in {row[0] for row in rows}, rows
    finally:
        fallback.close()


def test_retired_load_never_materializes_more_than_its_admitted_population():
    """D3/T6': a reentrant writer cannot grow the admitted retired set mid-load.

    The operation-start count admits exactly count*8 bytes for the owned id Vec, and
    the load enforces that population before one row is appended
    (push_retired_entry), so a reentrant writer that makes the load statement return
    an extra row fails closed instead of materializing bytes outside the admitted
    shape. A progress handler inserts (generation_id, max_entry_id + 1) into
    v_diskann_generation_retired while the load statement runs:
      * with a fine progress quantum the reentrant row is visible to the active
        cursor and the query fails closed with SQLITE_CORRUPT,
        "grew beyond its admitted population", serving no rows;
      * if SQLite hides the reentrant row from the active cursor (coarser quantum),
        the bound never sees an extra row and the query still fails closed through
        the documented D3 detector class (SQLITE_BUSY, "concurrent mutation"). This
        fallback is accepted because the structural per-row bound is unit-tested in
        Rust; either way the committed tamper is caught by the next
        operation-start count invariant.
    """
    db = _complete_fixture(rows=16)
    try:
        db.execute("DELETE FROM v WHERE rowid IN (14,15)")
        generation = db.execute(
            "SELECT generation_id FROM v_diskann_generations WHERE state=1"
        ).fetchone()[0]
        retired = db.execute(
            "SELECT entry_id FROM v_diskann_generation_retired WHERE generation_id=? ORDER BY entry_id",
            (generation,),
        ).fetchall()
        assert len(retired) >= 2, retired
        counters = (
            "vectorlite_test_retired_admissions",
            "vectorlite_test_retired_fallbacks",
        )
        before = _debug_counters(*counters)
        state = {"armed": False, "done": False, "inserted": None}

        def tracer(sql):
            if state["armed"] or state["done"]:
                return
            if (
                "generation_retired" in sql
                and "ORDER BY entry_id" in sql
                and "count(*)" not in sql
            ):
                state["armed"] = True

        def handler():
            if not state["armed"] or state["done"]:
                return 0
            state["done"] = True
            try:
                top = db.execute(
                    "SELECT max(entry_id) FROM v_diskann_generation_retired WHERE generation_id=?",
                    (generation,),
                ).fetchone()[0]
                db.execute(
                    "INSERT INTO v_diskann_generation_retired VALUES(?,?)",
                    (generation, top + 1),
                )
                state["inserted"] = "committed"
            except sqlite3.Error as error:
                state["inserted"] = f"{type(error).__name__}: {error}"
            return 0

        db.set_trace_callback(tracer)
        db.set_progress_handler(handler, 5)
        try:
            served, error = _drain_knn(db, 1, 4, project=False)
        finally:
            db.set_progress_handler(None, 0)
            db.set_trace_callback(None)
        after = _debug_counters(*counters)
        assert state["armed"], "the retired-set load statement must be traced"
        assert state["inserted"] == "committed", state["inserted"]
        assert served == [], served
        assert error is not None, "the grown retired set must fail the query closed"
        if error.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT:
            assert "grew beyond its admitted population" in str(error), error
            outcome = "admitted-population bound"
        else:
            assert error.sqlite_errorcode & 255 == sqlite3.SQLITE_BUSY, error
            assert "concurrent mutation" in str(error).lower(), error
            outcome = "d3-detector-fallback"
        if before[0] is not None:
            # The graph route admitted the retired set (the bound is reachable) and
            # never fell back to the three-table reader.
            assert after[0] - before[0] == 1, (before, after)
            assert after[1] - before[1] == 0, (before, after)
        # The reentrant row committed: the durable set now exceeds
        # total_count - eligible_count, so the next operation fails the D3 invariant.
        assert db.execute(
            "SELECT count(*) FROM v_diskann_generation_retired WHERE generation_id=?",
            (generation,),
        ).fetchone()[0] == len(retired) + 1
        served_next, next_error = _drain_knn(db, 1, 4, project=False)
        assert served_next == [], served_next
        assert next_error is not None, "the committed count tamper must fail closed"
        assert next_error.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, next_error
        assert "mismatch" in str(next_error).lower(), next_error
        print(
            "T6_RETIRED_LOAD_BOUND "
            + json.dumps(
                {
                    "outcome": outcome,
                    "admitted": None if before[0] is None else after[0] - before[0],
                    "retired_before": len(retired),
                    "retired_after": db.execute(
                        "SELECT count(*) FROM v_diskann_generation_retired WHERE generation_id=?",
                        (generation,),
                    ).fetchone()[0],
                    "next_error": str(next_error),
                },
                sort_keys=True,
            )
        )
    finally:
        db.close()


def test_forced_pair_cache_off_preserves_results_and_visit_charges():
    """D6/T6': the forced pair-cache-off hook changes no result or visit charge.

    The per-expansion native->entry pair cache is charged after the carry with a
    non-closing decline; when it is disabled (tight budget or the Debug hook) each
    scored candidate takes one NodeIdentity read at the same visit charge. The
    on/off differential must keep rowids, order, projection bytes and distance
    bits identical, keep owner-visit totals identical, and show exactly one
    disabled operation plus one extra miss/identity read per cache hit saved.
    """
    db = _complete_fixture()
    try:
        vector = [3, 0]
        counters = (
            "vectorlite_test_pair_cache_hits",
            "vectorlite_test_pair_cache_misses",
            "vectorlite_test_pair_cache_disabled_operations",
            "vectorlite_test_node_identity_executions",
            "vectorlite_test_scored_candidates",
        )
        on_before = _debug_counters(*counters)
        on_rows = query(db, vector=vector, k=4, l=8, project=True)
        on_after = _debug_counters(*counters)
        on_visits = _debug_counter("vectorlite_test_last_query_visits")
        # The Debug-only hook is absent from release artifacts; the result and
        # visit-parity assertions below hold either way.
        _debug_hook("vectorlite_test_force_pair_cache_off")
        off_before = _debug_counters(*counters)
        off_rows = query(db, vector=vector, k=4, l=8, project=True)
        off_after = _debug_counters(*counters)
        off_visits = _debug_counter("vectorlite_test_last_query_visits")
        assert _rows_signature(on_rows, True) == _rows_signature(off_rows, True)
        assert len(on_rows) == 4
        if on_before[0] is not None:
            on_hits = on_after[0] - on_before[0]
            on_misses = on_after[1] - on_before[1]
            off_hits = off_after[0] - off_before[0]
            off_misses = off_after[1] - off_before[1]
            on_ids = on_after[3] - on_before[3]
            off_ids = off_after[3] - off_before[3]
            assert on_hits > 0
            assert off_hits == 0
            assert (off_after[2] - off_before[2]) - (on_after[2] - on_before[2]) == 1
            assert off_misses - on_misses == on_hits
            assert off_ids - on_ids == on_hits
        if on_visits is not None:
            assert off_visits == on_visits
        print(
            "T6_PAIR_CACHE "
            + json.dumps(
                {
                    "rows_identical": True,
                    "pair_on_visits": on_visits,
                    "pair_off_visits": off_visits,
                    "pair_on_hits": None if on_before[0] is None else on_after[0] - on_before[0],
                    "pair_off_hits": None if off_before[0] is None else off_after[0] - off_before[0],
                    "pair_off_extra_misses": None
                    if on_before[1] is None
                    else (off_after[1] - off_before[1]) - (on_after[1] - on_before[1]),
                },
                sort_keys=True,
            )
        )
    finally:
        db.close()
