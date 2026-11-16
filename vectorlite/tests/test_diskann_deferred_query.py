"""Published Format4 query qualification against an explicit or deployed artifact.

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


def inject_csr(db, entries=None, adjacency=None, root=None, generation=1, published=True, schema="main", source_revision=None):
    """Trusted SQL fixture retaining native dense IDs and exact adjacency order."""
    prefix = f'"{schema}".v_diskann_'
    if entries is None:
        entries = db.execute(f'SELECT c.entry_id,e.public_rowid,e.vector FROM {prefix}current c JOIN {prefix}entries e ON e.entry_id=c.entry_id ORDER BY c.entry_id').fetchall()
    n = len(entries)
    if adjacency is None:
        adjacency = [[j for j in range(n + 1) if j != i] for i in range(n + 1)] if n <= 4 else [[i + 1] if i < n else [0] for i in range(n + 1)]
    if root is None and n:
        root = entries[0][2]
    revision = db.execute(f'SELECT revision FROM {prefix}meta').fetchone()[0] if source_revision is None else source_revision
    eligible = sum(db.execute(f'SELECT entry_id FROM {prefix}current WHERE public_rowid=?', (rowid,)).fetchone() == (entry,) for entry, rowid, _ in entries)
    # Opaque nonzero fixture hashes, not production digest validation evidence.
    digest = bytes(range(32))
    db.execute(f'INSERT INTO {prefix}generations(generation_id,job_id,state,source_revision,source_hash,config_hash,total_count,eligible_count,root_local_id) VALUES(?,?,?,?,?,?,?,?,?)', (generation, generation, 1 if published else 0, revision, digest, digest[::-1], n, eligible, n if n else None))
    for i, (entry, _, _) in enumerate(entries):
        db.execute(f'INSERT INTO {prefix}graph_nodes VALUES(?,?,?,0,NULL,?)', (generation, i, entry, struct.pack(f"<{len(adjacency[i])}Q", *adjacency[i])))
    if n:
        db.execute(f'INSERT INTO {prefix}graph_nodes VALUES(?,?,NULL,2,?,?)', (generation, n, root, struct.pack(f"<{len(adjacency[n])}Q", *adjacency[n])))
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


@pytest.mark.parametrize("damage", ["vector-size", "vector-text", "vector-nan", "entry-missing", "data-frozen-payload", "root-role", "root-vector", "root-count", "neighbor-dangling", "neighbor-self", "neighbor-duplicate", "neighbor-range"])
def test_corruption_rejected_not_empty_or_missing(db, damage):
    for i in range(16):
        insert(db, i, [i, 0])
    inject_csr(db)
    db.execute("PRAGMA ignore_check_constraints=ON")
    if damage == "vector-size":
        db.execute("UPDATE v_diskann_entries SET vector=zeroblob(1048576) WHERE entry_id=1")
    elif damage == "vector-text":
        db.execute("UPDATE v_diskann_entries SET vector='bad' WHERE entry_id=1")
    elif damage == "vector-nan":
        db.execute("UPDATE v_diskann_entries SET vector=? WHERE entry_id=1", (blob([np.nan, 0]),))
    elif damage == "entry-missing":
        db.execute("DELETE FROM v_diskann_entries WHERE entry_id=1")
    elif damage == "data-frozen-payload":
        db.execute("UPDATE v_diskann_graph_nodes SET frozen_vector=? WHERE native_local_id=0", (blob([1, 0]),))
    elif damage == "root-role":
        db.execute("UPDATE v_diskann_graph_nodes SET state=0 WHERE native_local_id=16")
    elif damage == "root-vector":
        db.execute("UPDATE v_diskann_graph_nodes SET frozen_vector=zeroblob(1048576) WHERE native_local_id=16")
    elif damage == "root-count":
        db.execute("UPDATE v_diskann_generations SET root_local_id=0")
    elif damage == "neighbor-dangling":
        db.execute("DELETE FROM v_diskann_graph_nodes WHERE native_local_id=0")
    else:
        neighbors = [16] if damage == "neighbor-self" else [0, 0] if damage == "neighbor-duplicate" else [1000]
        db.execute("UPDATE v_diskann_graph_nodes SET neighbors=? WHERE native_local_id=16", (struct.pack(f"<{len(neighbors)}Q", *neighbors),))
    with pytest.raises(sqlite3.DatabaseError) as error:
        query(db, k=2, l=4)
    assert error.value.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT
    # Error tears down the query; ordinary SQL still works.
    assert db.execute("SELECT 1").fetchone() == (1,)


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
    ["entry-missing", "neighbor-dangling", "entry-vector-nan", "entry-vector-size"],
)
def test_single_fault_corruption_is_corrupt_without_rows(damage):
    """C1 single-fault classes: injected corruption fails SQLITE_CORRUPT, no rows.

    Missing identity rows, dangling adjacency, and vector payload damage (NaN
    and wrong length already covered by the qualification damage cases) must
    stay a corruption class error and must never serve a row first.
    """
    db = open_db()
    create(db, options="degree=16,build_list_size=20,search_list_size=4,max_visits=4096")
    for i in range(16):
        insert(db, i, [i, 0])
    inject_csr(db)
    db.execute("PRAGMA ignore_check_constraints=ON")
    if damage == "entry-missing":
        db.execute("DELETE FROM v_diskann_entries WHERE entry_id=1")
    elif damage == "neighbor-dangling":
        db.execute("DELETE FROM v_diskann_graph_nodes WHERE native_local_id=0")
    elif damage == "entry-vector-nan":
        db.execute("UPDATE v_diskann_entries SET vector=? WHERE entry_id=1", (blob([np.nan, 0]),))
    else:
        db.execute("UPDATE v_diskann_entries SET vector=zeroblob(1048576) WHERE entry_id=1")

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


@pytest.mark.parametrize("damage", ["entry-missing", "neighbor-dangling", "entry-vector-nan"])
def test_coincident_corruption_and_visit_budget_records_error_without_rows(damage):
    """C1 coincident faults: error and no rows; which error fires is recorded.

    Under the approved relaxed contract a corruption that coincides with a
    tight max_visits may surface either SQLITE_CORRUPT or SQLITE_TOOBIG. The
    test records the observed class instead of requiring one, and requires
    fail-closed behaviour (an error with no served rows).
    """
    db = open_db()
    create(db, options="degree=2,build_list_size=4,search_list_size=4,max_visits=4")
    for i in range(16):
        insert(db, i, [i, 0])
    inject_csr(db)
    db.execute("PRAGMA ignore_check_constraints=ON")
    if damage == "entry-missing":
        db.execute("DELETE FROM v_diskann_entries WHERE entry_id=1")
    elif damage == "neighbor-dangling":
        db.execute("DELETE FROM v_diskann_graph_nodes WHERE native_local_id=0")
    else:
        db.execute("UPDATE v_diskann_entries SET vector=? WHERE entry_id=1", (blob([np.nan, 0]),))
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


def test_reentrant_mid_query_shadow_corruption_fails_closed_without_stale_rows(tmp_path):
    """Reentrancy/staleness: mid-query shadow corruption never serves stale rows.

    A progress handler on the querying connection commits a corruption of the
    published generation's entry-1 vector (NaN) while one k=10/L=100 KNN
    statement runs. Three deterministic variants are driven:
      * a second connection commits a direct shadow UPDATE;
      * a second connection performs an incremental sqlite3_blob_write and
        commits it;
      * the querying connection itself executes the shadow UPDATE from the
        progress callback (same-connection reentrant write; pysqlite permits
        this here and SQLite makes the autocommit write visible to later reads
        of the same statement).

    The in-flight statement must either fail closed or return exactly the
    pre-mutation rows (a second connection's commit cannot invalidate the
    querying connection's read snapshot), never a mixture or stale bytes. A
    fresh query after the commit must observe the corrupted row and fail
    closed, proving no reader or payload cache survives one query.

    Limitation: a mutation scheduled exactly between two extension reads of
    one in-flight statement cannot be timed deterministically from Python; the
    progress handler fires at VM instruction boundaries, not at the
    extension's read sites. The same-connection variant is the strongest
    deterministic driver available (its autocommit write is immediately
    visible to the same connection), and the second-connection variants cover
    the committed-write/staleness half of the contract.
    """
    sql = 'SELECT rowid,embedding,distance FROM v WHERE knn_search(embedding,knn_param(?,?,?))'

    def setup(name):
        path = tmp_path / name
        db = open_db(path)
        db.execute("PRAGMA journal_mode=WAL")
        create(db, dim=2, options="degree=128,build_list_size=130,search_list_size=4,max_visits=4096")
        for i in range(128):
            insert(db, i, [i, 0])
        inject_csr(db, adjacency=[[j for j in range(128) if j != i] for i in range(128)] + [list(range(128))])
        return db

    def drain(db):
        served = []
        error = None
        try:
            cursor = db.execute(sql, (blob([0, 0]), 10, json.dumps({"search_list_size": 100})))
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

    for scenario in ("second-connection-update", "second-connection-blob", "same-connection-update"):
        db = setup(f"reentrant-{scenario}.db")
        mutator = None
        blob_handle = None
        progress = {"fired": 0, "done": False, "mutation": None}
        try:
            base = query(db, vector=[0, 0], k=10, l=100, project=True)
            assert base
            if scenario != "same-connection-update":
                mutator = open_db(tmp_path / f"reentrant-{scenario}.db")
            target_entry = db.execute("SELECT entry_id FROM v_diskann_current WHERE public_rowid=1").fetchone()[0]
            if scenario == "second-connection-blob":
                blob_handle = mutator.blobopen("v_diskann_entries", "vector", target_entry)

            def handler():
                progress["fired"] += 1
                if progress["done"] or progress["fired"] < 3:
                    return 0
                progress["done"] = True
                try:
                    if scenario == "same-connection-update":
                        db.execute(
                            "UPDATE v_diskann_entries SET vector=? WHERE entry_id=?",
                            (blob([np.nan, 0]), target_entry),
                        )
                    elif scenario == "second-connection-blob":
                        blob_handle.write(blob([np.nan, 0]))
                        blob_handle.close()
                    else:
                        mutator.execute(
                            "UPDATE v_diskann_entries SET vector=? WHERE entry_id=?",
                            (blob([np.nan, 0]), target_entry),
                        )
                    progress["mutation"] = "committed"
                except sqlite3.Error as caught:
                    progress["mutation"] = f"{type(caught).__name__}: {caught}"
                return 0

            db.set_progress_handler(handler, 50)
            try:
                served, error = drain(db)
            finally:
                db.set_progress_handler(None, 0)
            assert progress["fired"] > 0, scenario
            assert progress["mutation"] == "committed", (scenario, progress, error)
            assert db.execute(
                "SELECT vector FROM v_diskann_entries WHERE entry_id=?", (target_entry,)
            ).fetchone() == (blob([np.nan, 0]),)
            if error is None:
                assert [(int(row[0]), struct.pack("<f", row[2]), bytes(row[1])) for row in served] == [
                    (int(row[0]), struct.pack("<f", row[2]), bytes(row[1])) for row in base
                ], scenario
            else:
                assert served == [], (scenario, served)
                assert error.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, (scenario, error)
            fresh_served, fresh_error = drain(db)
            assert fresh_error is not None, scenario
            assert fresh_error.sqlite_errorcode & 255 == sqlite3.SQLITE_CORRUPT, (scenario, fresh_error)
            assert fresh_served == [], (scenario, fresh_served)
            print(
                "T4_REENTRANT "
                + json.dumps(
                    {
                        "scenario": scenario,
                        "progress_callbacks": progress["fired"],
                        "mutation": progress["mutation"],
                        "in_flight_outcome": "error" if error is not None else "rows",
                        "in_flight_error": None if error is None else str(error),
                        "fresh_outcome": "error",
                        "fresh_error": str(fresh_error),
                    }
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


def test_reentrant_mid_query_finite_mutation_never_serves_stale_rows(tmp_path):
    """Reentrancy/staleness: committed mid-query bytes are never stale afterwards.

    A second connection replaces entry 1's published vector with [50, 0] while
    a k=10/L=100 query runs. The in-flight statement either fails closed or
    returns only self-consistent rows (each rowid's embedding/distance pair is
    one committed version of that rowid). The next query must return exactly
    the post-mutation committed result (row 1 leaves the top 10), which is
    impossible if any reader or payload cache retained the old bytes.
    """
    path = tmp_path / "finite-stale.db"
    db = open_db(path)
    db.execute("PRAGMA journal_mode=WAL")
    create(db, dim=2, options="degree=128,build_list_size=130,search_list_size=4,max_visits=4096")
    for i in range(128):
        insert(db, i, [i, 0])
    inject_csr(db, adjacency=[[j for j in range(128) if j != i] for i in range(128)] + [list(range(128))])
    target_entry = db.execute("SELECT entry_id FROM v_diskann_current WHERE public_rowid=1").fetchone()[0]
    assert db.execute("SELECT public_rowid FROM v_diskann_current WHERE entry_id=?", (target_entry,)).fetchone() == (1,)
    mutator = open_db(path)

    def drain():
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

    progress = {"fired": 0, "done": False, "mutation": None}

    def handler():
        progress["fired"] += 1
        if progress["done"] or progress["fired"] < 3:
            return 0
        progress["done"] = True
        try:
            mutator.execute(
                "UPDATE v_diskann_entries SET vector=? WHERE entry_id=?", (blob([50, 0]), target_entry)
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
            served, error = drain()
        finally:
            db.set_progress_handler(None, 0)
        assert progress["fired"] > 0
        assert progress["mutation"] == "committed", (progress, error)
        if error is not None:
            assert served == []
            assert error.sqlite_errorcode & 255 in (sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_TOOBIG), error
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
            "T4_STALE "
            + json.dumps(
                {
                    "progress_callbacks": progress["fired"],
                    "mutation": progress["mutation"],
                    "in_flight_outcome": "error" if error is not None else "rows",
                    "in_flight_rowids": [int(row[0]) for row in served],
                    "fresh_rowids": [int(row[0]) for row in fresh],
                }
            )
        )
    finally:
        mutator.close()
        db.close()


# ---------------------------------------------------------------------------
# T4(f) contract additions: transparent reprepare under a mid-query schema
# change, terminal reprepare failure after a reader was used, and small-budget
# scalar-fallback equivalence. Successful-path results stay strict; failure
# paths must stay fail-closed under C1.
# ---------------------------------------------------------------------------


# Identity-only reader SQL (the three-table join without the vector payload)
# as prepared by the query adapter. SQLite's statement trace reports the exact
# prepared SQL, so observing this marker proves a reader statement started
# executing before the handler below commits its DDL.
_IDENTITY_READER_MARKER = "c.entry_id FROM "


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
    published Format 4 generation installed by this file's trusted inject_csr
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
            if _IDENTITY_READER_MARKER in sql:
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
    gated on the traced identity-reader SQL commits ONE schema change that
    invalidates the readers' SQL on the same connection: ALTER TABLE
    v_diskann_entries RENAME TO v_diskann_entries_renamed. Retained readers are
    expired and the next reader step must reprepare against a schema where
    v_diskann_entries no longer exists, failing terminally after a reader was
    used.

    Contract: the statement raises sqlite3.Error and serves no rows, the
    connection stays usable (simple SELECT works, the renamed table still holds
    every canonical row), and a fresh KNN query does not serve stale rows (it
    fails closed again). The source database is unaffected. Debug builds also
    assert and record prepare/finalize balance after the failure.

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
        entry_rows = source.execute("SELECT count(*) FROM v_diskann_entries").fetchone()
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
            if _IDENTITY_READER_MARKER in sql:
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
                        "ALTER TABLE v_diskann_entries RENAME TO v_diskann_entries_renamed"
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
            "SELECT count(*) FROM v_diskann_entries_renamed"
        ).fetchone() == entry_rows
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
            "SELECT count(*) FROM sqlite_master WHERE name='v_diskann_entries'"
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


def test_small_budget_scalar_fallback_equivalence():
    """T4(f) limits: the smallest admitted budget falls back to scalar readers.

    Two databases get identical seeded data and one published generation; the
    first uses the default cache_bytes (retained readers admitted), the second
    carries the smallest cache_bytes that still admits the same plain, filter
    and projected KNN queries. That boundary is derived at runtime by binary
    search over the admission predicate, not hard-coded (the illustrative
    cache_bytes=71000 cannot be admitted by this configuration: the core
    scratch floor plus the fixed 64 KiB transient owner buffer already exceed
    it; the observed boundary in this build is roughly 712 KiB and is recorded
    in the test output).

    At exactly the admission floor the documented envelope arithmetic leaves a
    zero reader allowance, so the small-budget queries must run HEAD's scalar
    path. Assertions: every query raises no error on either database and returns
    identical rowids, order, projection bytes and distance bits; the budget one
    byte lower is rejected (proving the boundary), and the binding shape is
    recorded. When the Debug reader-execution counter is verified live, the
    small-budget queries must show zero retained-reader executions while the
    default-budget control shows non-zero ones (the scalar fallback itself);
    release builds export no counter and only record the equality evidence.
    """
    cases = _SMALL_BUDGET_CASES
    low, high = 4096, 64 * 1024 * 1024
    assert _small_budget_outcomes(low, cases)[1] is not None
    assert _small_budget_outcomes(high, cases)[1] is None
    while low + 1 < high:
        middle = (low + high) // 2
        if _small_budget_outcomes(middle, cases)[1] is None:
            high = middle
        else:
            low = middle
    chosen = high

    binding = []
    for label, case in cases:
        try:
            probe = _small_budget_db(chosen - 1)
        except sqlite3.Error:
            binding.append(label)
            continue
        try:
            query(probe, **case)
        except sqlite3.Error:
            binding.append(label)
        finally:
            probe.close()
    assert binding, "smallest admitted budget rejected no query shape"

    default = _small_budget_db(None)
    small = _small_budget_db(chosen)
    records = []
    try:
        for label, case in cases:
            default_before = _debug_reader_executions()
            default_rows = query(default, **case)
            default_after = _debug_reader_executions()
            small_before = _debug_reader_executions()
            small_rows = query(small, **case)
            small_after = _debug_reader_executions()
            assert small_rows, label
            assert default_rows, label
            assert _rows_signature(default_rows, case["project"]) == _rows_signature(
                small_rows, case["project"]
            ), label
            default_delta = (
                None
                if default_before is None or default_after is None
                else default_after - default_before
            )
            small_delta = (
                None
                if small_before is None or small_after is None
                else small_after - small_before
            )
            if default_delta:
                assert small_delta == 0, (label, small_delta)
            records.append(
                {
                    "case": label,
                    "default_reader_executions": default_delta,
                    "small_reader_executions": small_delta,
                    "rows": len(small_rows),
                    "identical_rowids_order_distance_bits": True,
                }
            )
        print(
            "T4_SMALL_BUDGET "
            + json.dumps(
                {
                    "min_admitted_cache_bytes": chosen,
                    "binding_shapes": binding,
                    "default_cache_bytes": "default",
                    "scalar_fallback_verified": any(
                        record["default_reader_executions"] for record in records
                    ),
                    "cases": records,
                },
                sort_keys=True,
            )
        )
    finally:
        small.close()
        default.close()
