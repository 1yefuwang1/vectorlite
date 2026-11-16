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
