"""T4 public frontend tests against an explicit or deployed extension.

No successful candidate is synthesized by editing shadows; labelled corruption
injection tests intentionally bypass unsupported shadow editing protections.
"""
import json
import os
import re
import sqlite3
import struct
import sys
from pathlib import Path

import pytest

BUDGET = 128 * 1024 * 1024


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
    library = artifact()
    assert library.is_file(), library
    db = sqlite3.connect(f"file:{path}?mode=ro" if readonly else str(path),
                         isolation_level=None, uri=readonly)
    db.enable_load_extension(True)
    db.load_extension(str(library))
    return db


def qi(text):
    return '"' + text.replace('"', '""') + '"'


def create(db, name="v", schema="main", metric="l2"):
    db.execute(f"CREATE VIRTUAL TABLE {qi(schema)}.{qi(name)} USING vectorlite(embedding float32[2] {metric},diskann(degree=4,build_list_size=8))")


def status(db, schema="main", name="v"):
    return json.loads(db.execute("SELECT vectorlite_index_status(?,?)", (schema, name)).fetchone()[0])


def control(db, operation, opts=None, schema="main", name="v"):
    db.execute(f"INSERT INTO {qi(schema)}.{qi(name)}(operation,path) VALUES(?,?)",
               (operation, json.dumps({} if opts is None else opts)))


def current(db, schema="main", name="v"):
    return status(db, schema, name)["current_job"]


def jf(job):
    return {"job_id": job["job_id"], "fence": job["fence"]}


def prepared(db, population=6, schema="main", name="v", claim=True):
    for i in range(population):
        db.execute(f"INSERT INTO {qi(schema)}.{qi(name)}(rowid,embedding) VALUES(?,?)", (i, blob((i, 1))))
    control(db, "index_prepare", schema=schema, name=name)
    job = current(db, schema, name)
    for _ in range(100):
        if job["phase"] == 1:
            break
        control(db, "index_snapshot_step", {**jf(job), "max_records": 2}, schema, name)
        job = current(db, schema, name)
    assert job["phase"] == 1
    if claim:
        control(db, "index_claim_build", jf(job), schema, name)
        job = current(db, schema, name)
    return job


def build(db, job, schema="main", name="v", **changes):
    opts = {**jf(job), "memory_budget_bytes": BUDGET, **changes}
    return json.loads(db.execute("SELECT vectorlite_index_build(?,?,?)", (schema, name, json.dumps(opts))).fetchone()[0])


def published(db, job, candidate, schema="main", name="v"):
    for operation, phase in [("index_import_step", 2), ("index_validate_step", 3)]:
        for _ in range(100):
            j = current(db, schema, name)
            if j["phase"] != phase:
                break
            control(db, operation, {**jf(j), "candidate_token": candidate["candidate_token"], "max_records": 1}, schema, name)
        else:
            pytest.fail("bounded control failed to progress")
    assert current(db, schema, name)["phase"] == 4
    control(db, "index_publish", {**jf(job), "candidate_token": candidate["candidate_token"]}, schema, name)
    assert current(db, schema, name) is None


@pytest.fixture
def db():
    connection = connect()
    create(connection)
    yield connection
    connection.close()


@pytest.mark.parametrize("population", [0, 1, 6])
@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_actual_native_full_pipeline_and_ready_replay(population, metric):
    db = connect()
    try:
        create(db, metric=metric)
        job = prepared(db, population)
        assert status(db)["pending_count"] == population
        candidate = build(db, job)
        assert candidate == build(db, job)
        assert re.fullmatch(r"cand:[A-Za-z0-9._-]+", candidate["candidate_token"])
        assert candidate["population"] == population
        assert candidate["root"] == (population if population else None)
        assert all(re.fullmatch(r"[0-9a-f]{64}", candidate[k]) for k in ["input_hash", "config_hash", "candidate_hash"])
        published(db, job, candidate)
        s = status(db)
        assert s["format"] == 5
        assert (s["canonical_count"], s["eligible_count"], s["pending_count"], s["published_count"]) == (population, population, 0, population)
        rows = db.execute("SELECT rowid,embedding FROM v WHERE knn_search(embedding,knn_param(?,20))", (blob((0, 1)),)).fetchall()
        assert {r[0] for r in rows} == set(range(population))
        if population:
            db.execute("UPDATE v SET embedding=? WHERE rowid=0", (blob((5, 5)),))
            assert status(db)["pending_count"] == 1
            assert struct.unpack("<2f", db.execute("SELECT embedding FROM v WHERE rowid=0").fetchone()[0]) == pytest.approx((5, 5) if metric == "l2" else (2 ** -.5, 2 ** -.5))
            assert 0 not in {r[0] for r in db.execute("SELECT rowid FROM v WHERE knn_search(embedding,knn_param(?,20))", (blob((5, 5)),))}
    finally:
        db.close()


@pytest.mark.parametrize("schema,name", [("main", "self"), ("temp", "Blob"), ("main", "Node"), ("main", 'a"b')])
def test_exact_names_and_runtime_leases(schema, name):
    db = connect()
    try:
        create(db, name, schema)
        job = prepared(db, 1, schema, name)
        candidate = build(db, job, schema, name)
        published(db, job, candidate, schema, name)
        assert status(db, schema, name)["eligible_count"] == 1
    finally:
        db.close()


@pytest.mark.parametrize("schema,name", [(None, "v"), (1, "v"), (b"main", "v"), ("main", None), ("main", b"v"), ("", "v"), ("main", ""), ("no_such_schema", "v"), ("main", "missing"), ("main", "v; DROP TABLE v"), ("main", "v\x00"), ("x" * 1025, "v")])
def test_strict_target_argument_types_and_names(db, schema, name):
    with pytest.raises(sqlite3.Error):
        status(db, schema, name)
    assert status(db)["canonical_count"] == 0


@pytest.mark.parametrize("ddl,name", [("CREATE TABLE plain(x)", "plain"), ("CREATE VIEW view_v AS SELECT 'USING vectorlite'", "view_v"), ("CREATE VIRTUAL TABLE other USING fts5(x)", "other"), ("CREATE VIRTUAL TABLE h USING vectorlite(embedding float32[2],hnsw(max_elements=8))", "h")])
def test_other_table_view_module_and_hnsw_never_authorize(db, ddl, name):
    db.execute(ddl)
    with pytest.raises(sqlite3.Error):
        status(db, "main", name)
    with pytest.raises(sqlite3.Error):
        db.execute("SELECT vectorlite_index_build('main',?,?)", (name, json.dumps({"job_id": 1, "fence": 1, "memory_budget_bytes": BUDGET}))).fetchone()


@pytest.mark.parametrize("path", [None, 1, b"{}", "[]", "null", '{"unknown":1}', '{"workers":1}', '{"owner_nonce":"fake"}', '{"x":1,"x":2}', " " * 4097])
def test_prepare_strict_json_unknown_keys_and_bytes(db, path):
    with pytest.raises(sqlite3.Error):
        db.execute("INSERT INTO v(operation,path) VALUES('index_prepare',?)", (path,))
    assert current(db) is None


@pytest.mark.parametrize("options", [{}, {"job_id": True, "fence": 1}, {"job_id": 1.0, "fence": 1}, {"job_id": "1", "fence": 1}, {"job_id": 1, "fence": False}, {"job_id": 1, "fence": 0}, {"job_id": 9223372036854775808, "fence": 1}, {"job_id": 1, "fence": 1, "max_records": 0}, {"job_id": 1, "fence": 1, "max_records": 257}, {"job_id": 1, "fence": 1, "max_records": True}, {"job_id": 1, "fence": 1, "max_bytes": 0}, {"job_id": 1, "fence": 1, "max_bytes": 1 << 40}])
def test_snapshot_strict_typed_fence_limits(db, options):
    prepared(db, 1, claim=False)
    with pytest.raises(sqlite3.Error):
        control(db, "index_snapshot_step", options)


@pytest.mark.parametrize("patch", [{"workers": 2}, {"workers": True}, {"memory_budget_bytes": 0}, {"memory_budget_bytes": 1}, {"memory_budget_bytes": True}, {"memory_budget_bytes": 1.0}, {"extra": 1}, {"fence": 999}, {"job_id": 999}])
def test_build_strict_options_and_admission_no_sql_mutation(db, patch):
    job = prepared(db, 1)
    before = current(db)
    with pytest.raises(sqlite3.Error):
        build(db, job, **patch)
    assert current(db) == before
    assert build(db, job)["population"] == 1


@pytest.mark.parametrize("operation", ["index_import_step", "index_validate_step", "index_publish"])
@pytest.mark.parametrize("token", [None, 42, "cand:unknown", "not-a-token"])
def test_foreign_candidate_tokens_and_missing_ram_fail_closed(db, operation, token):
    job = prepared(db, 1)
    build(db, job)
    before = current(db)
    with pytest.raises(sqlite3.Error):
        control(db, operation, {**jf(job), "candidate_token": token})
    assert current(db) == before


@pytest.mark.parametrize("kind", ["begin", "savepoint", "constant_cursor", "table_cursor", "outer_from"])
def test_native_build_rejects_transaction_cursor_and_outer_from(db, kind):
    job = prepared(db, 6)
    cursor = None
    try:
        if kind == "begin":
            db.execute("BEGIN")
        elif kind == "savepoint":
            db.execute("SAVEPOINT x")
        elif kind == "constant_cursor":
            cursor = db.execute("SELECT 1 UNION ALL SELECT 2")
            cursor.fetchone()
        elif kind == "table_cursor":
            cursor = db.execute("SELECT embedding FROM v WHERE rowid IN (0,1,2)")
            cursor.fetchone()
        with pytest.raises(sqlite3.Error):
            if kind == "outer_from":
                db.execute("SELECT vectorlite_index_build('main','v',?) FROM v_diskann_current", (json.dumps({**jf(job), "memory_budget_bytes": BUDGET}),)).fetchall()
            else:
                build(db, job)
    finally:
        if cursor:
            cursor.close()
        if db.in_transaction:
            db.execute("ROLLBACK")
    assert build(db, job)["population"] == 6


SHADOW_SUFFIXES = (
    "meta",
    "entries",
    "current",
    "generations",
    "graph_nodes",
    "generation_retired",
    "jobs",
    "snapshot_map",
    "txn",
)


def test_status_reports_format5_and_registers_nine_shadow_names(db):
    """Format 5: the status JSON reports format 5 and all nine shadows exist.

    The status document is the public format signal (Format-5 readers must never
    depend on a guessed format), and the ninth shadow layout
    (v_diskann_generation_retired) is registered with the other eight so create,
    rename, drop and status enumerate the same fixed set.
    """
    assert status(db)["format"] == 5
    names = {row[0] for row in db.execute("SELECT name FROM main.sqlite_schema WHERE type='table'")}
    expected = {"v"} | {"v_diskann_" + suffix for suffix in SHADOW_SUFFIXES}
    assert expected <= names, expected - names
    job = prepared(db, 1)
    candidate = build(db, job)
    published(db, job, candidate)
    assert status(db)["format"] == 5
    names = {row[0] for row in db.execute("SELECT name FROM main.sqlite_schema WHERE type='table'")}
    assert expected <= names, expected - names


def test_status_readonly_bounded_and_no_recovery(tmp_path):
    path = tmp_path / "status.db"
    db = connect(path)
    create(db)
    prepared(db, 6)
    before = db.total_changes
    statements = []
    db.set_trace_callback(statements.append)
    for _ in range(3):
        assert status(db)["pending_count"] == 6
    db.set_trace_callback(None)
    assert db.total_changes == before
    population_reads = [sql for sql in statements if re.search(r"\bFROM\s+[^;]*diskann_(entries|snapshot_map|graph_nodes)", sql, re.IGNORECASE)]
    assert not population_reads, population_reads
    assert not any("SELECT * FROM" in sql for sql in statements)
    db.close()
    db = connect(path, readonly=True)
    try:
        assert status(db)["pending_count"] == 6
        with pytest.raises(sqlite3.Error):
            control(db, "index_prepare")
    finally:
        db.close()


def test_import_publish_rollback_replays_fresh_sql_progress(db):
    job = prepared(db, 6)
    candidate = build(db, job)
    opts = {**jf(job), "candidate_token": candidate["candidate_token"], "max_records": 2}
    db.execute("BEGIN")
    db.execute("SAVEPOINT chunk")
    control(db, "index_import_step", opts)
    assert current(db)["import_cursor"] == 2
    db.execute("ROLLBACK TO chunk")
    db.execute("RELEASE chunk")
    db.execute("COMMIT")
    assert current(db)["import_cursor"] == 0
    # Publication keeps candidate RAM until explicit cleanup, so rollback can retry.
    for op, phase in [("index_import_step", 2), ("index_validate_step", 3)]:
        while current(db)["phase"] == phase:
            control(db, op, opts)
    db.execute("BEGIN")
    control(db, "index_publish", {**jf(job), "candidate_token": candidate["candidate_token"]})
    assert status(db)["eligible_count"] == 6
    db.execute("ROLLBACK")
    assert current(db)["phase"] == 4
    control(db, "index_publish", {**jf(job), "candidate_token": candidate["candidate_token"]})
    assert status(db)["eligible_count"] == 6


def test_connection_close_lost_ram_requires_new_fenced_attempt(tmp_path):
    path = tmp_path / "lost.db"
    db = connect(path)
    create(db)
    old = prepared(db, 1)
    old_candidate = build(db, old)
    db.close()
    db = connect(path)
    try:
        assert current(db)["fence"] == old["fence"]
        with pytest.raises(sqlite3.Error):
            build(db, old)
        with pytest.raises(sqlite3.Error):
            control(db, "index_import_step", {**jf(old), "candidate_token": old_candidate["candidate_token"]})
        with pytest.raises(sqlite3.Error):
            control(db, "index_restart", jf(old))
        control(db, "index_restart", {**jf(old), "force": True})
        new = current(db)
        assert new["job_id"] > old["job_id"] and new["fence"] > old["fence"]
        while new["phase"] == 0:
            control(db, "index_snapshot_step", jf(new))
            new = current(db)
        control(db, "index_claim_build", jf(new))
        published(db, new, build(db, new))
    finally:
        db.close()


def test_same_connection_backup_copy_attach_and_detach_invalidation(tmp_path):
    path = tmp_path / "copy.db"
    db = connect()
    try:
        create(db)
        job = prepared(db, 1)
        candidate = build(db, job)
        copy = sqlite3.connect(path)
        db.backup(copy)
        copy.close()
        db.execute("ATTACH ? AS aux", (str(path),))
        assert status(db, "aux")["index_uuid"] == status(db)["index_uuid"]
        with pytest.raises(sqlite3.Error):
            control(db, "index_import_step", {**jf(job), "candidate_token": candidate["candidate_token"]}, "aux")
        db.execute("DETACH aux")
        db.execute("ATTACH ? AS aux", (str(path),))
        with pytest.raises(sqlite3.Error):
            control(db, "index_import_step", {**jf(job), "candidate_token": candidate["candidate_token"]}, "aux")
    finally:
        db.close()


def test_cancel_restart_rollback_ram_loss_is_explicit_and_recoverable(db):
    job = prepared(db, 1)
    candidate = build(db, job)
    db.execute("BEGIN")
    control(db, "index_cancel", jf(job))
    db.execute("ROLLBACK")
    assert current(db)["fence"] == job["fence"]
    with pytest.raises(sqlite3.Error, match="rebuild-needed"):
        control(db, "index_import_step", {**jf(job), "candidate_token": candidate["candidate_token"]})
    control(db, "index_restart", jf(job))
    assert current(db)["fence"] > job["fence"]


def test_rename_drop_recreate_invalidate_identity(db):
    job = prepared(db, 1)
    candidate = build(db, job)
    control(db, "index_cancel", jf(job))
    old_uuid = status(db)["index_uuid"]
    db.execute("ALTER TABLE v RENAME TO renamed")
    assert status(db, name="renamed")["index_uuid"] == old_uuid
    with pytest.raises(sqlite3.Error):
        control(db, "index_import_step", {**jf(job), "candidate_token": candidate["candidate_token"]}, name="renamed")
    db.execute("DROP TABLE renamed")
    create(db)
    assert status(db)["index_uuid"] != old_uuid
    with pytest.raises(sqlite3.Error):
        control(db, "index_import_step", {**jf(job), "candidate_token": candidate["candidate_token"]})


def test_consolidate_requests_same_prepare_protocol(db):
    db.execute("INSERT INTO v(operation) VALUES('consolidate')")
    assert current(db)["phase"] == 0
    assert db.execute("SELECT count(*) FROM v_diskann_graph_nodes").fetchone()[0] == 0


def test_canonical_whole_statement_rollback_and_integer_batch_rejection(db):
    with pytest.raises(sqlite3.Error):
        db.execute("INSERT INTO v(rowid,embedding) VALUES(1,?),(1,?)", (blob((1,2)), blob((3,4))))
    assert status(db)["canonical_count"] == 0
    with pytest.raises(sqlite3.Error):
        db.execute("INSERT INTO v(operation,embedding) VALUES('insert_batch',12345)")
    assert status(db)["canonical_count"] == 0


def test_directonly_nondeterministic_scalar_registration(db):
    rows = db.execute("PRAGMA function_list").fetchall()
    for name, argc in [("vectorlite_index_status", 2), ("vectorlite_index_build", 3)]:
        row = next(r for r in rows if r[0] == name)
        assert row[4] == argc and row[5] & 0x80000 and not row[5] & 0x800
    db.execute("CREATE VIEW bad AS SELECT vectorlite_index_status('main','v')")
    with pytest.raises(sqlite3.Error, match="unsafe use"):
        db.execute("SELECT * FROM bad").fetchall()


@pytest.mark.parametrize("operation", [123, b"index_prepare", "index_undefined", "INDEX_PREPARE", "index_build"])
def test_control_module_operation_types_and_names(db, operation):
    with pytest.raises(sqlite3.Error):
        db.execute("INSERT INTO v(operation,path) VALUES(?,?)", (operation, "{}"))
    assert current(db) is None


def test_build_requires_committed_same_session_claim(db):
    job = prepared(db, 1, claim=False)
    with pytest.raises(sqlite3.Error):
        build(db, job)
    with pytest.raises(sqlite3.Error):
        control(db, "index_claim_build", {**jf(job), "fence": job["fence"] + 1})
    assert current(db)["owner_nonce"] is None
    control(db, "index_claim_build", jf(job))
    assert re.fullmatch(r"[0-9a-f]{32}", current(db)["owner_nonce"])
    assert build(db, job)["population"] == 1


def test_multirow_control_whole_statement_rollback(db):
    before = status(db)
    with pytest.raises(sqlite3.Error):
        db.execute("INSERT INTO v(operation,path) VALUES('index_prepare','{}'),('index_prepare','{}')")
    assert status(db) == before
    assert db.execute("SELECT count(*) FROM v_diskann_jobs").fetchone()[0] == 0


@pytest.mark.parametrize("operation", ["index_import_step", "index_validate_step", "index_publish"])
def test_stale_fence_never_advances_candidate(db, operation):
    job = prepared(db, 1)
    candidate = build(db, job)
    before = current(db)
    with pytest.raises(sqlite3.Error):
        control(db, operation, {**jf(job), "fence": job["fence"] + 1, "candidate_token": candidate["candidate_token"]})
    assert current(db) == before


def test_source_digest_corruption_is_readonly_error_not_recovery(db):
    job = prepared(db, 1)
    # Unsupported direct shadow edit is explicit corruption injection.
    db.execute("UPDATE v_diskann_jobs SET source_hash=zeroblob(32)")
    before = db.total_changes
    with pytest.raises(sqlite3.Error):
        build(db, job)
    assert db.total_changes == before
    assert db.execute("SELECT import_cursor FROM v_diskann_jobs").fetchone()[0] == 0


def test_native_build_reentry_rejected_before_nested_constructor(db):
    job = prepared(db, 6)
    errors = []
    def recurse():
        db.set_progress_handler(None, 0)
        try:
            build(db, job)
        except sqlite3.Error as error:
            errors.append(str(error))
        return 0
    db.set_progress_handler(recurse, 50)
    try:
        candidate = build(db, job)
    finally:
        db.set_progress_handler(None, 0)
    assert len(errors) == 1 and "recursive DiskANN build" in errors[0]
    assert candidate["stats"]["coordinator_joined"]
    assert candidate["stats"]["native_alive_tasks_at_teardown"] == 0
    assert candidate == build(db, job)


def test_attach_target_detach_reattach_invalidates_old_source_binding(db):
    db.execute("ATTACH ':memory:' AS aux")
    create(db, schema="aux")
    job = prepared(db, 1, "aux")
    candidate = build(db, job, "aux")
    db.execute("DETACH aux")
    db.execute("ATTACH ':memory:' AS aux")
    create(db, schema="aux")
    new = prepared(db, 1, "aux")
    with pytest.raises(sqlite3.Error):
        control(db, "index_import_step", {**jf(new), "candidate_token": candidate["candidate_token"]}, "aux")
    published(db, new, build(db, new, "aux"), "aux")


@pytest.mark.parametrize("name,argc", [("vector_distance",3),("vector_from_json",1),("vector_to_json",1),("knn_search",2),("knn_param",-1),("vectorlite_info",0),("vectorlite_atomic_write",2),("vectorlite_index_status",2),("vectorlite_index_build",3)])
def test_loader_custom_function_collision_preserves_host_callback(name, argc):
    connection = sqlite3.connect(":memory:", isolation_level=None)
    calls = []
    def host(*args):
        calls.append(args)
        return "host-owned"
    connection.create_function(name, argc, host)
    connection.enable_load_extension(True)
    try:
        with pytest.raises(sqlite3.Error, match="already registered"):
            connection.load_extension(str(artifact()))
        assert calls == [], "preflight executed an untrusted host scalar"
        args = ",".join(["NULL"] * (2 if argc < 0 else argc))
        assert connection.execute(f"SELECT {name}({args})").fetchone() == ("host-owned",)
        assert len(calls) == 1
        assert connection.execute("SELECT count(*) FROM sqlite_schema").fetchone() == (0,)
    finally:
        connection.close()


def test_repeated_loader_rejection_preserves_hnsw_diskann_and_ready_candidate(db):
    db.execute("CREATE VIRTUAL TABLE h USING vectorlite(embedding float32[2],hnsw(max_elements=8))")
    db.execute("INSERT INTO h(rowid,embedding) VALUES(7,?)", (blob((1,2)),))
    job = prepared(db, 1)
    candidate = build(db, job)
    before = status(db)
    with pytest.raises(sqlite3.Error, match="already registered"):
        db.load_extension(str(artifact()))
    assert status(db) == before and build(db, job) == candidate
    assert db.execute("SELECT rowid FROM h WHERE knn_search(embedding,knn_param(?,1))", (blob((1,2)),)).fetchall() == [(7,)]
    published(db, job, candidate)


def test_build_required_keys_and_duplicate_args(db):
    job = prepared(db, 1)
    for text in ["{}", json.dumps(jf(job)), '{"job_id":1,"fence":1,"memory_budget_bytes":134217728,"fence":1}']:
        with pytest.raises(sqlite3.Error):
            db.execute("SELECT vectorlite_index_build('main','v',?)", (text,)).fetchone()
    assert build(db, job)["population"] == 1
