"""D7 Format-4 -> Format-5 migration surface against preserved artifacts.

Format-4 databases are created with the preserved baseline artifact; the
upgrade scalar and the republished Format-5 index are driven through public SQL
with the explicitly selected current artifact. Every test asserts that a
refused or aborted upgrade leaves the Format-4 file usable.
"""
import json
import os
import random
import sqlite3
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "build/benchmarks/knn-format5-20261123/baseline-artifact/vectorlite-format4-51b0aeb0.dylib"
DIMENSION = 4
# Format-5 inserts generation_retired; Format-4 has the other eight layouts.
FORMAT4_SUFFIXES = ("meta", "entries", "current", "generations", "graph_nodes", "jobs", "snapshot_map", "txn")
FORMAT5_SUFFIXES = FORMAT4_SUFFIXES[:5] + ("generation_retired",) + FORMAT4_SUFFIXES[5:]
CREATE = f'CREATE VIRTUAL TABLE "main"."v" USING vectorlite(embedding float32[{DIMENSION}] l2,diskann(degree=4,build_list_size=8))'
BUILD_OPTIONS = json.dumps({"memory_budget_bytes": 128 * 1024 * 1024, "workers": 1})


def artifact():
    """Explicit VECTORLITE_RUST_EXTENSION override, else the deployed extension."""
    override = os.environ.get("VECTORLITE_RUST_EXTENSION")
    if override:
        return Path(override).resolve()
    import vectorlite_py

    suffix = "dll" if sys.platform == "win32" else "dylib" if sys.platform == "darwin" else "so"
    return Path(vectorlite_py.vectorlite_path()).with_suffix(f".{suffix}")


def legacy_artifact():
    """The preserved Format-4 baseline artifact, or skip on platforms without it."""
    if not BASELINE.is_file():
        pytest.skip(f"preserved Format-4 baseline artifact is unavailable: {BASELINE}")
    return BASELINE


def blob(values):
    return struct.pack(f"<{len(values)}f", *values)


def connect(path, library=None, timeout=5.0):
    library = Path(library or artifact()).resolve()
    assert library.is_file(), f"build or install the extension first: {library}"
    db = sqlite3.connect(str(path), isolation_level=None, timeout=timeout)
    db.row_factory = sqlite3.Row
    db.enable_load_extension(True)
    db.load_extension(str(library))
    return db


def plain_connect(path, timeout=5.0):
    return sqlite3.connect(str(path), isolation_level=None, timeout=timeout)


def remove_database(path):
    for candidate in (path, f"{path}-wal", f"{path}-shm"):
        Path(candidate).unlink(missing_ok=True)


def create_format4(path, rows):
    remove_database(path)
    db = connect(path, legacy_artifact())
    db.execute(CREATE)
    for rowid, values in rows:
        db.execute('INSERT INTO v(rowid,embedding) VALUES(?,?)', (rowid, blob(values)))
    return db


def rows(count=24, seed=20261123):
    rng = random.Random(seed)
    return [(rowid, [rng.uniform(-1.0, 1.0) for _ in range(DIMENSION)]) for rowid in range(count)]


def queries(count=6, seed=7):
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(DIMENSION)] for _ in range(count)]


def catalog(db):
    return tuple(tuple(row) for row in db.execute('SELECT type,name,tbl_name,sql FROM main.sqlite_schema ORDER BY type,name'))


def meta_row(db):
    return dict(db.execute('SELECT * FROM v_diskann_meta').fetchone())


def shadow_snapshot(db, suffixes):
    return tuple(
        tuple(tuple(row) for row in db.execute(f'SELECT * FROM v_diskann_{suffix} ORDER BY 1,2'))
        for suffix in suffixes
    )


def table_counts(db, suffixes):
    return {suffix: db.execute(f'SELECT count(*) FROM v_diskann_{suffix}').fetchone()[0] for suffix in suffixes}


def upgrade(db, schema="main", table="v"):
    return json.loads(db.execute('SELECT vectorlite_index_upgrade(?,?)', (schema, table)).fetchone()[0])


def upgrade_error(db):
    with pytest.raises(sqlite3.Error) as error:
        db.execute('SELECT vectorlite_index_upgrade(?,?)', ("main", "v")).fetchone()
    return str(error.value)


def control(db, operation, **options):
    if "job_id" in options and "fence" not in options:
        options["fence"] = job(db, options["job_id"])["fence"]
    db.execute('INSERT INTO "main"."v"(operation,path) VALUES (?,?)', (operation, json.dumps(options)))


def job(db, job_id=None):
    if job_id is None:
        job_id = db.execute('SELECT active_job FROM v_diskann_meta').fetchone()[0]
    row = db.execute('SELECT * FROM v_diskann_jobs WHERE job_id=?', (job_id,)).fetchone()
    assert row is not None
    return dict(row)


def build_candidate(db, records=32):
    control(db, "index_prepare")
    current = job(db)
    for _ in range(1000):
        if current["phase"] == 1:
            break
        control(db, "index_snapshot_step", job_id=current["job_id"], max_records=records, max_bytes=65536)
        current = job(db, job_id=current["job_id"])
    else:
        pytest.fail("snapshot did not finish within the bounded history")
    control(db, "index_claim_build", job_id=current["job_id"])
    result = json.loads(
        db.execute(
            "SELECT vectorlite_index_build(?,?,?)",
            ("main", "v", json.dumps({"job_id": current["job_id"], "fence": current["fence"], **json.loads(BUILD_OPTIONS)})),
        ).fetchone()[0]
    )
    return current["job_id"], result


def advance_import(db, job_id, result, records=32, validate=True):
    steps = [(2, "index_import_step", "import_cursor")]
    if validate:
        steps.append((3, "index_validate_step", "validation_cursor"))
    for phase, operation, cursor in steps:
        for _ in range(1000):
            current = job(db, job_id=job_id)
            if current["phase"] != phase:
                break
            before = current[cursor]
            control(db, operation, job_id=job_id, candidate_token=result["candidate_token"], max_records=records, max_bytes=65536)
            after = job(db, job_id=job_id)
            assert 0 <= after[cursor] - before <= records
        else:
            pytest.fail(f"{operation} did not complete within the bounded history")


def publish(db, records=32):
    job_id, result = build_candidate(db, records)
    advance_import(db, job_id, result, records)
    assert job(db, job_id=job_id)["phase"] == 4
    control(db, "index_publish", job_id=job_id, candidate_token=result["candidate_token"])
    assert job(db, job_id=job_id)["phase"] == 5
    return job_id, result


def make_active_job(db, records=32):
    job_id, result = build_candidate(db, records)
    advance_import(db, job_id, result, records)
    current = job(db, job_id=job_id)
    assert current["phase"] == 4
    return current, result


def route_results(db, probes):
    results = []
    for query in probes:
        for k in (1, 5, 10):
            results.append(tuple(tuple(row) for row in db.execute(
                'SELECT rowid,embedding,distance FROM v WHERE knn_search(embedding,knn_param(?,?))',
                (blob(query), k),
            )))
            results.append(tuple(tuple(row) for row in db.execute(
                'SELECT rowid,vector_to_json(embedding),distance FROM v WHERE knn_search(embedding,knn_param(?,?)) AND rowid IN (?,?,?,?)',
                (blob(query), k, 0, 5, 11, 23),
            )))
            results.append(tuple(tuple(row) for row in db.execute(
                'SELECT rowid,distance FROM v WHERE knn_search(embedding,knn_param(?,?)) AND rowid=?',
                (blob(query), k, 4),
            )))
    return results


def test_new_build_rejects_format4_then_upgrade_preserves_and_refuses_second(tmp_path):
    path = tmp_path / "legacy.sqlite"
    data = rows(6)
    legacy = create_format4(path, data)
    before_meta = meta_row(legacy)
    before_entries = tuple(tuple(row) for row in legacy.execute('SELECT entry_id,public_rowid,created_revision,retired_revision,vector FROM v_diskann_entries ORDER BY entry_id'))
    before_current = tuple(tuple(row) for row in legacy.execute('SELECT public_rowid,entry_id FROM v_diskann_current ORDER BY public_rowid'))
    before_descriptor = json.loads(before_meta["descriptor"])
    assert before_meta["format_version"] == 4
    assert before_descriptor["format_version"] == 4
    legacy.close()

    db = connect(path)
    # Connect-time behaviour is unchanged and fail-closed: the Format-4 file
    # cannot xConnect, and the failure happens before any mutation.
    with pytest.raises(sqlite3.DatabaseError, match="format version is unsupported"):
        db.execute('SELECT embedding FROM v WHERE rowid=1').fetchall()
    assert meta_row(db)["format_version"] == 4

    payload = upgrade(db)
    assert payload["format"] == 5
    assert payload["upgraded"] is True
    assert payload["schema"] == "main" and payload["table"] == "v"
    assert payload["index_uuid"] == "idx:" + before_meta["instance_id"].hex()
    for key in ("revision", "next_entry_id", "next_job_id", "next_generation_id", "canonical_count", "retired_count", "cleanup_entry_id"):
        assert payload[key] == before_meta[key]
    assert payload["next_fence"] == before_meta["next_fence"] + 1
    assert payload["abandoned_published_generation"] is None
    assert payload["abandoned_candidate_generation"] is None
    assert payload["abandoned_active_job"] is None

    after_meta = meta_row(db)
    assert after_meta["format_version"] == 5
    assert after_meta["instance_id"] == before_meta["instance_id"]
    assert after_meta["descriptor"] != before_meta["descriptor"]
    assert json.loads(after_meta["descriptor"]) == {**before_descriptor, "format_version": 5}
    for key in ("revision", "next_entry_id", "next_job_id", "next_generation_id", "canonical_count", "retired_count", "cleanup_entry_id"):
        assert after_meta[key] == before_meta[key]
    assert after_meta["next_fence"] == before_meta["next_fence"] + 1
    assert after_meta["published_generation"] is None
    assert after_meta["candidate_generation"] is None
    assert after_meta["active_job"] is None
    # entries and current are untouched; derived tables are empty; the ninth
    # Format-5 layout exists.
    assert tuple(tuple(row) for row in db.execute('SELECT entry_id,public_rowid,created_revision,retired_revision,vector FROM v_diskann_entries ORDER BY entry_id')) == before_entries
    assert tuple(tuple(row) for row in db.execute('SELECT public_rowid,entry_id FROM v_diskann_current ORDER BY public_rowid')) == before_current
    assert table_counts(db, FORMAT5_SUFFIXES) == {
        "meta": 1, "entries": 6, "current": 6, "generations": 0, "graph_nodes": 0,
        "generation_retired": 0, "jobs": 0, "snapshot_map": 0, "txn": 2,
    }
    # The Format-5 store now connects and serves canonical reads.
    assert tuple(db.execute('SELECT embedding FROM v WHERE rowid=3').fetchone()) == (blob(data[3][1]),)
    assert db.execute('SELECT rowid FROM v WHERE knn_search(embedding,knn_param(?,?))', (blob(data[0][1]), 5)).fetchall() == []

    # A second invocation refuses at the format_version check with zero mutation.
    before_second_catalog = catalog(db)
    before_second_rows = shadow_snapshot(db, FORMAT5_SUFFIXES)
    before_second_meta = meta_row(db)
    message = upgrade_error(db)
    assert "Format-4" in message
    assert catalog(db) == before_second_catalog
    assert shadow_snapshot(db, FORMAT5_SUFFIXES) == before_second_rows
    assert meta_row(db) == before_second_meta
    db.close()

    # And a fresh connection refuses too, with the same zero-mutation evidence.
    db = connect(path)
    assert "Format-4" in upgrade_error(db)
    assert meta_row(db)["format_version"] == 5
    db.close()


@pytest.mark.parametrize(
    "label,statements,message",
    [
        ("descriptor-mismatch", ["UPDATE v_diskann_meta SET descriptor=x'7b7d'"], "descriptor does not match"),
        ("guard-row-missing", ["DELETE FROM v_diskann_txn WHERE singleton=2"], "atomic guard"),
        ("extra-index", ["CREATE INDEX probe_extra ON v_diskann_entries(public_rowid)"], "shadow index is not canonical"),
        ("trigger", ["CREATE TRIGGER probe_trigger AFTER INSERT ON v_diskann_current BEGIN SELECT 1; END"], "cannot have triggers"),
        (
            "dropped-check",
            [
                "ALTER TABLE v_diskann_entries RENAME TO replaced_entries",
                "CREATE TABLE v_diskann_entries (entry_id INTEGER PRIMARY KEY,public_rowid INTEGER NOT NULL,created_revision INTEGER NOT NULL,retired_revision INTEGER,vector BLOB NOT NULL)",
                "DROP TABLE replaced_entries",
            ],
            "definition is not canonical",
        ),
        (
            "meta-without-checks",
            [
                "ALTER TABLE v_diskann_meta RENAME TO replaced_meta",
                "CREATE TABLE v_diskann_meta (singleton INTEGER PRIMARY KEY,format_version INTEGER NOT NULL,instance_id BLOB NOT NULL,descriptor BLOB NOT NULL,revision INTEGER NOT NULL,next_entry_id INTEGER NOT NULL,next_job_id INTEGER NOT NULL,next_generation_id INTEGER NOT NULL,canonical_count INTEGER NOT NULL,retired_count INTEGER NOT NULL,published_generation INTEGER,candidate_generation INTEGER,active_job INTEGER,next_fence INTEGER NOT NULL,cleanup_entry_id INTEGER NOT NULL)",
                "INSERT INTO v_diskann_meta SELECT * FROM replaced_meta",
                "DROP TABLE replaced_meta",
            ],
            "definition is not canonical",
        ),
        (
            "format5-table-present",
            ["CREATE TABLE v_diskann_generation_retired (generation_id INTEGER NOT NULL,entry_id INTEGER NOT NULL,PRIMARY KEY(generation_id,entry_id)) WITHOUT ROWID"],
            "already contains a Format-5",
        ),
    ],
)
def test_format4_catalog_validation_refuses_tampered_database(tmp_path, label, statements, message):
    path = tmp_path / f"{label}.sqlite"
    create_format4(path, rows(6)).close()
    tamper = plain_connect(path)
    for statement in statements:
        tamper.execute(statement)
    tamper.close()

    db = connect(path)
    before_fence = db.execute('SELECT next_fence FROM v_diskann_meta').fetchone()[0]
    before_entries = db.execute('SELECT count(*) FROM v_diskann_entries').fetchone()[0]
    before_retired = db.execute("SELECT count(*) FROM main.sqlite_schema WHERE name='v_diskann_generation_retired'").fetchone()[0]
    refused = upgrade_error(db)
    assert message in refused, refused
    # The refused upgrade never starts its transaction.
    assert db.execute('SELECT format_version FROM v_diskann_meta').fetchone()[0] == 4
    assert db.execute('SELECT next_fence FROM v_diskann_meta').fetchone()[0] == before_fence
    assert db.execute("SELECT count(*) FROM main.sqlite_schema WHERE name='v_diskann_generation_retired'").fetchone()[0] == before_retired
    assert db.execute('SELECT count(*) FROM v_diskann_entries').fetchone()[0] == before_entries
    db.close()


def test_upgrade_then_republish_matches_fresh_format5(tmp_path):
    data = rows(24)
    probes = queries(6)
    legacy_path = tmp_path / "legacy.sqlite"
    fresh_path = tmp_path / "fresh.sqlite"

    legacy = create_format4(legacy_path, data)
    publish(legacy)
    before_meta = meta_row(legacy)
    assert before_meta["published_generation"] is not None
    legacy.close()

    db = connect(legacy_path)
    payload = upgrade(db)
    assert payload["abandoned_published_generation"] == before_meta["published_generation"]
    db.close()

    upgraded = connect(legacy_path)
    # The abandoned publication is gone until the normal fenced machinery runs.
    assert upgraded.execute('SELECT rowid FROM v WHERE knn_search(embedding,knn_param(?,?))', (blob(probes[0]), 5)).fetchall() == []
    publish(upgraded)
    upgraded_results = route_results(upgraded, probes)
    upgraded_status = json.loads(upgraded.execute('SELECT vectorlite_index_status(?,?)', ("main", "v")).fetchone()[0])
    upgraded_descriptor = meta_row(upgraded)["descriptor"]
    upgraded.close()

    # Identical data and configuration built directly as Format 5.
    fresh = connect(fresh_path)
    fresh.execute(CREATE)
    for rowid, values in data:
        fresh.execute('INSERT INTO v(rowid,embedding) VALUES(?,?)', (rowid, blob(values)))
    publish(fresh)
    fresh_results = route_results(fresh, probes)
    fresh_status = json.loads(fresh.execute('SELECT vectorlite_index_status(?,?)', ("main", "v")).fetchone()[0])
    fresh_descriptor = meta_row(fresh)["descriptor"]
    fresh.close()

    assert upgraded_results == fresh_results
    # The derived descriptor is the carried Format-4 descriptor with the version
    # bumped, byte-identical to what a fresh Format-5 build stores.
    assert upgraded_descriptor == fresh_descriptor
    assert json.loads(upgraded_descriptor) == {**json.loads(before_meta["descriptor"]), "format_version": 5}
    for key in ("format", "canonical_count", "retired_count", "published_count", "eligible_count"):
        assert upgraded_status[key] == fresh_status[key]
    assert upgraded_status["format"] == 5


def test_active_job_state_is_abandoned_with_bumped_fence(tmp_path):
    path = tmp_path / "active.sqlite"
    data = rows(12)
    legacy = create_format4(path, data)
    active, result = make_active_job(legacy)
    before_meta = meta_row(legacy)
    before_entries = tuple(tuple(row) for row in legacy.execute('SELECT entry_id,public_rowid,created_revision,retired_revision,vector FROM v_diskann_entries ORDER BY entry_id'))
    before_current = tuple(tuple(row) for row in legacy.execute('SELECT public_rowid,entry_id FROM v_diskann_current ORDER BY public_rowid'))
    before_counts = table_counts(legacy, FORMAT4_SUFFIXES)
    assert before_meta["active_job"] == active["job_id"]
    assert before_meta["candidate_generation"] == active["generation_id"]
    assert before_meta["published_generation"] is None
    assert before_counts["generations"] == 1
    assert before_counts["jobs"] == 1
    assert before_counts["graph_nodes"] > 0
    assert before_counts["snapshot_map"] == len(data)
    legacy.close()

    db = connect(path)
    payload = upgrade(db)
    assert payload["abandoned_active_job"] == active["job_id"]
    assert payload["abandoned_candidate_generation"] == active["generation_id"]
    assert payload["abandoned_published_generation"] is None
    after_meta = meta_row(db)
    assert after_meta["next_fence"] == before_meta["next_fence"] + 1
    assert after_meta["next_fence"] > active["fence"]
    assert after_meta["active_job"] is None
    assert after_meta["candidate_generation"] is None
    assert after_meta["published_generation"] is None
    assert table_counts(db, FORMAT5_SUFFIXES) == {
        "meta": 1, "entries": len(data), "current": len(data), "generations": 0, "graph_nodes": 0,
        "generation_retired": 0, "jobs": 0, "snapshot_map": 0, "txn": 2,
    }
    assert tuple(tuple(row) for row in db.execute('SELECT entry_id,public_rowid,created_revision,retired_revision,vector FROM v_diskann_entries ORDER BY entry_id')) == before_entries
    assert tuple(tuple(row) for row in db.execute('SELECT public_rowid,entry_id FROM v_diskann_current ORDER BY public_rowid')) == before_current
    # The abandoned lease cannot reattach: its job row is gone and the bumped
    # fence is above every value the Format-4 database handed out.
    with pytest.raises(sqlite3.Error):
        db.execute(
            'INSERT INTO "main"."v"(operation,path) VALUES (?,?)',
            ("index_cancel", json.dumps({"job_id": active["job_id"], "fence": active["fence"]})),
        )
    # The user republishes through the normal fenced machinery and queries it.
    republish_job, _ = publish(db)
    assert republish_job != active["job_id"]
    assert job(db, job_id=republish_job)["fence"] > active["fence"]
    assert len(db.execute('SELECT rowid FROM v WHERE knn_search(embedding,knn_param(?,?))', (blob(data[0][1]), 5)).fetchall()) == 5
    db.close()


def test_progress_handler_abort_rolls_back_savepoint(tmp_path):
    path = tmp_path / "abort.sqlite"
    data = rows(8)
    legacy = create_format4(path, data)
    make_active_job(legacy)
    before_meta = meta_row(legacy)
    before_catalog = catalog(legacy)
    before_rows = shadow_snapshot(legacy, FORMAT4_SUFFIXES)
    legacy.close()

    db = connect(path)
    baseline_changes = db.total_changes
    state = {"calls": 0, "aborts": 0}

    def handler():
        state["calls"] += 1
        # Abort once, only after the transaction has already deleted a row:
        # the first DELETE moves the counter, so the abort lands mid-upgrade.
        if state["aborts"] == 0 and db.total_changes > baseline_changes:
            state["aborts"] += 1
            return 1
        return 0

    db.set_progress_handler(handler, 5)
    with pytest.raises(sqlite3.DatabaseError, match="interrupted"):
        db.execute('SELECT vectorlite_index_upgrade(?,?)', ("main", "v")).fetchone()
    db.set_progress_handler(None, 0)
    assert state["aborts"] == 1, "the abort must land after the savepoint started mutating"
    assert not db.in_transaction
    # The savepoint rollback left the Format-4 store byte-identical.
    assert catalog(db) == before_catalog
    assert shadow_snapshot(db, FORMAT4_SUFFIXES) == before_rows
    assert meta_row(db) == before_meta
    db.close()

    # The Format-4 file is still usable by the baseline artifact.
    check = connect(path, legacy_artifact())
    assert check.execute('SELECT format_version FROM v_diskann_meta').fetchone()[0] == 4
    assert check.execute('SELECT count(*) FROM v_diskann_entries').fetchone()[0] == len(data)
    assert check.execute('SELECT count(*) FROM v_diskann_graph_nodes').fetchone()[0] > 0
    check.close()

    # And the aborted upgrade can be retried to completion.
    db = connect(path)
    payload = upgrade(db)
    assert payload["upgraded"] is True
    assert meta_row(db)["format_version"] == 5
    db.close()


def test_upgrade_refuses_outer_transaction_and_busy_cursor(tmp_path):
    path = tmp_path / "refuse.sqlite"
    create_format4(path, rows(6)).close()

    db = connect(path)
    db.execute("BEGIN")
    assert "autocommit" in upgrade_error(db)
    db.execute("ROLLBACK")
    assert meta_row(db)["format_version"] == 4
    db.close()

    db = connect(path)
    cursor = db.execute('SELECT singleton FROM v_diskann_txn ORDER BY singleton')
    cursor.fetchone()
    refused = upgrade_error(db)
    assert "transaction state NONE" in refused or "busy" in refused.lower()
    cursor.close()
    assert meta_row(db)["format_version"] == 4
    # With no other cursor open the same connection can upgrade.
    assert upgrade(db)["upgraded"] is True
    db.close()

    path = tmp_path / "journal.sqlite"
    create_format4(path, rows(6)).close()
    db = connect(path)
    assert db.execute("PRAGMA main.journal_mode=off").fetchone()[0] == "off"
    assert "journal_mode=OFF" in upgrade_error(db)
    assert meta_row(db)["format_version"] == 4
    db.close()


def test_concurrent_second_connection_writer_refuses_cleanly(tmp_path):
    path = tmp_path / "busy.sqlite"
    data = rows(6)
    create_format4(path, data).close()
    marker = plain_connect(path)
    marker.execute("CREATE TABLE probe_marker(value INTEGER)")
    marker.close()

    writer = plain_connect(path)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("INSERT INTO probe_marker VALUES(1)")
    db = connect(path, timeout=0.05)
    with pytest.raises(sqlite3.DatabaseError, match="locked"):
        db.execute('SELECT vectorlite_index_upgrade(?,?)', ("main", "v")).fetchone()
    # The refusal is clean: nothing was committed and the Format-4 file stays intact.
    assert meta_row(db)["format_version"] == 4
    assert db.execute('SELECT count(*) FROM v_diskann_entries').fetchone()[0] == len(data)
    writer.execute("ROLLBACK")
    writer.close()
    # After the competing writer releases its lock the upgrade completes.
    assert upgrade(db)["upgraded"] is True
    db.close()


def test_upgrade_refuses_a_committed_intervening_writer(tmp_path):
    """D7: an intervening committed writer makes the upgrade refuse without mutation.

    The upgrade validates the carried Format-4 catalog and metadata before it takes
    the write reservation, so a supported INSERT committed on another connection in
    that window moves the preserved counters (revision, canonical_count,
    next_entry_id). The SAVEPOINT body re-reads the state under the reservation and
    refuses with the stable SQLITE_BUSY "intervening writer" class instead of
    rewriting Format-5 metadata from the stale pre-transaction snapshot; the
    savepoint is rolled back, so the Format-4 file stays byte-identical. Once the
    writer is quiescent the same connection retries and upgrades against the
    post-INSERT state (revision/canonical_count/next_entry_id preserved, next_fence
    bumped by one).
    """
    path = tmp_path / "intervening.sqlite"
    data = rows(6)
    # The second connection runs the preserved baseline artifact: its public INSERT
    # is an ordinary supported Format-4 mutation.
    legacy = create_format4(path, data)
    before_meta = meta_row(legacy)
    before_catalog = catalog(legacy)
    before_counts = table_counts(legacy, FORMAT4_SUFFIXES)
    assert before_meta["format_version"] == 4

    db = connect(path)
    state = {"fired": 0, "committed": None}

    def commit_intervening_writer(sql):
        # The trace callback runs before the traced statement executes; the
        # SAVEPOINT itself takes no lock, so the writer can still commit here.
        if state["fired"] or "SAVEPOINT" not in sql:
            return
        state["fired"] += 1
        try:
            legacy.execute(
                'INSERT INTO v(rowid,embedding) VALUES(?,?)',
                (99, blob([0.5] * DIMENSION)),
            )
            state["committed"] = "committed"
        except sqlite3.Error as error:
            state["committed"] = f"{type(error).__name__}: {error}"

    db.set_trace_callback(commit_intervening_writer)
    try:
        with pytest.raises(sqlite3.Error) as error:
            db.execute('SELECT vectorlite_index_upgrade(?,?)', ("main", "v")).fetchone()
    finally:
        db.set_trace_callback(None)
    assert state["fired"] == 1, "the writer must commit while the upgrade opens its savepoint"
    assert state["committed"] == "committed", state["committed"]
    assert error.value.sqlite_errorcode & 255 == sqlite3.SQLITE_BUSY, error.value
    assert "intervening writer" in str(error.value), error.value
    assert not db.in_transaction

    # The refusal is clean and leaves the Format-4 file byte-identical: no format
    # bump, no ninth layout, unchanged catalog, and the writer's committed entry is
    # the only difference in the canonical tables.
    assert meta_row(db)["format_version"] == 4
    assert meta_row(db)["next_fence"] == before_meta["next_fence"]
    assert db.execute(
        "SELECT count(*) FROM main.sqlite_schema WHERE name='v_diskann_generation_retired'"
    ).fetchone()[0] == 0
    assert catalog(db) == before_catalog
    assert table_counts(db, FORMAT4_SUFFIXES) == {
        **before_counts,
        "entries": before_counts["entries"] + 1,
        "current": before_counts["current"] + 1,
    }
    assert db.execute(
        'SELECT entry_id FROM v_diskann_current WHERE public_rowid=99'
    ).fetchone() is not None
    assert db.execute('SELECT count(*) FROM v_diskann_entries').fetchone()[0] == len(data) + 1

    # With the writer quiescent the retry consumes the post-INSERT state: the
    # preserved counters are exactly the committed metadata and the fence bumps once.
    after_insert_meta = meta_row(legacy)
    payload = upgrade(db)
    assert payload["format"] == 5 and payload["upgraded"] is True
    for key in ("revision", "next_entry_id", "next_job_id", "next_generation_id", "canonical_count", "retired_count", "cleanup_entry_id"):
        assert payload[key] == after_insert_meta[key], key
    assert payload["next_fence"] == before_meta["next_fence"] + 1
    assert payload["revision"] > before_meta["revision"]
    upgraded_meta = meta_row(db)
    assert upgraded_meta["format_version"] == 5
    assert upgraded_meta["revision"] == after_insert_meta["revision"]
    assert upgraded_meta["canonical_count"] == after_insert_meta["canonical_count"]
    assert upgraded_meta["next_entry_id"] == after_insert_meta["next_entry_id"]
    assert db.execute('SELECT count(*) FROM v_diskann_entries').fetchone()[0] == len(data) + 1
    print(
        "D7_INTERVENING_WRITER "
        + json.dumps(
            {
                "refusal_code": error.value.sqlite_errorcode,
                "preserved_revision": payload["revision"],
                "preserved_canonical_count": payload["canonical_count"],
                "next_fence": payload["next_fence"],
            },
            sort_keys=True,
        )
    )
    db.close()
    legacy.close()


def test_upgrade_preserves_retired_history_and_republishes_live_rows(tmp_path):
    path = tmp_path / "churn.sqlite"
    data = rows(10)
    legacy = create_format4(path, data)
    # Supported Format-4 churn: one version replacement and one deletion.
    legacy.execute('UPDATE v SET embedding=? WHERE rowid=3', (blob([9.0, -9.0, 4.0, 1.0]),))
    legacy.execute('DELETE FROM v WHERE rowid=7')
    before_meta = meta_row(legacy)
    before_entries = tuple(tuple(row) for row in legacy.execute('SELECT entry_id,public_rowid,created_revision,retired_revision,vector FROM v_diskann_entries ORDER BY entry_id'))
    before_current = tuple(tuple(row) for row in legacy.execute('SELECT public_rowid,entry_id FROM v_diskann_current ORDER BY public_rowid'))
    assert before_meta["canonical_count"] == 9
    assert before_meta["retired_count"] >= 2
    assert len(before_entries) == 11
    legacy.close()

    db = connect(path)
    payload = upgrade(db)
    assert payload["canonical_count"] == before_meta["canonical_count"]
    assert payload["retired_count"] == before_meta["retired_count"]
    assert payload["cleanup_entry_id"] == before_meta["cleanup_entry_id"]
    after_meta = meta_row(db)
    assert after_meta["canonical_count"] == before_meta["canonical_count"]
    assert after_meta["retired_count"] == before_meta["retired_count"]
    assert after_meta["cleanup_entry_id"] == before_meta["cleanup_entry_id"]
    assert tuple(tuple(row) for row in db.execute('SELECT entry_id,public_rowid,created_revision,retired_revision,vector FROM v_diskann_entries ORDER BY entry_id')) == before_entries
    assert tuple(tuple(row) for row in db.execute('SELECT public_rowid,entry_id FROM v_diskann_current ORDER BY public_rowid')) == before_current

    publish(db)
    live = {rowid for rowid, _ in data if rowid != 7}
    found = {row[0] for row in db.execute('SELECT rowid FROM v WHERE knn_search(embedding,knn_param(?,?))', (blob(data[0][1]), 20))}
    assert found == live
    assert db.execute('SELECT embedding FROM v WHERE rowid=7').fetchone() is None
    db.close()


def test_reverse_route_rejects_the_upgraded_database_cleanly(tmp_path):
    path = tmp_path / "upgraded.sqlite"
    data = rows(6)
    db = create_format4(path, data)
    db.close()
    db = connect(path)
    assert upgrade(db)["upgraded"] is True
    db.close()
    # Reverting to the Format-4 build rejects the file instead of misreading it.
    legacy = connect(path, legacy_artifact())
    with pytest.raises(sqlite3.DatabaseError, match="format version is unsupported"):
        legacy.execute('SELECT embedding FROM v WHERE rowid=1').fetchall()
    legacy.close()


def test_upgrade_runs_on_a_wal_database(tmp_path):
    path = tmp_path / "wal.sqlite"
    data = rows(8)
    legacy = create_format4(path, data)
    assert legacy.execute("PRAGMA journal_mode=wal").fetchone()[0] == "wal"
    legacy.close()

    db = connect(path)
    assert db.execute("PRAGMA main.journal_mode").fetchone()[0] == "wal"
    assert upgrade(db)["upgraded"] is True
    assert meta_row(db)["format_version"] == 5
    publish(db)
    assert len(db.execute('SELECT rowid FROM v WHERE knn_search(embedding,knn_param(?,?))', (blob(data[0][1]), 5)).fetchall()) == 5
    db.close()


def test_upgrade_supports_an_attached_schema_target(tmp_path):
    main_path = tmp_path / "main.sqlite"
    side_path = tmp_path / "side.sqlite"
    data = rows(8)
    remove_database(main_path)
    remove_database(side_path)
    legacy = connect(main_path, legacy_artifact())
    legacy.execute("ATTACH DATABASE ? AS aux", (str(side_path),))
    legacy.execute('CREATE VIRTUAL TABLE "aux"."v" USING vectorlite(embedding float32[4] l2,diskann(degree=4,build_list_size=8))')
    for rowid, values in data:
        legacy.execute('INSERT INTO aux.v(rowid,embedding) VALUES(?,?)', (rowid, blob(values)))
    legacy.close()

    db = connect(main_path)
    db.execute("ATTACH DATABASE ? AS aux", (str(side_path),))
    payload = upgrade(db, schema="aux", table="v")
    assert payload["upgraded"] is True and payload["schema"] == "aux"
    assert db.execute('SELECT format_version FROM aux.v_diskann_meta').fetchone()[0] == 5
    assert db.execute('SELECT count(*) FROM aux.v_diskann_entries').fetchone()[0] == len(data)
    assert db.execute("SELECT count(*) FROM aux.sqlite_schema WHERE name='v_diskann_generation_retired'").fetchone()[0] == 1
    # The upgraded auxiliary table connects and serves canonical reads; the job
    # machinery (whose helpers target main) can republish it from a connection
    # where it is the only attached index.
    assert tuple(db.execute('SELECT embedding FROM aux.v WHERE rowid=3').fetchone()) == (blob(data[3][1]),)
    assert db.execute('SELECT rowid FROM aux.v WHERE knn_search(embedding,knn_param(?,?))', (blob(data[0][1]), 5)).fetchall() == []
    db.close()


def test_upgrade_refuses_a_database_that_was_born_format5(tmp_path):
    path = tmp_path / "fresh5.sqlite"
    db = connect(path)
    db.execute(CREATE)
    db.execute('INSERT INTO v(rowid,embedding) VALUES(1,?)', (blob([1.0, 2.0, 3.0, 4.0]),))
    before = meta_row(db)
    message = upgrade_error(db)
    assert "requires an unupgraded Format-4 database" in message
    assert "found format_version 5" in message
    assert meta_row(db) == before
    db.close()


def test_upgrade_supports_an_empty_format4_database(tmp_path):
    path = tmp_path / "empty.sqlite"
    legacy = create_format4(path, [])
    before = meta_row(legacy)
    legacy.close()
    db = connect(path)
    payload = upgrade(db)
    assert payload["upgraded"] is True
    assert payload["canonical_count"] == 0
    after = meta_row(db)
    assert after["format_version"] == 5
    assert after["canonical_count"] == 0
    assert after["next_fence"] == before["next_fence"] + 1
    db.close()
