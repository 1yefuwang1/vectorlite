"""Artifact-independent checks for the matched native identity-cache benchmark."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import struct
import subprocess
from unittest import mock

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "benchmark" / "diskann_identity_benchmark.py"
SPEC = importlib.util.spec_from_file_location("diskann_identity_benchmark", MODULE_PATH)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def descriptor():
    return {"backend": "diskann", "format_version": 3, "dimension": 2,
            "vector_type": "float32", "metric": "l2_squared", "normalize": False,
            "adapter_version": "0.60", "atomic_write_protocol": "guard-two-row-v1",
            "deletion_policy": "retained-topology-v1", "maintenance": "streamed-rebuild-v1",
            "options": bench.OPTIONS}


def snapshot(path):
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE v_diskann_meta(singleton INTEGER,format_version INTEGER,instance_id BLOB,"
            "descriptor BLOB,revision INTEGER,next_node_id INTEGER,live_count INTEGER,deleted_count INTEGER,entrypoints BLOB);"
            "CREATE TABLE v_diskann_nodes(node_id INTEGER PRIMARY KEY,public_rowid INTEGER,state INTEGER,vector BLOB,neighbors BLOB);"
            "CREATE TABLE v_diskann_txn(singleton INTEGER,value INTEGER);"
            "INSERT INTO v_diskann_txn VALUES(1,0),(2,0);"
            "CREATE TABLE v_diskann_rebuild(public_rowid INTEGER,vector BLOB);")
        connection.execute("INSERT INTO v_diskann_meta VALUES(1,3,?,?,1,11,10,0,?)",
                           (b"a" * 16, json.dumps(descriptor()).encode(), struct.pack("<Q", 0)))
        for node in range(11):
            connection.execute("INSERT INTO v_diskann_nodes VALUES(?,?,?,?,?)",
                               (node, node - 1 if node else None, 0 if node else 2,
                                struct.pack("<2f", node, 1), struct.pack("<2Q", 1, 2) if not node else b""))
    return path


def update(path, sql, parameters=()):
    with sqlite3.connect(path) as connection:
        connection.execute(sql, parameters)


def graph(path):
    return bench.graph_snapshot(path, 10, 2, "l2")


def result():
    return {"mode": "batch8", "count": 10, "dimension": 2, "metric": "l2", "sqlite": "3.53.4",
            "storage": "memory", "database": ":memory:", "snapshot": "snapshot.sqlite",
            "tracing": False, "max_visits": 65536, "build_seconds": 1,
            "query_windows": [{"window": window, "milliseconds": [0.1], "labels": [list(range(10))]}
                              for window in (10, 50, 100)]}


def validate(value):
    case = {**bench.CASES["l2-batch"], "count": 10, "dimension": 2}
    bench.validate_result(value, case, [list(range(10))], ":memory:", "snapshot.sqlite")


def test_hash_excludes_only_instance_id_and_preserves_adjacency_order(tmp_path):
    path = snapshot(tmp_path / "snapshot.sqlite")
    original = graph(path)
    update(path, "UPDATE v_diskann_meta SET instance_id=?", (b"b" * 16,))
    assert graph(path) == original
    update(path, "UPDATE v_diskann_nodes SET neighbors=? WHERE node_id=0", (struct.pack("<2Q", 2, 1),))
    changed = graph(path)
    assert changed["sha256"] != original["sha256"]
    update(path, "UPDATE v_diskann_nodes SET vector=? WHERE node_id=1", (struct.pack("<2f", 0.5, 1),))
    assert graph(path)["sha256"] != changed["sha256"]


@pytest.mark.parametrize("sql,parameters", [
    ("UPDATE v_diskann_meta SET revision=2", ()),
    ("UPDATE v_diskann_meta SET descriptor=?", (json.dumps(descriptor(), indent=2).encode(),)),
])
def test_hash_includes_stable_metadata_bytes(tmp_path, sql, parameters):
    path = snapshot(tmp_path / "snapshot.sqlite")
    original = graph(path)
    update(path, sql, parameters)
    assert graph(path)["sha256"] != original["sha256"]


def test_typed_hash_framing_distinguishes_all_record_fields():
    original = [1, 0, 0, b"vector", b"adjacency"]
    def digest(row):
        value = hashlib.sha256()
        for field in row:
            bench.hash_value(value, field)
        return value.hexdigest()
    for index, value in enumerate([2, None, 2, b"changed", b"different"]):
        changed = original.copy()
        changed[index] = value
        assert digest(changed) != digest(original)
    assert digest([b"ab", b"c"]) != digest([b"a", b"bc"])
    assert digest([1]) != digest([b"1"])


@pytest.mark.parametrize("sql,parameters", [
    ("ALTER TABLE v_diskann_nodes ADD COLUMN extra INTEGER", ()),
    ("UPDATE v_diskann_meta SET live_count=9", ()),
    ("UPDATE v_diskann_meta SET instance_id=?", (b"short",)),
    ("UPDATE v_diskann_nodes SET public_rowid=99 WHERE node_id=1", ()),
    ("UPDATE v_diskann_nodes SET state=1 WHERE node_id=1", ()),
    ("UPDATE v_diskann_nodes SET vector=? WHERE node_id=1", (struct.pack("<2f", float("nan"), 1),)),
    ("UPDATE v_diskann_nodes SET neighbors=? WHERE node_id=1", (struct.pack("<Q", 99),)),
    ("UPDATE v_diskann_txn SET value=1", ()),
    ("INSERT INTO v_diskann_rebuild VALUES(1,?)", (b"12345678",)),
])
def test_snapshot_validation_rejects_inconsistency(tmp_path, sql, parameters):
    path = snapshot(tmp_path / "snapshot.sqlite")
    update(path, sql, parameters)
    with pytest.raises(ValueError):
        graph(path)


def test_cosine_oracle_float64_and_stable_ties():
    vectors = np.array([[10, 0], [1, 1], [0, 0], [-1, 0]], dtype=np.float32)
    queries = np.array([[1, 0], [0, 0]], dtype=np.float32)
    assert bench.oracle(vectors, queries, "cosine") == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert bench.oracle(vectors, queries[:1], "l2") == [[1, 2, 3, 0]]
    with pytest.raises(ValueError):
        bench.oracle(vectors, queries, "invalid")


def test_result_validation_and_recall():
    value = result()
    validate(value)
    assert [row["recall_at_10"] for row in value["query_windows"]] == [1, 1, 1]


@pytest.mark.parametrize("change", [
    lambda value: value.update(metric="cosine"),
    lambda value: value.update(storage="file"),
    lambda value: value.update(tracing=True),
    lambda value: value.update(tracing=0),
    lambda value: value.update(build_seconds=float("nan")),
    lambda value: value.update(query_windows=[None, {}, {}]),
    lambda value: value["query_windows"].pop(),
    lambda value: value["query_windows"][0].update(window=11),
    lambda value: value["query_windows"][0].update(milliseconds=[]),
    lambda value: value["query_windows"][0].update(milliseconds=[-1]),
    lambda value: value["query_windows"][0].update(labels=[list(range(9))]),
    lambda value: value["query_windows"][0].update(labels=[[0] * 10]),
    lambda value: value["query_windows"][0].update(labels=[[True] + list(range(1, 10))]),
    lambda value: value["query_windows"][0].update(labels=[list(range(1, 11))]),
])
def test_result_validation_rejects_invalid_queries_and_provenance(change):
    value = result()
    change(value)
    with pytest.raises(ValueError):
        validate(value)


def test_strict_graph_and_ordered_query_comparison():
    native = result()
    validate(native)
    reference = {"graph": {"sha256": "first"}, "native": native}
    bench.verify_same(reference, copy.deepcopy(reference))
    changed = copy.deepcopy(reference)
    changed["graph"]["sha256"] = "second"
    with pytest.raises(ValueError, match="graph"):
        bench.verify_same(reference, changed)
    changed = copy.deepcopy(reference)
    changed["native"]["query_windows"][0]["labels"][0].reverse()
    with pytest.raises(ValueError, match="ordered query"):
        bench.verify_same(reference, changed)


def test_ordering_baseline_selfcheck_then_warmups_then_alternating_pairs():
    order = list(bench.run_order(2, 1))
    assert order == [("selfcheck", 0, "baseline"), ("selfcheck", 1, "baseline"),
                     ("warmup", 0, "baseline"), ("warmup", 0, "optimized"),
                     ("measured", 0, "baseline"), ("measured", 0, "optimized"),
                     ("measured", 1, "optimized"), ("measured", 1, "baseline")]
    assert len(list(bench.run_order(2, 0))) == 6


def test_summary_uses_successful_measured_pairs_and_retains_all_times():
    def run(index, variant, seconds, phase="measured", status="ok"):
        return {"case": "l2-batch", "index": index, "variant": variant, "phase": phase,
                "status": status, "native": {"build_seconds": seconds}}
    runs = [run(0, "baseline", 4), run(0, "optimized", 2), run(1, "optimized", 2), run(1, "baseline", 6),
            run(0, "baseline", 100, phase="warmup"), run(2, "baseline", 9), run(2, "optimized", 1, status="error")]
    value = bench.summary(runs)["l2-batch"]
    assert value["build_seconds"] == {"baseline": [4, 6, 9], "optimized": [2, 2]}
    assert value["median_build_seconds"] == {"baseline": 6, "optimized": 2}
    assert value["paired_speedups"] == [2, 3]
    assert value["median_paired_speedup"] == 2.5
    assert value["paired_speedup_min"] == 2
    assert value["paired_speedup_max"] == 3


def test_fresh_output_refuses_existing_directory_file_and_dangling_symlink(tmp_path):
    fresh = tmp_path / "fresh"
    assert bench.fresh_output(fresh) == fresh.resolve()
    with pytest.raises(ValueError):
        bench.fresh_output(fresh)
    file = tmp_path / "file"
    file.write_text("keep")
    with pytest.raises(ValueError):
        bench.fresh_output(file)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError):
        bench.fresh_output(link)
    assert file.read_text() == "keep"


def test_inputs_are_matched_immutable_and_repeatable(tmp_path):
    first = bench.make_inputs(tmp_path, 10, 2)
    assert (tmp_path / "vectors-10.bin").stat().st_mode & 0o222 == 0
    second_path = tmp_path / "other"
    second_path.mkdir()
    second = bench.make_inputs(second_path, 10, 2)
    for key in ("vectors", "queries"):
        assert first["files"][key]["sha256"] == second["files"][key]["sha256"]
    with pytest.raises(FileExistsError):
        bench.make_inputs(tmp_path, 10, 2)


@pytest.mark.parametrize("behavior", ["nonzero", "timeout", "invalid-json", "invalid-result", "nonfinite"])
def test_errors_preserve_raw_process_evidence(tmp_path, behavior):
    inputs = {"files": {"vectors": {"path": "vectors.bin"}, "queries": {"path": "queries.bin"}}}
    record = {"id": "run", "case": "l2-batch", "phase": "selfcheck", "index": 0, "variant": "baseline"}
    case = {**bench.CASES["l2-batch"], "count": 10, "dimension": 2, "queries": 1}
    if behavior == "timeout":
        response = subprocess.TimeoutExpired("host", 2, output=b"partial", stderr=b"timeout details")
    else:
        stdout = {"invalid-result": "{}", "nonfinite": '{"build_seconds":1e999}'}.get(behavior, "partial")
        response = subprocess.CompletedProcess("host", 7 if behavior == "nonzero" else 0, stdout, "details")
    with mock.patch.object(bench.subprocess, "run") as native:
        if isinstance(response, Exception):
            native.side_effect = response
        else:
            native.return_value = response
        bench.execute_run(record, Path("host"), Path("library"), case, inputs, tmp_path, 2, [list(range(10))], None)
    saved = json.loads((tmp_path / "run.json").read_text())
    assert saved["status"] == "error"
    assert saved["stderr"] in ("details", "timeout details")
    assert saved["stdout"] in ("partial", "{}", '{"build_seconds":1e999}')
    assert saved["command"][:2] == ["host", "library"]
    assert saved["environment"]["VECTORLITE_BENCH_METRIC"] == "l2"


def test_main_stops_first_failure_retains_prior_success_and_partial_summary(tmp_path, capsys):
    executable = tmp_path / "host"
    library = tmp_path / "vectorlite.dylib"
    executable.write_bytes(b"host")
    library.write_bytes(b"library")
    output = tmp_path / "output"
    calls = []
    def execute(record, *args):
        calls.append(record["id"])
        if len(calls) == 1:
            record.update(status="ok", native={"build_seconds": 1}, graph={"sha256": "first"})
        else:
            record.update(status="error", error="deliberate nondeterminism")
    with mock.patch.object(bench, "execute_run", side_effect=execute):
        status = bench.main(["--executable", str(executable), "--baseline", str(library), "--optimized", str(library),
                             "--output-dir", str(output), "--count", "10", "--queries", "1", "--pairs", "2", "--warmups", "0"])
    assert status == 1
    assert calls == ["l2-batch-selfcheck-00-baseline", "l2-batch-selfcheck-01-baseline"]
    report = json.loads((output / "results.json").read_text())
    assert report["status"] == "error"
    assert report["runs"][0]["status"] == "ok"
    assert report["runs"][1]["error"] == "deliberate nondeterminism"
    assert report["summary"] == {}
    with pytest.raises(SystemExit) as error:
        bench.main(["--executable", str(executable), "--baseline", str(library), "--optimized", str(library),
                    "--output-dir", str(output)])
    assert error.value.code == 2
    assert len(calls) == 2
