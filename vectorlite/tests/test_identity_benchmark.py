"""Artifact-independent checks for the matched native identity-cache benchmark."""
import argparse
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


def result(*, storage="memory", controls=None, milliseconds=None, count=10, dimension=2, timeout=600):
    controls = bench.normalized_controls(controls)
    milliseconds = [0.1] if milliseconds is None else milliseconds
    defaults = {"page_size": 4096, "cache_size": -2000, "cache_spill": 483,
                "journal_mode": "memory" if storage == "memory" else "delete", "synchronous": 2,
                "mmap_size": None if storage == "memory" else 0, "temp_store": 0, "wal_autocheckpoint": 1000}
    effective = defaults.copy()
    for key in ("page_size", "cache_size", "mmap_size", "journal_mode"):
        if controls[key] is not None:
            effective[key] = controls[key]
    if storage == "memory":
        effective["mmap_size"] = None
    if controls["cache_size"] is not None or controls["page_size"] is not None:
        effective["cache_spill"] = max(1, abs(effective["cache_size"]) // 4)
    if controls["cache_spill"] is not None:
        effective["cache_spill"] = effective["cache_spill"] if controls["cache_spill"] == "on" else 0
    if controls["synchronous"] is not None:
        effective["synchronous"] = {"full": 2, "extra": 3}[controls["synchronous"]]
    if controls["temp_store"] is not None:
        effective["temp_store"] = {"default": 0, "file": 1, "memory": 2}[controls["temp_store"]]
    checkpoint = {"performed": False, "mode": None, "after_queries": False,
                  "seconds": None, "rc": None, "log_frames": None, "checkpointed_frames": None}
    if effective["journal_mode"] == "wal":
        effective["wal_autocheckpoint"] = 0
        checkpoint.update(performed=True, mode="full", after_queries=True, seconds=0.01, rc=0,
                          log_frames=9, checkpointed_frames=9)
    return {"mode": "batch8", "count": count, "dimension": dimension, "metric": "l2", "sqlite": "3.53.4",
            "storage": storage, "database": ":memory:" if storage == "memory" else "database.sqlite", "snapshot": "snapshot.sqlite",
            "tracing": False, "max_visits": 65536, "build_seconds": 1, "timeout_seconds": timeout,
            "sqlite_source_id": "2026-01-01 fake-test-source-id", "sqlite_compile_options": ["THREADSAFE=1", "MAX_MMAP_SIZE=0x7fff0000"],
            "pager": {"requested": controls, "defaults": defaults, "effective": effective},
            "ann_settings": bench.OPTIONS.copy(), "final_checkpoint": checkpoint,
            "query_lifecycle": {"connection": "same_open_post_build", "warmup_passes": 1, "timed_passes": 1,
                                "queries_per_pass": len(milliseconds), "k": 10, "windows": [10, 50, 100]},
            "query_windows": [{"window": window, "milliseconds": milliseconds.copy(), "labels": [list(range(10)) for _ in milliseconds],
                               "distance_bits": [[0] * 10 for _ in milliseconds]}
                              for window in (10, 50, 100)]}


def validate(value, controls=None):
    case = {**bench.CASES["l2-batch"], "storage": value["storage"], "count": 10, "dimension": 2}
    bench.validate_result(value, case, [list(range(10))] * value["query_lifecycle"]["queries_per_pass"],
                          ":memory:" if case["storage"] == "memory" else "database.sqlite", "snapshot.sqlite", controls, 600)


def measured(index, variant, seconds=1, *, phase="measured", status="ok", native=None, name="l2-batch"):
    native = result() if native is None else native
    native["build_seconds"] = seconds
    for row in native["query_windows"]:
        row.update(bench.query_percentiles(native)[row["window"]])
        row["recall_at_10"] = 1
    return {"case": name, "index": index, "variant": variant, "phase": phase, "status": status,
            "native": native, "graph": {"sha256": "same-graph"}}


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
    lambda value: value["query_windows"][0].pop("distance_bits"),
    lambda value: value["query_windows"][0].update(distance_bits=None),
    lambda value: value["query_windows"][0].update(distance_bits=[]),
    lambda value: value["query_windows"][0].update(distance_bits=[[0] * 9]),
    lambda value: value["query_windows"][0].update(distance_bits=[[True] + [0] * 9]),
    lambda value: value["query_windows"][0].update(distance_bits=[[-1] + [0] * 9]),
])
def test_result_validation_rejects_invalid_queries_and_provenance(change):
    value = result()
    change(value)
    with pytest.raises(ValueError):
        validate(value)


def test_strict_graph_and_ordered_query_comparison():
    native = result()
    for row in native["query_windows"]:
        row["distance_bits"] = [[0] * 10]
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
    changed = copy.deepcopy(reference)
    changed["native"]["query_windows"][0]["distance_bits"][0][0] = 1
    with pytest.raises(ValueError, match="distance_bits"):
        bench.verify_same(reference, changed)
    bad_nan = copy.deepcopy(native)
    bad_nan["query_windows"][0]["distance_bits"] = [[0x7FC00000] + [0] * 9]
    with pytest.raises(ValueError, match="distance_bits"):
        validate(bad_nan)


def test_ordering_baseline_selfcheck_then_warmups_then_alternating_pairs():
    order = list(bench.run_order(2, 1))
    assert order == [("selfcheck", 0, "baseline"), ("selfcheck", 1, "baseline"),
                     ("warmup", 0, "baseline"), ("warmup", 0, "optimized"),
                     ("measured", 0, "baseline"), ("measured", 0, "optimized"),
                     ("measured", 1, "optimized"), ("measured", 1, "baseline")]
    assert len(list(bench.run_order(2, 0))) == 6


def test_summary_uses_only_complete_successful_measured_pairs():
    runs = [measured(0, "baseline", 4), measured(0, "optimized", 2), measured(1, "optimized", 2), measured(1, "baseline", 6),
            measured(0, "baseline", 100, phase="warmup"), measured(0, "baseline", 200, phase="selfcheck"),
            measured(0, "optimized", 300, phase="diagnostic"),
            measured(2, "baseline", 9), measured(2, "optimized", 1, status="error")]
    value = bench.summary(runs)["l2-batch"]
    assert value["validation"] == "partial"
    assert value["build_seconds"] == {"baseline": [4, 6], "optimized": [2, 2]}
    assert value["median_build_seconds"] == {"baseline": 5, "optimized": 2}
    assert value["pair_indices"] == [0, 1]
    assert value["incomplete_pair_indices"] == [2]
    assert value["paired_speedups"] == [2, 3]
    assert value["median_paired_speedup"] == 2.5
    assert value["paired_speedup_min"] == 2
    assert value["paired_speedup_max"] == 3
    assert value["build_candidate_over_baseline"]["paired_candidate_over_baseline"] == [0.5, 1 / 3]
    assert value["build_candidate_over_baseline"]["median_paired_candidate_over_baseline"] == pytest.approx(5 / 12)
    cases = {"l2-batch": {**bench.CASES["l2-batch"], "count": 10, "dimension": 2, "pairs": 3}}
    with pytest.raises(ValueError, match="incomplete/failed"):
        bench.summary(runs, cases, final=True)


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
    def execute(record, *args, **kwargs):
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


@pytest.mark.parametrize("controls", [
    {"page_size": 0}, {"page_size": 511}, {"page_size": 1000}, {"page_size": 131072}, {"page_size": True},
    {"cache_size": 0}, {"cache_size": -0x80000000}, {"cache_size": 0x80000000}, {"cache_size": "-2048"},
    {"cache_size": False}, {"cache_spill": 1}, {"cache_spill": "ON"},
    {"journal_mode": "off"}, {"journal_mode": "truncate"}, {"synchronous": "normal"}, {"synchronous": "off"},
    {"mmap_size": -1}, {"mmap_size": 0x8000000000000000}, {"mmap_size": 1.0},
    {"temp_store": 2}, {"temp_store": "disk"}, {"wal_autocheckpoint": 1000},
])
def test_invalid_pager_controls_fail_closed(controls):
    with pytest.raises(ValueError):
        bench.normalized_controls(controls)


@pytest.mark.parametrize("text", ["", "-", "+2", " 2", "2 ", "-2048x", "0", "-2147483648", "2147483648", "１２"])
def test_cache_cli_rejects_malformed_or_unsafe_values(text):
    with pytest.raises(argparse.ArgumentTypeError):
        bench.cache_argument(text)


def test_valid_control_schema_and_cli_defaults():
    assert bench.cache_argument("default") is None
    assert bench.cache_argument("-65536") == -65536
    assert bench.cache_argument("256") == 256
    controls = bench.normalized_controls({"page_size": 65536, "cache_size": -65536, "cache_spill": "off",
                                          "journal_mode": "wal", "synchronous": "full", "mmap_size": 0,
                                          "temp_store": "memory"})
    assert set(controls) == set(bench.CONTROL_KEYS)
    args = bench.parser().parse_args(["--executable", "host", "--baseline", "library", "--optimized", "library", "--output-dir", "new"])
    assert args.timeout == 600
    assert args.queries == 100
    assert args.warmups == 1
    assert all(case["pairs"] == 5 for case in bench.CASES.values())
    assert bench.variant_controls(args, "file")["baseline"]["journal_mode"] == "delete"
    assert bench.variant_controls(args, "memory")["baseline"]["journal_mode"] is None
    assert bench.variant_controls(args, "memory")["baseline"]["synchronous"] == "full"


@pytest.mark.parametrize("change", [
    lambda value: value.pop("pager"),
    lambda value: value["pager"].update(requested=None),
    lambda value: value["pager"]["requested"].pop("cache_size"),
    lambda value: value["pager"]["requested"].update(wal_autocheckpoint=0),
    lambda value: value["pager"]["defaults"].pop("temp_store"),
    lambda value: value["pager"]["effective"].update(cache_size=-8192),
    lambda value: value["pager"]["effective"].update(cache_spill=0),
    lambda value: value["pager"]["effective"].update(synchronous=1),
    lambda value: value["pager"]["effective"].update(page_size=8192),
    lambda value: value["pager"]["effective"].update(temp_store=True),
    lambda value: value["pager"]["effective"].update(journal_mode="off"),
    lambda value: value["pager"]["effective"].update(wal_autocheckpoint=0),
    lambda value: value.update(sqlite_source_id=""),
    lambda value: value.pop("sqlite_source_id"),
    lambda value: value.update(sqlite_compile_options=[]),
    lambda value: value.update(sqlite_compile_options=["THREADSAFE=1", "THREADSAFE=1"]),
    lambda value: value.update(sqlite_compile_options=[False]),
    lambda value: value.update(timeout_seconds=0),
    lambda value: value.update(timeout_seconds=599),
    lambda value: value["ann_settings"].update(alpha=1.3),
    lambda value: value["ann_settings"].update(degree=True),
    lambda value: value.pop("ann_settings"),
    lambda value: value["query_lifecycle"].update(connection="reopened"),
    lambda value: value["query_lifecycle"].update(warmup_passes=0),
    lambda value: value["query_lifecycle"].update(timed_passes=2),
    lambda value: value["query_lifecycle"].update(k=True),
    lambda value: value["query_lifecycle"].update(windows=[10, 100, 50]),
    lambda value: value["query_windows"][0].update(milliseconds=[0]),
    lambda value: value["query_windows"][0].update(milliseconds=[float("inf")]),
    lambda value: value["query_windows"][0].update(milliseconds=[True]),
    lambda value: value.update(build_seconds=0),
])
def test_strict_effective_settings_and_native_context_validation(change):
    value = result()
    change(value)
    with pytest.raises(ValueError):
        validate(value)


@pytest.mark.parametrize("change", [
    lambda value: value.pop("final_checkpoint"),
    lambda value: value["final_checkpoint"].update(performed=False),
    lambda value: value["final_checkpoint"].update(mode="passive"),
    lambda value: value["final_checkpoint"].update(after_queries=False),
    lambda value: value["final_checkpoint"].update(rc=5),
    lambda value: value["final_checkpoint"].update(rc=False),
    lambda value: value["final_checkpoint"].update(log_frames=-1),
    lambda value: value["final_checkpoint"].update(checkpointed_frames=8),
    lambda value: value["final_checkpoint"].update(seconds=0),
    lambda value: value["final_checkpoint"].update(seconds=float("nan")),
    lambda value: value["pager"]["effective"].update(wal_autocheckpoint=1000),
])
def test_wal_requires_disabled_autocheckpoint_and_successful_final_full_checkpoint(change):
    controls = {"journal_mode": "wal", "synchronous": "full"}
    value = result(storage="file", controls=controls)
    validate(value, controls)
    change(value)
    with pytest.raises(ValueError):
        validate(value, controls)


def test_nonwal_must_not_claim_a_checkpoint_and_mmap_readback_is_required_for_file():
    value = result()
    value["final_checkpoint"].update(performed=True)
    with pytest.raises(ValueError, match="checkpoint"):
        validate(value)
    value = result(storage="file")
    value["pager"]["effective"]["mmap_size"] = None
    with pytest.raises(ValueError, match="mmap_size"):
        validate(value)


def test_explicit_control_readbacks_and_clamped_values():
    controls = {"page_size": 8192, "cache_size": -8192, "cache_spill": "off", "journal_mode": "delete",
                "synchronous": "extra", "mmap_size": 0, "temp_store": "memory"}
    value = result(storage="file", controls=controls)
    validate(value, controls)
    value["pager"]["effective"]["cache_spill"] = 1
    with pytest.raises(ValueError, match="cache_spill"):
        validate(value, controls)
    value = result(storage="file", controls={"mmap_size": 4096})
    value["pager"]["effective"]["mmap_size"] = 0
    with pytest.raises(ValueError, match="mmap_size"):
        validate(value, {"mmap_size": 4096})
    value = result(controls={"mmap_size": 0})
    validate(value, {"mmap_size": 0})  # In-memory mmap is explicitly not applicable.


def test_pager_ablation_allows_only_declared_cache_size_difference_and_derived_spill():
    first = {"native": result(storage="file"), "graph": {"sha256": "same"}}
    second = {"native": result(storage="file", controls={"cache_size": -8192}), "graph": {"sha256": "same"}}
    validate(first["native"])
    validate(second["native"], {"cache_size": -8192})
    with pytest.raises(ValueError, match="settings mismatch"):
        bench.verify_same(first, second)
    bench.verify_same(first, second, allow_cache_difference=True)
    second["native"]["pager"]["effective"]["mmap_size"] = 4096
    with pytest.raises(ValueError, match="mmap_size"):
        bench.verify_same(first, second, allow_cache_difference=True)


@pytest.mark.parametrize("key", ["sqlite", "sqlite_source_id", "sqlite_compile_options", "ann_settings", "query_lifecycle", "timeout_seconds"])
def test_same_configuration_rejects_native_provenance_drift(key):
    first = {"native": result()}
    second = copy.deepcopy(first)
    second["native"][key] = "drift"
    with pytest.raises(ValueError, match="settings/SQLite"):
        bench.verify_settings(first, second)


def test_pager_configuration_selfchecks_are_separate_and_precede_warmups():
    order = list(bench.run_order(1, 1, variant_selfchecks=True))
    assert order[:4] == [("selfcheck", 0, "baseline"), ("selfcheck", 1, "baseline"),
                         ("selfcheck", 0, "optimized"), ("selfcheck", 1, "optimized")]
    assert order[4:] == [("warmup", 0, "baseline"), ("warmup", 0, "optimized"),
                         ("measured", 0, "baseline"), ("measured", 0, "optimized")]


def test_linear_per_process_percentiles_and_median_of_paired_ratios_not_pooling():
    first = result(milliseconds=[1, 2, 3, 10])
    validate(first)
    assert first["query_windows"][0]["p50_ms"] == 2.5
    assert first["query_windows"][0]["p95_ms"] == pytest.approx(8.95)
    runs = [measured(0, "baseline", 4, native=first),
            measured(0, "optimized", 2, native=result(milliseconds=[2, 4, 6, 20])),
            measured(1, "baseline", 6, native=result(milliseconds=[10, 20, 30, 100])),
            measured(1, "optimized", 2, native=result(milliseconds=[5, 10, 15, 50])),
            measured(2, "baseline", 1, phase="warmup", native=result(milliseconds=[1000] * 4)),
            measured(2, "optimized", 1, phase="selfcheck", native=result(milliseconds=[2000] * 4))]
    cases = {"l2-batch": {**bench.CASES["l2-batch"], "pairs": 2, "count": 10, "dimension": 2, "queries": 4}}
    value = bench.summary(runs, cases, final=True)["l2-batch"]
    assert value["complete"] is True
    assert value["validation"] == "final"
    for window in ("10", "50", "100"):
        for percentile in ("p50_ms", "p95_ms"):
            metrics = value["query_percentiles"][window][percentile]
            assert metrics["paired_candidate_over_baseline"] == pytest.approx([2, 0.5])
            assert metrics["median_paired_candidate_over_baseline"] == pytest.approx(1.25)
            assert metrics["regression_over_5_percent"] is True
    assert value["query_percentiles"]["10"]["p50_ms"]["milliseconds"] == {"baseline": [2.5, 25], "optimized": [5, 12.5]}
    pooled_ratio = np.percentile([2, 4, 6, 20, 5, 10, 15, 50], 50, method="linear") / np.percentile([1, 2, 3, 10, 10, 20, 30, 100], 50, method="linear")
    assert pooled_ratio == 0.8
    assert value["query_percentiles"]["10"]["p50_ms"]["median_paired_candidate_over_baseline"] != pooled_ratio
    assert value["final_checkpoint_seconds"] == {"baseline": [None, None], "optimized": [None, None]}


def test_final_summary_requires_predeclared_complete_unique_pair_set():
    runs = [measured(0, "baseline"), measured(0, "optimized")]
    case = {**bench.CASES["l2-batch"], "count": 10, "dimension": 2, "pairs": 1}
    with pytest.raises(ValueError, match="predeclared"):
        bench.summary(runs, final=True)
    with pytest.raises(ValueError, match="duplicate"):
        bench.summary([*runs, runs[0]], {"l2-batch": case}, final=True)
    with pytest.raises(ValueError, match="incomplete/failed"):
        bench.summary(runs[:1], {"l2-batch": case}, final=True)
    with pytest.raises(ValueError, match="missing measured"):
        bench.summary(runs, {"l2-batch": {**case, "pairs": 2}}, final=True)
    with pytest.raises(ValueError, match="missing measured"):
        bench.summary(runs, {"l2-batch": case, "missing-case": case}, final=True)
    with pytest.raises(ValueError, match="unexpected measured"):
        bench.summary(runs, {"other": case}, final=True)
    value = bench.summary(runs, {"l2-batch": case}, final=True)["l2-batch"]
    assert value["expected_pairs"] == 1
    assert value["complete"] is True
    assert value["decision_metrics"]["final_primary_target_at_least_1_10"] is None


@pytest.mark.parametrize("change", [
    lambda run: run.update(index=True),
    lambda run: run.update(index=-1),
    lambda run: run.update(index=1),
    lambda run: run.update(variant="candidate"),
    lambda run: run.update(status="error"),
    lambda run: run["native"].update(build_seconds=0),
    lambda run: run["native"].update(build_seconds=float("nan")),
    lambda run: run["native"].update(build_seconds=float("inf")),
    lambda run: run["native"].update(build_seconds=True),
    lambda run: run["native"]["query_windows"][0].pop("p50_ms"),
    lambda run: run["native"]["query_windows"][0].update(p95_ms=float("nan")),
    lambda run: run["native"]["query_windows"][0].update(p50_ms=0.2),
    lambda run: run["native"]["query_windows"][0].update(milliseconds=[0]),
    lambda run: run["native"]["query_windows"][0].update(milliseconds=[0.1, 0.1]),
    lambda run: run["native"]["query_windows"].pop(),
    lambda run: run["native"].pop("sqlite_compile_options"),
    lambda run: run["native"]["query_lifecycle"].update(connection="reopened"),
    lambda run: run["native"]["pager"]["effective"].update(cache_size=-8192),
    lambda run: run["native"].update(dimension=3),
])
def test_final_summary_rejects_bad_identity_missing_metrics_and_settings(change):
    runs = [measured(0, "baseline"), measured(0, "optimized")]
    change(runs[1])
    cases = {"l2-batch": {**bench.CASES["l2-batch"], "count": 10, "dimension": 2, "pairs": 1, "queries": 1}}
    with pytest.raises(ValueError):
        bench.summary(runs, cases, final=True)


def test_required_ratio_rejects_nonfinite_arithmetic_even_with_finite_inputs():
    with pytest.raises(ValueError, match="paired ratio"):
        bench.positive_ratio(1e308, 1e-308)
    with pytest.raises(ValueError):
        bench.positive_ratio(0, 1)


@pytest.mark.parametrize("pager_tuned", [False, True])
def test_primary_target_requires_five_pairs_default_pager_and_all_query_metrics(pager_tuned):
    controls = {"journal_mode": "delete", "synchronous": "full"}
    if pager_tuned:
        controls["cache_size"] = -8192
    runs = []
    for index in range(5):
        for variant in bench.VARIANTS:
            native = result(storage="file", controls=controls, count=10000, dimension=128, milliseconds=[100] * 100)
            if variant == "optimized":
                native["query_windows"][1]["milliseconds"] = [105] * 100
                native["query_windows"][2]["milliseconds"] = [106] * 100
            runs.append(measured(index, variant, 1.1 if variant == "baseline" else 1, name="l2-file", native=native))
    cases = {"l2-file": {**bench.CASES["l2-file"], "dimension": 128, "queries": 100, "pairs": 5,
                          "controls": {key: bench.normalized_controls(controls) for key in bench.VARIANTS}}}
    value = bench.summary(runs, cases, final=True)["l2-file"]
    metrics = value["decision_metrics"]
    assert metrics["primary_default_cache_configuration"] is (not pager_tuned)
    assert metrics["final_primary_target_at_least_1_10"] is (None if pager_tuned else True)
    assert metrics["screening_primary_strict_improvement"] is None
    assert metrics["regression_flags_over_1_05"] == ["query/100/p50_ms", "query/100/p95_ms"]
    assert value["query_percentiles"]["50"]["p50_ms"]["median_paired_candidate_over_baseline"] == 1.05


@pytest.mark.parametrize("behavior", ["duplicate-json", "scalar-json"])
def test_native_json_parser_rejects_duplicate_fields_and_nonobjects(tmp_path, behavior):
    stdout = '{"build_seconds":1,"build_seconds":2}' if behavior == "duplicate-json" else "[]"
    record = {"id": "json", "variant": "baseline"}
    case = {**bench.CASES["l2-batch"], "count": 10, "dimension": 2, "queries": 1}
    inputs = {"files": {key: {"path": key + ".bin"} for key in ("vectors", "queries")}}
    with mock.patch.object(bench.subprocess, "run", return_value=subprocess.CompletedProcess("host", 0, stdout, "details")):
        bench.execute_run(record, Path("host"), Path("library"), case, inputs, tmp_path, 600, [list(range(10))], None)
    assert record["status"] == "error"
    saved = json.loads((tmp_path / "json.json").read_text())
    assert saved["stdout"] == stdout
    assert saved["stderr"] == "details"


def main_args(tmp_path):
    executable = tmp_path / "host"
    library = tmp_path / "library"
    executable.write_bytes(b"host")
    library.write_bytes(b"library")
    return ["--executable", str(executable), "--baseline", str(library), "--optimized", str(library),
            "--output-dir", str(tmp_path / "output"), "--case", "l2-batch", "--dimension", "2", "--count", "10",
            "--queries", "1", "--pairs", "2", "--warmups", "0"]


def successful_fake_run(record, executable, library, case, inputs, output, timeout, truth, reference, **kwargs):
    native = result(storage=case["storage"], controls=case["controls"][record["variant"]],
                    count=case["count"], dimension=case["dimension"], timeout=timeout, milliseconds=[0.1] * case["queries"])
    native.update(mode=case["mode"], metric=case["metric"])
    bench.validate_result(native, case, truth, native["database"], native["snapshot"], case["controls"][record["variant"]], timeout)
    record.update(status="ok", native=native, graph={"sha256": "same"})


def test_main_validates_final_declared_pairs_and_provenance(tmp_path, capsys):
    args = main_args(tmp_path)
    with mock.patch.object(bench, "execute_run", side_effect=successful_fake_run) as execute:
        assert bench.main(args) == 0
    report = json.loads((tmp_path / "output/results.json").read_text())
    assert len(report["runs"]) == 6
    assert execute.call_count == 6
    assert report["status"] == "ok"
    assert report["summary_validated"] is True
    assert report["summary"]["l2-batch"]["complete"] is True
    assert report["summary"]["l2-batch"]["validation"] == "final"
    assert report["context"]["settings"]["query_percentile_method"] == "linear"


@pytest.mark.parametrize("mutation", ["host", "library", "vectors", "queries"])
def test_main_detects_immutable_native_and_input_mutation_between_runs(tmp_path, capsys, mutation):
    args = main_args(tmp_path)
    calls = []
    def execute(record, executable, library, case, inputs, *remaining, **kwargs):
        successful_fake_run(record, executable, library, case, inputs, *remaining, **kwargs)
        path = {"host": executable, "library": library}.get(mutation)
        if path is None:
            path = Path(inputs["files"][mutation]["path"])
            path.chmod(0o644)
        path.write_bytes(b"changed")
        calls.append(record["id"])
    with mock.patch.object(bench, "execute_run", side_effect=execute):
        assert bench.main(args) == 1
    report = json.loads((tmp_path / "output/results.json").read_text())
    assert len(calls) == 1
    assert report["status"] == "error"
    assert report["summary_validated"] is False
    assert "immutable input/native executable/library changed" in report["error"]
    assert report["runs"][0]["status"] == "ok"  # Prior evidence is not discarded/reclassified.


@pytest.mark.parametrize("extra", [
    ["--page-size", "1000"], ["--cache-size=0"], ["--mmap-size=-1"], ["--cache-spill", "maybe"],
    ["--journal-mode", "off"], ["--synchronous", "normal"], ["--temp-store", "bogus"],
    ["--optimized-cache-size=-8192"], ["--comparison", "pager-cache"],
])
def test_main_invalid_settings_stop_before_any_process_or_output(tmp_path, extra):
    args = main_args(tmp_path)
    with mock.patch.object(bench.subprocess, "run") as native, pytest.raises(SystemExit) as error:
        bench.main([*args, *extra])
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()
    native.assert_not_called()


def test_pager_cli_requires_same_library_hash_and_file_cases(tmp_path, capsys):
    args = main_args(tmp_path)
    args[args.index("l2-batch")] = "l2-file"
    other = tmp_path / "other-library"
    other.write_bytes(b"different")
    with pytest.raises(SystemExit) as error:
        bench.main([*args, "--optimized", str(other), "--comparison", "pager-cache", "--optimized-cache-size=-8192"])
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


def test_pager_cli_selfchecks_and_final_summary_keep_differences_separate(tmp_path, capsys):
    args = main_args(tmp_path)
    args[args.index("l2-batch")] = "l2-file"
    with mock.patch.object(bench, "execute_run", side_effect=successful_fake_run):
        assert bench.main([*args, "--comparison", "pager-cache", "--optimized-cache-size=-8192"]) == 0
    report = json.loads((tmp_path / "output/results.json").read_text())
    assert [(run["phase"], run["variant"]) for run in report["runs"][:4]] == [("selfcheck", "baseline")] * 2 + [("selfcheck", "optimized")] * 2
    value = report["summary"]["l2-file"]
    assert value["configuration"]["comparison"] == "pager-cache"
    assert value["configuration"]["pager"]["baseline"]["effective"]["cache_size"] == -2000
    assert value["configuration"]["pager"]["optimized"]["effective"]["cache_size"] == -8192
    assert value["decision_metrics"]["primary_default_cache_configuration"] is False
    assert report["context"]["libraries"]["baseline"]["sha256"] == report["context"]["libraries"]["optimized"]["sha256"]


def test_execute_run_propagates_controls_and_rejects_inherited_tracing(tmp_path):
    controls = bench.normalized_controls({"page_size": 4096, "cache_size": -8192, "cache_spill": "on",
                                          "journal_mode": "memory", "synchronous": "full", "mmap_size": 0,
                                          "temp_store": "default"})
    case = {**bench.CASES["l2-batch"], "count": 10, "dimension": 2, "queries": 1, "controls": {"baseline": controls}}
    inputs = {"files": {key: {"path": key + ".bin"} for key in ("vectors", "queries")}}
    record = {"id": "controlled", "variant": "baseline"}
    def process(command, **kwargs):
        environment = kwargs["env"]
        assert kwargs["timeout"] == 600
        assert "VECTORLITE_BENCH_TRACING" not in environment
        assert "VECTORLITE_BENCH_UNKNOWN" not in environment
        assert environment["VECTORLITE_BENCH_CACHE_SIZE"] == "-8192"
        assert environment["VECTORLITE_BENCH_CACHE_SPILL"] == "on"
        assert environment["VECTORLITE_BENCH_SYNCHRONOUS"] == "full"
        path = Path(environment["VECTORLITE_BENCH_SNAPSHOT"])
        snapshot(path)
        value = result(controls=controls)
        value.update(database=environment["VECTORLITE_BENCH_DATABASE"], snapshot=str(path))
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")
    with mock.patch.dict(bench.os.environ, {"VECTORLITE_BENCH_TRACING": "1", "VECTORLITE_BENCH_UNKNOWN": "1"}), mock.patch.object(bench.subprocess, "run", side_effect=process):
        bench.execute_run(record, Path("host"), Path("library"), case, inputs, tmp_path, 600, [list(range(10))], None)
    assert record["status"] == "ok", record.get("error")
    saved = json.loads((tmp_path / "controlled.json").read_text())
    assert saved["native"]["query_windows"][0]["p50_ms"] == 0.1
    assert saved["native"]["query_windows"][0]["p95_ms"] == 0.1
    original = (tmp_path / "controlled.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        bench.execute_run(record, Path("host"), Path("library"), case, inputs, tmp_path, 600, [list(range(10))], None)
    assert (tmp_path / "controlled.json").read_bytes() == original



@pytest.mark.parametrize("behavior", ["graph", "labels", "distance_bits", "settings"])
def test_execute_run_preserves_evidence_when_baseline_comparison_fails(tmp_path, behavior):
    controls = bench.normalized_controls()
    native = result()
    validate(native)
    reference_path = snapshot(tmp_path / "reference.sqlite")
    reference = {"native": copy.deepcopy(native), "graph": graph(reference_path)}
    record = {"id": "mismatch", "variant": "baseline"}
    case = {**bench.CASES["l2-batch"], "count": 10, "dimension": 2, "queries": 1}
    inputs = {"files": {key: {"path": key + ".bin"} for key in ("vectors", "queries")}}
    destination = tmp_path / "mismatch-snapshot.sqlite"
    snapshot(destination)
    native.update(snapshot=str(destination))
    if behavior == "graph":
        update(destination, "UPDATE v_diskann_nodes SET vector=? WHERE node_id=1", (struct.pack("<2f", 0.5, 1),))
    elif behavior == "labels":
        native["query_windows"][0]["labels"][0].reverse()
    elif behavior == "distance_bits":
        native["query_windows"][0]["distance_bits"][0][0] = 1
    else:
        native["sqlite_source_id"] = "different SQLite build"
    response = subprocess.CompletedProcess("host", 0, json.dumps(native), "comparison details")
    with mock.patch.object(bench.subprocess, "run", return_value=response):
        bench.execute_run(record, Path("host"), Path("library"), case, inputs, tmp_path, 600, [list(range(10))], reference)
    assert record["status"] == "error"
    saved = json.loads((tmp_path / "mismatch.json").read_text())
    assert saved["stdout"] == response.stdout
    assert saved["stderr"] == "comparison details"
    assert saved["native"]["query_windows"]
    assert destination.exists()
    assert {"graph": "graph/metadata", "labels": "ordered query", "distance_bits": "distance_bits", "settings": "provenance drift"}[behavior] in saved["error"]


def test_main_cannot_finish_with_only_an_observed_partial_pair(tmp_path, capsys):
    args = main_args(tmp_path)
    order = [("selfcheck", 0, "baseline"), ("selfcheck", 1, "baseline"), ("measured", 0, "baseline")]
    with mock.patch.object(bench, "execute_run", side_effect=successful_fake_run), mock.patch.object(bench, "run_order", return_value=order):
        assert bench.main(args) == 1
    report = json.loads((tmp_path / "output/results.json").read_text())
    assert report["status"] == "error"
    assert report["summary_validated"] is False
    assert "missing measured pairs" in report["error"]
    assert report["summary"] == {}


def test_missing_query_samples_cannot_be_hidden_by_recomputed_percentiles():
    runs = [measured(0, "baseline", native=result(milliseconds=[1, 2])),
            measured(0, "optimized", native=result(milliseconds=[2, 4]))]
    runs[1]["native"]["query_windows"][0]["milliseconds"] = [2]
    runs[1]["native"]["query_windows"][0].update(p50_ms=2, p95_ms=2)
    cases = {"l2-batch": {**bench.CASES["l2-batch"], "count": 10, "dimension": 2, "pairs": 1, "queries": 2}}
    with pytest.raises(ValueError, match="cardinality"):
        bench.summary(runs, cases, final=True)


def test_summary_medians_and_large_integer_metrics_must_remain_finite():
    assert bench.finite_number(10 ** 400) is False
    with pytest.raises(ValueError, match="summary median"):
        bench.metric_median([1e308, 1e308])
    with pytest.raises(ValueError, match="paired ratio"):
        bench.positive_ratio(1e-308, 1e308)


def test_nondefault_page_mmap_temp_or_spill_cannot_satisfy_primary_target():
    controls = {"page_size": 8192, "mmap_size": 0, "temp_store": "memory", "cache_spill": "off"}
    runs = [measured(index, variant, 1.2 if variant == "baseline" else 1, name="l2-file",
                     native=result(storage="file", controls=controls, count=10000, dimension=128, milliseconds=[1] * 100))
            for index in range(5) for variant in bench.VARIANTS]
    cases = {"l2-file": {**bench.CASES["l2-file"], "dimension": 128, "queries": 100,
                          "controls": {key: bench.normalized_controls(controls) for key in bench.VARIANTS}}}
    value = bench.summary(runs, cases, final=True)["l2-file"]
    assert value["median_paired_speedup"] == 1.2
    assert value["decision_metrics"]["primary_default_cache_configuration"] is False
    assert value["decision_metrics"]["final_primary_target_at_least_1_10"] is None


@pytest.mark.parametrize("pairs,speedup,screening,final", [(3, 1, False, None), (3, 1.001, True, None), (5, 1.099, None, False)])
def test_section6_screening_and_final_thresholds_are_distinct(pairs, speedup, screening, final):
    runs = [measured(index, variant, speedup if variant == "baseline" else 1, name="l2-file",
                     native=result(storage="file", count=10000, dimension=128, milliseconds=[1] * 100))
            for index in range(pairs) for variant in bench.VARIANTS]
    cases = {"l2-file": {**bench.CASES["l2-file"], "dimension": 128, "queries": 100, "pairs": pairs}}
    metrics = bench.summary(runs, cases, final=True)["l2-file"]["decision_metrics"]
    assert metrics["screening_primary_strict_improvement"] is screening
    assert metrics["final_primary_target_at_least_1_10"] is final
    assert metrics["regression_flags_over_1_05"] == []
