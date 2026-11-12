#!/usr/bin/env python3
"""Matched release A/B DiskANN builds with strict baseline repeatability checks.

Bounded datasets and the float64 oracle are materialized outside native timing.
This is not a physical beyond-RAM or cold-disk benchmark. Failed/partial evidence
is retained; a graph, query, validation or process failure stops further builds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import struct
import subprocess
import sys

import numpy as np

CASES = {
    "l2-batch": {"metric": "l2", "mode": "batch8", "storage": "memory", "count": 3000, "pairs": 5},
    "l2-single": {"metric": "l2", "mode": "single", "storage": "memory", "count": 3000, "pairs": 3},
    "cosine-batch": {"metric": "cosine", "mode": "batch8", "storage": "memory", "count": 3000, "pairs": 3},
    "l2-file": {"metric": "l2", "mode": "batch8", "storage": "file", "count": 10000, "pairs": 3},
}
OPTIONS = {"degree": 32, "build_list_size": 100, "search_list_size": 64,
           "alpha": float(np.float32(1.2)), "cache_bytes": 67108864, "max_visits": 65536}
META_COLUMNS = ["singleton", "format_version", "instance_id", "descriptor", "revision",
                "next_node_id", "live_count", "deleted_count", "entrypoints"]
NODE_COLUMNS = ["node_id", "public_rowid", "state", "vector", "neighbors"]


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def save_json(path: Path, value) -> None:
    # Each process result is saved before parsing/verification, then updated.
    with path.open("w", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, allow_nan=False)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())


def fresh_output(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise ValueError(f"output directory already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def oracle(vectors: np.ndarray, queries: np.ndarray, metric: str) -> list[list[int]]:
    data = vectors.astype(np.float64)
    if metric == "cosine":
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        data = np.divide(data, norms, out=np.zeros_like(data), where=norms != 0)
    elif metric != "l2":
        raise ValueError("unsupported oracle metric")
    expected = []
    for vector in queries.astype(np.float64):
        if metric == "l2":
            distances = np.square(data - vector).sum(axis=1)
        else:
            norm = np.linalg.norm(vector)
            unit = vector / norm if norm else np.zeros_like(vector)
            distances = 1.0 - data @ unit
        expected.append(np.argsort(distances, kind="stable")[:10].tolist())
    return expected


def make_inputs(output: Path, count: int, queries: int, dimension: int = 128) -> dict:
    if type(dimension) is not int or not 1 <= dimension <= 3000:
        raise ValueError("dimension must be in [1,3000]")
    rng = np.random.RandomState(42)
    vectors = rng.random((count, dimension)).astype("<f4")
    query_vectors = rng.random((queries, dimension)).astype("<f4")
    suffix = f"{count}" if dimension == 128 else f"{count}-{dimension}d"
    paths = [output / f"vectors-{suffix}.bin", output / f"queries-{suffix}.bin"]
    for path, data in zip(paths, [vectors, query_vectors]):
        with path.open("xb") as destination:
            destination.write(data.tobytes())
        path.chmod(0o444)
    return {"vectors": vectors, "queries": query_vectors,
            "files": {"vectors": fingerprint(paths[0]), "queries": fingerprint(paths[1])}}


def hash_value(digest, value) -> None:
    """Typed, length-framed encoding; no sorting/normalization of BLOB payloads."""
    if value is None:
        digest.update(b"n")
    elif type(value) is int:
        digest.update(b"i" + struct.pack("<q", value))
    elif isinstance(value, bytes):
        digest.update(b"b" + struct.pack("<Q", len(value)) + value)
    elif isinstance(value, str):
        data = value.encode("utf-8")
        digest.update(b"s" + struct.pack("<Q", len(data)) + data)
    else:
        raise ValueError(f"unexpected SQLite value type: {type(value)}")


def check_columns(connection, table: str, names: list[str], types: list[str]) -> None:
    columns = connection.execute(f"PRAGMA table_xinfo({table})").fetchall()
    if [(row[1], row[2], row[6]) for row in columns] != list(zip(names, types, [0] * len(names))):
        raise ValueError(f"unexpected schema: {table}")


def graph_snapshot(path: Path, count: int, dimension: int, metric: str) -> dict:
    """Read a consistent, read-only snapshot and stream ordered graph records."""
    connection = sqlite3.connect(path.resolve(strict=True).as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ValueError("snapshot quick_check failed")
        check_columns(connection, "v_diskann_meta", META_COLUMNS,
                      ["INTEGER", "INTEGER", "BLOB", "BLOB"] + ["INTEGER"] * 4 + ["BLOB"])
        check_columns(connection, "v_diskann_nodes", NODE_COLUMNS,
                      ["INTEGER", "INTEGER", "INTEGER", "BLOB", "BLOB"])
        rows = connection.execute("SELECT * FROM v_diskann_meta").fetchall()
        if len(rows) != 1:
            raise ValueError("metadata must have one row")
        meta = dict(zip(META_COLUMNS, rows[0]))
        for key in ["singleton", "format_version", "revision", "next_node_id", "live_count", "deleted_count"]:
            if type(meta[key]) is not int:
                raise ValueError(f"invalid metadata integer: {key}")
        if (meta["singleton"], meta["format_version"], meta["next_node_id"],
                meta["live_count"], meta["deleted_count"]) != (1, 3, count + 1, count, 0) or meta["revision"] < 1:
            raise ValueError("metadata counters/version inconsistent")
        if not isinstance(meta["instance_id"], bytes) or len(meta["instance_id"]) != 16:
            raise ValueError("invalid instance_id")
        if meta["entrypoints"] != struct.pack("<Q", 0) or not isinstance(meta["descriptor"], bytes):
            raise ValueError("invalid metadata BLOBs")
        descriptor = json.loads(meta["descriptor"])
        expected = {"backend": "diskann", "format_version": 3, "dimension": dimension,
                    "vector_type": "float32", "metric": "cosine" if metric == "cosine" else "l2_squared",
                    "normalize": metric == "cosine", "adapter_version": "0.60",
                    "atomic_write_protocol": "guard-two-row-v1", "deletion_policy": "retained-topology-v1",
                    "maintenance": "streamed-rebuild-v1", "options": OPTIONS}
        if descriptor != expected:
            raise ValueError("descriptor/settings mismatch")
        digest = hashlib.sha256()
        for key in META_COLUMNS:
            if key != "instance_id":  # The ONLY deliberately excluded metadata field.
                hash_value(digest, key)
                hash_value(digest, meta[key])
        counts = [0, 0, 0]
        node_count = 0
        for node, public, state, vector, adjacency in connection.execute(
                "SELECT node_id,public_rowid,state,vector,neighbors FROM v_diskann_nodes ORDER BY node_id"):
            if type(node) is not int or node != node_count or type(state) is not int or state not in (0, 2):
                raise ValueError("invalid node ID/state in insertion-only fixture")
            if (node == 0 and (state != 2 or public is not None)) or (node != 0 and (
                    state != 0 or type(public) is not int or public != node - 1)):
                raise ValueError("invalid frozen/live public identity")
            if not isinstance(vector, bytes) or len(vector) != dimension * 4 or not np.isfinite(np.frombuffer(vector, dtype="<f4")).all():
                raise ValueError("invalid vector payload")
            if not isinstance(adjacency, bytes) or len(adjacency) % 8 or len(adjacency) > 41 * 8:
                raise ValueError("invalid adjacency payload")
            neighbors = struct.unpack(f"<{len(adjacency) // 8}Q", adjacency)
            if len(set(neighbors)) != len(neighbors) or any(other > count or other == node for other in neighbors):
                raise ValueError("invalid adjacency identity")
            for value in (node, public, state, vector, adjacency):
                hash_value(digest, value)
            counts[state] += 1
            node_count += 1
        if counts != [count, 0, 1] or node_count != count + 1:
            raise ValueError("frozen/live counts inconsistent")
        if connection.execute("SELECT singleton,value FROM v_diskann_txn ORDER BY singleton").fetchall() != [(1, 0), (2, 0)]:
            raise ValueError("atomic guard is not quiescent")
        if connection.execute("SELECT count(*) FROM v_diskann_rebuild").fetchone() != (0,):
            raise ValueError("rebuild staging is not empty")
        stable = {key: value.hex() if isinstance(value, bytes) else value
                  for key, value in meta.items() if key != "instance_id"}
        return {"sha256": digest.hexdigest(), "nodes": node_count, "live": counts[0],
                "deleted": counts[1], "frozen": counts[2], "stable_metadata": stable}
    finally:
        connection.close()


def finite_number(value, positive=False) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def validate_result(result: dict, case: dict, truth: list[list[int]], database: str, snapshot: str) -> None:
    expected = {key: case[key] for key in ("mode", "metric", "storage", "count", "dimension")}
    expected.update(database=database, snapshot=snapshot, tracing=False, max_visits=65536)
    if not isinstance(result, dict) or result.get("tracing") is not False or any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("native result provenance mismatch")
    if not isinstance(result.get("sqlite"), str) or not finite_number(result.get("build_seconds"), positive=True):
        raise ValueError("invalid native build time/SQLite provenance")
    windows = result.get("query_windows")
    if not isinstance(windows, list) or not all(isinstance(row, dict) for row in windows) or [row.get("window") for row in windows] != [10, 50, 100]:
        raise ValueError("invalid query windows")
    for row in windows:
        labels, times, bits = row.get("labels"), row.get("milliseconds"), row.get("distance_bits")
        if not isinstance(labels, list) or len(labels) != len(truth) or not isinstance(times, list) or len(times) != len(truth):
            raise ValueError("invalid query cardinality")
        for values in labels:
            if not isinstance(values, list) or len(values) != 10 or any(type(value) is not int or not 0 <= value < case["count"] for value in values) or len(set(values)) != 10:
                raise ValueError("invalid query labels")
        if not isinstance(bits, list) or len(bits) != len(truth):
            raise ValueError("invalid distance_bits cardinality")
        for values in bits:
            if not isinstance(values, list) or len(values) != 10 or any(
                type(value) is not int or not 0 <= value <= 0xFFFFFFFF or not math.isfinite(struct.unpack("<f", struct.pack("<I", value))[0])
                for value in values
            ):
                raise ValueError("invalid query distance_bits")
        if not all(finite_number(value) for value in times):
            raise ValueError("invalid query latency")
        row["recall_at_10"] = statistics.mean(len(set(found) & set(expected)) / 10
                                             for found, expected in zip(labels, truth))


def verify_same(reference: dict, current: dict) -> None:
    if current["graph"] != reference["graph"]:
        raise ValueError("graph/metadata mismatch against first baseline (including adjacency order)")
    for before, after in zip(reference["native"]["query_windows"], current["native"]["query_windows"]):
        if before["labels"] != after["labels"] or before["recall_at_10"] != after["recall_at_10"]:
            raise ValueError("ordered query output/recall mismatch against first baseline")
        if before["distance_bits"] != after["distance_bits"]:
            raise ValueError("ordered query distance_bits mismatch against first baseline")


def run_order(pairs: int, warmups: int):
    # These two verification builds precede ANY candidate build and measurements.
    yield "selfcheck", 0, "baseline"
    yield "selfcheck", 1, "baseline"
    for index in range(warmups):
        for variant in ("baseline", "optimized"):
            yield "warmup", index, variant
    for index in range(pairs):
        for variant in (("baseline", "optimized") if index % 2 == 0 else ("optimized", "baseline")):
            yield "measured", index, variant


def summary(runs: list[dict]) -> dict:
    measured = [run for run in runs if run["phase"] == "measured" and run.get("status") == "ok"]
    result = {}
    for name in dict.fromkeys(run["case"] for run in measured):
        selected = [run for run in measured if run["case"] == name]
        times = {variant: [run["native"]["build_seconds"] for run in selected if run["variant"] == variant]
                 for variant in ("baseline", "optimized")}
        ratios = []
        for index in sorted(set(run["index"] for run in selected)):
            pair = {run["variant"]: run["native"]["build_seconds"] for run in selected if run["index"] == index}
            if len(pair) == 2:
                ratios.append(pair["baseline"] / pair["optimized"])
        result[name] = {"build_seconds": times,
                        "median_build_seconds": {key: statistics.median(values) if values else None for key, values in times.items()},
                        "paired_speedups": ratios, "median_paired_speedup": statistics.median(ratios) if ratios else None,
                        "paired_speedup_min": min(ratios) if ratios else None,
                        "paired_speedup_max": max(ratios) if ratios else None}
    return result


def text(value) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else value or ""


def execute_run(record: dict, executable: Path, library: Path, case: dict, inputs: dict,
                output: Path, timeout: float, truth: list[list[int]], reference: dict | None) -> None:
    run_path = output / (record["id"] + ".json")
    snapshot = output / (record["id"] + "-snapshot.sqlite")
    database = str(output / (record["id"] + "-database.sqlite")) if case["storage"] == "file" else ":memory:"
    command = [str(executable), str(library), case["mode"], str(case["count"]), str(case["dimension"]),
               inputs["files"]["vectors"]["path"], inputs["files"]["queries"]["path"], str(case["queries"])]
    settings = {"VECTORLITE_BENCH_METRIC": case["metric"], "VECTORLITE_BENCH_DATABASE": database,
                "VECTORLITE_BENCH_SNAPSHOT": str(snapshot), "VECTORLITE_BENCH_MAX_VISITS": "65536",
                "VECTORLITE_BENCH_TIMEOUT_SECONDS": str(timeout)}
    # Override inherited benchmark controls; timed runs never enable tracing.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("VECTORLITE_BENCH_")}
    environment.update(settings)
    record.update(command=command, environment=settings, started_at=datetime.now(timezone.utc).isoformat(), status="started")
    save_json(run_path, record)
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False, env=environment)
        record.update(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr, status="process-complete")
        save_json(run_path, record)
        if process.returncode:
            raise ValueError(f"native process exited {process.returncode}")
        native = json.loads(process.stdout)
        # Reject NaN/Infinity (including exponent overflow) before retaining a
        # parsed object, so malformed output cannot poison JSON error evidence.
        json.dumps(native, allow_nan=False)
        validate_result(native, case, truth, database, str(snapshot))
        record["native"] = native
        record["graph"] = graph_snapshot(snapshot, case["count"], case["dimension"], case["metric"])
        record["snapshot"] = fingerprint(snapshot)
        if case["storage"] == "file":
            original = graph_snapshot(Path(database), case["count"], case["dimension"], case["metric"])
            if original != record["graph"]:
                raise ValueError("file database and backup graph differ")
        if reference is not None:
            verify_same(reference, record)
        record["status"] = "ok"
    except subprocess.TimeoutExpired as error:
        record.update(status="error", error=f"native timeout after {timeout}s", stdout=text(error.stdout), stderr=text(error.stderr))
    except Exception as error:
        record.update(status="error", error=f"{type(error).__name__}: {error}")
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_json(run_path, record)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "baseline", "optimized", "output-dir"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--case", action="append", choices=list(CASES), help="repeatable; default: all four cases")
    result.add_argument("--dimension", type=int, default=128, help="vector dimension in [1,3000]; default: 128")
    result.add_argument("--pairs", type=int, help="override measured pair count for every selected case")
    result.add_argument("--count", type=int, help="override vector count for every selected case")
    result.add_argument("--queries", type=int, default=100)
    result.add_argument("--warmups", type=int, default=1, help="per variant/case; excludes two baseline selfchecks")
    result.add_argument("--timeout", type=float, default=120, help="hard timeout per native process, seconds")
    return result


def main(argv=None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    if not 1 <= args.dimension <= 3000:
        arguments.error("dimension must be in [1,3000]")
    if args.count is not None and not 10 <= args.count <= 100000:
        arguments.error("count must be in [10,100000]")
    if (args.pairs is not None and not 1 <= args.pairs <= 100) or not 0 <= args.warmups <= 10 or not 1 <= args.queries <= 1000 or not math.isfinite(args.timeout) or args.timeout <= 0:
        arguments.error("invalid pairs, warmups, queries or timeout")
    try:
        executable = args.executable.resolve(strict=True)
        libraries = {key: getattr(args, key).resolve(strict=True) for key in ("baseline", "optimized")}
        if not executable.is_file() or not all(path.is_file() for path in libraries.values()):
            raise ValueError("executable and libraries must be regular files")
        output = fresh_output(args.output_dir)
    except (ValueError, OSError) as error:
        arguments.error(str(error))
    context = {"command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
               "working_directory": str(Path.cwd()), "started_at": datetime.now(timezone.utc).isoformat(),
               "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
               "machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
               "sqlite_python_verifier": sqlite3.sqlite_version,
               "runner": fingerprint(Path(__file__).resolve()),
               "executable": fingerprint(executable), "libraries": {key: fingerprint(path) for key, path in libraries.items()},
               "settings": {"dimension": args.dimension, "seed": 42, "queries": args.queries, "k": 10,
                            "query_windows": [10, 50, 100], "diskann": OPTIONS, "tracing": False,
                            "warmups_per_variant_case": args.warmups, "baseline_selfchecks_per_case": 2,
                            "query_warmup_passes": 1, "query_timed_passes": 1, "timeout_seconds": args.timeout},
               "notes": ["Same executable and immutable inputs for both libraries; alternating AB/BA measured pairs.",
                         "Native timer: BEGIN + INSERT(s) + COMMIT, including statement prepare/finalize.",
                         "Data generation/oracle, queries, snapshot backup and streaming hashing are outside build timing.",
                         "Graph hash excludes ONLY random instance_id; adjacency ordering is significant.",
                         "Small samples and warm OS caches; no physical cold-disk, beyond-RAM or cross-platform execution claim."]}
    report = {"schema_version": 1, "context": context, "cases": {}, "runs": [], "summary": {}, "status": "running"}
    report_path = output / "results.json"
    save_json(report_path, report)
    inputs_by_count = {}
    try:
        for name in dict.fromkeys(args.case or CASES):
            case = {**CASES[name], "dimension": args.dimension, "queries": args.queries}
            if args.count is not None:
                case["count"] = args.count
            if args.pairs is not None:
                case["pairs"] = args.pairs
            if case["count"] not in inputs_by_count:
                inputs_by_count[case["count"]] = make_inputs(output, case["count"], args.queries, args.dimension)
            inputs = inputs_by_count[case["count"]]
            truth = oracle(inputs["vectors"], inputs["queries"], case["metric"])
            oracle_suffix = f"{name}" if args.dimension == 128 else f"{name}-{args.dimension}d"
            oracle_path = output / (oracle_suffix + "-oracle.json")
            save_json(oracle_path, {"metric": case["metric"], "precision": "float64", "k": 10, "labels": truth})
            report["cases"][name] = {**case, "inputs": inputs["files"], "oracle": fingerprint(oracle_path)}
            reference = None
            for phase, index, variant in run_order(case["pairs"], args.warmups):
                record = {"id": f"{name}-{phase}-{index:02d}-{variant}", "case": name,
                          "phase": phase, "index": index, "variant": variant}
                report["runs"].append(record)
                save_json(report_path, report)
                execute_run(record, executable, libraries[variant], case, inputs, output, args.timeout, truth, reference)
                report["summary"] = summary(report["runs"])
                save_json(report_path, report)
                if record["status"] != "ok":
                    raise ValueError(f"{record['id']}: {record['error']}")
                if reference is None:
                    reference = record
                print(f"{record['id']}: {record['native']['build_seconds']:.6f}s graph={record['graph']['sha256']}", flush=True)
            # Detect any accidental/external mutation of supposedly immutable files.
            if {key: fingerprint(Path(value["path"])) for key, value in inputs["files"].items()} != inputs["files"]:
                raise ValueError("input files changed during case")
        if fingerprint(executable) != context["executable"] or any(fingerprint(path) != context["libraries"][key] for key, path in libraries.items()):
            raise ValueError("native executable or library changed during comparison")
        report["status"] = "ok"
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        print(report["error"], file=sys.stderr, flush=True)
    save_json(report_path, report)
    print(json.dumps({"status": report["status"], "output": str(output), "summary": report["summary"]}, indent=2), flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
