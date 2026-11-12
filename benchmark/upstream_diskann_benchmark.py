#!/usr/bin/env python3
"""Matched-data build measurements: upstream RAM, genuine SSD format, and SQLite.

Fresh subprocesses run sequentially. Different persistence, kernels, roots and
static-builder policies mean these are implementation comparisons, NOT an isolated
storage-overhead ratio. macOS upstream SSD search is validation-only (RAM fallback).
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
import shutil
import statistics
import struct
import subprocess
import time

import numpy as np

from diskann_identity_benchmark import graph_snapshot, oracle

CASES = {"3000x128": (3000, 128), "10000x128": (10000, 128), "10000x768": (10000, 768)}
BACKENDS = ("memory", "ssd", "sqlite-memory", "sqlite-file")
REVISION = "97a828a500848018d8be28c3ea6d5a585b07362f"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def save_json(path: Path, value) -> None:
    with path.open("w") as destination:
        json.dump(value, destination, indent=2, allow_nan=False)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())


def inputs(directory: Path, count: int, dimension: int, queries: int) -> dict:
    rng = np.random.RandomState(42)
    vectors = rng.random((count, dimension)).astype("<f4")
    query_vectors = rng.random((queries, dimension)).astype("<f4")
    paths = {}
    for name, array in (("vectors", vectors), ("queries", query_vectors)):
        raw = directory / f"{name}.f32"
        bigann = directory / f"{name}.bin"
        payload = array.tobytes()
        with raw.open("xb") as destination:
            destination.write(payload)
        with bigann.open("xb") as destination:
            destination.write(struct.pack("<II", *array.shape))
            destination.write(payload)
        require(raw.read_bytes() == bigann.read_bytes()[8:], "input formats differ")
        raw.chmod(0o444)
        bigann.chmod(0o444)
        paths[name] = {"raw": fingerprint(raw), "bigann": fingerprint(bigann)}
    return {"vectors": vectors, "queries": query_vectors, "files": paths,
            "truth": oracle(vectors, query_vectors, "l2")}


def normalized_searches(worker: dict, backend: str) -> list[dict]:
    if backend.startswith("sqlite"):
        return [{"window": item["window"], "ids": item["labels"], "bits": item["distance_bits"]}
                for item in worker["query_windows"]]
    result = []
    for item in worker["result"]["searches"]:
        rows = item["results"]
        require([row["query_index"] for row in rows] == list(range(len(rows))), "query order mismatch")
        result.append({"window": item["search_l"], "ids": [row["ids"] for row in rows],
                       "bits": [row["distance_bits"] for row in rows]})
    return result


def validate_searches(searches: list[dict], dataset: dict) -> list[dict]:
    require([item["window"] for item in searches] == [10, 50, 100], "search windows mismatch")
    data = dataset["vectors"]
    queries = dataset["queries"]
    truth = dataset["truth"]
    quality = []
    for item in searches:
        require(len(item["ids"]) == len(queries) == len(item["bits"]), "query count mismatch")
        hits = 0
        max_error = 0.0
        for q, (labels, bits) in enumerate(zip(item["ids"], item["bits"])):
            require(len(labels) == len(bits) == 10 and len(set(labels)) == 10, "invalid top-k size/duplicates")
            require(all(type(label) is int and 0 <= label < len(data) for label in labels), "invalid label")
            require(all(type(value) is int and 0 <= value <= 0xFFFFFFFF for value in bits), "invalid bits")
            distances = [struct.unpack("<f", struct.pack("<I", value))[0] for value in bits]
            require(all(math.isfinite(value) for value in distances), "non-finite distance")
            require(all(a <= b for a, b in zip(distances, distances[1:])), "unordered distances")
            exact = np.square(data[labels].astype(np.float64) - queries[q].astype(np.float64)).sum(axis=1)
            for actual, expected in zip(distances, exact):
                error = abs(actual - float(expected))
                require(error <= 2e-6 * max(1.0, abs(float(expected))), "distance disagrees with f64 oracle")
                max_error = max(max_error, error)
            hits += len(set(labels) & set(truth[q]))
        quality.append({"window": item["window"], "recall_at_10": hits / (10 * len(queries)),
                        "max_returned_distance_oracle_error": max_error})
    return quality


def validate_memory_integrity(result: dict, count: int) -> dict:
    require(result["vectors_inserted"] == count, "native insertion count mismatch")
    require(result["validated_live_vectors"] == count and result["validated_nodes"] == count + 1,
            "native exhaustive stored-data validation missing")
    reachable = result["reachable_nodes"]
    unreachable = result["unreachable_nodes"]
    ids = result["unreachable_ids"]
    require(type(reachable) is int and 1 <= reachable <= count + 1, "invalid rooted reachability")
    require(type(unreachable) is int and unreachable == count + 1 - reachable, "invalid unreachable count")
    require(len(ids) == unreachable and len(set(ids)) == unreachable, "invalid unreachable IDs")
    require(all(type(value) is int and 0 <= value < count for value in ids), "invalid unreachable live ID")
    storage = result["storage_validation"]
    require(storage["roundtrip_audit_matches"] is True and storage["post_build"] == storage["post_load"],
            "native exhaustive save/load audit differs")
    for field in ("validated_live_vectors", "validated_nodes", "reachable_nodes", "unreachable_nodes", "unreachable_ids"):
        require(storage["post_build"][field] == result[field], "native integrity evidence inconsistent")
    return {"validated_live_vectors": count, "validated_nodes": count + 1,
            "reachable_nodes_including_root": reachable, "unreachable_live_nodes": unreachable,
            "unreachable_live_ids": ids}


def run_one(backend: str, phase: str, repetition: int, directory: Path, dataset: dict,
            executable: Path, sqlite_host: Path, extension: Path, timeout: float) -> dict:
    directory.mkdir()
    n, dim = dataset["vectors"].shape
    query_count = len(dataset["queries"])
    result_path = directory / "worker.json"
    env = os.environ.copy()
    env.update(RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    # Never inherit unrelated benchmark overrides into the controlled SQLite host.
    for name in list(env):
        if name.startswith("VECTORLITE_BENCH_"):
            del env[name]
    if backend in ("memory", "ssd"):
        command = [str(executable), "--mode", backend, "--count", str(n), "--dimension", str(dim),
                   "--workers", "1", "--batch-size", "8", "--vectors", dataset["files"]["vectors"]["bigann"]["path"],
                   "--queries", dataset["files"]["queries"]["bigann"]["path"],
                   "--output-dir", str(directory / "index"), "--output", str(result_path)]
    else:
        database = str(directory / "index.sqlite") if backend == "sqlite-file" else ":memory:"
        snapshot = str(directory / "snapshot.sqlite")
        env.update(VECTORLITE_BENCH_METRIC="l2", VECTORLITE_BENCH_DATABASE=database,
                   VECTORLITE_BENCH_SNAPSHOT=snapshot, VECTORLITE_BENCH_TIMEOUT_SECONDS=str(timeout),
                   VECTORLITE_BENCH_MAX_VISITS="65536")
        command = [str(sqlite_host), str(extension), "batch8", str(n), str(dim),
                   dataset["files"]["vectors"]["raw"]["path"], dataset["files"]["queries"]["raw"]["path"], str(query_count)]
    record = {"backend": backend, "phase": phase, "repetition": repetition, "status": "running",
              "command": command, "directory": str(directory)}
    process_path = directory / "process.json"
    save_json(process_path, record)
    started = time.monotonic()
    try:
        with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
            process = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
        record.update(returncode=process.returncode, process_wall_seconds=time.monotonic() - started)
        require(process.returncode == 0, f"{backend} failed with exit {process.returncode}; see {directory}")
        if backend.startswith("sqlite"):
            worker = json.loads((directory / "stdout.log").read_text())
            save_json(result_path, worker)
            require(worker["mode"] == "batch8" and worker["metric"] == "l2" and worker["tracing"] is False, "SQLite mode mismatch")
            require(worker["count"] == n and worker["dimension"] == dim, "SQLite shape mismatch")
            require(worker["database"] == database and worker["snapshot"] == snapshot, "SQLite storage mismatch")
            primary = worker["build_seconds"]
            graph = graph_snapshot(Path(snapshot), n, dim, "l2")
            structure = {"graph_sha256": graph["sha256"], "nodes": graph["nodes"], "live": graph["live"]}
            output_bytes = Path(snapshot).stat().st_size
            output_size_kind = "SQLite backup snapshot bytes"
            original_database_bytes = Path(database).stat().st_size if backend == "sqlite-file" else None
            reachability = None
        else:
            worker = json.loads(result_path.read_text())
            require(worker["status"] == "complete" and worker["upstream_revision"] == REVISION, "native provenance mismatch")
            require(worker["mode"] == backend and worker["count"] == n and worker["dimension"] == dim, "native shape mismatch")
            require(worker["query_count"] == query_count and worker["metric"] == "l2_squared", "native query/metric mismatch")
            expected_config = {"degree": 32, "build_list_size": 100, "alpha": float(np.float32(1.2)),
                               "seed": 42, "workers": 1, "batch_size": 8}
            require(worker["requested_config"] == expected_config, "native config mismatch")
            if worker["architecture"] == "aarch64":
                require(worker["compiled_neon"] and worker["compiled_dotprod"], "native SIMD backend not enabled")
            primary = worker["result"]["timings_seconds"]["total_ready"]
            reachability = None
            if backend == "memory":
                reachability = validate_memory_integrity(worker["result"], n)
                persistence = worker["result"]["persistence"]
                require(persistence["roundtrip_exact"] is True, "native roundtrip failed")
                paths = [Path(persistence["graph_path"]), Path(persistence["data_path"])]
            else:
                require(worker["result"]["build_quantization"] == "FP" and worker["result"]["search_validation_only"] is True, "SSD mode mismatch")
                paths = [Path(item["path"]) for item in worker["result"]["files"]]
                require(len(paths) == 3, "incomplete SSD artifact set")
            structure = {"files": [{"name": path.name, **fingerprint(path)} for path in paths]}
            output_bytes = sum(path.stat().st_size for path in paths)
            output_size_kind = "native saved index artifact bytes"
            original_database_bytes = None
        require(type(primary) in (float, int) and math.isfinite(primary) and primary > 0, "invalid build time")
        searches = normalized_searches(worker, backend)
        quality = validate_searches(searches, dataset)
        predictions_sha = hashlib.sha256(json.dumps(searches, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if "files" in structure:
            stable_structure = [(item["name"], item["bytes"], item["sha256"]) for item in structure["files"]]
        else:
            stable_structure = structure
        structure_sha = hashlib.sha256(json.dumps(stable_structure, sort_keys=True).encode()).hexdigest()
        record.update(status="validated", build_seconds=primary, quality=quality, output_bytes=output_bytes,
                      output_size_kind=output_size_kind, original_database_bytes=original_database_bytes,
                      reachability=reachability,
                      predictions_sha256=predictions_sha, structure_sha256=structure_sha, structure=structure,
                      worker_result=fingerprint(result_path))
        save_json(process_path, record)
        return record
    except Exception as error:
        record.update(status="failed", error=str(error), process_wall_seconds=time.monotonic() - started)
        save_json(process_path, record)
        raise


def summarize(report: dict) -> dict:
    require(report["status"] == "complete", "cannot summarize incomplete campaign")
    summary = {"schema_version": 1, "cases": []}
    for case in report["cases"]:
        rows = []
        for backend in report["backends"]:
            runs = [run for run in case["runs"] if run["backend"] == backend and run["phase"] == "measured"]
            require(len(runs) == report["repeats"] and all(run["status"] == "validated" for run in runs), "missing measured runs")
            times = [run["build_seconds"] for run in runs]
            rows.append({"backend": backend, "runs": len(runs), "median_seconds": statistics.median(times),
                         "min_seconds": min(times), "max_seconds": max(times),
                         "recall_at_10": {str(item["window"]): item["recall_at_10"] for item in runs[0]["quality"]},
                         "median_output_bytes": statistics.median(run["output_bytes"] for run in runs),
                         "output_size_kind": runs[0].get("output_size_kind"),
                         "median_original_database_bytes": statistics.median(run["original_database_bytes"] for run in runs)
                         if all(run.get("original_database_bytes") is not None for run in runs) else None,
                         "reachability": runs[0].get("reachability"),
                         "same_backend_reachability_exact": all(run.get("reachability") == runs[0].get("reachability") for run in runs),
                         "same_backend_predictions_exact": len({run["predictions_sha256"] for run in runs}) == 1,
                         "same_backend_structure_exact": len({run["structure_sha256"] for run in runs}) == 1})
        summary["cases"].append({"name": case["name"], "count": case["count"], "dimension": case["dimension"], "backends": rows})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-executable", required=True, type=Path)
    parser.add_argument("--sqlite-executable", required=True, type=Path)
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case", action="append", choices=list(CASES))
    parser.add_argument("--backend", action="append", choices=BACKENDS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--count", type=int, help="correctness-smoke override; not a full-size result")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    require(1 <= args.repeats <= 20 and 0 <= args.warmups <= 5, "invalid repetitions")
    require(1 <= args.queries <= 1000 and math.isfinite(args.timeout) and args.timeout > 0, "invalid queries/timeout")
    require(args.count is None or 256 <= args.count <= 100_000, "count override must be 256..100000")
    require(not args.output_dir.exists() and not args.output_dir.is_symlink(), "output directory must be new")
    artifacts = [("upstream", args.upstream_executable.resolve(strict=True)),
                 ("sqlite-host", args.sqlite_executable.resolve(strict=True)),
                 (args.extension.name, args.extension.resolve(strict=True))]
    args.output_dir.mkdir(parents=True)
    out = args.output_dir.resolve()
    frozen = out / "artifacts"
    frozen.mkdir()
    frozen_paths = []
    for name, source in artifacts:
        destination = frozen / name
        shutil.copyfile(source, destination)
        destination.chmod(0o555 if name in ("upstream", "sqlite-host") else 0o444)
        frozen_paths.append(destination)
    backends = list(dict.fromkeys(args.backend or BACKENDS))
    report = {"schema_version": 1, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "revision": REVISION, "repeats": args.repeats,
              "warmups": args.warmups, "queries": args.queries, "count_override": args.count,
              "backends": backends, "artifacts": [fingerprint(path) for path in frozen_paths],
              "environment_controls": {"RAYON_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
              "timing_boundaries": {"memory": "provider setup plus graph insertion; serialization separate",
                                    "ssd": "configuration, constructor and full PQ/graph/disk-layout pipeline",
                                    "sqlite": "BEGIN, batch INSERT and COMMIT; table DDL/input loading excluded"},
              "cases": []}
    save_json(out / "results.json", report)
    try:
        for name in list(dict.fromkeys(args.case or CASES)):
            count, dim = CASES[name]
            if args.count is not None:
                count = args.count
            case_dir = out / name
            case_dir.mkdir()
            dataset = inputs(case_dir, count, dim, args.queries)
            case = {"name": name, "count": count, "dimension": dim, "inputs": dataset["files"], "runs": []}
            report["cases"].append(case)
            references = {}
            for phase, repetitions in (("warmup", args.warmups), ("measured", args.repeats)):
                for repetition in range(repetitions):
                    offset = repetition % len(backends)
                    order = backends[offset:] + backends[:offset]
                    for backend in order:
                        directory = case_dir / f"{phase}-{repetition}-{backend}"
                        run = run_one(backend, phase, repetition, directory, dataset, *frozen_paths, args.timeout)
                        reference = references.setdefault(backend, (run["predictions_sha256"], run["structure_sha256"]))
                        require(reference == (run["predictions_sha256"], run["structure_sha256"]),
                                f"same-backend repeat changed predictions/structure: {backend}; investigate before measuring further")
                        case["runs"].append(run)
                        save_json(out / "results.json", report)
                        print(f"{name} {phase}:{repetition} {backend}: {run['build_seconds']:.6f}s; "
                              f"recall100={run['quality'][-1]['recall_at_10']:.4f}", flush=True)
        for artifact in report["artifacts"]:
            require(fingerprint(Path(artifact["path"])) == artifact, "frozen artifact changed")
        report["status"] = "complete"
        save_json(out / "results.json", report)
        save_json(out / "summary.json", summarize(report))
    except Exception as error:
        report.update(status="failed", error=str(error))
        save_json(out / "results.json", report)
        raise


if __name__ == "__main__":
    main()
