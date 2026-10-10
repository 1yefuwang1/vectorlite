#!/usr/bin/env python3
"""Time-budgeted native batch INSERT comparison on the shared 3,000-vector data.

Build each variant once, save its result immediately, and reuse its index for all
query windows. This is an in-memory overhead comparison, not a beyond-RAM test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


def fingerprint(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="NEW directory; partial completed measurements are retained")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-visits", type=int, default=65536,
                        help="Explicit DiskANN per-chunk work limit")
    parser.add_argument("--timeout", type=float, default=120,
                        help="Hard per-process time limit in seconds")
    parser.add_argument("--modes", nargs="+", choices=["hnsw", "single", "batch8", "batch32"],
                        default=["hnsw", "single", "batch8", "batch32"])
    args = parser.parse_args()
    if not 10 <= args.count <= 100000 or not 1 <= args.dimension <= 3000:
        parser.error("count or dimension outside the bounded native runner range")
    if not 1 <= args.queries <= 1000 or args.timeout <= 0 or not 100 <= args.max_visits <= 2**32 - 1:
        parser.error("queries, timeout, or max-visits is invalid")
    executable = args.executable.resolve(strict=True)
    extension = args.extension.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("output directory already exists")
    output.mkdir(parents=True)

    # Match the older benchmark's RandomState data for 128D, including generating
    # the other default dimensions before producing the query vectors.
    rng = np.random.RandomState(args.seed)
    if args.dimension == 128 and args.count == 3000:
        vectors = np.float32(rng.random((args.count, 128)))
        for dimension in [512, 1536, 3000]:
            _unused = np.float32(rng.random((args.count, dimension)))
        del _unused
    else:
        vectors = np.float32(rng.random((args.count, args.dimension)))
    queries = np.float32(rng.random((args.queries, args.dimension)))
    vector_path, query_path = output / "vectors.bin", output / "queries.bin"
    vector_path.write_bytes(vectors.astype("<f4", copy=False).tobytes())
    query_path.write_bytes(queries.astype("<f4", copy=False).tobytes())
    # Exact float64 L2 oracle avoids candidate-ranking approximations. Only these
    # small test matrices are materialized; this runner makes no memory claim.
    expected = []
    vectors64 = vectors.astype(np.float64)
    for query in queries:
        distances = np.square(vectors64 - query.astype(np.float64)).sum(axis=1)
        expected.append(np.argsort(distances, kind="stable")[:10].tolist())
    context = {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "extension": fingerprint(extension), "executable": fingerprint(executable),
        "dataset": {"count": args.count, "dimension": args.dimension,
                    "queries": args.queries, "seed": args.seed, "metric": "l2", "k": 10,
                    "vectors_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
                    "queries_sha256": hashlib.sha256(query_path.read_bytes()).hexdigest()},
        "storage": ":memory:", "timeout_seconds_per_mode": args.timeout,
        "builds_per_mode": 1, "query_warmup_passes": 1, "query_timed_passes": 1,
        "hnsw": {"M": 30, "ef_construction": 100},
        "diskann": {"degree": 32, "build_list_size": 100, "search_list_size": 64,
                    "cache_bytes": 67108864, "max_visits": args.max_visits},
        "notes": ["Native C INSERT host; no Python per-vector binding overhead.",
                  "Equal query windows do not imply matched recall.",
                  "No physical cold-cache, durable commit, or beyond-RAM claim.",
                  "Per-query times include stepping/materializing rowids, exclude binding/reset.",
                  "Single build/pass per mode; host load and graph topology are not controlled."],
    }
    (output / "context.json").write_text(json.dumps(context, indent=2) + "\n")
    results = []
    failure = False
    environment = {**os.environ, "VECTORLITE_BENCH_MAX_VISITS": str(args.max_visits)}
    for mode in args.modes:
        command = [str(executable), str(extension), mode, str(args.count), str(args.dimension),
                   str(vector_path), str(query_path), str(args.queries)]
        try:
            process = subprocess.run(command, capture_output=True, text=True,
                                     timeout=args.timeout, check=False, env=environment)
        except subprocess.TimeoutExpired as error:
            failure = True
            result = {"mode": mode, "error": "per-mode timeout", "timeout": args.timeout}
            if error.stderr:
                result["stderr"] = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr
        else:
            if process.returncode:
                failure = True
                result = {"mode": mode, "error": "native process failed",
                          "returncode": process.returncode, "stderr": process.stderr,
                          "stdout": process.stdout}
            else:
                result = json.loads(process.stdout)
                for row in result["query_windows"]:
                    labels = row["labels"]
                    if len(labels) != args.queries or any(len(set(values)) != 10 for values in labels):
                        raise ValueError("Invalid query output cardinality")
                    row["recall_at_10"] = float(np.mean([
                        len(set(predicted) & set(truth)) / 10
                        for predicted, truth in zip(labels, expected)]))
                    row["mean_ms"] = float(np.mean(row["milliseconds"]))
                    row["p50_ms"] = float(np.quantile(row["milliseconds"], 0.5))
                    row["p95_ms"] = float(np.quantile(row["milliseconds"], 0.95))
                print(f"{mode}: build {result['build_seconds']:.3f}s; " + "; ".join(
                    f"window {row['window']}: {row['mean_ms']:.3f}ms recall {row['recall_at_10']:.1%}"
                    for row in result["query_windows"]), flush=True)
        if "error" in result:
            print(f"{mode}: {result}", flush=True)
        (output / f"{mode}.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        (output / "results.json").write_text(json.dumps({"context": context, "results": results}, indent=2) + "\n")
        # A failed native mode may expose a correctness issue; don't repeatedly
        # rerun expensive variants after the first failure.
        if "error" in result:
            break
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
