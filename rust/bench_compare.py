"""Micro-benchmark: Rust-ported vectorlite .so vs C++ vectorlite .so.

Both extensions share the same hnswlib + Highway `ops` core, so this isolates
the cost of the virtual-table glue layer (Rust vs C++). Identical workloads,
same seed, same SQLite (host stdlib sqlite3).
"""
import gc
import os
import sqlite3
import statistics
import sys
import time

import numpy as np

DIMS = [128, 512, 1536]
N = 5000          # vectors inserted per table
Q = 1000          # knn queries
K = 10
EF_CONSTRUCTION = 100
M = 30
SEED = 12345
REPEATS = 5       # measured repetitions per phase (median reported)


def load(path):
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.enable_load_extension(True)
    conn.load_extension(path)
    return conn


def bench_one(path, dim, vectors, queries):
    conn = load(path)
    cur = conn.cursor()
    cur.execute(
        f"create virtual table t using vectorlite("
        f"v float32[{dim}] l2, hnsw(max_elements={N}, ef_construction={EF_CONSTRUCTION}, M={M}, random_seed={SEED}))"
    )
    rows = [(i, vectors[i].tobytes()) for i in range(N)]

    gc.disable()
    t0 = time.perf_counter()
    cur.executemany("insert into t(rowid, v) values (?, ?)", rows)
    insert_s = time.perf_counter() - t0

    qparams = [(queries[i].tobytes(), K) for i in range(Q)]
    t0 = time.perf_counter()
    for qb, k in qparams:
        cur.execute(
            "select rowid, distance from t where knn_search(v, knn_param(?, ?))",
            (qb, k),
        ).fetchall()
    query_s = time.perf_counter() - t0
    gc.enable()

    cur.execute("drop table t")
    conn.close()
    return insert_s, query_s


def main():
    backends = {
        "C++ ": sys.argv[1],
        "Rust": sys.argv[2],
    }
    for name, p in backends.items():
        print(f"{name}: {p}  ({os.path.getsize(p)/1e6:.2f} MB)")
    print()

    rng = np.random.default_rng(SEED)
    header = f"{'dim':>5} | {'backend':<5} | {'insert µs/vec':>14} | {'query µs/query':>15} | {'insert tot s':>12} | {'query tot s':>11}"
    print(header)
    print("-" * len(header))

    for dim in DIMS:
        vectors = np.float32(rng.random((N, dim)))
        queries = np.float32(rng.random((Q, dim)))
        samples = {name: ([], []) for name in backends}
        # Warm up both backends once.
        for path in backends.values():
            bench_one(path, dim, vectors, queries)
        # Interleave backends within each repeat and alternate the starting
        # backend so monotonic thermal/cache drift cancels out between them.
        items = list(backends.items())
        for r in range(REPEATS):
            order = items if r % 2 == 0 else list(reversed(items))
            for name, path in order:
                i_s, q_s = bench_one(path, dim, vectors, queries)
                samples[name][0].append(i_s)
                samples[name][1].append(q_s)
        results = {}
        for name in backends:
            i_med = statistics.median(samples[name][0])
            q_med = statistics.median(samples[name][1])
            results[name] = (i_med, q_med)
            print(
                f"{dim:>5} | {name:<5} | {i_med/N*1e6:>14.2f} | {q_med/Q*1e6:>15.2f} | "
                f"{i_med:>12.3f} | {q_med:>11.3f}"
            )
        # ratio (Rust / C++)
        ci, cq = results["C++ "]
        ri, rq = results["Rust"]
        print(
            f"{'':>5} | ratio | {ri/ci:>13.3f}x | {rq/cq:>14.3f}x | "
            f"{'(Rust/C++)':>12} |"
        )
        print()


if __name__ == "__main__":
    main()
