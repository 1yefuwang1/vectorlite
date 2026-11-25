#!/usr/bin/env python3
"""Streamed DiskANN measurements, with each measured stage in a fresh process.

No corpus-sized Python/NumPy matrix is constructed. The optional storage-linear
fixture writes private shadow tables ONLY to test lazy storage/working memory;
it is not a supported import API, production ANN build, or ANN quality benchmark.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import platform
import random
import sqlite3
import struct
import subprocess
import sys
import time
from typing import Iterable, Iterator, Sequence

TABLE = "vectorbench"
MANIFEST = "diskann_benchmark_manifest"
SQLITE_CACHE_BYTES = 8 * 1024 * 1024
QUERY_SEED_XOR = 0x53A91D07
UPDATE_SEED_XOR = 0xA78E241B
MAX_BATCH_SIZE = 128


def pack_vector(values: Iterable[float]) -> bytes:
    data = array.array("f", values)
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def unpack_vector(blob: bytes, dimension: int) -> array.array:
    if len(blob) != dimension * 4:
        raise ValueError("stored vector has an unexpected length")
    data = array.array("f")
    data.frombytes(blob)
    if sys.byteorder != "little":
        data.byteswap()
    return data


def random_vectors(count: int, dimension: int, seed: int) -> Iterator[bytes]:
    rng = random.Random(seed)
    for _ in range(count):
        yield pack_vector(rng.uniform(-1.0, 1.0) for _ in range(dimension))


def batches(rows: Iterable, size: int) -> Iterator[list]:
    if not 1 <= size <= MAX_BATCH_SIZE:
        raise ValueError("batch size must be between 1 and 128")
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    if not 0 <= fraction <= 1:
        raise ValueError("percentile fraction must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary(seconds: Sequence[float]) -> dict:
    milliseconds = [value * 1000 for value in seconds]
    return {"samples": len(seconds), "p50_ms": percentile(milliseconds, 0.5),
            "p95_ms": percentile(milliseconds, 0.95)}


def rss_bytes(value: int | float, system: str) -> int:
    """POSIX ru_maxrss is bytes on macOS and KiB on Linux."""
    if system == "Darwin":
        return int(value)
    if system == "Linux":
        return int(value * 1024)
    raise ValueError(f"unverified ru_maxrss units on {system}")


def peak_rss() -> dict:
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                                   ctypes.POINTER(Counters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters),
                                            counters.cb):
                raise ctypes.WinError(ctypes.get_last_error())
            return {"bytes": int(counters.PeakWorkingSetSize),
                    "source": "GetProcessMemoryInfo.PeakWorkingSetSize"}
        except (OSError, AttributeError) as error:
            return {"bytes": None, "source": "unavailable", "error": str(error)}
    try:
        import resource
        maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"bytes": rss_bytes(maximum, platform.system()),
                "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss"}
    except (ImportError, ValueError) as error:
        return {"bytes": None, "source": "unverified", "error": str(error)}


def query_memory_limit(mebibytes: int | None) -> dict:
    if mebibytes is None:
        return {"requested_bytes": None, "enforced": False}
    if sys.platform != "linux":
        raise ValueError("--query-memory-limit-mib supports Linux RLIMIT_AS only; "
                         "macOS address-space limits are not verified")
    import resource
    limit = mebibytes * 1024 * 1024
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY and hard < limit:
        raise ValueError("requested query memory limit exceeds the inherited hard limit")
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    if resource.getrlimit(resource.RLIMIT_AS)[0] != limit:
        raise RuntimeError("RLIMIT_AS limit was not applied")
    return {"requested_bytes": limit, "enforced": True,
            "mechanism": "Linux RLIMIT_AS (virtual address space, not an RSS limit)"}


def database_sizes(database: Path) -> dict:
    def size(path: Path) -> int:
        return path.stat().st_size if path.exists() else 0
    return {"database_bytes": size(database), "wal_bytes": size(Path(str(database) + "-wal")),
            "shm_bytes": size(Path(str(database) + "-shm"))}


def extension_fingerprint(extension: Path) -> dict:
    digest = hashlib.sha256()
    with extension.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(extension), "bytes": extension.stat().st_size,
            "sha256": digest.hexdigest()}


def connect(config: dict, *, readonly: bool = False) -> sqlite3.Connection:
    database = Path(config["database"])
    mode = "ro" if readonly else "rw"
    connection = sqlite3.connect(database.as_uri() + f"?mode={mode}", uri=True,
                                 isolation_level=None, timeout=30)
    try:
        connection.execute("PRAGMA cache_size=-8192")
        connection.execute("PRAGMA mmap_size=0")
        connection.enable_load_extension(True)
        connection.load_extension(config["extension"])
        connection.enable_load_extension(False)
        return connection
    except BaseException:
        connection.close()
        raise


def read_manifest(connection: sqlite3.Connection) -> dict:
    row = connection.execute(f"SELECT config FROM {MANIFEST} WHERE singleton=1").fetchone()
    if row is None:
        raise ValueError("database does not contain this benchmark's manifest")
    config = json.loads(row[0])
    if config.get("manifest_version") != 1:
        raise ValueError("unsupported benchmark manifest version")
    return config


def dataset_config(connection: sqlite3.Connection, runtime: dict) -> dict:
    saved = read_manifest(connection)
    if "storage_descriptor" not in saved or saved.get("storage_format_version") != 5:
        raise ValueError("benchmark manifest lacks a Format 5 descriptor")
    descriptor(connection, saved)
    # An explicit extension can be changed to compare compatible implementations.
    return {**saved, **{key: runtime[key] for key in (
        "database", "extension", "queries", "k", "warm_runs", "query_memory_limit_mib",
        "numpy_ground_truth")}}


def write_batch(connection: sqlite3.Connection, sql: str, rows: list) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.executemany(sql, rows)
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def linear_vector(coordinate: float, dimension: int) -> bytes:
    return struct.pack("<f", coordinate) + bytes((dimension - 1) * 4)


def linear_neighbors(rowid: int, count: int, entries: dict[int, int]) -> bytes:
    """Format 5 slots: 16-byte (native_local_id, entry_id) pairs.

    Native IDs are dense 0..count-1, with the frozen root at count. The root has
    no entry id, so its slot carries 0; every live target resolves through the
    published current map.
    """
    node = rowid - 1
    neighbors = [count if node == 0 else node - 1]
    if rowid < count:
        neighbors.append(node + 1)
    return b"".join(
        struct.pack("<QQ", target, 0 if target == count else entries[target + 1])
        for target in neighbors)


def descriptor(connection: sqlite3.Connection, config: dict) -> tuple[int, dict]:
    version, raw = connection.execute(
        f"SELECT format_version,descriptor FROM {TABLE}_diskann_meta WHERE singleton=1").fetchone()
    value = json.loads(raw)
    if (version != 5 or value.get("format_version") != 5
            or value.get("backend") != "diskann" or value.get("dimension") != config["dim"]
            or value.get("metric") != ("l2_squared" if config["metric"] == "l2" else "cosine")
            or value.get("maintenance") != "deferred-oneshot-v1"):
        raise ValueError("unsupported or mismatched Format 5 storage descriptor")
    options = value.get("options", {})
    for key in ("degree", "build_list_size", "search_list_size", "cache_bytes", "max_visits"):
        if options.get(key) != config[key]:
            raise ValueError(f"storage descriptor option mismatch: {key}")
    if options.get("alpha") != struct.unpack("<f", struct.pack("<f", config["alpha"]))[0]:
        raise ValueError("storage descriptor option mismatch: alpha")
    if "storage_descriptor" in config and (config.get("storage_format_version") != version
                                             or config["storage_descriptor"] != value):
        raise ValueError("benchmark manifest and storage descriptor differ")
    return version, value


def eligible_rows_sql() -> str:
    return (f"FROM {TABLE}_diskann_meta m JOIN {TABLE}_diskann_graph_nodes g "
            "ON g.generation_id=m.published_generation AND g.state=0 "
            f"JOIN {TABLE}_diskann_current c ON c.entry_id=g.entry_id "
            f"JOIN {TABLE}_diskann_entries e ON e.entry_id=c.entry_id AND e.public_rowid=c.public_rowid ")


def eligible_count(connection: sqlite3.Connection) -> int:
    return connection.execute("SELECT count(*) " + eligible_rows_sql()).fetchone()[0]


def maintenance(config: dict, started_at: float | None = None) -> dict:
    # A separate file-backed worker owns the fenced candidate throughout the cycle.
    worker = connect(config)
    timings = {}
    def status():
        return json.loads(worker.execute("SELECT vectorlite_index_status(?,?)", ("main", TABLE)).fetchone()[0])
    def control(operation, options):
        worker.execute(f"INSERT INTO {TABLE}(operation,path) VALUES(?,?)", (operation, json.dumps(options)))
    def timed(name, action):
        start = time.perf_counter()
        value = action()
        timings[name] = time.perf_counter() - start
        return value
    try:
        timed("prepare_seconds", lambda: control("index_prepare", {}))
        job = status()["current_job"]
        key = {"job_id": job["job_id"], "fence": job["fence"]}
        # Every step is bounded; a nonprogressing or unexpected phase fails closed.
        # Each successful step can be byte-limited to a single record; include
        # the frozen root and final phase transition in the worst-case bound.
        limit = job["cutoff_entry_id"] + 4
        def steps(operation, phase):
            for _ in range(limit):
                current = status()["current_job"]
                if current is None or current["phase"] != phase:
                    return
                control(operation, {**key, **token, "max_records": 256,
                                    "max_bytes": min(1024 * 1024, config["cache_bytes"])})
            raise RuntimeError(f"{operation} failed to complete within bounded steps")
        token = {}
        timed("snapshot_seconds", lambda: steps("index_snapshot_step", 0))
        if status()["current_job"]["phase"] != 1:
            raise RuntimeError("snapshot did not reach build phase")
        timed("claim_seconds", lambda: control("index_claim_build", key))
        candidate = timed("build_seconds", lambda: json.loads(worker.execute(
            "SELECT vectorlite_index_build(?,?,?)", ("main", TABLE,
            json.dumps({**key, "memory_budget_bytes": max(128 * 1024 * 1024, config["cache_bytes"])}))).fetchone()[0]))
        token = {"candidate_token": candidate["candidate_token"]}
        timed("import_seconds", lambda: steps("index_import_step", 2))
        timed("validate_seconds", lambda: steps("index_validate_step", 3))
        if status()["current_job"]["phase"] != 4:
            raise RuntimeError("candidate validation did not reach publication")
        timed("publish_seconds", lambda: control("index_publish", {**key, **token}))
        elapsed_to_publication = time.perf_counter() - started_at if started_at is not None else None
        result = status()
        if result["current_job"] is not None or result["published_generation"] is None:
            raise RuntimeError("maintenance did not publish")
        timed("cleanup_seconds", lambda: control(
            "index_cleanup_step", {"max_bytes": min(1024 * 1024, config["cache_bytes"])}))
        return {"phase_seconds": timings, "published_generation": result["published_generation"],
                "eligible_count": result["eligible_count"], "candidate_population": candidate["population"],
                "elapsed_to_publication_seconds": elapsed_to_publication}
    finally:
        worker.close()


def build_storage_fixture(connection: sqlite3.Connection, config: dict) -> None:
    """Unsupported synthetic Format 5 generation, exclusively for storage tests."""
    count, dimension = config["count"], config["dim"]
    descriptor(connection, config)
    if connection.execute(f"SELECT singleton,value FROM {TABLE}_diskann_txn ORDER BY singleton").fetchall() != [(1, 0), (2, 0)]:
        raise ValueError("bootstrap carrier is not canonical")
    revision = connection.execute(f"SELECT revision FROM {TABLE}_diskann_meta").fetchone()[0]
    digest = bytes(range(32))  # Fixture hashes are opaque, not validation evidence.
    connection.execute(f"INSERT INTO {TABLE}_diskann_generations VALUES(1,1,1,?,?,?,?,?,?)",
                       (revision, digest, digest[::-1], count, count, count))
    entries = {rowid: entry for entry, rowid in connection.execute(
        f"SELECT entry_id,public_rowid FROM {TABLE}_diskann_current").fetchall()}
    if len(entries) != count:
        raise ValueError("fixture canonical population does not match manifest")
    cursor = connection.execute(f"SELECT entry_id,public_rowid FROM {TABLE}_diskann_current ORDER BY public_rowid")
    seen = 0
    while rows := cursor.fetchmany(config["batch_size"]):
        graph = []
        for entry, rowid in rows:
            seen += 1
            if rowid != seen:
                raise ValueError("fixture requires contiguous canonical rowids")
            graph.append((1, rowid - 1, entry, rowid, 0,
                          linear_neighbors(rowid, count, entries), None))
        write_batch(connection, f"INSERT INTO {TABLE}_diskann_graph_nodes VALUES(?,?,?,?,?,?,?)", graph)
    connection.execute(f"INSERT INTO {TABLE}_diskann_graph_nodes VALUES(1,?,NULL,NULL,2,?,?)",
                       (count, struct.pack("<QQ", 0, entries[1]), linear_vector(0, dimension)))
    connection.execute(f"UPDATE {TABLE}_diskann_meta SET next_generation_id=2,next_job_id=2,published_generation=1")


def build_worker(config: dict) -> dict:
    connection = connect(config)
    try:
        mode = connection.execute("PRAGMA journal_mode=" + config["journal_mode"]).fetchone()[0]
        connection.execute("PRAGMA wal_autocheckpoint=1000")
        options = ",".join(f"{name}={config[name]}" for name in (
            "degree", "build_list_size", "search_list_size", "alpha", "cache_bytes", "max_visits"))
        connection.execute(f"CREATE VIRTUAL TABLE {TABLE} USING vectorlite("
                           f"embedding float32[{config['dim']}] {config['metric']},diskann({options}))")
        vectors = ((linear_vector(i, config["dim"]) for i in range(1, config["count"] + 1))
                   if config["fixture"] == "storage-linear" else
                   random_vectors(config["count"], config["dim"], config["seed"]))
        begin = time.perf_counter()
        for batch in batches(enumerate(vectors, 1), config["batch_size"]):
            write_batch(connection, f"INSERT INTO {TABLE}(rowid,embedding) VALUES(?,?)", batch)
        ingest_seconds = time.perf_counter() - begin
        if config["fixture"] == "storage-linear":
            begin_fixture = time.perf_counter()
            build_storage_fixture(connection, config)
            publication = {"synthetic_fixture_seconds": time.perf_counter() - begin_fixture}
        else:
            publication = maintenance(config, started_at=begin)
        ingest_to_publication_seconds = (publication["elapsed_to_publication_seconds"]
                                         if config["fixture"] == "none" else None)
        storage_format, storage_descriptor = descriptor(connection, config)
        connection.execute(f"CREATE TABLE {MANIFEST}(singleton INTEGER PRIMARY KEY CHECK(singleton=1),config TEXT NOT NULL)")
        saved = {**config, "manifest_version": 1, "storage_format_version": storage_format,
                 "storage_descriptor": storage_descriptor,
                 "generator": "python-random-uniform-float32-v1" if config["fixture"] == "none"
                 else "affine-first-coordinate-linear-storage-v1"}
        connection.execute(f"INSERT INTO {MANIFEST} VALUES(1,?)", (json.dumps(saved),))
        result = {"process_id": os.getpid(), "seconds": ingest_seconds,
                  "seconds_per_vector": ingest_seconds / config["count"], "ingest_seconds": ingest_seconds,
                  "maintenance": publication, "ingest_to_publication_seconds": ingest_to_publication_seconds,
                  "rows": config["count"], "journal_mode": mode,
                  "peak_rss": peak_rss(), "storage_format_version": storage_format,
                  "storage_descriptor": storage_descriptor,
                  "sizes_connection_open": database_sizes(Path(config["database"])),
                  "production_ann_build": config["fixture"] == "none"}
    finally:
        connection.close()
    result["sizes_after_close"] = database_sizes(Path(config["database"]))
    return result

def query_vectors(config: dict) -> Iterator[bytes]:
    if config["fixture"] == "storage-linear":
        for query in range(config["queries"]):
            yield linear_vector(1.25 + query % 8, config["dim"])
    else:
        yield from random_vectors(config["queries"], config["dim"], config["seed"] ^ QUERY_SEED_XOR)


def linear_truth(coordinate: float, count: int, k: int) -> list[int]:
    # Fixture queries deliberately address the reachable start prefix. No N scan.
    limit = min(count, math.ceil(coordinate) + k + 1)
    return heapq.nsmallest(k, range(1, limit + 1), key=lambda row: ((row - coordinate) ** 2, row))


def query_worker(runtime: dict) -> dict:
    enforcement = query_memory_limit(runtime["query_memory_limit_mib"])
    load_begin = time.perf_counter()
    connection = connect(runtime, readonly=True)
    load_seconds = time.perf_counter() - load_begin
    try:
        config = dataset_config(connection, runtime)
        storage_format, description = descriptor(connection, config)
        status = json.loads(connection.execute("SELECT vectorlite_index_status(?,?)", ("main", TABLE)).fetchone()[0])
        live_count = status["eligible_count"]
        if config["fixture"] == "storage-linear" and live_count != config["count"]:
            raise ValueError("linear fixture no longer has its original eligible population")
        k = min(config["k"], max(1, live_count))
        sql = f"SELECT rowid,distance FROM {TABLE} WHERE knn_search(embedding,knn_param(?,?,?))"
        options = json.dumps({"search_list_size": config["search_list_size"]})
        first, warm, predictions, distances = [], [], [], []
        for query, blob in enumerate(query_vectors(config)):
            begin = time.perf_counter()
            rows = connection.execute(sql, (blob, k, options)).fetchall()
            first.append(time.perf_counter() - begin)
            labels = [row[0] for row in rows]
            predictions.append(labels)
            distances.append([row[1] for row in rows])
            if config["fixture"] == "storage-linear":
                coordinate = struct.unpack_from("<f", blob)[0]
                expected = linear_truth(coordinate, live_count, k)
                if labels != expected:
                    raise AssertionError(f"linear fixture nearest-prefix labels differ: {labels} != {expected}")
                for rowid, distance in rows:
                    if not math.isclose(distance, (rowid - coordinate) ** 2, rel_tol=1e-5, abs_tol=1e-4):
                        raise AssertionError("linear fixture squared-distance invariant failed")
        first_rss = peak_rss()
        for _ in range(config["warm_runs"]):
            for blob in query_vectors(config):
                begin = time.perf_counter()
                connection.execute(sql, (blob, k, options)).fetchall()
                warm.append(time.perf_counter() - begin)
        return {"process_id": os.getpid(), "sqlite_version": sqlite3.sqlite_version,
                "load_seconds": load_seconds, "effective_k": min(k, live_count), "live_count": live_count,
                "eligible_count": live_count, "canonical_count": status["canonical_count"],
                "pending_count": status["pending_count"],
                "eligible_coverage": live_count / status["canonical_count"] if status["canonical_count"] else None,
                "storage_format_version": storage_format, "storage_descriptor": description,
                "cache_label": "fresh-process first pass; NOT cold OS cache",
                "first_query_ms": first[0] * 1000,
                "fresh_process_first_pass": latency_summary(first),
                "warm_passes": latency_summary(warm), "first_pass_peak_rss": first_rss,
                "peak_rss": peak_rss(), "memory_limit": enforcement,
                "predictions": predictions, "distances": distances,
                "fixture_affine_invariants_verified": config["fixture"] == "storage-linear",
                "sizes_connection_open": database_sizes(Path(config["database"]))}
    finally:
        connection.close()


def offer_neighbor(heap: list, distance: float, rowid: int, k: int) -> None:
    entry = (-distance, -rowid)
    if len(heap) < k:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def truth_labels(heap: list) -> list[int]:
    return [-rowid for _, rowid in sorted(heap, reverse=True)]


def recall(predicted: Sequence[Sequence[int]], expected: Sequence[Sequence[int]]) -> float:
    if len(predicted) != len(expected) or not expected or any(not row for row in expected):
        raise ValueError("recall requires equally sized nonempty query/ground-truth lists")
    return sum(len(set(found).intersection(truth)) / len(truth)
               for found, truth in zip(predicted, expected)) / len(expected)


def truth_worker(runtime: dict) -> dict:
    connection = connect(runtime, readonly=True)
    begin = time.perf_counter()
    try:
        config = dataset_config(connection, runtime)
        live_count = eligible_count(connection)
        k = min(config["k"], live_count)
        queries = [unpack_vector(blob, config["dim"]) for blob in query_vectors(config)]
        if config["fixture"] == "storage-linear":
            if live_count != config["count"]:
                raise ValueError("linear fixture no longer has its original eligible population")
            return {"labels": [linear_truth(query[0], live_count, k) for query in queries],
                    "method": "analytic nearest-prefix affine fixture; NOT production ANN recall",
                    "seconds": time.perf_counter() - begin, "peak_rss": peak_rss()}
        if config["metric"] == "cosine":
            normalized = []
            for query in queries:
                norm = math.sqrt(sum(value * value for value in query))
                if not norm:
                    raise ValueError("cosine ground truth requires a nonzero query")
                normalized.append(unpack_vector(pack_vector(value / norm for value in query),
                                                config["dim"]))
            queries = normalized
        heaps = [[] for _ in queries]
        cursor = connection.execute("SELECT c.public_rowid,e.vector " + eligible_rows_sql() + "ORDER BY c.public_rowid")
        numpy = None
        if config["numpy_ground_truth"]:
            import numpy
            query_arrays = [numpy.asarray(query, dtype=numpy.float64) for query in queries]
        visited = 0
        while rows := cursor.fetchmany(config["batch_size"]):
            visited += len(rows)
            if numpy is not None:
                matrix = numpy.frombuffer(b"".join(row[1] for row in rows), dtype="<f4").reshape(
                    len(rows), config["dim"]).astype(numpy.float64)
                for heap, query in zip(heaps, query_arrays):
                    distances = (numpy.sum((matrix - query) ** 2, axis=1)
                                 if config["metric"] == "l2" else 1.0 - matrix @ query)
                    for (rowid, _), distance in zip(rows, distances):
                        offer_neighbor(heap, float(distance), rowid, k)
            else:
                for rowid, blob in rows:
                    vector = unpack_vector(blob, config["dim"])
                    for heap, query in zip(heaps, queries):
                        distance = (sum((x - y) ** 2 for x, y in zip(vector, query))
                                    if config["metric"] == "l2" else
                                    1.0 - sum(x * y for x, y in zip(vector, query)))
                        offer_neighbor(heap, distance, rowid, k)
        if visited != live_count:
            raise ValueError("ground-truth scan and metadata live populations differ")
        return {"labels": [truth_labels(heap) for heap in heaps], "rows_scanned": visited,
                "method": "streamed stored-float32 vectors, float64 distances, per-query top-k heaps",
                "numpy": numpy is not None, "seconds": time.perf_counter() - begin,
                "peak_rss": peak_rss()}
    finally:
        connection.close()


def mutation_worker(runtime: dict) -> dict:
    connection = connect(runtime)
    try:
        config = dataset_config(connection, runtime)
        count = config["count"]
        updates = min(runtime["updates"], count)
        deletes = min(runtime["deletes"], max(0, count - updates - 1))
        update_rows = ((blob, rowid) for rowid, blob in enumerate(
            random_vectors(updates, config["dim"], config["seed"] ^ UPDATE_SEED_XOR), 1))
        begin = time.perf_counter()
        for batch in batches(update_rows, config["batch_size"]):
            write_batch(connection, f"UPDATE {TABLE} SET embedding=? WHERE rowid=?", batch)
        update_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        for batch in batches(((rowid,) for rowid in range(count, count - deletes, -1)),
                             config["batch_size"]):
            write_batch(connection, f"DELETE FROM {TABLE} WHERE rowid=?", batch)
        delete_seconds = time.perf_counter() - begin
        publication = None
        if not runtime["skip_consolidation"]:
            publication = maintenance(config)
        return {"measured": True, "updates": updates, "update_seconds": update_seconds,
                "seconds_per_update": update_seconds / updates if updates else None,
                "deletes": deletes, "delete_seconds": delete_seconds,
                "seconds_per_delete": delete_seconds / deletes if deletes else None,
                "maintenance": publication, "peak_rss": peak_rss(),
                "sizes_connection_open": database_sizes(Path(config["database"]))}
    finally:
        connection.close()


WORKERS = {"build": build_worker, "query": query_worker, "truth": truth_worker,
           "mutate": mutation_worker}


def run_worker(stage: str, config: dict) -> dict:
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", stage,
               "--config-json", json.dumps(config, separators=(",", ":"))]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"{stage} worker exited {completed.returncode}:\n{completed.stderr.strip()}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{stage} worker did not emit valid JSON: {completed.stdout[:1000]}") from error


def measured_queries(config: dict) -> dict:
    result = run_worker("query", config)
    # Exact ground truth is AFTER queries in a separate process: its corpus scan,
    # optional NumPy import, and heaps never inflate measured-query RSS.
    truth = run_worker("truth", config)
    predicted = result.pop("predictions")
    result.pop("distances")
    expected = truth.pop("labels")
    coverage = recall(predicted, expected) if all(expected) else None
    result["recall_at_k"] = coverage if config["fixture"] == "none" else None
    if config["fixture"] == "storage-linear":
        result["fixture_nearest_prefix_coverage_at_k"] = coverage
    result["ground_truth"] = truth
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--extension", type=Path, help="explicit freshly built extension; required")
    result.add_argument("--database", type=Path, help="new SQLite file (existing only with --query-only)")
    result.add_argument("--query-only", action="store_true", help="read an existing benchmark database without mutations")
    result.add_argument("--fixture", choices=("none", "storage-linear"), default="none")
    result.add_argument("--count", type=int, default=10000)
    result.add_argument("--dim", type=int, default=128)
    result.add_argument("--metric", choices=("l2", "cosine"), default="l2")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--queries", type=int, default=32)
    result.add_argument("--k", type=int, default=10)
    result.add_argument("--warm-runs", type=int, default=3)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--degree", type=int, default=16)
    result.add_argument("--build-list-size", type=int, default=64)
    result.add_argument("--search-list-size", type=int, default=64)
    result.add_argument("--alpha", type=float, default=1.2)
    result.add_argument("--cache-bytes", type=int, default=64 * 1024 * 1024)
    result.add_argument("--max-visits", type=int, default=2048)
    result.add_argument("--journal-mode", choices=("wal", "delete"), default="wal")
    result.add_argument("--updates", type=int, default=16)
    result.add_argument("--deletes", type=int, default=16)
    result.add_argument("--skip-consolidation", action="store_true",
                        help="skip post-mutation maintenance; updated rows remain pending and ineligible")
    result.add_argument("--numpy-ground-truth", action="store_true", help="optional bounded-batch NumPy acceleration in truth worker only")
    result.add_argument("--query-memory-limit-mib", type=int, help="Linux query-worker RLIMIT_AS only; virtual address space, not RSS")
    result.add_argument("--require-beyond-ram", action="store_true", help="fail unless DB file exceeds measured query peak RSS and configured working budgets")
    result.add_argument("--output", type=Path, help="write JSON to a NEW file; stdout always contains JSON")
    result.add_argument("--worker", choices=tuple(WORKERS), help=argparse.SUPPRESS)
    result.add_argument("--config-json", help=argparse.SUPPRESS)
    return result


def validate_config(config: dict) -> None:
    for name in ("count", "dim", "queries", "k", "warm_runs", "build_list_size", "search_list_size",
                 "cache_bytes", "max_visits"):
        if config[name] < 1:
            raise ValueError(f"{name} must be positive")
    if not 1 <= config["batch_size"] <= MAX_BATCH_SIZE:
        raise ValueError("batch_size must be between 1 and 128")
    if not 2 <= config["degree"] <= 10000:
        raise ValueError("degree must be between 2 and 10000")
    if config["queries"] > 1024 or config["k"] > 1024 or config["warm_runs"] > 100:
        raise ValueError("queries/k are limited to 1024 and warm_runs to 100")
    if config["dim"] > 65536:
        raise ValueError("dimension is limited to 65536")
    if config["cache_bytes"] < 4096 or config["max_visits"] > 2**32 - 1:
        raise ValueError("invalid DiskANN cache or visit budget")
    if max(config["build_list_size"], config["search_list_size"]) > config["max_visits"]:
        raise ValueError("list sizes must not exceed max_visits")
    if config["search_list_size"] < config["k"]:
        raise ValueError("search_list_size must be at least k")
    if not math.isfinite(config["alpha"]) or config["alpha"] < 1:
        raise ValueError("alpha must be finite and at least one")
    if min(config["updates"], config["deletes"]) < 0:
        raise ValueError("update/delete counts must be nonnegative")
    if config["query_memory_limit_mib"] is not None:
        if config["query_memory_limit_mib"] < 64:
            raise ValueError("query memory limit must be at least 64 MiB")
        if sys.platform != "linux":
            raise ValueError("query memory limits support Linux RLIMIT_AS only; macOS/Windows are unverified")
    if config["fixture"] == "storage-linear" and (config["metric"] != "l2" or config["count"] > 2**24):
        raise ValueError("storage-linear is L2-only and requires count <= 2**24 (exact float32 integer coordinates)")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.worker:
        try:
            config = json.loads(arguments.config_json)
            print(json.dumps(WORKERS[arguments.worker](config), allow_nan=False))
            return 0
        except Exception as error:
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
            return 1
    if arguments.extension is None or arguments.database is None:
        parser().error("--extension and --database are required")
    config = vars(arguments).copy()
    config.pop("worker")
    config.pop("config_json")
    output = config.pop("output")
    extension = arguments.extension.resolve()
    database = arguments.database.resolve()
    config["extension"], config["database"] = str(extension), str(database)
    try:
        validate_config(config)
        if not extension.is_file():
            raise ValueError("explicit extension file does not exist")
        if output is not None and output.exists():
            raise ValueError("output already exists; select a new JSON file")
        if config["query_only"]:
            if not database.is_file():
                raise ValueError("--query-only requires an existing database")
            connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
            try:
                saved = read_manifest(connection)
            finally:
                connection.close()
            for key in ("fixture", "count", "dim", "metric", "seed", "batch_size", "degree",
                        "build_list_size", "search_list_size", "alpha", "cache_bytes", "max_visits"):
                config[key] = saved[key]
            validate_config(config)
        else:
            # Exclusive creation prevents silently modifying a prior database.
            with database.open("xb"):
                pass
        if any(part.lower() in ("dev", "debug") for part in extension.parts):
            print("WARNING: extension path looks like a debug build; timings are not release performance.",
                  file=sys.stderr)
        result = {"schema_version": 1, "configuration": config,
                  "extension": extension_fingerprint(extension), "python": sys.version,
                  "platform": platform.platform(), "sqlite_cache_bytes": SQLITE_CACHE_BYTES,
                  "sqlite_mmap_bytes": 0,
                  "methodology": "fresh subprocesses; never drops OS cache; no corpus-sized in-memory matrix",
                  "fixture_warning": ("UNSUPPORTED private-shadow-table storage fixture; NOT production ANN build/quality"
                                      if config["fixture"] == "storage-linear" else None)}
        result["build"] = None if config["query_only"] else run_worker("build", config)
        result["queries_before_mutation"] = measured_queries(config)
        if not config["query_only"] and config["fixture"] == "none":
            result["mutations"] = run_worker("mutate", config)
            result["queries_after_mutation"] = measured_queries(config)
        else:
            result["mutations"] = {"measured": False, "reason": "query-only or storage-only fixture"}
        sizes = database_sizes(database)
        result["sizes_final"] = sizes
        rss = result["queries_before_mutation"]["peak_rss"]["bytes"]
        budget = config["cache_bytes"] + SQLITE_CACHE_BYTES
        final_queries = result.get("queries_after_mutation", result["queries_before_mutation"])
        result["beyond_ram"] = {"database_bytes": sizes["database_bytes"],
                                "stored_vector_payload_bytes": final_queries["live_count"] * config["dim"] * 4,
                                "query_peak_rss_bytes": rss,
                                "configured_graph_plus_sqlite_cache_bytes": budget,
                                "verified": rss is not None and sizes["database_bytes"] > max(rss, budget),
                                "meaning": "file larger than measured query process peak RSS and configured working budgets; not an OS-cache claim"}
        requirement_failed = config["require_beyond_ram"] and not result["beyond_ram"]["verified"]
        result["beyond_ram"]["required"] = config["require_beyond_ram"]
        encoded = json.dumps(result, indent=2, allow_nan=False)
        if output is not None:
            with output.open("x", encoding="utf-8") as destination:
                destination.write(encoded + "\n")
        print(encoded)
        if requirement_failed:
            print("AssertionError: database does not exceed measured query RSS and configured working budgets",
                  file=sys.stderr)
            return 1
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
