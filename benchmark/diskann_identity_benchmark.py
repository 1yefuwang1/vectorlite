#!/usr/bin/env python3
"""Matched release A/B DiskANN builds with strict baseline repeatability checks.

Bounded datasets and the float64 oracle are materialized outside native timing.
This is not a physical beyond-RAM or cold-disk benchmark. Failed/partial evidence
is retained; a graph, query, validation or process failure stops further builds.

Use --comparison pager-cache with the SAME baseline/optimized library and
--optimized-cache-size=-8192 (baseline defaults) for a separate pager ablation.
Other controls are common to both variants; memory cells retain memory journals.
Each invocation has one dimension/configuration; use separate 128D/768D outputs.
Progress summaries include complete successful pairs only; final summaries require
the entire predeclared pair set. A successful runner is not a ship-gate verdict.
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
    "l2-single": {"metric": "l2", "mode": "single", "storage": "memory", "count": 3000, "pairs": 5},
    "cosine-batch": {"metric": "cosine", "mode": "batch8", "storage": "memory", "count": 3000, "pairs": 5},
    "l2-file": {"metric": "l2", "mode": "batch8", "storage": "file", "count": 10000, "pairs": 5},
}
OPTIONS = {"degree": 32, "build_list_size": 100, "search_list_size": 64,
           "alpha": float(np.float32(1.2)), "cache_bytes": 67108864, "max_visits": 65536}
META_COLUMNS = ["singleton", "format_version", "instance_id", "descriptor", "revision",
                "next_node_id", "live_count", "deleted_count", "entrypoints"]
NODE_COLUMNS = ["node_id", "public_rowid", "state", "vector", "neighbors"]
VARIANTS = ("baseline", "optimized")
WINDOWS = (10, 50, 100)
CONTROL_KEYS = ("page_size", "cache_size", "cache_spill", "journal_mode", "synchronous", "mmap_size", "temp_store")
PAGER_KEYS = (*CONTROL_KEYS, "wal_autocheckpoint")


def normalized_controls(controls: dict | None = None) -> dict:
    controls = {} if controls is None else controls
    if not isinstance(controls, dict) or set(controls) - set(CONTROL_KEYS):
        raise ValueError("unknown pager control")
    value = {key: controls.get(key) for key in CONTROL_KEYS}
    for key in ("page_size", "cache_size", "mmap_size"):
        item = value[key]
        if item is not None and type(item) is not int:
            raise ValueError(f"invalid {key}")
    size = value["page_size"]
    if size is not None and (not 512 <= size <= 65536 or size & (size - 1)):
        raise ValueError("page_size must be a power of two in [512,65536]")
    size = value["cache_size"]
    if size is not None and (size == 0 or not -0x7FFFFFFF <= size <= 0x7FFFFFFF):
        raise ValueError("cache_size must be nonzero signed pages/negative KiB")
    size = value["mmap_size"]
    if size is not None and not 0 <= size <= 0x7FFFFFFFFFFFFFFF:
        raise ValueError("mmap_size must be in [0,INT64_MAX]")
    for key, choices in (("cache_spill", ("on", "off")), ("journal_mode", ("delete", "wal", "memory")),
                         ("synchronous", ("full", "extra")), ("temp_store", ("default", "file", "memory"))):
        if value[key] is not None and (type(value[key]) is not str or value[key] not in choices):
            raise ValueError(f"invalid {key}")
    return value


def cache_argument(text: str) -> int | None:
    if text == "default":
        return None
    if not text or not text.removeprefix("-").isascii() or not text.removeprefix("-").isdigit():
        raise argparse.ArgumentTypeError("cache size must be 'default' or a nonzero signed integer")
    try:
        value = int(text)
        normalized_controls({"cache_size": value})
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def variant_controls(args, storage: str) -> dict:
    common = {key: getattr(args, key) for key in CONTROL_KEYS}
    if storage == "memory":
        common["journal_mode"] = None  # DELETE/WAL selections apply only to file cells.
    controls = {}
    for variant in VARIANTS:
        selected = {**common, "cache_size": getattr(args, variant + "_cache_size", common["cache_size"])}
        controls[variant] = normalized_controls(selected)
    if args.comparison == "code" and controls["baseline"] != controls["optimized"]:
        raise ValueError("code comparison requires identical per-variant settings; use --comparison pager-cache")
    return controls


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
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        return False


def validate_pager(result: dict, controls: dict, storage: str) -> None:
    pager = result.get("pager")
    if not isinstance(pager, dict) or set(pager) != {"requested", "defaults", "effective"}:
        raise ValueError("missing pager settings readback")
    if (not isinstance(pager["requested"], dict) or set(pager["requested"]) != set(CONTROL_KEYS)
            or normalized_controls(pager["requested"]) != controls):
        raise ValueError("requested pager settings mismatch")
    for phase in ("defaults", "effective"):
        values = pager[phase]
        if not isinstance(values, dict) or set(values) != set(PAGER_KEYS):
            raise ValueError("missing effective/default pager settings")
        for key in PAGER_KEYS:
            item = values[key]
            if key == "journal_mode":
                allowed = ("memory",) if storage == "memory" else ("delete", "wal")
                if type(item) is not str or item not in allowed:
                    raise ValueError("invalid journal_mode readback")
            elif key == "mmap_size" and item is None and storage == "memory":
                continue
            elif type(item) is not int:
                raise ValueError(f"invalid {key} readback")
            elif key == "page_size" and (not 512 <= item <= 65536 or item & (item - 1)):
                raise ValueError("invalid page_size readback")
            elif key == "cache_size" and (item == 0 or not -0x7FFFFFFF <= item <= 0x7FFFFFFF):
                raise ValueError("invalid cache_size readback")
            elif key in ("cache_spill", "mmap_size", "wal_autocheckpoint") and not 0 <= item <= 0x7FFFFFFFFFFFFFFF:
                raise ValueError(f"invalid {key} readback")
            elif key == "synchronous" and item not in (0, 1, 2, 3):
                raise ValueError("invalid synchronous readback")
            elif key == "temp_store" and item not in (0, 1, 2):
                raise ValueError("invalid temp_store readback")
    before, after = pager["defaults"], pager["effective"]
    for key in CONTROL_KEYS:
        requested = controls[key]
        if requested is None:
            if key == "cache_spill" and any(controls[name] is not None for name in ("cache_size", "page_size")):
                # SQLite derives its spill threshold from the effective cache/page size.
                matches = (after[key] > 0) == (before[key] > 0)
            else:
                matches = after[key] == before[key]
        elif key == "cache_spill":
            matches = after[key] > 0 if requested == "on" else after[key] == 0
        elif key == "synchronous":
            matches = after[key] == {"full": 2, "extra": 3}[requested]
        elif key == "temp_store":
            matches = after[key] == {"default": 0, "file": 1, "memory": 2}[requested]
        elif key == "mmap_size" and storage == "memory" and after[key] is None:
            matches = requested == 0
        else:
            matches = after[key] == requested
        if not matches:
            raise ValueError(f"effective {key} differs from requested/default settings")
    expected_auto = 0 if after["journal_mode"] == "wal" else before["wal_autocheckpoint"]
    if after["wal_autocheckpoint"] != expected_auto:
        raise ValueError("WAL autocheckpoint settings mismatch")


def query_percentiles(native: dict, *, require_stored: bool = False) -> dict:
    windows = native.get("query_windows")
    if (not isinstance(windows, list) or not all(isinstance(row, dict) for row in windows)
            or [row.get("window") for row in windows] != list(WINDOWS)
            or any(type(row["window"]) is not int for row in windows)):
        raise ValueError("invalid query windows")
    result = {}
    for row in windows:
        samples = row.get("milliseconds")
        if not isinstance(samples, list) or not samples or not all(finite_number(value, positive=True) for value in samples):
            raise ValueError("invalid query latency")
        p50, p95 = np.percentile(np.asarray(samples, dtype=np.float64), [50, 95], method="linear")
        metrics = {"p50_ms": float(p50), "p95_ms": float(p95)}
        if not all(finite_number(value, positive=True) for value in metrics.values()):
            raise ValueError("invalid query percentile")
        if require_stored and any(not finite_number(row.get(key), positive=True) or row[key] != value
                                  for key, value in metrics.items()):
            raise ValueError("missing/incorrect per-process query percentile")
        result[row["window"]] = metrics
    return result


def validate_checkpoint(result: dict) -> None:
    checkpoint = result.get("final_checkpoint")
    keys = {"performed", "mode", "after_queries", "seconds", "rc", "log_frames", "checkpointed_frames"}
    if not isinstance(checkpoint, dict) or set(checkpoint) != keys:
        raise ValueError("missing final checkpoint metadata")
    if result["pager"]["effective"]["journal_mode"] != "wal":
        expected = dict.fromkeys(keys)
        expected.update(performed=False, after_queries=False)
        if checkpoint != expected or checkpoint["performed"] is not False or checkpoint["after_queries"] is not False:
            raise ValueError("unexpected checkpoint for non-WAL run")
    elif (checkpoint["performed"] is not True or checkpoint["after_queries"] is not True or checkpoint["mode"] != "full"
          or not finite_number(checkpoint["seconds"], positive=True)
          or type(checkpoint["rc"]) is not int or checkpoint["rc"] != 0
          or type(checkpoint["log_frames"]) is not int or checkpoint["log_frames"] < 0
          or type(checkpoint["checkpointed_frames"]) is not int or checkpoint["checkpointed_frames"] != checkpoint["log_frames"]):
        raise ValueError("failed/incomplete final FULL checkpoint")


def validate_native_context(result: dict, case: dict, query_count: int, controls: dict,
                            timeout: float | None = None) -> None:
    expected = {key: case[key] for key in ("mode", "metric", "storage", "count", "dimension")}
    expected.update(tracing=False, max_visits=65536)
    if not isinstance(result, dict) or any(type(result.get(key)) is not type(value) or result[key] != value
                                           for key, value in expected.items()):
        raise ValueError("native result provenance mismatch")
    if (not isinstance(result.get("sqlite"), str) or not result["sqlite"]
            or not isinstance(result.get("sqlite_source_id"), str) or not result["sqlite_source_id"]
            or not finite_number(result.get("build_seconds"), positive=True)
            or not finite_number(result.get("timeout_seconds"), positive=True)
            or (timeout is not None and result["timeout_seconds"] != timeout)):
        raise ValueError("invalid native build time/SQLite/timeout provenance")
    compile_options = result.get("sqlite_compile_options")
    if (not isinstance(compile_options, list) or not compile_options
            or any(type(option) is not str or not option for option in compile_options)
            or len(set(compile_options)) != len(compile_options)):
        raise ValueError("invalid SQLite compile options provenance")
    if result.get("ann_settings") != OPTIONS or any(type(result["ann_settings"].get(key)) is not type(value)
                                                   for key, value in OPTIONS.items()):
        raise ValueError("ANN settings mismatch")
    if type(query_count) is not int or not 1 <= query_count <= 1000:
        raise ValueError("invalid query count")
    lifecycle = {"connection": "same_open_post_build", "warmup_passes": 1, "timed_passes": 1,
                 "queries_per_pass": query_count, "k": 10, "windows": list(WINDOWS)}
    if json.dumps(result.get("query_lifecycle"), sort_keys=True) != json.dumps(lifecycle, sort_keys=True):
        raise ValueError("query lifecycle/settings mismatch")
    validate_pager(result, controls, case["storage"])
    validate_checkpoint(result)


def validate_query_rows(result: dict, count: int, query_count: int, *, require_stored: bool = False) -> dict:
    metrics = query_percentiles(result, require_stored=require_stored)
    for row in result["query_windows"]:
        labels, times, bits = row.get("labels"), row["milliseconds"], row.get("distance_bits")
        if not isinstance(labels, list) or len(labels) != query_count or len(times) != query_count:
            raise ValueError("invalid query cardinality")
        for values in labels:
            if not isinstance(values, list) or len(values) != 10 or any(type(value) is not int or not 0 <= value < count for value in values) or len(set(values)) != 10:
                raise ValueError("invalid query labels")
        if not isinstance(bits, list) or len(bits) != query_count:
            raise ValueError("invalid distance_bits cardinality")
        for values in bits:
            if not isinstance(values, list) or len(values) != 10 or any(
                type(value) is not int or not 0 <= value <= 0xFFFFFFFF or not math.isfinite(struct.unpack("<f", struct.pack("<I", value))[0])
                for value in values
            ):
                raise ValueError("invalid query distance_bits")
    return metrics


def validate_result(result: dict, case: dict, truth: list[list[int]], database: str, snapshot: str,
                    controls: dict | None = None, timeout: float | None = None) -> None:
    validate_native_context(result, case, len(truth), normalized_controls(controls), timeout)
    if result.get("database") != database or result.get("snapshot") != snapshot:
        raise ValueError("native database/snapshot provenance mismatch")
    metrics = validate_query_rows(result, case["count"], len(truth))
    for row in result["query_windows"]:
        row.update(metrics[row["window"]])
        row["recall_at_10"] = statistics.mean(len(set(found) & set(expected)) / 10
                                             for found, expected in zip(row["labels"], truth))


def verify_settings(reference: dict, current: dict, *, allow_cache_difference: bool = False) -> None:
    before, after = reference["native"], current["native"]
    keys = ("mode", "count", "dimension", "metric", "storage", "tracing", "max_visits", "timeout_seconds",
            "sqlite", "sqlite_source_id", "sqlite_compile_options", "ann_settings", "query_lifecycle")
    if any(before.get(key) != after.get(key) for key in keys):
        raise ValueError("native settings/SQLite provenance drift")
    if before["pager"]["defaults"] != after["pager"]["defaults"]:
        raise ValueError("connection-default pager settings drift")
    for phase in ("requested", "effective"):
        for key, value in before["pager"][phase].items():
            if allow_cache_difference and key == "cache_size":
                continue
            if (allow_cache_difference and key == "cache_spill" and phase == "effective"
                    and before["pager"]["effective"]["cache_size"] != after["pager"]["effective"]["cache_size"]
                    and value > 0 and after["pager"][phase][key] > 0):
                continue  # SQLite-derived positive threshold, not a changed spill control.
            if after["pager"][phase].get(key) != value:
                raise ValueError(f"{phase} pager settings mismatch: {key}")


def verify_same(reference: dict, current: dict, *, allow_cache_difference: bool = False) -> None:
    verify_settings(reference, current, allow_cache_difference=allow_cache_difference)
    if current["graph"] != reference["graph"]:
        raise ValueError("graph/metadata mismatch against first baseline (including adjacency order)")
    before, after = reference["native"]["query_windows"], current["native"]["query_windows"]
    if len(before) != len(after) or [row["window"] for row in before] != [row["window"] for row in after]:
        raise ValueError("query windows mismatch against first baseline")
    for first, second in zip(before, after):
        if first["labels"] != second["labels"] or first["recall_at_10"] != second["recall_at_10"]:
            raise ValueError("ordered query output/recall mismatch against first baseline")
        if first["distance_bits"] != second["distance_bits"]:
            raise ValueError("ordered query distance_bits mismatch against first baseline")


def run_order(pairs: int, warmups: int, *, variant_selfchecks: bool = False):
    # Establish the baseline before any candidate/configuration build.
    for index in range(2):
        yield "selfcheck", index, "baseline"
    if variant_selfchecks:
        for index in range(2):
            yield "selfcheck", index, "optimized"
    for index in range(warmups):
        for variant in VARIANTS:
            yield "warmup", index, variant
    for index in range(pairs):
        for variant in (VARIANTS if index % 2 == 0 else tuple(reversed(VARIANTS))):
            yield "measured", index, variant


def positive_ratio(numerator: float, denominator: float) -> float:
    if not finite_number(numerator, positive=True) or not finite_number(denominator, positive=True):
        raise ValueError("nonpositive/nonfinite required metric")
    value = numerator / denominator
    if not finite_number(value, positive=True):
        raise ValueError("nonpositive/nonfinite paired ratio")
    return value


def metric_median(values: list) -> float | None:
    if not values:
        return None
    value = statistics.median(values)
    if not finite_number(value, positive=True):
        raise ValueError("nonpositive/nonfinite summary median")
    return value


def ratio_metrics(values: dict) -> dict:
    ratios = [positive_ratio(candidate, baseline) for baseline, candidate in zip(values["baseline"], values["optimized"])]
    median = metric_median(ratios)
    return {"paired_candidate_over_baseline": ratios, "median_paired_candidate_over_baseline": median,
            "paired_ratio_min": min(ratios) if ratios else None, "paired_ratio_max": max(ratios) if ratios else None,
            "regression_over_5_percent": median > 1.05 if median is not None else None}


def summary(runs: list[dict], cases: dict | None = None, *, final: bool = False) -> dict:
    """Never pool query samples across processes or include an incomplete pair.

    Progress may omit not-yet-complete pairs. Final validation REQUIRES the
    predeclared cases/pair counts, and rejects every failed/missing/duplicate run.
    Selfchecks, warmups and diagnostics are deliberately not timing samples.
    """
    if final and (not isinstance(cases, dict) or not cases):
        raise ValueError("final summary requires predeclared cases and pair counts")
    measured = {}
    for run in runs:
        if run.get("phase") != "measured":
            continue
        name, index, variant = run.get("case"), run.get("index"), run.get("variant")
        if not isinstance(name, str) or type(index) is not int or index < 0 or variant not in VARIANTS:
            raise ValueError("invalid measured pair identity")
        if cases is not None and (name not in cases or not 0 <= index < cases[name]["pairs"]):
            raise ValueError("unexpected measured case/pair index")
        identity = (index, variant)
        selected = measured.setdefault(name, {})
        if identity in selected:
            raise ValueError("duplicate measured pair member")
        selected[identity] = run
    result = {}
    names = cases if final else measured
    for name in names:
        selected = measured.get(name, {})
        expected = cases[name]["pairs"] if cases is not None else None
        if expected is not None and (type(expected) is not int or expected < 1):
            raise ValueError("invalid predeclared pair count")
        indices = sorted({index for index, variant in selected})
        if final and indices != list(range(expected)):
            raise ValueError(f"missing measured pairs: {name}")
        complete, incomplete, metrics = [], [], {}
        variant_reference = {}
        for index in indices:
            pair = {variant: selected.get((index, variant)) for variant in VARIANTS}
            if any(run is None or run.get("status") != "ok" for run in pair.values()):
                if final:
                    raise ValueError(f"incomplete/failed measured pair: {name}/{index}")
                incomplete.append(index)
                continue
            for variant, run in pair.items():
                native = run.get("native")
                if not isinstance(native, dict) or not finite_number(native.get("build_seconds"), positive=True):
                    raise ValueError("missing/invalid required build metric")
                declared = cases[name].get("controls", {}).get(variant) if cases is not None else None
                controls = normalized_controls(declared if declared is not None else native.get("pager", {}).get("requested"))
                configuration = cases[name] if cases is not None else native
                query_count = configuration.get("queries", native.get("query_lifecycle", {}).get("queries_per_pass"))
                validate_native_context(native, configuration, query_count, controls, configuration.get("timeout_seconds"))
                metrics[(index, variant)] = validate_query_rows(native, configuration["count"], query_count, require_stored=True)
                if variant in variant_reference:
                    verify_settings(variant_reference[variant], run)
                else:
                    variant_reference[variant] = run
            comparison = cases[name].get("comparison", "code") if cases is not None else "code"
            verify_settings(pair["baseline"], pair["optimized"], allow_cache_difference=comparison == "pager-cache")
            if final:
                verify_same(variant_reference["baseline"], pair["baseline"])
                verify_same(variant_reference["baseline"], pair["optimized"], allow_cache_difference=comparison == "pager-cache")
            complete.append(index)
        if not complete:
            if final:
                raise ValueError(f"no complete measured pairs: {name}")
            continue
        times = {variant: [selected[(index, variant)]["native"]["build_seconds"] for index in complete] for variant in VARIANTS}
        speedups = [positive_ratio(baseline, candidate) for baseline, candidate in zip(times["baseline"], times["optimized"])]
        build_ratios = ratio_metrics(times)
        queries, flags = {}, []
        if build_ratios["regression_over_5_percent"]:
            flags.append("build_seconds")
        for window in WINDOWS:
            queries[str(window)] = {}
            for percentile in ("p50_ms", "p95_ms"):
                values = {variant: [metrics[(index, variant)][window][percentile] for index in complete] for variant in VARIANTS}
                paired = ratio_metrics(values)
                queries[str(window)][percentile] = {"milliseconds": values, **paired}
                if paired["regression_over_5_percent"]:
                    flags.append(f"query/{window}/{percentile}")
        natives = {variant: selected[(complete[0], variant)]["native"] for variant in VARIANTS}
        comparison = cases[name].get("comparison", "code") if cases is not None else "code"
        primary = (comparison == "code" and all(
            (native["storage"], native["metric"], native["mode"], native["count"], native["dimension"]) ==
            ("file", "l2", "batch8", 10000, 128)
            and native["pager"]["effective"]["journal_mode"] == "delete"
            and native["pager"]["effective"]["synchronous"] == 2
            and all(native["pager"]["effective"][key] == native["pager"]["defaults"][key]
                    for key in ("page_size", "cache_size", "cache_spill", "mmap_size", "temp_store"))
            and native["query_lifecycle"]["queries_per_pass"] == 100 and native["timeout_seconds"] == 600
            for native in natives.values()))
        speedup = metric_median(speedups)
        is_complete = expected is not None and complete == list(range(expected)) and not incomplete
        checkpoints = {variant: [selected[(index, variant)]["native"]["final_checkpoint"]["seconds"] for index in complete]
                       for variant in VARIANTS}
        result[name] = {
            "validation": "final" if final else "partial", "complete": is_complete,
            "expected_pairs": expected, "pair_indices": complete, "incomplete_pair_indices": incomplete,
            "configuration": {**{key: natives["baseline"][key] for key in ("mode", "metric", "storage", "count", "dimension")},
                              "comparison": comparison, "pager": {variant: native["pager"] for variant, native in natives.items()}},
            "build_seconds": times, "median_build_seconds": {key: metric_median(values) for key, values in times.items()},
            "paired_speedups": speedups, "median_paired_speedup": speedup,
            "paired_speedup_min": min(speedups), "paired_speedup_max": max(speedups),
            "build_candidate_over_baseline": build_ratios, "query_percentiles": queries,
            "final_checkpoint_seconds": checkpoints,
            "median_final_checkpoint_seconds": {key: metric_median(values) if all(value is not None for value in values) else None
                                                for key, values in checkpoints.items()},
            "decision_metrics": {
                "primary_default_cache_configuration": primary,
                "screening_primary_strict_improvement": speedup > 1 if primary and is_complete and expected == 3 else None,
                "final_primary_target_at_least_1_10": speedup >= 1.10 if primary and is_complete and expected == 5 else None,
                "regression_flags_over_1_05": flags,
                "final_regression_flags": [flag for flag in flags if flag != "build_seconds" or not primary],
                "note": "Per-configuration metrics only: screening drops on any flag; final flags require exactly one complete 5-pair confirmation set. No ship verdict."
            }}
    return result


def unique_object(items) -> dict:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate native JSON field: {key}")
        result[key] = value
    return result


def assert_unchanged(expected: list[dict]) -> None:
    for original in expected:
        if fingerprint(Path(original["path"])) != original:
            raise ValueError(f"immutable input/native executable/library changed: {original['path']}")


def text(value) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else value or ""


def execute_run(record: dict, executable: Path, library: Path, case: dict, inputs: dict,
                output: Path, timeout: float, truth: list[list[int]], reference: dict | None,
                *, settings_reference: dict | None = None) -> None:
    run_path = output / (record["id"] + ".json")
    if run_path.exists() or run_path.is_symlink():
        raise ValueError(f"run evidence already exists: {run_path}")
    controls = normalized_controls(case.get("controls", {}).get(record["variant"]))
    snapshot = output / (record["id"] + "-snapshot.sqlite")
    database = str(output / (record["id"] + "-database.sqlite")) if case["storage"] == "file" else ":memory:"
    command = [str(executable), str(library), case["mode"], str(case["count"]), str(case["dimension"]),
               inputs["files"]["vectors"]["path"], inputs["files"]["queries"]["path"], str(case["queries"])]
    settings = {"VECTORLITE_BENCH_METRIC": case["metric"], "VECTORLITE_BENCH_DATABASE": database,
                "VECTORLITE_BENCH_SNAPSHOT": str(snapshot), "VECTORLITE_BENCH_MAX_VISITS": "65536",
                "VECTORLITE_BENCH_TIMEOUT_SECONDS": str(timeout)}
    settings.update({"VECTORLITE_BENCH_" + key.upper(): str(value) for key, value in controls.items() if value is not None})
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
        native = json.loads(process.stdout, object_pairs_hook=unique_object)
        # Reject NaN/Infinity (including exponent overflow) before retaining a
        # parsed object, so malformed output cannot poison JSON error evidence.
        json.dumps(native, allow_nan=False)
        validate_result(native, case, truth, database, str(snapshot), controls, timeout)
        record["native"] = native
        record["graph"] = graph_snapshot(snapshot, case["count"], case["dimension"], case["metric"])
        record["snapshot"] = fingerprint(snapshot)
        if case["storage"] == "file":
            original = graph_snapshot(Path(database), case["count"], case["dimension"], case["metric"])
            if original != record["graph"]:
                raise ValueError("file database and backup graph differ")
        if reference is not None:
            verify_same(reference, record, allow_cache_difference=case.get("comparison") == "pager-cache")
        if settings_reference is not None:
            verify_settings(settings_reference, record)
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
    result.add_argument("--pairs", type=int, help="measured pairs per case: screening=3, final=5 (default)")
    result.add_argument("--count", type=int, help="override vector count for every selected case")
    result.add_argument("--queries", type=int, default=100)
    result.add_argument("--warmups", type=int, default=1, help="per variant/case; excludes selfchecks")
    result.add_argument("--timeout", type=float, default=600, help="hard timeout for the COMPLETE native process, seconds")
    result.add_argument("--comparison", choices=("code", "pager-cache"), default="code",
                        help="pager-cache requires the same library hash, file cells, and separate attribution")
    result.add_argument("--page-size", type=int, help="SQLite page bytes; unset preserves the connection default")
    result.add_argument("--cache-size", type=cache_argument, default=None, metavar="SIGNED|default",
                        help="common SQLite cache_size (negative KiB, positive pages); default preserves read-back default")
    for variant in VARIANTS:
        result.add_argument("--" + variant + "-cache-size", type=cache_argument, default=argparse.SUPPRESS,
                            metavar="SIGNED|default", help="per-variant override; differences require --comparison pager-cache")
    result.add_argument("--cache-spill", choices=("on", "off"), help="unset preserves SQLite spill policy")
    result.add_argument("--journal-mode", choices=("delete", "wal"), default="delete", help="file cells only; memory cells retain memory journals")
    result.add_argument("--synchronous", choices=("full", "extra"), default="full")
    result.add_argument("--mmap-size", type=int, help="SQLite mmap bytes; effective readback must match")
    result.add_argument("--temp-store", choices=("default", "file", "memory"))
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
        planned_cases = {}
        for name in dict.fromkeys(args.case or CASES):
            case = {**CASES[name], "dimension": args.dimension, "queries": args.queries, "comparison": args.comparison, "timeout_seconds": args.timeout}
            if args.count is not None:
                case["count"] = args.count
            if args.pairs is not None:
                case["pairs"] = args.pairs
            if args.comparison == "pager-cache" and case["storage"] != "file":
                raise ValueError("pager-cache comparisons require file cases (--case l2-file)")
            case["controls"] = variant_controls(args, case["storage"])
            case["variant_selfchecks"] = args.comparison == "pager-cache"
            planned_cases[name] = case
        executable = args.executable.resolve(strict=True)
        libraries = {key: getattr(args, key).resolve(strict=True) for key in VARIANTS}
        if not executable.is_file() or not all(path.is_file() for path in libraries.values()):
            raise ValueError("executable and libraries must be regular files")
        library_fingerprints = {key: fingerprint(path) for key, path in libraries.items()}
        if args.comparison == "pager-cache" and len({value["sha256"] for value in library_fingerprints.values()}) != 1:
            raise ValueError("pager-cache comparison requires identical immutable library hashes")
        output = fresh_output(args.output_dir)
    except (ValueError, OSError) as error:
        arguments.error(str(error))
    context = {"command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
               "working_directory": str(Path.cwd()), "started_at": datetime.now(timezone.utc).isoformat(),
               "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
               "machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
               "sqlite_python_verifier": sqlite3.sqlite_version,
               "runner": fingerprint(Path(__file__).resolve()), "executable": fingerprint(executable),
               "libraries": library_fingerprints,
               "settings": {"dimension": args.dimension, "seed": 42, "queries": args.queries, "k": 10,
                            "query_windows": list(WINDOWS), "diskann": OPTIONS, "tracing": False,
                            "comparison": args.comparison, "warmups_per_variant_case": args.warmups,
                            "selfchecks_per_configuration": 2, "selfcheck_variants": list(VARIANTS) if args.comparison == "pager-cache" else ["baseline"],
                            "query_warmup_passes": 1, "query_timed_passes": 1, "timeout_seconds": args.timeout,
                            "query_percentile_method": "linear", "query_connection": "same_open_post_build",
                            "wal_autocheckpoint": 0, "wal_final_checkpoint": "FULL after queries before snapshot/close"},
               "notes": ["Same executable and immutable inputs; alternating AB/BA measured pairs, fresh process per build.",
                         "Native timer: BEGIN + INSERT(s) + COMMIT, including statement prepare/finalize; CREATE is excluded.",
                         "Queries use the SAME open, post-build, build-warmed connection: one untimed then one timed pass per window.",
                         "Query p50/p95 are per-process numpy.percentile(method=linear); regression medians use matched candidate/baseline ratios.",
                         "Data generation/oracle, queries, final FULL checkpoint, backup and streaming hashing are outside build timing.",
                         "WAL build/queries/final checkpoint are separate costs, not total physical bytes written or COMMIT-only checkpoint cost.",
                         "Pager-cache comparisons are separate attribution and cannot satisfy the primary code speedup gate.",
                         "Graph hash excludes ONLY random instance_id; adjacency ordering is significant.",
                         "Small samples and warm OS caches; no cold-disk, beyond-RAM, power-loss or cross-platform execution claim.",
                         "Do not measure concurrently with intentional heavy builds/reviews; record external contention separately."]}
    report = {"schema_version": 2, "context": context, "cases": planned_cases, "runs": [], "summary": {},
              "summary_validated": False, "status": "running"}
    report_path = output / "results.json"
    save_json(report_path, report)
    immutable = [context["runner"], context["executable"], *context["libraries"].values()]
    inputs_by_count = {}  # One dimension/query count/seed per invocation, recorded above.
    try:
        for name, case in planned_cases.items():
            if case["count"] not in inputs_by_count:
                inputs_by_count[case["count"]] = make_inputs(output, case["count"], args.queries, args.dimension)
                immutable.extend(inputs_by_count[case["count"]]["files"].values())
            inputs = inputs_by_count[case["count"]]
            truth = oracle(inputs["vectors"], inputs["queries"], case["metric"])
            oracle_suffix = f"{name}" if args.dimension == 128 else f"{name}-{args.dimension}d"
            oracle_path = output / (oracle_suffix + "-oracle.json")
            save_json(oracle_path, {"metric": case["metric"], "precision": "float64", "k": 10, "labels": truth})
            case.update(inputs=inputs["files"], oracle=fingerprint(oracle_path))
            immutable.append(case["oracle"])
            reference, settings_references = None, {}
            for phase, index, variant in run_order(case["pairs"], args.warmups, variant_selfchecks=case["variant_selfchecks"]):
                assert_unchanged(immutable)
                record = {"id": f"{name}-{phase}-{index:02d}-{variant}", "case": name,
                          "phase": phase, "index": index, "variant": variant}
                report["runs"].append(record)
                save_json(report_path, report)
                execute_run(record, executable, libraries[variant], case, inputs, output, args.timeout, truth, reference,
                            settings_reference=settings_references.get(variant))
                report["summary"] = summary(report["runs"], planned_cases)
                save_json(report_path, report)
                if record["status"] != "ok":
                    raise ValueError(f"{record['id']}: {record['error']}")
                assert_unchanged(immutable)
                if reference is None:
                    reference = record
                settings_references.setdefault(variant, record)
                print(f"{record['id']}: {record['native']['build_seconds']:.6f}s graph={record['graph']['sha256']}", flush=True)
        assert_unchanged(immutable)
        report["summary"] = summary(report["runs"], planned_cases, final=True)
        report.update(status="ok", summary_validated=True)
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        print(report["error"], file=sys.stderr, flush=True)
    report["context"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_json(report_path, report)
    print(json.dumps({"status": report["status"], "output": str(output), "summary": report["summary"]}, indent=2), flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
