#!/usr/bin/env python3
"""Retired Format 3 identity-cache A/B runner.

The former runner and graph snapshot verifier assumed synchronous graph INSERT,
Format 3 node shadows, and a cache no longer used by the current deferred format. It must not be
used to make current-format performance claims. Generic, artifact-independent input,
pager and hashing helpers remain for their own tests.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import struct
import sys
import numpy as np

CONTROL_KEYS = ("page_size", "cache_size", "cache_spill", "journal_mode", "synchronous", "mmap_size", "temp_store")


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


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


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


def finite_number(value, positive=False) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print("The Format 3 identity-cache benchmark is retired; use diskann_benchmark.py for current-format lifecycle measurements.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
