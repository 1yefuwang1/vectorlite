"""Artifact-independent checks for retained generic helpers; Format 3 runner is retired."""
import argparse
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "benchmark" / "diskann_identity_benchmark.py"
SPEC = importlib.util.spec_from_file_location("diskann_identity_benchmark", MODULE_PATH)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_retired_runner_fails_closed(capsys):
    assert bench.main([]) == 2
    assert "retired" in capsys.readouterr().err
    with pytest.raises(SystemExit) as result:
        bench.main(["--baseline", "not-a-real-library"])
    assert result.value.code == 2


def test_pager_defaults_and_valid_controls():
    assert all(value is None for value in bench.normalized_controls().values())
    values = {"page_size": 4096, "cache_size": -8192, "mmap_size": 0,
              "cache_spill": "on", "journal_mode": "wal", "synchronous": "full", "temp_store": "memory"}
    assert bench.normalized_controls(values) == values


@pytest.mark.parametrize("value", [
    {"unknown": 1}, {"page_size": 1000}, {"page_size": True}, {"cache_size": 0},
    {"cache_size": 2**31}, {"mmap_size": -1}, {"cache_spill": "yes"},
    {"journal_mode": "off"}, {"synchronous": "normal"}, {"temp_store": "bad"},
])
def test_invalid_pager_controls_fail_closed(value):
    with pytest.raises(ValueError):
        bench.normalized_controls(value)


def test_cache_argument():
    assert bench.cache_argument("default") is None
    assert bench.cache_argument("-8192") == -8192
    for value in ("", "0", "1.5", "--1"):
        with pytest.raises(argparse.ArgumentTypeError):
            bench.cache_argument(value)


def test_hash_is_typed_and_length_framed():
    def digest(values):
        value = hashlib.sha256()
        for item in values:
            bench.hash_value(value, item)
        return value.hexdigest()
    assert digest([b"ab", b"c"]) != digest([b"a", b"bc"])
    assert digest([1]) != digest([b"1"])
    assert digest([None]) != digest([0])
    with pytest.raises(ValueError):
        digest([1.5])


def test_oracle_distance_and_ties():
    vectors = np.array([[10, 0], [1, 1], [0, 0], [-1, 0]], dtype=np.float32)
    query = np.array([[1, 0]], dtype=np.float32)
    assert bench.oracle(vectors, query, "cosine") == [[0, 1, 2, 3]]
    assert bench.oracle(vectors, query, "l2") == [[1, 2, 3, 0]]
    with pytest.raises(ValueError):
        bench.oracle(vectors, query, "unknown")


def test_finite_number_rejects_nonfinite_and_bool():
    assert bench.finite_number(0)
    assert bench.finite_number(1, positive=True)
    assert not bench.finite_number(0, positive=True)
    for value in (True, -1, float("nan"), float("inf"), 10 ** 400):
        assert not bench.finite_number(value)


def test_fingerprint_detects_changes(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"first")
    first = bench.fingerprint(path)
    path.write_bytes(b"second")
    assert first["sha256"] != bench.fingerprint(path)["sha256"]
