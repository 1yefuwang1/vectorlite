"""Small correctness checks for shared-data DiskANN benchmark integration."""
from __future__ import annotations

import csv
import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

import benchmark as harness
from benchmark import BenchmarkData, VectorliteDiskAnnBackend
from test_benchmark import _run_search


class RecordingCursor:
    def __init__(self, rows=None, insert_error=None):
        self.calls = []
        self.rows = [(0,)] if rows is None else rows
        self.insert_error = insert_error

    def execute(self, sql, parameters=()):
        self.calls.append((sql, parameters))
        return self

    def executemany(self, sql, rows):
        self.calls.append((sql, list(rows)))
        if self.insert_error is not None:
            raise self.insert_error
        return self

    def fetchall(self):
        return list(self.rows)


def sample_data(num_queries=2):
    return SimpleNamespace(
        num_elements=2,
        data_bytes={128: [b"vector-0", b"vector-1"]},
        query_bytes={128: [b"query"] * num_queries},
        seed=42,
    )


def test_diskann_setup_and_labels():
    cursor = RecordingCursor()
    backend = VectorliteDiskAnnBackend(cursor, sample_data())
    backend.setup("cosine", 128, None, None)
    sql, parameters = cursor.calls[0]
    assert "embedding float32[128] cosine" in sql
    assert "diskann(degree=32, build_list_size=100)" in sql
    assert parameters == ()
    assert not backend.supports_ef_search
    assert backend.supports_search_list_size
    assert backend.query_label("cosine", 100) == "vectorlite_diskann_cosine_L_100"
    assert backend.insertion_label("l2") == "vectorlite_diskann_l2"


@pytest.mark.parametrize("window", [None, 10, 50, 100])
def test_diskann_search_uses_named_tuning(monkeypatch, window):
    monkeypatch.setattr(harness, "NUM_QUERIES", 2)
    cursor = RecordingCursor()
    backend = VectorliteDiskAnnBackend(cursor, sample_data())
    backend.setup("l2", 128, None, None)
    cursor.calls.clear()
    assert backend.do_search("l2", 128, window) == [[(0,)], [(0,)]]
    assert len(cursor.calls) == 2
    for sql, parameters in cursor.calls:
        assert parameters[:2] == (b"query", harness.K)
        if window is None:
            assert "knn_param(?, ?)" in sql
            assert len(parameters) == 2
        else:
            assert "knn_param(?, ?, ?)" in sql
            assert isinstance(parameters[2], str)
            assert json.loads(parameters[2]) == {"search_list_size": window}


def test_diskann_insert_uses_shared_transaction_helper():
    cursor = RecordingCursor()
    backend = VectorliteDiskAnnBackend(cursor, sample_data())
    backend.setup("l2", 128, None, None)
    cursor.calls.clear()
    backend.do_insert("l2", 128)
    assert cursor.calls[0] == ("BEGIN TRANSACTION;", ())
    assert cursor.calls[1][1] == [(0, b"vector-0"), (1, b"vector-1")]
    assert cursor.calls[2] == ("COMMIT;", ())


def test_diskann_insert_error_rolls_back():
    cursor = RecordingCursor(insert_error=sqlite3.IntegrityError("failed insert"))
    backend = VectorliteDiskAnnBackend(cursor, sample_data())
    backend.setup("l2", 128, None, None)
    cursor.calls.clear()
    with pytest.raises(sqlite3.IntegrityError, match="failed insert"):
        backend.do_insert("l2", 128)
    assert cursor.calls[-1] == ("ROLLBACK;", ())
    assert not any(sql == "COMMIT;" for sql, _ in cursor.calls)


def test_seeded_data_is_reproducible():
    arguments = ([4], 16, 2, 3, ["l2", "cosine"])
    first = BenchmarkData.generate(*arguments, seed=42)
    second = BenchmarkData.generate(*arguments, seed=42)
    different = BenchmarkData.generate(*arguments, seed=43)
    assert first.seed == 42
    assert first.data_bytes == second.data_bytes
    assert first.query_bytes == second.query_bytes
    assert first.data_bytes != different.data_bytes
    for metric in ["l2", "cosine"]:
        np.testing.assert_array_equal(first.correct_labels[metric][4],
                                      second.correct_labels[metric][4])


def test_search_metadata_distinguishes_diskann_window():
    class BenchmarkRecorder:
        def __init__(self):
            self.extra_info = {}

        def __call__(self, function, *args):
            return function(*args)

    data = sample_data(harness.NUM_QUERIES)
    data.correct_labels = {
        "l2": {128: np.tile(np.arange(harness.K), (harness.NUM_QUERIES, 1))}}
    cursor = RecordingCursor(rows=[(i,) for i in range(harness.K)])
    backend = VectorliteDiskAnnBackend(cursor, data)
    recorder = BenchmarkRecorder()
    _run_search(recorder, backend, data, "l2", 128, None, None, 100)
    info = recorder.extra_info
    assert info["ef_search"] is None
    assert info["search_list_size"] == 100
    assert info["plot_label"] == "vectorlite_diskann_l2_L_100"
    assert info["num_elements"] == 2
    assert info["dataset_seed"] == 42
    assert info["recall"] == 1.0


def test_recall_csv_preserves_diskann_window(tmp_path):
    from plot import _write_recall_csv

    path = tmp_path / "recall.csv"
    _write_recall_csv([{"extra_info": {
        "product": "vectorlite_diskann", "distance_type": "l2", "dim": 128,
        "ef_search": None, "search_list_size": 100, "recall": 0.95,
    }}], path)
    with path.open(newline="") as source:
        row, = csv.DictReader(source)
    assert row["ef_search"] == ""
    assert row["search_list_size"] == "100"
    assert row["recall"] == "0.95"
