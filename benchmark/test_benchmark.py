"""pytest-benchmark suite for vectorlite and comparable vector backends.

Run::

    # Default backends only (vectorlite, hnswlib, vectorlite brute force):
    pytest benchmark/test_benchmark.py --benchmark-json=bench.json

    # All backends:
    BENCHMARK_VSS=1 BENCHMARK_SQLITE_VEC=1 BENCHMARK_MILVUS_LITE=1 \\
        pytest benchmark/test_benchmark.py --benchmark-json=bench.json

    # Render PNG plots from the JSON output:
    python benchmark/plot.py bench.json

Each parametrised test cell contributes one benchmark to the JSON output.
The benchmark's ``extra_info`` dict carries the metadata that ``plot.py``
needs to group results into the existing two figures
(insertion-time-per-vector and query-time-per-query, both vs. dim):

    extra_info = {
        "plot_kind":               "insertion" | "query",
        "plot_label":              <legend label>,
        "product":                 backend.name,
        "distance_type":           "l2" | "cosine",
        "dim":                     int,
        "ef_search":               int | None,            # query only
        "recall":                  float,                 # query only
        "include_in_query_plot":   bool,                  # query only
    }
"""

from __future__ import annotations

from typing import Optional

import pytest

from benchmark import (
    Backend,
    BenchmarkData,
    DIMS,
    DISTANCE_TYPES,
    EF_SEARCH_VALUES,
    HNSW_PARAMS,
    K,
    NUM_QUERIES,
    compute_recall,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _skip_if_unsupported(backend: Backend, distance_type: str) -> None:
    if distance_type not in backend.supported_distances:
        pytest.skip(f"{backend.name} does not support {distance_type}")


def _run_insert(benchmark, backend: Backend, data: BenchmarkData,
                distance_type: str, dim: int,
                ef_construction: Optional[int], M: Optional[int]) -> None:
    """Time one bulk insert into a fresh index.

    Uses ``pedantic`` with a per-round setup that recreates the underlying
    storage so each round measures the same cold-insert cost. The wall-clock
    time of teardown/setup itself is not measured.
    """
    _skip_if_unsupported(backend, distance_type)

    def setup() -> tuple:
        backend.teardown()  # safe to call before first setup
        backend.setup(distance_type, dim, ef_construction, M)
        return (), {}

    try:
        benchmark.pedantic(
            lambda: backend.do_insert(distance_type, dim),
            setup=setup,
            rounds=3,
            iterations=1,
            warmup_rounds=1,
        )
    finally:
        backend.teardown()

    benchmark.extra_info.update({
        "plot_kind": "insertion",
        "plot_label": backend.insertion_label(distance_type),
        "product": backend.name,
        "distance_type": distance_type,
        "dim": dim,
        "num_elements": data.num_elements,
    })


def _run_search(benchmark, backend: Backend, data: BenchmarkData,
                distance_type: str, dim: int,
                ef_construction: Optional[int], M: Optional[int],
                ef_search: Optional[int]) -> None:
    """Time one batch of NUM_QUERIES kNN searches against a built index.

    pytest-benchmark auto-calibrates rounds and warm-up. We build the index
    once outside the timer.
    """
    _skip_if_unsupported(backend, distance_type)
    if ef_search is not None and not backend.supports_ef_search:
        pytest.skip(f"{backend.name} does not parametrise on ef_search")

    backend.setup(distance_type, dim, ef_construction, M)
    try:
        backend.do_insert(distance_type, dim)
        results = benchmark(
            backend.do_search, distance_type, dim, ef_search)
        recall = compute_recall(
            results, data.correct_labels[distance_type][dim], K)
    finally:
        backend.teardown()

    benchmark.extra_info.update({
        "plot_kind": "query",
        "plot_label": backend.query_label(distance_type, ef_search),
        "product": backend.name,
        "distance_type": distance_type,
        "dim": dim,
        "ef_search": ef_search,
        "recall": recall,
        "include_in_query_plot": backend.include_in_query_plot,
        "num_queries": NUM_QUERIES,
    })


# Common parametrisation marks
_ec, _M = HNSW_PARAMS[0]
_param_dim = pytest.mark.parametrize("dim", DIMS)
_param_distance = pytest.mark.parametrize("distance_type", DISTANCE_TYPES)
_param_distance_l2 = pytest.mark.parametrize("distance_type", ["l2"])
_param_ef = pytest.mark.parametrize("ef_search", EF_SEARCH_VALUES)


# ---------------------------------------------------------------------------
# vectorlite (HNSW virtual table)
# ---------------------------------------------------------------------------


@_param_dim
@_param_distance
def test_insert_vectorlite(benchmark, vectorlite_backend, benchmark_data,
                           distance_type, dim):
    _run_insert(benchmark, vectorlite_backend, benchmark_data,
                distance_type, dim, _ec, _M)


@_param_ef
@_param_dim
@_param_distance
def test_search_vectorlite(benchmark, vectorlite_backend, benchmark_data,
                           distance_type, dim, ef_search):
    _run_search(benchmark, vectorlite_backend, benchmark_data,
                distance_type, dim, _ec, _M, ef_search)


# ---------------------------------------------------------------------------
# hnswlib (in-memory baseline)
# ---------------------------------------------------------------------------


@_param_dim
@_param_distance
def test_insert_hnswlib(benchmark, hnswlib_backend, benchmark_data,
                        distance_type, dim):
    _run_insert(benchmark, hnswlib_backend, benchmark_data,
                distance_type, dim, _ec, _M)


@_param_ef
@_param_dim
@_param_distance
def test_search_hnswlib(benchmark, hnswlib_backend, benchmark_data,
                        distance_type, dim, ef_search):
    _run_search(benchmark, hnswlib_backend, benchmark_data,
                distance_type, dim, _ec, _M, ef_search)


# ---------------------------------------------------------------------------
# vectorlite brute force (plain SQL table + vector_distance ORDER BY)
# ---------------------------------------------------------------------------


@_param_dim
@_param_distance_l2
def test_insert_vectorlite_brute_force(benchmark, vectorlite_bf_backend,
                                       benchmark_data, distance_type, dim):
    _run_insert(benchmark, vectorlite_bf_backend, benchmark_data,
                distance_type, dim, None, None)


@_param_dim
@_param_distance_l2
def test_search_vectorlite_brute_force(benchmark, vectorlite_bf_backend,
                                       benchmark_data, distance_type, dim):
    _run_search(benchmark, vectorlite_bf_backend, benchmark_data,
                distance_type, dim, None, None, None)


# ---------------------------------------------------------------------------
# Optional backends - skipped unless the matching env var is set
# ---------------------------------------------------------------------------


@_param_dim
@_param_distance_l2
def test_insert_sqlite_vss(benchmark, sqlite_vss_backend, benchmark_data,
                           distance_type, dim):
    _run_insert(benchmark, sqlite_vss_backend, benchmark_data,
                distance_type, dim, None, None)


@_param_dim
@_param_distance_l2
def test_search_sqlite_vss(benchmark, sqlite_vss_backend, benchmark_data,
                           distance_type, dim):
    _run_search(benchmark, sqlite_vss_backend, benchmark_data,
                distance_type, dim, None, None, None)


@_param_dim
@_param_distance_l2
def test_insert_sqlite_vec(benchmark, sqlite_vec_backend, benchmark_data,
                           distance_type, dim):
    _run_insert(benchmark, sqlite_vec_backend, benchmark_data,
                distance_type, dim, None, None)


@_param_dim
@_param_distance_l2
def test_search_sqlite_vec(benchmark, sqlite_vec_backend, benchmark_data,
                           distance_type, dim):
    _run_search(benchmark, sqlite_vec_backend, benchmark_data,
                distance_type, dim, None, None, None)


@_param_dim
@_param_distance
def test_insert_milvus_lite(benchmark, milvus_lite_backend, benchmark_data,
                            distance_type, dim):
    _run_insert(benchmark, milvus_lite_backend, benchmark_data,
                distance_type, dim, None, None)


@_param_dim
@_param_distance
def test_search_milvus_lite(benchmark, milvus_lite_backend, benchmark_data,
                            distance_type, dim):
    _run_search(benchmark, milvus_lite_backend, benchmark_data,
                distance_type, dim, None, None, None)


# ---------------------------------------------------------------------------
# libSQL (DiskANN vector index)
# ---------------------------------------------------------------------------


@_param_dim
@_param_distance
def test_insert_libsql(benchmark, libsql_backend, benchmark_data,
                       distance_type, dim):
    _run_insert(benchmark, libsql_backend, benchmark_data,
                distance_type, dim, None, None)


@_param_dim
@_param_distance
def test_search_libsql(benchmark, libsql_backend, benchmark_data,
                       distance_type, dim):
    _run_search(benchmark, libsql_backend, benchmark_data,
                distance_type, dim, None, None, None)


# ---------------------------------------------------------------------------
# sqlite-vector (SIMD full scan)
# ---------------------------------------------------------------------------


@_param_dim
@_param_distance
def test_insert_sqlite_vector(benchmark, sqlite_vector_backend, benchmark_data,
                              distance_type, dim):
    _run_insert(benchmark, sqlite_vector_backend, benchmark_data,
                distance_type, dim, None, None)


@_param_dim
@_param_distance
def test_search_sqlite_vector(benchmark, sqlite_vector_backend, benchmark_data,
                              distance_type, dim):
    _run_search(benchmark, sqlite_vector_backend, benchmark_data,
                distance_type, dim, None, None, None)
