"""Benchmark harness library: synthetic data, ground truth, and pluggable backends.

This module is consumed by ``test_benchmark.py``. Run the benchmarks with::

    pytest benchmark/test_benchmark.py --benchmark-json=bench.json
    python benchmark/plot.py bench.json

Methodology follows
https://github.com/nmslib/hnswlib/blob/v0.8.0/TESTING_RECALL.md.

Configuration is driven by environment variables (read by ``conftest.py``):

    NUM_ELEMENTS           number of random vectors to index (default 3000)
    VECTORLITE_PATH        path to a locally built vectorlite extension
    BENCHMARK_VSS=1        enable sqlite_vss backend (Linux/macOS only)
    BENCHMARK_SQLITE_VEC=1 enable sqlite-vec backend (Linux/macOS only)
    BENCHMARK_MILVUS_LITE=1 enable milvus-lite backend (Linux/macOS only)

SQLite driver
-------------

The benchmark uses Python's standard-library ``sqlite3`` module rather than
``apsw``. Loading the vectorlite extension via ``conn.load_extension(...)``
requires a Python interpreter built with
``--enable-loadable-sqlite-extensions`` (standard on Homebrew, python.org
and modern Linux distribution Python builds). If your interpreter does not
enable that, ``conn.enable_load_extension(True)`` raises ``AttributeError``
or ``OperationalError`` and you will need a different Python build (or
``apsw``).

Vectorlite's metadata-filter (rowid pushdown) feature requires
SQLite >= 3.38; the benchmark itself does not exercise that path, so any
SQLite version that vectorlite loads on will work. The bundled SQLite
version is reported by ``sqlite3.sqlite_version``.
"""

from __future__ import annotations

import dataclasses
import platform
import sqlite3
from typing import List, Optional, Sequence, Tuple

import hnswlib
import numpy as np

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

NUM_QUERIES: int = 100
K: int = 10  # number of nearest neighbours retrieved per query

DIMS: List[int] = [128, 512, 1536, 3000]
# 'ip' (inner product) is intentionally excluded - it is not a true metric.
DISTANCE_TYPES: List[str] = ["l2", "cosine"]
HNSW_PARAMS: List[Tuple[int, int]] = [(100, 30)]  # (ef_construction, M)
EF_SEARCH_VALUES: List[int] = [10, 50, 100]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_recall(predicted: Sequence[Sequence[int]],
                   expected: np.ndarray, k: int) -> float:
    """Mean fraction of true k-NN labels recovered per query."""
    return float(np.mean([
        np.intersect1d(predicted[i], expected[i]).size / k
        for i in range(len(predicted))
    ]))


def is_supported_platform() -> bool:
    return platform.system().lower() in ("linux", "darwin")


# ---------------------------------------------------------------------------
# Synthetic dataset and ground truth
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BenchmarkData:
    """Random vectors plus their precomputed brute-force k-NN ground truth."""

    num_elements: int
    data: dict          # dim -> np.ndarray (NUM_ELEMENTS x dim, float32)
    data_bytes: dict    # dim -> list[bytes] (one per element, for SQL inserts)
    queries: dict       # dim -> np.ndarray (NUM_QUERIES x dim, float32)
    query_bytes: dict   # dim -> list[bytes]
    correct_labels: dict  # distance_type -> dim -> np.ndarray (NUM_QUERIES x K)

    @classmethod
    def generate(cls, dims: Sequence[int], num_elements: int,
                 num_queries: int, k: int,
                 distance_types: Sequence[str]) -> "BenchmarkData":
        data = {d: np.float32(np.random.random((num_elements, d))) for d in dims}
        data_bytes = {
            d: [data[d][i].tobytes() for i in range(num_elements)] for d in dims
        }
        queries = {
            d: np.float32(np.random.random((num_queries, d))) for d in dims
        }
        query_bytes = {
            d: [queries[d][i].tobytes() for i in range(num_queries)] for d in dims
        }

        correct_labels: dict = {}
        for dt in distance_types:
            correct_labels[dt] = {}
            for d in dims:
                bf = hnswlib.BFIndex(space=dt, dim=d)
                bf.init_index(max_elements=num_elements)
                bf.add_items(data[d])
                labels, _ = bf.knn_query(queries[d], k=k)
                assert len(labels) == num_queries and len(labels[0]) == k
                correct_labels[dt][d] = labels
                del bf

        return cls(num_elements, data, data_bytes,
                   queries, query_bytes, correct_labels)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class Backend:
    """Common interface for every system under test.

    A backend handles the lifecycle of one (distance_type, dim, hnsw_params)
    configuration. ``test_benchmark.py`` drives it through:

        backend.setup(distance_type, dim, ef_construction, M)
        backend.do_insert(distance_type, dim)              # measured
        for ef in EF_SEARCH_VALUES (or [None]):
            backend.do_search(distance_type, dim, ef)      # measured
        backend.teardown()
    """

    name: str = "<unknown>"
    plot_label: str = "<unknown>"  # may include {distance_type} / {ef_search}
    uses_hnsw_params: bool = False
    supports_ef_search: bool = False
    supported_distances: Sequence[str] = DISTANCE_TYPES
    # Some backends should not appear in the query plot at high N because
    # their search time dwarfs every other point. Override per-backend.
    include_in_query_plot: bool = True

    # --- lifecycle (override) -------------------------------------------------

    def setup(self, distance_type: str, dim: int,
              ef_construction: Optional[int], M: Optional[int]) -> None:
        raise NotImplementedError

    def do_insert(self, distance_type: str, dim: int) -> None:
        raise NotImplementedError

    def do_search(self, distance_type: str, dim: int,
                  ef_search: Optional[int]) -> List[Sequence[int]]:
        raise NotImplementedError

    def teardown(self) -> None:
        pass

    # --- helpers --------------------------------------------------------------

    def insertion_label(self, distance_type: str) -> str:
        return self.plot_label.format(distance_type=distance_type, ef_search="")

    def query_label(self, distance_type: str, ef_search: Optional[int]) -> str:
        return self.plot_label.format(
            distance_type=distance_type,
            ef_search=f"_ef_{ef_search}" if ef_search is not None else "",
        )


# ---------------------------------------------------------------------------
# Concrete backends
# ---------------------------------------------------------------------------


class _SqlBackend(Backend):
    """Shared scaffolding for backends that use the sqlite3 cursor."""

    def __init__(self, cursor: sqlite3.Cursor, data: BenchmarkData) -> None:
        self.cursor = cursor
        self.data = data
        self._table: Optional[str] = None
        self._dim: int = 0

    def _insert_rows(self) -> None:
        rows = [(i, self.data.data_bytes[self._dim][i])
                for i in range(self.data.num_elements)]
        self.cursor.execute("BEGIN TRANSACTION;")
        self.cursor.executemany(
            f"insert into {self._table}(rowid, embedding) values (?, ?)", rows)
        self.cursor.execute("COMMIT;")

    def teardown(self) -> None:
        if self._table is not None:
            self.cursor.execute(f"drop table {self._table}")
            self._table = None


class VectorliteBackend(_SqlBackend):
    name = "vectorlite"
    plot_label = "vectorlite_{distance_type}{ef_search}"
    uses_hnsw_params = True
    supports_ef_search = True

    def setup(self, distance_type: str, dim: int,
              ef_construction: Optional[int], M: Optional[int]) -> None:
        self._dim = dim
        self._table = f"table_{distance_type}_{dim}_{ef_construction}_{M}"
        self.cursor.execute(
            f"create virtual table {self._table} using vectorlite("
            f"embedding float32[{dim}] {distance_type}, "
            f"hnsw(max_elements={self.data.num_elements}, "
            f"ef_construction={ef_construction}, M={M}))"
        )

    def do_insert(self, distance_type: str, dim: int) -> None:
        self._insert_rows()

    def do_search(self, distance_type: str, dim: int,
                  ef_search: Optional[int]) -> List[Sequence[int]]:
        results = []
        for i in range(NUM_QUERIES):
            results.append(self.cursor.execute(
                f"select rowid from {self._table} "
                f"where knn_search(embedding, knn_param(?, ?, ?))",
                (self.data.query_bytes[dim][i], K, ef_search),
            ).fetchall())
        return results


class VectorliteBruteForceBackend(_SqlBackend):
    name = "vectorlite_brute_force"
    plot_label = "vectorlite_scalar_brute_force"
    supported_distances = ["l2"]

    def setup(self, distance_type: str, dim: int,
              ef_construction: Optional[int], M: Optional[int]) -> None:
        self._dim = dim
        self._table = f"table_vectorlite_bf_{dim}"
        self.cursor.execute(
            f"create table {self._table}"
            f"(rowid integer primary key, embedding blob)"
        )

    def do_insert(self, distance_type: str, dim: int) -> None:
        self._insert_rows()

    def do_search(self, distance_type: str, dim: int,
                  ef_search: Optional[int]) -> List[Sequence[int]]:
        results = []
        for i in range(NUM_QUERIES):
            results.append(self.cursor.execute(
                f"select rowid from {self._table} "
                f"order by vector_distance(?, embedding, 'l2') asc limit {K}",
                [self.data.query_bytes[dim][i]],
            ).fetchall())
        return results


class HnswlibBackend(Backend):
    name = "hnswlib"
    plot_label = "hnswlib_{distance_type}{ef_search}"
    uses_hnsw_params = True
    supports_ef_search = True

    def __init__(self, data: BenchmarkData) -> None:
        self.data = data
        self._index: Optional[hnswlib.Index] = None

    def setup(self, distance_type: str, dim: int,
              ef_construction: Optional[int], M: Optional[int]) -> None:
        self._index = hnswlib.Index(space=distance_type, dim=dim)
        self._index.init_index(
            max_elements=self.data.num_elements,
            ef_construction=ef_construction,
            M=M,
        )

    def do_insert(self, distance_type: str, dim: int) -> None:
        assert self._index is not None
        self._index.add_items(self.data.data[dim])

    def do_search(self, distance_type: str, dim: int,
                  ef_search: Optional[int]) -> List[Sequence[int]]:
        assert self._index is not None
        if ef_search is not None:
            self._index.set_ef(ef_search)
        results = []
        for i in range(NUM_QUERIES):
            labels, _ = self._index.knn_query(self.data.queries[dim][i], k=K)
            results.append(labels)
        return results

    def teardown(self) -> None:
        self._index = None


class SqliteVssBackend(_SqlBackend):
    name = "sqlite_vss"
    plot_label = "sqlite_vss"
    supported_distances = ["l2"]

    def setup(self, distance_type: str, dim: int,
              ef_construction: Optional[int], M: Optional[int]) -> None:
        self._dim = dim
        self._table = f"table_vss_{dim}"
        self.cursor.execute(
            f"create virtual table {self._table} using vss0(embedding({dim}))"
        )

    def do_insert(self, distance_type: str, dim: int) -> None:
        self._insert_rows()

    def do_search(self, distance_type: str, dim: int,
                  ef_search: Optional[int]) -> List[Sequence[int]]:
        results = []
        for i in range(NUM_QUERIES):
            results.append(self.cursor.execute(
                f"select rowid from {self._table} "
                f"where vss_search(embedding, ?) limit {K}",
                (self.data.query_bytes[dim][i],),
            ).fetchall())
        return results


class SqliteVecBackend(_SqlBackend):
    name = "sqlite_vec"
    plot_label = "sqlite_vec"
    supported_distances = ["l2"]

    def setup(self, distance_type: str, dim: int,
              ef_construction: Optional[int], M: Optional[int]) -> None:
        self._dim = dim
        self._table = f"table_vec_{dim}"
        self.cursor.execute(
            f"create virtual table {self._table} using vec0("
            f"rowid integer primary key, embedding float[{dim}])"
        )

    def do_insert(self, distance_type: str, dim: int) -> None:
        self._insert_rows()

    def do_search(self, distance_type: str, dim: int,
                  ef_search: Optional[int]) -> List[Sequence[int]]:
        results = []
        for i in range(NUM_QUERIES):
            results.append(self.cursor.execute(
                f"select rowid from {self._table} "
                f"where embedding match ? and k = {K}",
                (self.data.query_bytes[dim][i],),
            ).fetchall())
        return results


class MilvusLiteBackend(Backend):
    name = "milvus_lite"
    plot_label = "milvuslite_{distance_type}{ef_search}"

    def __init__(self, data: BenchmarkData,
                 db_path: str = "./milvus_lite_demo.db") -> None:
        from pymilvus import MilvusClient
        self.data = data
        self._client = MilvusClient(db_path)
        self._collection: Optional[str] = None

    def setup(self, distance_type: str, dim: int,
              ef_construction: Optional[int], M: Optional[int]) -> None:
        from pymilvus import DataType
        self._collection = f"collection_{distance_type}_{dim}"

        schema = self._client.create_schema(enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type=distance_type.upper(),
        )
        self._client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
            dimension=dim,
        )
        self._client.get_load_state(collection_name=self._collection)

    def do_insert(self, distance_type: str, dim: int) -> None:
        rows = [
            {"id": i, "embedding": self.data.data[dim][i].tolist()}
            for i in range(self.data.num_elements)
        ]
        self._client.insert(collection_name=self._collection, data=rows)

    def do_search(self, distance_type: str, dim: int,
                  ef_search: Optional[int]) -> List[Sequence[int]]:
        results = []
        for i in range(NUM_QUERIES):
            response = self._client.search(
                collection_name=self._collection,
                data=[self.data.queries[dim][i].tolist()],
                search_params={"metric_type": distance_type.upper()},
            )
            results.append([hit["id"] for hit in response[0]])
        return results

    def teardown(self) -> None:
        if self._collection is not None:
            self._client.drop_collection(collection_name=self._collection)
            self._collection = None
