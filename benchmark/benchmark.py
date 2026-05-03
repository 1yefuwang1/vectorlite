"""Benchmark vectorlite's performance and recall against alternative backends.

Methodology follows
https://github.com/nmslib/hnswlib/blob/v0.8.0/TESTING_RECALL.md.

Configuration is driven by environment variables:

    NUM_ELEMENTS           number of random vectors to index (default 3000)
    VECTORLITE_PATH        path to a locally built vectorlite extension
    BENCHMARK_VSS=1        also benchmark sqlite_vss (Linux/macOS only)
    BENCHMARK_SQLITE_VEC=1 also benchmark sqlite-vec (Linux/macOS only)
    BENCHMARK_MILVUS_LITE=1 also benchmark milvus-lite (Linux/macOS only)

SQLite driver
-------------

The benchmark uses Python's standard-library ``sqlite3`` module rather than
``apsw``. The module loads the vectorlite extension via
``conn.enable_load_extension(True)`` followed by ``conn.load_extension(...)``,
which requires a Python interpreter built with
``--enable-loadable-sqlite-extensions``. Standard Homebrew, python.org and
modern Linux distribution Python builds all enable this; if your interpreter
does not, ``conn.enable_load_extension(True)`` raises ``AttributeError`` or
``OperationalError`` and you will need a different Python build (or ``apsw``).

Vectorlite's optional metadata-filter (rowid pushdown) feature requires
SQLite >= 3.38; the benchmark itself does not exercise that path, so any
SQLite version that vectorlite loads on will work here. The bundled SQLite
version is reported by ``sqlite3.sqlite_version``.
"""

from __future__ import annotations

import dataclasses
import os
import platform
import sqlite3
import time
from collections import defaultdict
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import hnswlib
import numpy as np
import vectorlite_py
from rich.console import Console, ConsoleOptions, RenderResult
from rich.table import Table

import plot

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_ELEMENTS: int = int(os.environ.get("NUM_ELEMENTS", 3000))
NUM_QUERIES: int = 100
K: int = 10  # number of nearest neighbours retrieved per query

DIMS: List[int] = [128, 512, 1536, 3000]
# 'ip' (inner product) is intentionally excluded - it is not a true metric.
DISTANCE_TYPES: List[str] = ["l2", "cosine"]
HNSW_PARAMS: List[Tuple[int, int]] = [(100, 30)]  # (ef_construction, M)
EF_SEARCH_VALUES: List[int] = [10, 50, 100]

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def timeit(func: Callable[[], object]) -> Tuple[float, object]:
    """Run ``func`` once. Return (elapsed_microseconds, return_value).

    Avoids ``timeit.timeit`` which compiles strings and discards return values.
    """
    start_us = time.perf_counter_ns() / 1000
    retval = func()
    end_us = time.perf_counter_ns() / 1000
    return end_us - start_us, retval


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

    data: dict          # dim -> np.ndarray (NUM_ELEMENTS x dim, float32)
    data_bytes: dict    # dim -> list[bytes]   (one per element, for SQL inserts)
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

        return cls(data, data_bytes, queries, query_bytes, correct_labels)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BenchmarkResult:
    product: str
    distance_type: str
    dim: int
    insert_time_us: float
    search_time_us: float
    recall_rate: float
    ef_construction: Optional[int] = None
    M: Optional[int] = None
    ef_search: Optional[int] = None


@dataclasses.dataclass
class ResultTable:
    """Render a list of ``BenchmarkResult`` as a rich Table."""

    results: List[BenchmarkResult]

    def __rich_console__(self, console: Console,
                         options: ConsoleOptions) -> RenderResult:
        if not self.results:
            yield Table()
            return

        show_hnsw = self.results[0].ef_construction is not None

        table = Table()
        table.add_column("product\nname")
        table.add_column("distance\ntype")
        table.add_column("vector\ndimension")
        if show_hnsw:
            table.add_column("ef\nconstruction")
            table.add_column("M")
            table.add_column("ef\nsearch")
        table.add_column("insert_time\nper vector")
        table.add_column("search_time\nper query")
        table.add_column("recall\nrate")

        for r in self.results:
            row = [r.product, r.distance_type, str(r.dim)]
            if show_hnsw:
                row += [str(r.ef_construction), str(r.M), str(r.ef_search)]
            row += [
                f"{r.insert_time_us:.2f} us",
                f"{r.search_time_us:.2f} us",
                f"{r.recall_rate * 100:.2f}%",
            ]
            table.add_row(*row)

        yield table


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class Backend:
    """Common interface for every system under test.

    A backend handles the lifecycle of one (distance_type, dim, hnsw_params)
    configuration. ``run_backend`` drives it end to end; the backend just
    declares its capabilities and implements the four lifecycle hooks.
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

    def _insert_rows(self) -> None:
        rows = [(i, self.data.data_bytes[self._dim][i])
                for i in range(NUM_ELEMENTS)]
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
            f"hnsw(max_elements={NUM_ELEMENTS}, "
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
            max_elements=NUM_ELEMENTS,
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

    @property
    def include_in_query_plot(self) -> bool:  # type: ignore[override]
        # sqlite-vec's brute force search dominates the plot at high N.
        return NUM_ELEMENTS < 10000


class MilvusLiteBackend(Backend):
    name = "milvus_lite"
    plot_label = "milvuslite_{distance_type}{ef_search}"

    def __init__(self, data: BenchmarkData, db_path: str = "./milvus_lite_demo.db") -> None:
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
            for i in range(NUM_ELEMENTS)
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


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PlotData:
    time_taken_us: float
    column: str


@dataclasses.dataclass
class BenchmarkSuite:
    """Accumulates results and per-dim plot points across backends."""

    plot_insertion: dict = dataclasses.field(default_factory=lambda: defaultdict(list))
    plot_query: dict = dataclasses.field(default_factory=lambda: defaultdict(list))

    def record_insertion(self, dim: int, label: str, time_us: float) -> None:
        self.plot_insertion[dim].append(PlotData(time_us, label))

    def record_query(self, dim: int, label: str, time_us: float) -> None:
        self.plot_query[dim].append(PlotData(time_us, label))

    def write_plots(self, num_elements: int) -> None:
        for kind, points in (("insertion", self.plot_insertion),
                             ("query", self.plot_query)):
            if not points:
                continue
            dims = sorted(points.keys())
            columns = ["dim"] + [p.column for p in points[dims[0]]]
            rows = [[d] + [p.time_taken_us for p in points[d]] for d in dims]
            plot.plot(f"vector_{kind}_{num_elements}_vectors", columns, rows)


def run_backend(backend: Backend, data: BenchmarkData,
                suite: BenchmarkSuite,
                distance_types: Sequence[str] = DISTANCE_TYPES,
                dims: Sequence[int] = DIMS,
                hnsw_params: Sequence[Tuple[int, int]] = HNSW_PARAMS,
                ef_values: Sequence[int] = EF_SEARCH_VALUES,
                ) -> List[BenchmarkResult]:
    """Run one backend across all configurations; return its results."""
    param_combos: Iterable[Tuple[Optional[int], Optional[int]]] = (
        hnsw_params if backend.uses_hnsw_params else [(None, None)]
    )
    ef_iter: Sequence[Optional[int]] = (
        ef_values if backend.supports_ef_search else [None]
    )
    relevant_distances = [
        d for d in distance_types if d in backend.supported_distances
    ]

    results: List[BenchmarkResult] = []

    for distance_type in relevant_distances:
        for dim in dims:
            for ef_construction, M in param_combos:
                backend.setup(distance_type, dim, ef_construction, M)
                try:
                    insert_total_us, _ = timeit(
                        lambda: backend.do_insert(distance_type, dim))
                    insert_time_us = insert_total_us / NUM_ELEMENTS
                    suite.record_insertion(
                        dim, backend.insertion_label(distance_type),
                        insert_time_us)

                    for ef in ef_iter:
                        search_total_us, query_results = timeit(
                            lambda: backend.do_search(distance_type, dim, ef))
                        search_time_us = search_total_us / NUM_QUERIES
                        recall = compute_recall(
                            query_results,
                            data.correct_labels[distance_type][dim], K)

                        results.append(BenchmarkResult(
                            product=backend.name,
                            distance_type=distance_type,
                            dim=dim,
                            insert_time_us=insert_time_us,
                            search_time_us=search_time_us,
                            recall_rate=recall,
                            ef_construction=ef_construction,
                            M=M,
                            ef_search=ef,
                        ))
                        if backend.include_in_query_plot:
                            suite.record_query(
                                dim,
                                backend.query_label(distance_type, ef),
                                search_time_us)
                finally:
                    backend.teardown()

    return results


# ---------------------------------------------------------------------------
# Optional backend loaders (gated by env vars / platform)
# ---------------------------------------------------------------------------


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0") != "0"


def collect_optional_backends(cursor: sqlite3.Cursor, conn: sqlite3.Connection,
                              data: BenchmarkData) -> List[Backend]:
    backends: List[Backend] = []
    if not is_supported_platform():
        return backends

    if _env_flag("BENCHMARK_VSS"):
        # sqlite_vss is not self-contained; on Debian/Ubuntu install:
        #   sudo apt-get install -y libgomp1 libatlas-base-dev liblapack-dev
        import sqlite_vss
        sqlite_vss.load(conn)
        backends.append(SqliteVssBackend(cursor, data))

    if _env_flag("BENCHMARK_SQLITE_VEC"):
        import sqlite_vec
        conn.load_extension(sqlite_vec.loadable_path())
        backends.append(SqliteVecBackend(cursor, data))

    if _env_flag("BENCHMARK_MILVUS_LITE"):
        backends.append(MilvusLiteBackend(data))

    return backends


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    vectorlite_path = os.environ.get(
        "VECTORLITE_PATH", vectorlite_py.vectorlite_path())
    if vectorlite_path != vectorlite_py.vectorlite_path():
        console.print(f"Using local vectorlite: {vectorlite_path}")

    # ``isolation_level=None`` puts sqlite3 in autocommit mode so that the
    # explicit ``BEGIN TRANSACTION`` / ``COMMIT`` statements issued by
    # ``_SqlBackend._insert_rows`` are honoured rather than wrapped in an
    # implicit transaction by the driver.
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.enable_load_extension(True)
    conn.load_extension(vectorlite_path)
    cursor = conn.cursor()

    console.print(
        f"Benchmarking with {NUM_ELEMENTS} random vectors. "
        f"{NUM_QUERIES} {K}-nearest-neighbour queries per case."
    )

    data = BenchmarkData.generate(
        DIMS, NUM_ELEMENTS, NUM_QUERIES, K, DISTANCE_TYPES)

    suite = BenchmarkSuite()

    # The order here defines column order in plot CSV / PNG output.
    primary_backends: List[Tuple[Backend, str]] = [
        (VectorliteBackend(cursor, data),
         "vectorlite (HNSW virtual table)"),
        (HnswlibBackend(data),
         "hnswlib (in-memory, comparison)"),
        (VectorliteBruteForceBackend(cursor, data),
         "vectorlite brute force "
         "(SELECT ... ORDER BY vector_distance(...))"),
    ]

    optional_descriptions = {
        "sqlite_vss": "sqlite_vss (comparison)",
        "sqlite_vec": "sqlite_vec (comparison)",
        "milvus_lite": "milvus-lite (comparison)",
    }

    all_backends: List[Tuple[Backend, str]] = list(primary_backends)
    for backend in collect_optional_backends(cursor, conn, data):
        all_backends.append(
            (backend, optional_descriptions.get(backend.name, backend.name)))

    for backend, description in all_backends:
        console.print(f"Benchmarking {description}.")
        results = run_backend(backend, data, suite)
        if results:
            console.print(ResultTable(results))

    suite.write_plots(NUM_ELEMENTS)


if __name__ == "__main__":
    main()
