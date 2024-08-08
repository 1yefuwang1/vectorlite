import time
from typing import Literal, Optional
import numpy as np
import vectorlite_py
import apsw
import dataclasses
import hnswlib
from rich.console import Console, ConsoleOptions, RenderResult
from rich.table import Table
import os


"""
Benchamrk vectorlite's performance and recall rate using method described in https://github.com/nmslib/hnswlib/blob/v0.8.0/TESTING_RECALL.md
"""


# Roll our own timeit function to measure time in us and get return value of the func.
# Why Python's built-in timeit.timeit is not used:
# 1. it includes unnecessary overheads, because compiles the code passed to it
# 2. func's return value cannot be obtained directly
def timeit(func):
    start_us = time.perf_counter_ns() / 1000
    retval = func()
    end_us = time.perf_counter_ns() / 1000
    return end_us - start_us, retval


conn = apsw.Connection(":memory:")
conn.enable_load_extension(True)  # enable extension loading
# conn.load_extension(vectorlite_py.vectorlite_path())  # loads vectorlite
conn.load_extension('build/release/vectorlite')  # loads vectorlite

cursor = conn.cursor()

NUM_ELEMENTS = 10000  # number of vectors
NUM_QUERIES = 100  # number of queries

DIMS = [256, 1024]
data = {dim: np.float32(np.random.random((NUM_ELEMENTS, dim))) for dim in DIMS}
data_bytes = {dim: [data[dim][i].tobytes() for i in range(NUM_ELEMENTS)] for dim in DIMS}

query_data = {dim: np.float32(np.random.random((NUM_QUERIES, dim))) for dim in DIMS}
query_data_bytes = {dim: [query_data[dim][i].tobytes() for i in range(NUM_QUERIES)] for dim in DIMS}

# search for k nearest neighbors in this benchmark
k = 10

# (ef_construction, M)
hnsw_params = [(200, 64)]

# ef_search
efs = [10, 50, 100, 150]


# 'ip'(inner product) is not tested as it is not an actual metric that measures the distance between two vectors
distance_types = ["l2", "cosine"]

# Calculate correct results using Brute Force index
correct_labels = {}
for distance_type in distance_types:
    correct_labels[distance_type] = {}
    for dim in DIMS:
        bf_index = hnswlib.BFIndex(space=distance_type, dim=dim)
        bf_index.init_index(max_elements=NUM_ELEMENTS)
        bf_index.add_items(data[dim])

        labels, distances = bf_index.knn_query(query_data[dim], k=k)
        assert len(labels) == NUM_QUERIES and len(labels[0]) == k
        correct_labels[distance_type][dim] = labels
        del bf_index

console = Console()

@dataclasses.dataclass
class BenchmarkResult:
    distance_type: Literal["l2", "cosine"]
    dim: int
    ef_construction: int
    M: int
    ef_search: int
    insert_time_us: float  # in micro seconds, per vector
    search_time_us: float  # in micro seconds, per query
    recall_rate: float  # in micro seconds


@dataclasses.dataclass
class ResultTable:
    results: list[BenchmarkResult]

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        table = Table()
        table.add_column("distance\ntype")
        table.add_column("vector\ndimension")
        table.add_column("ef\nconstruction")
        table.add_column("M")
        table.add_column("ef\nsearch")
        table.add_column("insert_time\nper vector")
        table.add_column("search_time\nper query")
        table.add_column("recall\nrate")
        for result in self.results:
            table.add_row(
                result.distance_type,
                str(result.dim),
                str(result.ef_construction),
                str(result.M),
                str(result.ef_search),
                f"{result.insert_time_us:.2f} us",
                f"{result.search_time_us:.2f} us",
                f"{result.recall_rate * 100:.2f}%",
            )
        yield table


benchmark_results = []


def benchmark(distance_type, dim, ef_constructoin, M):
    result = BenchmarkResult(distance_type, dim, ef_constructoin, M, 0, 0, 0, 0)
    table_name = f"table_{distance_type}_{dim}_{ef_constructoin}_{M}"
    cursor.execute(
        f"create virtual table {table_name} using vectorlite(embedding float32[{dim}] {distance_type}, hnsw(max_elements={NUM_ELEMENTS}, ef_construction={ef_constructoin}, M={M}))"
    )

    # measure insert time
    insert_time_us, _ = timeit(
        lambda: cursor.executemany(
            f"insert into {table_name}(rowid, embedding) values (?, ?)",
            [(i, data_bytes[dim][i]) for i in range(NUM_ELEMENTS)],
        )
    )
    result.insert_time_us = insert_time_us / NUM_ELEMENTS

    for ef in efs:

        def search():
            result = []
            for i in range(NUM_QUERIES):
                result.append(
                    cursor.execute(
                        f"select rowid from {table_name} where knn_search(embedding, knn_param(?, ?, ?))",
                        (query_data_bytes[dim][i], k, ef),
                    ).fetchall()
                )
            return result

        search_time_us, results = timeit(search)
        # console.log(results)
        recall_rate = np.mean(
            [
                np.intersect1d(results[i], correct_labels[distance_type][dim][i]).size
                / k
                for i in range(NUM_QUERIES)
            ]
        )
        result = dataclasses.replace(
            result,
            ef_search=ef,
            search_time_us=search_time_us / NUM_QUERIES,
            recall_rate=recall_rate,
        )
        benchmark_results.append(result)


for distance_type in distance_types:
    for dim in DIMS:
        for ef_construction, M in hnsw_params:
            benchmark(distance_type, dim, ef_construction, M)


result_table = ResultTable(benchmark_results)
console.print(result_table)


@dataclasses.dataclass
class BruteForceBenchmarkResult:
    dim: int
    insert_time_us: float  # in micro seconds, per vector
    search_time_us: float  # in micro seconds, per query
    recall_rate: float  # in micro seconds


@dataclasses.dataclass
class BruteForceResultTable:
    results: list[BruteForceBenchmarkResult]

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        table = Table()
        table.add_column("vector dimension")
        table.add_column("insert_time(per vector)")
        table.add_column("search_time(per query)")
        table.add_column("recall_rate")
        for result in self.results:
            table.add_row(
                str(result.dim),
                f"{result.insert_time_us:.2f} us",
                f"{result.search_time_us:.2f} us",
                f"{result.recall_rate * 100:.2f}%",
            )
        yield table


brute_force_benchmark_results = []


def benchmark_brute_force(dim: int):
    benchmark_result = BruteForceBenchmarkResult(dim, 0, 0, 0)
    table_name = f"table_vectorlite_bf_{dim}"
    cursor.execute(
        f"create table {table_name}(rowid integer primary key, embedding blob)"
    )

    insert_time_us, _ = timeit(
        lambda: cursor.executemany(
            f"insert into {table_name}(rowid, embedding) values (?, ?)",
            [(i, data_bytes[dim][i]) for i in range(NUM_ELEMENTS)],
        )
    )
    benchmark_result.insert_time_us = insert_time_us / NUM_ELEMENTS

    def search():
        result = []
        for i in range(NUM_QUERIES):
            result.append(
                cursor.execute(
                    f"select rowid from {table_name} order by vector_distance(?, embedding, 'l2') asc limit {k}",
                    [query_data_bytes[dim][i]],
                ).fetchall()
            )
        return result

    search_time_us, results = timeit(search)
    # console.log(results)
    benchmark_result.search_time_us = search_time_us / NUM_QUERIES
    recall_rate = np.mean(
        [
            np.intersect1d(results[i], correct_labels["l2"][dim][i]).size / k
            for i in range(NUM_QUERIES)
        ]
    )
    benchmark_result.recall_rate = recall_rate
    brute_force_benchmark_results.append(benchmark_result)

for dim in DIMS:
    benchmark_brute_force(dim)
brute_force_table = BruteForceResultTable(brute_force_benchmark_results)
console.print(brute_force_table)


# Benchmark sqlite_vss as compariso.
# pip install sqlite-vss
import platform

benchmark_vss = os.environ.get("BENCHMARK_VSS", "0") != "0"
if benchmark_vss and platform.system().lower() == "linux":
    # note sqlite_vss is not self-contained.
    # Need to install dependencies manually using: sudo apt-get install -y libgomp1 libatlas-base-dev liblapack-dev
    console.print("Bencharmk sqlite_vss as comparison.")
    import sqlite_vss

    sqlite_vss.load(conn)

    vss_benchmark_results = []

    def benchmark_sqlite_vss(dim: int):
        benchmark_result = BruteForceBenchmarkResult(dim, 0, 0, 0)
        table_name = f"table_vss_{dim}"
        cursor.execute(
            f"create virtual table {table_name} using vss0(embedding({dim}))"
        )

        # measure insert time
        insert_time_us, _ = timeit(
            lambda: cursor.executemany(
                f"insert into {table_name}(rowid, embedding) values (?, ?)",
                [(i, data_bytes[dim][i]) for i in range(NUM_ELEMENTS)],
            )
        )
        benchmark_result.insert_time_us = insert_time_us / NUM_ELEMENTS

        def search():
            result = []
            for i in range(NUM_QUERIES):
                result.append(
                    cursor.execute(
                        f"select rowid from {table_name} where vss_search(embedding, ?) limit {k}",
                        (query_data_bytes[dim][i],),
                    ).fetchall()
                )
            return result

        search_time_us, results = timeit(search)
        benchmark_result.search_time_us = search_time_us / NUM_QUERIES
        recall_rate = np.mean(
            [
                np.intersect1d(results[i], correct_labels["l2"][dim][i]).size / k
                for i in range(NUM_QUERIES)
            ]
        )
        benchmark_result.recall_rate = recall_rate
        vss_benchmark_results.append(benchmark_result)

    for dim in DIMS:
        benchmark_sqlite_vss(dim)

    vss_result_table = BruteForceResultTable(vss_benchmark_results)
    console.print(vss_result_table)

# benchmark sqlite-vec
# pip install sqlite-vec
benchmark_sqlite_vec = os.environ.get("BENCHMARK_SQLITE_VEC", "0") != "0"
if benchmark_sqlite_vec and platform.system().lower() == "linux":
    # VssBenchamrkResult and VssResultTable can be reused
    vec_benchmark_results = []
    console.print("Bencharmk sqlite_vec as comparison.")
    import sqlite_vec

    conn.load_extension(sqlite_vec.loadable_path())

    def benchmark_sqlite_vec(dim: int):
        benchmark_result = BruteForceBenchmarkResult(dim, 0, 0, 0)
        table_name = f"table_vec_{dim}"
        cursor.execute(
            f"create virtual table {table_name} using vec0(rowid integer primary key, embedding float[{dim}])"
        )

        # measure insert time
        insert_time_us, _ = timeit(
            lambda: cursor.executemany(
                f"insert into {table_name}(rowid, embedding) values (?, ?)",
                [(i, data_bytes[dim][i]) for i in range(NUM_ELEMENTS)],
            )
        )
        benchmark_result.insert_time_us = insert_time_us / NUM_ELEMENTS

        def search():
            result = []
            for i in range(NUM_QUERIES):
                result.append(
                    cursor.execute(
                        f"select rowid from {table_name} where embedding match ? and k = {k}",
                        (query_data_bytes[dim][i],),
                    ).fetchall()
                )
            return result

        search_time_us, results = timeit(search)
        benchmark_result.search_time_us = search_time_us / NUM_QUERIES
        recall_rate = np.mean(
            [
                np.intersect1d(results[i], correct_labels["l2"][dim][i]).size / k
                for i in range(NUM_QUERIES)
            ]
        )
        benchmark_result.recall_rate = recall_rate
        vec_benchmark_results.append(benchmark_result)

    for dim in DIMS:
        benchmark_sqlite_vec(dim)

    vec_result_table = BruteForceResultTable(vec_benchmark_results)
    console.print(vec_result_table)
