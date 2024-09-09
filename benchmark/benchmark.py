import json
import time
from typing import Literal, Optional, List
import numpy as np
import vectorlite_py
import apsw
import dataclasses
import hnswlib
from rich.console import Console, ConsoleOptions, RenderResult
from rich.table import Table
import os
from collections import defaultdict


"""
Benchamrk vectorlite's performance and recall rate using method described in https://github.com/nmslib/hnswlib/blob/v0.8.0/TESTING_RECALL.md
"""


# Roll our own timeit function to measure time in us and get return value of the func.
# Why Python's built-in timeit.timeit is not used:
# 1. it compiles the code passed to it, which is unnecessary overhead.
# 2. func's return value cannot be obtained directly
def timeit(func):
    start_us = time.perf_counter_ns() / 1000
    retval = func()
    end_us = time.perf_counter_ns() / 1000
    return end_us - start_us, retval

vectorlite_path = os.environ.get("VECTORLITE_PATH", vectorlite_py.vectorlite_path())

if vectorlite_path != vectorlite_py.vectorlite_path():
    print(f"Using local vectorlite: {vectorlite_path}")

conn = apsw.Connection(":memory:")
conn.enable_load_extension(True)  # enable extension loading
conn.load_extension(vectorlite_path)  # loads vectorlite

cursor = conn.cursor()

NUM_ELEMENTS = int(os.environ.get("NUM_ELEMENTS", 3000)) # number of vectors 
NUM_QUERIES = 100  # number of queries

DIMS = [128, 512, 1536, 3000]
data = {dim: np.float32(np.random.random((NUM_ELEMENTS, dim))) for dim in DIMS}
data_bytes = {dim: [data[dim][i].tobytes() for i in range(NUM_ELEMENTS)] for dim in DIMS}

query_data = {dim: np.float32(np.random.random((NUM_QUERIES, dim))) for dim in DIMS}
query_data_bytes = {dim: [query_data[dim][i].tobytes() for i in range(NUM_QUERIES)] for dim in DIMS}
query_data_for_milvus = {dim: [query_data[dim][i].tolist() for i in range(NUM_QUERIES)] for dim in DIMS}

# search for k nearest neighbors in this benchmark
k = 10

# (ef_construction, M)
hnsw_params = [(100, 30)]

# ef_search
efs = [10, 50, 100]


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
console.print(f"Benchmarking using {NUM_ELEMENTS} randomly vectors. {NUM_QUERIES} {k}-neariest neighbor queries will be performed on each case.")

@dataclasses.dataclass
class BenchmarkResult:
    distance_type: Literal["l2", "cosine"]
    dim: int
    ef_construction: Optional[int]
    M: Optional[int]
    ef_search: Optional[int]
    insert_time_us: float  # in micro seconds, per vector
    search_time_us: float  # in micro seconds, per query
    recall_rate: float
    product: Optional[str]


@dataclasses.dataclass
class ResultTable:
    results: List[BenchmarkResult]

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        table = Table()
        table.add_column("product\nname")
        table.add_column("distance\ntype")
        table.add_column("vector\ndimension")
        if self.results[0].ef_construction is not None:
            table.add_column("ef\nconstruction")
            table.add_column("M")
            table.add_column("ef\nsearch")
        table.add_column("insert_time\nper vector")
        table.add_column("search_time\nper query")
        table.add_column("recall\nrate")
        for result in self.results:
            if self.results[0].ef_construction is not None:
                table.add_row(
                    result.product,
                    result.distance_type,
                    str(result.dim),
                    str(result.ef_construction),
                    str(result.M),
                    str(result.ef_search),
                    f"{result.insert_time_us:.2f} us",
                    f"{result.search_time_us:.2f} us",
                    f"{result.recall_rate * 100:.2f}%",
                )
            else:
                table.add_row(
                    result.product,
                    result.distance_type,
                    str(result.dim),
                    f"{result.insert_time_us:.2f} us",
                    f"{result.search_time_us:.2f} us",
                    f"{result.recall_rate * 100:.2f}%",
                )
        yield table

@dataclasses.dataclass
class PlotData:
    time_taken_us: float
    column: str

benchmark_milvus_results = []
benchmark_results = []
plot_data_for_insertion = defaultdict(list)
plot_data_for_query = defaultdict(list)

def transactional(func):
    # def wrapper():
    #     with conn:
    #         func()
    # return wrapper
    def wrapper():
        cursor.execute("BEGIN TRANSACTION;")
        func()
        cursor.execute("COMMIT;")
    return wrapper



def benchmark(distance_type, dim, ef_constructoin, M):
    result = BenchmarkResult(distance_type, dim, ef_constructoin, M, 0, 0, 0, 0, "vectorLite")
    table_name = f"table_{distance_type}_{dim}_{ef_constructoin}_{M}"
    cursor.execute(
        f"create virtual table {table_name} using vectorlite(embedding float32[{dim}] {distance_type}, hnsw(max_elements={NUM_ELEMENTS}, ef_construction={ef_constructoin}, M={M}))"
    )

    # measure insert time
    insert_time_us, _ = timeit(
        transactional(lambda: cursor.executemany(
            f"insert into {table_name}(rowid, embedding) values (?, ?)",
            [(i, data_bytes[dim][i]) for i in range(NUM_ELEMENTS)],
        ))
    )
    result.insert_time_us = insert_time_us / NUM_ELEMENTS
    plot_data_for_insertion[dim].append(PlotData(result.insert_time_us, f"vectorlite_{distance_type}"))

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
        plot_data_for_query[dim].append(PlotData(result.search_time_us, f"vectorlite_{distance_type}_ef_{ef}"))
    cursor.execute(f"drop table {table_name}")

for distance_type in distance_types:
    for dim in DIMS:
        for ef_construction, M in hnsw_params:
            benchmark(distance_type, dim, ef_construction, M)


result_table = ResultTable(benchmark_results)
console.print(result_table)

hnswlib_benchmark_results = []
console.print("Bencharmk hnswlib as comparison.")
def benchmark_hnswlib(distance_type, dim, ef_construction, M):
    result = BenchmarkResult(distance_type, dim, ef_construction, M, 0, 0, 0, 0, "vectorLite")
    hnswlib_index = hnswlib.Index(space=distance_type, dim=dim)
    hnswlib_index.init_index(max_elements=NUM_ELEMENTS, ef_construction=ef_construction, M=M)

    # measure insert time
    insert_time_us, _ = timeit(
        lambda: hnswlib_index.add_items(data[dim])
    )
    result.insert_time_us = insert_time_us / NUM_ELEMENTS
    plot_data_for_insertion[dim].append(PlotData(result.insert_time_us, f"hnswlib_{distance_type}"))

    for ef in efs:
        hnswlib_index.set_ef(ef)
        def search():
            result = []
            for i in range(NUM_QUERIES):
                labels, distances = hnswlib_index.knn_query(query_data[dim][i], k=k)
                result.append(labels)
            return result

        search_time_us, results = timeit(search)
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
        hnswlib_benchmark_results.append(result)
        plot_data_for_query[dim].append(PlotData(result.search_time_us, f"hnswlib_{distance_type}_ef_{ef}"))
    del hnswlib_index

for distance_type in distance_types:
    for dim in DIMS:
        for ef_construction, M in hnsw_params:
            benchmark_hnswlib(distance_type, dim, ef_construction, M)

hnswlib_result_table = ResultTable(hnswlib_benchmark_results)
console.print(hnswlib_result_table)

brute_force_benchmark_results = []

console.print("Bencharmk vectorlite brute force(select rowid from my_table order by vector_distance(query_vector, embedding, 'l2')) as comparison.")

def benchmark_brute_force(dim: int):
    benchmark_result = BenchmarkResult("l2", dim, None, None, None, 0, 0, 0, "vectorLite")
    table_name = f"table_vectorlite_bf_{dim}"
    cursor.execute(
        f"create table {table_name}(rowid integer primary key, embedding blob)"
    )

    insert_time_us, _ = timeit(
        transactional(
            lambda: cursor.executemany(
                f"insert into {table_name}(rowid, embedding) values (?, ?)",
                [(i, data_bytes[dim][i]) for i in range(NUM_ELEMENTS)],
        ))
    )
    benchmark_result.insert_time_us = insert_time_us / NUM_ELEMENTS
    plot_data_for_insertion[dim].append(PlotData(benchmark_result.insert_time_us, "vectorlite_scalar_brute_force"))

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
    plot_data_for_query[dim].append(PlotData(benchmark_result.search_time_us, "vectorlite_scalar_brute_force"))
    cursor.execute(f"drop table {table_name}")

for dim in DIMS:
    benchmark_brute_force(dim)
brute_force_table = ResultTable(brute_force_benchmark_results)
console.print(brute_force_table)


# Benchmark sqlite_vss as compariso.
# pip install sqlite-vss
import platform

benchmark_vss = os.environ.get("BENCHMARK_VSS", "0") != "0"
if benchmark_vss and (platform.system().lower() == "linux" or platform.system().lower() == "darwin"):
    # note sqlite_vss is not self-contained.
    # Need to install dependencies manually using: sudo apt-get install -y libgomp1 libatlas-base-dev liblapack-dev
    console.print("Bencharmk sqlite_vss as comparison.")
    import sqlite_vss

    sqlite_vss.load(conn)

    vss_benchmark_results = []

    def benchmark_sqlite_vss(dim: int):
        benchmark_result = BenchmarkResult("l2", dim, None, None, None, 0, 0, 0, "vectorLite")
        table_name = f"table_vss_{dim}"
        cursor.execute(
            f"create virtual table {table_name} using vss0(embedding({dim}))"
        )

        # measure insert time
        insert_time_us, _ = timeit(
            transactional(
                lambda: cursor.executemany(
                    f"insert into {table_name}(rowid, embedding) values (?, ?)",
                    [(i, data_bytes[dim][i]) for i in range(NUM_ELEMENTS)],
            ))
        )
        benchmark_result.insert_time_us = insert_time_us / NUM_ELEMENTS
        # insertion for sqlite_vss is so slow that it makes other data points insignificant
        plot_data_for_insertion[dim].append(PlotData(benchmark_result.insert_time_us, "sqlite_vss"))

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
        plot_data_for_query[dim].append(PlotData(benchmark_result.search_time_us, "sqlite_vss"))
        cursor.execute(f"drop table {table_name}")

    for dim in DIMS:
        benchmark_sqlite_vss(dim)

    vss_result_table = ResultTable(vss_benchmark_results)
    console.print(vss_result_table)

# benchmark sqlite-vec
# pip install sqlite-vec
benchmark_sqlite_vec = os.environ.get("BENCHMARK_SQLITE_VEC", "0") != "0"
if benchmark_sqlite_vec and (platform.system().lower() == "linux" or platform.system().lower() == "darwin"):
    # VssBenchamrkResult and VssResultTable can be reused
    vec_benchmark_results = []
    console.print("Bencharmk sqlite_vec as comparison.")
    import sqlite_vec

    conn.load_extension(sqlite_vec.loadable_path())

    def benchmark_sqlite_vec(dim: int):
        benchmark_result = BenchmarkResult("l2", dim, None, None, None, 0, 0, 0, "vectorLite")
        table_name = f"table_vec_{dim}"
        cursor.execute(
            f"create virtual table {table_name} using vec0(rowid integer primary key, embedding float[{dim}])"
        )

        # measure insert time
        insert_time_us, _ = timeit(
            transactional(
                lambda: cursor.executemany(
                    f"insert into {table_name}(rowid, embedding) values (?, ?)",
                    [(i, data_bytes[dim][i]) for i in range(NUM_ELEMENTS)],
            ))
        )
        benchmark_result.insert_time_us = insert_time_us / NUM_ELEMENTS
        plot_data_for_insertion[dim].append(PlotData(benchmark_result.insert_time_us, "sqlite_vec"))

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
        if NUM_ELEMENTS < 10000:
            plot_data_for_query[dim].append(PlotData(benchmark_result.search_time_us, "sqlite_vec"))

        cursor.execute(f"drop table {table_name}")

    for dim in DIMS:
        benchmark_sqlite_vec(dim)

    vec_result_table = ResultTable(vec_benchmark_results)
    console.print(vec_result_table)


import os
from pymilvus import MilvusClient
import numpy as np

client = MilvusClient("./milvus_demo6.db")

def milvus_insert(client, collection_name, data):
    client.insert(collection_name=collection_name, data=data)

def milvus_insert_many(client, collection_name, dim):
    insert_data = [{"id": i, "embedding": data[dim][i].tolist()} for i in range(NUM_ELEMENTS)]
    client.insert(collection_name=collection_name, data=insert_data)

def milvus_search(client, collection_name, distance_type, search_data):
    res = client.search(
        collection_name=collection_name,
        data=[search_data],
        search_params={"metric_type": distance_type.upper()}, # Search parameters
    )
    rowids = [result['id'] for result in res[0]]
    return rowids

def milvus_create_table(client, collection_name, distance_type, dim):
    from pymilvus import DataType

    schema = client.create_schema(enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="embedding", 
        metric_type=distance_type.upper(),
    )
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        dimension=dim  # The vectors we will use in this demo has 384 dimensions
    )
    
    res = client.get_load_state(
        collection_name=collection_name
    )
    # print(client.describe_index(collection_name,"embedding"))
    # print(client.describe_index(collection_name,"id"))

console.print("Bencharmk milvuslite.")

def benchmark_milvus(distance_type, dim):
    result = BenchmarkResult(distance_type=distance_type, dim=dim, insert_time_us=0, search_time_us=0, recall_rate=0, product="milvusLite", ef_construction=None, M=None, ef_search=None)
    collection_name = f"collection_{distance_type}_{dim}"

    milvus_create_table(client=client, collection_name=collection_name, distance_type=distance_type, dim=dim)
    
    # measure insert time
    insert_time_us, _ = timeit(
        transactional(lambda: milvus_insert_many(client=client, collection_name=collection_name, dim=dim))
    )

    result.insert_time_us = insert_time_us / NUM_ELEMENTS
    plot_data_for_insertion[dim].append(PlotData(result.insert_time_us, f"milvuslite_{distance_type}"))


    def search():
        result = []
        for i in range(NUM_QUERIES):
            result.append(
                milvus_search(client=client, collection_name=collection_name, distance_type=distance_type, search_data=query_data_for_milvus[dim][i])
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
        search_time_us=search_time_us / NUM_QUERIES,
        recall_rate=recall_rate,
    )

    benchmark_milvus_results.append(result)
    plot_data_for_query[dim].append(PlotData(result.search_time_us, f"milvuslite_{distance_type}"))
    client.drop_collection(collection_name=collection_name)

for distance_type in distance_types:
    for dim in DIMS:
        benchmark_milvus(distance_type, dim)

console.print(ResultTable(benchmark_milvus_results))

import plot
def plot_figures():
    vector_insertion_columns = ["dim"] + [plot_data.column for plot_data in plot_data_for_insertion[DIMS[0]]]
    vector_insertion_data = [[dim] + [plot_data.time_taken_us for plot_data in plot_data_for_insertion[dim]] for dim in DIMS]
    plot.plot(f"vector_insertion_{NUM_ELEMENTS}_vectors", vector_insertion_columns, vector_insertion_data)

    vector_query_columns = ["dim"] + [plot_data.column for plot_data in plot_data_for_query[DIMS[0]]]
    vector_query_data = [[dim] + [plot_data.time_taken_us for plot_data in plot_data_for_query[dim]] for dim in DIMS]
    plot.plot(f"vector_query_{NUM_ELEMENTS}_vectors", vector_query_columns, vector_query_data)

plot_figures()