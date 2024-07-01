import sqlite3
from typing import Optional
import apsw 
import numpy as np
import os
import timeit
"""
Example of using vectorlite extension to perform KNN search on a table of vectors.
"""

use_apsw = os.environ.get('USE_BUILTIN_SQLITE3', '0') == '0'

DIM = 1000
NUM_ELEMENTS = 10000

# Generating sample data
data = np.float32(np.random.random((NUM_ELEMENTS, DIM)))

def create_connection():
    # create connection to in-memory database
    conn = apsw.Connection(':memory:') if use_apsw else sqlite3.connect(':memory:')
    conn.enable_load_extension(True)
    conn.load_extension('../build/release/vectorlite.so')
    return conn

conn = create_connection()

cur = conn.cursor()

index_file_path = 'index_file.bin'
if os.path.exists(index_file_path):
    print('Removing existing index file.')
    os.remove(index_file_path)

print('Trying to create virtual table for vector search.')
# Below statement creates a virtual table named 'x' with one column called 'my_embedding' which has a dimension of 1000.
# my_embedding holds vectors that can be searched based on L2 squared distance(which is the default distance) using HNSW index.
# One can explicitly specify which distance type to use. 
# cur.execute(f'create virtual table x using vectorlite(my_embedding float32[{dim}]  l2, hnsw(max_elements={num_elements}))')
# cur.execute(f'create virtual table x using vectorlite(my_embedding float32[{dim}]  cosine, hnsw(max_elements={num_elements}))')
# cur.execute(f'create virtual table x using vectorlite(my_embedding float32[{dim}]  ip, hnsw(max_elements={num_elements}))')
# Note: the virtual table has an implict rowid column, which is used to uniquely identify a vector and as a "foreign key" to relate to the vector's metadata.
# For example, you could have another table with metadata columns and rowid column with the same value as the corresponding rowid in a vectorlite virutal table.
# The "hnsw(max_elements=10000)" part configures HNSW index parameters, which can be used to tune the performance of the index.
# Please check https://github.com/nmslib/hnswlib/blob/v0.8.0/ALGO_PARAMS.md for more information about HNSW parameters.
# Only max_elements is required.
# the 3rd argument is an optional index file path. Vectorlite will try to load the index from the file if it exists and save the index to the file when database connetion closes.
# If the index file path is not provided, the index will be stored in memory and will be lost when the database connection closes.
# The index file will be deleted if the table is dropped.
cur.execute(f'create virtual table x using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}), {index_file_path})')

print("Adding %d vectors" % (len(data)))
def insert_vectors():
    # rowid MUST be explicitly set when inserting vectors and cannot be auto-generated.
    cur.executemany('insert into x (rowid, my_embedding) values (?, ?)', [(i, data[i].tobytes()) for i in range(NUM_ELEMENTS)])

time_taken = timeit.timeit(insert_vectors, number=1)
print(f'time taken for inserting {NUM_ELEMENTS} vectors: {time_taken} seconds')

# Search for 10 nearest neighbors of data[0]
# distance here is a hidden column. The result is already sorted by distance in ascending order
# vectorlite treat a vector as an array of float32. So, we need to convert it to bytes before passing it to the query.
cur.execute('select rowid, distance from x where knn_search(my_embedding, knn_param(?, ?))', (data[0].tobytes(), 10))
print(f'10 nearest neighbors of row 0 is {cur.fetchall()}')

# test recall rate by iterating over all vectors and checking whether the nearest neighbor is itself.
def test_recall(table_name: str, vector_name: str,ef: Optional[int] = None):
    matches = 0
    for i in range(NUM_ELEMENTS):
        if ef is None:
            cur.execute(f'select rowid from {table_name} where knn_search({vector_name}, knn_param(?, ?))', (data[i].tobytes(), 1))
        else:
            cur.execute(f'select rowid from {table_name} where knn_search({vector_name}, knn_param(?, ?, ?))', (data[i].tobytes(), 1, ef))
        if cur.fetchone()[0] == i:
            matches += 1
    recall_rate = matches / NUM_ELEMENTS
    print(f'recall rate with ef = {10 if ef is None else ef} is {recall_rate * 100}%')

# calculate recall rate
time_taken = timeit.timeit(lambda: test_recall('x', 'my_embedding'), number=1)
print(f'time taken for calculating recall rate: {time_taken} seconds')

# Optionally, we can set ef to a higher value to improve recall rate without rebuilding index, at the cost of higher search time.
# It can be achieved by passing ef to the 3rd argument of knn_param.
# For more info on ef, please check https://github.com/nmslib/hnswlib/blob/v0.8.0/ALGO_PARAMS.md
# The default value of ef is 10. In this example, we set ef to 32. 
# Note: ef is not part of the index, modifying it is an imperative operation. If it is not set in later queries, it will always be 32.
time_taken = timeit.timeit(lambda: test_recall('x', 'my_embedding', 32), number=1)
print(f'time taken for calculating recall rate with ef=32: {time_taken} seconds')


if use_apsw:
    # Optionally, rowid can be filtered using 'rowid in (...)'. The rowid filter is pushed down to HNSW index and is efficient.
    # Please note: 'rowid in (...)' is only supported for sqlite3 version >= 3.38.0. The built-in sqlite3 module usually doesn't support it. 
    # Please use apsw module if you want to use rowid filtering.
    cur.execute(f'select rowid, distance from x where knn_search(my_embedding, knn_param(?, ?)) and rowid in (0, 1)', (data[0].tobytes(), 10))
    print(cur.fetchall())

    # Multiple 'rowid in (...)' is not supported.
    # cur.execute(f'select rowid, distance from x where knn_search(my_embedding, knn_param(?, ?)) and rowid in (0, 1) and rowid in (1, 2)', (data[0].tobytes(), 10))
    # print(cur.fetchall())


    # delete a row
    cur.execute(f'delete from x where rowid = 1')
    cur.execute(f'select rowid, vector_to_json(my_embedding) from x where rowid = 1')
    assert (len(cur.fetchall()) == 0)
    print('row 1 is deleted')

    # Because vectors are bytes, we need to use vector_to_json to print a vector.
    cur.execute(f'select rowid, vector_to_json(my_embedding) from x where rowid = 2')
    result = cur.fetchone()
    vector2 = result[1]
    print(f'vector of row 2 is {vector2}')

    # update a row
    cur.execute(f'update x set my_embedding = ? where rowid = 2', (data[0].tobytes(),))
    print('vector of row 2 is updated')

    cur.execute(f'select rowid, vector_to_json(my_embedding) from x where rowid = 2')
    result = cur.fetchone()
    vector2_updated = result[1]
    assert (vector2 != vector2_updated)

conn.close()
# If database connection is closed, the index will be saved to the index file.
assert os.path.exists(index_file_path) and os.path.getsize(index_file_path) > 0
conn = create_connection()
cur = conn.cursor()
# We could load the index from the index file by providing the index file path when creating the virtual table.
# When loading the index from the file, vector dimension MUST stay the same. But table name, vector name can be changed. 
# HNSW parameters can't be changed even if different values are set, they will be owverwritten by the value from the index file, 
# except that max_elements can be increased.
# Distance type can be changed too.
cur.execute(f'create virtual table table_reloaded using vectorlite(vec_reloaded float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS * 2}), {index_file_path})')
print(f'index is loaded from {index_file_path} with higher max_elements.')
# Because the index is loaded from the file, we can query the table without inserting any data.
result = cur.execute('select rowid, distance from table_reloaded where knn_search(vec_reloaded, knn_param(?, ?))', (data[0].tobytes(), 10)).fetchall()
print(f'10 nearest neighbors of row 0 is {result}')

# test recall again
# calculate recall rate. Note: ef defaults to 10 after reloading.
time_taken = timeit.timeit(lambda: test_recall('table_reloaded', 'vec_reloaded'), number=1)
print(f'time taken for calculating recall rate after reloading from file: {time_taken} seconds')

time_taken = timeit.timeit(lambda: test_recall('table_reloaded', 'vec_reloaded', 32), number=1)
print(f'time taken for calculating recall rate after reloading from file with ef=32: {time_taken} seconds')

# index file will be deleted when the table is dropped.
cur.execute('drop table table_reloaded')