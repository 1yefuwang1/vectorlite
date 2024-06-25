import sqlite3
from typing import Optional
import apsw 
import numpy as np
import os
import timeit
"""
Example of using vectorlite extension to perform KNN search on a table of vectors.
"""

use_apsw = os.environ.get('USE_APSW', '0') == '1'

dim = 1000
num_elements = 10000

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))

# create connection to in-memory database
conn = apsw.Connection(':memory:') if use_apsw else sqlite3.connect(':memory:')
conn.enable_load_extension(True)

conn.load_extension('../build/release/libvectorlite.so')

cur = conn.cursor()

print('Trying to create virtual table for vector search.')
# Below statement creates a virtual table named 'x' with one column called 'my_embedding' which has a dimension of 1000.
# my_embedding holds vectors that can be searched based on L2 distance(which is the default distance) using HNSW index.
# Note: the virtual table has an implict rowid column, which is used as a "foreign key" to make connections to other tables.
# The "hnsw(max_elements=10000, ef_construction=50, M=32)" part configures HNSW index parameters.
# Please check https://github.com/nmslib/hnswlib/blob/v0.8.0/ALGO_PARAMS.md for more information about HNSW parameters.
# Actually, only max_elements is required.
cur.execute(f'create virtual table x using vectorlite(my_embedding float32[{dim}], hnsw(max_elements={num_elements},ef_construction=32,M=32))')
# You can explicitly create a vectorlite table specifying the space. The default space is 'l2' if not specified.
#cur.execute(f'create virtual table x using vectorlite(my_embedding float32[{dim}]  l2, hnsw(max_elements={num_elements},ef_construction=32,M=32))')
#cur.execute(f'create virtual table x using vectorlite(my_embedding float32[{dim}]  cosine, hnsw(max_elements={num_elements},ef_construction=32,M=32))')
#cur.execute(f'create virtual table x using vectorlite(my_embedding float32[{dim}]  ip, hnsw(max_elements={num_elements},ef_construction=32,M=32))')
print("Adding %d vectors" % (len(data)))
def insert_vectors():
    for i in range(num_elements):
        cur.execute('insert into x (rowid, my_embedding) values (?, ?)', (i, data[i].tobytes()))
time_taken = timeit.timeit(insert_vectors, number=1)
print(f'time taken for inserting {num_elements} vectors: {time_taken} seconds')

# Search for 10 nearest neighbors of data[0]
# distance here is a hidden column. The result is already sorted by distance in ascending order
# vectorlite treat a vector as an array of float32. So, we need to convert it to bytes before passing it to the query.
cur.execute('select rowid, distance from x where knn_search(my_embedding, knn_param(?, ?))', (data[0].tobytes(), 10))
print(f'10 nearest neighbors of row 0 is {cur.fetchall()}')

# test recall rate by iterating over all vectors and checking whether the nearest neighbor is itself.
def test_recall(ef: Optional[int] = None):
    matches = 0
    for i in range(num_elements):
        if ef is None:
            cur.execute('select rowid from x where knn_search(my_embedding, knn_param(?, ?))', (data[i].tobytes(), 1))
        else:
            cur.execute('select rowid from x where knn_search(my_embedding, knn_param(?, ?, ?))', (data[i].tobytes(), 1, ef))
        if cur.fetchone()[0] == i:
            matches += 1
    recall_rate = matches / num_elements
    print(f'recall rate with ef = {10 if ef is None else ef} is {recall_rate * 100}%')

# calculate recall rate
time_taken = timeit.timeit(test_recall, number=1)
print(f'time taken for calculating recall rate: {time_taken} seconds')

# Optionally, we can set ef to a higher value to improve recall rate, at the cost of higher search time.
# It can be achieved by passing ef to the 3rd argument of knn_param.
# For more info on ef, please check https://github.com/nmslib/hnswlib/blob/v0.8.0/ALGO_PARAMS.md
# The default value of ef is 10. In this example, we set ef to 32. 
# Note: if ef is not set in later queries, it will always be 32.
time_taken = timeit.timeit(lambda: test_recall(32), number=1)
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

cur.close()