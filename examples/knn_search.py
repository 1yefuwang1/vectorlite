import sqlite3
import numpy as np
"""
Example of using sqlite-vector extension to perform kNN search on a table of vectors.
"""

dim = 16
num_elements = 10000

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))

# Declaring index
conn = sqlite3.connect(':memory:')
conn.enable_load_extension(True)

conn.load_extension('../build/dev/libsqlite-vector.so')

cur = conn.cursor()

print('Trying to create virtual table for vector search.')
# Below statement creates a virtual table named 'x' with one column called 'my_embedding' which has a dimension of 16, a an implict rowid column.
# my_embedding holds vectors that can be searched based on L2 distance using HNSW index.
cur.execute(f'create virtual table x using vector_search(my_embedding({dim}, "l2"), hnsw(max_elements={num_elements},ef_construction=100,M=16))')
print("Adding %d elements" % (len(data)))
for i in range(num_elements):
    cur.execute('insert into x (rowid, my_embedding) values (?, ?)', (i, data[i].tobytes()))

# Search for 10 nearest neighbors of data[0]
cur.execute('select rowid, distance from x where knn_search(my_embedding, knn_param(?, ?))', (data[0].tobytes(), 10))
print(cur.fetchall())

# Optionally, rowid can be filtered using 'rowid in (...)'. The rowid filter is pushed down to HNSW index and is efficient.
# Please note: 'rowid in (...)' is only supported for sqlite3 version >= 3.38.0
cur.execute(f'select rowid, distance from x where knn_search(my_embedding, knn_param(?, ?)) and rowid in (0, 1)', (data[0].tobytes(), 10))
print(cur.fetchall())

cur.close()