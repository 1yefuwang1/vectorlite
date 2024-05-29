import sqlite3
import apsw 
import numpy as np
import os
"""
Example of using vectorlite extension to perform KNN search on a table of vectors.
"""

use_apsw = os.environ.get('USE_APSW', '0') == '1'

dim = 16
num_elements = 10000

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))

# create connection to in-memory database
conn = apsw.Connection(':memory:') if use_apsw else sqlite3.connect(':memory:')
conn.enable_load_extension(True)

conn.load_extension('../build/dev/libvectorlite.so')

cur = conn.cursor()

print('Trying to create virtual table for vector search.')
# Below statement creates a virtual table named 'x' with one column called 'my_embedding' which has a dimension of 16, a an implict rowid column.
# my_embedding holds vectors that can be searched based on L2 distance using HNSW index.
cur.execute(f'create virtual table x using vectorlite(my_embedding({dim}, "l2"), hnsw(max_elements={num_elements},ef_construction=100,M=16))')
print("Adding %d elements" % (len(data)))
for i in range(num_elements):
    cur.execute('insert into x (rowid, my_embedding) values (?, ?)', (i, data[i].tobytes()))

# Search for 10 nearest neighbors of data[0]
# distance here is a hidden column.
cur.execute('select rowid, distance from x where knn_search(my_embedding, knn_param(?, ?))', (data[0].tobytes(), 10))
print(cur.fetchall())

# Optionally, rowid can be filtered using 'rowid in (...)'. The rowid filter is pushed down to HNSW index and is efficient.
# Please note: 'rowid in (...)' is only supported for sqlite3 version >= 3.38.0. The built-in sqlite3 module usually doesn't support it. 
# Please use apsw module if you want to use rowid filtering.
cur.execute(f'select rowid, distance from x where knn_search(my_embedding, knn_param(?, ?)) and rowid in (0, 1)', (data[0].tobytes(), 10))
print(cur.fetchall())

# Multiple 'rowid in (...)' is also supported.
cur.execute(f'select rowid, distance from x where knn_search(my_embedding, knn_param(?, ?)) and rowid in (0, 1) and rowid in (1, 2)', (data[0].tobytes(), 10))
print(cur.fetchall())


# 'rowid = (...)' is also supported
cur.execute(f'select rowid, vector_to_json(my_embedding) from x where rowid = 1')
print(cur.fetchall())

cur.execute(f'delete from x where rowid = 1')
print(cur.fetchall())

cur.execute(f'select rowid, vector_to_json(my_embedding) from x where rowid = 1 or rowid = 2')
print(cur.fetchall())

cur.execute(f'update x set my_embedding = ? where rowid = 2', (data[0].tobytes(),))
print(cur.fetchall())

cur.execute(f'select rowid, vector_to_json(my_embedding) from x where rowid = 1 or rowid = 2')
print(cur.fetchall())

cur.close()