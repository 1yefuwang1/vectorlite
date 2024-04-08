import sqlite3
import json
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
cur.execute(f'create virtual table x using vector_search(my_embedding({dim}, "l2"), hnsw(max_elements={num_elements},ef_construction=100,M=16))')
print("Adding %d elements" % (len(data)))
for i in range(num_elements):
    cur.execute(f'insert into x (rowid, my_embedding) values ({i}, vector_from_json("{json.dumps(data[i].tolist())}"))')

cur.execute(f'select rowid, distance from x where knn_search(my_embedding, knn_param(vector_from_json("{json.dumps(data[0].tolist())}"), 10))')
print(cur.fetchall())
cur.close()