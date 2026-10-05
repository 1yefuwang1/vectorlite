import vectorlite_py
import sqlite3
import numpy as np
import os
"""
This is an example of setting HNSW parameters in vectorlite.
"""


vectorlite_path = os.environ.get("VECTORLITE_PATH", vectorlite_py.vectorlite_path())

if vectorlite_path != vectorlite_py.vectorlite_path():
    print(f"Using local vectorlite: {vectorlite_path}")

conn = sqlite3.connect(':memory:')
conn.enable_load_extension(True) # enable extension loading
conn.load_extension(vectorlite_path) # loads vectorlite

cursor = conn.cursor()

DIM = 32 # dimension of the vectors
NUM_ELEMENTS = 1000 # number of vectors

# Create virtual table with customized HNSW parameters. Please check https://github.com/nmslib/hnswlib/blob/v0.8.0/ALGO_PARAMS.md for more info.
cursor.execute(f'create virtual table vector_table using vectorlite(embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}, ef_construction=100, M=32, random_seed=1000))')

data = np.float32(np.random.random((NUM_ELEMENTS, DIM))) # SQL inputs are float32 blobs; storage can also be float16/bfloat16
embeddings = [(i, data[i].tobytes()) for i in range(NUM_ELEMENTS)]
cursor.executemany('insert into vector_table(rowid, embedding) values (?, ?)', embeddings)

# Search for the 10 nearest neighbors of the first embedding with customized ef = 32. Increasing ef will increase the search accuracy but also increase the search time.
# knn_param(V, K, ef) is used to pass the query vector V, the number of nearest neighbors K to find and an optional ef parameter to tune the performance of the search.
# If ef is not specified, ef defaults to 10.
# An ef override applies only to this query; later queries without an override use 10.
# ef is not serialized in the index file, so a loaded table also starts with 10.
vector_to_search = data[0].tobytes()
k = 10
ef = 32
result = cursor.execute(f'select rowid, distance from vector_table where knn_search(embedding, knn_param(?, ?, ?))', [vector_to_search, k, ef]).fetchall()
print(result)

