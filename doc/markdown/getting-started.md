# Getting started
The quickest way to get started is to install vectorlite using python.
```shell
# Note: vectorlite-py not vectorlite. vectorlite is another project.
pip install vectorlite-py numpy
```
The packaged extension is implemented in Rust, with native hnswlib and Highway SIMD operations. Installing a prebuilt wheel requires no Rust toolchain. The `vectorlite_py` package and its `vectorlite_path()` API are unchanged.

Use a Python build with loadable SQLite extensions enabled. Vectorlite requires SQLite >= 3.20; rowid lookups and metadata filtering require SQLite >= 3.38. Python 3.14 with a recent bundled SQLite is recommended. On SQLite >= 3.31, vectorlite tables cannot be accessed from views or triggers; issue queries and save/load commands directly from application SQL.

Below is a minimal example of using vectorlite. It can also be found in the [examples folder](https://github.com/1yefuwang1/vectorlite/tree/main/examples).

```python
import vectorlite_py
import sqlite3
import numpy as np
"""
Quick start of using vectorlite extension.
"""

conn = sqlite3.connect(':memory:')
conn.enable_load_extension(True) # enable extension loading
conn.load_extension(vectorlite_py.vectorlite_path()) # load vectorlite

cursor = conn.cursor()
# check if vectorlite is loaded
print(cursor.execute('select vectorlite_info()').fetchall())

# Vector distance calculation
for distance_type in ['l2', 'cosine', 'ip']:
    v1 = "[1, 2, 3]"
    v2 = "[4, 5, 6]"
    # Note vector_from_json can be used to convert a JSON string to a vector
    distance = cursor.execute(f'select vector_distance(vector_from_json(?), vector_from_json(?), "{distance_type}")', (v1, v2)).fetchone()
    print(f'{distance_type} distance between {v1} and {v2} is {distance[0]}')

# generate some test data
DIM = 32 # dimension of the vectors
NUM_ELEMENTS = 10000 # number of vectors
data = np.float32(np.random.random((NUM_ELEMENTS, DIM))) # SQL inputs use float32 blobs; stored types may also be float16/bfloat16.

# Create a virtual table using vectorlite using l2 distance (default distance type) and default HNSW parameters
cursor.execute(f'create virtual table my_table using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
# Vector distance type can be explicitly set to cosine using:
# cursor.execute(f'create virtual table my_table using vectorlite(my_embedding float32[{DIM}] cosine, hnsw(max_elements={NUM_ELEMENTS}))')

# Insert the test data into the virtual table. Note that the rowid MUST be explicitly set when inserting vectors and cannot be auto-generated.
# The rowid is used to uniquely identify a vector and serve as a "foreign key" to relate to the vector's metadata.
# Vectorlite takes vectors in raw bytes, so a numpy vector need to be converted to bytes before inserting into the table.
cursor.executemany('insert into my_table(rowid, my_embedding) values (?, ?)', [(i, data[i].tobytes()) for i in range(NUM_ELEMENTS)])

# Query the virtual table to get the vector at rowid 12345. Note the vector needs to be converted back to json using vector_to_json() to be human-readable. 
result = cursor.execute('select vector_to_json(my_embedding) from my_table where rowid = 1234').fetchone()
print(f'vector at rowid 1234: {result[0]}')

# Find 10 approximate nearest neighbors of data[0] and there distances from data[0].
# knn_search() is used to tell vectorlite to do a vector search.
# knn_param(V, K, ef) is used to pass the query vector V, the number of nearest neighbors K to find and an optional ef parameter to tune the performance of the search.
# If ef is not specified, it defaults to 10. An explicit ef applies only to that query.
# For more info on ef, see https://github.com/nmslib/hnswlib/blob/v0.8.0/ALGO_PARAMS.md
result = cursor.execute('select rowid, distance from my_table where knn_search(my_embedding, knn_param(?, 10))', [data[0].tobytes()]).fetchall()
print(f'10 nearest neighbors of row 0 is {result}')

# Find 10 approximate nearest neighbors of the first embedding in vectors with rowid within [1000, 2000) using metadata(rowid) filtering.
rowids = ','.join([str(rowid) for rowid in range(1000, 2000)])
result = cursor.execute(f'select rowid, distance from my_table where knn_search(my_embedding, knn_param(?, 10)) and rowid in ({rowids})', [data[0].tobytes()]).fetchall()
print(f'10 nearest neighbors of row 0 in vectors with rowid within [1000, 2000) is {result}')

conn.close()

```

More examples can be found in the [examples](https://github.com/1yefuwang1/vectorlite/tree/main/examples) folder. For persistence, issue explicit `INSERT INTO <table>(operation, path) VALUES('save'|'load', ...)` commands from application SQL. New saves have a versioned schema descriptor; raw legacy files use the receiving table's declared schema. See the [API reference](<api.md>) for details.

## Building from source

Source builds require latest stable Rust, C/C++17 compilers, CMake >= 3.22, Ninja, Git and vcpkg. The Cargo workspace uses root [Cargo.toml](<../../Cargo.toml>) and [Cargo.lock](<../../Cargo.lock>); Rust sources and retained native ops live under `vectorlite/`. See the [contributor guide](<../../vectorlite/README.md>) for the layout and direct Cargo checks. Run the following commands from the repository root:

```shell
git submodule update --init --recursive
python3 bootstrap_vcpkg.py
# scikit-build-core runs CMake, which invokes Cargo for the extension:
python3 -m pip install .
# Or create a wheel:
python3 -m pip wheel . --wheel-dir dist
```

For development builds and tests, install the development requirements and use the root scripts:

```shell
python3 -m pip install -r requirements-dev.txt
sh build.sh          # Debug build + CTest + both Python suites
sh build_release.sh  # Release build + the same tests
```

There is no prerequisite C++ virtual-table build or separate Rust redeployment step. C++ is retained only for hnswlib, Highway ops and their thin native shim.
