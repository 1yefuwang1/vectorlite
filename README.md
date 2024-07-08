# Overview
Vectorlite is a [Runtime-loadable extension](https://www.sqlite.org/loadext.html) for SQLite that enables fast vector search based on [hnswlib](https://github.com/nmslib/hnswlib) and works on Windows, MacOS and Linux.

Currently, vectorlite is pre-compiled for Windows-x64, Linux-x64, MacOS-x64, MacOS-arm64 and distributed as python wheels.
For other languages, vectorlite.[so|dll|dylib] can be extracted from the wheel for your platform, given that a *.whl file is actually a zip archive.

Vectorlite is currently in beta. There could be breaking changes.
## Highlights
1. Fast ANN-search backed by hnswlib. Please see benchmark below.
2. Works on Windows, Linux and MacOS.
3. SIMD accelerated vector distance calculation for x86 platform, using `vector_distance()`
4. Supports all vector distance types provided by hnswlib: l2(squared l2), cosine, ip(inner product. I do not recomend you to use it though). For more info please check [hnswlib's doc](https://github.com/nmslib/hnswlib/tree/v0.8.0?tab=readme-ov-file#supported-distances).
3. Full control over HNSW parameters for performance tuning.
4. Metadata filter support (requires sqlite version >= 3.38).
5. Index serde support. A vectorlite table can be saved to a file, and be reloaded from it. Index files created by hnswlib can also be loaded by vectorlite.
6. Vector json serde support using `vector_from_json()` and `vector_to_json()`.

## Benchamrk
Vectorlite is fast. Compared with [sqlite-vss](https://github.com/facebookresearch/faiss), vectorlite is 10x faster in inserting vectors and 2x-10x faster in searching , and offers much better recall rate if proper HNSW parameters are set.

The benchmark method is:
1. Insert 10000 randomly-generated vectors into a vectorlite table.
2. Randomly generate 100 vectors and then query the table with them for 10 nearest neighbors.
3. Calculate recall rate by comparing the result with the neighbors calculated using brute force.

The benchmark is run on my PC with a i5-12600KF intel CPU and 16G RAM and on WSL.
The benchmark code can be found in benchmark folder, which can be used as an example of how to improve recall rate for your scenario by tuning HNSW parameters.

HNSW parameters are crucial for vectorlite's performance! Please benchmark and find the best HNSW parameters for your scenario.


```
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ distance_type ┃ vector dimension ┃ ef_construction ┃ M  ┃ ef_search ┃ insert_time(per vector) ┃ search_time(per query) ┃ recall_rate ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ l2            │ 256              │ 200             │ 32 │ 10        │ 297.24 us               │ 41.43 us               │ 32.00%      │
│ l2            │ 256              │ 200             │ 32 │ 50        │ 297.24 us               │ 144.02 us              │ 71.00%      │
│ l2            │ 256              │ 200             │ 32 │ 100       │ 297.24 us               │ 194.07 us              │ 89.10%      │
│ l2            │ 256              │ 200             │ 32 │ 150       │ 297.24 us               │ 258.51 us              │ 95.70%      │
│ l2            │ 256              │ 200             │ 48 │ 10        │ 308.29 us               │ 41.44 us               │ 40.00%      │
│ l2            │ 256              │ 200             │ 48 │ 50        │ 308.29 us               │ 135.68 us              │ 79.90%      │
│ l2            │ 256              │ 200             │ 48 │ 100       │ 308.29 us               │ 212.45 us              │ 94.30%      │
│ l2            │ 256              │ 200             │ 48 │ 150       │ 308.29 us               │ 288.03 us              │ 98.10%      │
│ l2            │ 256              │ 200             │ 64 │ 10        │ 315.36 us               │ 51.16 us               │ 45.30%      │
│ l2            │ 256              │ 200             │ 64 │ 50        │ 315.36 us               │ 176.78 us              │ 83.80%      │
│ l2            │ 256              │ 200             │ 64 │ 100       │ 315.36 us               │ 262.93 us              │ 96.50%      │
│ l2            │ 256              │ 200             │ 64 │ 150       │ 315.36 us               │ 350.85 us              │ 98.80%      │
│ l2            │ 1024             │ 200             │ 32 │ 10        │ 1467.88 us              │ 172.84 us              │ 22.70%      │
│ l2            │ 1024             │ 200             │ 32 │ 50        │ 1467.88 us              │ 556.84 us              │ 56.60%      │
│ l2            │ 1024             │ 200             │ 32 │ 100       │ 1467.88 us              │ 988.22 us              │ 77.40%      │
│ l2            │ 1024             │ 200             │ 32 │ 150       │ 1467.88 us              │ 1374.78 us             │ 87.20%      │
│ l2            │ 1024             │ 200             │ 48 │ 10        │ 1565.23 us              │ 240.40 us              │ 28.30%      │
│ l2            │ 1024             │ 200             │ 48 │ 50        │ 1565.23 us              │ 805.00 us              │ 67.20%      │
│ l2            │ 1024             │ 200             │ 48 │ 100       │ 1565.23 us              │ 1321.40 us             │ 85.80%      │
│ l2            │ 1024             │ 200             │ 48 │ 150       │ 1565.23 us              │ 1711.43 us             │ 93.50%      │
│ l2            │ 1024             │ 200             │ 64 │ 10        │ 1495.51 us              │ 359.36 us              │ 31.20%      │
│ l2            │ 1024             │ 200             │ 64 │ 50        │ 1495.51 us              │ 1031.87 us             │ 71.60%      │
│ l2            │ 1024             │ 200             │ 64 │ 100       │ 1495.51 us              │ 1493.56 us             │ 89.30%      │
│ l2            │ 1024             │ 200             │ 64 │ 150       │ 1495.51 us              │ 2142.30 us             │ 95.10%      │
│ cosine        │ 256              │ 200             │ 32 │ 10        │ 269.65 us               │ 36.94 us               │ 37.50%      │
│ cosine        │ 256              │ 200             │ 32 │ 50        │ 269.65 us               │ 103.17 us              │ 79.50%      │
│ cosine        │ 256              │ 200             │ 32 │ 100       │ 269.65 us               │ 155.54 us              │ 91.50%      │
│ cosine        │ 256              │ 200             │ 32 │ 150       │ 269.65 us               │ 218.37 us              │ 96.60%      │
│ cosine        │ 256              │ 200             │ 48 │ 10        │ 276.83 us               │ 62.64 us               │ 42.40%      │
│ cosine        │ 256              │ 200             │ 48 │ 50        │ 276.83 us               │ 113.28 us              │ 85.00%      │
│ cosine        │ 256              │ 200             │ 48 │ 100       │ 276.83 us               │ 193.28 us              │ 95.70%      │
│ cosine        │ 256              │ 200             │ 48 │ 150       │ 276.83 us               │ 251.09 us              │ 98.20%      │
│ cosine        │ 256              │ 200             │ 64 │ 10        │ 286.84 us               │ 56.21 us               │ 47.60%      │
│ cosine        │ 256              │ 200             │ 64 │ 50        │ 286.84 us               │ 131.34 us              │ 88.70%      │
│ cosine        │ 256              │ 200             │ 64 │ 100       │ 286.84 us               │ 226.67 us              │ 97.00%      │
│ cosine        │ 256              │ 200             │ 64 │ 150       │ 286.84 us               │ 302.13 us              │ 98.90%      │
│ cosine        │ 1024             │ 200             │ 32 │ 10        │ 1261.69 us              │ 163.41 us              │ 28.90%      │
│ cosine        │ 1024             │ 200             │ 32 │ 50        │ 1261.69 us              │ 477.67 us              │ 67.90%      │
│ cosine        │ 1024             │ 200             │ 32 │ 100       │ 1261.69 us              │ 890.98 us              │ 84.10%      │
│ cosine        │ 1024             │ 200             │ 32 │ 150       │ 1261.69 us              │ 1138.74 us             │ 91.30%      │
│ cosine        │ 1024             │ 200             │ 48 │ 10        │ 1324.49 us              │ 218.41 us              │ 33.40%      │
│ cosine        │ 1024             │ 200             │ 48 │ 50        │ 1324.49 us              │ 660.30 us              │ 73.70%      │
│ cosine        │ 1024             │ 200             │ 48 │ 100       │ 1324.49 us              │ 1127.36 us             │ 89.40%      │
│ cosine        │ 1024             │ 200             │ 48 │ 150       │ 1324.49 us              │ 1446.57 us             │ 95.10%      │
│ cosine        │ 1024             │ 200             │ 64 │ 10        │ 1317.50 us              │ 262.43 us              │ 36.50%      │
│ cosine        │ 1024             │ 200             │ 64 │ 50        │ 1317.50 us              │ 809.18 us              │ 78.60%      │
│ cosine        │ 1024             │ 200             │ 64 │ 100       │ 1317.50 us              │ 1333.40 us             │ 92.90%      │
│ cosine        │ 1024             │ 200             │ 64 │ 150       │ 1317.50 us              │ 1753.07 us             │ 96.50%      │
└───────────────┴──────────────────┴─────────────────┴────┴───────────┴─────────────────────────┴────────────────────────┴─────────────┘
```
The result of the same benchmark is also run for [sqlite-vss](https://github.com/asg017/sqlite-vss) using its default index: 
```
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ vector dimension ┃ insert_time(per vector) ┃ search_time(per query) ┃ recall_rate ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ 256              │ 3999.12 us              │ 869.13 us              │ 52.60%      │
│ 1024             │ 18820.58 us             │ 4293.13 us             │ 50.80%      │
└──────────────────┴─────────────────────────┴────────────────────────┴─────────────┘
```
I believe the performance difference is mainly caused by the underlying vector search library.
Sqlite-vss uses [faiss](https://github.com/facebookresearch/faiss), which is optimized for batched scenarios.
Vectorlite uses [hnswlib](https://github.com/facebookresearch/faiss), which is optimized for realtime vector searching.

# Quick Start
The quickest way to get started is to install vectorlite using python.
```shell
# Note: vectorlite-py not vectorlite. vectorlite is another project.
pip install vectorlite-py apsw numpy
```
Vectorlite's metadata filter feature requires sqlite>=3.38. Python's builtin `sqlite` module is usually built with old sqlite versions. So `apsw` is used here as sqlite driver, because it provides bindings to latest sqlite. Vectorlite still works with old sqlite versions if metadata filter support is not required.
Below is a minimal example of using vectorlite. It can also be found in the examples folder.

```python
import vectorlite_py
import apsw
import numpy as np
"""
Quick start of using vectorlite extension.
"""

conn = apsw.Connection(':memory:')
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
data = np.float32(np.random.random((NUM_ELEMENTS, DIM))) # Only float32 vectors are supported by vectorlite for now

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
# If ef is not specified, ef defaults to 10. For more info on ef, please check https://github.com/nmslib/hnswlib/blob/v0.8.0/ALGO_PARAMS.md
result = cursor.execute('select rowid, distance from my_table where knn_search(my_embedding, knn_param(?, 10))', [data[0].tobytes()]).fetchall()
print(f'10 nearest neighbors of row 0 is {result}')

# Find 10 approximate nearest neighbors of the first embedding in vectors with rowid within [1001, 2000) using metadata(rowid) filtering.
rowids = ','.join([str(rowid) for rowid in range(1000, 2000)])
result = cursor.execute(f'select rowid, distance from my_table where knn_search(my_embedding, knn_param(?, 10)) and rowid in ({rowids})', [data[0].tobytes()]).fetchall()
print(f'10 nearest neighbors of row 0 in vectors with rowid within [1000, 2000) is {result}')

conn.close()

```

More examples can be found in examples and integration_test folder.

# Build Instructions
If you want to contribute or compile vectorlite for your own platform, you can follow following instructions to build it.
## Prerequisites
1. CMake >= 3.22
2. Ninja
3. A C++ compiler in PATH that supports c++17
4. Python3
## Build
### Build sqlite extension only
```shell
git clone --recurse-submodules git@github.com:1yefuwang1/vectorlite.git

python3 bootstrap_vcpkg.py

# install dependencies for running integration tests
python3 -m pip install -r requirements-dev.txt

sh build.sh # for debug build
sh build_release.sh # for release build

```
`vecorlite.[so|dll|dylib]` can be found in `build/release` or `build/dev` folder

### Build wheel

```shell
python3 -m build -w

```
vectorlite_py wheel can be found in `dist` folder

# Roadmap
- [ ] SIMD support for ARM platform
- [ ] Support user defined metadata/rowid filter
- [ ] Support Multi-vector document search and epsilon search
- [ ] Support multi-threaded search
- [ ] Release vectorlite to more package managers.
- [ ] Support more vector types, e.g. float16, int8.

# Known limitations
1. On a single query, a knn_search vector constraint can only be paired with at most one rowid constraint and vice versa. 
For example, The following queries will fail:
```sql
select rowid from my_table where rowid in (1,2,3) and rowid in (2, 3, 4) # multiple rowid constraints

select rowid, distance from my_table where knn_search(my_embedding, knn_param(vector_from_json('[1,2,3]'), 10)) and rowid in (1,2,3) and rowid in (3, 4) # multiple rowid constraints

select rowid, distance from my_table where knn_search(my_embedding, knn_param(vector_from_json('[1,2,3]'), 10)) and knn_search(my_embedding, knn_param(vector_from_json('[1,2,3]'), 10)) # multiple vector constraints

``` 
However, multiple constrains can be combined with `or`, because the query will search the underlying hnsw index multiple times. The results will be conbined by sqlite.
The following queries will work.
```sql
select rowid from my_table where rowid in (1,2,3) or rowid in (2, 3, 4)

select rowid, distance from my_table where knn_search(my_embedding, knn_param(vector_from_json('[1,2,3]'), 10)) and rowid in (1,2,3) or rowid in (3, 4)

select rowid, distance from my_table where knn_search(my_embedding, knn_param(vector_from_json('[1,2,3]'), 10)) or knn_search(my_embedding, knn_param(vector_from_json('[1,2,3]'), 10)) 
```
2. Only float32 vectors are supported for now.
3. SIMD is only enabled on x86 platforms. Because the default implementation in hnswlib doesn't support SIMD on ARM. Vectorlite is 3x-4x slower on MacOS-ARM than MacOS-x64. I plan to improve it in the future.
4. rowid in sqlite3 is of type int64_t and can be negative. However, rowid in a vectorlite table should be in this range `[0, min(max value of size_t, max value of int64_t)]`. The reason is rowid is used as `labeltype` in hnsw index, which has type `size_t`(usually 32-bit or 64-bit depending on the platform).
5. Transaction is not supported.
6. Metadata filter(rowid filter) requires sqlite3 >= 3.38. Python's built-in `sqlite` module is usually built with old versions. Please use a newer sqlite binding such as `apsw` if you want to use metadata filter. knn_search() without rowid fitler still works for old sqlite3.
7. The vector index is held in memory.
8. Deleting a row only marks the vector as deleted and doesn't free the memory. The vector will not be included in later queries. However, if another vector is inserted with the same rowid, the memory will be reused.
9. A vectorlite table can only have one vector column.

# Acknowledgement
This project is greatly inspired by following projects
- [sqlite-vss](https://github.com/asg017/sqlite-vss)
- [hnsqlite](https://github.com/jiggy-ai/hnsqlite)
- [ChromaDB](https://github.com/chroma-core/chroma)