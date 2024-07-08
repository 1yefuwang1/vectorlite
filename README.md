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
Vectorlite is fast. Compared with [sqlite-vss](https://github.com/facebookresearch/faiss), vectorlite is 10x faster in insertion and 2x-10x faster in searching with much better recall rate.
The benchmark method is that:
1. Insert 10000 randomly-generated vectors into a vectorlite table.
2. Randomly generate 100 vectors and then query the table with them.

The benchmark is run on my PC with a i5-12600KF CPU and 16G RAM and on WSL.
The benchmark code can be found in benchmark folder, which can be used as an example of how to improve recall_rate for your scenario by tuning HNSW parameters.

```
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ distance_type ┃ vector dimension ┃ ef_construction ┃ M   ┃ ef_search ┃ insert_time(per vector) ┃ search_time(per query) ┃ recall_rate ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ l2            │ 256              │ 16              │ 200 │ 10        │ 310.31 us               │ 81.46 us               │ 52.40%      │
│ l2            │ 256              │ 16              │ 200 │ 50        │ 310.31 us               │ 217.17 us              │ 88.40%      │
│ l2            │ 256              │ 16              │ 200 │ 100       │ 310.31 us               │ 314.81 us              │ 97.40%      │
│ l2            │ 256              │ 32              │ 200 │ 10        │ 327.48 us               │ 89.16 us               │ 52.40%      │
│ l2            │ 256              │ 32              │ 200 │ 50        │ 327.48 us               │ 213.95 us              │ 88.40%      │
│ l2            │ 256              │ 32              │ 200 │ 100       │ 327.48 us               │ 349.63 us              │ 97.40%      │
│ l2            │ 1024             │ 16              │ 200 │ 10        │ 1460.37 us              │ 445.21 us              │ 42.70%      │
│ l2            │ 1024             │ 16              │ 200 │ 50        │ 1460.37 us              │ 1362.84 us             │ 81.90%      │
│ l2            │ 1024             │ 16              │ 200 │ 100       │ 1460.37 us              │ 1989.38 us             │ 92.90%      │
│ l2            │ 1024             │ 32              │ 200 │ 10        │ 1436.74 us              │ 415.00 us              │ 42.70%      │
│ l2            │ 1024             │ 32              │ 200 │ 50        │ 1436.74 us              │ 1282.99 us             │ 81.90%      │
│ l2            │ 1024             │ 32              │ 200 │ 100       │ 1436.74 us              │ 1904.94 us             │ 92.90%      │
│ cosine        │ 256              │ 16              │ 200 │ 10        │ 268.53 us               │ 63.51 us               │ 52.40%      │
│ cosine        │ 256              │ 16              │ 200 │ 50        │ 268.53 us               │ 163.83 us              │ 89.40%      │
│ cosine        │ 256              │ 16              │ 200 │ 100       │ 268.53 us               │ 264.20 us              │ 96.40%      │
│ cosine        │ 256              │ 32              │ 200 │ 10        │ 286.86 us               │ 64.63 us               │ 52.40%      │
│ cosine        │ 256              │ 32              │ 200 │ 50        │ 286.86 us               │ 192.57 us              │ 89.40%      │
│ cosine        │ 256              │ 32              │ 200 │ 100       │ 286.86 us               │ 338.05 us              │ 96.40%      │
│ cosine        │ 1024             │ 16              │ 200 │ 10        │ 1235.72 us              │ 411.42 us              │ 47.30%      │
│ cosine        │ 1024             │ 16              │ 200 │ 50        │ 1235.72 us              │ 1113.31 us             │ 85.20%      │
│ cosine        │ 1024             │ 16              │ 200 │ 100       │ 1235.72 us              │ 1652.70 us             │ 95.60%      │
│ cosine        │ 1024             │ 32              │ 200 │ 10        │ 1152.72 us              │ 378.64 us              │ 47.30%      │
│ cosine        │ 1024             │ 32              │ 200 │ 50        │ 1152.72 us              │ 1142.82 us             │ 85.20%      │
│ cosine        │ 1024             │ 32              │ 200 │ 100       │ 1152.72 us              │ 1634.47 us             │ 95.60%      │
└───────────────┴──────────────────┴─────────────────┴─────┴───────────┴─────────────────────────┴────────────────────────┴─────────────┘
```
The result of the same benchmark for [sqlite-vss](https://github.com/asg017/sqlite-vss) is below: 
```
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ vector dimension ┃ insert_time(per vector) ┃ search_time(per query) ┃ recall_rate ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ 256              │ 3694.37 us              │ 755.17 us              │ 55.40%      │
│ 1024             │ 18598.29 us             │ 3848.64 us             │ 48.60%      │
└──────────────────┴─────────────────────────┴────────────────────────┴─────────────┘
```
I believe the performance difference is mainly caused by the underlying vector search library.
Sqlite-vss uses [faiss](https://github.com/facebookresearch/faiss), which is optimized for batched scenarios.
Vectorlite uses [hnswlib](https://github.com/facebookresearch/faiss), which is optimized for online vector searching.

# Quick Start
The quickest way to get started is to install vectorlite using python.
```shell
pip install vectorlite apsw numpy
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