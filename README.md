# Overview
Vectorlite is a [Runtime-loadable extension](https://www.sqlite.org/loadext.html) for SQLite that enables fast vector search based on [hnswlib](https://github.com/nmslib/hnswlib) and works on Windows, MacOS and Linux.

Currently, vectorlite is pre-compiled for Windows-x64, Linux-x64, MacOS-x64, MacOS-arm64 and distributed in python wheels.
For other languages, vectorlite.[so|dll|dylib] can be extracted from the wheel for your platform, given that a *.whl file is actually a zip archive.

Vectorlite is currently in beta. There could be breaking changes.
More examples can be found in examples and integration_test folder.
# Features
1. Fast ANN-search backed by hnswlib.
2. SIMD accelerated vector distance(l2squared, cosine and inner product) calculation for x86 platform, using `vector_distance(vec1, vec2, 'l2')`
3. Full control over HNSW parameters for performance tuning.
4. Metadata filter support (requires sqlite version >= 3.38).
5. Index serde support. A vectorlite table can be saved to a file, and be reloaded from it. Index files created by hnswlib can also be loaded by vectorlite.
6. Vector json serde support using `vector_from_json()` and `vector_to_json()`.

# Quick Start
todo...

# Build Instructions
If you want to compile vectorlite yourself, you can follow following instructions to build it.
## Prerequisites
1. CMake >= 3.22
2. Ninja
3. A C++ compiler in PATH that supports c++17
4. Python3
## Build
### Build sqlite extension only
```
git clone --recurse-submodules git@github.com:1yefuwang1/vectorlite.git

python3 bootstrap_vcpkg.py

# install dependencies for running integration tests
python3 -m pip install -r requirements-dev.txt

sh build.sh # for debug build
sh build_release.sh # for release build

```
`vecorlite.[so|dll|dylib]` can be found in `build/release` or `build/dev` folder

### Build wheel

```
python3 -m build -w
```
vectorlite_py wheel can be found in `dist` folder
# Known limitations
1. On a single query, a knn_search vector constraint can only be paired with at most one rowid constraint and vice versa. 
For example, The following queries will fail:
```
select rowid from my_table where rowid in (1,2,3) and rowid in (2, 3, 4) # multiple rowid constraints
select rowid, distance from my_table where knn_search(my_embedding, knn_param(vector_from_json('[1,2,3]'), 10)) and rowid in (1,2,3) and rowid in (3, 4) # multiple rowid constraints
select rowid, distance from my_table where knn_search(my_embedding, knn_param(vector_from_json('[1,2,3]'), 10)) and knn_search(my_embedding, knn_param(vector_from_json('[1,2,3]'), 10)) # multiple vector constraints

``` 
However, multiple constrains can be combined with `or`, because the query will search the underlying hnsw index multiple times. The results will be conbined by sqlite.
The following queries will work.
```
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

# Acknowledgement
This project is greatly inspired by following projects
- [sqlite-vss](https://github.com/asg017/sqlite-vss)
- [hnsqlite](https://github.com/jiggy-ai/hnsqlite)
- [ChromaDB](https://github.com/chroma-core/chroma)