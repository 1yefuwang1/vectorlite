# Overview
Vectorlite is a [Runtime-loadable extension](https://www.sqlite.org/loadext.html) for SQLite that enables fast vector search based on [hnswlib](https://github.com/nmslib/hnswlib).
It works on Windows, MacOS and Linux.
Currently pre-compiled python wheels are provided for Windows-x64, Linux-x64, MacOS-x64 and MacOS-arm64.
Vectorlite is currently of beta quality. There could be breaking changes and bugs.
Examples can be found in examples folder.
# Build Instructions
If you want to compile vectorlite for platforms other than currently supported ones, you can follow following instructions to build it.
Vectorlite should build on all platforms.
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
1. On a single search to the virtual table, at most one vector constraint with at most one rowid constraint can be used. 
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
3. SIMD is only enabled on x86 platforms. Because the default implementation in hnswlib doesn't support SIMD on ARM. Please expect slower search on ARM platform. I plan to improve it in the future.
4. rowid in sqlite3 is of type int64_t and can be negative. However, rowid in a vectorlite table should be in this range `[0, min(max value of size_t, max value of int64_t)]`. The reason is rowid is used as `labeltype` in hnsw index, which has type `size_t`(usually 32-bit or 64-bit depending on the platform).
5. Transaction is not supported.
6. rowid filter requires sqlite3 >= 3.38. Python's built-in `sqlite` module is usually built with old versions. Please use a newer sqlite binding such as `apsw`.

# Acknowledgement
This project is greatly inspired by following projects
- [sqlite-vss](https://github.com/asg017/sqlite-vss)
- [hnsqlite](https://github.com/jiggy-ai/hnsqlite)
- [ChromaDB](https://github.com/chroma-core/chroma)