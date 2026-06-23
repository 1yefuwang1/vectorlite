# Overview of vectorlite
## Quick overview
Vectorlite is a [Runtime-loadable extension](https://www.sqlite.org/loadext.html) for SQLite that enables fast vector search based on [hnswlib](https://github.com/nmslib/hnswlib) and works on Windows, MacOS and Linux. It provides fast vector search capabilities with a SQL interface and runs on every language with a SQLite driver.

For motivation and background of this project, please check [here](https://dev.to/yefuwang/introducing-vectorlite-a-fast-and-tunable-vector-search-extension-for-sqlite-4dcl).

Below is an example of using it in sqlite CLI shell:

```sql
-- Load vectorlite
.load path/to/vectorlite.[so|dll|dylib]
-- shows vectorlite version and build info.
select vectorlite_info(); 
-- Calculate vector l2(squared) distance
select vector_distance(vector_from_json('[1,2,3]'), vector_from_json('[3,4,5]'), 'l2');
-- Create a virtual table named my_table with one vector column my_embedding with dimention of 3
create virtual table my_table using vectorlite(my_embedding float32[3], hnsw(max_elements=100));
-- Insert vectors into my_table. rowid can be used to relate to a vector's metadata stored elsewhere, e.g. another table.
insert into my_table(rowid, my_embedding) values (0, vector_from_json('[1,2,3]'));
insert into my_table(rowid, my_embedding) values (1, vector_from_json('[2,3,4]'));
insert into my_table(rowid, my_embedding) values (2, vector_from_json('[7,7,7]'));
-- Find 2 approximate nearest neighbors of vector [3,4,5] with distances
select rowid, distance from my_table where knn_search(my_embedding, knn_param(vector_from_json('[3,4,5]'), 2));
-- Find the nearest neighbor of vector [3,4,5] among vectors with rowid 0 and 1. (requires sqlite_version>=3.38)
-- It is called metadata filter in vectorlite, because you could get rowid set beforehand based on vectors' metadata and then perform vector search.
-- Metadata filter is pushed down to the underlying index when traversing the HNSW graph.
select rowid, distance from my_table where knn_search(my_embedding, knn_param(vector_from_json('[3,4,5]'), 1)) and rowid in (0, 1) ;

```

Currently, vectorlite is pre-compiled for Windows-x64, Linux-x64, MacOS-x64, MacOS-arm64 and distributed as python wheels and npm packages. It can be installed simply by:
```shell
# For python
pip install vectorlite-py
# for nodejs
npm i vectorlite
```
For other languages, `vectorlite.[so|dll|dylib]` can be extracted from the wheel for your platform, given that a *.whl file is actually a zip archive.

Vectorlite is currently in beta. There could be breaking changes.

## Highlights
1. Fast ANN(approximate nearest neighbors) search backed by [hnswlib](https://github.com/nmslib/hnswlib). Vector query is significantly faster than similar projects like [sqlite-vec](https://github.com/asg017/sqlite-vec) and [sqlite-vss](https://github.com/asg017/sqlite-vss). Please see benchmark [below](https://github.com/1yefuwang1/vectorlite?tab=readme-ov-file#benchmark).
2. Works on Windows, Linux and MacOS(x64 and ARM).
3. A fast and portable SIMD accelerated vector distance implementation using Google's [highway](https://github.com/google/highway) library. On my PC(i5-12600KF with AVX2 support), vectorlite's implementation is 1.5x-3x faster than hnswlib's when dealing vectors with dimension >= 256.
4. Supports all vector distance types provided by hnswlib: l2(squared l2), cosine, ip(inner product. I do not recomend you to use it though). For more info please check [hnswlib's doc](https://github.com/nmslib/hnswlib/tree/v0.8.0?tab=readme-ov-file#supported-distances).
3. Full control over [HNSW parameters](https://github.com/nmslib/hnswlib/blob/v0.8.0/ALGO_PARAMS.md) for performance tuning. Please check [this example](https://github.com/1yefuwang1/vectorlite/blob/main/examples/hnsw_param.py).
4. Predicate pushdown support for vector metadata(rowid) filter (requires sqlite version >= 3.38). Please check [this example](https://github.com/1yefuwang1/vectorlite/blob/main/examples/metadata_filter.py);
5. Index serde support. A vectorlite table can be saved to a file, and be reloaded from it. Index files created by hnswlib can also be loaded by vectorlite. Please check [this example](https://github.com/1yefuwang1/vectorlite/blob/main/examples/index_serde.py);
6. Vector json serde support using `vector_from_json()` and `vector_to_json()`.
