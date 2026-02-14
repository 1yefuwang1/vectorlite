![NPM Version](https://img.shields.io/npm/v/vectorlite)
![PyPI - Version](https://img.shields.io/pypi/v/vectorlite-py)

# Overview
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

## API reference
Vectorlite provides the following APIs. 
Please note vectorlite is currently in beta. There could be breaking changes.
### Free-standing Application Defined SQL functions
The following functions can be used in any context.
``` sql
vectorlite_info() -- prints version info, compile-time SIMD, and Highway runtime SIMD target.
vector_from_json(json_string) -- converts a json array of type TEXT into BLOB(a c-style float32 array)
vector_to_json(vector_blob) -- converts a vector of type BLOB(c-style float32 array) into a json array of type TEXT
vector_distance(vector_blob1, vector_blob2, distance_type_str) -- calculate vector distance between two vectors, distance_type_str could be 'l2', 'cosine', 'ip' 
```
In fact, one can easily implement brute force searching using `vector_distance`, which returns 100% accurate search results:
```sql
-- use a normal sqlite table
create table my_table(rowid integer primary key, embedding blob);

-- insert 
insert into my_table(rowid, embedding) values (0, {your_embedding});
-- search for 10 nearest neighbors using l2 squared distance
select rowid from my_table order by vector_distance({query_vector}, embedding, 'l2') asc limit 10

```

### Virtual Table
The core of vectorlite is the [virtual table](https://www.sqlite.org/vtab.html) module, which is used to hold vector index and way faster than brute force approach at the cost of not being 100% accurate.
A vectorlite table can be created using:

```sql
-- Required fields: table_name, vector_name, dimension, max_elements
-- Optional fields:
-- 1. distance_type: defaults to l2
-- 2. ef_construction: defaults to 200
-- 3. M: defaults to 16
-- 4. random_seed: defaults to 100
-- 5. allow_replace_deleted: defaults to true
-- 6. index_file_path: no default value. If not provided, the table will be memory-only. If provided, vectorlite will try to load index from the file and save to it when db connection is closed.
create virtual table {table_name} using vectorlite({vector_name} float32[{dimension}] {distance_type}, hnsw(max_elements={max_elements}, {ef_construction=200}, {M=16}, {random_seed=100}, {allow_replace_deleted=true}), {index_file_path});
```
You can insert, update and delete a vectorlite table as if it's a normal sqlite table. 
```sql
-- rowid is required during insertion, because rowid is used to connect the vector to its metadata stored elsewhere. Auto-generating rowid doesn't makes sense.
insert into my_vectorlite_table(rowid, vector_name) values ({your_rowid}, {vector_blob});
-- Note: update and delete statements that uses rowid filter require sqlite3_version >= 3.38 to run.  
update my_vectorlite_table set vector_name = {new_vector_blob} where rowid = {your_rowid};
delete from my_vectorlite_table where rowid = {your_rowid};
```
The following functions should be only used when querying a vectorlite table
```sql
-- returns knn_parameter that will be passed to knn_search(). 
-- vector_blob: vector to search
-- k: how many nearest neighbors to search for
-- ef: optional. A HNSW parameter that controls speed-accuracy trade-off. Defaults to 10 at first. If set to another value x, it will remain x if not specified again in another query within a single db connection.
knn_param(vector_blob, k, ef)
-- Should only be used in the `where clause` in a `select` statement to tell vectorlite to speed up the query using HNSW index
-- vector_name should match the vectorlite table's definition
-- knn_parameter is usually constructed using knn_param()
knn_search(vector_name, knn_parameter)
-- An example of vector search query. `distance` is an implicit column of a vectorlite table.
select rowid, distance from my_vectorlite_table where knn_search(vector_name, knn_param({vector_blob}, {k}))
-- An example of vector search query with pushed-down metadata(rowid) filter, requires sqlite_version >= 3.38 to run.
select rowid, distance from my_vectorlite_table where knn_search(vector_name, knn_param({vector_blob}, {k})) and rowid in (1,2,3,4,5)
```

## Benchmark
Please note only small datasets(with 3000 or 20000 vectors) are used because it would be unfair to benchmark against [sqlite-vec](https://github.com/asg017/sqlite-vec) using larger datasets. Sqlite-vec only uses brute-force, which doesn't scale with large datasets, while vectorlite uses ANN(approximate nearest neighbors), which scales to large datasets at the cost of not being 100% accurate.

How the benchmark is done:
1. Insert 3000/20000 randomly-generated vectors of dimension 128,512,1536 and 3000 into a vectorlite table with HNSW parameters ef_construction=100, M=30.
2. Randomly generate 100 vectors and then query the table with them for 10 nearest neighbors with ef=10,50,100 to see how ef impacts recall rate.
3. Calculate recall rate by comparing the result with the neighbors calculated using brute force.
4. vectorlite_scalar_brute_force(which is just inserting vectors into a normal sqlite table and do `select rowid from my_table order by vector_distance(query_vector, embedding, 'l2') limit 10`) is benchmarked as the baseline to see how much hnsw speeds up vector query.
5. [hnswlib](https://github.com/nmslib/hnswlib) is also benchmarked to see how much cost SQLite adds to vectorlite.
The benchmark is run in WSL on my PC with a i5-12600KF intel CPU and 16G RAM.


TL;DR:
1. Vectorlite's vector query is 3x-100x faster than [sqlite-vec](https://github.com/asg017/sqlite-vec) at the cost of lower recall rate. The difference gets larger when the dataset size grows, which is expected because sqlite-vec only supports brute force. 
2. Surprisingly, vectorlite_scalar_brute_force's vector query is about 1.5x faster for vectors with dimension >= 512 but slower than sqlite-vec for 128d vectors. vectorlite_scalar_brute_force's vector insertion is 3x-8x faster than sqlite-vec.
3. Compared with [hnswlib](https://github.com/nmslib/hnswlib), vectorlite provides almost identical recall rate. Vector query speed with L2 distance is on par with 128d vectors and is 1.5x faster when dealing with 3000d vectors. Mainly because vectorlite's vector distance implementation is faster. But vectorlite's vector insertion is about 4x-5x slower, which I guess is the cost added by SQLite.
4. Compared with brute force baseline(vectorlite_scalar_brute_force), vectorlite's knn query is 8x-80x faster.

The benchmark code can be found in [benchmark folder](https://github.com/1yefuwang1/vectorlite/tree/main/benchmark), which can be used as an example of how to improve recall rate for your scenario by tuning HNSW parameters.
### 3000 vectors
When dealing with 3000 vectors(which is a fairly small dataset):
1. Compared with [sqlite-vec](https://github.com/asg017/sqlite-vec), vectorlite's vector query can be 3x-15x faster with 128-d vectors, 6x-26x faster with 512-d vectors, 7x-30x faster with 1536-d vectors and 6x-24x faster with 3000-d vectors. But vectorlite's vector insertion is 6x-16x slower, which is expected because sqlite-vec uses brute force only and doesn't do much indexing.
2. Compared with vectorlite_scalar_brute_force, hnsw provides about 10x-40x speed up.
3. Compared with [hnswlib](https://github.com/nmslib/hnswlib), vectorlite provides almost identical recall rate. Vector query speed is on par with 128d vectors and is 1.5x faster when dealing with 3000d vectors. Mainly because vectorlite's vector distance implementation is faster. But vector insertion is about 4x-5x slower.
4. vectorlite_scalar_brute_force's vector insertion 4x-7x is faster than [sqlite-vec](https://github.com/asg017/sqlite-vec), and vector query is about 1.7x faster when dealing with vectors of dimension >= 512.



![vecter insertion](https://github.com/1yefuwang1/vectorlite/blob/main/media/vector_insertion_3000_vectors.png)
![vector query](https://github.com/1yefuwang1/vectorlite/blob/main/media/vector_query_3000_vectors.png)

<details>
<summary>Check raw data</summary>


```
Using local vectorlite: ../build/release/vectorlite/vectorlite.so
Benchmarking using 3000 randomly vectors. 100 10-nearest neighbor queries will be performed on each case.
┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┓
┃ distance ┃ vector    ┃ ef           ┃    ┃ ef     ┃ insert_time ┃ search_time ┃ recall ┃
┃ type     ┃ dimension ┃ construction ┃ M  ┃ search ┃ per vector  ┃ per query   ┃ rate   ┃
┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━╇━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━┩
│ l2       │ 128       │ 100          │ 30 │ 10     │ 62.41 us    │ 12.96 us    │ 56.40% │
│ l2       │ 128       │ 100          │ 30 │ 50     │ 62.41 us    │ 42.95 us    │ 93.30% │
│ l2       │ 128       │ 100          │ 30 │ 100    │ 62.41 us    │ 62.06 us    │ 99.40% │
│ l2       │ 512       │ 100          │ 30 │ 10     │ 146.40 us   │ 38.05 us    │ 46.60% │
│ l2       │ 512       │ 100          │ 30 │ 50     │ 146.40 us   │ 95.96 us    │ 86.50% │
│ l2       │ 512       │ 100          │ 30 │ 100    │ 146.40 us   │ 148.46 us   │ 96.70% │
│ l2       │ 1536      │ 100          │ 30 │ 10     │ 463.56 us   │ 124.51 us   │ 38.10% │
│ l2       │ 1536      │ 100          │ 30 │ 50     │ 463.56 us   │ 355.70 us   │ 78.50% │
│ l2       │ 1536      │ 100          │ 30 │ 100    │ 463.56 us   │ 547.84 us   │ 92.70% │
│ l2       │ 3000      │ 100          │ 30 │ 10     │ 1323.25 us  │ 391.57 us   │ 36.60% │
│ l2       │ 3000      │ 100          │ 30 │ 50     │ 1323.25 us  │ 1041.37 us  │ 78.60% │
│ l2       │ 3000      │ 100          │ 30 │ 100    │ 1323.25 us  │ 1443.10 us  │ 93.10% │
│ cosine   │ 128       │ 100          │ 30 │ 10     │ 59.75 us    │ 15.27 us    │ 58.30% │
│ cosine   │ 128       │ 100          │ 30 │ 50     │ 59.75 us    │ 36.72 us    │ 94.60% │
│ cosine   │ 128       │ 100          │ 30 │ 100    │ 59.75 us    │ 63.67 us    │ 99.30% │
│ cosine   │ 512       │ 100          │ 30 │ 10     │ 148.19 us   │ 36.98 us    │ 51.00% │
│ cosine   │ 512       │ 100          │ 30 │ 50     │ 148.19 us   │ 102.46 us   │ 88.10% │
│ cosine   │ 512       │ 100          │ 30 │ 100    │ 148.19 us   │ 143.41 us   │ 96.90% │
│ cosine   │ 1536      │ 100          │ 30 │ 10     │ 427.21 us   │ 106.94 us   │ 42.10% │
│ cosine   │ 1536      │ 100          │ 30 │ 50     │ 427.21 us   │ 285.50 us   │ 83.30% │
│ cosine   │ 1536      │ 100          │ 30 │ 100    │ 427.21 us   │ 441.66 us   │ 95.60% │
│ cosine   │ 3000      │ 100          │ 30 │ 10     │ 970.17 us   │ 289.00 us   │ 42.20% │
│ cosine   │ 3000      │ 100          │ 30 │ 50     │ 970.17 us   │ 848.03 us   │ 83.90% │
│ cosine   │ 3000      │ 100          │ 30 │ 100    │ 970.17 us   │ 1250.29 us  │ 95.60% │
└──────────┴───────────┴──────────────┴────┴────────┴─────────────┴─────────────┴────────┘
Bencharmk hnswlib as comparison.
┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┓
┃ distance ┃ vector    ┃ ef           ┃    ┃ ef     ┃ insert_time ┃ search_time ┃ recall ┃
┃ type     ┃ dimension ┃ construction ┃ M  ┃ search ┃ per vector  ┃ per query   ┃ rate   ┃
┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━╇━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━┩
│ l2       │ 128       │ 100          │ 30 │ 10     │ 12.84 us    │ 12.83 us    │ 56.90% │
│ l2       │ 128       │ 100          │ 30 │ 50     │ 12.84 us    │ 41.93 us    │ 93.60% │
│ l2       │ 128       │ 100          │ 30 │ 100    │ 12.84 us    │ 65.84 us    │ 99.40% │
│ l2       │ 512       │ 100          │ 30 │ 10     │ 29.34 us    │ 47.37 us    │ 47.00% │
│ l2       │ 512       │ 100          │ 30 │ 50     │ 29.34 us    │ 126.29 us   │ 86.40% │
│ l2       │ 512       │ 100          │ 30 │ 100    │ 29.34 us    │ 198.30 us   │ 96.80% │
│ l2       │ 1536      │ 100          │ 30 │ 10     │ 90.05 us    │ 149.35 us   │ 37.20% │
│ l2       │ 1536      │ 100          │ 30 │ 50     │ 90.05 us    │ 431.53 us   │ 78.00% │
│ l2       │ 1536      │ 100          │ 30 │ 100    │ 90.05 us    │ 765.03 us   │ 92.50% │
│ l2       │ 3000      │ 100          │ 30 │ 10     │ 388.87 us   │ 708.98 us   │ 36.30% │
│ l2       │ 3000      │ 100          │ 30 │ 50     │ 388.87 us   │ 1666.87 us  │ 78.90% │
│ l2       │ 3000      │ 100          │ 30 │ 100    │ 388.87 us   │ 2489.98 us  │ 93.40% │
│ cosine   │ 128       │ 100          │ 30 │ 10     │ 10.90 us    │ 11.14 us    │ 58.10% │
│ cosine   │ 128       │ 100          │ 30 │ 50     │ 10.90 us    │ 37.39 us    │ 94.30% │
│ cosine   │ 128       │ 100          │ 30 │ 100    │ 10.90 us    │ 62.45 us    │ 99.40% │
│ cosine   │ 512       │ 100          │ 30 │ 10     │ 25.46 us    │ 38.92 us    │ 50.70% │
│ cosine   │ 512       │ 100          │ 30 │ 50     │ 25.46 us    │ 109.84 us   │ 87.90% │
│ cosine   │ 512       │ 100          │ 30 │ 100    │ 25.46 us    │ 151.00 us   │ 97.10% │
│ cosine   │ 1536      │ 100          │ 30 │ 10     │ 77.53 us    │ 119.48 us   │ 42.00% │
│ cosine   │ 1536      │ 100          │ 30 │ 50     │ 77.53 us    │ 340.78 us   │ 84.00% │
│ cosine   │ 1536      │ 100          │ 30 │ 100    │ 77.53 us    │ 510.02 us   │ 95.50% │
│ cosine   │ 3000      │ 100          │ 30 │ 10     │ 234.79 us   │ 453.12 us   │ 43.20% │
│ cosine   │ 3000      │ 100          │ 30 │ 50     │ 234.79 us   │ 1380.79 us  │ 83.80% │
│ cosine   │ 3000      │ 100          │ 30 │ 100    │ 234.79 us   │ 1520.92 us  │ 95.70% │
└──────────┴───────────┴──────────────┴────┴────────┴─────────────┴─────────────┴────────┘
Bencharmk vectorlite brute force(select rowid from my_table order by vector_distance(query_vector, embedding, 'l2')) as comparison.
┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ distance ┃ vector    ┃ insert_time ┃ search_time ┃ recall  ┃
┃ type     ┃ dimension ┃ per vector  ┃ per query   ┃ rate    ┃
┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━┩
│ l2       │ 128       │ 2.38 us     │ 299.14 us   │ 100.00% │
│ l2       │ 512       │ 3.69 us     │ 571.19 us   │ 100.00% │
│ l2       │ 1536      │ 4.86 us     │ 2237.64 us  │ 100.00% │
│ l2       │ 3000      │ 7.69 us     │ 5135.63 us  │ 100.00% │
└──────────┴───────────┴─────────────┴─────────────┴─────────┘
Bencharmk sqlite_vss as comparison.
┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ distance ┃ vector    ┃ insert_time ┃ search_time ┃ recall  ┃
┃ type     ┃ dimension ┃ per vector  ┃ per query   ┃ rate    ┃
┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━┩
│ l2       │ 128       │ 395.24 us   │ 2508.52 us  │ 99.90%  │
│ l2       │ 512       │ 2824.89 us  │ 1530.77 us  │ 100.00% │
│ l2       │ 1536      │ 8931.72 us  │ 1602.36 us  │ 100.00% │
│ l2       │ 3000      │ 17498.60 us │ 3142.38 us  │ 100.00% │
└──────────┴───────────┴─────────────┴─────────────┴─────────┘
Bencharmk sqlite_vec as comparison.
┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ distance ┃ vector    ┃ insert_time ┃ search_time ┃ recall  ┃
┃ type     ┃ dimension ┃ per vector  ┃ per query   ┃ rate    ┃
┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━┩
│ l2       │ 128       │ 10.21 us    │ 202.05 us   │ 100.00% │
│ l2       │ 512       │ 14.43 us    │ 989.64 us   │ 100.00% │
│ l2       │ 1536      │ 31.68 us    │ 3856.08 us  │ 100.00% │
│ l2       │ 3000      │ 59.94 us    │ 9503.91 us  │ 100.00% │
└──────────┴───────────┴─────────────┴─────────────┴─────────┘

```
</details>

### 20000 vectors
When dealing with 20000 vectors, 
1. Compared with [sqlite-vec](https://github.com/asg017/sqlite-vec), vectorlite's vector query can be 8x-100x faster depending on vector dimension.
2. Compared with vectorlite_scalar_brute_force, hnsw provides about 8x-80x speed up with reduced recall rate at 13.8%-85% depending on vector dimension.
3. Compared with hnswlib, vectorlite provides almost identical recall rate. Vector query is on par with 128d vectors and can be 1.5x faster with 3000d vectors. But vector insertion is 3x-9x slower.
4. vectorlite_scalar_brute_force's vector insertion is 4x-8x faster than sqlite-vec. sqlite-vec's vector query is 1.5x faster with 128d vectors and 1.8x slower when vector dimension>=512.


Please note:
1. sqlite-vss is not benchmarked with 20000 vectors because its index creation takes so long that it doesn't finish in hours.
2. sqlite-vec's vector query is benchmarked and included in the raw data, but not plotted in the figure because it's search time is disproportionally long.

![vecter insertion](https://github.com/1yefuwang1/vectorlite/blob/main/media/vector_insertion_20000_vectors.png)
![vector query](https://github.com/1yefuwang1/vectorlite/blob/main/media/vector_query_20000_vectors.png)

<details>
<summary>Check raw data</summary>

```
Using local vectorlite: ../build/release/vectorlite/vectorlite.so
Benchmarking using 20000 randomly vectors. 100 10-neariest neighbor queries will be performed on each case.
┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┓
┃ distance ┃ vector    ┃ ef           ┃    ┃ ef     ┃ insert_time ┃ search_time ┃ recall ┃
┃ type     ┃ dimension ┃ construction ┃ M  ┃ search ┃ per vector  ┃ per query   ┃ rate   ┃
┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━╇━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━┩
│ l2       │ 128       │ 100          │ 30 │ 10     │ 187.41 us   │ 46.58 us    │ 29.10% │
│ l2       │ 128       │ 100          │ 30 │ 50     │ 187.41 us   │ 95.16 us    │ 70.20% │
│ l2       │ 128       │ 100          │ 30 │ 100    │ 187.41 us   │ 179.51 us   │ 85.70% │
│ l2       │ 512       │ 100          │ 30 │ 10     │ 820.80 us   │ 105.80 us   │ 18.10% │
│ l2       │ 512       │ 100          │ 30 │ 50     │ 820.80 us   │ 361.83 us   │ 50.40% │
│ l2       │ 512       │ 100          │ 30 │ 100    │ 820.80 us   │ 628.88 us   │ 67.00% │
│ l2       │ 1536      │ 100          │ 30 │ 10     │ 2665.31 us  │ 292.39 us   │ 13.70% │
│ l2       │ 1536      │ 100          │ 30 │ 50     │ 2665.31 us  │ 1069.47 us  │ 42.40% │
│ l2       │ 1536      │ 100          │ 30 │ 100    │ 2665.31 us  │ 1744.79 us  │ 59.50% │
│ l2       │ 3000      │ 100          │ 30 │ 10     │ 5236.76 us  │ 558.56 us   │ 13.80% │
│ l2       │ 3000      │ 100          │ 30 │ 50     │ 5236.76 us  │ 1787.83 us  │ 39.30% │
│ l2       │ 3000      │ 100          │ 30 │ 100    │ 5236.76 us  │ 3039.94 us  │ 56.60% │
│ cosine   │ 128       │ 100          │ 30 │ 10     │ 164.31 us   │ 25.35 us    │ 34.70% │
│ cosine   │ 128       │ 100          │ 30 │ 50     │ 164.31 us   │ 78.33 us    │ 71.20% │
│ cosine   │ 128       │ 100          │ 30 │ 100    │ 164.31 us   │ 133.75 us   │ 87.60% │
│ cosine   │ 512       │ 100          │ 30 │ 10     │ 711.35 us   │ 100.90 us   │ 19.00% │
│ cosine   │ 512       │ 100          │ 30 │ 50     │ 711.35 us   │ 406.08 us   │ 51.10% │
│ cosine   │ 512       │ 100          │ 30 │ 100    │ 711.35 us   │ 582.51 us   │ 71.50% │
│ cosine   │ 1536      │ 100          │ 30 │ 10     │ 2263.96 us  │ 283.88 us   │ 22.60% │
│ cosine   │ 1536      │ 100          │ 30 │ 50     │ 2263.96 us  │ 919.98 us   │ 54.50% │
│ cosine   │ 1536      │ 100          │ 30 │ 100    │ 2263.96 us  │ 1674.77 us  │ 72.40% │
│ cosine   │ 3000      │ 100          │ 30 │ 10     │ 4541.09 us  │ 566.31 us   │ 19.80% │
│ cosine   │ 3000      │ 100          │ 30 │ 50     │ 4541.09 us  │ 1672.82 us  │ 49.30% │
│ cosine   │ 3000      │ 100          │ 30 │ 100    │ 4541.09 us  │ 2855.43 us  │ 65.40% │
└──────────┴───────────┴──────────────┴────┴────────┴─────────────┴─────────────┴────────┘
Bencharmk hnswlib as comparison.
┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┓
┃ distance ┃ vector    ┃ ef           ┃    ┃ ef     ┃ insert_time ┃ search_time ┃ recall ┃
┃ type     ┃ dimension ┃ construction ┃ M  ┃ search ┃ per vector  ┃ per query   ┃ rate   ┃
┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━╇━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━┩
│ l2       │ 128       │ 100          │ 30 │ 10     │ 23.06 us    │ 39.96 us    │ 29.60% │
│ l2       │ 128       │ 100          │ 30 │ 50     │ 23.06 us    │ 75.02 us    │ 69.80% │
│ l2       │ 128       │ 100          │ 30 │ 100    │ 23.06 us    │ 160.01 us   │ 85.40% │
│ l2       │ 512       │ 100          │ 30 │ 10     │ 146.58 us   │ 167.31 us   │ 18.10% │
│ l2       │ 512       │ 100          │ 30 │ 50     │ 146.58 us   │ 392.12 us   │ 50.80% │
│ l2       │ 512       │ 100          │ 30 │ 100    │ 146.58 us   │ 781.50 us   │ 67.20% │
│ l2       │ 1536      │ 100          │ 30 │ 10     │ 657.41 us   │ 298.71 us   │ 12.70% │
│ l2       │ 1536      │ 100          │ 30 │ 50     │ 657.41 us   │ 1031.61 us  │ 40.60% │
│ l2       │ 1536      │ 100          │ 30 │ 100    │ 657.41 us   │ 1764.34 us  │ 57.90% │
│ l2       │ 3000      │ 100          │ 30 │ 10     │ 1842.77 us  │ 852.88 us   │ 13.80% │
│ l2       │ 3000      │ 100          │ 30 │ 50     │ 1842.77 us  │ 2905.57 us  │ 39.60% │
│ l2       │ 3000      │ 100          │ 30 │ 100    │ 1842.77 us  │ 4936.35 us  │ 56.50% │
│ cosine   │ 128       │ 100          │ 30 │ 10     │ 19.25 us    │ 23.27 us    │ 34.20% │
│ cosine   │ 128       │ 100          │ 30 │ 50     │ 19.25 us    │ 72.66 us    │ 71.40% │
│ cosine   │ 128       │ 100          │ 30 │ 100    │ 19.25 us    │ 134.11 us   │ 87.60% │
│ cosine   │ 512       │ 100          │ 30 │ 10     │ 112.80 us   │ 106.90 us   │ 22.70% │
│ cosine   │ 512       │ 100          │ 30 │ 50     │ 112.80 us   │ 341.23 us   │ 54.20% │
│ cosine   │ 512       │ 100          │ 30 │ 100    │ 112.80 us   │ 609.93 us   │ 72.40% │
│ cosine   │ 1536      │ 100          │ 30 │ 10     │ 615.04 us   │ 268.00 us   │ 22.50% │
│ cosine   │ 1536      │ 100          │ 30 │ 50     │ 615.04 us   │ 898.82 us   │ 54.00% │
│ cosine   │ 1536      │ 100          │ 30 │ 100    │ 615.04 us   │ 1557.51 us  │ 71.90% │
│ cosine   │ 3000      │ 100          │ 30 │ 10     │ 1425.49 us  │ 546.18 us   │ 20.60% │
│ cosine   │ 3000      │ 100          │ 30 │ 50     │ 1425.49 us  │ 2008.53 us  │ 49.20% │
│ cosine   │ 3000      │ 100          │ 30 │ 100    │ 1425.49 us  │ 3106.51 us  │ 65.00% │
└──────────┴───────────┴──────────────┴────┴────────┴─────────────┴─────────────┴────────┘
Bencharmk vectorlite brute force(select rowid from my_table order by vector_distance(query_vector, embedding, 'l2')) as comparison.
┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ distance ┃ vector    ┃ insert_time ┃ search_time ┃ recall  ┃
┃ type     ┃ dimension ┃ per vector  ┃ per query   ┃ rate    ┃
┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━┩
│ l2       │ 128       │ 0.93 us     │ 2039.69 us  │ 100.00% │
│ l2       │ 512       │ 2.73 us     │ 7177.23 us  │ 100.00% │
│ l2       │ 1536      │ 4.64 us     │ 17163.25 us │ 100.00% │
│ l2       │ 3000      │ 6.62 us     │ 25378.79 us │ 100.00% │
└──────────┴───────────┴─────────────┴─────────────┴─────────┘
Bencharmk sqlite_vec as comparison.
┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ distance ┃ vector    ┃ insert_time ┃ search_time ┃ recall  ┃
┃ type     ┃ dimension ┃ per vector  ┃ per query   ┃ rate    ┃
┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━┩
│ l2       │ 128       │ 3.49 us     │ 1560.17 us  │ 100.00% │
│ l2       │ 512       │ 6.73 us     │ 7778.39 us  │ 100.00% │
│ l2       │ 1536      │ 17.13 us    │ 26344.76 us │ 100.00% │
│ l2       │ 3000      │ 35.30 us    │ 60652.58 us │ 100.00% │
└──────────┴───────────┴─────────────┴─────────────┴─────────┘
```

</details>

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

# Find 10 approximate nearest neighbors of the first embedding in vectors with rowid within [1000, 2000) using metadata(rowid) filtering.
rowids = ','.join([str(rowid) for rowid in range(1000, 2000)])
result = cursor.execute(f'select rowid, distance from my_table where knn_search(my_embedding, knn_param(?, 10)) and rowid in ({rowids})', [data[0].tobytes()]).fetchall()
print(f'10 nearest neighbors of row 0 in vectors with rowid within [1000, 2000) is {result}')

conn.close()

```

More examples can be found in examples and bindings/python/vectorlite_py/test folder.

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

# install dependencies for running python tests
python3 -m pip install -r requirements-dev.txt

sh build.sh # for debug build
sh build_release.sh # for release build

```
`vecorlite.[so|dll|dylib]` can be found in `build/release/vectorlite` or `build/dev/vectorlite` folder

### Build wheel

```shell
python3 -m build -w

```
vectorlite_py wheel can be found in `dist` folder

# Roadmap
- [x] SIMD support for ARM platform
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
3. ~~SIMD is only enabled on x86 platforms. Because the default implementation in hnswlib doesn't support SIMD on ARM. Vectorlite is 3x-4x slower on MacOS-ARM than MacOS-x64. I plan to improve it in the future.~~
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
