# API reference
Vectorlite provides the following APIs. 
Please note vectorlite is currently in beta. There could be breaking changes.
The SQL API is implemented by the Rust extension; the public `vectorlite` module and function names are unchanged. Loading requires SQLite >= 3.20, with SQLite >= 3.38 required for rowid lookup/filtering.

## Free-standing Application Defined SQL functions
The following functions can be used in any context.
``` sql
vectorlite_info() -- prints version info and the best SIMD target chosen by Highway at runtime.
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
## Virtual Table
The core of vectorlite is the [virtual table](https://www.sqlite.org/vtab.html) module, which is used to hold vector index and way faster than brute force approach at the cost of not being 100% accurate.
A vectorlite table can be created using:

SQL vector inputs and outputs are little-endian float32 blobs. The declared storage type can be `float32`, `float16` or `bfloat16`; half-precision tables quantize on insert/query and dequantize when reading the vector column. Supported metrics are `l2` (squared L2), `ip` and `cosine` (normalized inner-product distance).

```sql
-- Required fields: table_name, vector_name, dimension, max_elements
-- Optional fields:
-- 1. distance_type: defaults to l2
-- 2. ef_construction: defaults to 200
-- 3. M: defaults to 16
-- 4. random_seed: defaults to 100
-- 5. allow_replace_deleted: defaults to true
-- The index is always held in memory. Persist or restore it explicitly with the
-- operation/path commands shown below.
create virtual table {table_name} using vectorlite({vector_name} float32[{dimension}] {distance_type}, hnsw(max_elements={max_elements}, {ef_construction=200}, {M=16}, {random_seed=100}, {allow_replace_deleted=true}));
```
### Breaking change: explicit index persistence

The optional **third file-path argument to `CREATE VIRTUAL TABLE` has been removed**. Existing SQL using that argument is rejected. Indexes are no longer automatically loaded when a table is created or automatically saved when the connection closes.

To migrate, remove the path argument, then issue an explicit `load` command after creating the table when you want to restore an existing index:

```sql
-- Old API (no longer supported):
CREATE VIRTUAL TABLE my_table USING vectorlite(
    embedding float32[128], hnsw(max_elements=10000), '/path/to/index.bin'
);

-- Current API:
CREATE VIRTUAL TABLE my_table USING vectorlite(
    embedding float32[128], hnsw(max_elements=10000)
);
INSERT INTO my_table(operation, path) VALUES ('load', '/path/to/index.bin');
```

For a new index, skip `load`. **Explicitly save any changes you want to retain before closing the connection**; closing without saving loses the in-memory changes. If an existing SQLite database stores a virtual-table declaration with the old third argument, that declaration also needs migration; changing only new table-creation SQL is not sufficient. Legacy index-file compatibility is described below.

Persist an index to disk, or restore a saved index into an in-memory table:
```sql
-- Save the current in-memory index to a file (overwrites if it exists).
insert into {table_name}(operation, path) values ('save', '/path/to/index.bin');
-- Load a saved index into a freshly created table. Loading replaces the table's
-- current in-memory index; on any error the existing index is left unchanged.
insert into {table_name}(operation, path) values ('load', '/path/to/index.bin');
```
New saves use a **versioned envelope** recording the vector dimension, element type (`float32`, `float16` or `bfloat16`), distance metric, normalization policy and native word size/endianness. These must match the receiving table on load. A successful save atomically replaces the destination; a failed load leaves the live index unchanged.

Loading also accepts **legacy raw HNSW files** from older Vectorlite builds or hnswlib. Raw files have no Vectorlite schema descriptor, so the receiving table's declaration is authoritative. The per-vector byte size and native layout must match, but equal-width types or dimensions with the same total byte size cannot be distinguished. Loading does not convert or re-normalize vectors or rebuild the graph: declare the intended schema. To upgrade, load the raw file and save again to write a versioned envelope.

The receiving table's `max_elements` and `allow_replace_deleted` control capacity and deleted-slot reuse after loading. Capacity is at least the loaded element count; use a larger `max_elements` to allow growth. Other graph-construction parameters come from the saved graph.

The in-memory index is held per database connection and survives schema reparses (e.g. `VACUUM`, `ALTER TABLE`, or DDL from other connections). It is lost when the connection closes unless you explicitly save it. On SQLite **3.31 or newer**, vectorlite tables are direct-only: application SQL can access them, but views and triggers cannot.

Note: `operation`, `path`, and `distance` are reserved column names and cannot be used as the vector column name.

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
-- ef: optional, positive HNSW speed/accuracy parameter. Defaults to 10.
-- An override applies only to this query; later queries without ef use the default.
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
