# API reference
Vectorlite provides the following APIs. 
Please note vectorlite is currently in beta. There could be breaking changes.
## Free-standing Application Defined SQL functions
The following functions can be used in any context.
``` sql
vectorlite_info() -- prints version info and some compile time info. e.g. Is SSE, AVX enabled.
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
