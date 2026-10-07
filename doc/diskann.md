# SQLite-backed DiskANN

The `diskann(...)` backend uses Microsoft DiskANN3's Rust graph algorithms with
SQLite-owned vector, graph, and index metadata storage. It is an alternative to
HNSW, not a change to HNSW's in-memory persistence or transaction behavior.

This first milestone is experimental. It supports `float32` storage with squared
L2 or cosine distance, interactive inserts/updates/deletes, and explicit
consolidation. It does not include product quantization, half-precision storage,
inner-product indexing, parallel bulk graph construction, or background repair.
The upstream algorithm/provider API is pinned to DiskANN 0.60.0.

## Create and query

Use a **file-backed SQLite database** for data larger than memory. SQLite
`:memory:` and memory-backed temporary databases still store their pages in RAM.
DiskANN requires SQLite 3.38.0 or newer; loading the extension for HNSW retains
its existing minimum-version policy.

```sql
CREATE VIRTUAL TABLE embeddings USING vectorlite(
    embedding float32[768] cosine,
    diskann(degree=32, build_list_size=100,
            search_list_size=64, cache_bytes=67108864)
);

INSERT INTO embeddings(rowid, embedding) VALUES (123, ?);

SELECT rowid, distance
FROM embeddings
WHERE knn_search(embedding, knn_param(?, 10));

-- DiskANN's named speed/recall setting, applied to this query only.
SELECT rowid, distance
FROM embeddings
WHERE knn_search(embedding,
                 knn_param(?, 10, '{"search_list_size":100}'));

-- Only matching rows may enter the output. Other graph nodes can still provide
-- navigation bridges; this is not filtering an already-selected global top-k.
SELECT rowid, distance
FROM embeddings
WHERE knn_search(embedding, knn_param(?, 10))
  AND rowid IN (123, 456, 789);
```

The two-argument `knn_param(vector, k)` form works with either backend. A numeric
third argument remains **HNSW `ef`** and is rejected by DiskANN. DiskANN accepts a
JSON object containing only a positive integer `search_list_size`.

Like existing vectorlite tables, queries require a KNN or rowid equality/IN
constraint. Arbitrary SQL predicates are not automatically pushed into graph
search. Request `ORDER BY distance, rowid` when SQL-level ordering is required.

## Options

All DiskANN options are optional; `diskann()` uses these defaults:

| Option | Default | Meaning |
|---|---:|---|
| `degree` | 32 | Target pruned graph degree; physical lists allow the pinned algorithm's 1.3× slack. |
| `build_list_size` | 100 | Candidate window during graph insertion. |
| `search_list_size` | 64 | Default query window; effective size is at least effective k. |
| `alpha` | 1.2 | Finite pruning factor, at least 1. |
| `cache_bytes` | 67108864 | Adapter workspace budget in bytes. |
| `max_visits` | 65536 | Conservative per-operation/traversal work limit. |

Duplicate and unknown options are errors. Degree must be between 2 and 10000.
List sizes must be positive and no greater than `max_visits`, which is at most
`u32::MAX`. Cache values below 4096 are rejected. The minimum syntactically valid
cache size is **not** a guarantee that graph work fits: scratch admission also
accounts for degree, list sizes, visits, and vector dimension. Increase the budget
or lower the tuning limits when an operation reports `SQLITE_TOOBIG`.

There is no HNSW-style preallocated `max_elements`. Sparse public rowids are
looked up through SQLite's index, and committed internal graph IDs are monotonic
and never reused.

## Vector representation

SQL inputs and outputs remain little-endian float32 BLOBs. L2 stores float32
values unchanged and reports **squared** distance. Cosine normalizes both stored
vectors and queries using the existing SIMD policy and reports `1 - dot`;
reading the vector column returns the normalized representation, not the original
unscaled input. Zero vectors retain the existing zero-normalization behavior.

DiskANN rejects NaN/infinity and dimensions that do not match the declaration.
To avoid overflow in graph distances, L2 inputs require squared norm no greater
than `f32::MAX / 4`. Cosine inputs require a finite float32-compatible norm.
Nonfinite computed distances are errors. These DiskANN checks do not change
HNSW's numeric policy.

## Transactions and persistence

```sql
BEGIN;
INSERT INTO documents(id, title) VALUES (123, 'Example');
INSERT INTO embeddings(rowid, embedding) VALUES (123, ?);
COMMIT;

BEGIN;
UPDATE embeddings SET embedding=? WHERE rowid=123;
-- The new representation is visible to queries on this connection.
ROLLBACK;
-- Both the vector and graph return to their previous state.
```

Vectors, labels, adjacency lists, ID allocation, and counters live in the same
SQLite database and participate in its transactions and savepoints. A private
ordinary-table, multirow statement provides the rollback boundary for each
DiskANN callback's multiple shadow writes. No extra database connection,
independent journal, or application-issued nested transaction is used.

Mutation requires a rollback-capable journal mode. `journal_mode=OFF` is
rejected rather than silently changing host configuration. `MEMORY` can provide
statement rollback but does not provide an on-disk crash-recovery journal;
commit/power-loss guarantees also depend on the host's SQLite durability settings.
The atomic boundary covers vectors, graph, metadata, and the carrier in their
own schema, not arbitrary host-function side effects in other attached/TEMP
schemas.

Initial constraint handling is ABORT-only. A duplicate/constraint failure does
not implement SQL `OR IGNORE`, `OR FAIL`, or `OR REPLACE` semantics; the failed
statement must not retain a partially changed graph. SQLite may roll back an
entire transaction for I/O, interruption, or other severe errors. Inspect the
connection's transaction state and handle SQLite's error rather than assuming
all failures preserve earlier statements.

Data automatically reopens with the database. DiskANN has no external index
`save`/`load` command: use normal SQLite backup/checkpoint procedures. A running
WAL database can have ordinary SQLite WAL/shared-memory files; these are not
separate ANN index sidecars. The transaction guarantees above apply to DiskANN
and ordinary SQLite tables, **not** to HNSW mutations in the same transaction.

Virtual tables retain the existing direct-only restriction: access them from
application SQL rather than through views or triggers. Join metadata by rowid,
and issue coordinated writes explicitly within your application transaction.

## Updates, deletion, and consolidation

Public rowids are required, nonnegative, and no greater than `i64::MAX`; UPDATE
cannot change a rowid. Updating a vector retires its old internal node and inserts
a new node under the same public rowid, atomically. DELETE removes the row from
query results immediately. Retired vectors remain available to graph navigation
until maintenance proves their remaining incoming references have been removed.

```sql
INSERT INTO embeddings(operation) VALUES ('consolidate');
```

Consolidation is a synchronous, transactional **rebuild of the live graph**.
A SQLite-owned staging shadow stores the live canonical vectors while graph
insertion reads them in bounded batches. New internal IDs are allocated; the
old graph remains recoverable through SQLite's journal until the new graph and
reclamation checks succeed. Retired topology is never cleared during ordinary
DELETE/UPDATE, because approximate OneHop repair can miss a one-way incoming
edge and disconnect live nodes from the entrypoint.

Maintenance requires temporary disk space for staged vectors and a new graph,
and its cost can approach a full rebuild. It does not reuse committed internal
IDs. SQLite can reuse freed pages; use ordinary `VACUUM` separately when file
shrinking is needed. Consolidation holds the SQLite writer transaction. Schedule
it explicitly; this backend does not perform hidden background writes.

## Memory, snapshots, and error handling

Opening the index reads bounded schema/metadata/root information, not all graph
nodes or vectors. Graph records are fetched on demand with operation-local
workspaces. Small rowid filters and underfilled filtered graph searches can use a
streaming exact top-k fallback; a request covering the whole candidate population
can also take the exact path. These paths may perform more I/O but do not require
a full in-memory vector matrix.

`cache_bytes` is an **adapter workspace** budget, including admitted graph
scratch, owned vector snapshots, filter/result/projection accounting, and bounded
SQL-adapter buffers. It is not a process-wide RSS cap. SQLite's pager cache, OS
file cache, caller-supplied SQL BLOBs, the backend-agnostic `knn_param` producer,
and allocator/code overhead have their own memory costs. Work/visit limits are
conservative, not an exact public count of expanded graph nodes. Exceeding limits
returns an explicit error; writes do not silently accept a truncated graph repair.

Cursors retain a SQLite snapshot anchor. Projected vector bytes are materialized
with their result distances, so a later same-connection update does not mix an old
distance with a changed vector. Close cursors promptly to release WAL snapshots.
Other connections follow ordinary SQLite isolation and `SQLITE_BUSY_SNAPSHOT`
rules; arbitrary interleaved writes on one connection do not gain stronger
isolation than SQLite provides.

I/O, corruption, and allocation failures are not reported as missing rows. Invalid
record lengths, node IDs, mappings, or formats fail closed when accessed. The
adapter checks its frozen entrypoint at open, but does not eagerly verify every
node in a larger-than-memory graph. Internal shadow tables, journal-carrier rows,
and the private atomic function are **not a supported editing/import API**.
Changing them directly can invalidate the index; attempted guard corruption or
unexpected guard triggers are rejected before/within mutation.

## Platforms and validation

The release targets remain Linux x64, Windows x64, and macOS arm64 ≥11. Native
macOS dependencies must be built with the pinned deployment floor, not merely
relinked into a dylib tagged 11.0. CI/package checks inspect the resulting native
artifacts and notices. Tests on a current macOS host are not evidence of execution
on an actual macOS 11 machine.

Performance is workload-dependent. Use file-backed, release-built measurements
and report recall alongside latency, update/consolidation cost, database/WAL size,
and memory. Small correctness tests or a database merely larger than the pager
cache do not establish a physical beyond-RAM benchmark. Do not interpret the
repository's historical HNSW figures as new DiskANN performance results.
