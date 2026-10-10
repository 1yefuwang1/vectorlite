# SQLite-backed DiskANN

The `diskann(...)` backend uses Microsoft DiskANN3's Rust graph algorithms with
SQLite-owned vector, graph, and index metadata storage. It is an alternative to
HNSW, not a change to HNSW's in-memory persistence or transaction behavior.

This first milestone is experimental. It supports `float32` storage with squared
L2 or cosine distance, interactive inserts/updates/deletes, native pointer-based
batch insertion, and explicit consolidation. It does not include product
quantization, half-precision storage, parallel CPU graph construction, inner-product
indexing, or background repair. The upstream algorithm/provider API is pinned to
DiskANN 0.60.0 with a recorded local batch error-propagation/task-draining patch.

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

## Native batch INSERT

Native C/Rust callers can bind a **tagged pointer to a versioned descriptor**
containing a contiguous row-major float32 matrix and explicit int64 rowids:

```sql
INSERT INTO embeddings(operation, embedding)
VALUES ('insert_batch', ?1);

-- Optional working-chunk size; default 8, accepted range 1 through 32.
INSERT INTO embeddings(operation, embedding, path)
VALUES ('insert_batch', ?1, '{"batch_size":8}');
```

`?1` must be bound using `sqlite3_bind_pointer()` with type tag
`vectorlite.batch.f32.v1`, not as an integer address or BLOB. The standalone
[public native header](<../vectorlite/include/vectorlite_batch.h>) is also installed
under `vectorlite_py/include` with the extension. Example for a two-dimensional
DiskANN table:

```c
#include <sqlite3.h>
#include "vectorlite_batch.h"

float vectors[] = {1.f, 0.f, 0.f, 2.f, 3.f, 4.f};
int64_t rowids[] = {123, 456, 789};
BatchF32V1 batch = {
    VECTORLITE_BATCH_F32_V1_ABI_VERSION, (uint32_t)sizeof(BatchF32V1),
    3, 2, vectors, rowids
};
sqlite3_stmt *stmt = NULL;
int rc = sqlite3_prepare_v2(db,
    "INSERT INTO embeddings(operation,embedding) VALUES('insert_batch',?1)",
    -1, &stmt, NULL);
if (rc == SQLITE_OK) {
    rc = sqlite3_bind_pointer(stmt, 1, &batch,
                             VECTORLITE_BATCH_F32_V1_TAG, NULL);
}
if (rc == SQLITE_OK) {
    do { rc = sqlite3_step(stmt); } while (rc == SQLITE_ROW);
    /* Success is SQLITE_DONE. SQLITE_ROW may come from count_changes=ON. */
}
/* Preserve any error before cleanup, and check finalize's result too. */
int finalize_rc = sqlite3_finalize(stmt);
```

The descriptor is 40 bytes on official 64-bit targets, with fields
`uint32_t abi_version`, `uint32_t struct_size`, `uint64_t count`,
`uint64_t dimension`, `const float *vectors`, and `const int64_t *rowids`.
Rust callers use an exact `#[repr(C)]` counterpart and the same static tag.
Native inputs are native IEEE-754 float32 values; stored/output BLOBs retain the
little-endian float32 format.

**The caller must keep the descriptor and both readable, aligned buffers alive
and immutable until the binding is cleared, replaced, or finalized.** Resetting
the statement does not release bindings. With a NULL destructor the caller owns
all buffers. Allocation validity and actual buffer length cannot be proved from
a pointer tag; an arbitrary address with the correct tag violates the native
contract and can cause undefined behavior. The extension checks version, layout,
dimensions, nullness/alignment, length arithmetic and rowids before accessing the
appropriate bounded input. It never interprets SQL numbers as addresses.

Only `operation`, the vector column and optional `path` may be supplied; rowids
come from the descriptor and a `distance` input is rejected. Duplicate rowids
within/across chunks or already in the table abort the operation; this is
insert-only, not upsert. Empty batches are no-ops. Numeric/range checks and cosine
normalization follow ordinary DiskANN insertion and do not modify caller buffers.

A single INSERT may describe more than 32 vectors. The extension copies and
normalizes working chunks of 8 by default (at most 32), seeds an empty live graph
with ordinary single insertion, and runs true DiskANN batch candidate/backedge aggregation for
subsequent chunks. The **entire INSERT has one rollback boundary**: a failure in a
later chunk restores earlier chunks' vectors, graph, ID allocation and counters.
Chunking does not commit partial results. A successful nonempty command bumps the
revision once. Callers needing separate commits can issue several bounded INSERTs,
optionally under one explicit application transaction.

Batch tasks run on a fresh **private current-thread Tokio runtime** inside the
original SQLite callback. No task accesses SQLite from a worker thread; no tasks
or borrowed input pointers remain after callback return. An ambient Tokio runtime
visible to the extension's linked Tokio instance is rejected before graph writes;
this is not detection of a separately linked host runtime. The pinned upstream
patch drains tasks and propagates all batch phase failures independently of
tracing, rather than accepting partial graph construction as success.

`cache_bytes` admits owned chunk inputs, batch edges/maps/guards, graph scratch and
vector snapshots. Runtime/container overhead is conservatively estimated, not an
allocator-enforced process RSS cap. `max_visits` applies across each true chunk,
not independently to every vector. Lower `batch_size` or adjust resource settings
when admission/work limits return `SQLITE_TOOBIG`; no graph repair is silently
truncated. Larger chunks consume one shared visit allowance and may need a larger
explicit `max_visits`; a failed chunk is not automatically retried at a smaller
size after writes. The ordinary SQL API and HNSW behavior are unchanged.

Python's standard-library `sqlite3.execute()` cannot bind tagged native pointers;
this interface is for native C/Rust clients, not a raw-address Python workaround.
There is no automatic graph batching of ordinary multi-row VALUES or `executemany`.

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
