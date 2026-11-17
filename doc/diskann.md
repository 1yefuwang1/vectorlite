# SQLite-backed DiskANN

The `diskann(...)` backend stores vectors in SQLite and builds its **global
DiskANN graph** with unmodified Microsoft DiskANN 0.60.0 Rust algorithms. It is an
alternative to HNSW, not a change to HNSW's in-memory persistence or transaction
behavior.

This milestone is experimental and uses a **store-first, deferred-indexing** design:

1. **Ingest** — ordinary INSERT/UPDATE/DELETE (including the native batch
   descriptor) commit canonical vectors transactionally. No graph is built.
2. **Maintenance** — an explicit SQL job builds one global graph from a frozen
   snapshot in owned RAM, then publishes it generation by generation.

Vector search sees only a **published** generation. Until a generation is
published, ingested vectors are stored and readable by rowid but do not appear in
`knn_search` results.

Supported: `float32` storage, squared L2 and cosine distance, interactive
inserts/updates/deletes, native pointer-based batch ingestion, one native OneShot
build worker, and explicit publication. Not included: product quantization,
half-precision storage, parallel native graph construction (`workers` must be 1),
inner-product indexing, Merged/partitioned building, background daemons, or
automatic migration from the earlier experimental disk layout.

The dependency is the published crates.io `diskann` / `diskann-providers` /
`diskann-utils` / `diskann-vector` 0.60.0 packages, pinned exactly in
`Cargo.lock`. No algorithm source is vendored or patched in this repository, and
`DiskANNIndex::multi_insert` is not used.

## Create a table

Use a **file-backed SQLite database** for data larger than memory. `:memory:`
and memory-backed temporary databases store their pages in RAM, and a second
connection cannot open them — see [Maintenance on a worker connection](#maintenance-on-a-worker-connection).
DiskANN requires SQLite 3.38.0 or newer; loading the extension for HNSW retains
its existing minimum-version policy.

```sql
CREATE VIRTUAL TABLE embeddings USING vectorlite(
    embedding float32[768] cosine,
    diskann(degree=32, build_list_size=100,
            search_list_size=64, cache_bytes=67108864)
);
```

All options are optional; `diskann()` uses these defaults:

| Option | Default | Meaning |
|---|---:|---|
| `degree` | 32 | Target pruned graph degree; physical lists allow the pinned algorithm's 1.3× slack. |
| `build_list_size` | 100 | Candidate window during native graph insertion. |
| `search_list_size` | 64 | Default query window; effective size is at least effective k. |
| `alpha` | 1.2 | Finite pruning factor, at least 1. |
| `cache_bytes` | 67108864 | Adapter/query workspace budget in bytes. |
| `max_visits` | 65536 | Conservative per-operation/traversal work limit. |

Duplicate and unknown options are errors. Degree must be between 2 and 10000.
List sizes must be positive and no greater than `max_visits`, which is at most
`u32::MAX`. Cache values below 4096 are rejected. These options define the
declaration descriptor; a graph built for one declaration is never published into
a table with a different one.

There is no HNSW-style preallocated `max_elements`. Sparse public rowids are
looked up through SQLite's index, and committed version identifiers are monotonic
and never reused.

## How indexing works

| Phase | Command | Committed work |
|---|---|---|
| Prepare | `index_prepare` | Creates a fenced job with a fixed cutoff and expected population. |
| Snapshot | `index_snapshot_step` (repeat) | Records the frozen `version -> entry` map and a resumable digest chain. |
| Claim | `index_claim_build` | Marks the job as owned by one worker connection. |
| Build | `vectorlite_index_build(...)` | Materializes the snapshot into owned RAM and runs native OneShot in **no** SQLite transaction. |
| Import | `index_import_step` (repeat) | Validates and stages bounded graph chunks into a candidate generation. |
| Validate | `index_validate_step` (repeat) | Re-reads staged rows against the in-memory candidate. |
| Publish | `index_publish` | One fenced metadata transition makes the candidate the published generation. |
| Cleanup | `index_cleanup_step` (repeat) | Bounded reclamation of garbage generations and retired history. |

Two invariants matter:

* **Ingest never builds a graph.** INSERT/UPDATE/DELETE only write canonical
  versioned rows, so they stay O(log N) index writes.
* **Build never borrows SQLite.** The expensive phase runs in owned memory with
  the connection's transaction state verified `NONE`; other connections can keep
  reading and writing, including WAL checkpoints, throughout the build.

Everything remains inside your database file. There is no sidecar index file, no
scratch directory, and no background thread started by the extension.

### Visibility while a generation is pending

* A row inserted after the published generation was built is **canonical but not
  searchable**: `SELECT embedding ... WHERE rowid = ?` returns it, `knn_search`
  does not.
* An **UPDATE** immediately hides the old published version from search and the
  new version stays pending. The row disappears from vector search until the next
  publication rather than returning a stale vector.
* A **DELETE** removes the row from later queries according to ordinary SQLite
  snapshot visibility; it is not removed retroactively from an already-open
  cursor.
* Unchanged rows keep returning results from the published generation.

Query the current state with [`vectorlite_index_status`](#status).

## Ingest

```sql
BEGIN;
INSERT INTO documents(id, title) VALUES (123, 'Example');
INSERT INTO embeddings(rowid, embedding) VALUES (123, ?);
COMMIT;
```

Public rowids are required, nonnegative, and no greater than `i64::MAX`; UPDATE
cannot change a rowid. The vector, its immutable version record, the current
rowid mapping and the canonical/eligible counters change inside the ordinary
journaled statement, so a failed statement or a rolled-back transaction restores
the previous canonical state exactly.

Mutation requires a rollback-capable journal mode. `journal_mode=OFF` is rejected
rather than silently changing host configuration. `MEMORY` provides statement
rollback but no on-disk crash-recovery journal; commit/power-loss guarantees also
depend on the host's SQLite durability settings.

Initial constraint handling is ABORT-only. A duplicate/constraint failure does not
implement SQL `OR IGNORE`, `OR FAIL`, or `OR REPLACE` semantics. SQLite may roll
back an entire transaction for I/O, interruption, or other severe errors: inspect
the connection's transaction state and handle SQLite's error rather than assuming
every failure preserves earlier statements.

Virtual tables retain the existing direct-only restriction: access them from
application SQL rather than through views or triggers. Join metadata by rowid and
issue coordinated writes explicitly within your application transaction.

### Native batch INSERT

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
[public native header](<../vectorlite/include/vectorlite_batch.h>) is installed
under `vectorlite_py/include` with the extension:

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
all buffers. Allocation validity and actual buffer length cannot be proved from a
pointer tag; an arbitrary address with the correct tag violates the native
contract and can cause undefined behavior. The extension checks version, layout,
dimensions, nullness/alignment, length arithmetic and rowids before accessing the
bounded input. It never interprets SQL numbers as addresses.

Only `operation`, the vector column and optional `path` may be supplied; rowids
come from the descriptor and a `distance` input is rejected. Duplicate rowids
within/across chunks or already in the table abort the operation; this is
insert-only, not upsert. Empty batches are no-ops. Numeric/range checks and cosine
normalization follow ordinary DiskANN ingestion and do not modify caller buffers.

A single INSERT may describe any number of vectors. The extension copies and
normalizes bounded working chunks (8 by default, at most 32) and commits them as
one statement: **the entire INSERT has one rollback boundary**, so a failure in a
later chunk restores earlier chunks' vectors, mappings and counters. Chunking
does not commit partial results. Callers needing separate commits can issue
several bounded INSERTs, optionally under one explicit application transaction.

Canonical ingestion performs no graph work and needs no runtime: it is available
from any SQLite callback context, including inside an application Tokio runtime.
Python's standard-library `sqlite3.execute()` cannot bind tagged native pointers;
this interface is for native C/Rust clients, not a raw-address Python workaround.

## Maintenance on a worker connection

Build steps are ordinary SQL, issued by your application. This worker helper runs
the whole lifecycle for a **quiescent bulk load** - the application has finished
writing before maintenance starts - and elects a no-op if the table is empty.
Writes that land on another connection after the captured cutoff are valid and
simply stay pending; see [Post-cutoff writes](#post-cutoff-writes).

```python
import json
import sqlite3

TABLE = "embeddings"          # the DiskANN virtual table
SCHEMA = "main"
PHASE_CAPTURING, PHASE_IMPORT, PHASE_VALIDATING, PHASE_VALIDATED = 0, 2, 3, 4
MAX_RECORDS = 256             # default bounded step size
BUDGET = 512 * 1024 * 1024    # admission estimate, not an RSS cap


def status(worker):
    # Read-only and bounded. Always refresh from SQL after a committed step:
    # the value you read before a step is stale once that step commits.
    return json.loads(worker.execute(
        "SELECT vectorlite_index_status(?,?)", (SCHEMA, TABLE)).fetchone()[0])


def control(worker, operation, **options):
    # One autocommit statement per step; never wrap the campaign in BEGIN.
    worker.execute(f"INSERT INTO {TABLE}(operation, path) VALUES (?, ?)",
                   (operation, json.dumps(options, separators=(",", ":"))))


def maintain(database, extension):
    worker = sqlite3.connect(database, isolation_level=None)
    try:
        worker.enable_load_extension(True)
        worker.load_extension(extension)

        # This helper elects a no-op on an empty table: a bulk loader has nothing
        # to capture. The protocol itself CAN publish an explicit empty generation
        # (a build of population 0 yields zero graph nodes); it is just pointless
        # work here.
        if status(worker)["canonical_count"] == 0:
            return status(worker)

        control(worker, "index_prepare")                 # or "consolidate"
        current = status(worker)["current_job"]
        job, fence = current["job_id"], current["fence"]

        # 1. Capture the frozen snapshot in bounded steps until it is sealed.
        while status(worker)["phase"] == PHASE_CAPTURING:
            control(worker, "index_snapshot_step",
                    job_id=job, fence=fence, max_records=MAX_RECORDS)

        # 2. Claim the job for this connection, then build in owned RAM with no
        #    SQLite transaction. The candidate token is bound to this connection.
        control(worker, "index_claim_build", job_id=job, fence=fence)
        candidate = json.loads(worker.execute(
            "SELECT vectorlite_index_build(?,?,?)",
            (SCHEMA, TABLE, json.dumps(
                {"job_id": job, "fence": fence,
                 "memory_budget_bytes": BUDGET}, separators=(",", ":"))),
        ).fetchone()[0])
        token = candidate["candidate_token"]

        # 3. Import bounded chunks, then validate every chunk. Both phases are
        #    repeated: one step can never cover a population larger than a chunk.
        while status(worker)["phase"] == PHASE_IMPORT:
            control(worker, "index_import_step", job_id=job, fence=fence,
                    candidate_token=token, max_records=MAX_RECORDS)
        while status(worker)["phase"] == PHASE_VALIDATING:
            control(worker, "index_validate_step", job_id=job, fence=fence,
                    candidate_token=token, max_records=MAX_RECORDS)

        # 4. Publish atomically, then release the retained candidate RAM only
        #    after the publishing transaction has committed.
        control(worker, "index_publish", job_id=job, fence=fence,
                candidate_token=token)

        # Publication clears the active job and makes the captured population
        # searchable. pending_count is whatever arrived after the cutoff, so it is
        # reported rather than asserted: a concurrent writer legitimately leaves a
        # backlog.
        final = status(worker)
        assert final["active_job"] is None
        assert final["published_generation"] is not None
        control(worker, "index_cleanup_step", max_records=MAX_RECORDS)
        return final
    finally:
        worker.close()   # closing the connection also drops any candidate RAM


final = maintain("vectors.db", "vectorlite.dylib")
print(final["published_generation"], final["eligible_count"])
```

The phase numbers match the durable job encoding (0 capturing, 1 sealed, 2
building/importing, 3 validating, 4 validated, 5 completed, 6 abandoned), and
`index_status` reports them with the fence and cursors. Notes:

* Capturing is phase 0 and ends when the snapshot is sealed (phase 1); the loops
  above therefore stop on the first committed step that leaves their phase.
* Every loop condition is re-read from SQL after the previous step committed, so a
  finished phase is never re-run and a partially imported generation is never
  published.
* `index_publish` only accepts a fully validated candidate, so the validation
  loop must terminate on phase 4 before publication; skipping it fails closed.
* A successful publication clears the active job, so `index_status` then reports
  `active_job: null` and a `published_generation`. `pending_count` is NOT
  necessarily 0 - it counts canonical rows the published generation does not cover,
  which includes everything written after the captured cutoff. Import and validation
  cursors count native graph nodes: the captured population plus the retained
  frozen root.
* A lost candidate (worker killed, connection closed, job superseded) needs
  `index_restart` with a fresh fence and a new build rather than reusing partial
  SQL progress. `index_cancel` abandons the job; `index_cleanup_step` reclaims
  garbage generations and retired history in bounded chunks.
* The example elects a no-op on an empty table because a bulk loader has nothing
  to index. The protocol itself supports population 0: prepare, capture (nothing to
  capture), claim, build (an explicit empty generation, no native constructor),
  import and validate (no nodes) and publish all succeed, which is how to publish
  an empty generation deliberately.

### Post-cutoff writes

Native compute holds no SQLite transaction, so it does not stop ingestion: a row
committed on another connection after the job captured its cutoff is ordinary
canonical data - readable by rowid, counted in `pending_count`, absent from
`knn_search` results of the currently published generation, and picked up by the
next maintenance cycle. The bounded capture, import, validation and publication
steps are ordinary journaled SQLite statements, so like any writer they hold the
database write lock briefly and a concurrent writer can see `SQLITE_BUSY`; that is
ordinary locking, not maintenance blocking ingestion for a whole build. Publishing a candidate built from an older cutoff is therefore still a
**success**; its post-publication status is `active_job: null`,
`published_generation` set and `pending_count` equal to the rows written after
the cutoff.

An UPDATE behaves the same way for eligibility: the previous version stops
matching the current-version check immediately, so that row is absent from vector
search until its new version is published. Rows the published generation does
cover are unaffected.

Rules that matter:

* **Do not wrap the campaign in one `BEGIN`/`COMMIT`.** Each step must commit
  before it is durable. A control call that returns inside your transaction is
  not committed until you commit, and the writer lock is held until then.
* **Use a separate worker connection for a file-backed database.** The build call
  occupies its own connection for the whole native build while holding no SQLite
  transaction; your application connection keeps working. `:memory:` and
  `TEMP` databases cannot be opened by a second connection, so run maintenance on
  the owning connection there.
* **The build call is compute-only.** While it runs, the worker connection is
  busy inside that statement, but its transaction state must be `NONE`: it is
  rejected before any native construction if the connection is in an explicit
  transaction, inside a `SAVEPOINT`, holding another cursor, or already building.
  Those rejections are deliberate, not incidental.
* **The candidate lives in worker RAM.** The opaque `candidate_token` is only
  valid for that connection, schema/table lifetime, job, fence and attempt. If the
  worker process exits, the candidate is gone: cancel or restart the job (which
  allocates a fresh fence) and run a new build; durable canonical data and the
  job's SQL progress are unaffected.
* **Publication is a bounded metadata transition.** It does not rescan the
  population; full validation happened in the bounded validation steps.
* **Loading the extension twice on one connection fails closed** rather than
  overwriting the existing Vectorlite functions.

Applications can also run these calls inline, for example once after a bulk
import, accepting that the calling connection is busy during the build.

### Control operations

`path` is a TEXT JSON object. Every identifier is a JSON integer or string, never
a pointer or address.

| Operation | JSON |
|---|---|
| `index_prepare` (or `consolidate`, which is an alias) | `{}` |
| `index_snapshot_step` | `job_id`, `fence`; optional `max_records`, `max_bytes` |
| `index_claim_build` | `job_id`, `fence` |
| `index_import_step` | `job_id`, `fence`, `candidate_token`; optional `max_records`, `max_bytes` |
| `index_validate_step` | `job_id`, `fence`, `candidate_token`; optional `max_records`, `max_bytes` |
| `index_publish` | `job_id`, `fence`, `candidate_token` |
| `index_cancel` | `job_id`, `fence`; optional `force` (default false) |
| `index_restart` | `job_id`, `fence`; optional `force` (default false) |
| `index_cleanup_step` | optional `max_records`, `max_bytes` |

`max_records` defaults to 256 and accepts 1 through 256. `max_bytes` defaults to
1 MiB and must be positive and no larger than the declared `cache_bytes`.
Only one job may be active per index. Stale fences, other connections' claims and
lost candidates are reported as conflicts (`SQLITE_BUSY`) rather than silently
reused; state-changing calls always re-read the current owner and fence inside the
journaled statement.

`consolidate` is retained as an alias for `index_prepare`: it no longer performs
a synchronous rebuild.

### Build

```sql
SELECT vectorlite_index_build('main', 'embeddings', ?);
```

The third argument is a JSON object with required positive integers `job_id` and
`fence`, required positive `memory_budget_bytes`, and optional `workers` which
must be 1. The result is bounded JSON describing the candidate:

```json
{"candidate_token":"cand:...","population":10000,"node_count":10001,"root":10000,
 "medoid_ordinal":123,"edge_count":320032,"input_hash":"...","config_hash":"...",
 "candidate_hash":"...","retained_bytes":409152,"stats":{...}}
```

`root` is the native frozen start point (equal to the population), and native
node IDs `0..population-1` are ordinary data. An empty population produces an
explicit empty generation with no native constructor call.

### Status

```sql
SELECT vectorlite_index_status('main', 'embeddings');
```

Read-only and bounded (no corpus count, no vector scan); it performs no recovery
writes and can be called on a read-only connection:

```json
{"format":4,"index_uuid":"idx:...","canonical_count":10100,
 "pending_count":100,"eligible_count":10000,"retired_count":3,"revision":412,
 "published_generation":7,"published_count":10000,"published_source_count":10000,
 "published_source_hash":"...","candidate_generation":null,"active_job":null,
 "current_job":null,"phase":null,"fence":null,"failure":null,"cleanup_entry_id":0}
```

`current_job`, when present, reports `job_id`, `generation_id`,
`parent_generation`, `phase`, `fence`, `cutoff_revision`, `cutoff_entry_id`,
`expected_count`, `captured_count`, `scan_entry_id`, `import_cursor`,
`validation_cursor`, `cleanup_cursor`, `source_hash`, `config_hash`,
`candidate_hash`, `snapshot_chain` and `failure_code`.

`pending_count = canonical_count - eligible_count` is the number of stored rows
that no published generation currently covers. It is the freshness backlog.

## Query

```sql
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

Every KNN path restricts results to the published generation **and** to the
current version of each row: pending, superseded and deleted versions are rejected
before the top-k is truncated. This includes small rowid filters, large-k requests
and underfilled filtered searches that take the exact streaming fallback, so
approximate and exact paths agree on eligibility. Navigation may read retired
immutable vectors as bridges, but they cannot appear as results. An empty
published population returns an empty result even when rows are pending.

Projected vector bytes are materialized with their distances, so a later
same-connection update cannot mix an old distance with a changed vector. Cursors
retain a SQLite snapshot anchor; close them promptly to release WAL snapshots.
Other connections follow ordinary SQLite isolation and `SQLITE_BUSY_SNAPSHOT`
rules.

## Vector representation and numeric policy

SQL inputs and outputs remain little-endian float32 BLOBs. L2 stores float32
values unchanged and reports **squared** distance. Cosine normalizes both stored
vectors and queries once, at ingest, using the existing SIMD policy and reports
`1 - dot`; reading the vector column returns the normalized representation, not
the original unscaled input. Zero vectors retain the existing zero-normalization
behavior. A build consumes already-normalized bytes and never renormalizes them.

DiskANN rejects NaN/infinity and dimensions that do not match the declaration. To
avoid overflow in graph distances, L2 inputs require squared norm no greater than
`f32::MAX / 4`. Cosine inputs require a finite float32-compatible norm. Nonfinite
computed distances are errors. These DiskANN checks do not change HNSW's numeric
policy.

## Memory and limits

`cache_bytes` is an **adapter/query workspace** budget: admitted result sets,
filter/projection accounting, and bounded SQL-adapter buffers. It is not a
process-wide RSS cap. Prepared query readers are retained only for the duration of
one operation and are admitted from this same workspace after the owner reservation
and the core scratch floor, with a 256 KiB reserve bound; when they cannot be
admitted the adapter silently uses its scalar read path, so admission outcomes do
not change. `max_visits` bounds traversal work per operation across
roughly shared allowances. Exceeding a limit returns an explicit error; nothing is
silently truncated.

The build has a separate, explicit `memory_budget_bytes` covering the owned
snapshot copy, native provider storage (including the frozen dense buffer), graph
and construction scratch, extraction overlap and the retained candidate. It is an
admission estimate, not an allocator-enforced RSS cap, and it does not include the
2 MiB worker stack or the native coordinator stack, which admission accounts for
separately. Two coexistence costs are charged before the native constructor runs:
the guard's owned worker stack and any other retained candidate on the connection.

Practical consequences:

* If admission rejects the build with `SQLITE_TOOBIG`, raise
  `memory_budget_bytes` or reduce the population; the previous published
  generation and all canonical rows are untouched, and those rows stay pending.
* Native allocation failures and panics are **not** recoverable errors. This
  library is built with `panic = "abort"`, and the native builder uses infallible
  Rust allocations, so an out-of-memory condition can terminate the host process.
  Applications that cannot accept that should run maintenance in their own worker
  process. The database file is unaffected because the build holds no SQLite
  transaction.
* Cancellation is cooperative: it is checked between completed native phases and
  fenced at publication, and it does not preempt arbitrary native CPU. To stop a
  build hard, terminate the worker process.

### Optional host-owned pager tuning

SQLite pager memory is separate from both budgets. For an index in `main`, this
is an explicit application-owned example, not an extension default:

```sql
PRAGMA main.cache_size = -32768; -- suggested 32 MiB pager budget
```

Negative `cache_size` uses KiB. It is a SQLite cache hint, not a hard RSS cap.
The setting affects other tables in that database on the connection. For an
attached database, use its schema instead of `main`, and budget concurrent
connections separately. The extension never silently sets this or changes the
application's journal, synchronous, spill or mmap policy. Keep durability settings
fixed when comparing builds.

## Failure, recovery and cleanup

* **Statement/transaction rollback** restores canonical rows and counters exactly,
  including inside `SAVEPOINT`s.
* **A failed build** abandons the candidate and reports the failure; the previously
  published generation keeps serving queries and canonical rows stay pending.
  Repeat computation needs an explicit fresh fence (`index_restart`), never reuse
  of a partly imported candidate.
* **A worker crash** leaves SQL progress durable but the RAM candidate lost. Reopen
  the database, read `vectorlite_index_status` to confirm the job's phase and
  fence, then continue with `index_restart` (fresh fence) followed by a new build,
  import and publish. A crashed build never exposes a partial graph.
* **SQLite rollback cannot undo RAM side effects.** If a transaction that released
  candidate memory later rolls back, the restored job cannot reuse that memory;
  cancel/restart and build again. Commit publication before calling
  `index_cleanup_step` so the retained candidate is still available for a retry.
* **Read-only connections** can open, read, query the published generation and call
  `vectorlite_index_status`; build and control operations need a writable
  connection and fail with `SQLITE_READONLY`.
* **Retired history** is reclaimed only by bounded `index_cleanup_step` calls,
  never by unbounded `DELETE`/`DROP`. Active jobs, the current mapping, published
  and staged graph references and snapshot inputs are protected.

Data automatically reopens with the database; there is no external index
`save`/`load` command. Use normal SQLite backup/checkpoint procedures — the
canonical vectors, published graph and job progress are all inside the database
file. A running WAL database can have ordinary SQLite WAL/shared-memory files;
these are not separate ANN index sidecars. The transaction guarantees above apply
to DiskANN and ordinary SQLite tables, **not** to HNSW mutations in the same
transaction.

Internal shadow tables, journal-carrier rows, the private atomic function and the
candidate token internals are **not a supported editing/import API**. Changing
them directly can invalidate the index; attempted guard corruption or unexpected
guard triggers are rejected before/within mutation. Invalid record lengths, node
IDs, mappings or formats fail closed when accessed and are never reported as
missing rows.

## Platforms and validation

The release targets remain Linux x64, Windows x64, and macOS arm64 ≥11. Native
macOS dependencies must be built with the pinned deployment floor, not merely
relinked into a dylib tagged 11.0. CI/package checks inspect the resulting native
artifacts and notices. Tests on a current macOS host are not evidence of execution
on an actual macOS 11 machine.

Performance is workload-dependent. Report, separately: canonical ingest latency and
lock hold time, maintenance phase times, time-to-searchable and freshness coverage
(`pending_count`), native build wall time and peak admission, query latency and
recall, and database/WAL size. Recall must be compared against unmodified native
DiskANN OneShot at matched inputs and settings; a database merely larger than the
pager cache does not establish a physical beyond-RAM benchmark. Do not interpret
the repository's historical HNSW figures as new DiskANN performance results.

### Measured query latency (experimental, one host)

On the release build (Apple M4, macOS arm64, file-backed WAL/FULL, 100 seeded
queries, k=10, degree 32, build list 100, `cache_bytes` 64 MiB, five interleaved
AB/BA runs) the SQL `knn_search` p50 fell from **14.276 ms to 1.903 ms** at
10,000 x 128 with L=10 (7.50x) and from **77.667 ms to 10.749 ms** at L=100 (7.23x),
with identical recall@10. At 10,000 x 768 the p50 is **2.078 / 6.974 / 12.637 ms**
at L=10/50/100 (6.2-6.5x versus the previous adapter). The improvement comes from
preparing a small set of readers once per operation and reusing them, and from
reading a candidate's identity and vector in one statement.

These are single-host, warm-cache measurements, not a platform, cold-disk or
beyond-RAM claim. The SQL surface, published-generation/current-version eligibility,
`max_visits` and `cache_bytes` remain unchanged. Query readers exist only for the
duration of one operation; there is no cross-query vector or identity cache, so every
query reads through SQLite and observes its own snapshot. When two faults coincide
(for example corruption at a node whose visit would also exhaust `max_visits`), the
reported error may differ from earlier versions; a failing query never returns rows,
and stale or superseded versions are never returned.