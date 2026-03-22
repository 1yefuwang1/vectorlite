# Async Compact/Rebuild for Shadow Table Storage

## Problem

The shadow table storage `compact` and `rebuild` commands can run for a long time on large indices. During execution they block the calling connection, preventing the user from running queries or inserting/deleting vectors.

## Goals

- `compact` and `rebuild` return immediately, running work on a background thread.
- Users can continue reading (KNN queries) and writing (inserts/deletes) while the background operation runs.
- Users can poll the status of the background operation via `vectorlite_task_status()`.
- Crash safety: if the process dies mid-operation, the database is left in a consistent state.

## Constraints

- **Shadow table mode only.** Async compact/rebuild is only available when `use_shadow_tables_ == true`. If issued on a file-based virtual table, `ExecuteCommand` returns `SQLITE_ERROR` with an appropriate message.
- **SQLite WAL journal mode is not required** but strongly recommended for best concurrency. In rollback journal mode, the background thread's writes and the main thread's WAL appends will serialize via SQLite's database-level lock, causing brief `SQLITE_BUSY` retries handled by the busy timeout. In WAL mode, concurrent writes from separate connections proceed with finer-grained locking.

## Design

### BackgroundTask Class

A generic, reusable thread lifecycle manager. Knows nothing about WAL, indexes, or SQLite. All public methods must be called from a single thread (the SQLite connection's thread). Only the internal thread body runs on the background thread.

```cpp
class BackgroundTask {
 public:
  enum class State { idle, running, completed, failed };

  BackgroundTask() = default;
  ~BackgroundTask();  // joins thread if joinable

  BackgroundTask(const BackgroundTask&) = delete;
  BackgroundTask& operator=(const BackgroundTask&) = delete;

  // Launch a task. Returns false if one is already running.
  // Uses compare-and-swap on state_ to guarantee at most one
  // background operation in flight.
  bool Start(std::function<absl::Status()> fn);

  State state() const { return state_.load(); }
  const std::string& error() const { return error_; }

  // Reset completed/failed back to idle. Returns false if running.
  bool Reset();

 private:
  std::atomic<State> state_{State::idle};
  // Memory ordering note: error_ is written by the background thread BEFORE
  // storing state_ = failed (seq_cst). The main thread reads state_ BEFORE
  // reading error_. This establishes a happens-before relationship making
  // the non-atomic access to error_ safe.
  std::string error_;
  std::thread thread_;
};
```

**Thread lifecycle:**

- `Start()`: CAS `state_` from non-`running` to `running`. If already `running`, return false. If previous thread is joinable, join it (instant — it already finished). Launch new thread.
- Thread body: call the provided `fn`. On success, store `completed`. On failure, write error message to `error_`, then store `failed`.
- `~BackgroundTask()`: if thread is joinable, join it (blocks until background work finishes).

### VirtualTable Changes

New members:

```cpp
BackgroundTask bg_task_;
std::string db_name_;  // schema/database name from argv[1] in xCreate/xConnect
```

`db_name_` is captured from `argv[1]` during `xCreate`/`xConnect` and used to obtain the database file path via `sqlite3_db_filename(db_, db_name_.c_str())`. This correctly handles both the default `"main"` database and attached databases.

### Refactoring of Existing Methods

The existing synchronous `Compact()` and `Rebuild()` methods are removed. Their logic is split into reusable helpers:

- `SerializeIndex()` — already exists, unchanged. Returns serialized index as `std::string`.
- `QueryMaxWalSeq()` — new helper. Returns `max(seq)` from the WAL table, or 0 if empty.
- `SnapshotVectors()` — new helper. Collects all non-deleted vectors (label + data) from the in-memory index. Returns a `std::vector<VectorEntry>`.
- `WriteIndexAndCleanupWal(sqlite3* conn, const std::string& serialized, int64_t wal_seq)` — new static helper. Writes serialized chunks to `_index` and deletes WAL entries `<= wal_seq` in a single transaction on the given connection.

`ExecuteCommand("compact")` and `ExecuteCommand("rebuild")` call these helpers to prepare data synchronously, then dispatch the background thread.

### Async Compact Flow

1. **Synchronous (on calling thread):**
   - Capture `wal_seq` via `QueryMaxWalSeq()`.
   - Serialize the in-memory index to a byte buffer via `SerializeIndex()`. Note: this creates a full copy of the index in memory. For a large index (e.g., 1M 384-dim float32 vectors), this is roughly 1.5 GB. This is acceptable because compact is already an explicit user action, and the buffer is freed as soon as the background thread finishes writing it.
2. **Background thread (lambda captures `serialized`, `db_path`, table names, and `wal_seq` by value — no `this` capture):**
   - Open a second `sqlite3*` connection to `db_path` via `sqlite3_open_v2` with `SQLITE_OPEN_READWRITE`. Set a busy timeout (e.g., 5000ms) via `sqlite3_busy_timeout` to handle contention with the main thread.
   - Call `WriteIndexAndCleanupWal(conn, serialized, wal_seq)`.
   - Close the second connection.
   - Set state to `completed` or `failed`.

### Async Rebuild Flow

1. **Synchronous (on calling thread):**
   - Capture `wal_seq` via `QueryMaxWalSeq()`.
   - Snapshot all non-deleted vectors via `SnapshotVectors()`. Note: for large indices, this iterates all elements and copies their data. The duration is proportional to the index size but involves only memory reads (no DB access).
2. **Background thread (lambda captures `snapshot`, `db_path`, table names, `wal_seq`, and index construction parameters by value — no `this` capture):**
   - Build a new HNSW index from the snapshot (CPU-only, no DB access).
   - Serialize the new index to a byte buffer.
   - Open a second `sqlite3*` connection (same approach as compact).
   - Call `WriteIndexAndCleanupWal(conn, serialized, wal_seq)`.
   - Close the second connection.
   - Set state to `completed` or `failed`.

### No In-Memory Index Swap

The background operation writes the rebuilt/compacted index only to the shadow tables. The current in-memory index continues serving queries and accepting inserts/deletes. Those new operations go to the WAL as usual. On next connect (e.g., reopen the database), the fresh index is loaded from the shadow table and any remaining WAL entries are replayed.

**Trade-off:** For rebuild, this means the user pays the rebuild cost but does not see improved query recall or reduced memory usage from deleted-node removal until they disconnect and reconnect. This is the price for avoiding a mutex on `index_` and keeping the design simple.

### Concurrency Model

- **Main thread reads `index_`**: KNN queries read the in-memory index. No contention because the background thread never touches it.
- **Main thread writes `index_`**: Inserts/deletes modify the in-memory index and append to `_wal`. No contention with the background thread.
- **Background thread writes to shadow tables**: Uses its own `sqlite3*` connection. The main thread's WAL appends and the background thread's chunk writes are on separate connections. SQLite serializes concurrent writers with its internal locking (WAL mode allows finer-grained concurrency; rollback mode uses a database-level lock). The busy timeout on the second connection handles transient `SQLITE_BUSY` from contention.
- **No `this` capture in background lambda**: The lambda captures all needed data by value/move. The background thread never accesses `index_`, `db_`, or any other VirtualTable state. This eliminates data race risks entirely.

### Status Polling

Implemented as a virtual table overloaded function via `xFindFunction`, the same mechanism used for `knn_search`.

```sql
SELECT vectorlite_task_status() FROM my_vectors LIMIT 1;
-- Returns: 'idle' | 'running' | 'completed' | 'failed: <error message>'
```

**BestIndex change:** Currently `BestIndex` rejects queries with no recognized WHERE constraint. To support status polling (and unconstrained queries in general), `BestIndex` allows an unconstrained plan:

- When no constraints are present, return a plan with `idxNum = 0` (which cannot occur under the current encoding where `idxNum = constraint_count * 2`, since at least one constraint was previously required).
- Set `estimatedCost` to a very high value so the optimizer prefers constrained plans when available.
- In `Filter`, when `idxNum == 0`, push a single dummy entry `{0.0f, 0}` into the cursor's result set.
- In `Column`, when serving a full-scan row, return `NULL` for the vector and distance columns (the caller only cares about function results like `vectorlite_task_status()`).

**Behavioral note:** This means `SELECT * FROM my_vectors` (no WHERE clause) now returns a single dummy row instead of an error. This is a user-visible change but is necessary for status polling to work.

**Registration:** Add a case in `VirtualTable::FindFunction` for `vectorlite_task_status` with `nArg == 0`. The implementation casts `ppArg` to `VirtualTable*`, reads `bg_task_.state()`, and returns the state string.

**Status reset:** Reading the status does NOT auto-reset state. Instead, a separate `reset_task` command is provided:

```sql
INSERT INTO my_vectors(command) VALUES('reset_task');
```

This resets `completed`/`failed` back to `idle`. Returns `SQLITE_ERROR` if the task is still `running`. This avoids TOCTOU races and problems with the function being evaluated multiple times in a single query.

### Disconnect/Destroy Behavior

- **`Disconnect` while task is running:** The `~BackgroundTask` destructor joins the thread, which blocks until the background work finishes. This is a silent hang but is the correct behavior — the background thread must complete its transaction. The user should poll `vectorlite_task_status()` and wait for completion before disconnecting.
- **`Destroy` while task is running:** `Destroy` must check `bg_task_.state()`. If `running`, return `SQLITE_BUSY` without calling `delete vtab` to prevent dropping shadow tables while the background thread is writing to them. The user must wait for the task to complete before dropping the table.

### Command Routing

`ExecuteCommand` dispatches commands as follows:

| Command      | Behavior                                                         |
|--------------|------------------------------------------------------------------|
| `compact`    | Start async compact. Error if already running or file-based mode.|
| `rebuild`    | Start async rebuild. Error if already running or file-based mode.|
| `reset_task` | Reset completed/failed state to idle. Error if running.          |

There is no synchronous path. Both commands always run asynchronously.

### Crash Safety

**Background thread uses an explicit transaction** on the second connection. The entire set of shadow table writes (delete old chunks, insert new chunks, delete WAL entries) is wrapped in `BEGIN`/`COMMIT`. If the process crashes mid-transaction, SQLite rolls back automatically. The old index chunks and WAL entries remain intact, and on next connect the index loads correctly.

**WAL cleanup is bounded** by the `wal_seq` captured at launch time. Only WAL entries with `seq <= wal_seq` are deleted. Entries added by the main thread during the background operation are preserved.

**WAL sequence numbers grow monotonically** because the `seq` column uses `INTEGER PRIMARY KEY AUTOINCREMENT`. SQLite never reuses deleted rowids, so the captured `wal_seq` remains a valid cutoff regardless of what the main thread does.

### Error Handling

If the background operation fails (disk full, serialization error, etc.):

- State is set to `failed`.
- Error message is stored and returned by `vectorlite_task_status()`.
- The shadow table transaction is rolled back. Index and WAL are unchanged.
- The user can query the status, see the error, reset, and retry.

## Files to Change

- `vectorlite/background_task.h` — new file, `BackgroundTask` class declaration
- `vectorlite/background_task.cpp` — new file, `BackgroundTask` implementation
- `vectorlite/virtual_table.h` — add `BackgroundTask bg_task_` member, `std::string db_name_`, declare async helper methods
- `vectorlite/virtual_table.cpp` — remove synchronous `Compact()`/`Rebuild()`, add `QueryMaxWalSeq()`, `SnapshotVectors()`, `WriteIndexAndCleanupWal()`, async dispatch in `ExecuteCommand`, `xFindFunction` addition, `BestIndex` full-scan plan, `Filter`/`Column` dummy-row handling, `Destroy` guard
- `vectorlite/CMakeLists.txt` — add new source files
- `vectorlite/background_task_test.cpp` — unit tests for `BackgroundTask`

## Test Plan

- **BackgroundTask unit tests:** start/complete/fail lifecycle, reject concurrent start, reset semantics, destructor join.
- **Async compact:** insert vectors, trigger compact, verify inserts still work during compact, verify status transitions (idle -> running -> completed).
- **Async rebuild:** insert + delete vectors, trigger rebuild, insert more during rebuild, verify status, disconnect/reconnect to verify rebuilt index loads correctly with post-rebuild WAL replayed.
- **Reject while running:** trigger rebuild, immediately trigger compact, verify error.
- **Destroy guard:** trigger rebuild, attempt DROP TABLE, verify SQLITE_BUSY.
- **File-based mode guard:** create file-based virtual table, attempt compact, verify error.
- **Status polling:** verify `SELECT vectorlite_task_status() FROM vtab LIMIT 1` returns correct state, verify Column returns NULL for vector/distance on full-scan rows.
