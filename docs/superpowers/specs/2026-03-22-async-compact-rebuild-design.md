# Async Compact/Rebuild for Shadow Table Storage

## Problem

The shadow table storage `compact` and `rebuild` commands can run for a long time on large indices. During execution they block the calling connection, preventing the user from running queries or inserting/deleting vectors.

## Goals

- `compact` and `rebuild` return immediately, running work on a background thread.
- Users can continue reading (KNN queries) and writing (inserts/deletes) while the background operation runs.
- Users can poll the status of the background operation via a scalar function.
- Crash safety: if the process dies mid-operation, the database is left in a consistent state.

## Design

### BackgroundTask Class

A generic, reusable thread lifecycle manager. Knows nothing about WAL, indexes, or SQLite.

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
  std::string error_;
  std::thread thread_;
};
```

**Thread lifecycle:**

- `Start()`: CAS `state_` from non-`running` to `running`. If already `running`, return false. Join any previous thread (instant, since it already finished). Launch new thread.
- Thread body: call the provided `fn`. On success, store `completed`. On failure, store error message and `failed`.
- `~BackgroundTask()`: join thread if joinable (blocks until background work finishes).

### VirtualTable Changes

New member:

```cpp
BackgroundTask bg_task_;
```

### Async Compact Flow

1. **Synchronous (on calling thread):**
   - Capture `wal_seq = max(seq) FROM _wal`.
   - Serialize the in-memory index to a byte buffer via `SerializeIndex()`.
2. **Background thread:**
   - Open a second `sqlite3*` connection to the same database file.
   - In a single transaction:
     - `DELETE FROM _index`.
     - Insert all serialized chunks into `_index`.
     - `DELETE FROM _wal WHERE seq <= wal_seq`.
   - Close the second connection.
   - Set state to `completed` or `failed`.

### Async Rebuild Flow

1. **Synchronous (on calling thread):**
   - Capture `wal_seq = max(seq) FROM _wal`.
   - Snapshot all non-deleted vectors from the in-memory index (label + data).
2. **Background thread:**
   - Build a new HNSW index from the snapshot (CPU-only, no DB access).
   - Serialize the new index to a byte buffer.
   - Open a second `sqlite3*` connection to the same database file.
   - In a single transaction:
     - `DELETE FROM _index`.
     - Insert all serialized chunks into `_index`.
     - `DELETE FROM _wal WHERE seq <= wal_seq`.
   - Close the second connection.
   - Set state to `completed` or `failed`.

### No In-Memory Index Swap

The background operation writes the rebuilt/compacted index only to the shadow tables. The current in-memory index continues serving queries and accepting inserts/deletes. Those new operations go to the WAL as usual. On next connect (e.g., reopen the database), the fresh index is loaded from the shadow table and any remaining WAL entries are replayed.

### Concurrency Model

- **Main thread reads `index_`**: KNN queries read the in-memory index. No contention because the background thread never touches it.
- **Main thread writes `index_`**: Inserts/deletes modify the in-memory index and append to `_wal`. No contention with the background thread.
- **Background thread writes to shadow tables**: Uses its own `sqlite3*` connection. In WAL mode, SQLite allows concurrent writes from separate connections (one writer at a time with busy-wait). The main thread's WAL appends and the background thread's chunk writes are on separate connections, so they serialize via SQLite's WAL locking.

### Status Polling

Implemented as a virtual table overloaded function via `xFindFunction`, the same mechanism used for `knn_search`.

```sql
SELECT vectorlite_task_status() FROM my_vectors LIMIT 1;
-- Returns: 'idle' | 'running' | 'completed' | 'failed: <error message>'
```

- Registered in `VirtualTable::FindFunction` for `nArg == 0`.
- The implementation casts `ppArg` to `VirtualTable*`, reads `bg_task_.state()`, and returns the state string.
- Reading `completed` or `failed` resets state back to `idle`, allowing a new operation to be launched.

### Command Routing

`ExecuteCommand` dispatches `compact` and `rebuild` to the async path. If a background operation is already running, it returns `SQLITE_ERROR` with an appropriate error message.

| Command   | Behavior                                              |
|-----------|-------------------------------------------------------|
| `compact` | Start async compact. Error if already running.        |
| `rebuild` | Start async rebuild. Error if already running.        |

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
- The user can query the status, see the error, and retry.

## Files to Change

- `vectorlite/background_task.h` — new file, `BackgroundTask` class declaration
- `vectorlite/background_task.cpp` — new file, `BackgroundTask` implementation
- `vectorlite/virtual_table.h` — add `BackgroundTask bg_task_` member, declare async helper methods
- `vectorlite/virtual_table.cpp` — async compact/rebuild implementations, `xFindFunction` addition, `ExecuteCommand` changes
- `vectorlite/CMakeLists.txt` — add new source files
- `vectorlite/background_task_test.cpp` — unit tests for `BackgroundTask`
