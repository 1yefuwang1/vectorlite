# Explicit Index Persistence Design

Date: 2026-06-14
Status: Approved (design)

## Problem

The HNSW index that backs a vectorlite virtual table is an in-memory
structure. Today its on-disk location is supplied as the third argument to
`CREATE VIRTUAL TABLE`:

```sql
CREATE VIRTUAL TABLE my_table USING vectorlite(
    my_vec float32[4],
    hnsw(max_elements=1000),
    'index.bin'
);
```

SQLite persists the full text of that statement in `sqlite_master`. On every
subsequent connection SQLite replays the statement verbatim, so the index file
path is effectively baked into the schema. Consequences:

- The index file cannot be moved or renamed without editing the stored schema.
- The `.db` file and the index file are separate artifacts that must travel
  together, yet nothing ties them together.
- The path is interpreted relative to the process working directory, which is
  fragile.

The actual vector data lives entirely in the external index file; the `.db`
only stores the schema. So a "moved file" silently breaks the table.

## Goal

Decouple index persistence from the schema. The path must never be stored
anywhere. Persistence becomes an explicit, user-driven operation that can target
any location at any time.

## Approach

Persistence is **explicit only**. A vectorlite table is always a pure in-memory
index. There is no automatic load on connect, no automatic save on disconnect,
and no automatic file deletion on drop. The user persists and restores the index
through explicit SQL commands, choosing the path each time.

Binding is done through the **INSERT-command idiom** (the same pattern FTS5 uses
for `INSERT INTO ft(ft) VALUES('optimize')`). The command is delivered through
hidden columns, which routes it to `xUpdate`, where the `VirtualTable*` is
directly available. This avoids a table-name registry and any "force connect"
machinery, and it works correctly on empty tables (unlike a scalar function in
`SELECT ... FROM vtab`, which is never called when there are no rows).

This is a **breaking change** and warrants a major version bump.

## Detailed Design

### 1. CREATE VIRTUAL TABLE signature (clean break)

The module accepts exactly two arguments: the vector space and the index
options. The previous optional third path argument is removed.

```sql
CREATE VIRTUAL TABLE my_table USING vectorlite(
    my_vec float32[4],
    hnsw(max_elements=1000)
);
```

- `InitVirtualTable` requires `argc == 2 + kModuleParamOffset`. A 3-argument
  form is rejected with a clear error that points the user to the new
  `operation`/`path` INSERT-command mechanism.
- Existing databases created with the 3-argument form will fail to connect until
  recreated. This is intentional and documented in the changelog under the major
  version bump.
- `Create`, `Connect`, `Disconnect`, and `Destroy` no longer touch the
  filesystem.

### 2. Persistence via INSERT-commands

Two hidden columns are appended to the declared schema:

```sql
CREATE TABLE X(my_vec, distance REAL HIDDEN, operation TEXT HIDDEN, path TEXT HIDDEN)
```

The column index enum becomes:

```cpp
enum ColumnIndexInTable {
  kColumnIndexVector,     // 0
  kColumnIndexDistance,   // 1
  kColumnIndexOperation,  // 2
  kColumnIndexPath,       // 3
};
```

Usage:

```sql
INSERT INTO my_table(operation, path) VALUES ('save', '/path/index.bin');
INSERT INTO my_table(operation, path) VALUES ('load', '/path/index.bin');
```

In `xUpdate`, a declared column `i` is delivered as `argv[2 + i]`. For an INSERT
(`argv[0]` is NULL):

1. If `argv[2 + kColumnIndexOperation]` is a non-NULL TEXT value, treat the
   statement as a persistence command:
   - Read the operation string and the `path` string
     (`argv[2 + kColumnIndexPath]`). A missing or non-TEXT `path` is an error.
   - `'save'` → `SaveTo(path)`; `'load'` → `LoadFrom(path)`; any other value is
     an error.
   - Set `*pRowid = 0`. No vector row is inserted.
   - On failure, set `zErrMsg` and return `SQLITE_ERROR` so the statement fails.
2. Otherwise, fall through to the existing vector insert path unchanged.

`Column()` returns `sqlite3_result_null` for `kColumnIndexOperation` and
`kColumnIndexPath`; they are a write-only command channel.

`BestIndex` and `FindFunction` are unaffected: the vector column remains index 0,
and the new columns never participate in KNN constraints.

### 3. VirtualTable changes

Remove:

- The `file_path_` member.
- `LoadIndexFromFile()`, `SaveIndexToFile()`, `DeleteIndexFile()`.
- Their call sites in `Create`/`Connect`/`Disconnect`/`Destroy` and the
  `load_from_file` parameter threaded through `InitVirtualTable`.

Add:

- `absl::Status SaveTo(const std::string& path)` — serializes `index_` to
  `path`, overwriting any existing file. Does not create missing parent
  directories; a missing directory surfaces as an error.
- `absl::Status LoadFrom(const std::string& path)` — loads a serialized index
  from `path` into a freshly constructed index, then swaps it in only on
  success, replacing all current contents. Validates that the loaded per-vector
  data size (`label_offset_ - offsetData_`, both read from the file) matches the
  table's vector space (`space_.space->get_data_size()`); on mismatch it returns
  an error and leaves the table's current in-memory index unchanged. A missing
  or unreadable file is an error.

Both wrap hnswlib's `saveIndex`/`loadIndex` (which operate on a filename) and
translate exceptions into `absl::Status`, mirroring the existing helpers.

### 4. Edge cases and decisions

- `load` on a non-empty table replaces all existing contents.
- `save` overwrites an existing file; it does not create missing parent
  directories.
- `operation`, `path`, and `distance` are reserved and cannot be used as the
  vector column name. This is documented.
- INSERT-commands return no value (no row is inserted). Errors surface as a
  failed statement through `zErrMsg`.
- In-memory tables that are never saved lose their contents when the connection
  closes. This is expected and accepted.

### 5. Tests

C++ unit tests (`virtual_table` / extension level):

- save then load round-trip preserves vectors and KNN results.
- `load` replaces pre-existing data.
- dimension / vector-space mismatch on `load` errors and leaves the table's
  existing index unchanged.
- `load` from a missing file errors.
- unknown `operation` value errors.
- 3-argument `CREATE VIRTUAL TABLE` is rejected.
- `operation` and `path` are hidden from `SELECT *`.

Python integration tests (`bindings/python/vectorlite_py/test`):

- Drop the path argument from `CREATE VIRTUAL TABLE`.
- Round-trip an index across reconnects using the `save`/`load` INSERT-commands.

### 6. Documentation

Update to the new API and remove references to the path argument and implicit
persistence:

- `README.md`
- `doc/markdown/*` (getting-started, overview, api)
- `examples/index_serde.py`
- Node.js binding docs/tests that reference the path argument.

## Out of Scope

- Storing the index as BLOBs inside the SQLite database itself (a separate
  approach that was considered and set aside).
- Path resolution relative to the database file.
- Any backward-compatibility shim for the 3-argument form.
