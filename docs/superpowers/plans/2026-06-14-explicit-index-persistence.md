# Explicit Index Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the schema-baked index file path with explicit, user-driven persistence via `INSERT INTO <table>(operation, path) VALUES('save'|'load', '<path>')`.

**Architecture:** A vectorlite virtual table becomes pure in-memory. Two hidden columns (`operation`, `path`) route persistence commands through `xUpdate`, where the `VirtualTable*` is directly available, so no table-name registry is needed. `CREATE VIRTUAL TABLE` drops its optional 3rd (path) argument — a clean break warranting a major version bump.

**Tech Stack:** C++17, SQLite virtual table API, hnswlib 0.8.0, abseil `absl::Status`, GoogleTest, pytest + apsw.

---

## Background the engineer needs

- Source of the virtual table: `vectorlite/virtual_table.h` and `vectorlite/virtual_table.cpp`.
- The module is registered in `vectorlite/vectorlite.cpp` (`sqlite3_create_module(db, "vectorlite", ...)`).
- End-to-end behavior is tested through Python integration tests in `bindings/python/vectorlite_py/test/vectorlite_test.py`, which load the compiled extension into an in-memory SQLite DB via apsw. This is the established way to test full SQL behavior (the C++ `unit_test` target only tests components, not SQL).
- Build + test (from repo root): `sh build.sh`. This runs `cmake --preset dev`, builds, runs `ctest` (C++ unit tests), then `pytest bindings/python/vectorlite_py/test`. A POST_BUILD step copies the freshly built `vectorlite` shared library into `bindings/python/vectorlite_py/`, so pytest always exercises the latest build.
- To run only the Python integration tests after a build: `pytest bindings/python/vectorlite_py/test -v`.

### How SQLite delivers columns to `xUpdate`

For a declared table with columns `[c0, c1, ...]`, an INSERT calls `xUpdate` with
`argv[0]` = NULL (the rowid of the row being replaced; NULL for a pure insert),
`argv[1]` = the new rowid, and `argv[2 + i]` = the value for declared column `i`.

After this change the declared columns are:

| index | name (enum) | meaning |
|-------|-------------|---------|
| 0 | `kColumnIndexVector` | the vector column (`my_embedding`) |
| 1 | `kColumnIndexDistance` | `distance` (hidden, query-only) |
| 2 | `kColumnIndexOperation` | `operation` (hidden, write-only command) |
| 3 | `kColumnIndexPath` | `path` (hidden, write-only command) |

So in `xUpdate` the operation value is `argv[2 + kColumnIndexOperation]` (= `argv[4]`)
and the path value is `argv[2 + kColumnIndexPath]` (= `argv[5]`).

---

## Task 1: Failing Python integration test for explicit persistence

This is the spec-level red test. It will fail against the current extension because the new behavior doesn't exist yet and the old 3-arg/auto-persistence behavior is still present.

**Files:**
- Modify: `bindings/python/vectorlite_py/test/vectorlite_test.py` (replace the existing `test_index_file` function, lines 134-195, with the new tests below)

- [ ] **Step 1: Replace `test_index_file` with explicit-persistence tests**

Delete the entire existing `test_index_file(random_vectors)` function (the `def test_index_file(...)` block spanning lines 134-195) and insert this in its place:

```python
def test_save_and_load_round_trip(random_vectors):
    with tempfile.TemporaryDirectory() as tempdir:
        index_path = os.path.join(tempdir, 'index.bin')

        for vector_type in ['float32', 'bfloat16', 'float16']:
            assert not os.path.exists(index_path)

            # Build an in-memory index and save it explicitly.
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(f'create virtual table my_table using vectorlite(my_embedding {vector_type}[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
            for i in range(NUM_ELEMENTS):
                cur.execute('insert into my_table (rowid, my_embedding) values (?, ?)', (i, random_vectors[i].tobytes()))

            before = cur.execute('select rowid, distance from my_table where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10)).fetchall()
            assert len(before) == 10

            cur.execute('insert into my_table(operation, path) values (?, ?)', ('save', index_path))
            assert os.path.exists(index_path) and os.path.getsize(index_path) > 0
            conn.close()

            # Load it into a brand new in-memory table without re-inserting data.
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(f'create virtual table reloaded using vectorlite(my_embedding {vector_type}[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
            # No rows yet.
            assert cur.execute('select count(*) from reloaded').fetchone()[0] == 0
            cur.execute('insert into reloaded(operation, path) values (?, ?)', ('load', index_path))

            after = cur.execute('select rowid, distance from reloaded where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10)).fetchall()
            assert after == before
            conn.close()

            os.remove(index_path)


def test_load_replaces_existing_contents(random_vectors):
    with tempfile.TemporaryDirectory() as tempdir:
        index_path = os.path.join(tempdir, 'index.bin')

        conn = get_connection()
        cur = conn.cursor()
        # Save an index containing only rowids 0..9.
        cur.execute(f'create virtual table src using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
        for i in range(10):
            cur.execute('insert into src (rowid, my_embedding) values (?, ?)', (i, random_vectors[i].tobytes()))
        cur.execute('insert into src(operation, path) values (?, ?)', ('save', index_path))

        # A different table with different rowids gets fully replaced by load.
        cur.execute(f'create virtual table dst using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
        for i in range(100, 110):
            cur.execute('insert into dst (rowid, my_embedding) values (?, ?)', (i, random_vectors[i].tobytes()))
        cur.execute('insert into dst(operation, path) values (?, ?)', ('load', index_path))

        rowids = set(r[0] for r in cur.execute('select rowid from dst where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10)).fetchall())
        assert rowids == set(range(10))
        conn.close()


def test_load_dimension_mismatch_is_rejected(random_vectors):
    with tempfile.TemporaryDirectory() as tempdir:
        index_path = os.path.join(tempdir, 'index.bin')

        conn = get_connection()
        cur = conn.cursor()
        cur.execute(f'create virtual table src using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
        for i in range(10):
            cur.execute('insert into src (rowid, my_embedding) values (?, ?)', (i, random_vectors[i].tobytes()))
        cur.execute('insert into src(operation, path) values (?, ?)', ('save', index_path))

        # Declare a table with a different dimension and insert a marker row.
        cur.execute(f'create virtual table dst using vectorlite(my_embedding float32[{DIM * 2}], hnsw(max_elements={NUM_ELEMENTS}))')
        marker = np.float32(np.random.random(DIM * 2))
        cur.execute('insert into dst (rowid, my_embedding) values (?, ?)', (999, marker.tobytes()))

        with pytest.raises(apsw.SQLError):
            cur.execute('insert into dst(operation, path) values (?, ?)', ('load', index_path))

        # Existing contents are left intact after a rejected load.
        rowids = [r[0] for r in cur.execute('select rowid from dst where knn_search(my_embedding, knn_param(?, ?))', (marker.tobytes(), 1)).fetchall()]
        assert rowids == [999]
        conn.close()


def test_load_missing_file_is_rejected():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
    with pytest.raises(apsw.SQLError):
        cur.execute('insert into t(operation, path) values (?, ?)', ('load', '/no/such/index.bin'))
    conn.close()


def test_unknown_operation_is_rejected():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
    with pytest.raises(apsw.SQLError):
        cur.execute('insert into t(operation, path) values (?, ?)', ('frobnicate', '/tmp/index.bin'))
    conn.close()


def test_three_argument_create_is_rejected():
    conn = get_connection()
    cur = conn.cursor()
    with pytest.raises(apsw.SQLError):
        cur.execute(f"create virtual table t using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}), 'index.bin')")
    conn.close()


def test_command_columns_are_hidden(random_vectors):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
    cur.execute('insert into t (rowid, my_embedding) values (?, ?)', (0, random_vectors[0].tobytes()))
    # `select *` must not surface operation/path columns.
    cur.execute('select * from t where rowid = 0')
    column_names = [d[0] for d in cur.getdescription()]
    assert 'operation' not in column_names
    assert 'path' not in column_names
    conn.close()
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest bindings/python/vectorlite_py/test/vectorlite_test.py -k "save_and_load or load_replaces or dimension_mismatch or missing_file or unknown_operation or three_argument or command_columns" -v`

Expected: FAIL. Against the current extension, `operation`/`path` are not columns (so the command INSERTs raise "no column named operation"), the 3-arg create still succeeds (so `test_three_argument_create_is_rejected` fails to raise), etc.

- [ ] **Step 3: Commit the failing test**

```bash
git add bindings/python/vectorlite_py/test/vectorlite_test.py
git commit -m "test: explicit index persistence via save/load INSERT-commands"
```

---

## Task 2: Add the hidden command columns and their enum/Column handling

**Files:**
- Modify: `vectorlite/virtual_table.cpp` (column enum ~lines 36-39; schema string ~lines 122-123; `Column` ~lines 355-359)

- [ ] **Step 1: Extend the column index enum**

Replace the enum at lines 36-39:

```cpp
enum ColumnIndexInTable {
  kColumnIndexVector,
  kColumnIndexDistance,
};
```

with:

```cpp
enum ColumnIndexInTable {
  kColumnIndexVector,
  kColumnIndexDistance,
  kColumnIndexOperation,
  kColumnIndexPath,
};
```

- [ ] **Step 2: Add the hidden columns to the declared schema**

Replace the `sql` assignment (lines 122-123):

```cpp
  std::string sql = absl::StrFormat("CREATE TABLE X(%s, distance REAL hidden)",
                                    vector_space->vector_name);
```

with:

```cpp
  std::string sql = absl::StrFormat(
      "CREATE TABLE X(%s, distance REAL hidden, operation TEXT hidden, path "
      "TEXT hidden)",
      vector_space->vector_name);
```

- [ ] **Step 3: Return NULL for the command columns in `Column`**

In `VirtualTable::Column`, replace the trailing `else` branch (lines 355-359):

```cpp
  } else {
    std::string err = absl::StrFormat("Invalid column index: %d", N);
    sqlite3_result_text(pCtx, err.c_str(), err.size(), SQLITE_TRANSIENT);
    return SQLITE_ERROR;
  }
```

with:

```cpp
  } else if (kColumnIndexOperation == N || kColumnIndexPath == N) {
    // operation/path are a write-only command channel.
    sqlite3_result_null(pCtx);
    return SQLITE_OK;
  } else {
    std::string err = absl::StrFormat("Invalid column index: %d", N);
    sqlite3_result_text(pCtx, err.c_str(), err.size(), SQLITE_TRANSIENT);
    return SQLITE_ERROR;
  }
```

- [ ] **Step 4: Build to confirm it still compiles**

Run: `cmake --build build/dev -j8`
Expected: builds successfully. (Behavior is not yet complete — command INSERTs will still hit the old insert path, which is fixed in Task 4. Do not run the full test suite yet.)

- [ ] **Step 5: Commit**

```bash
git add vectorlite/virtual_table.cpp
git commit -m "feat: declare hidden operation/path command columns"
```

---

## Task 3: Replace auto-persistence with `SaveTo`/`LoadFrom`

This task changes the `VirtualTable` constructor and removes file-path state. The codebase will not fully compile until Task 4 updates `InitVirtualTable`/`Create`/`Connect`/`Destroy`/`Disconnect` to match. Tasks 3 and 4 are therefore verified together by a single build at the end of Task 4.

**Files:**
- Modify: `vectorlite/virtual_table.h` (constructor ~lines 48-62; method declarations ~lines 64-69; member ~line 106; private section ~line 100-102; includes ~line 4)
- Modify: `vectorlite/virtual_table.cpp` (replace `LoadIndexFromFile`/`DeleteIndexFile`/`SaveIndexToFile` ~lines 153-191)

- [ ] **Step 1: Update the header — constructor, methods, member**

In `vectorlite/virtual_table.h`, replace the constructor (lines 48-62):

```cpp
  VirtualTable(NamedVectorSpace space, const IndexOptions& options,
               std::string_view file_path)
      : space_(std::move(space)),
        index_(std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_.space.get(), options.max_elements, options.M,
            options.ef_construction, options.random_seed,
            options.allow_replace_deleted)),
        file_path_() {
    VECTORLITE_ASSERT(space_.space != nullptr);
    VECTORLITE_ASSERT(index_ != nullptr);
    if (!file_path.empty()) {
      // might throw
      file_path_ = file_path;
    }
  }
```

with:

```cpp
  VirtualTable(NamedVectorSpace space, const IndexOptions& options)
      : space_(std::move(space)),
        index_(std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_.space.get(), options.max_elements, options.M,
            options.ef_construction, options.random_seed,
            options.allow_replace_deleted)) {
    VECTORLITE_ASSERT(space_.space != nullptr);
    VECTORLITE_ASSERT(index_ != nullptr);
  }
```

Replace the three persistence method declarations (lines 64-69):

```cpp
  // Load index from file_path_.
  absl::Status LoadIndexFromFile();

  absl::Status DeleteIndexFile();

  absl::Status SaveIndexToFile();
```

with:

```cpp
  // Serialize the in-memory index to `path`, overwriting any existing file.
  absl::Status SaveTo(const std::string& path);

  // Replace the in-memory index with one loaded from `path`. On any error the
  // current index is left unchanged.
  absl::Status LoadFrom(const std::string& path);
```

Remove the `file_path_` member (line 106):

```cpp
  std::filesystem::path file_path_;
```

Add a private helper declaration. Replace the private section header (lines 100-102):

```cpp
 private:
  absl::StatusOr<Vector> GetVectorByRowid(int64_t rowid) const;
  int InsertOrUpdateVector(VectorView vector, Cursor::Rowid rowid);
```

with:

```cpp
 private:
  absl::StatusOr<Vector> GetVectorByRowid(int64_t rowid) const;
  int InsertOrUpdateVector(VectorView vector, Cursor::Rowid rowid);
  // Handles an INSERT carrying a non-NULL `operation` column.
  int ExecutePersistenceCommand(sqlite3_value** argv);
```

Remove the now-unused `#include <filesystem>` (line 4 of the header):

```cpp
#include <filesystem>
```

- [ ] **Step 2: Replace the persistence implementations in the .cpp**

In `vectorlite/virtual_table.cpp`, replace the three functions `LoadIndexFromFile`, `DeleteIndexFile`, and `SaveIndexToFile` (lines 153-191) with:

```cpp
absl::Status VirtualTable::SaveTo(const std::string& path) {
  VECTORLITE_ASSERT(index_ != nullptr);
  if (path.empty()) {
    return absl::InvalidArgumentError("path must not be empty");
  }
  try {
    index_->saveIndex(path);
  } catch (const std::exception& ex) {
    return absl::InternalError(ex.what());
  }
  return absl::OkStatus();
}

absl::Status VirtualTable::LoadFrom(const std::string& path) {
  VECTORLITE_ASSERT(index_ != nullptr);
  if (path.empty()) {
    return absl::InvalidArgumentError("path must not be empty");
  }
  if (!std::filesystem::exists(path)) {
    return absl::NotFoundError(
        absl::StrFormat("index file does not exist: %s", path));
  }

  std::unique_ptr<hnswlib::HierarchicalNSW<float>> new_index;
  try {
    // This constructor loads the index from `path`; it throws on failure.
    new_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space_.space.get(), path);
  } catch (const std::exception& ex) {
    return absl::InternalError(ex.what());
  }

  // The file stores offsetData_ and label_offset_; their difference is the
  // per-vector data size recorded when the index was created. Compare it to
  // what this table's vector space expects to catch dimension/type mismatches.
  size_t file_data_size = new_index->label_offset_ - new_index->offsetData_;
  if (file_data_size != space_.space->get_data_size()) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "index data size mismatch: file has %d bytes per vector, table expects "
        "%d",
        file_data_size, space_.space->get_data_size()));
  }

  index_ = std::move(new_index);
  return absl::OkStatus();
}
```

- [ ] **Step 3: Commit (compilation completes in Task 4)**

```bash
git add vectorlite/virtual_table.h vectorlite/virtual_table.cpp
git commit -m "feat: add SaveTo/LoadFrom and drop file-path state"
```

---

## Task 4: Wire up creation (2-arg only) and command dispatch in `xUpdate`

**Files:**
- Modify: `vectorlite/virtual_table.cpp` (`InitVirtualTable` ~lines 59-151; `Create`/`Connect` ~lines 193-223; `Destroy` ~lines 205-217; `Disconnect` ~lines 219-237; `Update` ~lines 675-679; add `ExecutePersistenceCommand`)

- [ ] **Step 1: Make `InitVirtualTable` accept exactly 2 args and drop path handling**

Replace the `InitVirtualTable` signature and arg-count check (lines 59-85):

```cpp
// Shared by Create and Connect
static int InitVirtualTable(bool load_from_file, sqlite3* db, void* pAux,
                            int argc, const char* const* argv,
                            sqlite3_vtab** ppVTab, char** pzErr) {
```
...
```cpp
  constexpr int kModuleParamOffset = 3;

  if (argc != 2 + kModuleParamOffset && argc != 3 + kModuleParamOffset) {
    *pzErr = sqlite3_mprintf("vectorlite expects 2 or 3 arguments, got %d",
                             argc - kModuleParamOffset);
    return SQLITE_ERROR;
  }
```

with:

```cpp
// Shared by Create and Connect
static int InitVirtualTable(sqlite3* db, void* pAux, int argc,
                            const char* const* argv, sqlite3_vtab** ppVTab,
                            char** pzErr) {
```
...
```cpp
  constexpr int kModuleParamOffset = 3;

  if (argc != 2 + kModuleParamOffset) {
    *pzErr = sqlite3_mprintf(
        "vectorlite expects 2 arguments (a vector space and index options), "
        "got %d. The index file path argument has been removed; use INSERT "
        "INTO <table>(operation, path) VALUES('save', <path>) to persist an "
        "index and INSERT INTO <table>(operation, path) VALUES('load', <path>) "
        "to restore one.",
        argc - kModuleParamOffset);
    return SQLITE_ERROR;
  }
```

- [ ] **Step 2: Remove the index-file-path parsing block**

Delete the entire `index_file_path` block (lines 107-120):

```cpp
  std::string_view index_file_path;
  if (argc == 3 + kModuleParamOffset) {
    index_file_path = argv[2 + kModuleParamOffset];
    int size = index_file_path.size();
    // Handle cases where the index_file_path is enclosed in double/single
    // quotes. It is necessary for windows paths, because they contain ':', that
    // must be quoted for sqlite to parse correctly.
    if (size > 2) {
      if ((index_file_path[0] == '\"' && index_file_path[size - 1] == '\"') ||
          (index_file_path[0] == '\'' && index_file_path[size - 1] == '\'')) {
        index_file_path = index_file_path.substr(1, size - 2);
      }
    }
  }
```

(Leave the `sqlite3_declare_vtab` call that follows it in place — it now declares the schema with the hidden command columns from Task 2.)

- [ ] **Step 3: Construct the table with two args and drop the load-from-file block**

Replace the construction + load block (lines 130-149):

```cpp
  try {
    auto vtab = new VirtualTable(std::move(*vector_space), *index_options,
                                 index_file_path);
    *ppVTab = vtab;

    if (load_from_file) {
      auto status = vtab->LoadIndexFromFile();
      if (!status.ok()) {
        *pzErr = sqlite3_mprintf("Failed to load index from file: %s",
                                 absl::StatusMessageAsCStr(status));
        delete vtab;
        *ppVTab = nullptr;
        return SQLITE_ERROR;
      }
    }

  } catch (const std::exception& ex) {
    *pzErr = sqlite3_mprintf("Failed to create virtual table: %s", ex.what());
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}
```

with:

```cpp
  try {
    auto vtab = new VirtualTable(std::move(*vector_space), *index_options);
    *ppVTab = vtab;
  } catch (const std::exception& ex) {
    *pzErr = sqlite3_mprintf("Failed to create virtual table: %s", ex.what());
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}
```

- [ ] **Step 4: Update `Create` and `Connect` to the new signature**

Replace `Create` (lines 193-197) and `Connect` (lines 219-223):

```cpp
int VirtualTable::Create(sqlite3* db, void* pAux, int argc,
                         const char* const* argv, sqlite3_vtab** ppVTab,
                         char** pzErr) {
  return InitVirtualTable(true, db, pAux, argc, argv, ppVTab, pzErr);
}
```
...
```cpp
int VirtualTable::Connect(sqlite3* db, void* pAux, int argc,
                          const char* const* argv, sqlite3_vtab** ppVTab,
                          char** pzErr) {
  return InitVirtualTable(true, db, pAux, argc, argv, ppVTab, pzErr);
}
```

with:

```cpp
int VirtualTable::Create(sqlite3* db, void* pAux, int argc,
                         const char* const* argv, sqlite3_vtab** ppVTab,
                         char** pzErr) {
  return InitVirtualTable(db, pAux, argc, argv, ppVTab, pzErr);
}
```
...
```cpp
int VirtualTable::Connect(sqlite3* db, void* pAux, int argc,
                          const char* const* argv, sqlite3_vtab** ppVTab,
                          char** pzErr) {
  return InitVirtualTable(db, pAux, argc, argv, ppVTab, pzErr);
}
```

- [ ] **Step 5: Drop file deletion from `Destroy`**

Replace `Destroy` (lines 205-217):

```cpp
int VirtualTable::Destroy(sqlite3_vtab* pVTab) {
  DLOG(INFO) << "Destroy called";
  VECTORLITE_ASSERT(pVTab != nullptr);
  VirtualTable* vtab = static_cast<VirtualTable*>(pVTab);
  auto status = vtab->DeleteIndexFile();
  if (!status.ok()) {
    SetZErrMsg(&vtab->zErrMsg, "Failed to delete index file: %s",
               absl::StatusMessageAsCStr(status));
    return SQLITE_ERROR;
  }
  delete vtab;
  return SQLITE_OK;
}
```

with:

```cpp
int VirtualTable::Destroy(sqlite3_vtab* pVTab) {
  DLOG(INFO) << "Destroy called";
  VECTORLITE_ASSERT(pVTab != nullptr);
  delete static_cast<VirtualTable*>(pVTab);
  return SQLITE_OK;
}
```

- [ ] **Step 6: Drop auto-save from `Disconnect`**

Replace `Disconnect` (lines 225-237):

```cpp
int VirtualTable::Disconnect(sqlite3_vtab* pVTab) {
  DLOG(INFO) << "Disconnect called";
  VECTORLITE_ASSERT(pVTab != nullptr);
  VirtualTable* vtab = static_cast<VirtualTable*>(pVTab);
  auto status = vtab->SaveIndexToFile();
  if (!status.ok()) {
    SetZErrMsg(&vtab->zErrMsg, "Failed to save index to file: %s",
               absl::StatusMessageAsCStr(status));
    return SQLITE_ERROR;
  }
  delete vtab;
  return SQLITE_OK;
}
```

with:

```cpp
int VirtualTable::Disconnect(sqlite3_vtab* pVTab) {
  DLOG(INFO) << "Disconnect called";
  VECTORLITE_ASSERT(pVTab != nullptr);
  delete static_cast<VirtualTable*>(pVTab);
  return SQLITE_OK;
}
```

- [ ] **Step 7: Add the `ExecutePersistenceCommand` helper**

Add this function immediately before `int VirtualTable::Update(` (i.e. just before line 675):

```cpp
int VirtualTable::ExecutePersistenceCommand(sqlite3_value** argv) {
  sqlite3_value* op_value = argv[2 + kColumnIndexOperation];
  std::string operation(
      reinterpret_cast<const char*>(sqlite3_value_text(op_value)),
      sqlite3_value_bytes(op_value));

  sqlite3_value* path_value = argv[2 + kColumnIndexPath];
  if (sqlite3_value_type(path_value) != SQLITE_TEXT) {
    SetZErrMsg(&zErrMsg, "path must be provided as TEXT for '%s' operation",
               operation.c_str());
    return SQLITE_ERROR;
  }
  std::string path(
      reinterpret_cast<const char*>(sqlite3_value_text(path_value)),
      sqlite3_value_bytes(path_value));

  absl::Status status;
  if (operation == "save") {
    status = SaveTo(path);
  } else if (operation == "load") {
    status = LoadFrom(path);
  } else {
    SetZErrMsg(&zErrMsg, "unknown operation '%s'; expected 'save' or 'load'",
               operation.c_str());
    return SQLITE_ERROR;
  }

  if (!status.ok()) {
    SetZErrMsg(&zErrMsg, "%s failed: %s", operation.c_str(),
               absl::StatusMessageAsCStr(status));
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}
```

- [ ] **Step 8: Dispatch persistence commands from `xUpdate`**

In `VirtualTable::Update`, replace the start of the INSERT branch (lines 675-684):

```cpp
int VirtualTable::Update(sqlite3_vtab* pVTab, int argc, sqlite3_value** argv,
                         sqlite_int64* pRowid) {
  VirtualTable* vtab = static_cast<VirtualTable*>(pVTab);
  auto argv0_type = sqlite3_value_type(argv[0]);
  if (argc > 1 && argv0_type == SQLITE_NULL) {
    // Insert with a new row
    if (sqlite3_value_type(argv[1]) == SQLITE_NULL) {
      SetZErrMsg(&vtab->zErrMsg, "rowid must be specified during insertion");
      return SQLITE_ERROR;
    }
```

with:

```cpp
int VirtualTable::Update(sqlite3_vtab* pVTab, int argc, sqlite3_value** argv,
                         sqlite_int64* pRowid) {
  VirtualTable* vtab = static_cast<VirtualTable*>(pVTab);
  auto argv0_type = sqlite3_value_type(argv[0]);
  if (argc > 1 && argv0_type == SQLITE_NULL) {
    // An INSERT carrying a non-NULL `operation` column is a persistence
    // command (save/load), not a vector insert.
    if (sqlite3_value_type(argv[2 + kColumnIndexOperation]) == SQLITE_TEXT) {
      *pRowid = 0;
      return vtab->ExecutePersistenceCommand(argv);
    }
    // Insert with a new row
    if (sqlite3_value_type(argv[1]) == SQLITE_NULL) {
      SetZErrMsg(&vtab->zErrMsg, "rowid must be specified during insertion");
      return SQLITE_ERROR;
    }
```

- [ ] **Step 9: Build and run the Task 1 integration tests**

Run: `cmake --build build/dev -j8 && pytest bindings/python/vectorlite_py/test/vectorlite_test.py -k "save_and_load or load_replaces or dimension_mismatch or missing_file or unknown_operation or three_argument or command_columns" -v`

Expected: builds successfully; all seven tests PASS.

- [ ] **Step 10: Run the full test suite to catch regressions**

Run: `sh build.sh`
Expected: C++ `ctest` passes and all Python tests pass.

- [ ] **Step 11: Commit**

```bash
git add vectorlite/virtual_table.cpp
git commit -m "feat: explicit save/load persistence via INSERT-commands"
```

---

## Task 5: Update example and documentation

**Files:**
- Modify: `examples/index_serde.py`
- Modify: `README.md` (lines 78-90)
- Modify: `doc/markdown/api.md` (lines 30-40), and scan `doc/markdown/getting-started.md`, `doc/markdown/overview.md` for path-argument references
- Modify: `bindings/nodejs/packages/vectorlite/README.md` and `bindings/nodejs/packages/vectorlite/test/test.js` if they reference the 3rd path argument

- [ ] **Step 1: Rewrite `examples/index_serde.py` to use save/load commands**

Replace the table-creation comment block and statement (the comment lines explaining "the 3rd argument is an optional index file path ..." plus the `create virtual table x ... {index_file_path})` line) so the table is created without a path:

```python
print('Trying to create virtual table for vector search.')
# Creates an in-memory virtual table 'x' with a 1000-dim float32 vector column.
# The index lives in memory. Persist it explicitly with:
#   insert into x(operation, path) values ('save', '<path>')
# and restore it into another in-memory table with:
#   insert into x(operation, path) values ('load', '<path>')
cur.execute(f'create virtual table x using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
```

Replace the explicit-save point. After the vector inserts and queries, before `conn.close()`, add:

```python
# Persist the in-memory index to disk explicitly.
cur.execute('insert into x(operation, path) values (?, ?)', ('save', index_file_path))
```

Replace the reload block. Change the `create virtual table table_reloaded ... {index_file_path})` line and the comment above it to:

```python
# Create a fresh in-memory table, then load the previously saved index into it.
# The vector dimension MUST match the saved index. Table name and vector name
# may differ. Distance type may differ.
cur.execute(f'create virtual table table_reloaded using vectorlite(vec_reloaded float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS * 2}))')
cur.execute('insert into table_reloaded(operation, path) values (?, ?)', ('load', index_file_path))
print(f'index is loaded from {index_file_path}')
```

Update the post-`conn.close()` assertion comment (the line `# If database connection is closed, the index will be saved to the index file.`) to:

```python
# The index file was written by the explicit 'save' command above.
```

Remove the final `drop table table_reloaded` comment claiming the file is deleted on drop; dropping a table no longer deletes any file. Replace:

```python
# index file will be deleted when the table is dropped.
cur.execute('drop table table_reloaded')
```

with:

```python
# Dropping a table only frees the in-memory index; it never deletes files.
cur.execute('drop table table_reloaded')
os.remove(index_file_path)
```

- [ ] **Step 2: Run the example to verify it works end-to-end**

Run: `python examples/index_serde.py`
Expected: runs without error, prints recall rates and nearest-neighbor output, and exits 0.

- [ ] **Step 3: Update `README.md`**

Replace the create-table documentation block (lines 82-89):

```
-- Optional fields:
-- 1. distance_type: defaults to l2
-- 2. ef_construction: defaults to 200
-- 3. M: defaults to 16
-- 4. random_seed: defaults to 100
-- 5. allow_replace_deleted: defaults to true
-- 6. index_file_path: no default value. If not provided, the table will be memory-only. If provided, vectorlite will try to load index from the file and save to it when db connection is closed.
create virtual table {table_name} using vectorlite({vector_name} float32[{dimension}] {distance_type}, hnsw(max_elements={max_elements}, {ef_construction=200}, {M=16}, {random_seed=100}, {allow_replace_deleted=true}), {index_file_path});
```

with:

```
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

Immediately after the closing ``` of that block, add a new persistence example:

````
Persist an index to disk, or restore a saved index into an in-memory table:
```sql
-- Save the current in-memory index to a file (overwrites if it exists).
insert into {table_name}(operation, path) values ('save', '/path/to/index.bin');
-- Load a saved index into a freshly created table with a matching vector dimension.
insert into {table_name}(operation, path) values ('load', '/path/to/index.bin');
```
Note: `operation`, `path`, and `distance` are reserved column names and cannot be
used as the vector column name. The in-memory index is lost on connection close
unless you explicitly save it.
````

- [ ] **Step 4: Update `doc/markdown/api.md`**

Apply the same replacement as Step 3 to the matching block in `doc/markdown/api.md` (lines 30-37: the `index_file_path` comment line and the `create virtual table ... {index_file_path});` line), removing the `index_file_path` comment and trailing argument, and add the same save/load SQL example block beneath it.

- [ ] **Step 5: Scan the remaining docs for stale path references**

Run: `grep -rn "index_file\|index file\|3rd argument\|third argument\|saved to the file\|deleted if the table is dropped" README.md doc/markdown/ bindings/nodejs/packages/vectorlite/README.md`
Expected: no remaining references describing the removed path argument or auto-persistence. Fix any that remain by describing the explicit save/load commands instead. If `bindings/nodejs/packages/vectorlite/test/test.js` creates a table with a 3rd path argument, update it to create a path-less table and persist via `insert into <table>(operation, path) values('save'|'load', ...)`.

- [ ] **Step 6: Commit**

```bash
git add examples/index_serde.py README.md doc/markdown/ bindings/nodejs/packages/vectorlite/README.md bindings/nodejs/packages/vectorlite/test/test.js
git commit -m "docs: document explicit save/load index persistence"
```

---

## Task 6: Bump the major version

A clean break in the table-creation contract warrants a major version bump.

**Files:**
- Inspect/modify: the project version source. Find it first.

- [ ] **Step 1: Locate the version definition**

Run: `grep -rn "version" CMakeLists.txt vectorlite/version.h.in pyproject.toml bindings/nodejs/packages/vectorlite/package.json 2>/dev/null | grep -i "version" | head -20`
Expected: identifies where the version string lives (e.g. `version.h.in`, `pyproject.toml`, npm `package.json`).

- [ ] **Step 2: Bump the major version**

Edit each location found in Step 1 to increment the major version (e.g. `0.x.y` → `1.0.0` or, if already `>=1`, `N.x.y` → `(N+1).0.0`), keeping all locations in sync.

- [ ] **Step 3: Verify the version is consistent**

Run: `grep -rn "<new-version-number>" CMakeLists.txt vectorlite/version.h.in pyproject.toml bindings/nodejs/packages/vectorlite/package.json 2>/dev/null`
Expected: every version location reflects the new number.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: bump major version for explicit-persistence breaking change"
```

---

## Final verification

- [ ] **Run the full build + test suite**

Run: `sh build.sh`
Expected: C++ unit tests pass; all Python integration tests (including the seven new persistence tests) pass.

- [ ] **Run the example**

Run: `python examples/index_serde.py`
Expected: exits 0 with expected output.
