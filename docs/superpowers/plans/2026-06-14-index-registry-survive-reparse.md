# Connection-Scoped Index Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a vectorlite table's in-memory HNSW index survive SQLite schema reparses (VACUUM, ALTER, foreign DDL) within a connection, instead of being silently wiped.

**Architecture:** Move the stateful core (vector space + hnswlib index + flag) out of the short-lived `VirtualTable` object into a per-connection `IndexRegistry` owned by the module's `pAux` and freed by a `sqlite3_create_module_v2` destructor on connection close. `xConnect` reattaches to the existing registry entry instead of building an empty index; `xDisconnect` leaves the entry; `xDestroy` removes it.

**Tech Stack:** C++17, SQLite virtual-table API, hnswlib 0.8.0, abseil, GoogleTest, pytest + apsw.

---

## Background the engineer needs

- Core files: `vectorlite/virtual_table.h`, `vectorlite/virtual_table.cpp`, `vectorlite/vectorlite.cpp` (module registration). New files: `vectorlite/index_registry.h`, `vectorlite/index_registry.cpp`, `vectorlite/index_registry_test.cpp`.
- The design spec is `docs/superpowers/specs/2026-06-14-index-registry-survive-reparse-design.md`. Read it if any step is unclear.
- Build + test from repo root: `sh build.sh` (configures `build/dev`, builds, runs `ctest`, runs pytest). To iterate: `cmake --build build/dev -j8`.
- **Python tests require the repo venv:** `cd /Volumes/external/vectorlite && . .venv/bin/activate` before running `pytest` (apsw/numpy/pytest live there; system python lacks them). The build's POST_BUILD step copies the freshly built `vectorlite` dylib into `bindings/python/vectorlite_py/`, so pytest always uses the latest build.
- C++ unit tests: `ctest --test-dir build/dev/vectorlite --output-on-failure`. The `unit_test` target is built from `file(GLOB *.cpp)` in `vectorlite/CMakeLists.txt`, so new `*.cpp`/`*_test.cpp` files are picked up automatically. The **shared library** target lists its sources explicitly and must be edited to add `index_registry.cpp`.

### SQLite vtab facts relevant here

- `xConnect` is called on open AND on every schema reparse; `xDisconnect` tears the object down each time. `xCreate` runs once at `CREATE VIRTUAL TABLE`; `xDestroy` runs once at `DROP TABLE`.
- Callback `argv` layout (in `InitVirtualTable`): `argv[1]` = schema name (`main`/`temp`/attached), `argv[2]` = table name, `argv[3]` = vector-space string, `argv[4]` = index-options string. `kModuleParamOffset` is 3, so `argv[3]` = `argv[0 + kModuleParamOffset]`, `argv[4]` = `argv[1 + kModuleParamOffset]`.
- The module's `pAux` (4th arg to `sqlite3_create_module*`) is delivered to `xCreate`/`xConnect` as their `pAux` parameter.

### hnswlib lifetime trap (critical)

The hnswlib index caches a pointer into its space object (`dist_func_param_ = s->get_dist_func_param()`, which for L2/IP/Cosine returns `&dim_` inside the `SpaceInterface`). Therefore the index and its `NamedVectorSpace` must live together; an index must never be paired with a different/rebuilt space. Moving a `NamedVectorSpace` or an `IndexHandle` is safe because the `SpaceInterface` is held by `unique_ptr` — the move transfers the pointer, the pointee keeps its address. Storing handles as `std::unique_ptr<IndexHandle>` in a map keeps each handle's address stable across map mutations.

---

## Task 1: Failing Python regression tests

These encode the target behavior and fail against the current (reparse-wiping) extension.

**Files:**
- Modify: `bindings/python/vectorlite_py/test/vectorlite_test.py` (append new test functions at end of file)

- [ ] **Step 1: Append regression tests**

Add these functions at the end of `bindings/python/vectorlite_py/test/vectorlite_test.py`:

```python
def _make_table_and_fill(cur, name='surv', n=20):
    cur.execute(f'create virtual table {name} using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
    vectors = np.float32(np.random.random((n, DIM)))
    for i in range(n):
        cur.execute(f'insert into {name} (rowid, my_embedding) values (?, ?)', (i, vectors[i].tobytes()))
    return vectors


def _count(cur, name, query_vec, k=20):
    return len(cur.execute(f'select rowid from {name} where knn_search(my_embedding, knn_param(?, ?))', (query_vec.tobytes(), k)).fetchall())


def test_index_survives_vacuum():
    conn = get_connection()
    cur = conn.cursor()
    vectors = _make_table_and_fill(cur)
    assert _count(cur, 'surv', vectors[0]) == 20
    cur.execute('vacuum')
    assert _count(cur, 'surv', vectors[0]) == 20
    conn.close()


def test_index_survives_alter_table_on_other_table():
    conn = get_connection()
    cur = conn.cursor()
    vectors = _make_table_and_fill(cur)
    cur.execute('create table other(a)')
    assert _count(cur, 'surv', vectors[0]) == 20
    cur.execute('alter table other add column b')
    assert _count(cur, 'surv', vectors[0]) == 20
    conn.close()


def test_index_survives_foreign_connection_ddl():
    with tempfile.TemporaryDirectory() as tempdir:
        db_path = os.path.join(tempdir, 'shared.db')

        def open_conn():
            c = apsw.Connection(db_path)
            c.enable_load_extension(True)
            c.load_extension(vectorlite_py.vectorlite_path())
            return c

        conn_a = open_conn()
        cur_a = conn_a.cursor()
        vectors = _make_table_and_fill(cur_a)
        assert _count(cur_a, 'surv', vectors[0]) == 20

        # Another connection performs DDL, bumping the schema version.
        conn_b = open_conn()
        conn_b.cursor().execute('create table unrelated(a)')
        conn_b.close()

        # Connection A reparses on its next statement; the index must survive.
        assert _count(cur_a, 'surv', vectors[0]) == 20
        conn_a.close()


def test_name_reuse_with_different_shape_is_clean():
    conn = get_connection()
    cur = conn.cursor()
    # Create a dim-4 table, fill it, drop it.
    cur.execute('create virtual table reuse using vectorlite(emb float32[4], hnsw(max_elements=100))')
    for i in range(5):
        cur.execute('insert into reuse (rowid, emb) values (?, ?)', (i, np.float32(np.random.random(4)).tobytes()))
    cur.execute('drop table reuse')

    # Recreate with the SAME name but a different dimension.
    cur.execute('create virtual table reuse using vectorlite(emb float32[8], hnsw(max_elements=100))')
    # The new table is empty (no stale dim-4 data) and accepts dim-8 vectors.
    assert cur.execute('select rowid from reuse where knn_search(emb, knn_param(?, ?))', (np.float32(np.random.random(8)).tobytes(), 10)).fetchall() == []
    cur.execute('insert into reuse (rowid, emb) values (?, ?)', (0, np.float32(np.random.random(8)).tobytes()))
    assert len(cur.execute('select rowid from reuse where knn_search(emb, knn_param(?, ?))', (np.float32(np.random.random(8)).tobytes(), 10)).fetchall()) == 1
    conn.close()
```

- [ ] **Step 2: Run the new tests to verify they fail (red)**

Run:
```bash
cd /Volumes/external/vectorlite && . .venv/bin/activate && pytest bindings/python/vectorlite_py/test/vectorlite_test.py -k "survives or name_reuse" -v
```
Expected: `test_index_survives_vacuum`, `test_index_survives_alter_table_on_other_table`, and `test_index_survives_foreign_connection_ddl` FAIL (count is 0 after the reparse). `test_name_reuse_with_different_shape_is_clean` should already PASS (DROP currently frees the index), which is fine — it guards against regressions in the new code.

- [ ] **Step 3: Confirm collection still works**

Run: `pytest bindings/python/vectorlite_py/test/vectorlite_test.py --collect-only -q`
Expected: collects with no syntax/import errors.

- [ ] **Step 4: Commit**

```bash
git add bindings/python/vectorlite_py/test/vectorlite_test.py
git commit -m "test: index must survive schema reparse (VACUUM/ALTER/foreign DDL)"
```

---

## Task 2: Add IndexHandle and IndexRegistry

A self-contained, independently-compilable data structure plus its unit test.

**Files:**
- Create: `vectorlite/index_registry.h`
- Create: `vectorlite/index_registry.cpp`
- Create: `vectorlite/index_registry_test.cpp`
- Modify: `vectorlite/CMakeLists.txt` (add `index_registry.cpp` to the `vectorlite` shared library sources)

- [ ] **Step 1: Write the failing unit test**

Create `vectorlite/index_registry_test.cpp`:

```cpp
#include "index_registry.h"

#include <memory>
#include <utility>

#include "gtest/gtest.h"
#include "hnswlib/hnswlib.h"
#include "index_options.h"
#include "vector_space.h"

namespace vectorlite {
namespace {

IndexHandle MakeTestHandle(std::string_view space_str = "emb float32[4]",
                           std::string_view options_str = "hnsw(max_elements=100)") {
  auto space = NamedVectorSpace::FromString(space_str);
  EXPECT_TRUE(space.ok());
  auto options = IndexOptions::FromString(options_str);
  EXPECT_TRUE(options.ok());
  auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      space->space.get(), options->max_elements, options->M,
      options->ef_construction, options->random_seed,
      options->allow_replace_deleted);
  return IndexHandle{std::move(*space), std::move(index),
                     options->allow_replace_deleted, std::string(space_str),
                     std::string(options_str)};
}

TEST(IndexRegistry, FindReturnsNullForMissingKey) {
  IndexRegistry registry;
  EXPECT_EQ(registry.Find({"main", "absent"}), nullptr);
}

TEST(IndexRegistry, InsertThenFindReturnsSameHandle) {
  IndexRegistry registry;
  IndexHandle* inserted = registry.Insert({"main", "t"}, MakeTestHandle());
  ASSERT_NE(inserted, nullptr);
  EXPECT_EQ(registry.Find({"main", "t"}), inserted);
}

TEST(IndexRegistry, InsertReplacesExistingEntry) {
  IndexRegistry registry;
  IndexHandle* first = registry.Insert({"main", "t"}, MakeTestHandle());
  IndexHandle* second = registry.Insert({"main", "t"}, MakeTestHandle());
  EXPECT_NE(first, second);
  EXPECT_EQ(registry.Find({"main", "t"}), second);
}

TEST(IndexRegistry, EraseRemovesEntry) {
  IndexRegistry registry;
  registry.Insert({"main", "t"}, MakeTestHandle());
  registry.Erase({"main", "t"});
  EXPECT_EQ(registry.Find({"main", "t"}), nullptr);
}

TEST(IndexRegistry, KeysWithDifferentSchemasAreDistinct) {
  IndexRegistry registry;
  IndexHandle* main_handle = registry.Insert({"main", "t"}, MakeTestHandle());
  IndexHandle* temp_handle = registry.Insert({"temp", "t"}, MakeTestHandle());
  EXPECT_NE(main_handle, temp_handle);
  EXPECT_EQ(registry.Find({"main", "t"}), main_handle);
  EXPECT_EQ(registry.Find({"temp", "t"}), temp_handle);
}

}  // namespace
}  // namespace vectorlite
```

- [ ] **Step 2: Write the header**

Create `vectorlite/index_registry.h`:

```cpp
#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "hnswlib/hnswlib.h"
#include "vector_space.h"

namespace vectorlite {

// The stateful core of a vectorlite table. Owned by an IndexRegistry so that it
// outlives the short-lived VirtualTable object across schema reparses. The
// space and index are kept together because the hnswlib index caches a pointer
// into the space (see design doc).
struct IndexHandle {
  NamedVectorSpace space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
  // Runtime-only hnswlib flag (not serialized); retained to reapply on load.
  bool allow_replace_deleted;
  // The exact module-argument strings that defined this table. Used to detect a
  // table-name collision on xConnect.
  std::string vector_space_str;
  std::string index_options_str;
};

// (schema_name, table_name) uniquely identifies a table within a connection.
using RegistryKey = std::pair<std::string, std::string>;

// A per-connection map of live in-memory indexes. Not thread-safe; SQLite
// serializes access to a single connection.
class IndexRegistry {
 public:
  // Returns the handle for `key`, or nullptr if absent.
  IndexHandle* Find(const RegistryKey& key);

  // Stores `handle` under `key`, replacing any existing entry, and returns a
  // stable pointer to the stored handle.
  IndexHandle* Insert(const RegistryKey& key, IndexHandle handle);

  // Removes the entry for `key` if present.
  void Erase(const RegistryKey& key);

 private:
  std::map<RegistryKey, std::unique_ptr<IndexHandle>> handles_;
};

}  // namespace vectorlite
```

- [ ] **Step 3: Write the implementation**

Create `vectorlite/index_registry.cpp`:

```cpp
#include "index_registry.h"

#include <memory>
#include <utility>

namespace vectorlite {

IndexHandle* IndexRegistry::Find(const RegistryKey& key) {
  auto it = handles_.find(key);
  return it == handles_.end() ? nullptr : it->second.get();
}

IndexHandle* IndexRegistry::Insert(const RegistryKey& key, IndexHandle handle) {
  auto stored = std::make_unique<IndexHandle>(std::move(handle));
  IndexHandle* ptr = stored.get();
  handles_[key] = std::move(stored);
  return ptr;
}

void IndexRegistry::Erase(const RegistryKey& key) { handles_.erase(key); }

}  // namespace vectorlite
```

- [ ] **Step 4: Add the source to the shared library target**

In `vectorlite/CMakeLists.txt`, find the `add_library(vectorlite SHARED ...)` line:

```cmake
add_library(vectorlite SHARED vectorlite.cpp virtual_table.cpp util.cpp vector_space.cpp index_options.cpp sqlite_functions.cpp constraint.cpp quantization.cpp)
```

Add `index_registry.cpp` to it:

```cmake
add_library(vectorlite SHARED vectorlite.cpp virtual_table.cpp util.cpp vector_space.cpp index_options.cpp sqlite_functions.cpp constraint.cpp quantization.cpp index_registry.cpp)
```

(The `unit_test` target uses `file(GLOB *.cpp)`, so `index_registry.cpp` and `index_registry_test.cpp` are picked up automatically — no change needed there.)

- [ ] **Step 5: Reconfigure (new files added), build, and run the registry tests**

Run:
```bash
cd /Volumes/external/vectorlite && cmake --preset dev >/dev/null && cmake --build build/dev -j8 && ctest --test-dir build/dev/vectorlite -R IndexRegistry --output-on-failure
```
Expected: builds; all 5 `IndexRegistry.*` tests PASS.

- [ ] **Step 6: Commit**

```bash
git add vectorlite/index_registry.h vectorlite/index_registry.cpp vectorlite/index_registry_test.cpp vectorlite/CMakeLists.txt
git commit -m "feat: add per-connection IndexRegistry and IndexHandle"
```

---

## Task 3: Move index ownership into the registry and reattach on reparse

This is the core change. It rewires `VirtualTable` to reference a registry-owned handle, makes `xConnect` reattach, and registers the module with a `pAux` registry + destructor. The Task 1 regression tests pass at the end.

**Files:**
- Modify: `vectorlite/virtual_table.h`
- Modify: `vectorlite/virtual_table.cpp`
- Modify: `vectorlite/vectorlite.cpp`

- [ ] **Step 1: Update the VirtualTable class declaration**

In `vectorlite/virtual_table.h`, add the include near the other project includes (after `#include "index_options.h"`):

```cpp
#include "index_registry.h"
```

Replace the constructor (lines 47-56):

```cpp
  VirtualTable(NamedVectorSpace space, const IndexOptions& options)
      : space_(std::move(space)),
        index_(std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_.space.get(), options.max_elements, options.M,
            options.ef_construction, options.random_seed,
            options.allow_replace_deleted)),
        allow_replace_deleted_(options.allow_replace_deleted) {
    VECTORLITE_ASSERT(space_.space != nullptr);
    VECTORLITE_ASSERT(index_ != nullptr);
  }
```

with a constructor that binds to a registry-owned handle:

```cpp
  // `registry` and `handle` are owned by the connection's IndexRegistry, not by
  // this object. `handle` must already live in `registry` under `key`.
  VirtualTable(IndexRegistry* registry, RegistryKey key, IndexHandle* handle)
      : registry_(registry),
        key_(std::move(key)),
        handle_(handle),
        space_(handle->space),
        index_(handle->index),
        allow_replace_deleted_(handle->allow_replace_deleted) {
    VECTORLITE_ASSERT(registry_ != nullptr);
    VECTORLITE_ASSERT(handle_ != nullptr);
    VECTORLITE_ASSERT(space_.space != nullptr);
    VECTORLITE_ASSERT(index_ != nullptr);
  }
```

Replace the private member block (lines 100-104):

```cpp
  NamedVectorSpace space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
  // allow_replace_deleted is a runtime-only hnswlib flag that is not stored in
  // the serialized index, so it is retained here to reapply it after LoadFrom.
  bool allow_replace_deleted_;
```

with reference members bound to the registry-owned handle:

```cpp
  IndexRegistry* registry_;  // not owned
  RegistryKey key_;          // this table's (schema, name)
  IndexHandle* handle_;      // owned by registry_
  // References into *handle_, so existing member-access call sites are
  // unchanged. The handle has a stable address (owned via unique_ptr in the
  // registry map), so these references stay valid until the entry is erased.
  NamedVectorSpace& space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>>& index_;
  bool& allow_replace_deleted_;
```

Note: member initialization order follows declaration order, so `registry_`, `key_`, `handle_` are declared before the references that depend on `handle_`. Keep them in that order.

- [ ] **Step 2: Add a handle factory and rewire `InitVirtualTable` in the .cpp**

In `vectorlite/virtual_table.cpp`, add this static factory function just above `InitVirtualTable` (before the `// Shared by Create and Connect` comment at line 60):

```cpp
// Builds an IndexHandle (vector space + fresh empty index) from parsed args.
// Might throw (hnswlib allocation).
static IndexHandle MakeIndexHandle(NamedVectorSpace space,
                                   const IndexOptions& options,
                                   std::string_view vector_space_str,
                                   std::string_view index_options_str) {
  auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      space.space.get(), options.max_elements, options.M,
      options.ef_construction, options.random_seed,
      options.allow_replace_deleted);
  return IndexHandle{std::move(space), std::move(index),
                     options.allow_replace_deleted,
                     std::string(vector_space_str),
                     std::string(index_options_str)};
}
```

Change the `InitVirtualTable` signature (lines 60-63) to add an `is_create` flag:

```cpp
// Shared by Create and Connect
static int InitVirtualTable(sqlite3* db, void* pAux, int argc,
                            const char* const* argv, sqlite3_vtab** ppVTab,
                            char** pzErr) {
```

becomes:

```cpp
// Shared by Create and Connect
static int InitVirtualTable(bool is_create, sqlite3* db, void* pAux, int argc,
                            const char* const* argv, sqlite3_vtab** ppVTab,
                            char** pzErr) {
```

Then replace the construction block (lines 124-131):

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

with registry-aware find-or-create logic:

```cpp
  IndexRegistry* registry = static_cast<IndexRegistry*>(pAux);
  VECTORLITE_ASSERT(registry != nullptr);
  RegistryKey key{argv[1], argv[2]};

  IndexHandle* handle = nullptr;
  if (!is_create) {
    // On connect/reparse, reuse the existing in-memory index, but only if it
    // matches the current table definition. A stale entry left by a different
    // table that reused this name will not match and is rebuilt instead.
    handle = registry->Find(key);
    if (handle != nullptr &&
        (handle->vector_space_str != vector_space_str ||
         handle->index_options_str != index_options_str)) {
      handle = nullptr;
    }
  }

  if (handle == nullptr) {
    // xCreate always builds fresh (replacing any stale entry); xConnect builds
    // fresh only when no matching entry exists.
    try {
      handle = registry->Insert(
          key, MakeIndexHandle(std::move(*vector_space), *index_options,
                               vector_space_str, index_options_str));
    } catch (const std::exception& ex) {
      *pzErr = sqlite3_mprintf("Failed to create virtual table: %s", ex.what());
      return SQLITE_ERROR;
    }
  }

  auto vtab = new VirtualTable(registry, key, handle);
  *ppVTab = vtab;
  return SQLITE_OK;
}
```

(`vector_space_str` and `index_options_str` are the `std::string_view`s already declared earlier in `InitVirtualTable` at lines 94 and 104; they remain valid here.)

- [ ] **Step 3: Update `Create`, `Connect`, and `Destroy`**

Replace `VirtualTable::Create` (lines 187-191):

```cpp
int VirtualTable::Create(sqlite3* db, void* pAux, int argc,
                         const char* const* argv, sqlite3_vtab** ppVTab,
                         char** pzErr) {
  return InitVirtualTable(db, pAux, argc, argv, ppVTab, pzErr);
}
```

with:

```cpp
int VirtualTable::Create(sqlite3* db, void* pAux, int argc,
                         const char* const* argv, sqlite3_vtab** ppVTab,
                         char** pzErr) {
  return InitVirtualTable(/*is_create=*/true, db, pAux, argc, argv, ppVTab,
                          pzErr);
}
```

Replace `VirtualTable::Connect` (lines 206-210):

```cpp
int VirtualTable::Connect(sqlite3* db, void* pAux, int argc,
                          const char* const* argv, sqlite3_vtab** ppVTab,
                          char** pzErr) {
  return InitVirtualTable(db, pAux, argc, argv, ppVTab, pzErr);
}
```

with:

```cpp
int VirtualTable::Connect(sqlite3* db, void* pAux, int argc,
                          const char* const* argv, sqlite3_vtab** ppVTab,
                          char** pzErr) {
  return InitVirtualTable(/*is_create=*/false, db, pAux, argc, argv, ppVTab,
                          pzErr);
}
```

Replace `VirtualTable::Destroy` (lines 199-204):

```cpp
int VirtualTable::Destroy(sqlite3_vtab* pVTab) {
  DLOG(INFO) << "Destroy called";
  VECTORLITE_ASSERT(pVTab != nullptr);
  delete static_cast<VirtualTable*>(pVTab);
  return SQLITE_OK;
}
```

with a version that removes the registry entry (DROP TABLE):

```cpp
int VirtualTable::Destroy(sqlite3_vtab* pVTab) {
  DLOG(INFO) << "Destroy called";
  VECTORLITE_ASSERT(pVTab != nullptr);
  VirtualTable* vtab = static_cast<VirtualTable*>(pVTab);
  vtab->registry_->Erase(vtab->key_);
  delete vtab;
  return SQLITE_OK;
}
```

(`Disconnect` is left unchanged: it deletes only the `VirtualTable` wrapper and intentionally leaves the registry entry alive so the index survives the reparse.)

- [ ] **Step 4: Register the module with a per-connection registry and destructor**

In `vectorlite/vectorlite.cpp`, add the include near the top with the other project includes (after `#include "virtual_table.h"`):

```cpp
#include "index_registry.h"
```

Replace the module registration (lines 99-104):

```cpp
  rc = sqlite3_create_module(db, "vectorlite", &vector_search_module, nullptr);
  if (rc != SQLITE_OK) {
    *pzErrMsg = sqlite3_mprintf("Failed to create module vector_search: %s",
                                sqlite3_errstr(rc));
    return rc;
  }
```

with a `sqlite3_create_module_v2` call that owns a registry and frees it on connection close:

```cpp
  auto* registry = new vectorlite::IndexRegistry();
  rc = sqlite3_create_module_v2(
      db, "vectorlite", &vector_search_module, registry,
      [](void* p) { delete static_cast<vectorlite::IndexRegistry*>(p); });
  if (rc != SQLITE_OK) {
    *pzErrMsg = sqlite3_mprintf("Failed to create module vector_search: %s",
                                sqlite3_errstr(rc));
    return rc;
  }
```

(Note: `sqlite3_create_module_v2` calls the destructor even when it returns an error, so the registry is not leaked on failure.)

- [ ] **Step 5: Build**

Run: `cmake --build build/dev -j8`
Expected: compiles cleanly. If the compiler reports that `space_`/`index_`/`allow_replace_deleted_` are reference members needing initialization, confirm the member declaration order in the header (Step 1) places `handle_` before the references.

- [ ] **Step 6: Run the Task 1 regression tests**

Run:
```bash
cd /Volumes/external/vectorlite && . .venv/bin/activate && pytest bindings/python/vectorlite_py/test/vectorlite_test.py -k "survives or name_reuse" -v
```
Expected: all four PASS.

- [ ] **Step 7: Run the full suite (regression check)**

Run:
```bash
ctest --test-dir build/dev/vectorlite --output-on-failure
pytest bindings/python/vectorlite_py/test -v
```
Expected: all C++ tests pass; all Python tests pass (existing save/load, knn, rowid-filter, quantized, version tests included).

- [ ] **Step 8: Commit**

```bash
git add vectorlite/virtual_table.h vectorlite/virtual_table.cpp vectorlite/vectorlite.cpp
git commit -m "feat: keep index in per-connection registry so it survives reparse"
```

---

## Task 4: Documentation note and final verification

**Files:**
- Modify: `README.md`
- Modify: `doc/markdown/api.md`

- [ ] **Step 1: Add a lifetime note to README**

In `README.md`, locate the persistence note added by the previous change — the sentence ending "The in-memory index is lost on connection close unless you explicitly save it." Replace that sentence with:

```
The in-memory index is held per database connection and survives schema changes (e.g. `VACUUM`, `ALTER TABLE`, or DDL from other connections) for the life of the connection. It is lost when the connection closes unless you explicitly save it.
```

- [ ] **Step 2: Mirror the note in api.md**

In `doc/markdown/api.md`, find the same sentence ("The in-memory index is lost on connection close unless you explicitly save it.") and replace it with the same updated text from Step 1.

- [ ] **Step 3: Full build + test + example**

Run:
```bash
cd /Volumes/external/vectorlite && sh build.sh
```
Expected: C++ unit tests and all Python integration tests pass.

Then run the example:
```bash
. .venv/bin/activate && python examples/index_serde.py
```
Expected: exits 0 with recall/neighbor output.

- [ ] **Step 4: Commit**

```bash
git add README.md doc/markdown/api.md
git commit -m "docs: note that the in-memory index survives schema reparse"
```

---

## Final verification

- [ ] Run `sh build.sh` — all C++ and Python tests pass.
- [ ] Run `python examples/index_serde.py` — exits 0.
- [ ] Manually confirm the bug is fixed: a table with inserted vectors still returns results after `VACUUM` (covered by `test_index_survives_vacuum`).
