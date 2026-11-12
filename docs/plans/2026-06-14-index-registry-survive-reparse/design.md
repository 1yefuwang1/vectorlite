# Connection-Scoped Index Registry (Survive Schema Reparse) Design

Date: 2026-06-14
Status: Approved (design)

## Problem

After the explicit-persistence change (PR #51), a vectorlite virtual table is a
pure in-memory hnswlib index whose lifetime is tied to the `VirtualTable` C++
object. SQLite creates that object in `xConnect`/`xCreate` and destroys it in
`xDisconnect`/`xDestroy`.

The trap: SQLite destroys and recreates the `VirtualTable` object **on every
schema reparse**, not just on open/close. The SQLite docs state the `xConnect`
method "is invoked whenever a database connection attaches to or reparses a
schema." On reparse SQLite calls `xDisconnect` on the live object and `xConnect`
to build a fresh one — and the current `xConnect` builds a **fresh empty**
index. All inserted vectors are silently lost (no error).

This was verified empirically. Operations that trigger a reparse and wipe the
index:

| Operation | Same connection | Another connection |
|-----------|-----------------|--------------------|
| `CREATE TABLE` / `CREATE INDEX` (unrelated) | survives | **wipes** |
| `DROP TABLE` (unrelated) | survives | **wipes** |
| `ALTER TABLE` (unrelated) | **wipes** | **wipes** |
| `VACUUM` | **wipes** | **wipes** |
| `PRAGMA schema_version=N` | **wipes** | — |
| plain `INSERT`/`UPDATE` (data only) | survives | survives |

The impact is broader than concurrent writers: routine single-connection
maintenance (`VACUUM`, `ALTER TABLE` on any table) silently empties the index.
Failures are silent — queries just return fewer/zero neighbors, and rowid
uniqueness is lost (re-inserting a previously-existing rowid succeeds), so a
vectorlite index and a separate metadata table can silently desync.

Note: VACUUM does not operate on the virtual table's data (vtab content lives
outside the SQLite file); the loss is purely a side effect of the reparse that
VACUUM triggers.

## Goal

Make a table's in-memory index survive any `xDisconnect`/`xConnect` reparse
cycle within the life of a database connection, without changing the
explicit-persistence model or cross-connection semantics. The index should be
discarded only when the table is dropped or the connection closes.

## Approach

Decouple the index lifetime from the short-lived `VirtualTable` object by storing
each table's stateful core in a **per-connection registry** that outlives
reparse cycles. The `VirtualTable` object becomes a thin wrapper holding a
non-owning pointer to its registry entry.

The registry is owned by the module's `pAux` pointer and freed by a destructor
that SQLite invokes when the connection closes (via
`sqlite3_create_module_v2`). On reparse, `xDisconnect` leaves the entry in the
registry and the following `xConnect` reattaches to it instead of building an
empty index.

## Detailed Design

### 1. The registry entry owns space + index together

hnswlib caches a pointer into its space object: `loadIndex`/the constructor set
`dist_func_param_ = s->get_dist_func_param()`, and for the L2/IP/cosine spaces
`get_dist_func_param()` returns `&dim_` — a pointer **into** the
`SpaceInterface`. Therefore the index and its space must live and die together;
an index can never be paired with a freshly-rebuilt space.

The registry entry (call it `IndexHandle`) owns the full stateful core as a unit:

```cpp
struct IndexHandle {
  NamedVectorSpace space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
  bool allow_replace_deleted;
  // The exact module-argument strings (argv[3], argv[4]) that defined this
  // table, used to detect a name collision on xConnect (see section 4).
  std::string vector_space_str;
  std::string index_options_str;
};
```

### 2. The registry

```cpp
// Keyed by (schema_name, table_name) to disambiguate main/temp/attached.
using RegistryKey = std::pair<std::string, std::string>;
class IndexRegistry {
 public:
  IndexHandle* Find(const RegistryKey& key);
  IndexHandle* Insert(const RegistryKey& key, IndexHandle handle);  // replaces
  void Erase(const RegistryKey& key);
 private:
  std::map<RegistryKey, std::unique_ptr<IndexHandle>> handles_;
};
```

One `IndexRegistry` is created per connection in `sqlite3_extension_init` and
passed to `sqlite3_create_module_v2` as `pAux`, with a destructor that
`delete`s it on connection close:

```cpp
auto* registry = new IndexRegistry();
sqlite3_create_module_v2(db, "vectorlite", &vector_search_module, registry,
                         [](void* p) { delete static_cast<IndexRegistry*>(p); });
```

`pAux` is delivered to `xCreate`/`xConnect` as their `pAux` argument; from there
the `VirtualTable` stores the `IndexRegistry*` so all callbacks can reach it.

### 3. VirtualTable becomes a thin wrapper

`VirtualTable` no longer owns `space_`/`index_`. It holds:

```cpp
IndexRegistry* registry_;   // not owned
RegistryKey key_;           // this table's (schema, name)
IndexHandle* handle_;       // not owned; lives in registry_
```

All index access goes through `handle_->index`, `handle_->space`, etc. The
`space_`/`index_` members and the constructor that built them move into the
handle-construction path used by `xCreate`/`xConnect`.

### 4. Lifecycle wiring

`InitVirtualTable` takes a `bool is_create` (Create vs Connect) plus `pAux`. The
key is built from `argv[1]` (schema) and `argv[2]` (table name). The module
arguments are `argv[3]` (vector space) and `argv[4]` (index options).

| Callback | Trigger | Behavior |
|----------|---------|----------|
| `xCreate` | `CREATE VIRTUAL TABLE` | Build a fresh `IndexHandle` from `argv`; `Insert` it (replacing any stale entry under the same key — see below). Never reuse vectors. |
| `xConnect` | open **and every reparse** | **Find-or-create with validation:** if an entry exists for the key AND its stored `vector_space_str`/`index_options_str` equal the current `argv`, reuse it. Otherwise build a fresh handle from `argv` and `Insert` (replace). |
| `xDisconnect` | reparse / close | Delete only the `VirtualTable` wrapper. **Leave the entry in the registry.** |
| `xDestroy` | `DROP TABLE` | `Erase` the entry, then delete the wrapper. |
| connection close | — | `pAux` destructor frees the whole registry. |

The single line that fixes the bug is `xConnect` reusing an existing validated
entry instead of starting empty.

### 5. Name-collision safety (the two guards)

A registry entry survives `xDisconnect`, so we must ensure a stale entry can
never bind to a *different* table that happens to reuse the name.

- **Guard A — `xCreate` always replaces, never reuses.** `CREATE` means a new
  table by definition. If an entry already exists for the key at `xCreate`
  time, it is a stale leak; overwrite it with a fresh handle (assert in debug
  builds). A newly created table never inherits old vectors.

- **Guard B — `xConnect` validates before reuse.** SQLite reconstructs `argv`
  for `xConnect` from the authoritative stored schema. Reuse the entry only if
  the stored `vector_space_str` and `index_options_str` exactly match the
  current `argv[3]`/`argv[4]`. On any mismatch, discard the stale entry and
  build fresh from `argv`. This makes the dangerous case — reusing a dim-4
  index for a dim-128 table — impossible: a differently-shaped former table can
  never match, so it is rebuilt, not silently bound.

The normal collision path `CREATE x(dim4)` → `DROP x` → `CREATE x(dim128)` is
already safe because `DROP` routes to `xDestroy`, which `Erase`s the entry; the
second `CREATE` finds nothing. Guards A and B defend against any logic slip
where an entry lingers.

### 6. Interaction with explicit save/load

`SaveTo`/`LoadFrom` operate on `handle_->index`/`handle_->space` exactly as
today. `LoadFrom` still builds a new index and swaps it into the handle on
success, re-applying `handle_->allow_replace_deleted`. No semantic change for
the persistence commands; they remain the cross-session durability mechanism.

## Edge Cases and Decisions

- The registry is strictly **per-connection**; no index is shared across
  connections. Two connections to the same DB keep independent in-memory
  indexes (unchanged from today).
- `TEMP` and attached-database tables are disambiguated by including the schema
  name in the key.
- An index is freed only on `DROP TABLE` (`xDestroy`) or connection close
  (destructor). A reparse never frees it.
- Closing the connection without saving still loses the in-memory index — that
  is what `save`/`load` is for. Out of scope here.

## Testing

C++ unit tests are limited (no SQL harness), so behavior is verified through
Python integration tests, plus targeted C++ tests for the registry data
structure.

C++ unit tests (`IndexRegistry`):
- Insert/Find/Erase round-trip.
- Insert replaces an existing entry under the same key.

Python integration tests (`bindings/python/vectorlite_py/test`):
- **Regression:** insert vectors, run `VACUUM`, then query — rows survive.
- **Regression:** insert vectors, run `ALTER TABLE other ADD COLUMN`, then
  query — rows survive.
- **Regression:** insert on connection A; run DDL on connection B; query on A —
  rows survive.
- Name reuse with a different shape: `CREATE x(dim=4)`, insert, `DROP x`,
  `CREATE x(dim=8)` — the new table is empty and dim-8 (no stale dim-4 data).
- Existing save/load, knn, rowid-filter, and quantized tests continue to pass.
- `vectorlite_info()` / version test continues to pass.

## Out of Scope

- Storing the index inside the SQLite database (shadow-table persistence).
- Cross-connection sharing of an in-memory index.
- Surviving connection close without an explicit `save`.
