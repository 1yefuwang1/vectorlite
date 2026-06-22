# Comprehensive Python Integration Regression Suite — Design

## Purpose

Expand vectorlite's Python integration tests into a comprehensive
behavior-characterization suite that pins down the extension's *observable SQL
behavior*, so regressions in any future change are caught. The suite exercises
the public surface end-to-end through SQLite (via `apsw`) rather than internal
C++ APIs.

## Scope

In scope (categories requested):

- **Numerical correctness** — distance values vs. numpy reference, quantization
  tolerances, cosine normalization.
- **Query semantics** — `knn_search` ordering, `k`/`ef` parameters, rowid
  filters, multiple `knn_search`, empty results.
- **DDL & option parsing** — element types, spaces, all `hnsw(...)` options.
- **Persistence & lifecycle** — save/load, rename, vacuum, multi-table, reopen.
- **DML edge cases** — update, duplicate rowid, delete-then-search, capacity
  limits, large batches.
- **Data integrity** — vector round-trip across types/dims, byte-exact reads.
- **Error handling** — malformed inputs, dimension/type mismatches, invalid
  options and arguments.

Out of scope: C++ unit tests (already covered under `vectorlite/*_test.cpp`),
benchmarks, concurrency/threading stress.

## Determinism strategy (seeded-strict)

Assertions favour mathematically-guaranteed outcomes over brittle HNSW graph
internals — strict, yet robust to incidental implementation changes.

- All test data is generated with a fixed `numpy.random.default_rng(seed)`.
- Tables use a fixed `random_seed=` in `hnsw(...)` where graph determinism
  matters.
- For ordering/recall assertions, use a small `N` with `ef >= N` so HNSW
  returns *exact* KNN, then compare the full returned rowid ordering against a
  numpy brute-force reference.
- Distance values are asserted against numpy. Note hnswlib returns **squared**
  L2 distance (no square root), matching existing tests.
- Exact-match retrieval: inserting vector `v` at rowid `r` and querying
  `knn_param(v, 1)` must return `r` with distance ≈ 0 (for `l2`/`cosine`).
  Inner product (`ip`) is not a true metric, so exact-match assertions are
  skipped for it.

## Structure

New files under `bindings/python/vectorlite_py/test/`, plus a shared
`conftest.py`. The existing `vectorlite_test.py` is left untouched; new files
are complementary with minimal overlap.

| File | Categories | Contents |
|---|---|---|
| `conftest.py` | shared | `get_connection()` factory, seeded RNG fixtures, `brute_force_knn()` numpy reference, per-space distance helpers, element-type tolerance map, small shared constants |
| `test_scalar_functions.py` | numerical, errors | `vector_distance` for l2/ip/cosine vs numpy across dims; `vector_to_json`/`vector_from_json` round-trip and malformed-JSON / wrong-arg errors; `vectorlite_info` content |
| `test_vector_roundtrip.py` | data_integrity | byte-exact f32 read-back; bf16/f16 dequant tolerances; multiple dims; cosine normalization on read |
| `test_knn_query.py` | query_semantics, numerical | exact-match nearest (dist ≈ 0); full ordering vs brute force with `ef >= N`; `distance` column vs numpy; `k > N` caps to N; `ef` parameter; empty table; rowid filters (`in`, `or`, ranges); multiple `knn_search` |
| `test_ddl_options.py` | ddl_options, errors | all element types × spaces (`l2`/`ip`/`cosine`/empty); parse `M` / `ef_construction` / `random_seed` / `allow_replace_deleted`; missing `max_elements`; invalid keys/values; malformed column-type/dim syntax |
| `test_dml.py` | dml_edge, errors | update existing rowid reflected in re-search; duplicate-rowid → error; delete-then-search; capacity-exceeded → error; `allow_replace_deleted=false` vs default behaviour; wrong-dim insert → error; large batch insert |
| `test_persistence.py` | persistence | save/load across all element types; save empty index; overwrite save; load nonexistent / mismatched → error; multiple tables in one DB; file-backed reopen; rename; vacuum |

## Verified current behaviors to lock

Confirmed empirically against the built extension:

- Duplicate rowid insert → `SQLError: row N already exists`.
- `k <= 0` in `knn_param` → error `k should be greater than 0`.
- `k > N` → returns all N rows (capped).
- Inserting beyond `max_elements` (no deleted slots) → error
  `The number of elements exceeds the specified limit`.
- Wrong-dimension insert → error `Dimension mismatch: vector's dimension X,
  table's dimension Y`.
- `update ... set e = ?` on an existing rowid updates the stored vector.

## Testing / validation

- Run the full suite: `pytest bindings/python/vectorlite_py/test` (with the
  built `.so` available on the import path).
- All existing and new tests must pass before completion.
- New error-message assertions match on substrings to avoid coupling to exact
  phrasing where reasonable, but lock the meaningful part of each message.

## Non-goals / YAGNI

- No new test framework or runner; reuse pytest + apsw + numpy.
- No changes to extension source code (test-only change).
- No exact HNSW graph snapshotting (too brittle); rely on brute-force-checkable
  properties instead.
