# SQLite-Native DiskANN Optimization & Ablation Plan

Date: 2026-11-11
Status: Active execution (approved staged implementation → integrated GPT-6.1 Sol review → matched benchmark).

## Motivation and Baseline

The earlier [Format-4 Garnet bundle](2026-11-10-garnet-provider-optimization-results.md) regressed build times by 17–35% across all five workloads because splitting `(state, public_rowid)` into `_diskann_nodes` and `(vector)` into `_diskann_vectors` turned every `Identity`, `Vector`, and `Neighbors` read (~15.1M calls per 3k build) into a two-B-tree `LEFT JOIN`:
- SQLite VDBE steps jumped **+43.79%** (`228.9M → 329.1M`).
- Under SQLite's default 2 MiB page cache (`PRAGMA cache_size = -2000`), 10k×128D file builds suffered **+84.6% more pager `CACHE_MISS`es** (`22.33M → 41.21M` `pread` misses), overwhelming the 3× reduction in `CACHE_WRITE`s (`154.9k → 54.7k`).
- Meanwhile, the 256-entry `IdentityCache` and prune pool were wiped on every `SetNeighbors` write (`83,591` times per 3k build), leaving ~62+ MiB of the default 64 MiB `cache_bytes` workspace completely idle while SQLite's 2 MiB page cache thrashed.

The regressed Format-4 diff is preserved in ignored local evidence (`build/benchmarks/garnet-20261110/archived-format4-attempt/`), and the source tree starts cleanly from `4a657792f5e343cd22b659cf183744de5edc3718` (matching frozen baseline SHA-256 `e855093b8a45b78c066ac00e3516d4fe75deef3d12a8c5b8289bb19f652f76a2`).

**Primary performance target:** ≥1.5× median paired build speedup on the 10,000×128D file-backed L2 `batch8` workload, with **positive speedups across control workloads** (3k×128D L2 `batch8`, 3k×128D `single`, 3k×128D cosine `batch8`, 10k×768D file `batch8`), zero recall loss, and bit-exact graph/vector/distance equivalence.

## Non-Negotiable Invariants

1. **Exact graph & recall equivalence:** Identical ANN parameters (`degree=32`, `build_list_size=100`, `search_list_size=64`, `alpha=1.2f32`, `max_visits=65,536`, `batch_size=8`), identical float32 Highway SIMD kernels/normalization, identical ordered edges, labels, float32 distance bits, and recall.
2. **SQLite transactional & trigger safety:** Host `sqlite3_api_routines` only, SQLite ≥3.38, two-row `_diskann_txn` statement-journal carrier, whole-INSERT rollback, private current-thread Tokio runtime, vendored task-draining/first-error patch, and strict trigger/corruption fail-closed semantics.
3. **Strict `cache_bytes` budget:** All adapter caches (SQL plan workspace, `IdentityCache`, callback-local vector/adjacency caches, core scratch, and prune working set) are strictly bounded and accounted within `options.cache_bytes`. Tight `cache_bytes` tests (`4096`, `24576`, `32768`, `1048576`) must continue to behave deterministically without silent budget increases.
4. **Ablation discipline:** Every stage is A/B timed on 3k×128D `:memory:` and 10k×128D file before keeping it. Any sub-change that does not improve wall time is dropped or redesigned immediately.

## Staged Design & Ablations

### Stage 0 — Storage Layout Ablation (Format 3 Single-Table vs. Inverted Split-Edges)

Test whether keeping Format 3 (`_diskann_nodes` co-locating `node_id, public_rowid, state, vector, neighbors`) or inverting the split (`_diskann_nodes(node_id, public_rowid, state, vector)` + `_diskann_edges(node_id, neighbors)`) wins in wall time when combined with read/cache optimizations:
- Format 3 requires **zero** format-breaking migration, **zero** paired-table joins or `EXISTS` probes on any path, and keeps 100% single-B-tree locality when expanding and scoring nodes during read-only `knn` queries (where each query callback is fresh).
- Inverted split-edges reduces `SetNeighbors` page writes during build, at the cost of a second B-tree for `Neighbors` reads during both build and read-only `knn` queries.
- **Decision rule:** Keep Format 3 unless inverted split-edges demonstrates a clear net build **and** non-regressing query advantage on both 128D and 768D file workloads.

### Stage 1 — Low-Overhead SQL Projections & Borrowed-Row SIMD

1. **Single-pass bounded BLOB projection:**
   - Replace the double `CASE` projection (`CASE WHEN typeof(c)='blob' THEN length(c) ELSE -1 END, CASE WHEN typeof(c)='blob' AND length(c)=N THEN c ELSE NULL END`, 28 VDBE opcodes) with a single guarded column + compact status/length check or `CASE WHEN typeof(c)='blob' AND length(c)=N THEN c END` (cutting ~8 VDBE opcodes and multiple `Column`/`Function` calls per row while still preventing SQLite's `Column` opcode from materializing oversized corrupt BLOBs!).
2. **Synchronous borrowed-row BLOB lease (`Statement::with_column_blob`):**
   - Feed 4-byte-aligned little-endian SQLite payload slices directly into Highway `l2_sq_f32` / `ip_dist_f32` via `bytemuck::try_cast_slice`, with a reusable aligned `Vec<f32>` fallback for unaligned rows.
   - Stage pure math outcomes until SQL statement reset/cleanup completes so raw SQLite/cleanup errors always take precedence.

### Stage 2 — Conservative Write-Boundary Coherence, Multi-Row `IN (...)` SQL Reads, & Append Snapshot

1. **Strict baseline cache coherence (resolved during integrated review):**
   - Because SQLite permits reentrant host trace/progress callbacks to mutate shadow rows via cross-table triggers, attached-DB triggers, dynamic DDL, or `sqlite3_blob_write` (even during read statements, without incrementing `sqlite3_total_changes64`), `SqliteGraphStore` **never** caches vector payloads or adjacency lists.
   - Only validated node identities (`(NodeState, Option<u64>)`) are cached in `SqliteGraphStore`—cleared unconditionally before and after every attempted SQL write and on any `sqlite3_total_changes64` read-DML delta—preserving 100% baseline payload-check parity.
2. **Adaptive budget-respecting `IdentityCache` + Batched `IN` Identity Validation:**
   - Keep the fixed `IdentityCache` tiny on ultra-tight `cache_bytes` configurations (so `24,576` / `32,768` byte admission tests preserve exact boundaries), and dynamically allocate a larger 16,384-entry 4-way set-associative table charged against surplus `available_bytes`.
   - When `neighbors(id)` checks neighbor identities and finds `m > 1` cache misses, validate them in **one** prepared `SELECT node_id, state, public_rowid FROM _diskann_nodes WHERE node_id IN (?1, ..., ?K)` query (metered by `SQLITE_STMTSTATUS_MEMUSED` with scalar fallback) instead of `m` scalar queries.
3. **True Multi-Row `IN (...)` Vector Reads in Beam Search & Prune:**
   - Add `GraphStore::with_vectors_batch` using cached prepared statements for arities `4, 8, 16, 32`.
   - In `Accessor::expand_neighbors` (unfiltered and filtered), collect up to `32` admissible neighbor IDs (`remaining_visits` bounded), simulate exact sequential `DistanceCache` hit/clear-on-full state, and fetch misses in sorted/deduplicated `IN (...)` batches while delivering `(neighbor, distance)` to `callback` in the **exact original neighbor order**.
   - Apply the same `IN (...)` batch read to `PruneAccessor::fill`.
4. **Immediate append adjacency snapshot (`RecentNeighbors`):**
   - Preserve raw storage order from `get_neighbors` ONLY for an immediately following `append_vector` with no intervening `fill` or write (`self.recent` is cleared on `fill`, `set_neighbors`, and any `mutation_stamp` change).

## Verification & Delivery Gates

1. Per-stage A/B benchmarks on 3k×128D `:memory:` and 10k×128D file verifying monotonic wall-time improvements and identical semantic/physical graph hashes.
2. Full test suite via project `.venv`:
   - `sh build_release.sh` (CMake release build, CTest Rust unit + ABI + ops tests, both Python suites).
   - `cargo fmt --all --check` and `cargo clippy --locked --workspace --all-targets -- -D warnings`.
   - Native batch & SQLite 3.38.5 minimum-version test suite.
   - Node.js 22 (`node:sqlite` and `better-sqlite3`) test suites against the exact built `.dylib`.
   - Dedicated unit & SQL trigger tests covering trigger-present cache invalidation, partial `IN (...)` missing-node corruption, `max_visits` exact parity, and tight `cache_bytes` boundaries.
3. One integrated review by an independent `github-copilot/gpt-6.1-sol` reviewer; fix all findings.
4. Full matched untraced benchmark matrix (`5` workloads × `5` alternating AB/BA pairs + `2` selfchecks + warmups = `70` builds) + `6` supplemental WAL update/delete/consolidation runs, and honest repository report in `docs/`. No commit or push unless requested.
