# SQLite-Native DiskANN Optimization Results (Format 3 Preserved)

Date: 2026-11-12

## Executive Summary

**Target achieved with zero format-breaking changes, bit-identical physical graphs, and faster queries.**

Following the [Format-4 post-mortem](<../2026-11-10-garnet-provider-optimizations/results.md>)—which showed that splitting `(state, public_rowid)` and `(vector)` across two B-trees caused a 17–35% build regression via two-table `LEFT JOIN`s (`+43.8%` SQLite VDBE steps and `+84.6%` pager `CACHE_MISS`es under SQLite's default 2 MiB cache)—we redesigned the optimizations around SQLite's single-B-tree locality in the [Stage 1–3 plan](<plan.md>).

Across **70 primary untraced native builds** (5 workloads × [2 baseline selfchecks + 2 warmups + 5 alternating AB/BA pairs]) and **6 supplemental WAL mutation/consolidation runs**:
- **Primary target met (`10,000×128D` file-backed L2 `batch8`):** median build time dropped from **48.941013 s to 30.262014 s** (**−38.17%**, median paired speedup **1.6201×**, range **1.6070×–1.6254×**, exceeding the `≥1.5×` target).
- **Every workload sped up (`1.42×–1.83×` build, `1.11×–1.45×` query p50):**
  - `3,000×128D` L2 `batch8` (`:memory:`): **8.020381 s → 4.400331 s** (**1.8297×** median paired speedup).
  - `3,000×128D` cosine `batch8` (`:memory:`): **7.883110 s → 4.323529 s** (**1.8279×** median paired speedup).
  - `3,000×128D` L2 ordinary `INSERT` (`:memory:`): **8.682125 s → 6.115826 s** (**1.4232×** median paired speedup).
  - `10,000×768D` L2 `batch8` (file): **73.898130 s → 47.727184 s** (**1.5951×** median paired speedup).
  - Window-100 read-only KNN `p50` latency improved across all five workloads (**1.30×–1.45×** faster).
- **Bit-exact equivalence & Format-3 compatibility:** Every single run matched the baseline's **exact physical Format-3 graph SHA-256** (ordered records, edges, vectors, and metadata excluding random `instance_id`), **ordered top-10 labels**, **IEEE-754 float32 `distance_bits`**, and **float64-oracle `recall@10`**. Existing Format-3 databases remain 100% compatible with no migration required.

Local evidence: [Final validated summary](<../../../build/benchmarks/sqlite-opt-20261111/final-summary-r5.json>) · [128D raw runs](<../../../build/benchmarks/sqlite-opt-20261111/final-r5-128/results.json>) · [768D raw runs](<../../../build/benchmarks/sqlite-opt-20261111/final-r5-768/results.json>) · [Supplemental WAL runs](<../../../build/benchmarks/sqlite-opt-20261111/supplemental-r5-wal/summary.json>).

---

## What Changed (and What Was Deliberately Excluded)

1. **Preserved Single-Table Format 3 Co-Location ([diskann_store.rs](<../../../vectorlite/src/diskann_store.rs>)):**
   - Kept `(node_id, public_rowid, state, vector, neighbors)` in one `_diskann_nodes` rowid B-tree so every `Identity`, `Vector`, and `Neighbors` read touches a single B-tree with zero `LEFT JOIN`s and preserves leaf-page spatial locality under SQLite's default 2 MiB page cache.
2. **True Multi-Row `IN (?1..?K)` SQL Point Reads ([diskann_core.rs](<../../../vectorlite/src/diskann_core.rs>), [diskann_store.rs](<../../../vectorlite/src/diskann_store.rs>)):**
   - Added `GraphStore::with_vectors_batch` and batched neighbor identity checks (`validate_neighbor_identities`) using prepared `WHERE node_id IN (?1, ..., ?K)` plans for bucketed arities `4, 8, 16, 32`.
   - SQLite executes `IN (...)` on `INTEGER PRIMARY KEY` via an ephemeral sorted key list and sequential `SeekRowid` probes, amortizing VDBE startup/reset overhead by 4×–32× and traversing B-tree leaf pages in ascending `node_id` order while delivering `(neighbor, distance)` to DiskANN callbacks in exact original neighbor order.
   - Applied across unfiltered beam search (`Accessor::expand`), filtered beam search (`expand_beam_filtered`), and pruning (`PruneAccessor::fill_batch`).
3. **Single-`CASE` Guarded BLOB Projection & Synchronous Borrowed SIMD ([sqlite.rs](<../../../vectorlite/src/sqlite.rs>), [diskann_store.rs](<../../../vectorlite/src/diskann_store.rs>)):**
   - Replaced the two-`CASE` BLOB projection (`28` VDBE opcodes) on hot node reads with `single_blob_projection` (`CASE WHEN typeof(c)='blob' AND length(c)=N THEN c ELSE NULL END`, `22` opcodes), still preventing SQLite's `Column` opcode from materializing oversized corrupt BLOBs.
   - Added `Statement::with_column_blob` with exclusive `&mut Statement` borrow and checked `bytemuck::try_cast_slice` so aligned little-endian SQLite page payloads feed directly into Highway `l2_sq_f32` / `ip_dist_f32` without intermediate heap copies, staging pure math outcomes until SQL cleanup finishes so raw SQLite/cleanup errors always win.
4. **Sequential-Exact Inline `DistanceCache` Simulation & `RecentNeighbors` ([diskann_core.rs](<../../../vectorlite/src/diskann_core.rs>)):**
   - Replaced per-search `HashMap<u64, OwnedVector>` with an inline 31-entry `DistanceCache`, using `sim_cache` simulation during batched expansion so hit/miss visit charging, clear-on-full boundaries, and post-batch cache state match sequential scalar `distance()` 1-for-1.
   - Added `RecentNeighbors` so `PruneAccessor::get_neighbors` preserves raw storage order for an immediately consecutive `append_vector` (cleared on `fill`, `set_neighbors`, or any `mutation_stamp` change), eliminating duplicate adjacency reads during backedge appends.
5. **Budget-Metered `IdentityCache` Upgrade & Plan Admission ([identity_cache.rs](<../../../vectorlite/src/diskann_store/identity_cache.rs>), [diskann_store.rs](<../../../vectorlite/src/diskann_store.rs>)):**
   - Kept the inline 256-entry `IdentityCache` unchanged for tight `cache_bytes` budgets (`24,576` / `32,768` byte admission boundaries preserved), upgrading to a 16,384-entry (`4096 × 4`) table and metering optional `IN (...)` plans via `SQLITE_STMTSTATUS_MEMUSED` (with scalar fallback) only out of surplus `cache_bytes` *after* reserving full `worst_case_core_bytes` (including effective query beam `max(search_l, k)`).
6. **Why Store-Level Vector/Adjacency Caching Was Excluded (Review Finding `B1`):**
   - During integrated review, adversarial testing proved that SQLite permits reentrant host trace/progress callbacks to mutate shadow BLOBs mid-operation—even during a `SELECT` read statement—via `sqlite3_blob_open` / `sqlite3_blob_write` without incrementing `sqlite3_total_changes64`.
   - Any store-level vector/adjacency cache in `SqliteGraphStore` could therefore mask mid-callback `blobopen` corruption relative to baseline. Removing store-level payload/adjacency caching completely (retaining only validated `(NodeState, Option<u64>)` identities cleared pre/post every write and on `total_changes64` read-DML deltas) preserved 100% baseline fail-closed corruption detection while still exceeding the `1.5×` build target (`1.62×`).

---

## Verification & Integrated Review Summary

- **Automated validation (`r5`, SHA-256 `4499cc9d453009b1abc54c99c6f7aa6603ab6e672ff563de7a0520396f7f7e7e`):**
  - `cargo fmt --all --check` and `cargo clippy --locked --workspace --all-targets -- -D warnings -D clippy::undocumented_unsafe_blocks -D clippy::missing_safety_doc`: **0 warnings/errors** in Rust.
  - `cargo test --locked -p vectorlite-sqlite-sys --features abi-check`: **passed** (including `stmt_status` and `total_changes64` layout offset and guard-page host-prefix checks).
  - `sh build_release.sh`: **28/28 CTests** and **399/399 Python tests** passed.
  - Streamed benchmark helper unit tests (`benchmark/test_diskann_benchmark_utils.py`): **14/14 passed**.
  - Minimum supported SQLite **3.38.5** native pointer batch suite: **26/26 checks passed**.
  - Node.js **v22.23.4** (`node:sqlite` 3.53.4, `better-sqlite3` 11.10.0 / SQLite 3.49.2, platform & notice checks): **all passed**.
- **Independent GPT-6.1 Sol integrated review:**
  - Multi-round adversarial review uncovered five concrete issues prior to final benchmarking: cross-table/reentrant trigger & `sqlite3_blob_write` cache coherence (`B1`), batched `DistanceCache` clear-on-full visit parity (`M1`), `HashMap` FIFO tombstone capacity growth (`M2`), `IN (...)` prepared-plan `SQLITE_STMTSTATUS_MEMUSED` accounting & `k > search_l` effective-beam headroom (`M3`), and mandatory `distance_bits` benchmark gating (`m1`).
  - Every finding was fixed and covered by dedicated Rust/SQL regression tests; independent re-verification of deployed `r5` (`4499cc9d453009b1abc54c99c6f7aa6603ab6e672ff563de7a0520396f7f7e7e`) returned **APPROVED**.

---

## Primary Matched Benchmark Results (70 Untraced Native Builds)

Hardware & environment: Apple M4 (`arm64`, 10 logical CPUs, 16 GiB RAM), macOS, shared native host `diskann_batch_benchmark` (`83bb6ff8e424fa3ff158ce45008a4f6b9e9b1c785d122ed17cd07ceee23dad12`, SQLite `3.51.2`), Python verifier SQLite `3.50.4` (`.venv` Python `3.14.2`, NumPy `2.4.4`). Fixed ANN settings: `degree=32`, `build_list_size=100`, `search_list_size=64`, `alpha=1.2f32`, `cache_bytes=67,108,864`, `max_visits=65,536`, `seed=42`, `k=10`. Baseline SHA-256: `e855093b8a45b78c066ac00e3516d4fe75deef3d12a8c5b8289bb19f652f76a2`; Candidate `r5` SHA-256: `4499cc9d453009b1abc54c99c6f7aa6603ab6e672ff563de7a0520396f7f7e7e`.

### Build Throughput

| Workload | Baseline median (s) | Candidate `r5` median (s) | Time change | Ratio of medians | Median paired speedup (min–max) | Physical graph SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| 3,000×128D L2 `batch8`, `:memory:` | 8.020381 | **4.400331** | −45.14% | 1.8227× | **1.8297×** (1.8119–1.8355) | `8afcbc09c1fd…` |
| 3,000×128D L2 `single`, `:memory:` | 8.682125 | **6.115826** | −29.56% | 1.4196× | **1.4232×** (1.4155–1.4284) | `4dadb4855dc5…` |
| 3,000×128D cosine `batch8`, `:memory:` | 7.883110 | **4.323529** | −45.15% | 1.8233× | **1.8279×** (1.8051–1.8403) | `6a1a274c04d9…` |
| **10,000×128D L2 `batch8`, file (Primary)** | 48.941013 | **30.262014** | **−38.17%** | 1.6172× | **1.6201×** (1.6070–1.6254) | `508ea72a4b71…` |
| 10,000×768D L2 `batch8`, file | 73.898130 | **47.727184** | −35.41% | 1.5483× | **1.5951×** (1.5483–1.6399) | `3bb2f9cb8538…` |

### Read-Only KNN Query Latency & Recall (`k = 10`, 500 Timed Queries per Cell)

| Workload | Window `L=10` p50/p95 (ms), base → `r5` | Window `L=50` p50/p95 (ms), base → `r5` | Window `L=100` p50/p95 (ms), base → `r5` (`p50` speedup) | `recall@10` (`L=10 / 50 / 100`) |
|---|---|---|---|---|
| 3,000×128D L2 `batch8`, `:memory:` | 0.670/0.867 → **0.601/0.745** | 1.808/1.881 → **1.419/1.474** | 2.914/2.994 → **2.147/2.210** (**1.36×**) | `0.566 / 0.918 / 0.981` (exact match) |
| 3,000×128D L2 `single`, `:memory:` | 0.666/0.810 → **0.583/0.695** | 1.818/1.896 → **1.416/1.484** | 2.910/2.989 → **2.109/2.163** (**1.38×**) | `0.569 / 0.922 / 0.980` (exact match) |
| 3,000×128D cosine `batch8`, `:memory:` | 0.666/0.784 → **0.581/0.669** | 1.760/1.824 → **1.387/1.441** | 2.832/2.901 → **2.060/2.117** (**1.37×**) | `0.592 / 0.934 / 0.985` (exact match) |
| 10,000×128D L2 `batch8`, file | 0.988/1.331 → **0.874/1.151** | 2.881/3.122 → **2.389/2.589** | 5.099/5.284 → **3.918/4.058** (**1.30×**) | `0.364 / 0.758 / 0.892` (exact match) |
| 10,000×768D L2 `batch8`, file | 1.566/2.163 → **1.211/1.622** | 4.870/5.815 → **3.482/4.004** | 8.329/9.663 → **5.748/6.218** (**1.45×**) | `0.240 / 0.540 / 0.713` (exact match) |

### Persisted Database & Snapshot Sizes

Because Format 3's schema and codecs are unchanged, persisted database sizes are unchanged:
- `10,000×128D` file: `11,264,000` bytes (`10.742 MiB`) baseline median vs `11,272,192` bytes (`10.750 MiB`) candidate median (within 2 pages of B-tree split order variance; canonical graph SHA-256 is identical).
- `10,000×768D` file: `41,193,472` bytes (`39.285 MiB`) on both baseline and candidate.

---

## Supplemental WAL Interactive Mutation & Consolidation Results

Executed **after** the primary matrix finished (`6` isolated subprocess runs = `3` alternating AB/BA pairs on `1,000×128D` L2, ordinary `INSERT` transactions of 128, `journal_mode=WAL`, 8 MiB SQLite cache, `mmap_size=0`, `16` updates + `16` deletes + full `consolidate` to `984` live nodes):

| Phase | Baseline median (s) | Candidate `r5` median (s) | Time change | Median paired speedup |
|---|---:|---:|---:|---:|
| Ordinary `INSERT` build (1,000 rows) | 1.954316 | **1.467577** | −24.91% | **1.3317×** |
| 16 `UPDATE`s | 0.050081 | **0.037842** | −24.44% | **1.3234×** |
| 16 `DELETE`s (soft mark) | 0.002448 | 0.002658 | +8.60% (+0.21 ms) | 0.9146× (sub-3 ms noise) |
| Streamed `consolidate` (984 live) | 1.696236 | **1.155031** | **−31.91%** | **1.4673×** |

All 6 runs produced identical pre-mutation graph SHA-256 (`2e6091503f5c…`), post-consolidation graph SHA-256 (`0a92ca51c16a…`), ordered predictions/distances, `recall@10 = 0.99375`, final database size (`2,846,720` bytes), and open WAL high-water size (`3,831,632` bytes; `0` after close).

---

## Caveats and Delivery State

- Measurements are on local Apple Silicon (`arm64` macOS) with warm OS file caches and bounded datasets (`3,000` and `10,000` vectors); they do not establish cold-disk, beyond-RAM, Linux x64, Windows x64, or physical macOS 11 runtime performance.
- `cache_bytes` bounds the extension's Rust/SQLite-statement workspace, not total process RSS or SQLite's own pager cache.
- At measurement handoff, source, plan, and report changes were **uncommitted and unpushed** on branch `feat/sqlite-backed-diskann`. The implementation was subsequently published in commit `0b816c5`. The earlier regressed Format-4 diff and research report remain preserved for reference.
