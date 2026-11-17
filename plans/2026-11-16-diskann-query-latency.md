# DiskANN query latency: reused prepared readers (relaxed equivalence)

Canonical path: /Volumes/external/vectorlite/plans/2026-11-16-diskann-query-latency.md

Plan version: v8
Date: 2026-11-16 (Asia/Shanghai)
Owner: lead
Status: approved 2026-11-16 (v8); implemented; integrated review 3/3 APPROVED 2026-11-16

## 1. Original user request (verbatim)

`/review-gated-delivery plan first based on the findings`

Context: the user asked "how to speed up diskann search for vectorlite", an expert panel reviewed the query path, and a micro-measurement of prepared-statement cost was taken. Planning dialogue decisions: read-path-only scope; staged performance gate; and (later) **strict behavioural identity is not required for error paths** — see C1. The T0 baseline campaigns are prospective and not yet executed.

## 2. Goal

Cut SQLite-backed DiskANN `knn_search` latency by reusing prepared SQLite readers across all per-node reads of one KNN operation, collapsing HEAD's separate identity and vector statements into one combined per-node read, while keeping success-path results identical to baseline, keeping every visibility/eligibility rule, keeping fail-closed error behaviour, and keeping work and memory bounded.

## 3. Success criteria (all required)

**C1 Equivalence contract (relaxed by explicit user decision).**
- *Success path:* for every query where the baseline returns rows, the candidate must return the same rowids in the same order with the same distance bits, for all recorded workloads and the existing test suite. This is strict.
- *Failure path:* where the baseline returns an error, the candidate must also return an error and must never return partial or incorrect rows (fail closed). The error code, message and the point at which it fires may differ from baseline. Single-fault error classes remain stable (corruption still `SQLITE_CORRUPT`, budget exhaustion still `SQLITE_TOOBIG`); when two faults coincide (e.g. corruption at a node whose visit would exhaust the budget), which fault fires first may differ.
- *Never different results on success:* the candidate must not succeed with different rows, ordering or distances where the baseline succeeds.
- Eligibility semantics are non-negotiable: pending, superseded, deleted and rolled-back versions never appear in results; `current` matching is still performed per accepted node.
- Limits remain enforced: `max_visits` and `cache_bytes` still bound work and memory; exceeding them is an error, never silent truncation.

**C2 Performance gate (staged, hard; ratios from unrounded medians).** With the copied T8 protocol (file-backed WAL/FULL, 100 seeded queries, k=10, 5 interleaved AB/BA runs, p50 per run):
- Primary cells, `10000x128-wal-full`: `baseline_p50/candidate_p50 >= 4.0` at L=10 and `>= 3.0` at L=100.
- No-regression cells within 5% of the same-session baseline: `3000x128-wal-full` L=10/50/100, `10000x128-wal-full` L=50, `10000x768-wal-full` L=10/50/100. DELETE-mode configurations recorded only.
- Recall@10 per L identical to baseline. The 6-9x figure in 5.2 is an expectation, recorded but not gated.

**C3 Statement-shape gate (observable, Debug configuration).** One public KNN query over 10,000 rows with no filter and no projection performs **<= 12** `sqlite3_prepare_v2` calls at L=10 and at L=100, the counts differ by **<= 4**, the effective search list reported by instrumentation equals the requested L, and `prepares == finalizes` after cursor close on both a successful and a deterministic forced-error query. The forced-error fixture uses a valid core configuration — `diskann(degree=8, build_list_size=16, search_list_size=8, max_visits=16)` over at least 512 published rows — queried with `k=1` and `L=8`, whose first expansion executes at least one reader and then deterministically exceeds the visit budget, failing `SQLITE_TOOBIG` during traversal. A Debug-only reader-execution counter proves at least one reader ran before the error, and the observed error point, DDL and SQL are recorded in host output. The Release artifact exports no instrumentation. Inability to execute this test escalates as a delivery blocker.

**C4 Integration.** Strict fmt/Clippy with CI flags, `abi-check`, CTest Debug and Release, full Python suites, Node package tests against the SHA-verified candidate, docs updated, exact commands/exit codes recorded, including platform-specific Release symbol checks.

**C5 Evidence.** Two fresh `t8-accuracy` campaigns of 30 cells each on the frozen candidate artifact, compared by the artifact-local `compare_60_cells.py` against the two T0 baseline campaigns: ordered labels and label sets exactly equal, per-rowid distance bits exactly equal. Native-reference parity 30/30 per campaign.

## 4. Scope and non-goals

In scope (runtime query path only):
- Production statement reuse: `reset`/`clear_bindings` promoted from `#[cfg(test)]`, `stmt_status` memory reporting, strict lifecycle.
- Per-KNN-operation **reused per-ID readers plus the existing exact stream**: graph/adjacency, identity-only validation, combined identity+vector, exact scan (unfiltered stream and shared point shape), projection.
- Collapsing HEAD's identity-then-vector pair into one combined per-node read (permitted by C1's failure-path relaxation).
- A crate-internal `workspace_floor(effective_l)` helper exposing the existing core scratch floor.
- Debug-only instrumentation counters plus a CTest native host.
- Behavioural/parity tests under the C1 contract, measurement campaign, docs.

Non-goals: multi-id batch statements (removed: they cannot preserve even relaxed fault ordering without unbudgeted payload buffers, and each node still needs its own step); cross-query caches; identity/vector memo; eliding the `current` join; exact-scan crossover changes; Format 4 schema or `FORMAT_VERSION`; denormalising vectors/rowids; PQ/float16; changing `beam`, traversal or tie-breaking; removing duplicate top-k `rowid()` resolution (core-hook follow-up); new SQL functions/options/PRAGMAs; authoritative in-memory index; HNSW; 50k-100k or beyond-RAM benchmarking; platform performance claims.

## 5. Root cause and evidence

### 5.1 Cached-vs-unique SQL differential (new probe, this investigation)

Warm 10k x128 database, adapter-shaped statements ([probe](<../build/benchmarks/diskann-deferred-20261115-1615/prepare-cost-probe/measure.py>), [results](<../build/benchmarks/diskann-deferred-20261115-1615/prepare-cost-probe/prepare-cost.json>)):

| shape | cached-plan execute | unique-SQL execute | differential | share |
|---|---:|---:|---:|---:|
| `SELECT 1` | 0.48 us | 1.41 us | 0.93 us | 66% |
| graph row + frozen-vector projection | 1.80 us | 7.99 us | 6.19 us | 78% |
| 3-table identity join | 2.16 us | 10.75 us | 8.59 us | 80% |
| 3-table join + vector projection | 2.57 us | 18.26 us | 15.68 us | 86% |

**Limitations:** the differential includes Python string construction, statement-cache hash/eviction and fetch overhead; it reads one node; it omits the adjacency projection. It is a cached-versus-unique SQL differential, not a pure prepare timer. T5 re-runs it with the exact shapes used.

### 5.2 Expectation (model, not measurement)

HEAD performs about two prepared statements per scored candidate (~24 us of compilation at the measured differentials) plus one per neighbour validation. Reusing one reader per shape removes nearly all compilation, and merging identity+vector removes one execution per candidate. Expected **6-9x** on 10k x128 L=10, recorded but not gated; the gate is C2's 4x/3x. No replay artifact exists, so this is an estimate.

### 5.3 Verified code facts

- Every read prepares a new statement (no cache); `reset`/`clear_bindings` are `#[cfg(test)]` ([query.rs:130-217](<../vectorlite/src/diskann_deferred/query.rs#L130>), [sqlite.rs:46-71,276-308](<../vectorlite/src/sqlite.rs#L46>), [ffi.rs:349-352](<../vectorlite/src/ffi.rs#L349>)). `stmt_status` is in the required API prefix ([ffi.rs:102](<../vectorlite/src/ffi.rs#L102>)).
- HEAD's per-node order: `rowid(id)` charges 1 visit then prepares/steps the identity join ([query.rs:440-446,204-217](<../vectorlite/src/diskann_deferred/query.rs#L440>)); `with_vector(id)` charges 1 visit then prepares/steps the vector statement ([query.rs:467-491](<../vectorlite/src/diskann_deferred/query.rs#L467>)). `neighbors` charges 1 per expansion and 0 per target identity check ([query.rs:493-501](<../vectorlite/src/diskann_deferred/query.rs#L493>)). `exact_scan` charges per visited row, and point/IN probes charge 1 before stepping including misses; rowid > `i64::MAX` charges 0 ([query.rs:278-298](<../vectorlite/src/diskann_deferred/query.rs#L278>)). `project` charges 1 per row ([query.rs:335](<../vectorlite/src/diskann_deferred/query.rs#L335>)).
- `PublishedGraph` does not override `with_vectors_batch`; the default does `rowid` + `with_vector` per candidate ([diskann_core.rs:84-95](<../vectorlite/src/diskann_core.rs#L84>)). The combined reader replaces both with one statement per candidate.
- The fixed owner reservation includes SQL text/VM names, five vector buffers (`vector_bytes * 5`), `physical_degree * 24` adjacency storage, filter storage, per-result containers and a 64 KiB buffer ([query.rs:51-98](<../vectorlite/src/diskann_deferred/query.rs#L51>)); it does not include retained VMs of extra readers.
- `GraphConfig::scratch_bytes` rejects when `scratch + minimum_vector > max_vector_bytes` ([diskann_core.rs:177-222](<../vectorlite/src/diskann_core.rs#L177>)); core derives `l = override.unwrap_or(search_l).max(count)` inside `search` ([diskann_core.rs:1517-1523](<../vectorlite/src/diskann_core.rs#L1517>)). Exact routes bypass core admission ([query.rs:594-615](<../vectorlite/src/diskann_deferred/query.rs#L594>)).
- `prepare_v2` recompiles transparently on schema change and continues; only failed recompilation errors.
- `vectorlite` does not link libsqlite3; standalone C hosts link SQLite and load the extension ([build.rs:80-85](<../vectorlite/build.rs#L80>), [CMakeLists.txt:155-171](<../vectorlite/CMakeLists.txt#L155>)).
- Accuracy: one `--trials 2` campaign over five workloads x three L values yields 30 cells; 60 require two campaigns.

### 5.4 Historical anchors and panel

The T8 anchors (17.234/94.016 ms) were measured on artifact `a16a9710...` from the pre-delivery tree at `d442bf6`; the delivered runtime is unchanged from `df39315` to HEAD. They are historical context; the gate uses T0/T5 artifacts. Plan reviews v1-v5 established the root cause and progressively removed unsafe designs (identity memo, unbudgeted batch payloads, and combined multi-id statements whose error ordering cannot be preserved even under relaxed C1 — hence v6's per-node readers).

## 6. Baseline (captured 2026-11-16, before plan approval)

- HEAD `894c050470c3594bd46db4d5b27ca5d45ba9914e` on `feat/sqlite-backed-diskann`; tracked tree clean; 16 ignored entries; only the plan untracked.
- Runtime fingerprints: query.rs `39291f90...`, codec.rs `84aceaa0...`, diskann_store.rs `3e5bb67f...`, sqlite.rs `f5e9dd7a...`, ffi.rs `275e32ab...`, diskann_core.rs `2f69ff40...`, virtual_table.rs `758aa88d...`.
- Pre-approval investigation artifacts under gitignored `build/benchmarks/diskann-deferred-20261115-1615/{storage-probe,brute-force-probe,prepare-cost-probe}/` are evidence, not source changes.

## 7. Design decisions

**D1 Per-query ownership.** Readers live in `PublishedGraph` for one `xFilter` call and are finalized before `knn()` returns; only the metadata anchor remains open. No connection-level pool.

**D2 Reused per-ID readers plus the existing exact stream; no multi-ID batch statements.** Readers:
1. graph/adjacency read (single-table, frozen root aware);
2. identity-only validation (3-table join, no vector) for neighbour-target checks and standalone `rowid()` resolution;
3. **combined identity+vector** (3-table join with vector projection) for scored candidates — one step per candidate replaces HEAD's two;
4. exact scan, unfiltered shape (`point=false`, streams all eligible rows and is drained through `Done`);
5. exact point shape (`point=true`), which is shared by `exact_scan(Some(rowid))` (including `IN` loops) and by `project`, so filter elements and projections never re-prepare;
6. nothing else. Single-row readers are reused per read; the exact unfiltered reader is the one deliberate multi-row stream and is always drained through `Done` before reuse.
Each reuse cycle: `reset()` then `clear_bindings()` then bind. Combined-read behaviour is permitted only because C1 relaxes failure-path ordering: the vector payload is materialized by SQLite when the row is stepped, i.e. possibly before the vector visit charge and before identity validation. The row is fully validated before its vector is accepted or scored, and on any validation failure the query fails closed.

**D3 Keep every query-time validation.** Joins, blob guards, finite checks and `current` eligibility remain on every accepted node. No memo.

**D4 Visit accounting (totals) and relaxed error precedence.**

| operation | HEAD | after change |
|---|---|---|
| `neighbors(id)` expansion | 1 per call | 1 per call; per-target validations 0 |
| `state`/`identity`/`internal_id` | 0 | 0 |
| standalone `rowid(id)` (core decision, distance-cache hits, post-processing, final top-k) | 1 | 1, via the identity-only reader |
| `with_vector(id)` | 1 | 1, via the combined reader |
| `with_vectors_batch(..., need_rowid=true)` | 2 per id | 2 per id: rowid charge then vector charge, each before its validation |
| `with_vectors_batch(..., need_rowid=false)` | 1 per id | 1 per id (vector charge only) |
| `exact_scan` unfiltered / point / IN | 1 per visited row / 1 per probe incl. misses / 0 for >i64::MAX | unchanged |
| `project` | 1 per row | 1 per row |
| `start_points` | 0 | 0 |

Totals per query are identical to HEAD for the same traversal, so budget success/failure is identical on corruption-free data. Error precedence relaxes exactly as C1 states: with a single fault the class is preserved; when faults coincide the candidate may report a different one, but never wrong rows.

**D5 Lifecycle, construction failure, reprepare, and demotion.**
- Reuse only after `Step::Done`; never after `State::Failed` or a partial row. `Statement::reset` keeps its generic `Ready|Done` contract; readers reuse from `Done`.
- **Route-aware preconstruction before core:** the readers needed by the selected route are prepared and measured *before* `GraphConfig` is built, because core copies its allowance on entry and cannot absorb later readers. A prepare or accounting failure for any shape silently marks that shape scalar-only for the operation; no setup-only error is raised, and unused shapes are never prepared.
- `MEMUSED` is sampled after prepare and after each logical read; if the aggregate retained high-water exceeds the admitted total plus `REPREPARE_HEADROOM` (16 KiB), readers are demoted largest-first, mark-and-drain style: during demotion a reader is never finalized while on a `Row`; it is finalized after `Done` before the next bind, and that shape falls back to scalar for the rest of the operation. Ordinary fail-closed cleanup after a validation or SQL error may finalize immediately while preserving the primary error. Results and charges are unchanged by demotion.
- Terminal `step`/reprepare errors finalize the affected readers and surface the primary SQL error; partial results are discarded. No new error is invented for accounting reasons.

**D6 No memo.** No cached identities or payloads.

**D7 Admission and reserve composition (eager envelope, immutable core budget).**
1. Admission reuses HEAD's **unchanged full owner calculation**: SQL text/VM-name bytes, five vector buffers (`vector_bytes * 5`), `physical_degree * 24` adjacency storage, filter storage, per-result containers and the 64 KiB transient buffer ([query.rs:51-98](<../vectorlite/src/diskann_deferred/query.rs#L51>)). The formula is not redefined; only an additive reader term is introduced.
2. Route is decided exactly as HEAD decides it (eligible <= l, count >= eligible, filtered-small).
3. **Graph route:** before constructing `GraphConfig`, prepare and measure the graph-route reader set (graph row, identity-only, combined pair, exact-unfiltered fallback, plus the exact-point reader when a filter/projection needs it). Admission is `allowance = max(0, min(MAX_READER_BYTES, cache_bytes - owner - floor - REPREPARE_HEADROOM))` with `floor = workspace_floor(effective_l)` and `effective_l = override_l.unwrap_or(options.search_list_size).max(count)`, derived and validated identically to core (core rejects rather than clamps when `effective_l > max_visits`). Readers are admitted in priority order while `sum(measured MEMUSED) + next <= allowance`; a shape whose prepare fails or whose measurement is unavailable is silently scalar-only. Then `core_bytes = cache_bytes - owner - sum(admitted readers) - REPREPARE_HEADROOM` (guaranteed `>= floor`) is the only value given to `GraphConfig`; core's copied allowance and the reader set are disjoint by construction, and no additional retained reader is prepared after core starts (scalar fallback still prepares statements) — only demotion, which releases memory, may occur.
4. **Exact route:** no core admission. Owner + the exact/shared/projection readers + headroom must fit within `cache_bytes`; otherwise the operation uses the HEAD scalar path. Exact-route outcomes and error codes stay identical to HEAD.
5. If no reader is admitted on either route, the operation runs entirely on HEAD's scalar path; admission outcomes and error codes are identical to HEAD at every `cache_bytes`.
6. Readers are finalized before `knn()` returns; nothing persists across queries.
7. Exact-fallback shape selection is filter-aware: the unfiltered exact reader is prepared only when a `SearchFilter::None` fallback is reachable, and the point shape is prepared for filtered/projection paths, so "unused shapes are never prepared" stays literally true.
8. Tests assert `owner + admitted reader bytes + headroom + core_bytes <= cache_bytes` at every existing budget boundary.

**D8 Debug-only instrumentation.** `#[cfg(debug_assertions)]` counters at the `Statement` boundary in `sqlite.rs` (every `prepare_v2`, every finalize including `Drop`) plus adapter counters for effective search list; a Debug-only forced-demote hook. Debug builds export `vectorlite_test_reset_counters()`/`vectorlite_test_read_counters(...)` and the force-demote hook; the C host resolves them with `dlsym`/`GetProcAddress` after `sqlite3_load_extension`. The CTest target is Debug-only; Release asserts absence of `vectorlite_test_` symbols using `nm -D --defined-only` (Linux), `nm -gU` (macOS) or `dumpbin /exports` (Windows), with inspection-success and no-match handled as separate checks recorded in the command ledger.

**D9 Curated, parameterized, hashed harness copies.** T0 copies only sources/helpers (no prior results, no `reference/target`, no caches): `run_accuracy.py`, `verify_results.py`, `render_results.py`, `finalize_evidence.py`, `test_harness.py`, README, the `reference/` crate sources plus Cargo.toml/lock, the `t8-perf` driver, and `t1-native/src/reference.rs` at `harness/t1-native/src/reference.rs`. Parameterize `--repo-root`, `--artifact`, `--expected-sha256`, `--output-dir`; output must be a fresh directory under the campaign root; reuse the frozen cargo registry read-only with a fresh `CARGO_TARGET_DIR` (the rebuilt reference binary lives there, never copied). Artifact-local scripts `compare_60_cells.py` and `aggregate_gate.py` are hashed with the harness; baseline and candidate runs use identical finalized hashes.

**D10 Baseline and candidate built identically and measured interleaved.** T0 builds a Release extension at HEAD and records SHA-256/commands; T5 builds the candidate identically. C2 runs are interleaved AB/BA in the same session (5 runs each); T0 campaigns supply C1/C5 equivalence evidence.

**D11 Empty/small/exact paths unchanged** in outcomes and charges.

## 8. Task breakdown

| ID | Outcome | Deps |
|---|---|---|
| T0 | Baseline: source snapshot, HEAD Release artifact, baseline campaigns | - |
| T1 | Statement-reuse primitives, wrapper counters, workspace floor | T0 |
| T2 | Reused per-ID readers, eager envelope admission/failback, combined identity+vector read | T1 |
| T3 | Route wiring: neighbour validation, candidate scoring, exact/projection readers | T2 |
| T4 | Contract tests (success differential, fail-closed classes, limits, C3 host) | T2, T3 |
| T5 | Interleaved measurement campaign and gate evidence | T3, T4 |
| T6 | Integration verification, docs, integrated review | all |

**T0.** Recapture baseline; snapshot touched files; build the HEAD Release artifact and record SHA-256/commands; copy, parameterize and hash harnesses; run the five-run perf baseline and two 30-cell accuracy campaigns. Acceptance: artifact/campaign/harness hashes recorded; frozen T8 evidence untouched.

**T1.** Promote `reset`/`clear_bindings` with existing preconditions; add `stmt_status` wrapper and `Statement::memory_used()`; Debug-only prepare/finalize counters and exports; `workspace_floor(effective_l)` extracted verbatim. Files: sqlite.rs, ffi.rs, diskann_core.rs (helper only). Verification: reset/rebind equivalence, clear-before-bind, use-after-`Row`/`Failed` rejection, finalize-on-drop balance, memory fallback, API-prefix fixtures, `abi-check`. Acceptance: no behaviour change; strict Clippy.

**T2.** Add the per-ID readers (including both exact shapes); route-aware preconstruction and eager envelope admission per D7; demotion per D5; the combined reader changes one candidate read from two statements to one; set the effective-L counter. Files: query.rs. Verification: admission arithmetic unit tests (zero/negative allowance, unavailable MEMUSED, partial admission, aggregate headroom, effective-L cases including `k > search_l`); the disjointness assertion `owner + readers + headroom + core_bytes <= cache_bytes` at every existing boundary; construction-failure injection for used and unused shapes (silent scalar failback, no new error); forced-demotion results equality.

**T3.** Wire readers into all routes: neighbour validation loop (reused identity reader per target, 0 charges), standalone `rowid()` resolution (identity-only reader, 1 charge), candidate scoring via the combined reader, exact unfiltered/point scans, projection. The `with_vectors_batch` override must honour both modes explicitly: one vector-owner charge per id when `need_rowid=false`, rowid plus vector charges when true, each before its validation; it returns the same results as the core's default loop. Files: query.rs only. Verification: labels/bytes identical on the recorded topologies; both `need_rowid` modes and their visit boundaries; adjacency lengths 1/32/33/41/65 including dangling targets; duplicates; projection k=1/32/33/100; standalone-rowid charge parity; charge-total equality.

**T4.** Contract tests per C1: (a) success differential vs baseline artifact on 10k x128, 10k x768 and 3000 x128 fixtures (labels, order, distance bits, projections, filters, rowid IN, k>32); (b) limits on corruption-free data: a 1..256 `max_visits` sweep must produce the same success/failure outcome and the same rows on success as baseline; (c) single-fault error classes on corruption-free budgets: injected corruption still errors with `SQLITE_CORRUPT` and no rows; budget exhaustion still `SQLITE_TOOBIG`; (d) coincident faults (corruption at a node whose visit exhausts the budget) must error and never return rows; differences in which error fires are recorded, not failed; (e) construction/admission failure injection and forced demotion produce baseline-equal results; (f) reprepare and reentrancy: successful transparent reprepare after a schema change during a query (results baseline-equal or fail-closed), terminal step/reprepare failure after a reader has been used (fail-closed with primary error and balanced finalization), and reentrant trace/progress or mid-query incremental-BLOB mutation proving no reader serves stale bytes (fail-closed, never wrong rows); (g) C3 host `vectorlite/tests/diskann_query_counts.c` + `scripts/run_query_counts_test.py` + CMake Debug target: 10,000 published rows; exact public SQL for L=10 and L=100; effective-L assertion; `<=12` prepares; `|delta| <= 4`; finalization balance on success; and the valid forced-error fixture — `diskann(degree=8, build_list_size=16, search_list_size=8, max_visits=16)` over at least 512 published rows, queried with `k=1, L=8`, which executes at least one reader (reader-execution counter > 0) and then fails `SQLITE_TOOBIG` during traversal; DDL/SQL, error point and code recorded. Acceptance: all pass; C3 failure escalates.

**T5.** After T4 freezes source: build candidate Release artifact and record SHA; run the exact-shape differential probe; run perf with AB/BA interleaving for all configurations; run two 30-cell accuracy campaigns; run `compare_60_cells.py` and `aggregate_gate.py`; record C3 output. Acceptance: C2/C3/C5 evaluated; measured vs inferred labelled; gate miss escalates.

**T6.** Strict fmt/Clippy with CI flags, workspace tests, `abi-check`, Debug/Release builds and CTest, full Python suites, Node tests after staging and SHA-verifying the target's platform binary, Release symbol absence with the recorded platform command, doc updates, ledger, integrated review packet.

## 9. Verification commands

```sh
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets -- -D warnings \
  -D clippy::undocumented_unsafe_blocks -D clippy::missing_safety_doc
cargo test --locked --workspace
cargo test --locked -p vectorlite-sqlite-sys --features abi-check
sh build.sh
sh build_release.sh            # last, so the deployed artifact is Release
ctest --test-dir build/dev --output-on-failure
ctest --test-dir build/release --output-on-failure
.venv/bin/python -m pytest vectorlite/tests/
.venv/bin/python -m pytest bindings/python/vectorlite_py/test/
# stage and SHA-verify the target platform binary, then:
(cd bindings/nodejs/packages/vectorlite && npm test)
# Release instrumentation absence: inspect FIRST (failure of inspection fails the gate),
# then reject any matching export, then exit 0 only on verified absence.
# Linux:
test -f build/release/vectorlite/vectorlite.so || exit 1
command -v nm >/dev/null || exit 1
symbols=$(nm -D --defined-only build/release/vectorlite/vectorlite.so) || exit 1
if printf '%s\n' "$symbols" | grep -q vectorlite_test_; then echo "instrumentation exported"; exit 1; fi
exit 0
# macOS:
test -f build/release/vectorlite/vectorlite.dylib || exit 1
command -v nm >/dev/null || exit 1
symbols=$(nm -gU build/release/vectorlite/vectorlite.dylib) || exit 1
if printf '%s\n' "$symbols" | grep -q vectorlite_test_; then echo "instrumentation exported"; exit 1; fi
exit 0
# Windows (cmd):
if not exist build\release\vectorlite\vectorlite.dll exit /b 1
where dumpbin >nul || exit /b 1
dumpbin /exports build\release\vectorlite\vectorlite.dll > "%TEMP%\vl_exports.txt" 2>&1 || exit /b 1
findstr /C:"vectorlite_test_" "%TEMP%\vl_exports.txt" >nul && exit /b 1
exit /b 0
```

## 10. Measurement protocol and gate computation

Copied harness: file-backed WAL/FULL, 128-row autocommit batches, seeds 42/43, 100 queries, k=10, default options with `cache_bytes=64 MiB`, L per query, 5 interleaved runs, per-run p50, no tuning. Per configuration: rep1 A->B, rep2 B->A, rep3 A->B, rep4 B->A, rep5 A->B; aggregate each artifact's five runs. `aggregate_gate.py` computes C2 ratios and no-regression percentages from stored JSON without rounding. `compare_60_cells.py` compares campaign A-vs-A and B-vs-B (ordered labels and per-rowid distance bits; native parity 30/30 per campaign). Gated configurations listed in C2; DELETE-mode recorded only.

## 10b. Residual contract note

C1 intentionally permits failure-path ordering differences (user decision), so T4 records coincident-fault outcomes instead of failing them; every success path, every eligibility rule and every limit remains strict. This trade must be stated in the delivered documentation.

## 11. Risks and mitigations

- **R1 Gate miss.** Stop and escalate with T5 evidence; no threshold adjustment.
- **R2 Relaxed contract misuse.** Success-path equivalence is strict and differential-tested; failure paths are tested for fail-closed behaviour and single-fault classes; coincident-fault differences are recorded.
- **R3 Combined-read memory/visibility.** One row at a time, borrowed BLOB, no multi-vector buffer; validation still precedes accepting or scoring; no cached payloads across reads.
- **R4 Budget regressions.** Admission reuses HEAD's full owner calculation plus an additive reader term; existing boundaries and fallback tests guard it.
- **R5 Lifecycle/construction failures.** Eager route-aware preconstruction with silent scalar failback, demotion only between rows, finalization balance.
- **R6 Reentrancy/schema changes.** Per-query readers, no payload cache, transparent reprepare with headroom/demotion.
- **R7 Instrumentation leakage.** Release symbol checks with platform commands in T6.
- **R8 C3 execution.** Required Debug gate; inability to run it escalates.
- **R9 Evidence provenance.** Same build pipeline, identical harness hashes, AB/BA interleaving.

## 12. Assumptions

- A1 `reset`, `clear_bindings`, `stmt_status`, `next_stmt` are available at the 3.38 floor; `clear_bindings` clears unused parameters to NULL.
- A2 The core floor can be exposed without changing admission behaviour for any existing `cache_bytes`.
- A3 The CTest native-host pattern works in Debug; inability to execute C3 blocks delivery and escalates.
- A4 No concurrent user edits; T0 re-captures the baseline.
- A5 The relaxed C1 contract (section 3) is the user's explicit decision for this delivery; strict error-position identity is not required.
- A6 Debug counters exist in Debug and are absent from Release.

## 13. Rollback

Changes are confined to Rust adapter/wrapper/test-support files, the CMake test target, runner scripts and docs. Reverting the delivery commits restores HEAD; no schema/format/data change, so databases remain valid. Evidence artifacts stay under `build/`.

## 14. Review and approval process

Three independent read-only reviewers must all APPROVE this plan (v8 review round: high-effort `gpt-6.1-sol` and `gpt-5.6-sol`, plus `gemini-4-carbon`); then the user gate; implementation by separate subagents; completion requires full integrated verification and three APPROVED integrated verdicts.

## 15. Task ledger

| task | status |
|---|---|
| T0 baseline | implemented |
| T1 reuse primitives, counters, floor | implemented |
| T2 reused readers, admission, failback | implemented |
| T3 route wiring (combined read, validation loop) | implemented |
| T4 contract tests and C3 host | implemented |
| T5 interleaved measurement campaign | implemented |
| T6 integration, docs, integrated review | implemented |

## 16. Version history

- v1-v2 (2026-11-16): initial design; corrected visit map; removed identity memo; pre-core admission; C3 host; T0 baseline; harness retargeting.
- v3 (2026-11-16): exact-scan charges; prepare_v2 reprepare semantics; headroom/sampling/demotion; named reserve components and effective-L floor; explicit-L C3; harness copy; task graph fixes.
- v4 (2026-11-16): HEAD-identical precedence and per-id replay; safe mark-and-drain demotion; zero-row retry vs partial-row abort; same-session AB/BA; comparison scripts; counter ownership; Node staging.
- v5 (2026-11-16): slot-ordered VALUES-CTE streaming replacing the multi-row buffer; narrowed equivalence claim; curated harness copy; repeated demotion; target-selected Node staging.
- v6 (2026-11-16): user decision that strict error-path identity is not required; contract relaxed to strict success-path equivalence plus fail-closed errors with single-fault class stability; multi-id batching removed entirely; reused per-ID readers with a combined identity+vector read per candidate; full owner-reserve composition corrected; route-aware construction with scalar failback; C3 forced-error fixture and Release-symbol commands recorded; 6-9x estimate.
- v7 (2026-11-16): reader admission preconstructed and measured BEFORE `GraphConfig` (eager route envelope), so core's immutable allowance and retained readers are disjoint; exact route has its own calculation; demotion only releases memory; charge map covers both `need_rowid` modes and standalone `rowid()`; C3 forced-error fixture and platform Release-symbol commands recorded.
- v8 (2026-11-16): Release symbol checks now separate inspection success from symbol absence (capture tool output, fail on tool/artifact/inspection failure, fail on any `vectorlite_test_` match, exit 0 only on verified absence) on Linux/macOS/Windows. C3 forced-error fixture replaced with a valid core configuration (`degree=8, build_list_size=16, search_list_size=8, max_visits=16`, >=512 rows, `k=1, L=8`) that executes at least one reader before deterministically exceeding the visit budget, with a Debug reader-execution counter and recorded error point/code. T4 now explicitly specifies successful transparent reprepare after schema change, terminal step/reprepare failure after a reader was used, and reentrant trace/progress/incremental-BLOB mutation tests proving no stale bytes are served, under the relaxed fail-closed contract. Filter-aware exact-fallback shape selection keeps "unused shapes are never prepared" true; no additional retained reader is prepared after core starts (scalar fallback still prepares statements); stale "lazy" wording, `otool -IV`, and the scope/review-process wording are corrected.
