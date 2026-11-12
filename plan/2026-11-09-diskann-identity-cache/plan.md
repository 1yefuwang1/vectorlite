# DiskANN build acceleration with validated identity reuse

Plan version: v2

## Execution scheduling update

The fresh uncached baseline has now been captured and passed all 28 CTests and 333 Python tests using the project venv. T1's remaining benchmark-harness work and T2's cache implementation may proceed in parallel on disjoint files. This scheduling update supersedes T2's original dependency on the completed T1 review: T2 now requires the immutable baseline capture, while both T1 and T2 still require the one independent reviewer's approval before T3's final acceptance. No implementation scope, quality requirement, or verification requirement changes. This is within the user's instruction to implement now with one reviewer; no additional preliminary approval gate is introduced.

Current task ledger: T1 approved; corrected T2 approved; T3 integration APPROVED with no blockers. **All tasks are complete.** The single independent github-copilot/gpt-6.1-sol reviewer approved all three gates. Plan version v2 and implementation scope are unchanged. At implementation completion no commit or push had been made; the subsequent local commit and repository archival are authorized below.

Completion evidence: all 44 full-size alternating native A/B builds passed exact stable-graph, ordered-query-output and recall checks. Baseline/optimized median-time ratios are 1.3005x L2 batch8, 1.2810x ordinary L2, 1.2947x cosine batch8 and 1.1473x fresh-file L2: approximately 1.15–1.30x, not 2x. The final project-venv root release passed all 28 CTests and 384 Python tests; strict Clippy, rustfmt and diff checks exited 0. The separately built SQLite 3.38.5 host passed all 26 native checks. Preserved, deployed, source, runner and host fingerprints remained unchanged. The cache-dependent 24576-byte pre-DDL rejection and unchanged 32768-byte regression passed. See the [measured report](<../../build/benchmarks/identity-cache-20261108/matched-release/results.md>), [raw results](<../../build/benchmarks/identity-cache-20261108/matched-release/results.json>), [final validation](<../../build/benchmarks/identity-cache-20261108/final-validation-20261109/result.json>), [integrity proof](<../../build/benchmarks/identity-cache-20261108/final-validation-20261109/handoff.json>) and [compatibility evidence](<../../build/benchmarks/identity-cache-20261108/sqlite338-compat/result.json>). Claims remain limited to small-sample, warm-OS local macOS arm64 runs, not physical beyond-RAM/cold-disk or Linux/Windows execution; the 10k baseline's 89.2% recall is unchanged and tight budgets may reject because the fixed cache is fully charged.

## User requests and authority

Original implementation request, verbatim:

> implement your proposed change and verify what's the speed up

Quality requirement from the preceding discussion, verbatim:

> Figure out a way to speed up index build time without sacrificing recall performance

Process decision, verbatim:

> Implement now and use 1 gpt 6.1 sol reviewer

The user has confirmed the design direction, authorized implementation now, and replaced the default three-model review/advance-approval process with one independent `github-copilot/gpt-6.1-sol` reviewer. Maintain this written plan and one-reviewer implementation/integration checks, but do not wait for another preliminary plan-approval round. Reviewers remain read-only. Do not silently override a substantive reviewer blocker. No commit or push is authorized by this request.

## Repository archival and commit authorization

Subsequent user request, verbatim:

> commit your changes and move the plan to the repo

This completed plan is now stored with the repository's dated implementation plans. The follow-up authorizes a local commit of the implementation, tests, benchmark, documentation and this plan; it supersedes the earlier no-commit instruction for that local commit only. No push was requested.

Source and script links are repository-relative. Links to the ignored build directory are local verification artifacts retained from the measurements, not files included in the commit; the measured summary and validation outcomes are preserved above.

## Goal and acceptance criteria

Implement a small, bounded, callback-local cache of already validated DiskANN node identities and measure the actual index-build effect. Preserve the existing graph algorithm, persisted format and native batch ABI, float32 L2/cosine math, graph/metadata authority in SQLite, same-connection callback-thread access, strict batch task/error handling, whole-INSERT rollback, corruption detection and workspace/work limits.

Success requires:

1. A measured repeatable release-build improvement on matched inputs, not an extrapolation from SQL counts. Report regressions or inconclusive results honestly and revise the cache implementation within scope if needed.
2. Unchanged construction/search settings, candidate policy, vector normalization, distance kernels, graph-update order and logical visit charging. No reduced degree/list sizes, no worker-thread SQLite, no separate connection inside the extension, no scratch/runtime/algorithm changes in this iteration.
3. Identical stable graph snapshots and query outputs between baseline and candidate on deterministic fixtures, with recall evaluated against the same oracle. Check baseline repeatability first; investigate any nondeterminism instead of silently weakening comparisons.
4. Green required local Rust, SQLite/native ABI, native pointer/runtime, retained ops and both Python suites against the freshly built candidate. State that local macOS arm64 results are not Linux/Windows execution or a physical beyond-RAM/cold-disk validation.
5. A clean explanation of artifact provenance, settings, per-run times, paired/median speedups, recall/graph comparisons, and limitations. Repository changes remain uncommitted unless the user asks otherwise.

## Established evidence, not speedup claims

Starting checkout: repository root, branch `feat/sqlite-backed-diskann`, clean HEAD `4e64c20868188e2599ee8b88663d9ea72563cfcd`.

A read-only diagnostic on the pre-change release artifact, SQLite 3.53.4, 256 random float32 128D L2 vectors, degree 32, build_list_size 100, search_list_size 64, cache_bytes 64 MiB and max_visits 65536 observed:

| Path | Identity SELECTs | Share of SQL executions | Simulated 64-entry LRU reuse | Simulated 256-entry LRU reuse |
|---|---:|---:|---:|---:|
| Ordinary INSERT | 1,385,741 | 91.7% | 53.1% | 80.7% |
| Native batch size 8 | 1,004,353 | 90.7% | 49.1% | 87.6% |

The simulation cleared on every DML and seeded successful identity/vector/adjacency-owner reads. It did not skip any real queries. A 256-entry cache almost fits that small graph. These observations neither attribute CPU time nor demonstrate a wall-time improvement. The implementation may use set-associative replacement rather than exact global LRU; its actual hit rate and effect must be measured.

Historical 3,000-vector timings are context only. Do not compare a newly built candidate solely against an old binary/compiler/SQLite measurement.

## Scope and design

### Identity cache

- Add a private safe-Rust cache, preferably a separate child module under the store module.
- Fixed 64 sets of 4 entries (256 identities maximum). Each entry stores the full internal node ID and the checked `(NodeState, Option<public rowid>)`; valid-prefix lengths distinguish empty slots from the legitimate frozen node 0.
- Use deterministic multiplicative high-bit hashing, full-key equality, MRU-to-LRU order within each four-entry set, and bounded tail eviction. No HashMap allocation, unbounded container, generation counter or wrap hazard. No whole graph/vector data is cached.
- Cache only successful validated identities. Never cache absence, an SQL error, malformed state, a public-rowid mismatch, or an incompletely cleaned-up statement.
- `identity()` still validates input ID range, uses the cache on hits, and executes the original checked SELECT on misses.
- Seed identities from successful `vector()` and `neighbors()` owner rows, retaining their current identity/frozen-ID checks, BLOB type/length/finite/adjacency checks and statement cleanup. Do not skip any referenced identity's validation: use only an already validated cached tuple or perform the original query.
- Keep all cache access inside the existing store/callback/thread/token boundaries. The cache is never a field of the persistent table handle or registry and never survives callback return or rollback.

### Invalidation and errors

- Provide one private invalidating operation that clears before the action, captures its result without early `?` return, clears afterward and returns the original result. Add pure unit tests for both success and error paths.
- Exhaustively classify the six mutating `GraphStatement` variants: SetNeighbors, Frozen, InsertNode, AllocateMetadata, DeleteNode and DeleteMetadata. Apply bracketing to preparation, binding, stepping, draining, reset/clear-bindings and cleanup, not only successful row changes. Keep the existing statement reuse and first-error behavior.
- Retain the five consolidation bypass DML sites in their original order, but wrap all of them while a store is alive: retirement UPDATE, adjacency-clear UPDATE, metadata-reset UPDATE, tombstone DELETE and staging DELETE. Do not move finalization earlier merely to avoid invalidation, because that can change deferred-error precedence.
- Staging writes before constructing the store, revision writes after consuming `finish()`, lifecycle DDL outside a store, and carrier pre/post validation have no live identity cache. Audit this explicitly.
- No new total_changes/FFI/minimum-SQLite API requirement. Supported callbacks are serialized and the query anchor pins a snapshot; supported mutation/trigger effects occur within bracketed DML. Unsupported direct shadow edits or arbitrary callbacks that mutate during SELECT are not legitimized by this optimization.

### Resource accounting

- Reserve the full fixed cache footprint using actual `size_of::<IdentityCache>()` in the existing transient workspace estimate; include a conservative temporary-move allowance if appropriate.
- Inline storage cannot have an uncharged fictional disabled fallback. Slightly stricter near-boundary `cache_bytes` admission is an explicit risk of the added workspace and must be documented and tested before DDL/writes. Do not silently raise budgets, reduce graph work, remove checks or claim unchanged low-budget admission.
- Default benchmark settings remain identical to baseline, including 64 MiB cache_bytes and unchanged max_visits. Identity hits must not reduce logical core/vector visit charging.

### Benchmark method

- Before changing the engine, freshly build the baseline with the normal release frontend and record commit, Rust/compiler versions, CMake settings, SQLite host version and SHA-256 fingerprints. Preserve the public basename `vectorlite.dylib` in a new baseline subdirectory under a fresh ignored build artifact directory.
- Build the optimized library with the same toolchain, dependencies, target, flags and build directory, preserving it separately afterward. No checkout/reset/revert loop and no comparison against stale historical artifacts.
- Use the same native host executable for both extension variants and the same immutable dataset/query files. Extend the existing bounded native benchmark only as necessary to select L2/cosine, a fresh file-backed DB or :memory:, and an optional post-timing SQLite backup for streaming graph comparisons. Preserve the existing default interface/behavior.
- Any SQL counters/tracing are separate diagnostic runs, disabled during reported timing. Do not use Python trace callback timings as speed evidence.
- Database/snapshot destinations must be fresh and refuse overwrite. Snapshot/graph hashing, exact-oracle work, file preparation and other verification are outside the timed build. The timer continues to cover the same INSERT work and transaction commit on both variants.
- One warm-up per variant/case; alternate A/B and B/A order for measured pairs. Primary: 3,000x128D L2 native batch8, five measured pairs. Secondary: the same ordinary INSERT and cosine batch8, at least three measured pairs each; a 10,000x128D L2 fresh file-backed batch8 case, three pairs if it fits a bounded per-process timeout. Retain every completed result and stop expensive repetitions on the first correctness/process failure.
- Report all per-run times, median build times, median paired speedup and spread; do not cherry-pick the best run. Small samples and warm OS caches remain limitations.
- Compare streaming ordered node rows (`node_id`, public rowid, state, vector bytes, adjacency bytes) and relevant stable metadata, excluding only the deliberately random instance ID. Keep adjacency order significant. Pair checks within the same mode/metric; ordinary and batch graphs need not match each other.
- At fixed query windows 10/50/100, compare top-k outputs and recall against the same float64 L2/cosine oracle. Verify small deterministic L2/cosine, sparse-rowid and retired-node/update cases separately where useful. Do not silently accept a recall decrease.

## Tasks and ledger

### T1 — Fresh baseline and reproducible comparison harness

Status: approved

Outcome: immutable freshly built uncached baseline plus one matched native benchmark/verification path usable for both libraries.

Areas:
- [release build script](<../../build_release.sh>) and existing CMake/native dependency configuration (use, do not gratuitously change).
- [native batch benchmark](<../../benchmark/diskann_batch_benchmark.c>).
- [Matched A/B runner](<../../benchmark/diskann_identity_benchmark.py>) and focused benchmark-helper tests.
- New ignored artifacts under `build/benchmarks/identity-cache-*`.

Verification: fresh baseline build/test evidence; native host build; harness tests; two baseline self-comparisons with stable graph/query results; artifact fingerprints and input hashes. First reviewer checks provenance, timer boundaries, no trace overhead and no hidden graph-setting changes.

Acceptance: the optimized engine is not edited until baseline capture is complete; the comparison can distinguish library provenance and fail on graph/recall mismatch. Default existing benchmark behavior remains usable.

Dependencies: none.

### T2 — Bounded identity reuse and correctness regressions

Status: approved

Outcome: safe bounded identity cache integrated into all store paths with conservative invalidation, resource accounting and targeted regression coverage.

Areas:
- [DiskANN store](<../../vectorlite/src/diskann_store.rs>).
- [Identity cache](<../../vectorlite/src/diskann_store/identity_cache.rs>), the private safe-Rust cache module.
- [DiskANN Python regressions](<../../vectorlite/tests/test_diskann.py>) and [native batch tests](<../../vectorlite/tests/diskann_batch_native.c>) only where needed.
- [backend documentation](<../../doc/diskann.md>) for cache lifetime/workspace semantics.

Verification: pure cache unit tests (empty/frozen0, all states, full-key collisions, MRU/eviction, clearing, bounded capacity and success/error bracketing); exhaustive mutation-kind classification; targeted SQLite tests for all internal writes and consolidation bypasses, trigger-induced identity changes/corruption, deleted bridges/sparse rowids, failed writes, callback/connection/rollback separation and resource preflight. Existing pointer/host-runtime/error-draining regressions stay enabled. Run rustfmt, strict Clippy and the root release script against the candidate.

Acceptance: no new unsafe/Send/Sync/ABI/persistence-format change; no algorithm/settings/math change; all validated-record/error/rollback/visit rules preserved; one independent reviewer approves implementation and tests. Fix blockers before proceeding.

Dependencies: T1 baseline capture and review.

### T3 — Integration, speed/quality verification and report

Status: approved

Outcome: reproducible measured speedup and evidence that index/query quality did not regress.

Areas: benchmark runner/results and [benchmark documentation](<../../benchmark/README.md>), plus targeted fixes within T2 scope if needed.

Verification: matched release A/B runs and graph/query/oracle checks described above; final rustfmt/strict Clippy; full root release build script (CTest and both Python suites); native API/SQLite-3.38 compatibility where the existing harness supports it; final working-tree diff/scope review. The one independent reviewer reviews the integrated diff and evidence.

Acceptance: actual timings and quality comparisons reported with artifact/settings provenance and limitations; all required local tests green and reviewer blockers resolved. If no convincing speedup appears, report that and investigate rather than claim success. No commit/push without a new request.

Dependencies: T2.

## Commands and environment (verify actual executable paths first)

Use the existing workspace Cargo cache and release native triplet where available, without changing global package-manager settings. Keep toolchain/flags identical for A/B.

```bash
CARGO_HOME="$PWD/.cache/cargo-diskann" PYTHON="$PWD/.venv/bin/python" sh build_release.sh
cmake --build build/release --target diskann_batch_benchmark --parallel 8
cargo fmt --all --check
CARGO_HOME="$PWD/.cache/cargo-diskann" CARGO_TARGET_DIR="$PWD/build/release/vectorlite/cargo" CARGO_BUILD_TARGET=aarch64-apple-darwin VECTORLITE_VCPKG_TRIPLET_DIR="$PWD/build/release/vcpkg_installed/arm64-osx" MACOSX_DEPLOYMENT_TARGET=11.0 cargo clippy --locked --workspace --all-targets -- -D warnings -D clippy::undocumented_unsafe_blocks -D clippy::missing_safety_doc
```

All Python validation and benchmark runs used the project venv shown above. Exact completed commands and exit codes are retained in the linked local verification artifacts.

## Risks, decisions and rollback

- Query count share is not CPU share. Larger graphs can have lower cache hit rates; fixed four-way replacement is not the simulated global LRU. Measure rather than assume 2x.
- Extra fixed workspace may tighten near-boundary admission. Account it honestly, reject before mutations where applicable, document it, and do not weaken safety limits.
- Direct consolidation DML and trigger effects are the primary coherence hazards. Central invalidation and tests are mandatory.
- The cache's full internal-ID key and valid-prefix mask must handle legitimate frozen0, tombstones, sparse public rowids and collisions without confusing absence with `rowid=None`.
- A/B builds must use the same toolchain/host and shared data, not historical timings. Native benchmark enhancements must not accidentally include verification in one variant's timed region.
- Snapshot determinism must be checked, not assumed. No new randomized HashMap/cache allocation should perturb graph behavior; investigate any remaining nondeterminism.
- Physical cold-cache/beyond-RAM and Linux/Windows execution are outside the local measurement claim. This cache remains O(1) workspace and adds no O(N) Rust index.
- Keep baseline immutable for rollback/comparison. Revert only task-owned targeted changes if required; do not reset unrelated user changes. Verify exact absolute targets before any delete/move, retain failed/partial measurements, and do not overwrite existing artifact destinations.
