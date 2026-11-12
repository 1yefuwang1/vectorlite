# Garnet-inspired DiskANN optimizations — revised execution order

Plan version: v4. User-approved through plan review on 2026-11-10. The technical design was approved by one independent GPT-6.1 Sol reviewer in v3. This revision implements the user's requested process: **implement all coding tasks and tests → one integrated GPT-6.1 review → benchmark**. There are no task-by-task review gates.

## Original requests and decisions

> apply the optimizations found in garnet to vectorlite

> write the plan in the repo btw

> format breaking change is ok for now

> No need to review tasks 1 by 1. Implement all coding tasks and then review. Then benchmark

## Objective and baseline

Apply the optimizations from the [Garnet investigation](<superpowers/research/2026-11-09-garnet-diskann-storage.md>) without changing recall, graph decisions or SQLite transactional guarantees.

The repository was clean at `4a657792f5e343cd22b659cf183744de5edc3718` at approval. Preserve/fingerprint a release library from this baseline before changing extension code, but do not run performance comparisons until integrated review. It includes the existing identity cache. Historical medians were 8.42 seconds for 3,000×128D in-memory batch insertion and 53.71 seconds for 10,000×128D file-backed insertion; these are not fresh measurements.

**Performance target, not a promise:** at least 1.5× median paired build speedup on the 10,000×128D file-backed L2 batch-8 workload. Correctness is mandatory regardless of measured speedup.

## Fixed constraints

- A breaking experimental storage-format change is allowed. Introduce format 4; no automatic migration or silent modification of old indexes.
- Preserve SQL APIs, native batch ABI/tag, default batch size 8/range 1–32, monotonic internal IDs, public rowid policy, float32 L2/cosine math/normalization, degree/slack/candidates/order, and retained-topology deletion.
- Preserve SQLite ≥3.38 support, same-connection/callback-thread access, private current-thread runtime, whole-INSERT carrier rollback, and vendored first-error/task-draining behavior.
- No quantization, weaker corruption handling, persistent graph cache, independent connection/transaction, foreign-key/cascade dependence, SQLite-owned memory mutation, incremental BLOB I/O, or new upstream API patch.
- No automatic commit, push or release.

## Phase 1 — All coding tasks and verification

### C1. Format-4 storage and lifecycle

Primary areas: [SQLite store](<../vectorlite/src/diskann_store.rs>), [virtual table](<../vectorlite/src/virtual_table.rs>), [SQL tests](<../vectorlite/tests/test_diskann.py>), [native tests](<../vectorlite/tests/diskann_batch_native.c>), and Node tests.

Split storage into five shadows:

- `_diskann_nodes`: internal ID, unique public rowid, state and adjacency.
- New `_diskann_vectors`: internal ID primary key and canonical little-endian float32 vector.
- Existing metadata, transaction carrier and rebuild staging.

Adjacency is exactly `8 * (physical_degree + 1)` bytes: ordered active u64 slots, zero-filled inactive slots and trailing little-endian u64 count. Validate BLOB type/exact length, count, padding, ID range, duplicates, self-links and every referenced paired identity. Active ID 0 is legal; zero is not an ID sentinel. Empty adjacency is the complete zero-filled record. Keep metadata entrypoints on their existing separate variable-length codec/bounds.

Probe format-neutral metadata before validating new layouts. Reject old/unknown formats clearly with export/rebuild guidance; never create missing shadows or migrate/delete data during connect.

Every graph node must have a matching payload. Use checked graph-driven left joins/existence checks across both rowid lookup implementations, identity/state, neighbor owners/targets, vector reads, exact scans, rowid-only selection, projections and staging. Missing public rowid means absence; an existing graph node without payload means corruption. Opening remains bounded to schema/metadata/root and root neighbors; it is not a global orphan scan.

Allocate payload first, graph row second and metadata last, checking every affected-row count and resulting frozen/live pair identity after metadata writes. Keep intermediate states inside the carrier. Soft deletion retains vector/topology; UPDATE allocates a new pair.

Consolidation audits both paired-ID directions and counters **even without tombstones**. The revision-only branch invalidates its metadata write and performs final pairing/count/root checks. The rebuild branch stages checked canonical bytes without renormalization, retires mappings, clears topology using fixed zero records, rebuilds fresh pairs, compares rebuilt payloads with staging, verifies topology, deletes retired payloads before retired graph rows, clears staging and publishes counters/revision. Final pairing/count/root checks run after the last mutation. Never silently clean orphan payloads.

Update create/validate/drop/rename/reparse, shadow recognition, collision handling, five-table snapshots, backup/readonly/WAL/crash cases, corruption injection and empty-adjacency trigger predicates. Update both Node drivers and the [storage benchmark](<../benchmark/diskann_benchmark.py>) fixture/streamed oracle.

### C2. Shared coherence and resource accounting

Add a callback-local mutation stamp to the exhaustive write executor and maintenance bracket. Invalidate before and after every attempted DML on either table or metadata, including preparation/binding/step/drain/reset failures. Read errors invalidate reuse. Stamp overflow disables reuse rather than wrapping; metadata revision is not a coherence token.

Charge both qualified names, cached statements, fixed records, stamps, leases, retained buckets, pooled buffers, indexes, seed ownership and simultaneous views using checked arithmetic. Admit known workspace before DDL/first mutation. Reclaim unneeded retention before `TOOBIG`; never raise budgets silently or drop candidates.

### C3. Borrowed-row distance and ordered provider access

Areas: [core adapter](<../vectorlite/src/diskann_core.rs>), [SQLite wrapper](<../vectorlite/src/sqlite.rs>), store and tests.

Add a narrow synchronous BLOB lease callback. Validate identity/type/length/finite payload before use. Checked aligned little-endian data feeds the existing SIMD kernel directly; unaligned/non-little-endian data uses a charged reusable aligned decode buffer. No pointer/slice escapes its lease; no callback reenters SQL/store access or yields.

**All search/start/filtered/accept-only paths remain width 1.** Replace search's ≤31 owned-vector cache with query-specific distance entries, keeping its cap and clear-on-full policy. Invalidate by mutation stamp, charge retained metadata, and check context on hits.

Bounded multi-ID provider groups, at most eight, apply only to materialized prune fills. Internally reuse the scalar prepared statement and complete each ordinal before the next, under one scoped store boundary. This reduces allocations/copies and adapter crossings; it does not claim fewer SQLite VM executions. True multi-row SQL is deferred.

#### Logical work accounting

Add explicit private owner charge/remaining-visit methods; raw reads no longer charge implicitly. Audit callers so each logical fetch charges exactly once. Preserve owner counter lifetime across core scopes/exact fallback and existing chunk/per-rebuild-node reset sites.

- **Unfiltered:** predicate passes → expansion visit → cache lookup; miss adds core fetch visit, workspace admission, owner charge, read/cleanup and distance outcome → callback. Hits add no fetch charges.
- **Filtered navigation:** predicate passes → expansion visit → classification → distance processing even for rejected/nonmatching/tombstone nodes → callback. They remain navigation bridges.
- **Accept-only:** predicate → classification → accepted predicate → expansion visit/distance; only its existing rule can omit rejected classifications.
- **Pruning:** each unique ID charges once per fill, including retained/seeded hits. Duplicates within that fill do not charge again; later fills charge again.

Choose an admissible prune-group prefix before I/O using non-latching remaining-charge checks. Process actual charges/checks ordinal by ordinal. Flush earlier admissible work before handling a failing next ordinal at width 1. Never speculate a later fatal error or recursively access the store while borrowed.

#### Error publication

Borrowed-row distance computation is **pure and non-latching**. Store its private f32/error outcome; do not call the latching `Computer` or set the fatal latch while SQL is live.

Raw identity/payload validation retains its before-cleanup priority. End the lease and complete drain/reset/binding-clear/finalization using current primary-error rules. If validation or cleanup fails, discard staged math and preserve that original error. Only after successful cleanup re-check context and publish/latch a distance error, or publish cache/callback success. A staged math error must not become the raw statement action's error.

### C4. Pruning reuse and lazy owned-batch seeds

Use a small fallible local working set with existing lease patterns, not the upstream map's infallible allocation/immutable-overlay behavior.

Pin requested IDs, evict unneeded entries before misses and refill in original order. Capacity hints are retention targets, not hard active-fill limits. Reuse payloads only within a valid no-write stamp, especially batch candidate generation; do not promise reuse across backedge writes. Retained invalid buffers/buckets remain fully charged. Return a View only after complete validation.

Seed from bounded owned `Arc<FlatBatch>` input and actual guard internal IDs in corresponding order. At first requested lookup, validate authoritative stored data and SQL cleanup, then compare canonical bits: equal may borrow owned input; finite unequal uses stored data; malformed/missing errors. Revalidate after writes. Never eagerly check unused IDs, retain caller pointers, renormalize input or install an immutable overlay that survives invalidation.

### C5. Append snapshot and adjacency buffers

Retain one checked original-storage-order adjacency snapshot keyed by owner and mutation stamp. Populate generic adjacency output with its existing sorting behavior while reusing capacity.

Consume a matching snapshot for append, preserving the old prefix and existing deduplication order; otherwise use the original checked read. Invalidate/consume before DML and keep full store write validation. Do not replace append with a sorted generic list. Charge snapshot/serialization buffers.

### C6. Tests, benchmark support and documentation

Extend the existing [A/B runner](<../benchmark/diskann_identity_benchmark.py>), [native benchmark](<../benchmark/diskann_batch_benchmark.c>) and [helper tests](<../vectorlite/tests/test_identity_benchmark.py>) during coding, but do not run timing comparisons yet:

- Configurable dimensions and ordered float32 distance-bit output.
- Explicit baseline/candidate versions and independent physical schema/descriptor/codec validation.
- Cross-format canonical comparison of ordered IDs/rowids/states/vector bytes/active neighbors and unchanged logical counters/settings. Normalize only declared structural differences and random instance ID.
- Within-format raw repeatability, including descriptor bytes, and separate physical fingerprints.
- Genuine format-3 SQL fixture for read-only rejection tests without an old-binary CI dependency.

Required regressions include paired-record failures throughout lifecycle/selection, zero-tombstone maintenance and revision-trigger damage, five-table rollback, failed/ignored writes, sparse IDs/retired bridges, cache boundaries, duplicate fills, exact host/core charge traces, near-limit errors, mutation invalidation, over-hint fills, retained empty buckets/pools, partial failures and resource cleanup.

Explicitly test finite values causing distance overflow followed by injected SQL cleanup failure: cleanup wins without cache/callback publication. Successful cleanup exposes the math error; earlier raw-validation errors retain priority.

Use the project venv and configure CMake's Python executable explicitly. Run the root [release build script](<../build_release.sh>) covering CTest and both Python suites, rustfmt check, strict Clippy, ABI checks, native tests on SQLite 3.38.5, helper tests and both Node drivers against the same freshly built artifact. Preserve HNSW/package checks and official-platform CI; report unexecuted platforms honestly.

Update [current DiskANN documentation](<../doc/diskann.md>) for format 4, compatibility and memory/cache behavior. Leave historical research/plans historical.

## Phase 2 — One integrated review

After all coding and verification, one independent GPT-6.1 Sol reviewer inspects the full implementation, test evidence, format/lifecycle coverage, error precedence, coherence, memory accounting and graph-order preservation.

Fix blockers and obtain approval of the integrated result. Do not interrupt coding with individual task reviews. Do not start performance benchmarks before integrated approval.

## Phase 3 — Benchmark and report

After review, use frozen baseline/candidate artifacts and identical inputs/settings. Run two baseline selfchecks, per-variant warmup and five alternating AB/BA measured pairs for:

- 3,000×128D L2 batch 8 and single insert.
- 3,000×128D cosine batch 8.
- 10,000×128D file-backed L2 batch 8.
- 10,000×768D file-backed L2 batch 8.

Keep degree 32, build list 100, alpha 1.2, cache 64 MiB, max visits 65,536, seed 42, k=10 and windows 10/50/100 unchanged.

Require exact canonical graph/vector/counter equality, within-format raw repeatability, ordered labels/distance bits and float64-oracle recall equality for every workload. Investigate unexplained >5% control-workload build regressions. Any correctness failure stops measurement; code corrections must be reviewed before rerunning.

Keep diagnostics separate from untraced timing. Report medians, paired ranges, query latency, recall, storage/page/WAL effects and update/consolidation costs with hashes/settings. Save summary in the repository; keep large raw artifacts ignored.

If the target is missed, report that plainly. Do not change ANN settings, increase budgets silently, weaken validation or call an unmeasured improvement significant.

## Progress ledger

| Item | Status | Evidence |
|---|---|---|
| Save approved plan | completed | This repository file |
| Preserve unchanged baseline | completed | [Provenance](<../build/benchmarks/garnet-20261110/baseline/context.json>); SHA-256 `e855093b8a45b78c066ac00e3516d4fe75deef3d12a8c5b8289bb19f652f76a2`; captured before extension source changes |
| C1 — Format-4 storage/lifecycle | completed | Full release/native/SQL validation passed |
| C2 — Coherence/resource accounting | completed | Explicit charge/stamp/retention-pressure regressions passed |
| C3 — Borrowed-row/ordered access | completed | Staged-math/cleanup precedence and scalar charge traces passed |
| C4 — Prune/owned-batch reuse | completed | Both integrated corrections implemented; retained-only IDs hidden and old empty request capacity reclaimed |
| C5 — Append snapshot | completed | Unsorted-prefix, fast-path read-count and fallback tests passed |
| C6 — Tests/harness/docs | completed | Regression-first failures/fixes; final 151 Rust/28 CTests/509 Python, strict checks and same-artifact Node/minSQLite passed |
| Integrated GPT-6.1 review | completed | [r3: APPROVED](<../build/benchmarks/garnet-20261110/integrated-review-r3.md>), 151 Rust/28 CTests/509 Python plus exact-artifact Node/minSQLite reruns |
| Matched benchmarks/report | completed | [Final report](<2026-11-10-garnet-provider-optimization-results.md>); 70 strict-gate native builds + 6 isolated WAL mutation runs, quality unchanged; primary speed target MISSED |

### Final handoff outcome

All coding and verification completed before the continuing single reviewer's integration approvals; no task-by-task review gates were imposed. The initial negative campaign was retained, an optional pool growth correction was verified/reviewed, and the final frozen r3 library completed all five workloads with five matched pairs each. The 1.5× primary target was missed: 49.011265 s baseline → 66.059756 s candidate (0.7425× median paired ratio, +34.78% median time). Every final build case regressed by 17–35%; do not merge this bundle as a build-speed improvement. Recall/ordered graph/vector/distance equivalence passed throughout.

- [Final coding manifest](<../build/benchmarks/garnet-20261110/coding-validation/integrated-9_sqmeuy/result.json>), [r3 review](<../build/benchmarks/garnet-20261110/integrated-review-r3.md>), [r3 Node](<../build/benchmarks/garnet-20261110/coding-validation/node-pool-final-h5k3l3pl/results.json>) and [r3 SQLite 3.38.5](<../build/benchmarks/garnet-20261110/coding-validation/sqlite338-pool-final-20261110-2340/result.json>).
- [Final native summary](<../build/benchmarks/garnet-20261110/final-summary-r3.json>): 70 validated builds, 50 measured/10 baseline selfchecks/10 warmups; no ANN or budget changes.
- [Supplemental WAL evidence](<../build/benchmarks/garnet-20261110/supplemental-r3-wal/summary.json>): 3 alternating pairs, matching canonical before/after mutation graphs and ordered query results; separate host/pager/dataset/journal protocol.
- [Frozen r2 SQL diagnostics](<../build/benchmarks/garnet-20261110/diagnostic-sql-profile-20261110-2330/analysis.md>): inner statement counts −1.26%, VM steps +43.79%, with substantial paired-key identity-read overhead; instrumented evidence is not mixed with untraced timing.

Code/plan/report remain uncommitted and unpushed. Format 4 deliberately breaks stored-format compatibility and does not migrate old user indexes. The original task allowed an honestly reported missed target; no benchmark quality requirement was relaxed.

### Historical coding verification evidence (before initial integrated review)

- [Full verification manifest](<../build/benchmarks/garnet-20261110/coding-validation/integrated-l4m1n7_h/result.json>): all commands exit 0; source fingerprints identical before/after; root release build, 28 CTests, 509 Python tests, fmt, strict workspace/all-target Clippy and 17 additional benchmark-helper tests. CTest includes 145 main Rust tests, five build-support tests and SQLite ABI checks. Native C++ dependency warnings remain; this is not a warning-free C++ claim.
- [Final Node evidence](<../build/benchmarks/garnet-20261110/coding-validation/node-final-o16kn5n2/results.json>): Node 22.23.4, both built-in and better-sqlite3 drivers plus platform/notices tests passed. Release, Python and Node payload SHA-256 all match `8560a75b50a60e8926f4cb8dc4cc0646b1f805d79c524b532e03f4d05640fc63`.
- [Minimum SQLite verification](<../build/benchmarks/garnet-20261110/coding-validation/sqlite338-final-20261110-2236/result.json>): freshly compiled current native host passed 38 checks on SQLite 3.38.5 and the same final library. Identity/vector/neighbor query plans each use two integer-primary-key seeks and no population scan; all frozen source/artifact hashes remained unchanged.
- [Frozen candidate and shared benchmark host](<../build/benchmarks/garnet-20261110/candidate-r1/context.json>): candidate library SHA-256 above; host SHA-256 `83bb6ff8e424fa3ff158ce45008a4f6b9e9b1c785d122ed17cd07ceee23dad12`. No performance comparison had been run at that historical snapshot; it was subsequently rejected in integrated review and superseded by r2/r3.
- Explicit full unfiltered exact scans now compare their already-scanned live population with metadata (no additional scan). Filtered missing-public-mapping lookups remain legitimate absence; referenced internal pairs and explicit maintenance still fail closed.
- Maintenance's mandatory initial/final root validation reads consume the existing owner visit allowance, without resets or a raised budget. Redundant intermediate root payload reads were removed; unusually tight maintenance limits may reject rather than skip required validation.
