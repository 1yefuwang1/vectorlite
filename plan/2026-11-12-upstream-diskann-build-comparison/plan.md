# Upstream DiskANN in-memory and SSD build comparison

Supporting documents: [provider and SSD-builder research](<research.md>),
[measurement report](<results.md>).

Date: 2026-11-12
Status: Complete. The reviewed final campaign validated all 48 builds; results and reproducible tables are recorded in this bundle.

## Request

Measure both upstream DiskANN's standard in-memory provider and its SSD/disk-index
builder, using the same datasets as the SQLite integration where possible. Do not
confuse the earlier [kernel comparison](<../2026-11-11-sqlite-native-diskann-optimizations/distance-kernel-comparison.md>)
with complete index-construction cost.

## Fixed scope

- Pin upstream Rust DiskANN 0.60.0 to revision
  `97a828a500848018d8be28c3ea6d5a585b07362f`, matching vectorlite's component revision.
  Keep upstream implementation files unchanged; record source/archive and executable hashes.
- Standard in-memory baseline: `diskann-providers` full-precision provider with native
  float32 squared-L2 distances. Match the SQLite graph's degree/slack, build list,
  alpha, first-vector frozen seed, batch width 8, intra-batch candidates 8 and
  minibatch parallelism 1 where supported. Record ID/root-layout and runtime differences.
- SSD baseline: actual `diskann-disk` full-precision graph builder, including mandatory
  search-PQ training/compression, graph construction/finalization, and sectorized
  disk-layout creation. Do not substitute saving an ordinary in-memory graph.
- SQLite comparison: freshly fingerprinted current vectorlite extension and native
  batch-8 host, one transaction, `max_visits=65536`, adapter workspace 64 MiB.
  Measure `:memory:` and fresh file-backed databases separately.
- Primary inputs: seed-42 float32 L2 datasets of 3,000×128D, 10,000×128D, and
  10,000×768D; identical 100-query inputs per workload. Generate data outside timing.
  Store raw bytes for SQLite and BigANN-headered copies for upstream; verify payload equality.
- Shared graph settings: pruned degree 32 (preserve the existing slack), build list
  100, alpha float32 1.2, one worker for the controlled baseline. No silent reduction
  of degree, candidates, input size, validation or precision.

## Timing boundaries and quality

1. Report in-memory provider allocation/initialization, insertion, serialization,
   and total ready/persisted costs separately where the API allows. SQLite's timed
   build includes BEGIN/INSERT/COMMIT but excludes table DDL; do not hide that difference.
2. Report SSD constructor plus full build wall time, preserving official stage
   information where available. Include PQ processing even though graph building
   uses full precision. Publish the exact PQ settings and any root/batching differences.
3. Native SSD output uses normal file writes/flush, not necessarily fsync; SQLite
   commits have their own journal/synchronous contract. These timings do not establish
   identical durability guarantees or a storage-only overhead factor.
4. On macOS, the upstream saved-index search fallback loads the sectorized index
   into memory. Use it only to validate saved-index query results/recall, **not** to
   claim disk-resident SSD search latency. Real Linux io_uring/Windows IOCP search
   performance is outside this host's measurements.
5. Compute recall@10 against the same float64 L2 oracle at search windows 10/50/100.
   Verify label ranges, finite distances, result sizes and stable same-backend repeats.
   Cross-provider graph/distance equality is not assumed: kernels, roots, static
   final-pruning and SSD PQ navigation can differ. Report quality alongside time.

## Execution

- Implement a small standalone Rust API driver and a Python matched-data/process
  runner. Keep native build dependencies and outputs in ignored local directories;
  do not change production code or the existing Cargo workspace dependency graph.
- Check APIs and harness methodology with independent GPT-6.1 Sol review; fix findings
  before the full measurement campaign. Smoke tests are for correctness, not reported
  performance comparisons.
- Freeze executable/source provenance; run fresh subprocesses sequentially with
  warm-up and at least three measured repetitions per backend/workload. Rotate backend
  order. Do not build or run competing benchmarks during measured campaigns.
- Save each raw result immediately, retain failed/partial runs, and stop on correctness
  errors or explicit timeouts rather than dropping inconvenient samples.
- Run formatting/linting, artifact-independent harness tests and relevant release
  regression checks. Write the measured results and honest limitations in this bundle.
- Do not commit or push unless requested.

## Progress

- [x] Download the exact upstream archive and verify its recorded SHA-256.
- [x] Identify the standard memory provider and genuine SSD builder, including macOS
  search-fallback and flush-versus-fsync limitations.
- [x] Implement and review the standalone API driver and matched runner.
- [x] Build, smoke-test, and freeze artifacts.
- [x] Complete repeated measurements and query-quality validation.
- [x] Write the results report and reproduce its summary from raw JSON.
