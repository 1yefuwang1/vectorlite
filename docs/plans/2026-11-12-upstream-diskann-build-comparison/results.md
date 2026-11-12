# Upstream DiskANN RAM and SSD index-build measurements

Date: 2026-11-12
Status: Complete. All 48 final builds validated; statistics reproduced from raw JSON.

## Measurements

**Measured on this Apple M4 with one worker:** the standard upstream RAM provider
builds these indexes in about **0.18 s, 0.81 s and 2.82 s**. The genuine SSD-format
pipeline takes about **0.27 s, 1.11 s and 4.11 s**, including mandatory PQ processing
and disk-layout creation. The current SQLite integration takes substantially more
time on the same inputs, but these differences are not a pure storage-overhead
factor or an equal-recall comparison.

Validated 48 builds; 36 measured.
Medians are seconds; RAM readiness, SSD pipeline, and SQLite commit have different boundaries.

| Workload | Upstream RAM ready | Upstream SSD full build | SQLite :memory: | SQLite file |
|---|---:|---:|---:|---:|
| 3000×128D | 0.182262 | 0.272251 | 4.255075 | 4.674254 |
| 10000×128D | 0.813648 | 1.111919 | 22.327018 | 30.463744 |
| 10000×768D | 2.823018 | 4.114817 | 35.247292 | 46.964040 |

| Workload | Backend | Median seconds | Observed min–max | Recall@10 L10 / L50 / L100 |
|---|---|---:|---:|---|
| 3000×128D | memory | 0.182262 | 0.181494–0.183255 | 0.566 / 0.914 / 0.981 |
| 3000×128D | ssd | 0.272251 | 0.270233–0.279091 | 0.498 / 0.901 / 0.975 |
| 3000×128D | sqlite-memory | 4.255075 | 4.205114–4.305286 | 0.566 / 0.918 / 0.981 |
| 3000×128D | sqlite-file | 4.674254 | 4.620810–4.676362 | 0.566 / 0.918 / 0.981 |
| 10000×128D | memory | 0.813648 | 0.804990–0.817480 | 0.363 / 0.765 / 0.889 |
| 10000×128D | ssd | 1.111919 | 1.108480–1.138269 | 0.361 / 0.739 / 0.887 |
| 10000×128D | sqlite-memory | 22.327018 | 22.283499–22.389485 | 0.364 / 0.758 / 0.892 |
| 10000×128D | sqlite-file | 30.463744 | 30.439418–30.502320 | 0.364 / 0.758 / 0.892 |
| 10000×768D | memory | 2.823018 | 2.745231–2.902942 | 0.234 / 0.541 / 0.720 |
| 10000×768D | ssd | 4.114817 | 4.096159–4.142504 | 0.186 / 0.454 / 0.621 |
| 10000×768D | sqlite-memory | 35.247292 | 35.044421–35.809671 | 0.240 / 0.540 / 0.713 |
| 10000×768D | sqlite-file | 46.964040 | 46.594631–46.974493 | 0.240 / 0.540 / 0.713 |

**Quality caveat:** at 10k×768D and L=100, the configured SSD path reaches **62.1%**
recall@10, versus **72.0%** for native RAM and **71.3%** for SQLite. Its 96-byte PQ
navigation, medoid root and static graph policy differ; no query tuning was used
to force equal recall. Returned distance values still pass the float64 check—the
recall loss means true neighbors were missed, not that accepted distances were
silently inaccurate. Do not present the SSD time as an equal-quality speedup.

All 48 final builds passed retained-artifact, same-backend persisted-structure and
prediction checks. Each cell has three measured builds; ranges above are observed
sample min–max, not confidence intervals. The interrupted earlier campaign and
correctness replay are retained separately and **not pooled** into these tables.


## Native setup and serialization costs

Medians in seconds. These stages are reported separately; in particular, RAM-ready
time does not silently include serialization. SSD pipeline stages remain aggregated
because their individual methods are private in the public builder API.

| Workload | RAM setup | RAM insertion | RAM save | SSD constructor | SSD complete pipeline |
|---|---:|---:|---:|---:|---:|
| 3000×128D | 0.000029 | 0.182227 | 0.001208 | 0.000003 | 0.272246 |
| 10000×128D | 0.000052 | 0.813598 | 0.014001 | 0.000003 | 1.111914 |
| 10000×768D | 0.000276 | 2.822742 | 0.040972 | 0.000004 | 4.114811 |

Serialization is the upstream save API with ordinary buffered writes, not an
added fsync-equivalent durable commit. Save is a separately measured phase after
query validation; adding its median to RAM readiness is not a continuous wall-clock
build-and-save measurement.

## What the columns measure

- **Upstream RAM ready:** configuration/runtime/provider allocation, frozen-root
  setup and all insertions into the standard native full-precision provider.
  Input file reading, exhaustive post-build audits, queries and serialization
  are excluded. Serialization is measured separately.
- **Upstream SSD build:** configuration, constructor, mandatory search-PQ training
  and compression, graph construction, final prune, intermediate save, sectorized
  disk-layout writes/flush and cleanup. Full-precision graph building is used;
  search PQ is still mandatory. Individual private stage durations are not invented.
- **SQLite memory/file:** the existing native batch-8 host's BEGIN/INSERT/COMMIT
  timer. Input loading, virtual-table creation, queries, snapshot backup and
  verification are excluded. The file variant starts from a fresh database.

These are absolute implementation build costs, **not an isolated storage-only
overhead ratio**. The implementations differ in distance kernels, runtime lifetime,
resource/corruption checks, transaction guarantees and some graph policies. RAM
readiness is not a persisted/durable index. Native file flush and SQLite commit
are not asserted to have identical durability guarantees.

## Frozen comparison artifacts

| Artifact | SHA-256 |
|---|---|
| [Upstream driver](<../../../build/upstream-diskann-20261112/final-r3/artifacts/upstream>) | `453ef1f60f3494db9ae31cec2f9e0ca3b5f90bb5457e20934e2e55bbac88ddc2` |
| [SQLite native host](<../../../build/upstream-diskann-20261112/final-r3/artifacts/sqlite-host>) | `83bb6ff8e424fa3ff158ce45008a4f6b9e9b1c785d122ed17cd07ceee23dad12` |
| [Vectorlite extension](<../../../build/upstream-diskann-20261112/final-r3/artifacts/vectorlite.dylib>) | `4499cc9d453009b1abc54c99c6f7aa6603ab6e672ff563de7a0520396f7f7e7e` |

The SQLite extension is byte-identical to the approved r5 optimization candidate
used in the earlier [SQLite optimization measurements](<../2026-11-11-sqlite-native-diskann-optimizations/results.md>).
The native SQLite host reports SQLite **3.51.2**. The production integration was
not changed to make this comparison.

## Inputs and protocol

- Host: Apple M4, arm64, 10 logical CPUs, 16 GiB RAM, macOS 27.0.1.
- Rust 1.98.1 / LLVM 22.1.8; optimized standalone driver with one codegen unit,
  empty RUSTFLAGS, explicit one-worker execution. Native ARM NEON+dotprod features
  are checked in worker results; the native baseline is not forced scalar.
- Upstream Rust DiskANN 0.60.0 at
  `97a828a500848018d8be28c3ea6d5a585b07362f` with **no source changes**.
- Inputs: NumPy RandomState(42), uniform float32 values in [0,1), datasets
  3,000×128D, 10,000×128D and 10,000×768D. Each has 100 independently generated
  queries following its corpus from the same seeded generator. The raw SQLite
  inputs and BigANN-headered native inputs have byte-identical f32 payloads.
- Major graph settings: squared L2, pruned degree 32, default slack (physical
  degree 41), build list 100, alpha float32 1.2. RAM/SQLite use first-vector
  initialization and batch 8 after a single-point bootstrap, IBC 8, max minibatch
  parallelism 1. The native SSD path uses a medoid root, single insertions and
  static final pruning. All use one requested worker.
- SSD search-PQ budgets: 32 bytes at 128D, 96 bytes at 768D; FP graph construction,
  2 GiB build RAM budget, one-shot build strategy on these small datasets. These
  are explicit comparison settings, not universal upstream CLI defaults.
- SQLite adapter workspace: 64 MiB, max_visits 65,536, unmodified native host
  journal/pager/synchronous defaults. No cache or ANN parameter is silently
  increased to manufacture an improvement.
- Final campaign: four backends × three workloads × (one warm-up + three measured
  repetitions), **48 builds total, 36 measured**. Each run is a fresh subprocess;
  backend order rotates, and this session starts no competing build or benchmark
  during timed runs. Ambient OS load, core placement and frequency are not locked.
  There are only three measured repetitions per cell, not a confidence interval.

## Quality, integrity and the connectivity finding

Every run checks finite, unique, ordered top-10 results at search windows 10/50/100.
Returned distances must agree with float64 squared-L2 calculations from the exact
input f32 values within `2e-6 * max(1, abs(reference))`. Recall is reported against
the same float64 oracle, independently of timing. Cross-provider graphs, distance
bits and recall are **not** assumed identical.

The native RAM adapter additionally bit-checks all input vectors and its frozen
copy, validates every logical adjacency list (nonempty, bounded, in-range, no
self-edges or duplicates), and repeats the exhaustive audit after reload. Ordered
query labels and distance bits must match exactly before/after save/load. Same-backend
persisted-file SHA-256s and prediction hashes must repeat, including warm-ups.
SQLite snapshots pass the existing schema, metadata, vector/edge and quick_check
validation; stable graph hashes exclude only the random instance ID.

The first campaign stopped at an overly strong reachability assumption in the
10,000×768D native RAM case. Root reachability is an upstream analysis metric,
not a guaranteed insertion invariant: later pruning can discard a vertex's last
incoming edge. Rather than change the graph or hide that diagnostic, we added
stronger exhaustive stored-vector/edge checks and published reachability separately.
Independent source review approved the correction before fresh measurements.

The correctness replay passed all exhaustive checks before and after reload for
all **10,000 live vectors + frozen root**, while observing **9,999 reachable logical
nodes and two unreachable live vertices**, IDs **1702** and **5739**. Each has zero
incoming edges; their outgoing degrees are 35 and 32. The graph was not repaired
or reconfigured. This does not claim that exhaustive integrity auditing proves
absence of every suppressed internal upstream event; it establishes the recorded
stored-data/edge checks and query quality.

Audit equality uses the complete integrity checks plus counts and an order-sensitive
FNV-1a adjacency fingerprint; that fingerprint is explicitly noncryptographic. The
separate persisted-file SHA-256 and exact query roundtrip checks are retained. We
do not call FNV equality a bit-exact adjacency roundtrip proof.

## SSD build versus SSD query execution

The builder writes genuine disk-index sectors and PQ files. On macOS, however,
upstream's saved-index search fallback loads the full disk file into a Vec. Those
searches validate the saved format and recall **only**; they do not establish
SSD-resident query latency, io_uring/IOCP behavior, cold-disk or beyond-RAM performance.

## Verification

- **48/48 final builds validated**, including 36 measured runs and 12 warm-ups;
  all same-backend persisted-file/graph, ordered-prediction and reachability checks
  repeated exactly. No preliminary or diagnostic runs are pooled into the tables.
- The artifact-checked summary utility reverified shared inputs, frozen executables,
  worker JSON, saved native files, SQLite snapshots, and original SQLite file
  databases against their backup snapshots before reproducing these statistics.
- Independent GPT-6.1 Sol integrated review approved the harness and the exhaustive
  integrity correction before the final campaign. Final read-only review recomputed
  all 12 timing cells and stage medians directly from measured worker JSON and
  verified all 48 worker fingerprints; no blockers remained.
- Strict standalone-driver Clippy, including documented-unsafe and safety-doc lints,
  passed. Direct driver rustfmt, workspace rustfmt and `git diff --check` passed.
- Artifact-independent runner tests: **7/7 passed**, including disconnected-but-stored
  graphs, invalid integrity evidence, numerical failures, input encodings and summaries.
- Final root release regression script: **28/28 CTests and 399/399 Python tests passed**.
  The interrupted campaign is explicitly rejected by final-summary tooling.

## Reproducibility and evidence

See the [plan](<plan.md>), [source-selection research](<research.md>), and
[driver/build/run guide](<../../../benchmark/upstream_diskann/README.md>).
The [standalone lockfile](<../../../benchmark/upstream_diskann/Cargo.lock>) is separate
from vectorlite's production dependency graph.

- [Successful final build provenance](<../../../build/upstream-diskann-20261112/harness-r3/build-result.json>):
  exact compiler identity, manifest/lock/source fingerprints, full archive-to-tree
  checks of all 1,410 regular upstream files before and after building, successful
  command return code and executable fingerprint.
- [Final campaign raw results](<../../../build/upstream-diskann-20261112/final-r3/results.json>):
  per-process commands, logs, settings, fingerprints, build times and quality checks.
- [Artifact-checked analysis summary](<../../../build/upstream-diskann-20261112/final-r3/analysis-summary.json>)
  and [generated tables](<../../../build/upstream-diskann-20261112/final-r3/analysis-table.md>),
  reproduced by the [summary utility](<../../../benchmark/summarize_upstream_diskann.py>).
- [Final release regression log](<../../../.cache/upstream-comparison-final-release.log>)
  and [strict standalone Clippy log](<../../../.cache/upstream-diskann-clippy-r3.log>).
- [Retained interrupted campaign](<../../../build/upstream-diskann-20261112/final-r2/results.json>):
  two complete 128D workloads followed by the stopped 768D warm-up; not final timing data.
- [768D correctness replay](<../../../build/upstream-diskann-20261112/reachability-replay-r3/results.json>):
  two full-size native-only runs validating the corrected completeness criterion;
  excluded from final timing statistics.

Raw databases, native indexes, source archives and large logs are ignored local
artifacts, not files committed with this report. At measurement handoff, source,
tools and documentation changes were uncommitted and unpushed; publication is a
separate delivery step. No production kernel, provider, SQL API or transactional
implementation was changed for these measurements.
