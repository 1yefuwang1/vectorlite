# Garnet-inspired DiskANN provider optimizations: implementation and measurements

## Result

**Implemented and correctness-reviewed, but not a build-speed improvement.** The final candidate took **17–35% more build time** across all five measured workloads. The primary 10,000×128D file-backed target was missed: median build time increased from **49.011265 s to 66.059756 s**, with median paired baseline/candidate ratio **0.7425×**, not the targeted ≥1.5×.

All **70 final native builds** passed strict canonical graph/vector/counter equality, ordered labels, float32 distance-bit and recall checks, plus within-format raw repeatability. This preserves quality; it does not turn a performance regression into a successful optimization. **Do not merge this format-4 bundle for the purpose of speeding up builds.**

[Final validated summary](<../../../build/benchmarks/garnet-20261110/final-summary-r3.json>) · [128D raw evidence](<../../../build/benchmarks/garnet-20261110/matched-r3-128/results.json>) · [768D raw evidence](<../../../build/benchmarks/garnet-20261110/matched-r3-768/results.json>)

## Implementation and review

The [approved plan](<plan.md>) applies separate graph/vector records, fixed-capacity ordered adjacency, safe borrowed-row distances, bounded coherent pruning/buffer reuse, lazy owned-batch seeds, and removal of the duplicate append adjacency read.

Storage is deliberately **format 4**, with no automatic migration from format 3. Public SQL and native batch interfaces remain unchanged. Graph settings, numeric kernels/normalization, retained-topology deletion, host connection/thread, whole-INSERT rollback and vendored error/task cleanup are preserved.

Coding was completed before integrated review by one continuing independent GPT-6.1 Sol reviewer, rather than task-by-task review gates. Its initial blockers—current-fill visibility of internally retained vectors, and reclamation of an old empty requested header—were reproduced before correction and covered by regressions. The same reviewer [approved the corrected integration](<../../../build/benchmarks/garnet-20261110/integrated-review-r2.md>) before the first performance comparison.

After an initial negative timing campaign, a bounded allocation-growth correction was implemented and [reviewed by the same reviewer](<../../../build/benchmarks/garnet-20261110/integrated-review-r3.md>) before fresh final measurements. No correctness or performance finding was hidden by switching to the older pre-identity-cache baseline, reducing graph work, raising budgets or dropping validation.

### Final correctness evidence

- [Full manifest](<../../../build/benchmarks/garnet-20261110/coding-validation/integrated-9_sqmeuy/result.json>): **151 main Rust tests, 28 CTests, 509 Python tests**, strict workspace/all-target Clippy and unsafe-documentation lints, fmt, 17 additional streamed helper tests and diff checks; all commands exit 0 with stable source hashes. Existing native C++ dependency warnings remain; this is not a warning-free C++ claim.
- [Regression-before evidence](<../../../build/benchmarks/garnet-20261110/coding-validation/review-regressions-7315wmo4/results.json>) and [after evidence](<../../../build/benchmarks/garnet-20261110/coding-validation/review-regressions-pass-dr3xdk7w/results.json>): all four integrated regressions fail before fixes and pass afterward. The >750-record case compares every ordered identity/vector/edge against a no-retention reference and demonstrates the prior graph divergence.
- [Pool-growth before](<../../../build/benchmarks/garnet-20261110/coding-validation/pool-growth-before-nw4udzu0/results.json>) and [after](<../../../build/benchmarks/garnet-20261110/coding-validation/pool-growth-pass-fgp5731_/results.json>): growth-shape regression and pressure fallback pass, with all 42 core tests passing.
- [Final Node evidence](<../../../build/benchmarks/garnet-20261110/coding-validation/node-pool-final-h5k3l3pl/results.json>): both CI Node 22 drivers and platform/notices checks pass against identical final library bytes.
- [Final SQLite 3.38.5 evidence](<../../../build/benchmarks/garnet-20261110/coding-validation/sqlite338-pool-final-20261110-2340/result.json>): 38 native checks pass; identity/vector/neighbor plans use two integer-primary-key seeks each with no population scan.

## Frozen artifacts and host

| Role | SHA-256 |
|---|---|
| Current-code baseline, including identity cache | `e855093b8a45b78c066ac00e3516d4fe75deef3d12a8c5b8289bb19f652f76a2` |
| Initial reviewed candidate r2 — negative campaign | `aa4bf8453c556fc9f20cf320934e07d692d94c930b4ba8b422dc38aadea78fb4` |
| Final reviewed candidate r3 | `3ec0dc6a3aa7517bff8f8b9a9ec1c69586b1e6716213440e5ef029f442074bff` |
| Shared native host | `83bb6ff8e424fa3ff158ce45008a4f6b9e9b1c785d122ed17cd07ceee23dad12` |

See [baseline provenance](<../../../build/benchmarks/garnet-20261110/baseline/context.json>), [initial candidate provenance](<../../../build/benchmarks/garnet-20261110/candidate-r2/context.json>) and [final candidate/host provenance](<../../../build/benchmarks/garnet-20261110/candidate-r3/context.json>). The baseline is the current code at `4a657792f5e343cd22b659cf183744de5edc3718`, **not** the slower pre-identity-cache implementation.

Execution: local macOS 27 arm64, **Apple M4, 10 logical CPUs, 16 GiB RAM** ([hardware provenance](<../../../build/benchmarks/garnet-20261110/hardware-r3.json>)). Native timing host SQLite is **3.51.2**; Python verifier/supplemental host SQLite is **3.50.4**. Project venv: Python **3.14.2**, NumPy **2.4.4**. Official-target execution beyond this host remains CI-dependent.

## Matched measurement protocol

The existing [matched runner](<../../../benchmark/diskann_identity_benchmark.py>) uses one [native host](<../../../benchmark/diskann_batch_benchmark.c>) loading each frozen library. Inputs, settings, process timeout and query work are identical within each workload. Measurements are **untraced** and use project-venv Python.

Settings: degree 32, build list 100, search list 64, alpha float32 1.2, adapter cache 64 MiB, max visits 65,536, seed 42, k=10 and query windows 10/50/100. Native chunk size remains 8; ordinary insertion is a separate workload. The process timeout is 600 seconds for both variants. No ANN or resource setting was changed to manufacture an improvement.

Every workload completed two baseline selfchecks, one warmup per variant, and five measured pairs alternating AB/BA. The final matrix contains **70 validated builds: 50 measured, 10 baseline selfchecks, 10 warmups**. The primary 10k×128D file workload ran first, followed by 128D controls and the 768D file workload. The timer includes BEGIN, insertion/statement work and COMMIT; input generation, oracle, queries, snapshot backup and validation are outside build timing.

Format-3 and format-4 raw records intentionally differ. Each physical schema/descriptor/codec is checked independently. Cross-format comparison requires identical canonical vectors, identities, **ordered active edges**, logical metadata/settings/counters, ordered labels, float32 distance bits and float64-oracle recall. Within-format raw stable metadata/record repeatability remains strict; only explicitly declared structural differences and random instance ID are normalized for cross-format comparison. Equivalence is per workload, not between ordinary/batch graphs or different datasets.

## Final r3 build results

| Workload | Baseline median s | Candidate median s | Ratio of medians | More time | Median paired ratio (range) |
|---|---:|---:|---:|---:|---:|
| 3,000×128D L2 batch8, memory | 8.103215 | 9.963177 | 0.8133× | +22.95% | 0.8153× (0.8059–0.8236) |
| 3,000×128D L2 ordinary INSERT, memory | 8.738119 | 10.839560 | 0.8061× | +24.05% | 0.8111× (0.7955–0.8206) |
| 3,000×128D cosine batch8, memory | 8.023244 | 9.771471 | 0.8211× | +21.79% | 0.8211× (0.7935–0.8384) |
| 10,000×128D L2 batch8, fresh file | 49.011265 | 66.059756 | 0.7419× | +34.78% | 0.7425× (0.7419–0.7440) |
| 10,000×768D L2 batch8, fresh file | 87.084118 | 102.105130 | 0.8529× | +17.25% | 0.8508× (0.7842–0.8666) |

Ratios below 1 mean a slower candidate. Ratio of medians, median of within-pair ratios, and percent more time are deliberately distinct statistics. Five pairs are a small sample, not a confidence interval. **The ≥1.5× primary median-paired target was missed**; its observed value is **0.742506×**. No favorable historical baseline is substituted.

### Query latency and quality

Query window 100 below pools five measured runs × 100 queries for each variant. Times are milliseconds and include stepping/materializing rowid+distance, not binding/reset or oracle work.

| Workload | Baseline p50 / p95 | Candidate p50 / p95 | Recall@10, unchanged |
|---|---:|---:|---:|
| 3k×128D L2 batch | 2.912500 / 3.003000 | 3.406500 / 3.519000 | 0.981 |
| 3k×128D L2 ordinary INSERT | 2.912000 / 3.015100 | 3.402000 / 3.622100 | 0.980 |
| 3k×128D cosine batch | 2.860000 / 2.956100 | 3.346000 / 3.447050 | 0.985 |
| 10k×128D L2 file | 5.139000 / 5.800900 | 7.166000 / 7.530350 | 0.892 |
| 10k×768D L2 file | 9.801000 / 10.875050 | 11.698000 / 12.429350 | 0.713 |

Read-only searches do not use the pruning buffer pool. Their slowdown is additional evidence that the regression cannot be reduced to the pool-growth issue alone.

| Workload | Recall@10 window 10 | Window 50 | Window 100 |
|---|---:|---:|---:|
| 3k×128D L2 batch | 0.566 | 0.918 | 0.981 |
| 3k×128D ordinary INSERT | 0.569 | 0.922 | 0.980 |
| 3k×128D cosine batch | 0.592 | 0.934 | 0.985 |
| 10k×128D L2 file | 0.364 | 0.758 | 0.892 |
| 10k×768D L2 file | 0.240 | 0.540 | 0.713 |

Every successful final run matches its baseline's ordered labels and distance bits as well as recall. This is unchanged baseline ANN quality, not exact nearest-neighbor recall or a quality improvement.

### Storage sizes

Median post-timing SQLite backup sizes, MiB:

| Workload | Baseline | Candidate |
|---|---:|---:|
| 3k×128D L2 batch | 3.250 | 2.801 |
| 3k×128D ordinary INSERT | 3.273 | 2.801 |
| 3k×128D cosine batch | 3.203 | 2.801 |
| 10k×128D L2 file | 10.758 | 9.293 |
| 10k×768D L2 file | 39.285 | 42.852 |

Original file databases matched their backups logically. Median original file sizes were 11,280,384 → 9,744,384 bytes for 128D, and 41,193,472 → 44,933,120 bytes for 768D. The smaller 128D file did not produce faster builds or queries. These are allocated database sizes, not an attribution of individual copying/journaling costs.

## Initial r2 negative campaign, retained

The first campaign was intentionally stopped under the >5% regression rule. [Raw partial results](<../../../build/benchmarks/garnet-20261110/matched-128/results.json>) remain unchanged; the [cancellation record](<../../../build/benchmarks/garnet-20261110/matched-128/cancellation.json>) explains the interrupted `running` status and preserves 40 validated completed builds and one queued, unstarted run. No correctness gate failed. The primary 10k case and 768D case had not started then.

| Workload, 3k×128D in memory | Measured pairs | Baseline median s | r2 median s | More time |
|---|---:|---:|---:|---:|
| L2 batch8 | 5 | 7.945212 | 9.720126 | +22.34% |
| L2 ordinary INSERT | 5 | 8.524004 | 10.651722 | +24.96% |
| Cosine batch8 — incomplete workload | 4 | 7.838012 | 9.5836435 | +22.27% |

A functional probe then found 750 capacity-growth events for 750 retained pool slots in r2's exact-one-slot push path. Geometric optional growth with exact pressure fallback reduces that probe to nine events and capacity 1,024; the 160-byte test falls back to one slot and releases all leases. These are **capacity-growth observations, not guaranteed relocations or proof of the whole wall-time cause**. The reviewer approved r3 before its fresh complete campaign. The final negative results must not be hidden by that allocation-count improvement or by mixing r2/r3 samples.

## Separate instrumented SQL diagnostics (frozen r2)

[Diagnostic analysis](<../../../build/benchmarks/garnet-20261110/diagnostic-sql-profile-20261110-2330/analysis.md>) and [counter data](<../../../build/benchmarks/garnet-20261110/diagnostic-sql-profile-20261110-2330/comparison.instrumented.json>) use a separate traced host on the same baseline/r2 libraries and 3k×128D L2 batch8 inputs. They are **not untraced performance measurements** and are not mixed into build ratios.

| Inner SQL category | Baseline executions | r2 executions | Baseline VM steps/execution | r2 VM steps/execution |
|---|---:|---:|---:|---:|
| Identity | 10,592,885 | 10,592,417 | 10 | 17 |
| Vector | 4,146,452 | 4,020,540 | 26 | 33 |
| Neighbors | 454,557 | 381,878 | 26 | 33 |

Total inner executions fell **1.26%** (15,295,524 → 15,102,468), but VM steps rose **43.79%** (228,897,872 → 329,130,075). Identity lookups account for 74,142,239 of the extra 100,232,203 steps, principally the added paired-key join. New vector inserts are only 3,001 calls/63,021 steps. There is no population scan in the point-read plans: this is extra per-lookup work.

This demonstrates substantial added SQL bytecode work independent of pool growth. PROFILE nanoseconds are coarse/quantized, instrumentation changes costs, row callbacks include work between steps, and outer INSERT/carrier times overlap. They do not establish exclusive CPU attribution or a precise share of untraced wall-time regression. Diagnostic labels, distance bits and recall match the untraced outputs exactly. These counters describe frozen r2, not a separately measured r3 profile.

## Supplemental WAL update/delete/consolidation measurements

[Raw evidence](<../../../build/benchmarks/garnet-20261110/supplemental-r3-wal/results.json>) · [Validated summary](<../../../build/benchmarks/garnet-20261110/supplemental-r3-wal/summary.json>)

These six runs occurred **after the full primary matrix ended**, in three alternating baseline/candidate pairs using the existing reviewed streamed subprocess workers. This is a separate workload: **1,000×128D signed seeded float32 L2**, ordinary INSERTs in transactions of up to 128, **16 UPDATEs, 16 DELETEs, then one consolidation**. Degree/build list/alpha/adapter budget/max visits stay 32/100/1.2/64 MiB/65,536, but the host is Python SQLite **3.50.4**, pager cache **8 MiB**, mmap 0, journal **WAL**, search list 64, and 16 queries. Do not mix these timings with the native matrix.

Median times across three runs per variant:

| Stage | Baseline s | Candidate s | More time |
|---|---:|---:|---:|
| 16 updates | 0.053166 | 0.067321 | +26.62% |
| 16 deletes | 0.002928 | 0.002517 | −14.04% |
| Consolidate remaining 984 live vectors | 1.779837 | 2.194905 | +23.32% |

The tiny delete timings are noisy and do not compensate for build/update/consolidation regressions. All six runs matched canonical graph/vector/counters **before and after mutation**, with identical ordered query labels/distances and recall. Recall@10 was 0.99375 at both stages for both variants; final graphs have 984 live vectors, one frozen root and zero tombstones.

Median final database size was **2,846,720 → 2,555,904 bytes**. The connection-open post-mutation WAL file was **3,831,632 → 2,719,232 bytes**, with both WAL files absent after connection closure/checkpoint. These WAL figures are allocated/high-water file sizes under autocheckpoint, **not bytes written or exclusive write-amplification metrics**; final database size can include reusable free pages. Before-mutation graph/identity/vector fingerprint and after-mutation fingerprint both match across variants independently.

## Recommendation

**Do not merge this format-4 bundle as a build-speed improvement.** Correctness/review success is not a performance win. All final workload regressions and the rejected initial campaign are retained; the declared primary target is missed.

A separately approved next design could keep checked identity and canonical vector payloads together while separating frequently rewritten adjacency; it must retain the same coherence, corruption-detection and rollback guarantees when SQL fault triggers mutate those payloads. That may avoid paying an added payload-key join on every identity read. This is a hypothesis suggested by measured VM-step overhead, not an implemented or benchmarked alternative, and no new speedup is promised. Isolating access/pooling changes from storage-layout changes would also help attribute effects; current bundled measurements do not provide that ablation.

## Limitations and delivery state

Execution is on the local macOS arm64 host. Passing its tests and benchmark does not establish Linux/Windows performance or actual macOS 11 execution. Warm OS caches, small samples, thermal/frequency/host-load variation and per-pair ordering remain limitations; no physical cold-disk or beyond-RAM claim is made. The adapter budget is not total process RSS. All test/benchmark Python commands use the project venv.

Large raw databases, source snapshots and validation logs are ignored local artifacts; links to them are local evidence, not data committed with this report. Source, plan and report changes remain **uncommitted and unpushed**. Format 4 deliberately rejects earlier stored indexes; there is no automatic migration. The existing pre-change baseline library was preserved, and no existing user index was silently rewritten.
