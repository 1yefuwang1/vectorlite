# Format-preserving DiskANN insertion results

Date: 2026-11-14 (Asia/Shanghai)
Plan: [v2 canonical plan](<2026-11-13-diskann-insertion-performance.md>) with explicit user waiver W1.
Delivery status: **COMPLETE under explicit user waiver W1; all three integrated reviewers APPROVED with no blockers.** Verification and review were completed before the user’s separate commit-and-push request.

Raw-evidence links below refer to ignored local build artifacts preserved in the original workspace; those large artifacts are not included in the Git commit.

## Outcome and scope

- Retain the checked stored-vector proof path. Aligned little-endian SQL BLOBs are validated once per borrowed value, with core dimension checking and cleanup-error precedence preserved. Generic/copy fallbacks retain validation; no persistent payload cache or storage-format change.
- Reject the identity-epoch and bulk-encoder candidates under the predeclared independent screening rule. Identity-cache source is restored to baseline; bulk code was never applied to main. Their isolated code/tests/binaries and failed-attempt evidence remain available.
- The original default-pager primary target **was not met**: the valid five-pair result was **1.031279x**, not >=1.10x. The user explicitly accepted the smaller gain and waived only that numerical target. All exact-result, regression, safety, compatibility and final-review gates remain in force. [Waiver W1](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/user-waiver-W1.json>).
- Host-owned pager tuning produced a larger effect on these bounded datasets. It is measured separately with the **same baseline library on both sides** and is not substituted for code speedup. Extension defaults and application durability settings are unchanged.

## Fixed controls and provenance

Same immutable CMake-built native host/SQLite 3.51.2 for all acceptance measurements; Rust/Cargo 1.98.1, release `opt-level=3`, `panic=abort`, aarch64-apple-darwin, MACOSX_DEPLOYMENT_TARGET=11.0. Local execution is macOS arm64 only. Native CPU/OS/Python/NumPy/SQLite settings and compile options are captured in each raw report.

Fixed `degree=32`, `build_list_size=100`, `search_list_size=64`, alpha float32 1.2, adapter `cache_bytes=67108864`, `max_visits=65536`, `seed=42`, `k=10`; query windows 10/50/100 with 100 timed queries each after one complete untimed pass. Native build timer covers BEGIN, INSERT prepare/bind/step/finalize and COMMIT; CREATE/input generation, queries, snapshot/hash and final WAL checkpoint are outside it. Queries use the same open, build-warmed connection. WAL uses FULL durability and `wal_autocheckpoint=0`, and a separately timed final FULL checkpoint after queries.

| Artifact | SHA-256 |
|---|---|
| Source baseline HEAD | c2fababe2c56e37c4acbc4d6cdf0a6115d07c5cf |
| Fresh baseline library | 0fc529fb4ec8d386d3611ac6abd8bda42fc0d6217a6bb7906b54c56c0bc6442c |
| Final retained root-built library | 9918fc7357f519ccbd715dbf92b79dce4f09654894605daf376d904a841f9f10 |
| Acceptance host | 11f43989aa04f702dc13cbbd182e5f24fb3fca0b69ad14297de8171f4add401f |
| Matched runner | 70e0fb7cbfb60cd6d77f5fb341260877a92517693cea89169ac2c9cfff07dab3 |

The pre-implementation tree had no staged/unstaged changes and only this engagement's canonical plan untracked. [Baseline metadata/archive provenance](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/pre-implementation-baseline.json>) and per-candidate handoffs distinguish all source ownership.

## Independent candidate screening

Three alternating AB/BA measured pairs per each of five cells, two baseline selfchecks and one warmup per variant/cell: 150 native builds, 45 measured pairs. Every run passed exact ordered Format-3 graph/metadata (excluding only random instance_id), labels, float32 distance bits and float64-oracle recall checks.

| Candidate | Primary paired median | Disposition |
|---|---:|---|
| T2 validated-vector proof | 1.030965x | retain for integration; no screening flags |
| T3 identity epochs | 0.858560x | reject: no primary benefit and regression flags |
| T4 scalar encoder reuse | 0.987255x | reject: no primary benefit and query regression flags |

These are local screening estimates, not an uncontended causal ranking of cache algorithms. All pairs/outliers remain recorded. [Dispositions](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/screening/dispositions.json>) / [complete screening ledger](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/screening/controller.json>).

## Final default-pager code comparisons

Five alternating measured AB/BA pairs per cell; selfchecks/warmups excluded from timing summaries. The three memory cells use 3,000 vectors; file cells use 10,000. File journal modes are DELETE/FULL and WAL/FULL. Both binaries have identical read-back connection-default pager controls, including cache_size=-2000 KiB on this host.

**Speedup is the median of per-pair baseline/candidate ratios, not the ratio of separate medians.** The last column is the worst median candidate/baseline query ratio across all three windows and both p50/p95 metrics; lower is better. Percentiles use per-process numpy.percentile(method=linear), then paired ratios—not pooled samples.

| Workload | Baseline median s | Retained median s | Paired median speedup | Worst query median ratio |
|---|---:|---:|---:|---:|
| 128D l2-file / delete | 30.793 | 29.577 | **1.0313x** | 1.0034 |
| 128D l2-batch / memory | 4.169 | 4.032 | **1.0329x** | 0.9852 |
| 128D l2-single / memory | 5.930 | 5.772 | **1.0257x** | 0.9791 |
| 128D cosine-batch / memory | 4.201 | 4.106 | **1.0356x** | 0.9833 |
| 768D l2-file / delete | 47.013 | 42.939 | **1.1023x** | 0.9174 |
| 128D l2-file / wal | 31.197 | 31.159 | **1.0167x** | 0.9969 |
| 768D l2-file / wal | 47.871 | 43.345 | **1.1044x** | 0.9179 |

All seven configurations are complete and have no first-set median regression above 5%; therefore no confirmation set was triggered. Individual pairs vary; raw ranges and samples are retained. The valid primary paired range is 0.998259x–1.086421x. Its original 1.10x target miss remains visible and is accepted only through W1. It was not rerun or replaced.

[Complete final matrix](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/final-matrix-W1.json>), [original primary](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/final-primary-t2/results.json>), and [other six configurations](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/remaining-final-w1/controller.json>).

### Separately timed final WAL checkpoint

The build timer includes COMMIT but not this post-query FULL checkpoint. Median costs from the five measured processes are reported separately; they are not physical bytes written or a power-loss test. All checkpoints completed successfully.

| File workload | Baseline checkpoint s | Retained checkpoint s |
|---|---:|---:|
| 128D WAL | 0.022783 | 0.024167 |
| 768D WAL | 0.055998 | 0.054977 |

## Baseline-only pager-cache ablation

The same baseline library is compared against itself: default -2000 KiB versus explicit 2/8/32/64 MiB pager hints. Same durability/ANN settings; three measured pairs, two selfchecks per configuration and one warmup per variant. SQLite-derived positive spill thresholds follow the cache budget while spill policy stays on. These figures are **host-budget effects, not retained-code speedups**.

| Dimension | Explicit host cache | Paired median build speedup vs default | Exploratory >5% regression flags |
|---|---:|---:|---|
| 128D | 2 MiB | **1.0001x** | query/50/p95_ms, query/100/p50_ms, query/100/p95_ms |
| 128D | 8 MiB | **1.2929x** | none |
| 128D | 32 MiB | **1.3591x** | none |
| 128D | 64 MiB | **1.3696x** | none |
| 768D | 2 MiB | **1.0058x** | none |
| 768D | 8 MiB | **1.1238x** | none |
| 768D | 32 MiB | **1.3184x** | none |
| 768D | 64 MiB | **1.3180x** | none |

Explicit 2 MiB (-2048 KiB) is not exactly the host default (-2000 KiB); its 128D query flags and negligible build gain are reported rather than hidden. The larger-budget comparisons did not flag. A user-requested pause interrupted one configuration: two fully completed sets were deep-audited and preserved, and all six unfinished/unstarted sets ran in fresh output directories. No interrupted pairs were counted. Comparisons across budgets span that pause; use within-configuration paired ratios, not uncontrolled absolute times across cohorts.

[Complete eight-configuration record](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/pager-grid-complete.json>), [resume/deep audit](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/pager-resume-deep-audit-20261114.json>).

### Code verification at the 32 MiB documentation example

One bounded common host example—32 MiB—was predeclared for both file dimensions, not chosen to satisfy the waived target. Baseline and retained code receive identical 32 MiB controls; five pairs per cell.

| Dimension | Paired median code speedup at 32 MiB | Worst query median ratio | Regression flags |
|---|---:|---:|---|
| 128D | 1.0358x | 0.9980 | none |
| 768D | 1.1386x | 0.8865 | none |

These two sets also passed exact-result validation and all unchanged regression gates. Do not multiply independent medians across experiments and present that as a directly measured combined speedup. [Predeclaration](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/tuned32-predeclared.json>) / [results](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/tuned32-final-summary.json>).

## Separate resource diagnostics

One diagnostic-only run per row, using a distinct host with counter reads. None of its timings enter acceptance summaries. Peak RSS is macOS time -l maximum resident set size in bytes, converted to MiB below; SQLite cache-used bytes are captured after COMMIT. Pager hit/miss/write counters reset before BEGIN and are logical SQLite events, **not physical bytes written**. Counts can vary with legal write ordering; no precise code-level I/O attribution is claimed.

| Dimension | Host cache | Variant | Peak RSS MiB | SQLite cache-used MiB | Pager misses | Pager writes |
|---|---|---|---:|---:|---:|---:|
| 128D | default | baseline | 15.88 | 2.01 | 17,811,886 | 155,329 |
| 128D | default | retained | 15.89 | 2.01 | 17,813,317 | 155,729 |
| 128D | 32 MiB | baseline | 27.36 | 11.42 | 0 | 2,746 |
| 128D | 32 MiB | retained | 27.33 | 11.42 | 0 | 2,745 |
| 768D | default | baseline | 40.91 | 2.01 | 26,120,290 | 225,416 |
| 768D | default | retained | 41.02 | 2.01 | 26,123,878 | 225,534 |
| 768D | 32 MiB | baseline | 78.48 | 32.91 | 94,097 | 10,980 |
| 768D | 32 MiB | retained | 78.44 | 32.91 | 94,088 | 10,977 |

The 32 MiB hint is not a hard RSS cap: notably, the 768D retained process peaked at 78.44 MiB while SQLite reported 32.91 MiB cache usage. Adapter cache_bytes is also not a process-wide RSS cap. The measurements include caller input, SQLite/runtime/allocator costs and backup/query phases; they are illustrative resource observations, not a memory limit guarantee.

[All diagnostic runs, raw output/time logs and exact graphs](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/diagnostics/final-20261114/results.json>).

## Correctness, compatibility and failure evidence

- Retained root release build: **28/28 CTests and 620/620 Python tests** passed, with the deployed artifact hash matching the preserved measured library. rustfmt and strict Clippy passed. [Root build](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/retained-t2-integration/build-release.log>).
- T2 baseline/candidate SQL parity: 219 cases each, including 82 new malformed-payload/schema-guard cases; 135 Rust library + 5 build-support tests; proof lifetime/forgery/callback compile-fail checks. [Handoff](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/t2-work/handoff.json>). Initial 20 new-test failures reproduced identically on baseline; only test assumptions were corrected, preserving NOTNULL 1299 versus CORRUPT 11 and actual poison-hook execution. Initial failed evidence remains preserved.
- Fresh native hosts: 26/26 pointer-batch checks each on SQLite 3.38.5 and 3.51.2 using the final measured library. Product floor remains 3.38.0; 3.38.5 is the selected 3.38-series execution test. [Verification](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/final-compat-native-20261114/verification.json>).
- Node 22.23.4: unchanged npm test passed platform/notices/node:sqlite and better-sqlite3 HNSW/DiskANN transaction/rollback/reopen suites. Separate host probes verified the measured dylib hash with SQLite 3.49.2 and 3.53.4. The pre-existing Node artifact was restored exactly. [Node handoff](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/compatibility-node-20261114/run-p_q7fua9/handoff.json>). An initial extra probe incorrectly tried different SQLite runtimes in one process; the existing guard rejected it, and only the evidence-local probe was corrected. No runtime guard or npm-suite source was changed.
- Original callback-thread/runtime/scoped-token, whole-INSERT rollback, severe SQLite transaction-abort behavior, primary/extended error precedence, visit/budget/tiny-admission and reentrant corruption contracts remain covered. No new unsafe Rust implementation or SQLite API-table dependency was added to the extension.

## Limitations and delivery gate

These are small, deterministic, warm-OS-cache workloads on one macOS arm64 host. A file larger than the SQLite pager does not establish a physical beyond-RAM or cold-disk result. External/user contention could not be fully audited (process listing was denied); intentional builds/tests/review panels were stopped during acceptance timing. No Linux/Windows or actual macOS 11 execution, crash/power-loss simulation, physical write-amplification estimate or universal speed promise is claimed.

Format 3, SQL/batch/package APIs and extension filenames are unchanged. No application PRAGMAs are changed by the extension. At verification completion, source was uncommitted and had not been pushed or published. All three original independent reviewers approved the complete integrated code, documentation and evidence under W1: **GPT-6.1 Sol — APPROVED; Gemini 4 Carbon — APPROVED; Grok 4.7 — APPROVED. No blockers.** These are integrated implementation verdicts, not the earlier plan approvals.

[Detailed integrated receipts](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/final-verification-20261114/integrated-review-verdicts.json>) record all three explicit approvals, evidence and residual limitations.

The exactly reviewed technical [plan](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/final-verification-20261114/reviewed-plan-v2-W1.md>) and [results snapshot](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/final-verification-20261114/reviewed-results.md>) remain preserved. Subsequent status/receipt edits are metadata only. Non-blocking manifest labeling notes were corrected without changing any metric, with the [reviewed matrix retained](<../build/benchmarks/diskann-insert-20261113-1905.8jDmVI/final-matrix-W1.reviewed-snapshot.json>).

Remaining documented tradeoffs: the uncommon misaligned/big-endian copy fallback conservatively revalidates; private generic vector-only stores may make an intermediate Vec copy, unlike the production borrowed SQL path. These were reviewed as non-blocking. The exploratory 2 MiB query flags remain visible and that budget is not recommended by these results. Raw local evidence occupies about 16 GB and remains intact; no automatic cleanup was performed.
