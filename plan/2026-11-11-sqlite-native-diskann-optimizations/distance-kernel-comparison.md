# Highway versus DiskANN 0.60.0 float32 distance kernels

Date: 2026-11-12

## Conclusion

**No general Highway speed advantage was demonstrated on this Apple M4.**
DiskANN's production native provider is substantially faster at 32D, faster at
128D, slightly faster around 129D/384D, and mostly near parity with Highway at
768D–3000D. Highway's few-percent wins in some larger-dimensional cases do not
support a blanket claim that our distance calculation is faster.

For the 64-byte-aligned, 16-vector corpus, median paired comparisons show:

- **128D L2:** Highway takes about **23% more time** (8.226 vs 6.714 ns).
- **128D normalized cosine:** Highway takes about **17% more time** (7.306 vs 6.262 ns).
- **768D L2:** essentially tied (40.353 vs 40.129 ns).
- **768D normalized cosine:** Highway is about **3% faster** (37.897 vs 39.022 ns),
  but the 4,096-vector case is essentially tied.

These are callable-entry-point microbenchmarks, including their different real
call/dispatch boundaries. The production integration was **not** switched to the
native provider. Numerical results are close but not bit-identical; any proposed
switch would need an end-to-end graph/recall/transactional regression campaign.

## Primary results

Pooled medians from two full campaigns (16 pairs per cell), 64-byte-aligned inputs.
`D/H` is the median paired DiskANN-time / Highway-time ratio; it is not necessarily
the ratio of the two displayed timing medians. **D/H > 1 favors Highway.**

| Dimensions | Metric | Corpus vectors | Highway ns/distance | DiskANN ns/distance | Paired D/H |
|---:|---|---:|---:|---:|---:|
| 32 | L2 squared | 16 | 5.205 | 2.328 | 0.4505 |
| 32 | Normalized cosine | 16 | 3.807 | 2.211 | 0.5865 |
| 128 | L2 squared | 16 | 8.226 | 6.714 | 0.8127 |
| 128 | Normalized cosine | 16 | 7.306 | 6.262 | 0.8560 |
| 129 | L2 squared | 16 | 8.511 | 8.142 | 0.9554 |
| 129 | Normalized cosine | 16 | 7.664 | 7.276 | 0.9475 |
| 384 | L2 squared | 16 | 19.851 | 19.260 | 0.9676 |
| 384 | Normalized cosine | 16 | 19.674 | 19.091 | 0.9659 |
| 768 | L2 squared | 16 | 40.353 | 40.129 | 0.9994 |
| 768 | Normalized cosine | 16 | 37.897 | 39.022 | 1.0300 |
| 1536 | L2 squared | 16 | 80.673 | 80.913 | 1.0021 |
| 1536 | Normalized cosine | 16 | 75.659 | 75.419 | 0.9963 |
| 3000 | L2 squared | 16 | 171.461 | 171.009 | 1.0011 |
| 3000 | Normalized cosine | 16 | 165.648 | 164.792 | 0.9967 |
| 128 | L2 squared | 4096 | 10.267 | 9.603 | 0.9351 |
| 128 | Normalized cosine | 4096 | 10.738 | 9.398 | 0.8803 |
| 768 | L2 squared | 4096 | 55.578 | 55.856 | 0.9989 |
| 768 | Normalized cosine | 4096 | 53.665 | 53.914 | 0.9971 |
| 1536 | L2 squared | 4096 | 219.685 | 230.337 | 1.0107 |
| 1536 | Normalized cosine | 4096 | 183.239 | 186.211 | 1.0139 |
| 3000 | L2 squared | 4096 | 434.082 | 429.420 | 1.0022 |
| 3000 | Normalized cosine | 4096 | 411.504 | 410.345 | 1.0012 |

Across all 56 cells (including 4-byte-offset layouts), Highway's median paired
ratio exceeded 1 in only 11 cells, primarily near parity. This count is descriptive,
not a statistical significance test. Some larger-corpus samples have substantial
spread (for example 1536D aligned L2: paired D/H 0.569–1.555), reinforcing that tiny
wins should not be overinterpreted. The complete matrix with observed ranges is
included below, so the selected table does not hide other layouts or outcomes.

Local evidence: [first campaign](<../../build/kernel-comparison/native-r1.json>),
[second campaign](<../../build/kernel-comparison/native-r2.json>),
[validated combined summary](<../../build/kernel-comparison/native-summary.json>),
and [generated full table](<../../build/kernel-comparison/native-combined-table.md>).

## Default-target control

A separate optimized build with **empty `RUSTFLAGS`** measured 128D and 768D using
the same 16 combinations of metrics, layouts, and corpora, eight pairs per case.
Both `neon` and `dotprod` were enabled in this default Apple target, and Highway
again selected NEON_BF16. These samples are **not pooled** with the native-tuned
campaigns. The key conclusion reproduced without `target-cpu=native`:

| Dimensions | Metric | Corpus vectors | Highway ns/distance | DiskANN ns/distance | Paired D/H |
|---:|---|---:|---:|---:|---:|
| 128 | L2 squared | 16 | 8.277 | 6.811 | 0.8226 |
| 128 | Normalized cosine | 16 | 7.346 | 6.208 | 0.8475 |
| 128 | L2 squared | 4096 | 10.521 | 9.653 | 0.9119 |
| 128 | Normalized cosine | 4096 | 10.634 | 9.437 | 0.8778 |
| 768 | L2 squared | 16 | 39.883 | 39.883 | 0.9978 |
| 768 | Normalized cosine | 16 | 38.493 | 39.283 | 1.0207 |
| 768 | L2 squared | 4096 | 55.143 | 56.315 | 1.0248 |
| 768 | Normalized cosine | 4096 | 51.846 | 52.265 | 1.0005 |

Raw control: [default-r1.json](<../../build/kernel-comparison/default-r1.json>),
[full control table](<../../build/kernel-comparison/default-table.md>).
Default executable SHA-256:
`d3dfe19fe5798a94c50ff7b370a4aa571cbc2332a800ca0e54d46894c7b4135c`.
The control adds 128 pairs / 256 samples: **1,024 measured pairs / 2,048 samples**
across the two main campaigns and this control, excluding calibration and smoke tests.

## Scope

This measures the callable distance entry points, **not SQLite or end-to-end ANN
performance**. The production integration remains unchanged: DiskANN's graph
algorithms call vectorlite's Highway-backed distance adapter. The earlier
[SQLite optimization results](<results.md>)
compared two implementations that both used Highway; those results cannot prove
Highway is faster than DiskANN's own distance kernels.

The reproducible [Rust benchmark](<../../benchmark/distance_kernels.rs>) compares:

- **Highway:** the exact production `vl_ops_l2_sq_f32` and
  `vl_ops_ip_dist_f32` C ABI functions, built from the unmodified production native
  sources. The Rust length assertion mirrors the production safe wrapper.
- **DiskANN:** pinned `diskann-vector = 0.60.0`, using
  `f32::distance_comparer(metric, Some(dimension))`, constructed outside timing.
  This is the production provider with architecture dispatch and its available
  dimension specializations, not a scalar reference implementation.

Both compute squared L2, or `1 - dot(a,b)` on identically pre-normalized vectors.
DiskANN's full `Metric::Cosine` would additionally compute norms on every call;
using it here would unfairly charge only DiskANN for work vectorlite already
performs outside the distance kernel. `Metric::InnerProduct` returns `-dot`, so the
matching native comparator is `Metric::CosineNormalized`.

## Host and build

- Apple M4, arm64, 10 logical CPUs, 16 GiB RAM; macOS 27.0.1.
- Rust 1.98.1, LLVM 22.1.8; Apple Clang 21.0.0.
- Cargo optimized bench profile; main campaign uses `RUSTFLAGS='-C target-cpu=native'`.
- `MACOSX_DEPLOYMENT_TARGET=11.0` for compilation; actual execution is macOS 27,
  not a macOS 11 compatibility test.
- Rust's native flag is not an assertion that the C++ compiler used identical
  CPU flags. Highway retains the production C++ build and dynamic dispatch.

DiskANN 0.60.0's ARM64 architecture selection requires both `neon` and `dotprod`
at **compile time**, including through `DistanceProvider`. We verified both are
enabled in the main binary. **This host's current default Rust Apple target also
enables both**, so its default build is not an emulated/scalar baseline. This
must be checked rather than inferred from the CPU alone on other toolchains.

Local compiler/hardware/source provenance is recorded in
[provenance.json](<../../build/kernel-comparison/provenance.json>).

## Measurement method

- Dimensions: 32, 128, 129, 384, 768, 1536, 3000. The odd dimension exercises tails;
  128, 384, and 768 also exercise the provider's built-in dimension specializations.
- Metrics: squared L2 and normalized cosine (`1 - dot`).
- Input layouts: each vector starts at a 64-byte boundary or 4 bytes after it.
  Both remain valid aligned `f32` slices; the second is not 16/64-byte aligned.
- Access patterns: one fixed query and a deterministic shuffled traversal of 16
  or 4,096 distinct corpus vectors. Query and corpus use separate buffers to avoid
  Highway's pointer-equality self-distance shortcut. Rows use a padded 64-byte stride.
- Identical seed-42 float32 inputs are supplied to both paths. Cosine normalization
  uses Highway once outside timing. A stable FNV-1a input/order fingerprint checks
  repeated inputs; it is not a cryptographic integrity hash.
- Warm both paths and calibrate a **common operation count** against the faster
  implementation, targeting 100 ms per sample. Actual later durations can differ
  with frequency/load. Each case has eight alternating AB/BA measured pairs.
- Two sequential, fresh-process full campaigns: 56 cases per campaign, 448 paired
  measurements each, 896 pairs / 1,792 timed samples combined. No other benchmark
  or build runs concurrently with the campaign processes.
- Black-box both input slices and the output on every call; accumulate an f64
  checksum to consume results. No input generation, normalization, allocation,
  reference calculation, JSON writing, or SQLite call is inside the timed loop.
- **Timing still includes** real entry-point/dispatch overhead, slice addressing,
  black-box barriers and checksum accumulation. It is not an attribution of pure
  SIMD instruction execution time. No empty-loop subtraction is performed.
- The 4,096-vector corpora range from 0.5 MiB at 32D to 46.875 MiB at 3000D before
  padding; repeated traversals warm caches. Neither corpus is labelled cold-disk,
  cold-memory, or guaranteed L1/L2-resident. No core pinning or frequency lock is used.

Report medians of timings separately from medians of within-pair ratios. A ratio
`DiskANN time / Highway time > 1` favors Highway; `< 1` favors DiskANN. Observed
min–max spreads are not statistical confidence intervals, and repeated pairs are
not assumed independent.

## Numerical checks

Before timing each case, the Rust benchmark checks **every query/corpus pair**
against an f64 oracle calculated from the exact stored f32 inputs, requiring:

```text
abs(f32 result - f64 reference) <= 2e-6 * max(1, abs(f64 reference))
```

It also checks self, opposite, near-identical, and zero inputs. For normalized
cosine, the near/zero probes test the raw `1 - dot` kernel semantics, not valid
unit-vector cosine preprocessing. No clamping is added. Slightly negative
self-distances from f32 normalization/rounding are permitted by both implementations.

Cross-implementation bit equality is measured and reported, **not required or
assumed**. Different reduction/FMA order can change low bits. Passing this bounded
numerical test does not prove that replacing the production kernel would preserve
bit-exact ANN graphs, tie ordering, or recall.

Each full campaign passed **115,360 numerical comparisons**. The second campaign
reproduced the exact input fingerprints and numerical error summaries of the first.
These are repeated checks of the same deterministic fixtures, not independent
random datasets:

| Metric | Comparisons/campaign | Bit mismatches | Max H–D absolute difference | Max Highway f64-reference error | Max DiskANN f64-reference error |
|---|---:|---:|---:|---:|---:|
| L2 squared | 57,680 | 14,212 | 0.000244140625 | 0.000370148555 | 0.000409131273 |
| Normalized cosine | 57,680 | 1,430 | 1.192092896e-7 | 1.066984383e-7 | 1.066984383e-7 |

All differences satisfy the stated mixed absolute/relative tolerance. There is no
claim of bit-exact interchangeability between native DiskANN and Highway.

## Artifact identity and benchmark review

The main campaigns used one frozen executable:

```text
native executable SHA-256:
4da348110be91fc2594f900712f94bf6f05062eca661de18311fa0c918691fc4
native-r1.json SHA-256:
73b1a4656f52290ad3ed204ced34a26779ef983e3a293eb3cbca83fdeb896d07
native-r2.json SHA-256:
b074e84aec6a957dcbc5083d224b22f3120399367984de98a8c7e52e0c3e1967
```

The selected Highway target was **NEON_BF16** (the float32 functions are still
float32 kernels), and DiskANN used its **NEON** backend. A read-only independent
GPT-6.1 Sol review checked the native API selection, semantics, black-box placement,
alignment, calibration, and numerical checks before measurements. A second review
of the summary utility led to duplicate-artifact rejection, API/version compatibility
checks, and removal of an unsupported sample-independence claim.

Strict Clippy subsequently flagged two legal native-link attributes on the same
Rust `extern` block. Splitting the declarations fixed the lint without changing
benchmark logic. The rebuilt final executable's **entire 298,864-byte Mach-O
`__TEXT,__text` machine-code section is byte-identical** to the measured executable
(SHA-256 `fd2b3e8f35a0fd393eb8f14a82988b1da93b93e75f70f7d183d686e36bc0ef2b`).
The executable file hash differs because linkage/metadata changed; it is not silently
substituted for the frozen measured artifact. The exact
[measured source snapshot](<../../build/kernel-comparison/distance_kernels.measured.rs>)
and [code identity evidence](<../../build/kernel-comparison/final-code-identity.json>)
are retained locally.

## Verification and delivery

- Root `sh build_release.sh`: **28/28 CTests and 399/399 Python tests passed**
  ([release log](<../../build/kernel-comparison/release-validation.log>)).
- Benchmark target strict Clippy (`-D warnings`, documented unsafe blocks and safety
  documentation lints): **passed**, after separating the link declarations.
  Existing native C++ unused-alias/parameter warnings remain; this is not a
  warning-free C++ claim ([Clippy log](<../../build/kernel-comparison/clippy.log>)).
- `cargo fmt --all --check`, `git diff --check`: passed.
- Artifact-independent summary validation: **5/5 unit tests passed**.
- Both full primary matrices and the control matched their intended Cartesian case
  sets; raw reports passed sample/summary validation.

Only an opt-in benchmark target, benchmark tools/tests, and documentation were
added. No production kernel, graph, storage, SQL API, or distance-routing code was
changed. At measurement handoff, these additions were **uncommitted and unpushed**;
repository publication is a separate delivery step.

## Reproduction

Configure native dependencies as documented in the [benchmark guide](<../../benchmark/README.md>).
Output JSON paths must be new:

```bash
mkdir -p build/benchmarks/distance-kernels
RUSTFLAGS='-C target-cpu=native' MACOSX_DEPLOYMENT_TARGET=11.0 \
  cargo bench --locked --bench distance_kernels -- \
  --pairs 8 --milliseconds 100 \
  --output build/benchmarks/distance-kernels/native-r1.json
RUSTFLAGS='-C target-cpu=native' MACOSX_DEPLOYMENT_TARGET=11.0 \
  cargo bench --locked --bench distance_kernels -- \
  --pairs 8 --milliseconds 100 \
  --output build/benchmarks/distance-kernels/native-r2.json
python benchmark/summarize_distance_kernels.py \
  build/benchmarks/distance-kernels/native-r1.json \
  build/benchmarks/distance-kernels/native-r2.json
python -m unittest discover -s benchmark -p test_distance_kernel_results.py
# Separate default-target control matching the report's 128D/768D matrix:
RUSTFLAGS='' MACOSX_DEPLOYMENT_TARGET=11.0 \
  cargo bench --locked --bench distance_kernels -- \
  --dimensions 128,768 --pairs 8 --milliseconds 100 \
  --output build/benchmarks/distance-kernels/default-r1.json
python benchmark/summarize_distance_kernels.py \
  build/benchmarks/distance-kernels/default-r1.json
```

The [summary utility](<../../benchmark/summarize_distance_kernels.py>) validates report
completion, case/input/API consistency, sample ordering, and derived statistics.
The Rust producer enforces numerical tolerance per input; the Python utility does
not independently recompute the vector oracle from aggregate error fields. Before
pooling, also verify external executable hashes/compiler flags match; selected
SIMD features alone do not identify a build. Raw artifacts and build outputs are
ignored local evidence, not committed datasets.

## Appendix: complete native-tuned matrix

Campaigns: 2; cases per campaign: 56; Highway target: NEON_BF16; NEON=True, dotprod=True.

Ratios >1 favor Highway; <1 favor DiskANN. Pooled paired samples; range is observed min–max, not a confidence interval.

| Dimension | Metric | Offset B | Corpus | Highway median ns | DiskANN median ns | Median paired D/H (range) |
|---|---|---:|---:|---:|---:|---:|
| 32 | cosine_normalized | 0 | 16 | 3.807 | 2.211 | 0.5865 (0.5752–0.5903) |
| 32 | cosine_normalized | 0 | 4096 | 3.573 | 3.010 | 0.8492 (0.8304–0.8615) |
| 32 | cosine_normalized | 4 | 16 | 3.802 | 2.227 | 0.5873 (0.5386–0.5950) |
| 32 | cosine_normalized | 4 | 4096 | 4.159 | 3.477 | 0.8381 (0.7319–0.9320) |
| 32 | l2_squared | 0 | 16 | 5.205 | 2.328 | 0.4505 (0.4169–0.4781) |
| 32 | l2_squared | 0 | 4096 | 4.944 | 3.066 | 0.6248 (0.6008–0.6949) |
| 32 | l2_squared | 4 | 16 | 5.134 | 2.315 | 0.4552 (0.4381–0.4734) |
| 32 | l2_squared | 4 | 4096 | 5.077 | 3.586 | 0.7119 (0.6642–0.7531) |
| 128 | cosine_normalized | 0 | 16 | 7.306 | 6.262 | 0.8560 (0.8353–0.8645) |
| 128 | cosine_normalized | 0 | 4096 | 10.738 | 9.398 | 0.8803 (0.7017–0.9153) |
| 128 | cosine_normalized | 4 | 16 | 7.378 | 6.329 | 0.8541 (0.8006–0.9082) |
| 128 | cosine_normalized | 4 | 4096 | 12.476 | 10.764 | 0.8734 (0.7332–1.1565) |
| 128 | l2_squared | 0 | 16 | 8.226 | 6.714 | 0.8127 (0.7969–0.8378) |
| 128 | l2_squared | 0 | 4096 | 10.267 | 9.603 | 0.9351 (0.8972–0.9565) |
| 128 | l2_squared | 4 | 16 | 8.263 | 6.762 | 0.8194 (0.7619–0.9375) |
| 128 | l2_squared | 4 | 4096 | 11.721 | 11.222 | 0.9571 (0.9101–1.0634) |
| 129 | cosine_normalized | 0 | 16 | 7.664 | 7.276 | 0.9475 (0.9300–0.9752) |
| 129 | cosine_normalized | 0 | 4096 | 11.343 | 10.399 | 0.9229 (0.9024–0.9631) |
| 129 | cosine_normalized | 4 | 16 | 7.728 | 7.335 | 0.9452 (0.9278–1.1777) |
| 129 | cosine_normalized | 4 | 4096 | 12.716 | 11.773 | 0.9202 (0.8852–0.9416) |
| 129 | l2_squared | 0 | 16 | 8.511 | 8.142 | 0.9554 (0.9017–0.9804) |
| 129 | l2_squared | 0 | 4096 | 11.240 | 11.192 | 0.9898 (0.9613–1.0743) |
| 129 | l2_squared | 4 | 16 | 8.398 | 8.165 | 0.9767 (0.9414–1.0092) |
| 129 | l2_squared | 4 | 4096 | 12.331 | 12.170 | 0.9790 (0.9541–1.0143) |
| 384 | cosine_normalized | 0 | 16 | 19.674 | 19.091 | 0.9659 (0.9509–0.9971) |
| 384 | cosine_normalized | 0 | 4096 | 28.343 | 27.274 | 0.9552 (0.9327–1.0625) |
| 384 | cosine_normalized | 4 | 16 | 19.778 | 19.450 | 0.9854 (0.9673–1.1378) |
| 384 | cosine_normalized | 4 | 4096 | 32.041 | 30.114 | 0.9456 (0.7984–1.0015) |
| 384 | l2_squared | 0 | 16 | 19.851 | 19.260 | 0.9676 (0.9563–0.9816) |
| 384 | l2_squared | 0 | 4096 | 28.420 | 27.781 | 0.9650 (0.6118–1.0335) |
| 384 | l2_squared | 4 | 16 | 20.088 | 19.546 | 0.9730 (0.9585–0.9978) |
| 384 | l2_squared | 4 | 4096 | 31.208 | 31.002 | 0.9905 (0.9382–1.1663) |
| 768 | cosine_normalized | 0 | 16 | 37.897 | 39.022 | 1.0300 (0.8395–1.0518) |
| 768 | cosine_normalized | 0 | 4096 | 53.665 | 53.914 | 0.9971 (0.9637–1.0262) |
| 768 | cosine_normalized | 4 | 16 | 38.191 | 39.438 | 1.0372 (1.0192–1.0500) |
| 768 | cosine_normalized | 4 | 4096 | 60.272 | 61.190 | 0.9915 (0.8672–1.2562) |
| 768 | l2_squared | 0 | 16 | 40.353 | 40.129 | 0.9994 (0.9824–1.2563) |
| 768 | l2_squared | 0 | 4096 | 55.578 | 55.856 | 0.9989 (0.9635–1.3469) |
| 768 | l2_squared | 4 | 16 | 40.906 | 40.772 | 0.9985 (0.9876–1.0100) |
| 768 | l2_squared | 4 | 4096 | 61.887 | 61.222 | 0.9927 (0.9222–1.1277) |
| 1536 | cosine_normalized | 0 | 16 | 75.659 | 75.419 | 0.9963 (0.9904–1.0707) |
| 1536 | cosine_normalized | 0 | 4096 | 183.239 | 186.211 | 1.0139 (0.9218–1.4611) |
| 1536 | cosine_normalized | 4 | 16 | 76.201 | 75.398 | 0.9934 (0.7472–1.0031) |
| 1536 | cosine_normalized | 4 | 4096 | 193.837 | 189.797 | 0.9856 (0.7114–1.1902) |
| 1536 | l2_squared | 0 | 16 | 80.673 | 80.913 | 1.0021 (0.8208–1.0211) |
| 1536 | l2_squared | 0 | 4096 | 219.685 | 230.337 | 1.0107 (0.5691–1.5554) |
| 1536 | l2_squared | 4 | 16 | 82.174 | 82.566 | 1.0054 (0.9900–1.0232) |
| 1536 | l2_squared | 4 | 4096 | 202.783 | 201.131 | 0.9989 (0.7214–1.3420) |
| 3000 | cosine_normalized | 0 | 16 | 165.648 | 164.792 | 0.9967 (0.8470–1.0094) |
| 3000 | cosine_normalized | 0 | 4096 | 411.504 | 410.345 | 1.0012 (0.9084–1.0369) |
| 3000 | cosine_normalized | 4 | 16 | 172.061 | 171.956 | 0.9952 (0.8782–1.2712) |
| 3000 | cosine_normalized | 4 | 4096 | 471.115 | 454.559 | 1.0144 (0.8525–1.2296) |
| 3000 | l2_squared | 0 | 16 | 171.461 | 171.009 | 1.0011 (0.9910–1.0091) |
| 3000 | l2_squared | 0 | 4096 | 434.082 | 429.420 | 1.0022 (0.9034–1.2252) |
| 3000 | l2_squared | 4 | 16 | 175.001 | 176.861 | 1.0065 (0.7602–1.2434) |
| 3000 | l2_squared | 4 | 4096 | 438.017 | 434.683 | 0.9981 (0.9161–1.0955) |
