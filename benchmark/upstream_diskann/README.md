# Upstream DiskANN build comparison driver

This standalone driver uses the **unmodified upstream Rust DiskANN 0.60.0** memory
provider and full SSD-format build pipeline. It is not part of vectorlite's Cargo
workspace or production extension. The [comparison plan](<../../docs/plans/2026-11-12-upstream-diskann-build-comparison/plan.md>)
records scope and interpretation limits.

## Build

Obtain the source archive for revision `97a828a500848018d8be28c3ea6d5a585b07362f`
from [GitHub codeload](<https://codeload.github.com/microsoft/DiskANN/tar.gz/97a828a500848018d8be28c3ea6d5a585b07362f>).
The measured archive SHA-256 is
`88ea51a9db42d483b330d1be1ea764d0fa8ac7e39876aa9bb02878bd0c1377c1`.
Extract into a new directory without changing upstream source files.

```bash
# New build directory; --upstream-source is the extracted pinned repository root.
.venv/bin/python benchmark/build_upstream_diskann.py \
  --upstream-source .cache/diskann-upstream-0600/DiskANN-97a828a500848018d8be28c3ea6d5a585b07362f \
  --source-archive .cache/diskann-upstream-0600/source.tar.gz \
  --build-dir build/upstream-comparison-new \
  --cargo-home .cache/cargo-diskann
```

The [build helper](<../build_upstream_diskann.py>) generates a standalone manifest,
uses this directory's [dependency lock](<Cargo.lock>), and records executable,
manifest and lock fingerprints. It uses optimized Rust with one codegen unit and
empty `RUSTFLAGS`; verify that the emitted JSON has ARM `neon` and `dotprod` enabled
on an ARM host. No SIMD fallback should silently stand in for the native provider.
Upstream's relevant linear algebra uses pure-Rust faer; no BLAS installation is
needed. Default macOS deployment target is 11.0, but actual execution support must
be verified on the machine used for measurement.

## Run

Build the regular vectorlite release extension and opt-in native host first. Choose
a **new** campaign output directory. Processes run sequentially, with one warm-up
and three measured repetitions per backend/workload by default:

```bash
.venv/bin/python benchmark/upstream_diskann_benchmark.py \
  --upstream-executable build/upstream-comparison-new/target/release/upstream_diskann_benchmark \
  --sqlite-executable build/release/vectorlite/diskann_batch_benchmark \
  --extension build/release/vectorlite/vectorlite.dylib \
  --output-dir build/benchmarks/upstream-builds-new

# Small correctness-only fixture, not a full-size performance result:
.venv/bin/python benchmark/upstream_diskann_benchmark.py \
  --upstream-executable build/upstream-comparison-new/target/release/upstream_diskann_benchmark \
  --sqlite-executable build/release/vectorlite/diskann_batch_benchmark \
  --extension build/release/vectorlite/vectorlite.dylib \
  --output-dir build/benchmarks/upstream-smoke-new \
  --case 3000x128 --count 1000 --queries 16 --repeats 2 --warmups 0

.venv/bin/python -m unittest discover -s benchmark -p test_upstream_diskann_benchmark.py

# After status=complete: verify retained artifacts and reproduce timing/recall tables.
# --json-output must be a NEW path; omit it to print only the Markdown tables.
.venv/bin/python benchmark/summarize_upstream_diskann.py \
  build/benchmarks/upstream-builds-new/results.json \
  --json-output build/benchmarks/upstream-builds-new/analysis-summary.json
```

Use the platform extension suffix on Linux/Windows; local measurements are not
cross-platform performance evidence. `--case` and `--backend` can be repeated to
restrict a run. Backends are `memory`, `ssd`, `sqlite-memory`, and `sqlite-file`.
Data is seeded with NumPy RandomState(42), exactly matching the earlier SQLite
build runner. The headered upstream files and raw SQLite files have identical f32
payloads. Labels and returned distances are checked against the same f64 oracle;
recall@10 is recorded at L=10/50/100. Different backends need not produce identical
graphs or recall. Every backend must reproduce its own persisted structure and
ordered predictions on repeat; memory save/load must preserve exact predictions.
The RAM driver audits every logical stored vector and adjacency list before and
after reload. Root reachability is reported separately: upstream pruning does not
guarantee full connectivity, so unreachable vertices must not be confused with
missing or corrupt stored vectors.
Failures retain partial files/logs and stop the campaign, not silently retry.

## Interpretation

- **Memory:** standard `diskann-providers::FullPrecisionProvider<f32>`, first-vector
  frozen root, batch 8 after a single-point bootstrap, IBC 8, max minibatch parallelism
  1, one current-thread runtime. Setup, insertion and serialization are separate.
  The root ID differs from SQLite's; the distance kernels are upstream native, not Highway.
- **SSD:** actual full-precision graph builder plus mandatory search-PQ codebooks,
  compression, medoid root, sequential insertions, static final prune, and 4 KiB
  disk-layout packing. Explicit PQ chunks are `clamp(dim/4, 1, 96)` (32 at 128D, 96
  at 768D); build RAM budget is 2 GiB. These are benchmark choices, not a claimed
  universal upstream default. The total constructor/build timer includes the full
  pipeline; private stage timings are not invented.
- **SQLite:** one native batch-8 INSERT in one transaction. Timed BEGIN/INSERT/COMMIT
  excludes table creation/input loading. The extension adds corruption checks,
  workspace limits, SQL/pager work and transactional behavior absent from the
  standard RAM provider. Runtime lifetime and distance kernels also differ.
- Native saved indexes use ordinary writes/flush, with no explicit fsync added by
  this driver. SQLite uses its host library's unmodified journal/synchronous
  defaults. These are not equal durability contracts.
- On **macOS**, upstream's saved-SSD search fallback reads the entire disk index
  into a Vec. Those searches validate format/quality only, **not SSD I/O latency**.
- The small synthetic datasets and warm OS caches do not test physical beyond-RAM
  indexing, cold disks, large-scale partitioned builds or production service load.

Sources: [driver entry point](<src/main.rs>), [memory provider adapter](<src/memory.rs>),
[SSD builder adapter](<src/ssd.rs>), [matched runner](<../upstream_diskann_benchmark.py>).
