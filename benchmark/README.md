# Vectorlite benchmark

Compares vectorlite's vector-search performance and recall against:

- `hnswlib` (in-memory, the library vectorlite is built on)
- `vectorlite` brute-force (`SELECT ... ORDER BY vector_distance(...)`)
- SQLite-backed Rust DiskANN _(optional; `BENCHMARK_DISKANN=1`)_
- `sqlite_vss` _(optional)_
- `sqlite_vec` _(optional)_
- `milvus-lite` _(optional)_
- `libsql` _(optional)_ — Turso/libSQL with built-in DiskANN vector index
- `sqlite-vector` _(optional)_ — sqliteai SIMD-accelerated full scan

Each cell is benchmarked with `pytest-benchmark`, which auto-calibrates
warm-up and round count and reports min / max / mean / median / stddev / IQR
per cell.

Local CMake builds now benchmark the **Rust implementation** of the SQLite
extension, backed by the retained hnswlib/Highway native core. No separate C++
virtual-table build or Rust deployment step is needed. The Cargo workspace uses
the root [manifest](<../Cargo.toml>) and [lockfile](<../Cargo.lock>); extension
sources and retained native ops live under `vectorlite/`. See the
[contributor guide](<../vectorlite/README.md>). This source-layout move does not
change the public `build/<preset>/vectorlite` artifact paths. The native ops
microbenchmark remains a separate, opt-in executable, built from
[ops_benchmark.cpp](<../vectorlite/ops/ops_benchmark.cpp>). Enable its native
vcpkg benchmark dependency and CMake target explicitly:

```bash
cmake --preset release -DVECTORLITE_BUILD_BENCHMARKS=ON
cmake --build build/release --target ops_benchmark -j8
# Run from the repository root:
build/release/vectorlite/ops/ops_benchmark
```

The figures and raw tables in the repository's [main README](<../README.md#benchmark>)
are **historical measurements from before the Rust-primary migration**, not
new Rust performance results. Keep their recorded artifact paths and numbers
as provenance; rerun this suite to measure the current implementation.

## Small shared-data HNSW / DiskANN comparison

The existing pytest suite defaults to **3,000 vectors**, dimensions 128/512/1536/3000,
100 top-10 queries per configuration, and both L2 and cosine. Enable the
SQLite-backed DiskANN cases alongside the existing baselines:

```bash
NUM_ELEMENTS=3000 BENCHMARK_SEED=42 BENCHMARK_DISKANN=1 \
  pytest benchmark/test_benchmark.py \
    --vectorlite-path=build/release/vectorlite/vectorlite.dylib \
    -k 'vectorlite or hnswlib' --benchmark-json=bench-3000.json
```

All backends use the same generated vectors, queries, and exact ground truth.
Insertions use one transaction per complete dataset, with one warm-up and three
measured rebuilds. Query timings cover batches of 100 searches against already
built indexes; setup/build time is excluded. The query-window sweep is 10/50/100:
HNSW uses numeric `ef`, while DiskANN uses JSON `search_list_size`, recorded and
labelled separately. Equal window values are **not** a guarantee of equal recall;
compare recall as well as latency.

This suite uses an **in-memory SQLite database**, so it measures the current
SQL/graph adapter overhead, not cold-disk I/O, durable commit cost, or behavior
beyond RAM. For those workloads, use the separate file-backed runner below.
DiskANN requires SQLite 3.38 or newer and a build containing the new backend.
Its graph settings here are `degree=32, build_list_size=100`; other options use
the documented defaults. Existing HNSW settings remain `M=30, ef_construction=100`.

## Focused native batch INSERT comparison

Use [the bounded native runner](<diskann_batch_benchmark.py>) for pointer-based
batch ingestion, rather than repeating the full pytest build matrix:

```bash
# Configure the release tree with BUILD_TESTING=ON, then build this opt-in target.
cmake --build build/release --target diskann_batch_benchmark
python benchmark/diskann_batch_benchmark.py \
  --executable build/release/vectorlite/diskann_batch_benchmark \
  --extension build/release/vectorlite/vectorlite.dylib \
  --output-dir build/benchmarks/batch-insert-new --timeout 120
```

The native [C host](<diskann_batch_benchmark.c>) builds each HNSW / single DiskANN /
batch-8 / batch-32 variant exactly once, then reuses that index for query windows
10/50/100. Each variant runs in its own process with a hard timeout, and completed
results are saved immediately. A failed mode stops the remaining sequence. The
C host also has a 90-second insertion deadline via a SQLite progress handler.

Defaults are one shared seeded 3,000-vector 128D L2 dataset and 100 top-10 queries.
The Python driver computes exact ground truth and records individual-query
mean/p50/p95 plus recall and artifact/data fingerprints. These timings differ from
the pytest suite's repeated-batch medians: they exclude per-query bind/reset and
have only one measured build/pass, so do not compare them as controlled before/
after statistics. The database is `:memory:`; this is not cold-disk, durable commit,
physical beyond-RAM, or production-quality evidence. Pointer ingestion is a native
C/Rust API; this runner does not add a pointer bridge to Python's sqlite3 driver.
The work allowance is explicit (`--max-visits`, default 65536) and applies to an
entire true chunk; a batch-32 run may need a larger setting than batch-8. Failed
runs remain in their own result directory instead of being silently retried.

## Streamed SQLite-contained DiskANN benchmark

[The standalone DiskANN runner](<diskann_benchmark.py>) is separate from the
in-memory, multi-backend pytest suite below. It requires only Python's standard
library and an **explicit, freshly built extension** with the DiskANN backend;
SQLite 3.38 or newer is required. It never falls back to an installed wheel.

```bash
# Run after building the release extension; choose a NEW database and JSON file.
python benchmark/diskann_benchmark.py \
  --extension build/release/vectorlite/vectorlite.dylib \
  --database diskann-production.sqlite --output diskann-production.json \
  --count 10000 --dim 128 --metric l2 --queries 32 --k 10 \
  --degree 16 --build-list-size 64 --search-list-size 64 \
  --cache-bytes 67108864 --max-visits 2048 --updates 16 --deletes 16

# Reopen a previously generated database without changing its contents.
# Dataset/graph settings come from its persisted benchmark manifest.
python benchmark/diskann_benchmark.py \
  --extension build/release/vectorlite/vectorlite.dylib \
  --database diskann-production.sqlite --query-only --queries 32 --k 10
```

Use `.so` on Linux or `.dll` on Windows. New runs fail if the database already
exists; `--query-only` requires a database created by this runner. A JSON output
file must also be new. Failed builds leave their partial database for inspection,
not automatic reuse or deletion.

### Measurement method

- Seeded float32 L2 or cosine data is generated one vector at a time and inserted
  through the public virtual-table API in transactions of at most 128 vectors.
  There is no corpus-sized Python/NumPy matrix or vector list.
- Build, query, exact-ground-truth and mutation stages use **separate fresh
  interpreter processes**. Query workers do not inherit the builder's dataset
  or memory high-water mark. Exact ground truth runs **after** timed queries in
  another process, scanning stored vectors in bounded batches into per-query
  top-k heaps. `--numpy-ground-truth` optionally accelerates those bounded
  batches; NumPy is never imported by measured query workers.
- JSON records recall@k, the first query, first-pass and repeated warm-pass
  p50/p95 latency, extension path/SHA-256, SQLite/Python versions, build cost,
  update/delete/consolidation costs, database/WAL sizes, and per-stage peak RSS.
  Mutation measurements are followed by another fresh query/recall run.
- “Fresh-process first pass” **does not mean cold OS cache**. The builder and
  earlier workers can leave OS pages cached; this runner never drops global
  OS caches. Both first-pass and warm latency exclude synthetic query generation.
- Every connection uses an 8 MiB SQLite page-cache setting and `mmap_size=0`.
  `--cache-bytes` is the graph workspace budget, not total process RSS. Linux
  `ru_maxrss` is converted from KiB, macOS from bytes; Windows uses
  `GetProcessMemoryInfo.PeakWorkingSetSize`.
- `--query-memory-limit-mib` applies and verifies Linux `RLIMIT_AS` in query
  workers only. This limits **virtual address space, not RSS**. It is rejected
  on macOS/Windows, whose corresponding behavior is not verified here.
- Format-3 soft deletions retain vectors and adjacency until maintenance.
  Consolidation performs a SQLite-staged full graph rebuild with bounded Rust
  workspace, not merely deleted-neighbor cleanup; its cost can scale with the
  entire live population. Use `--skip-consolidation` if that cost is intentionally
  outside a run. The runtime descriptor/format is recorded so prototype format-2
  measurements are not confused with format-3 results. No default latency or
  recall SLA is implied by this benchmark.

### Controlled large-storage fixture

To exercise lazy storage/working memory without waiting for a large production
ANN build, the runner also offers a synthetic linear graph:

```bash
python benchmark/diskann_benchmark.py \
  --extension build/release/vectorlite/vectorlite.dylib \
  --database diskann-storage.sqlite --output diskann-storage.json \
  --fixture storage-linear --count 65536 --dim 1024 --metric l2 \
  --degree 4 --build-list-size 32 --search-list-size 64 \
  --cache-bytes 16777216 --max-visits 1024 --queries 32 --k 10 \
  --require-beyond-ram
```

**This is not a production ANN build or a general recall benchmark.** It uses a
public first-row bootstrap, then deliberately populates private format-2/3 shadow
tables in streamed batches. The runtime descriptor is preserved and its format
recorded; no descriptor is fabricated to bypass extension validation. Those
writes are benchmark fixture construction,
**not a supported import API**. The two-row atomic carrier and descriptor remain
unchanged. Frozen node 0 connects to a bidirectional chain of live nodes; vectors
have their row ID in the first float32 coordinate and zeros elsewhere. Queries
address only the reachable start prefix and verify analytic nearest labels and
squared distances after reopening in a fresh process. Mutations are skipped.
JSON uses fixture coverage rather than reporting production ANN recall.

The example stores **256 MiB of live vector payload**, plus SQLite and graph
storage overhead. `--require-beyond-ram` succeeds only when the measured database
file is larger than both the query worker's peak RSS and the configured graph
workspace plus SQLite cache. This is measured evidence about this controlled
workload, **not proof of random-query performance, a cold OS cache, or universal
memory bounds**. Small fixture smoke runs make no beyond-RAM claim.

Artifact-independent utility tests do not load an extension or benchmark deps:

```bash
python -B -m unittest discover -s benchmark -p 'test_diskann_benchmark_utils.py' -v
```

## Requirements

- **Python >= 3.14** to install the current `vectorlite_py` package and the
  benchmark dependencies. The harness's older Python >= 3.10 guard in
  [conftest.py](<conftest.py>) does not override package/dependency requirements.
- A Python interpreter built with `--enable-loadable-sqlite-extensions`
  (standard on Homebrew, python.org installer, and modern Linux distro
  Pythons; see [SQLite driver](#sqlite-driver) below)

## Choosing a Python interpreter

The benchmark uses Python's stdlib `sqlite3` module, which links against
whatever SQLite the interpreter was built with. SQLite versions vary
significantly across Python distributions, even at the same Python
version. Vectorlite requires **SQLite >= 3.20**, and its
metadata-filter (rowid pushdown) feature requires **SQLite >= 3.38** -
the benchmark does not exercise that path, so any SQLite that loads
the extension at all will run the benchmark. The session header reports
the loaded SQLite version and prints a NOTE line if it is below 3.38.

Empirically, here is what common Python distributions ship:

| Distribution                        | SQLite          | Metadata filter (>= 3.38)? |
|-------------------------------------|-----------------|----------------------------|
| python.org installer 3.10           | 3.36            | no                         |
| python.org installer 3.11           | 3.39            | yes                        |
| python.org installer 3.12           | 3.43            | yes                        |
| python.org installer 3.13+          | 3.45+           | yes                        |
| Homebrew Python (any version)       | tracks Homebrew's `sqlite` keg, currently ~3.53 | yes |
| pyenv-built Python (any version)    | tracks the Homebrew/system SQLite at build time | usually yes |
| Conda / Miniconda Python            | bundled, recent | yes                        |
| Ubuntu 20.04 system Python (3.8)    | 3.31            | no                         |
| Ubuntu 22.04 system Python (3.10)   | 3.37            | no (just under!)           |
| Ubuntu 24.04 system Python (3.12)   | 3.45            | yes                        |
| Debian 11 system Python             | 3.34            | no                         |
| Debian 12 system Python             | 3.40            | yes                        |
| RHEL/Rocky/Alma 9 system Python     | 3.34            | no                         |
| Official `python:3.X` Docker image  | tracks the Debian base; recent tags are fine | usually yes |

These distribution examples describe SQLite availability, not the current
package's Python support floor. Use Python 3.14 or newer for the current
`vectorlite_py` package, verify extension loading is enabled, and check the
actual SQLite version instead of relying on a distribution name.

To check what your interpreter has:

```bash
python -c "import sqlite3; print(sqlite3.sqlite_version)"
```

If SQLite is too old or extension loading is disabled, use another Python
3.14-or-newer build with a recent SQLite and loadable-extension support.

## Quick start

```bash
pip install -r benchmark/requirements.txt           # core deps (PyPI vectorlite)
pip install -r benchmark/requirements-extra.txt     # +sqlite_vss, sqlite_vec

# Run the default backends (vectorlite, hnswlib, vectorlite brute force):
pytest benchmark/test_benchmark.py --benchmark-json=bench.json

# Render the two PNG figures + a recall.csv companion next to them:
python benchmark/plot.py bench.json
```

## Choosing which vectorlite to benchmark

By default the benchmark loads the vectorlite shared library shipped with
the installed `vectorlite_py` wheel, which may be an older release. To measure
the current Rust implementation, first build it through CMake:

```bash
git submodule update --init --recursive
python3 bootstrap_vcpkg.py
cmake --preset release
cmake --build build/release -j8
```

This source build needs latest stable Rust, C/C++17 compilers, CMake >= 3.22,
Ninja and vcpkg. CMake invokes Cargo and keeps the public library filename;
there is no separate C++ virtual-table build. Use `.so` on Linux, `.dylib` on
macOS or `.dll` on Windows in the path below. To select the library explicitly,
there are two equivalent options:

```bash
# Command-line flag (highest priority):
pytest benchmark/test_benchmark.py \
    --vectorlite-path=build/release/vectorlite/vectorlite.dylib

# Environment variable (also picked up by the examples):
VECTORLITE_PATH=build/release/vectorlite/vectorlite.dylib \
    pytest benchmark/test_benchmark.py
```

The session header prints which file was actually loaded so you can
double-check before trusting the numbers:

```
vectorlite: /path/to/build/release/vectorlite/vectorlite.dylib
            (2.47 MiB, sqlite3 3.50.4)
```

### Don't benchmark a debug build

A debug artifact such as `build/dev/vectorlite/vectorlite.dylib` is not
representative of release performance; the slowdown depends on the workload
and toolchain. The benchmark detects path patterns
that look like debug builds (`/build/dev/`, `/Debug/`, `/debug/`) and
prints a `WARNING` line plus a Python `UserWarning`:

```
vectorlite: /path/to/build/dev/vectorlite/vectorlite.dylib
            (8.55 MiB, sqlite3 3.50.4)
            WARNING: path looks like a debug build; benchmark numbers will not be representative.
```

If you really do want to compare debug-build perf for some reason, the
warning is non-fatal — the run still proceeds.

## Comparing two builds (before / after)

`pytest-benchmark` saves results under `.benchmarks/` and can diff them.
Useful when changing vectorlite internals and you want to know whether a
change moved the needle:

```bash
# Take a baseline of the released library:
pytest benchmark/test_benchmark.py --benchmark-save=baseline

# ... edit vectorlite, rebuild ...
cmake --build build/release -j8

# Run again against the new local build:
pytest benchmark/test_benchmark.py \
    --vectorlite-path=build/release/vectorlite/vectorlite.dylib \
    --benchmark-compare=baseline \
    --benchmark-compare-fail=median:5%
```

`--benchmark-compare-fail=median:5%` makes pytest exit non-zero if any
median regressed by more than 5 %. Useful in a CI job.

To list saved baselines: `python -m pytest_benchmark list`. To delete
them: `rm -r .benchmarks/`.

## Configuration

Driven by environment variables; defaults in `benchmark/benchmark.py`.

| Variable | Default | Effect |
|---|---|---|
| `NUM_ELEMENTS` | `3000` | Number of random vectors indexed per case. |
| `BENCHMARK_SEED` | unset | Optional integer seed shared by every backend. |
| `VECTORLITE_PATH` | wheel default | Vectorlite shared library to load. |
| `BENCHMARK_DISKANN` | `0` | `1` enables vectorlite's SQLite-backed Rust DiskANN. |
| `BENCHMARK_VSS` | `0` | `1` enables the `sqlite_vss` backend (Linux/macOS). |
| `BENCHMARK_SQLITE_VEC` | `0` | `1` enables the `sqlite_vec` backend (Linux/macOS). |
| `BENCHMARK_MILVUS_LITE` | `0` | `1` enables the `milvus-lite` backend (Linux/macOS). |
| `BENCHMARK_LIBSQL` | `0` | `1` enables the `libsql` backend (Linux/macOS). |
| `BENCHMARK_SQLITE_VECTOR` | `0` | `1` enables the `sqlite-vector` backend (Linux/macOS). |

Other constants (vector dimensions, distance metrics, HNSW parameters,
`ef_search` values, query count) are at the top of `benchmark.py` and are
edited in source rather than via environment.

## Useful pytest-benchmark flags

```bash
# Filter to a subset (test parametrize ids: [<distance>-<dim>-<ef_search>]):
pytest benchmark/test_benchmark.py -k "search and 1536 and 50"

# Sort the printed table by a specific stat:
pytest benchmark/test_benchmark.py --benchmark-sort=mean

# Group rows by parameter (default groups by test):
pytest benchmark/test_benchmark.py --benchmark-group-by=group,param:dim

# Skip generating the JSON file (just print the table):
pytest benchmark/test_benchmark.py
```

See `pytest-benchmark`'s docs for the full list:
<https://pytest-benchmark.readthedocs.io>.

## Output files

After `python benchmark/plot.py bench.json`:

- `vector_insertion_<N>_vectors.png` — bar chart, per-vector insert time vs. dim, one bar per `(product, distance_type)`.
- `vector_query_<N>_vectors.png` — bar chart, per-query search time vs. dim, one bar per `(product, distance_type, ef_search)`.
- `vector_query_<N>_vectors_recall.csv` — recall per query cell. Recall isn't a timing metric so it isn't on the bar chart; this file makes sure it isn't lost.

`<N>` is `NUM_ELEMENTS`. The PNGs are written to the current directory by
default; pass `--output-dir=DIR` to `plot.py` to put them somewhere else.

## File layout

```
benchmark/
├─ benchmark.py        # library: BenchmarkData, Backend hierarchy, helpers
├─ conftest.py         # pytest fixtures and the --vectorlite-path option
├─ test_benchmark.py   # parametrized pytest-benchmark cases (one per cell)
├─ plot.py             # JSON -> PNG + recall CSV
├─ requirements.txt    # core dependencies
└─ requirements-extra.txt  # sqlite_vss, sqlite_vec
```

## SQLite driver

The benchmark uses Python's stdlib `sqlite3` rather than `apsw`. Loading
the vectorlite extension via `conn.load_extension(...)` requires a Python
interpreter built with `--enable-loadable-sqlite-extensions` (standard on
Homebrew, python.org installer and modern Linux distro Pythons). If your
interpreter does not enable that, `conn.enable_load_extension(True)`
raises `AttributeError` or `OperationalError` and you'll need a different
Python build.

The benchmark itself does not use vectorlite's metadata-filter (rowid
pushdown) feature. Existing HNSW cases work with any SQLite version that loads
the extension; opt-in DiskANN cases require SQLite 3.38 or newer. The bundled
SQLite version is reported in the session header.
