# Vectorlite benchmark

Compares vectorlite's vector-search performance and recall against:

- `hnswlib` (in-memory, the library vectorlite is built on)
- `vectorlite` brute-force (`SELECT ... ORDER BY vector_distance(...)`)
- `sqlite_vss` _(optional)_
- `sqlite_vec` _(optional)_
- `milvus-lite` _(optional)_
- `libsql` _(optional)_ — Turso/libSQL with built-in DiskANN vector index
- `sqlite-vector` _(optional)_ — sqliteai SIMD-accelerated full scan

Each cell is benchmarked with `pytest-benchmark`, which auto-calibrates
warm-up and round count and reports min / max / mean / median / stddev / IQR
per cell.

## Requirements

- **Python >= 3.10** (driven by NumPy 2.4 wheel availability; enforced
  at session start by `benchmark/conftest.py` so wrong-Python runs fail
  with a clear message instead of cryptic install errors)
- A Python interpreter built with `--enable-loadable-sqlite-extensions`
  (standard on Homebrew, python.org installer, and modern Linux distro
  Pythons; see [SQLite driver](#sqlite-driver) below)

## Choosing a Python interpreter

The benchmark uses Python's stdlib `sqlite3` module, which links against
whatever SQLite the interpreter was built with. SQLite versions vary
significantly across Python distributions, even at the same Python
version. Vectorlite itself loads on virtually any SQLite, but its
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

**Rule of thumb:** Python 3.11+ from python.org, Homebrew, or pyenv is
always fine. System Python on Linux is unreliable below
Ubuntu 24.04 / Debian 12 / RHEL 10 / Alpine 3.19+.

To check what your interpreter has:

```bash
python -c "import sqlite3; print(sqlite3.sqlite_version)"
```

If the version is too old and you cannot upgrade Python, switch to
[apsw](https://rogerbinns.github.io/apsw/), which bundles its own
SQLite (currently 3.53). The benchmark targets stdlib `sqlite3`
specifically, but the rest of vectorlite's documentation and examples
work with apsw.

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
the installed `vectorlite_py` wheel. To benchmark something else there are
two equivalent options:

```bash
# Command-line flag (highest priority):
pytest benchmark/test_benchmark.py \
    --vectorlite-path=build/release/vectorlite/vectorlite.dylib

# Environment variable (also picked up by examples/ and tests/):
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

`build/dev/vectorlite/vectorlite.dylib` exists right next to the release
build and is roughly **10× slower**. The benchmark detects path patterns
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
| `VECTORLITE_PATH` | wheel default | Vectorlite shared library to load. |
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
Python build (or `apsw`).

The benchmark itself does not use vectorlite's metadata-filter (rowid
pushdown) feature, so any SQLite version that vectorlite loads on works
here. The bundled SQLite version is reported in the session header.
