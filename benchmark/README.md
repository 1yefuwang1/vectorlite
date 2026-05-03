# Vectorlite benchmark

Compares vectorlite's vector-search performance and recall against:

- `hnswlib` (in-memory, the library vectorlite is built on)
- `vectorlite` brute-force (`SELECT ... ORDER BY vector_distance(...)`)
- `sqlite_vss` _(optional)_
- `sqlite_vec` _(optional)_
- `milvus-lite` _(optional)_

Each cell is benchmarked with `pytest-benchmark`, which auto-calibrates
warm-up and round count and reports min / max / mean / median / stddev / IQR
per cell.

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
