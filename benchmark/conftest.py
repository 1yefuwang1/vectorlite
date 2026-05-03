"""Pytest fixtures for the vectorlite benchmark suite.

Provides session-scoped, expensive resources (extension-loaded sqlite
connection, randomly generated test data, ground-truth labels) plus
``backend_factory`` fixtures that hand out fresh ``Backend`` instances to
each test.

Selecting the vectorlite library to benchmark
---------------------------------------------

Resolution order:

1. ``--vectorlite-path=<path>`` command-line option (highest priority)
2. ``VECTORLITE_PATH`` environment variable
3. ``vectorlite_py.vectorlite_path()`` from the installed wheel (fallback)

The session prints which file was loaded and its size; if the resolved
path looks like a debug build (path contains ``/build/dev/``,
``/Debug/`` or ``/debug/``) a warning is emitted because debug builds
are typically several times slower than release.
"""

from __future__ import annotations

import os
import re
import sqlite3
import warnings
from pathlib import Path
from typing import Iterator

import pytest
import vectorlite_py

from benchmark import (
    BenchmarkData,
    DIMS,
    DISTANCE_TYPES,
    HnswlibBackend,
    K,
    MilvusLiteBackend,
    NUM_QUERIES,
    SqliteVecBackend,
    SqliteVssBackend,
    VectorliteBackend,
    VectorliteBruteForceBackend,
    is_supported_platform,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0") != "0"


_DEBUG_PATH_RE = re.compile(r"(/build/dev/|/Debug/|/debug/)")


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("vectorlite", "vectorlite benchmark options")
    group.addoption(
        "--vectorlite-path",
        action="store",
        default=None,
        help="Path to the vectorlite shared library to benchmark. "
             "Overrides the VECTORLITE_PATH environment variable. "
             "Defaults to the wheel's bundled binary.",
    )


def _resolve_vectorlite_path(config: pytest.Config) -> str:
    cli = config.getoption("--vectorlite-path")
    if cli:
        return os.path.abspath(cli)
    env = os.environ.get("VECTORLITE_PATH")
    if env:
        return os.path.abspath(env)
    return vectorlite_py.vectorlite_path()


@pytest.fixture(scope="session")
def vectorlite_path(pytestconfig: pytest.Config) -> str:
    return _resolve_vectorlite_path(pytestconfig)


def pytest_report_header(config: pytest.Config) -> list[str]:
    """Print the resolved vectorlite library banner above the test list."""
    path = _resolve_vectorlite_path(config)

    # ``vectorlite_py.vectorlite_path()`` and the VECTORLITE_PATH convention
    # may omit the platform-specific suffix (SQLite's loader appends it).
    # Look for the actual file on disk so we can print its size.
    resolved = Path(path)
    if not resolved.exists():
        for ext in (".dylib", ".so", ".dll"):
            candidate = resolved.with_suffix(ext) if resolved.suffix else \
                Path(str(resolved) + ext)
            if candidate.exists():
                resolved = candidate
                break

    try:
        size_bytes = resolved.stat().st_size
        size_str = f"{size_bytes / (1024 * 1024):.2f} MiB"
    except OSError:
        size_str = "missing"

    lines = [
        f"vectorlite: {path}",
        f"            ({size_str}, sqlite3 {sqlite3.sqlite_version})",
    ]

    if _DEBUG_PATH_RE.search(path):
        lines.append(
            "            WARNING: path looks like a debug build; "
            "benchmark numbers will not be representative.")
        warnings.warn(
            f"Benchmarking what looks like a debug build: {path}. "
            "Use a release build (build/release/...) for representative numbers.",
            UserWarning,
            stacklevel=2,
        )

    return lines


@pytest.fixture(scope="session")
def num_elements() -> int:
    return int(os.environ.get("NUM_ELEMENTS", 3000))


# ---------------------------------------------------------------------------
# SQLite connection (session-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sqlite_conn(vectorlite_path: str) -> Iterator[sqlite3.Connection]:
    """A SQLite connection with vectorlite (and any optional extensions) loaded.

    ``isolation_level=None`` puts sqlite3 in autocommit mode so that the
    explicit ``BEGIN TRANSACTION`` / ``COMMIT`` issued by ``_SqlBackend``
    is honoured.
    """
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.enable_load_extension(True)
    conn.load_extension(vectorlite_path)

    if _env_flag("BENCHMARK_VSS") and is_supported_platform():
        # sqlite_vss is not self-contained; on Debian/Ubuntu install:
        #   sudo apt-get install -y libgomp1 libatlas-base-dev liblapack-dev
        import sqlite_vss
        sqlite_vss.load(conn)

    if _env_flag("BENCHMARK_SQLITE_VEC") and is_supported_platform():
        import sqlite_vec
        conn.load_extension(sqlite_vec.loadable_path())

    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="session")
def sqlite_cursor(sqlite_conn: sqlite3.Connection) -> sqlite3.Cursor:
    return sqlite_conn.cursor()


# ---------------------------------------------------------------------------
# Random vectors and ground truth (session-scoped, expensive)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def benchmark_data(num_elements: int) -> BenchmarkData:
    return BenchmarkData.generate(
        DIMS, num_elements, NUM_QUERIES, K, DISTANCE_TYPES)


# ---------------------------------------------------------------------------
# Backend factories
# ---------------------------------------------------------------------------
#
# Each factory yields a freshly-constructed backend per test. The backend's
# lifecycle (setup -> insert -> search -> teardown) is driven inside the test
# itself so that pytest-benchmark can time the relevant phase.


@pytest.fixture
def vectorlite_backend(sqlite_cursor, benchmark_data) -> VectorliteBackend:
    return VectorliteBackend(sqlite_cursor, benchmark_data)


@pytest.fixture
def vectorlite_bf_backend(sqlite_cursor,
                          benchmark_data) -> VectorliteBruteForceBackend:
    return VectorliteBruteForceBackend(sqlite_cursor, benchmark_data)


@pytest.fixture
def hnswlib_backend(benchmark_data) -> HnswlibBackend:
    return HnswlibBackend(benchmark_data)


@pytest.fixture
def sqlite_vss_backend(sqlite_cursor, benchmark_data) -> SqliteVssBackend:
    if not _env_flag("BENCHMARK_VSS"):
        pytest.skip("set BENCHMARK_VSS=1 to enable sqlite_vss benchmark")
    if not is_supported_platform():
        pytest.skip("sqlite_vss only supported on Linux/macOS")
    return SqliteVssBackend(sqlite_cursor, benchmark_data)


@pytest.fixture
def sqlite_vec_backend(sqlite_cursor, benchmark_data) -> SqliteVecBackend:
    if not _env_flag("BENCHMARK_SQLITE_VEC"):
        pytest.skip("set BENCHMARK_SQLITE_VEC=1 to enable sqlite_vec benchmark")
    if not is_supported_platform():
        pytest.skip("sqlite_vec only supported on Linux/macOS")
    return SqliteVecBackend(sqlite_cursor, benchmark_data)


@pytest.fixture
def milvus_lite_backend(benchmark_data) -> MilvusLiteBackend:
    if not _env_flag("BENCHMARK_MILVUS_LITE"):
        pytest.skip("set BENCHMARK_MILVUS_LITE=1 to enable milvus-lite benchmark")
    if not is_supported_platform():
        pytest.skip("milvus-lite only supported on Linux/macOS")
    return MilvusLiteBackend(benchmark_data)
