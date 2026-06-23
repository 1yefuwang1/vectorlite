# Comprehensive Python Integration Regression Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a comprehensive, behavior-characterizing Python integration test suite that locks down vectorlite's observable SQL behavior so future changes cannot silently regress it.

**Architecture:** New pytest files under `bindings/python/vectorlite_py/test/`, organized by category, sharing helpers from a new `helpers.py` and fixtures from a new `conftest.py`. The existing `vectorlite_test.py` is left untouched. These are characterization tests: each asserts the extension's *current, verified* behavior, so they pass immediately against the built `.so` and fail if behavior drifts.

**Tech Stack:** pytest, the Python standard-library `sqlite3` module (the project requires Python >= 3.14, which bundles SQLite 3.50.x with loadable-extension support; `apsw` was dropped in commit 96c5d37 / PR #54), numpy, the prebuilt `vectorlite.so`.

---

## Conventions for every task

**Run command** (from repo root, venv active):

```bash
source .venv/bin/activate
PYTHONPATH=bindings/python python -m pytest bindings/python/vectorlite_py/test/<file>.py -q
```

The test environment needs only `numpy` and `pytest`; database access uses the
stdlib `sqlite3` module. The interpreter must be a CPython 3.14 whose `sqlite3`
was compiled with loadable-extension support (the repo's `.venv` qualifies:
`sqlite3.sqlite_version == '3.50.4'` and `load_extension` works).

Because these are characterization tests against existing behavior, the "test
fails first" TDD step is replaced by: **run the new test file and confirm every
test PASSES** (the suite characterizes shipped behavior). A failure means either
the test is wrong or a genuine behavior gap was found — investigate before
committing.

**Connection pattern** (mirrors `vectorlite_test.py`):

```python
conn = sqlite3.connect(':memory:', isolation_level=None)
conn.enable_load_extension(True)
conn.load_extension(vectorlite_py.vectorlite_path())
```

**Error type:** every vectorlite-side failure surfaces as
`sqlite3.OperationalError` (verified for duplicate rowid, capacity, dimension
mismatch, invalid options, bad function args, malformed JSON, load failures, and
unconstrained scans).

---

## Import note (package layout)

The test directory is a Python package (`bindings/python/vectorlite_py/test/`
contains `__init__.py`), and the suite runs with `PYTHONPATH=bindings/python`.
Because of this, a bare `from conftest import ...` does **not** resolve — pytest
imports modules under their package-qualified names. Shared *functions and
constants* therefore live in `helpers.py` and are imported as
`from vectorlite_py.test.helpers import ...`. Shared *fixtures* (`conn`, `rng`)
live in `conftest.py` and are auto-discovered by pytest (no import needed). This
was verified empirically before writing the plan.

## File Structure

- Create: `bindings/python/vectorlite_py/test/helpers.py` — shared pure functions and constants (numpy reference helpers, connection factory).
- Create: `bindings/python/vectorlite_py/test/conftest.py` — shared pytest fixtures (`conn`, `rng`).
- Create: `bindings/python/vectorlite_py/test/test_scalar_functions.py`
- Create: `bindings/python/vectorlite_py/test/test_vector_roundtrip.py`
- Create: `bindings/python/vectorlite_py/test/test_knn_query.py`
- Create: `bindings/python/vectorlite_py/test/test_ddl_options.py`
- Create: `bindings/python/vectorlite_py/test/test_dml.py`
- Create: `bindings/python/vectorlite_py/test/test_persistence.py`

---

### Task 1: Shared helpers and fixtures (`helpers.py`, `conftest.py`)

**Files:**
- Create: `bindings/python/vectorlite_py/test/helpers.py`
- Create: `bindings/python/vectorlite_py/test/conftest.py`

- [ ] **Step 1a: Write `helpers.py` (pure functions and constants)**

```python
import sqlite3
import numpy as np
import vectorlite_py

SEED = 12345
ELEMENT_TYPES = ['float32', 'bfloat16', 'float16']
# '' (empty space) is treated as 'l2' by vectorlite.
SPACES = ['l2', 'ip', 'cosine', '']
# Reading a quantized vector back as float32 is lossy; float32 is exact.
DEQUANT_RTOL = {'float32': 0.0, 'bfloat16': 1e-2, 'float16': 1e-3}


def get_connection(path=':memory:'):
    conn = sqlite3.connect(path, isolation_level=None)
    conn.enable_load_extension(True)
    conn.load_extension(vectorlite_py.vectorlite_path())
    return conn


def random_vectors(rng, n, dim):
    return np.float32(rng.random((n, dim)))


def l2_squared(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.dot(d, d))


def ip_distance(a, b):
    return float(1.0 - np.dot(np.asarray(a, np.float64), np.asarray(b, np.float64)))


def cosine_distance(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# 'l2' uses squared L2 because hnswlib does not take the square root.
DISTANCE_FN = {'l2': l2_squared, 'ip': ip_distance, 'cosine': cosine_distance}


def brute_force_knn(vectors, query, k, space='l2'):
    fn = DISTANCE_FN[space]
    dists = [(i, fn(query, vectors[i])) for i in range(len(vectors))]
    dists.sort(key=lambda x: x[1])
    return dists[:k]
```

- [ ] **Step 1b: Write `conftest.py` (fixtures only)**

```python
import numpy as np
import pytest
from vectorlite_py.test.helpers import get_connection, SEED


@pytest.fixture
def conn():
    c = get_connection()
    yield c
    c.close()


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)
```

- [ ] **Step 2: Sanity-check the helpers import**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=bindings/python python -c "from vectorlite_py.test.helpers import brute_force_knn; print(brute_force_knn([[0,0],[1,1]], [0,0], 1))"
```

Expected: `[(0, 0.0)]`

- [ ] **Step 3: Commit**

```bash
git add bindings/python/vectorlite_py/test/helpers.py bindings/python/vectorlite_py/test/conftest.py
git commit -m "test: add shared helpers and fixtures for integration suite"
```

---

### Task 2: Scalar function tests (`test_scalar_functions.py`)

**Files:**
- Create: `bindings/python/vectorlite_py/test/test_scalar_functions.py`

Covers: `vector_distance` (l2/ip/cosine vs numpy), `vector_to_json` /
`vector_from_json` round-trip and errors, `vectorlite_info`.

- [ ] **Step 1: Write the test file**

```python
import json
import math
import sqlite3
import numpy as np
import pytest
import vectorlite_py
from vectorlite_py.test.helpers import l2_squared, ip_distance, cosine_distance


@pytest.mark.parametrize('dim', [1, 3, 4, 16, 128])
def test_vector_distance_matches_numpy(conn, dim):
    rng = np.random.default_rng(7)
    a = np.float32(rng.random(dim))
    b = np.float32(rng.random(dim))
    cur = conn.cursor()

    ip = cur.execute('select vector_distance(?, ?, "ip")', (a.tobytes(), b.tobytes())).fetchone()[0]
    assert np.isclose(ip, ip_distance(a, b), atol=1e-5)

    cos = cur.execute('select vector_distance(?, ?, "cosine")', (a.tobytes(), b.tobytes())).fetchone()[0]
    assert np.isclose(cos, cosine_distance(a, b), atol=1e-5)

    l2 = cur.execute('select vector_distance(?, ?, "l2")', (a.tobytes(), b.tobytes())).fetchone()[0]
    # vectorlite returns squared L2 distance.
    assert np.isclose(l2, l2_squared(a, b), rtol=1e-4, atol=1e-4)
    assert np.isclose(math.sqrt(l2), float(np.linalg.norm(a - b)), rtol=1e-4, atol=1e-4)


def test_vector_distance_self_is_zero_for_l2(conn):
    a = np.float32([1, 2, 3, 4])
    d = conn.cursor().execute('select vector_distance(?, ?, "l2")', (a.tobytes(), a.tobytes())).fetchone()[0]
    assert np.isclose(d, 0.0, atol=1e-6)


def test_vector_distance_wrong_arg_count_is_rejected(conn):
    with pytest.raises(sqlite3.OperationalError):
        conn.cursor().execute('select vector_distance(?, ?)', (b'', b'')).fetchone()


def test_vector_distance_unknown_space_is_rejected(conn):
    a = np.float32([1, 2, 3, 4]).tobytes()
    with pytest.raises(sqlite3.OperationalError):
        conn.cursor().execute('select vector_distance(?, ?, "manhattan")', (a, a)).fetchone()


def test_vector_distance_dimension_mismatch_is_rejected(conn):
    a = np.float32([1, 2, 3]).tobytes()
    b = np.float32([1, 2, 3, 4]).tobytes()
    with pytest.raises(sqlite3.OperationalError):
        conn.cursor().execute('select vector_distance(?, ?, "l2")', (a, b)).fetchone()


def test_vector_distance_non_float_blob_is_rejected(conn):
    # A 3-byte blob is not a multiple of sizeof(float).
    with pytest.raises(sqlite3.OperationalError):
        conn.cursor().execute('select vector_distance(?, ?, "l2")', (b'abc', b'abc')).fetchone()


@pytest.mark.parametrize('dim', [1, 4, 64])
def test_json_round_trip(conn, dim):
    rng = np.random.default_rng(11)
    v = np.float32(rng.random(dim))
    cur = conn.cursor()
    back = cur.execute('select vector_from_json(vector_to_json(?))', (v.tobytes(),)).fetchone()[0]
    assert np.allclose(v, np.frombuffer(back, dtype=np.float32), atol=1e-6)


def test_vector_from_json_accepts_plain_json_array(conn):
    v = np.float32([1.5, 2.5, 3.5, 4.5])
    back = conn.cursor().execute('select vector_from_json(?)', (json.dumps(v.tolist()),)).fetchone()[0]
    assert np.allclose(v, np.frombuffer(back, dtype=np.float32), atol=1e-6)


def test_vector_to_json_format(conn):
    j = conn.cursor().execute('select vector_to_json(?)', (np.float32([1, 2, 3, 4]).tobytes(),)).fetchone()[0]
    assert json.loads(j) == [1.0, 2.0, 3.0, 4.0]


def test_vector_from_json_rejects_malformed_json(conn):
    with pytest.raises(sqlite3.OperationalError):
        conn.cursor().execute("select vector_from_json('not json')").fetchone()


def test_vector_to_json_rejects_non_blob(conn):
    with pytest.raises(sqlite3.OperationalError):
        conn.cursor().execute('select vector_to_json(?)', ('a string',)).fetchone()


def test_vectorlite_info_reports_version_and_simd(conn):
    out = conn.cursor().execute('select vectorlite_info()').fetchone()[0]
    assert f'vectorlite extension version {vectorlite_py.__version__}' in out
    assert 'Best SIMD target in use:' in out
```

- [ ] **Step 2: Run the file and confirm all tests pass**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=bindings/python python -m pytest bindings/python/vectorlite_py/test/test_scalar_functions.py -q
```

Expected: all tests PASS (e.g. `N passed`).

- [ ] **Step 3: Commit**

```bash
git add bindings/python/vectorlite_py/test/test_scalar_functions.py
git commit -m "test: characterize vector_distance, json, and vectorlite_info scalar functions"
```

---

### Task 3: Vector round-trip / data integrity (`test_vector_roundtrip.py`)

**Files:**
- Create: `bindings/python/vectorlite_py/test/test_vector_roundtrip.py`

Covers: byte-exact float32 read-back, bf16/f16 dequant tolerances, multiple
dims, cosine normalization on read.

- [ ] **Step 1: Write the test file**

```python
import numpy as np
import pytest
from vectorlite_py.test.helpers import get_connection, random_vectors, ELEMENT_TYPES, DEQUANT_RTOL


@pytest.mark.parametrize('dim', [1, 4, 32, 128])
def test_float32_read_back_is_byte_exact(conn, dim):
    rng = np.random.default_rng(3)
    v = np.float32(rng.random(dim))
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(e float32[{dim}], hnsw(max_elements=10))')
    cur.execute('insert into t(rowid, e) values (?, ?)', (0, v.tobytes()))
    back = cur.execute('select e from t where rowid = 0').fetchone()[0]
    assert back == v.tobytes()


@pytest.mark.parametrize('vector_type', ELEMENT_TYPES)
def test_read_back_within_tolerance_for_all_types(vector_type):
    dim = 32
    rng = np.random.default_rng(4)
    v = np.float32(rng.random(dim))
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(e {vector_type}[{dim}], hnsw(max_elements=10))')
    cur.execute('insert into t(rowid, e) values (?, ?)', (0, v.tobytes()))
    result = cur.execute('select e from t where rowid = 0').fetchone()[0]
    back = np.frombuffer(result, dtype=np.float32)
    assert back.shape == (dim,)
    rtol = DEQUANT_RTOL[vector_type]
    if rtol == 0.0:
        assert back.tobytes() == v.tobytes()
    else:
        assert np.allclose(back, v, rtol=rtol, atol=rtol)
    conn.close()


def test_cosine_column_is_normalized_on_read(conn):
    dim = 4
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(e float32[{dim}] cosine, hnsw(max_elements=10))')
    v = np.float32([3, 0, 4, 0])  # L2 norm 5
    cur.execute('insert into t(rowid, e) values (?, ?)', (0, v.tobytes()))
    back = np.frombuffer(cur.execute('select e from t where rowid = 0').fetchone()[0], dtype=np.float32)
    assert np.isclose(np.linalg.norm(back), 1.0, atol=1e-5)
    assert np.allclose(back, v / np.linalg.norm(v), atol=1e-5)


def test_non_cosine_column_is_stored_verbatim(conn):
    dim = 4
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(e float32[{dim}] l2, hnsw(max_elements=10))')
    v = np.float32([3, 0, 4, 0])
    cur.execute('insert into t(rowid, e) values (?, ?)', (0, v.tobytes()))
    back = cur.execute('select e from t where rowid = 0').fetchone()[0]
    assert back == v.tobytes()


def test_round_trip_survives_many_rows(conn):
    dim = 16
    n = 200
    vectors = random_vectors(np.random.default_rng(5), n, dim)
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(e float32[{dim}], hnsw(max_elements={n}))')
    for i in range(n):
        cur.execute('insert into t(rowid, e) values (?, ?)', (i, vectors[i].tobytes()))
    for i in (0, n // 2, n - 1):
        back = cur.execute('select e from t where rowid = ?', (i,)).fetchone()[0]
        assert back == vectors[i].tobytes()
```

- [ ] **Step 2: Run the file and confirm all tests pass**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=bindings/python python -m pytest bindings/python/vectorlite_py/test/test_vector_roundtrip.py -q
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add bindings/python/vectorlite_py/test/test_vector_roundtrip.py
git commit -m "test: characterize vector column round-trip and quantization tolerances"
```

---

### Task 4: KNN query semantics (`test_knn_query.py`)

**Files:**
- Create: `bindings/python/vectorlite_py/test/test_knn_query.py`

Covers: exact-match nearest, full ordering vs brute force (ef >= N), distance
column vs numpy, k>N caps, ef parameter, empty table, rowid filters, multiple
knn_search.

- [ ] **Step 1: Write the test file**

```python
import sqlite3
import numpy as np
import pytest
from vectorlite_py.test.helpers import random_vectors, brute_force_knn, l2_squared

DIM = 16


def _fill(cur, vectors, space='l2', name='t', max_elements=None):
    n = len(vectors)
    max_elements = max_elements or n
    space_clause = '' if space in ('l2', '') else f' {space}'
    cur.execute(
        f'create virtual table {name} using vectorlite('
        f'e float32[{DIM}]{space_clause}, hnsw(max_elements={max_elements}, random_seed=42))')
    for i in range(n):
        cur.execute(f'insert into {name}(rowid, e) values (?, ?)', (i, vectors[i].tobytes()))


@pytest.mark.parametrize('space', ['l2', 'cosine'])
def test_exact_match_is_nearest_with_zero_distance(conn, space):
    vectors = random_vectors(np.random.default_rng(21), 50, DIM)
    cur = conn.cursor()
    _fill(cur, vectors, space=space)
    for probe in (0, 17, 49):
        row = cur.execute(
            'select rowid, distance from t where knn_search(e, knn_param(?, ?))',
            (vectors[probe].tobytes(), 1)).fetchone()
        assert row[0] == probe
        assert np.isclose(row[1], 0.0, atol=1e-4)


def test_full_ordering_matches_brute_force_with_high_ef(conn):
    n = 30
    vectors = random_vectors(np.random.default_rng(22), n, DIM)
    cur = conn.cursor()
    _fill(cur, vectors, space='l2')
    query = np.float32(np.random.default_rng(99).random(DIM))
    # ef >= n makes HNSW return the exact k nearest neighbors.
    result = cur.execute(
        'select rowid, distance from t where knn_search(e, knn_param(?, ?, ?))',
        (query.tobytes(), n, 200)).fetchall()
    expected = brute_force_knn(vectors, query, n, space='l2')
    assert [r[0] for r in result] == [e[0] for e in expected]
    for (rowid, dist), (_, exp_dist) in zip(result, expected):
        assert np.isclose(dist, exp_dist, rtol=1e-4, atol=1e-4)


def test_distance_column_matches_numpy(conn):
    vectors = random_vectors(np.random.default_rng(23), 40, DIM)
    cur = conn.cursor()
    _fill(cur, vectors, space='l2')
    query = vectors[3]
    result = cur.execute(
        'select rowid, distance from t where knn_search(e, knn_param(?, ?, ?))',
        (query.tobytes(), 10, 100)).fetchall()
    for rowid, dist in result:
        assert np.isclose(dist, l2_squared(query, vectors[rowid]), rtol=1e-4, atol=1e-4)


def test_k_greater_than_n_returns_all_rows(conn):
    n = 7
    vectors = random_vectors(np.random.default_rng(24), n, DIM)
    cur = conn.cursor()
    _fill(cur, vectors, space='l2', max_elements=100)
    result = cur.execute(
        'select rowid from t where knn_search(e, knn_param(?, ?))',
        (vectors[0].tobytes(), 1000)).fetchall()
    assert len(result) == n


def test_k_must_be_positive(conn):
    vectors = random_vectors(np.random.default_rng(25), 5, DIM)
    cur = conn.cursor()
    _fill(cur, vectors, space='l2', max_elements=10)
    for bad_k in (0, -1):
        with pytest.raises(sqlite3.OperationalError):
            cur.execute('select rowid from t where knn_search(e, knn_param(?, ?))',
                        (vectors[0].tobytes(), bad_k)).fetchall()


def test_ef_must_be_positive(conn):
    vectors = random_vectors(np.random.default_rng(26), 5, DIM)
    cur = conn.cursor()
    _fill(cur, vectors, space='l2', max_elements=10)
    with pytest.raises(sqlite3.OperationalError):
        cur.execute('select rowid from t where knn_search(e, knn_param(?, ?, ?))',
                    (vectors[0].tobytes(), 3, 0)).fetchall()


def test_knn_param_rejects_bad_arg_counts(conn):
    vectors = random_vectors(np.random.default_rng(27), 5, DIM)
    cur = conn.cursor()
    _fill(cur, vectors, space='l2', max_elements=10)
    with pytest.raises(sqlite3.OperationalError):
        cur.execute('select rowid from t where knn_search(e, knn_param(?))',
                    (vectors[0].tobytes(),)).fetchall()


def test_query_on_empty_table_returns_nothing(conn):
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(e float32[{DIM}], hnsw(max_elements=10))')
    result = cur.execute('select rowid from t where knn_search(e, knn_param(?, ?))',
                         (np.float32(np.random.default_rng(0).random(DIM)).tobytes(), 5)).fetchall()
    assert result == []


def test_rowid_in_filter_limits_candidates(conn):
    vectors = random_vectors(np.random.default_rng(28), 50, DIM)
    cur = conn.cursor()
    _fill(cur, vectors, space='l2')
    result = cur.execute(
        'select rowid from t where knn_search(e, knn_param(?, ?)) and rowid in (1,2,3,4,5)',
        (vectors[1].tobytes(), 10)).fetchall()
    rowids = set(r[0] for r in result)
    assert rowids <= {1, 2, 3, 4, 5}
    assert 1 in rowids


def test_plain_rowid_filter_without_knn(conn):
    vectors = random_vectors(np.random.default_rng(29), 20, DIM)
    cur = conn.cursor()
    _fill(cur, vectors, space='l2')
    result = cur.execute('select rowid from t where rowid == 1 or rowid == 2 order by rowid').fetchall()
    assert [r[0] for r in result] == [1, 2]


def test_multiple_knn_search_unions_results(conn):
    vectors = random_vectors(np.random.default_rng(30), 50, DIM)
    cur = conn.cursor()
    _fill(cur, vectors, space='l2')
    combined = cur.execute(
        'select rowid from t where knn_search(e, knn_param(?, ?)) or knn_search(e, knn_param(?, ?))',
        (vectors[0].tobytes(), 10, vectors[1].tobytes(), 10)).fetchall()
    n0 = cur.execute('select rowid from t where knn_search(e, knn_param(?, ?))',
                     (vectors[0].tobytes(), 10)).fetchall()
    n1 = cur.execute('select rowid from t where knn_search(e, knn_param(?, ?))',
                     (vectors[1].tobytes(), 10)).fetchall()
    assert set(r[0] for r in combined) == set(r[0] for r in n0) | set(r[0] for r in n1)


def test_unconstrained_scan_is_rejected(conn):
    vectors = random_vectors(np.random.default_rng(31), 5, DIM)
    cur = conn.cursor()
    _fill(cur, vectors, space='l2', max_elements=10)
    # vectorlite requires a knn_search or rowid constraint on every query.
    with pytest.raises(sqlite3.OperationalError):
        cur.execute('select rowid from t').fetchall()
```

- [ ] **Step 2: Run the file and confirm all tests pass**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=bindings/python python -m pytest bindings/python/vectorlite_py/test/test_knn_query.py -q
```

Expected: all tests PASS. If `test_full_ordering_matches_brute_force_with_high_ef`
is ever flaky, the `ef` value (200) should remain >= the number of elements.

- [ ] **Step 3: Commit**

```bash
git add bindings/python/vectorlite_py/test/test_knn_query.py
git commit -m "test: characterize knn_search ordering, parameters, and filters"
```

---

### Task 5: DDL and option parsing (`test_ddl_options.py`)

**Files:**
- Create: `bindings/python/vectorlite_py/test/test_ddl_options.py`

Covers: all element types x spaces, hnsw option parsing, missing/invalid
options, malformed column syntax.

- [ ] **Step 1: Write the test file**

```python
import sqlite3
import numpy as np
import pytest
from vectorlite_py.test.helpers import get_connection, ELEMENT_TYPES, SPACES

DIM = 8


@pytest.mark.parametrize('vector_type', ELEMENT_TYPES)
@pytest.mark.parametrize('space', SPACES)
def test_create_insert_query_for_every_type_and_space(vector_type, space):
    conn = get_connection()
    cur = conn.cursor()
    space_clause = '' if space == '' else f' {space}'
    cur.execute(
        f'create virtual table t using vectorlite('
        f'e {vector_type}[{DIM}]{space_clause}, hnsw(max_elements=10))')
    v = np.float32(np.random.default_rng(1).random(DIM))
    cur.execute('insert into t(rowid, e) values (?, ?)', (0, v.tobytes()))
    result = cur.execute('select rowid from t where knn_search(e, knn_param(?, ?))',
                         (v.tobytes(), 1)).fetchall()
    assert result[0][0] == 0
    conn.close()


@pytest.mark.parametrize('options', [
    'max_elements=100',
    'max_elements=100, M=8',
    'max_elements=100, ef_construction=50',
    'max_elements=100, random_seed=7',
    'max_elements=100, allow_replace_deleted=true',
    'max_elements=100, allow_replace_deleted=false',
    'max_elements=100, M=16, ef_construction=100, random_seed=1, allow_replace_deleted=true',
])
def test_valid_hnsw_options_are_accepted(conn, options):
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(e float32[{DIM}], hnsw({options}))')
    v = np.float32(np.random.default_rng(2).random(DIM))
    cur.execute('insert into t(rowid, e) values (?, ?)', (0, v.tobytes()))
    assert cur.execute('select rowid from t where knn_search(e, knn_param(?, ?))',
                       (v.tobytes(), 1)).fetchone()[0] == 0


def test_missing_max_elements_is_rejected(conn):
    with pytest.raises(sqlite3.OperationalError):
        conn.cursor().execute(f'create virtual table t using vectorlite(e float32[{DIM}], hnsw(M=16))')


def test_unknown_option_key_is_rejected(conn):
    with pytest.raises(sqlite3.OperationalError):
        conn.cursor().execute(
            f'create virtual table t using vectorlite(e float32[{DIM}], hnsw(max_elements=10, bogus=1))')


def test_non_numeric_option_value_is_rejected(conn):
    with pytest.raises(sqlite3.OperationalError):
        conn.cursor().execute(
            f'create virtual table t using vectorlite(e float32[{DIM}], hnsw(max_elements=abc))')


def test_trailing_garbage_in_options_is_rejected(conn):
    with pytest.raises(sqlite3.OperationalError):
        conn.cursor().execute(
            f'create virtual table t using vectorlite(e float32[{DIM}], hnsw(max_elements=10, gibberish))')


def test_non_hnsw_index_is_rejected(conn):
    with pytest.raises(sqlite3.OperationalError):
        conn.cursor().execute(f'create virtual table t using vectorlite(e float32[{DIM}], flat(max_elements=10))')


def test_unknown_element_type_is_rejected(conn):
    with pytest.raises(sqlite3.OperationalError):
        conn.cursor().execute(f'create virtual table t using vectorlite(e int8[{DIM}], hnsw(max_elements=10))')


def test_unknown_space_is_rejected(conn):
    with pytest.raises(sqlite3.OperationalError):
        conn.cursor().execute(
            f'create virtual table t using vectorlite(e float32[{DIM}] manhattan, hnsw(max_elements=10))')


def test_missing_dimension_is_rejected(conn):
    with pytest.raises(sqlite3.OperationalError):
        conn.cursor().execute('create virtual table t using vectorlite(e float32, hnsw(max_elements=10))')


def test_three_argument_create_is_rejected(conn):
    with pytest.raises(sqlite3.OperationalError):
        conn.cursor().execute(
            f"create virtual table t using vectorlite(e float32[{DIM}], hnsw(max_elements=10), 'index.bin')")


def test_command_columns_are_hidden(conn):
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(e float32[{DIM}], hnsw(max_elements=10))')
    cur.execute('insert into t(rowid, e) values (?, ?)',
                (0, np.float32(np.random.default_rng(3).random(DIM)).tobytes()))
    cur.execute('select * from t where rowid = 0')
    column_names = [d[0] for d in cur.description]
    assert 'operation' not in column_names
    assert 'path' not in column_names
```

- [ ] **Step 2: Run the file and confirm all tests pass**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=bindings/python python -m pytest bindings/python/vectorlite_py/test/test_ddl_options.py -q
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add bindings/python/vectorlite_py/test/test_ddl_options.py
git commit -m "test: characterize DDL element types, spaces, and hnsw option parsing"
```

---

### Task 6: DML edge cases (`test_dml.py`)

**Files:**
- Create: `bindings/python/vectorlite_py/test/test_dml.py`

Covers: update, duplicate rowid, delete-then-search, capacity limits,
allow_replace_deleted semantics, wrong-dim insert, large batch.

- [ ] **Step 1: Write the test file**

```python
import sqlite3
import numpy as np
import pytest
from vectorlite_py.test.helpers import random_vectors

DIM = 8


def _vec(seed):
    return np.float32(np.random.default_rng(seed).random(DIM))


def test_update_changes_stored_vector_and_search(conn):
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(e float32[{DIM}], hnsw(max_elements=10))')
    cur.execute('insert into t(rowid, e) values (?, ?)', (5, _vec(1).tobytes()))
    new = _vec(2)
    cur.execute('update t set e = ? where rowid = 5', (new.tobytes(),))
    back = cur.execute('select e from t where rowid = 5').fetchone()[0]
    assert back == new.tobytes()
    row = cur.execute('select rowid, distance from t where knn_search(e, knn_param(?, ?))',
                      (new.tobytes(), 1)).fetchone()
    assert row[0] == 5 and np.isclose(row[1], 0.0, atol=1e-4)


def test_duplicate_rowid_insert_is_rejected(conn):
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(e float32[{DIM}], hnsw(max_elements=10))')
    cur.execute('insert into t(rowid, e) values (?, ?)', (0, _vec(1).tobytes()))
    with pytest.raises(sqlite3.OperationalError):
        cur.execute('insert into t(rowid, e) values (?, ?)', (0, _vec(2).tobytes()))


def test_delete_removes_row_from_search(conn):
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(e float32[{DIM}], hnsw(max_elements=10))')
    v = _vec(1)
    cur.execute('insert into t(rowid, e) values (?, ?)', (0, v.tobytes()))
    assert cur.execute('select rowid from t where rowid = 0').fetchall() == [(0,)]
    cur.execute('delete from t where rowid = 0')
    assert cur.execute('select rowid from t where rowid = 0').fetchall() == []
    assert cur.execute('select rowid from t where knn_search(e, knn_param(?, ?))',
                       (v.tobytes(), 5)).fetchall() == []


def test_insert_beyond_capacity_is_rejected(conn):
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(e float32[{DIM}], hnsw(max_elements=2))')
    cur.execute('insert into t(rowid, e) values (?, ?)', (0, _vec(1).tobytes()))
    cur.execute('insert into t(rowid, e) values (?, ?)', (1, _vec(2).tobytes()))
    with pytest.raises(sqlite3.OperationalError):
        cur.execute('insert into t(rowid, e) values (?, ?)', (2, _vec(3).tobytes()))


def test_allow_replace_deleted_true_reuses_slot(conn):
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(e float32[{DIM}], hnsw(max_elements=2))')
    cur.execute('insert into t(rowid, e) values (?, ?)', (0, _vec(1).tobytes()))
    cur.execute('insert into t(rowid, e) values (?, ?)', (1, _vec(2).tobytes()))
    cur.execute('delete from t where rowid = 0')
    # Default allow_replace_deleted=true lets a new rowid reuse the freed slot.
    cur.execute('insert into t(rowid, e) values (?, ?)', (2, _vec(3).tobytes()))
    rowids = set(r[0] for r in cur.execute(
        'select rowid from t where knn_search(e, knn_param(?, ?))', (_vec(3).tobytes(), 5)).fetchall())
    assert rowids == {1, 2}


def test_allow_replace_deleted_false_rejects_reuse(conn):
    cur = conn.cursor()
    cur.execute(
        f'create virtual table t using vectorlite(e float32[{DIM}], '
        f'hnsw(max_elements=2, allow_replace_deleted=false))')
    cur.execute('insert into t(rowid, e) values (?, ?)', (0, _vec(1).tobytes()))
    cur.execute('insert into t(rowid, e) values (?, ?)', (1, _vec(2).tobytes()))
    cur.execute('delete from t where rowid = 0')
    with pytest.raises(sqlite3.OperationalError):
        cur.execute('insert into t(rowid, e) values (?, ?)', (2, _vec(3).tobytes()))


def test_wrong_dimension_insert_is_rejected(conn):
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(e float32[{DIM}], hnsw(max_elements=10))')
    wrong = np.float32(np.random.default_rng(1).random(DIM + 1))
    with pytest.raises(sqlite3.OperationalError):
        cur.execute('insert into t(rowid, e) values (?, ?)', (0, wrong.tobytes()))


def test_large_batch_insert_and_search(conn):
    n = 1000
    vectors = random_vectors(np.random.default_rng(50), n, DIM)
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(e float32[{DIM}], hnsw(max_elements={n}))')
    for i in range(n):
        cur.execute('insert into t(rowid, e) values (?, ?)', (i, vectors[i].tobytes()))
    result = cur.execute('select rowid, distance from t where knn_search(e, knn_param(?, ?))',
                         (vectors[123].tobytes(), 10)).fetchall()
    assert len(result) == 10
    assert result[0][0] == 123 and np.isclose(result[0][1], 0.0, atol=1e-4)
```

- [ ] **Step 2: Run the file and confirm all tests pass**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=bindings/python python -m pytest bindings/python/vectorlite_py/test/test_dml.py -q
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add bindings/python/vectorlite_py/test/test_dml.py
git commit -m "test: characterize DML edge cases (update, capacity, replace-deleted)"
```

---

### Task 7: Persistence and lifecycle (`test_persistence.py`)

**Files:**
- Create: `bindings/python/vectorlite_py/test/test_persistence.py`

Covers: save/load across element types, save empty, overwrite save, load
errors, multi-table, file-backed reopen, rename, vacuum.

- [ ] **Step 1: Write the test file**

```python
import os
import tempfile
import sqlite3
import numpy as np
import pytest
from vectorlite_py.test.helpers import get_connection, random_vectors, ELEMENT_TYPES

DIM = 16


@pytest.mark.parametrize('vector_type', ELEMENT_TYPES)
def test_save_and_load_round_trip(vector_type):
    n = 200
    vectors = random_vectors(np.random.default_rng(60), n, DIM)
    with tempfile.TemporaryDirectory() as d:
        index_path = os.path.join(d, 'index.bin')

        conn = get_connection()
        cur = conn.cursor()
        cur.execute(f'create virtual table t using vectorlite(e {vector_type}[{DIM}], hnsw(max_elements={n}))')
        for i in range(n):
            cur.execute('insert into t(rowid, e) values (?, ?)', (i, vectors[i].tobytes()))
        before = cur.execute('select rowid, distance from t where knn_search(e, knn_param(?, ?))',
                             (vectors[0].tobytes(), 10)).fetchall()
        cur.execute('insert into t(operation, path) values (?, ?)', ('save', index_path))
        assert os.path.getsize(index_path) > 0
        conn.close()

        conn = get_connection()
        cur = conn.cursor()
        cur.execute(f'create virtual table r using vectorlite(e {vector_type}[{DIM}], hnsw(max_elements={n}))')
        assert cur.execute('select rowid from r where knn_search(e, knn_param(?, ?))',
                           (vectors[0].tobytes(), 10)).fetchall() == []
        cur.execute('insert into r(operation, path) values (?, ?)', ('load', index_path))
        after = cur.execute('select rowid, distance from r where knn_search(e, knn_param(?, ?))',
                           (vectors[0].tobytes(), 10)).fetchall()
        assert after == before
        conn.close()


def test_save_empty_index_then_load(conn):
    with tempfile.TemporaryDirectory() as d:
        index_path = os.path.join(d, 'empty.bin')
        cur = conn.cursor()
        cur.execute(f'create virtual table t using vectorlite(e float32[{DIM}], hnsw(max_elements=10))')
        cur.execute('insert into t(operation, path) values (?, ?)', ('save', index_path))
        assert os.path.exists(index_path)

        cur.execute(f'create virtual table r using vectorlite(e float32[{DIM}], hnsw(max_elements=10))')
        cur.execute('insert into r(operation, path) values (?, ?)', ('load', index_path))
        q = np.float32(np.random.default_rng(0).random(DIM)).tobytes()
        assert cur.execute('select rowid from r where knn_search(e, knn_param(?, ?))', (q, 5)).fetchall() == []


def test_overwrite_save_reflects_latest_data(conn):
    with tempfile.TemporaryDirectory() as d:
        index_path = os.path.join(d, 'index.bin')
        vectors = random_vectors(np.random.default_rng(61), 20, DIM)
        cur = conn.cursor()
        cur.execute(f'create virtual table t using vectorlite(e float32[{DIM}], hnsw(max_elements=100))')
        for i in range(5):
            cur.execute('insert into t(rowid, e) values (?, ?)', (i, vectors[i].tobytes()))
        cur.execute('insert into t(operation, path) values (?, ?)', ('save', index_path))
        for i in range(5, 15):
            cur.execute('insert into t(rowid, e) values (?, ?)', (i, vectors[i].tobytes()))
        cur.execute('insert into t(operation, path) values (?, ?)', ('save', index_path))

        cur.execute(f'create virtual table r using vectorlite(e float32[{DIM}], hnsw(max_elements=100))')
        cur.execute('insert into r(operation, path) values (?, ?)', ('load', index_path))
        rowids = set(x[0] for x in cur.execute(
            'select rowid from r where knn_search(e, knn_param(?, ?))', (vectors[0].tobytes(), 100)).fetchall())
        assert rowids == set(range(15))


def test_load_replaces_existing_contents(conn):
    with tempfile.TemporaryDirectory() as d:
        index_path = os.path.join(d, 'index.bin')
        vectors = random_vectors(np.random.default_rng(62), 200, DIM)
        cur = conn.cursor()
        cur.execute(f'create virtual table src using vectorlite(e float32[{DIM}], hnsw(max_elements=1000))')
        for i in range(10):
            cur.execute('insert into src(rowid, e) values (?, ?)', (i, vectors[i].tobytes()))
        cur.execute('insert into src(operation, path) values (?, ?)', ('save', index_path))

        cur.execute(f'create virtual table dst using vectorlite(e float32[{DIM}], hnsw(max_elements=1000))')
        for i in range(100, 110):
            cur.execute('insert into dst(rowid, e) values (?, ?)', (i, vectors[i].tobytes()))
        cur.execute('insert into dst(operation, path) values (?, ?)', ('load', index_path))
        rowids = set(x[0] for x in cur.execute(
            'select rowid from dst where knn_search(e, knn_param(?, ?))', (vectors[0].tobytes(), 100)).fetchall())
        assert rowids == set(range(10))


def test_load_missing_file_is_rejected(conn):
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(e float32[{DIM}], hnsw(max_elements=10))')
    with pytest.raises(sqlite3.OperationalError):
        cur.execute('insert into t(operation, path) values (?, ?)', ('load', '/no/such/index.bin'))


def test_load_dimension_mismatch_is_rejected(conn):
    with tempfile.TemporaryDirectory() as d:
        index_path = os.path.join(d, 'index.bin')
        vectors = random_vectors(np.random.default_rng(63), 10, DIM)
        cur = conn.cursor()
        cur.execute(f'create virtual table src using vectorlite(e float32[{DIM}], hnsw(max_elements=100))')
        for i in range(10):
            cur.execute('insert into src(rowid, e) values (?, ?)', (i, vectors[i].tobytes()))
        cur.execute('insert into src(operation, path) values (?, ?)', ('save', index_path))

        cur.execute(f'create virtual table dst using vectorlite(e float32[{DIM * 2}], hnsw(max_elements=100))')
        with pytest.raises(sqlite3.OperationalError):
            cur.execute('insert into dst(operation, path) values (?, ?)', ('load', index_path))


def test_load_element_type_mismatch_is_rejected(conn):
    with tempfile.TemporaryDirectory() as d:
        index_path = os.path.join(d, 'index.bin')
        vectors = random_vectors(np.random.default_rng(64), 10, DIM)
        cur = conn.cursor()
        cur.execute(f'create virtual table src using vectorlite(e float32[{DIM}], hnsw(max_elements=100))')
        for i in range(10):
            cur.execute('insert into src(rowid, e) values (?, ?)', (i, vectors[i].tobytes()))
        cur.execute('insert into src(operation, path) values (?, ?)', ('save', index_path))

        cur.execute(f'create virtual table dst using vectorlite(e float16[{DIM}], hnsw(max_elements=100))')
        with pytest.raises(sqlite3.OperationalError):
            cur.execute('insert into dst(operation, path) values (?, ?)', ('load', index_path))


def test_unknown_operation_is_rejected(conn):
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(e float32[{DIM}], hnsw(max_elements=10))')
    with pytest.raises(sqlite3.OperationalError):
        cur.execute('insert into t(operation, path) values (?, ?)', ('frobnicate', '/tmp/x.bin'))


def test_two_tables_save_load_independently(conn):
    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, 'a.bin')
        p2 = os.path.join(d, 'b.bin')
        vectors = random_vectors(np.random.default_rng(65), 40, DIM)
        cur = conn.cursor()
        cur.execute(f'create virtual table a using vectorlite(e float32[{DIM}], hnsw(max_elements=100))')
        cur.execute(f'create virtual table b using vectorlite(e float32[{DIM}], hnsw(max_elements=100))')
        for i in range(10):
            cur.execute('insert into a(rowid, e) values (?, ?)', (i, vectors[i].tobytes()))
        for i in range(20, 30):
            cur.execute('insert into b(rowid, e) values (?, ?)', (i, vectors[i].tobytes()))
        cur.execute('insert into a(operation, path) values (?, ?)', ('save', p1))
        cur.execute('insert into b(operation, path) values (?, ?)', ('save', p2))

        cur.execute(f'create virtual table ra using vectorlite(e float32[{DIM}], hnsw(max_elements=100))')
        cur.execute('insert into ra(operation, path) values (?, ?)', ('load', p1))
        rowids = set(x[0] for x in cur.execute(
            'select rowid from ra where knn_search(e, knn_param(?, ?))', (vectors[0].tobytes(), 100)).fetchall())
        assert rowids == set(range(10))


def test_file_backed_index_is_not_auto_persisted():
    # The HNSW index lives in memory; a file-backed DB does not persist it
    # automatically. A fresh connection sees an empty index until 'load'.
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, 'x.db')
        vectors = random_vectors(np.random.default_rng(66), 5, DIM)
        c = get_connection(db_path)
        cur = c.cursor()
        cur.execute(f'create virtual table t using vectorlite(e float32[{DIM}], hnsw(max_elements=10))')
        for i in range(5):
            cur.execute('insert into t(rowid, e) values (?, ?)', (i, vectors[i].tobytes()))
        assert len(cur.execute('select rowid from t where knn_search(e, knn_param(?, ?))',
                               (vectors[0].tobytes(), 10)).fetchall()) == 5
        c.close()

        c2 = get_connection(db_path)
        cur2 = c2.cursor()
        assert cur2.execute('select rowid from t where knn_search(e, knn_param(?, ?))',
                           (vectors[0].tobytes(), 10)).fetchall() == []
        c2.close()


def test_index_survives_rename(conn):
    vectors = random_vectors(np.random.default_rng(67), 20, DIM)
    cur = conn.cursor()
    cur.execute(f'create virtual table surv using vectorlite(e float32[{DIM}], hnsw(max_elements=100))')
    for i in range(20):
        cur.execute('insert into surv(rowid, e) values (?, ?)', (i, vectors[i].tobytes()))
    cur.execute('alter table surv rename to renamed')
    assert len(cur.execute('select rowid from renamed where knn_search(e, knn_param(?, ?))',
                          (vectors[0].tobytes(), 20)).fetchall()) == 20


def test_index_survives_vacuum(conn):
    vectors = random_vectors(np.random.default_rng(68), 20, DIM)
    cur = conn.cursor()
    cur.execute(f'create virtual table surv using vectorlite(e float32[{DIM}], hnsw(max_elements=100))')
    for i in range(20):
        cur.execute('insert into surv(rowid, e) values (?, ?)', (i, vectors[i].tobytes()))
    cur.execute('vacuum')
    assert len(cur.execute('select rowid from surv where knn_search(e, knn_param(?, ?))',
                          (vectors[0].tobytes(), 20)).fetchall()) == 20
```

- [ ] **Step 2: Run the file and confirm all tests pass**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=bindings/python python -m pytest bindings/python/vectorlite_py/test/test_persistence.py -q
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add bindings/python/vectorlite_py/test/test_persistence.py
git commit -m "test: characterize save/load persistence and table lifecycle"
```

---

### Task 8: Full-suite verification

**Files:** none (verification only).

- [ ] **Step 1: Run the entire Python test suite**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=bindings/python python -m pytest bindings/python/vectorlite_py/test -q
```

Expected: every test passes (existing `vectorlite_test.py` plus all new files).
If an `L2DistanceSquared`-style random-vector check causes a one-off failure,
re-run once before treating it as a regression.

- [ ] **Step 2: Confirm no source files were modified**

Run:

```bash
git status --porcelain vectorlite/
```

Expected: empty output (this is a test-only change).

---

## Self-Review

**Spec coverage:**
- Numerical correctness → Task 2 (`vector_distance` vs numpy), Task 4 (distance column).
- Query semantics → Task 4.
- DDL & option parsing → Task 5.
- Persistence & lifecycle → Task 7.
- DML edge cases → Task 6.
- Data integrity → Task 3.
- Error handling → Tasks 2, 4, 5, 6, 7 (each has explicit `pytest.raises`).

All spec categories map to at least one task.

**Placeholder scan:** No TBD/TODO; every step contains complete code or an exact command.

**Type/name consistency:** Helper names (`get_connection`, `random_vectors`,
`brute_force_knn`, `l2_squared`, `ip_distance`, `cosine_distance`,
`DEQUANT_RTOL`, `ELEMENT_TYPES`, `SPACES`, `SEED`) defined in Task 1 are used
consistently in Tasks 2-7. `get_connection` accepts an optional `path` argument
so the file-backed persistence test reuses it. Every database error is asserted
as `sqlite3.OperationalError`.

**Verified behaviors:** All asserted error types/messages and outcomes were
confirmed empirically against the built extension using stdlib `sqlite3`
(SQLite 3.50.4) before writing this plan.
