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
