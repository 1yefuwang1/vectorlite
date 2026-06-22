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
