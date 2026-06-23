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
