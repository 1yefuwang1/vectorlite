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
