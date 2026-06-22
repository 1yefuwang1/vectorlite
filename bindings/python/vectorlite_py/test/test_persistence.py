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
