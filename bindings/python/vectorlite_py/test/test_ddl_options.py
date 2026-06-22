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
