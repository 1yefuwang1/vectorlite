import vectorlite_py
import apsw
import pytest
import numpy as np


@pytest.fixture(scope='module')
def conn() -> None:
    conn = apsw.Connection(':memory:')
    conn.enable_load_extension(True)
    conn.load_extension(vectorlite_py.vectorlite_path())
    return conn

DIM = 32
NUM_ELEMENTS = 1000

# Generating sample data
@pytest.fixture(scope='module')
def random_vectors():
    return np.float32(np.random.random((NUM_ELEMENTS, DIM)))

def test_vectorlite_info(conn):
    cur = conn.cursor()
    cur.execute('select vectorlite_info()')
    output = cur.fetchone()
    assert f'vectorlite extension version {vectorlite_py.__version__}' in output[0]

def test_l2_space_with_default_hnsw_param(conn, random_vectors):
    cur = conn.cursor()
    cur.execute(f'create virtual table x using vectorlite(my_embedding({DIM}, "l2"), hnsw(max_elements={NUM_ELEMENTS}))')

    for i in range(NUM_ELEMENTS):
        cur.execute('insert into x (rowid, my_embedding) values (?, ?)', (i, random_vectors[i].tobytes()))
    
    result = cur.execute('select my_embedding from x where rowid = 0').fetchone()
    assert result[0] == random_vectors[0].tobytes()

    cur.execute('delete from x where rowid = 0')
    result = cur.execute('select my_embedding from x where rowid = 0').fetchone()
    assert result is None

    cur.execute('insert into x (rowid, my_embedding) values (?, ?)', (0, random_vectors[0].tobytes()))
    result = cur.execute('select my_embedding from x where rowid = 0').fetchone()
    assert result[0] == random_vectors[0].tobytes()

    result = cur.execute('select rowid, distance from x where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10)).fetchall()
    assert len(result) == 10

    result = cur.execute('select rowid, distance from x where knn_search(my_embedding, knn_param(?, ?)) and rowid in (1,2,3,4,5)', (random_vectors[1].tobytes(), 10)).fetchall()
    # although we are searching for 10 nearest neighbors, rowid filter only has 5 elements
    assert len(result) == 5 and all([r[0] in (1, 2, 3, 4, 5) for r in result]) and result[0][0] == 1
