import vectorlite_py
import apsw
import pytest
import numpy as np
import json
import tempfile
import os
import platform

def get_connection():
    conn = apsw.Connection(':memory:')
    conn.enable_load_extension(True)
    conn.load_extension(vectorlite_py.vectorlite_path())
    return conn

@pytest.fixture(scope='module')
def conn() -> None:
    return get_connection()

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
    assert 'Best SIMD target in use:' in output[0]

def test_virtual_table_happy_path(conn, random_vectors):
    # Note: if space is '', it will be treated as 'l2'
    spaces = ['l2', 'ip', 'cosine', '']
    def test_with_space(space):
        cur = conn.cursor()
        cur.execute(f'create virtual table x using vectorlite(my_embedding float32[{DIM}] {space}, hnsw(max_elements={NUM_ELEMENTS}))')

        for i in range(NUM_ELEMENTS):
            cur.execute('insert into x (rowid, my_embedding) values (?, ?)', (i, random_vectors[i].tobytes()))
        
        # a vector will be normalized if space is cosine
        if space != 'cosine':
            result = cur.execute('select my_embedding from x where rowid = 0').fetchone()
            assert result[0] == random_vectors[0].tobytes()

        cur.execute('delete from x where rowid = 0')
        result = cur.execute('select my_embedding from x where rowid = 0').fetchone()
        assert result is None

        cur.execute('insert into x (rowid, my_embedding) values (?, ?)', (0, random_vectors[0].tobytes()))
        # a vector will be normalized if space is cosine
        if space != 'cosine':
            result = cur.execute('select my_embedding from x where rowid = 0').fetchone()
            assert result[0] == random_vectors[0].tobytes()

        result = cur.execute('select rowid, distance from x where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10)).fetchall()
        assert len(result) == 10

        result = cur.execute('select rowid, distance from x where knn_search(my_embedding, knn_param(?, ?)) and rowid in (1,2,3,4,5)', (random_vectors[1].tobytes(), 10)).fetchall()
        # although we are searching for 10 nearest neighbors, rowid filter only has 5 elements
        # Note that inner product is not an actual metric. An element can be closer to some other element than to itself. 
        if space != 'ip':
            assert len(result) == 5 and all([r[0] in (1, 2, 3, 4, 5) for r in result]) and result[0][0] == 1

        # test if multiple rowid filters work
        result = cur.execute('select rowid, distance from x where rowid == 1 or rowid == 2 order by rowid').fetchall()
        assert len(result) == 2 and result[0][0] == 1 and result[1][0] == 2

        # test if multiple knn_search works
        result = cur.execute('select rowid, distance from x where knn_search(my_embedding, knn_param(?, ?)) or knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10, random_vectors[1].tobytes(), 10)).fetchall()
        vector1_neighbors = cur.execute('select rowid, distance from x where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10)).fetchall()
        vector2_neighbors = cur.execute('select rowid, distance from x where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[1].tobytes(), 10)).fetchall()
        assert set([r[0] for r in result]) == set([r[0] for r in vector1_neighbors] + [r[0] for r in vector2_neighbors])

        # test if multiple knn_search works with rowid filter
        result = cur.execute('select rowid, distance from x where (knn_search(my_embedding, knn_param(?, ?)) and rowid in (1,2,3)) or (knn_search(my_embedding, knn_param(?, ?)) and rowid in (4,5,6))', (random_vectors[0].tobytes(), 10, random_vectors[1].tobytes(), 10)).fetchall()
        vector1_neighbors = cur.execute('select rowid, distance from x where knn_search(my_embedding, knn_param(?, ?)) and rowid in (1,2,3)', (random_vectors[0].tobytes(), 10)).fetchall()
        vector2_neighbors = cur.execute('select rowid, distance from x where knn_search(my_embedding, knn_param(?, ?)) and rowid in (4,5,6)', (random_vectors[1].tobytes(), 10)).fetchall()
        assert set([r[0] for r in result]) == set([r[0] for r in vector1_neighbors] + [r[0] for r in vector2_neighbors])

        cur.execute('drop table x')

    for space in spaces:
        test_with_space(space)

def test_read_vector_column_quantized(conn, random_vectors):
    # Reading the vector column back from a quantized (bfloat16/float16) table
    # must return the dequantized f32 vector, not garbage. Regression test for
    # GetVectorByRowid hardcoding getDataByLabel<float>.
    tolerances = {'bfloat16': 1e-2, 'float16': 1e-3}
    for vector_type, rtol in tolerances.items():
        cur = conn.cursor()
        cur.execute(f'create virtual table vq using vectorlite(my_embedding {vector_type}[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
        cur.execute('insert into vq (rowid, my_embedding) values (?, ?)', (0, random_vectors[0].tobytes()))
        result = cur.execute('select my_embedding from vq where rowid = 0').fetchone()
        assert result is not None
        read_back = np.frombuffer(result[0], dtype=np.float32)
        assert read_back.shape == (DIM,)
        assert np.allclose(read_back, random_vectors[0], rtol=rtol, atol=rtol)
        cur.execute('drop table vq')

def test_json_happy_path(conn):
    cur = conn.cursor()
    vector = np.float32(np.random.random(DIM))
    vec = cur.execute('select vector_from_json(vector_to_json(?))', (vector.tobytes(),)).fetchone()[0]
    assert np.allclose(vector, np.frombuffer(vec, dtype=np.float32))

    vec = cur.execute('select vector_from_json(?)', (json.dumps(vector.tolist()),)).fetchone()[0]
    assert np.allclose(vector, np.frombuffer(vec, dtype=np.float32))

def test_vector_distance(conn):
    vec1 = np.float32(np.random.random(DIM))
    vec2 = np.float32(np.random.random(DIM))

    inner_product_distance = 1 - np.dot(vec1, vec2)
    cur = conn.cursor()
    result = cur.execute('select vector_distance(?, ?, "ip")', (vec1.tobytes(), vec2.tobytes())).fetchone()[0]
    assert np.isclose(result, inner_product_distance)

    cosine_distance = 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    result = cur.execute('select vector_distance(?, ?, "cosine")', (vec1.tobytes(), vec2.tobytes())).fetchone()[0]
    assert np.isclose(result, cosine_distance)

    l2_distance = np.linalg.norm(vec1 - vec2)
    result = cur.execute('select vector_distance(?, ?, "l2")', (vec1.tobytes(), vec2.tobytes())).fetchone()[0]
    import math
    # hnswlib doesn't calculate sqaure root of l2 distance
    assert np.isclose(math.sqrt(result), l2_distance)

def test_save_and_load_round_trip(random_vectors):
    with tempfile.TemporaryDirectory() as tempdir:
        index_path = os.path.join(tempdir, 'index.bin')

        for vector_type in ['float32', 'bfloat16', 'float16']:
            assert not os.path.exists(index_path)

            # Build an in-memory index and save it explicitly.
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(f'create virtual table my_table using vectorlite(my_embedding {vector_type}[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
            for i in range(NUM_ELEMENTS):
                cur.execute('insert into my_table (rowid, my_embedding) values (?, ?)', (i, random_vectors[i].tobytes()))

            before = cur.execute('select rowid, distance from my_table where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10)).fetchall()
            assert len(before) == 10

            cur.execute('insert into my_table(operation, path) values (?, ?)', ('save', index_path))
            assert os.path.exists(index_path) and os.path.getsize(index_path) > 0
            conn.close()

            # Load it into a brand new in-memory table without re-inserting data.
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(f'create virtual table reloaded using vectorlite(my_embedding {vector_type}[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
            # The table is empty until we load: a knn_search returns nothing.
            # (vectorlite rejects unconstrained scans like `select count(*)`.)
            assert cur.execute('select rowid from reloaded where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10)).fetchall() == []
            cur.execute('insert into reloaded(operation, path) values (?, ?)', ('load', index_path))

            after = cur.execute('select rowid, distance from reloaded where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10)).fetchall()
            assert after == before
            conn.close()

            os.remove(index_path)


def test_load_replaces_existing_contents(random_vectors):
    with tempfile.TemporaryDirectory() as tempdir:
        index_path = os.path.join(tempdir, 'index.bin')

        conn = get_connection()
        cur = conn.cursor()
        # Save an index containing only rowids 0..9.
        cur.execute(f'create virtual table src using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
        for i in range(10):
            cur.execute('insert into src (rowid, my_embedding) values (?, ?)', (i, random_vectors[i].tobytes()))
        cur.execute('insert into src(operation, path) values (?, ?)', ('save', index_path))

        # A different table with different rowids gets fully replaced by load.
        cur.execute(f'create virtual table dst using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
        for i in range(100, 110):
            cur.execute('insert into dst (rowid, my_embedding) values (?, ?)', (i, random_vectors[i].tobytes()))
        cur.execute('insert into dst(operation, path) values (?, ?)', ('load', index_path))

        rowids = set(r[0] for r in cur.execute('select rowid from dst where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10)).fetchall())
        assert rowids == set(range(10))
        conn.close()


def test_allow_replace_deleted_preserved_after_load(random_vectors):
    # allow_replace_deleted is a runtime-only hnswlib flag that is NOT stored in
    # the serialized index. A loaded table must keep the table's configured value
    # (default true), otherwise inserting into a full index after a delete fails.
    with tempfile.TemporaryDirectory() as tempdir:
        index_path = os.path.join(tempdir, 'index.bin')
        capacity = 16

        conn = get_connection()
        cur = conn.cursor()
        # Fill an index to capacity (default allow_replace_deleted=true) and save.
        cur.execute(f'create virtual table src using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={capacity}))')
        for i in range(capacity):
            cur.execute('insert into src (rowid, my_embedding) values (?, ?)', (i, random_vectors[i].tobytes()))
        cur.execute('insert into src(operation, path) values (?, ?)', ('save', index_path))
        conn.close()

        # Load into a fresh table created with the default allow_replace_deleted=true.
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(f'create virtual table dst using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={capacity}))')
        cur.execute('insert into dst(operation, path) values (?, ?)', ('load', index_path))

        # The index is full. Delete one row, then insert a new rowid: this only
        # succeeds if allow_replace_deleted survived the load (it reuses the slot).
        cur.execute('delete from dst where rowid = 0')
        cur.execute('insert into dst (rowid, my_embedding) values (?, ?)', (capacity, random_vectors[0].tobytes()))

        result = cur.execute('select rowid from dst where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 1)).fetchall()
        assert result[0][0] == capacity
        conn.close()


def test_load_dimension_mismatch_is_rejected(random_vectors):
    with tempfile.TemporaryDirectory() as tempdir:
        index_path = os.path.join(tempdir, 'index.bin')

        conn = get_connection()
        cur = conn.cursor()
        cur.execute(f'create virtual table src using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
        for i in range(10):
            cur.execute('insert into src (rowid, my_embedding) values (?, ?)', (i, random_vectors[i].tobytes()))
        cur.execute('insert into src(operation, path) values (?, ?)', ('save', index_path))

        # Declare a table with a different dimension and insert a marker row.
        cur.execute(f'create virtual table dst using vectorlite(my_embedding float32[{DIM * 2}], hnsw(max_elements={NUM_ELEMENTS}))')
        marker = np.float32(np.random.random(DIM * 2))
        cur.execute('insert into dst (rowid, my_embedding) values (?, ?)', (999, marker.tobytes()))

        with pytest.raises(apsw.SQLError):
            cur.execute('insert into dst(operation, path) values (?, ?)', ('load', index_path))

        # Existing contents are left intact after a rejected load.
        rowids = [r[0] for r in cur.execute('select rowid from dst where knn_search(my_embedding, knn_param(?, ?))', (marker.tobytes(), 1)).fetchall()]
        assert rowids == [999]
        conn.close()


def test_load_missing_file_is_rejected():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
    with pytest.raises(apsw.SQLError):
        cur.execute('insert into t(operation, path) values (?, ?)', ('load', '/no/such/index.bin'))
    conn.close()


def test_unknown_operation_is_rejected():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
    with pytest.raises(apsw.SQLError):
        cur.execute('insert into t(operation, path) values (?, ?)', ('frobnicate', '/tmp/index.bin'))
    conn.close()


def test_three_argument_create_is_rejected():
    conn = get_connection()
    cur = conn.cursor()
    with pytest.raises(apsw.SQLError):
        cur.execute(f"create virtual table t using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}), 'index.bin')")
    conn.close()


def test_command_columns_are_hidden(random_vectors):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f'create virtual table t using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
    cur.execute('insert into t (rowid, my_embedding) values (?, ?)', (0, random_vectors[0].tobytes()))
    # `select *` must not surface operation/path columns.
    cur.execute('select * from t where rowid = 0')
    column_names = [d[0] for d in cur.getdescription()]
    assert 'operation' not in column_names
    assert 'path' not in column_names
    conn.close()
