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

def test_index_file(random_vectors):
    def remove_quote(s: str):
        return s.strip('\'').strip('\"')
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, 'index.bin')
        file_paths = [f'\"{file_path}\"', f'\'{file_path}\'']

        for vector_type in ['float32', 'bfloat16', 'float16']:
            for index_file_path in file_paths:
                assert not os.path.exists(remove_quote(index_file_path))

                conn = get_connection()
                cur = conn.cursor()
                cur.execute(f'create virtual table my_table using vectorlite(my_embedding {vector_type}[{DIM}], hnsw(max_elements={NUM_ELEMENTS}), {index_file_path})')

                for i in range(NUM_ELEMENTS):
                    cur.execute('insert into my_table (rowid, my_embedding) values (?, ?)', (i, random_vectors[i].tobytes()))

                result = cur.execute('select rowid, distance from my_table where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10)).fetchall()
                assert len(result) == 10

                conn.close()
                # The index file should be created
                index_file_size = os.path.getsize(remove_quote(index_file_path))
                assert os.path.exists(remove_quote(index_file_path)) and index_file_size > 0

                # test if the index file could be loaded with the same parameters without inserting data again
                conn = get_connection()
                cur = conn.cursor()
                cur.execute(f'create virtual table my_table using vectorlite(my_embedding {vector_type}[{DIM}], hnsw(max_elements={NUM_ELEMENTS}), {index_file_path})')
                result = cur.execute('select rowid, distance from my_table where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10)).fetchall()
                assert len(result) == 10
                conn.close()
                # The index file should be created
                assert os.path.exists(remove_quote(index_file_path)) and os.path.getsize(remove_quote(index_file_path)) == index_file_size

                # test if the index file could be loaded with different hnsw parameters and distance type without inserting data again
                # But hnsw parameters can't be changed even if different values are set, they will be owverwritten by the value from the index file
                # todo: test whether hnsw parameters are overwritten after more functions are introduced to provide runtime stats.
                conn = get_connection()
                cur = conn.cursor()
                cur.execute(f'create virtual table my_table2 using vectorlite(my_embedding {vector_type}[{DIM}] cosine, hnsw(max_elements={NUM_ELEMENTS},ef_construction=32,M=32), {index_file_path})')
                result = cur.execute('select rowid, distance from my_table2 where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10)).fetchall()
                assert len(result) == 10

                # test searching with ef_search = 30, which defaults to 10
                result = cur.execute('select rowid, distance from my_table2 where knn_search(my_embedding, knn_param(?, ?, ?))', (random_vectors[0].tobytes(), 10, 30)).fetchall()
                assert len(result) == 10
                conn.close()
                assert os.path.exists(remove_quote(index_file_path)) and os.path.getsize(remove_quote(index_file_path)) == index_file_size


                # test if `drop table` deletes the index file
                conn = get_connection()
                cur = conn.cursor()
                cur.execute(f'create virtual table my_table2 using vectorlite(my_embedding {vector_type}[{DIM}] cosine, hnsw(max_elements={NUM_ELEMENTS},ef_construction=64,M=32), {index_file_path})')
                result = cur.execute('select rowid, distance from my_table2 where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10)).fetchall()
                assert len(result) == 10

                cur.execute(f'drop table my_table2')
                assert not os.path.exists(remove_quote(index_file_path))
                conn.close()


def get_file_connection(db_path):
    conn = apsw.Connection(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(vectorlite_py.vectorlite_path())
    return conn


def test_shadow_table_basic(random_vectors):
    """Test basic shadow table flow: insert, disconnect, reconnect, query."""
    with tempfile.TemporaryDirectory() as tempdir:
        db_path = os.path.join(tempdir, 'test.db')

        # Create table and insert vectors (no file path = shadow table mode).
        conn = get_file_connection(db_path)
        cur = conn.cursor()
        cur.execute(f'create virtual table sv using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')

        for i in range(NUM_ELEMENTS):
            cur.execute('insert into sv (rowid, my_embedding) values (?, ?)', (i, random_vectors[i].tobytes()))

        # Verify search works before disconnect.
        result = cur.execute('select rowid, distance from sv where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10)).fetchall()
        assert len(result) == 10

        conn.close()

        # Reconnect: should load from shadow table + replay WAL.
        conn = get_file_connection(db_path)
        cur = conn.cursor()
        result = cur.execute('select rowid, distance from sv where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10)).fetchall()
        assert len(result) == 10

        # Verify exact vector retrieval.
        result = cur.execute('select my_embedding from sv where rowid = 0').fetchone()
        assert result[0] == random_vectors[0].tobytes()

        conn.close()


def test_shadow_table_compact(random_vectors):
    """Test that compact serializes index and clears WAL."""
    with tempfile.TemporaryDirectory() as tempdir:
        db_path = os.path.join(tempdir, 'test.db')

        conn = get_file_connection(db_path)
        cur = conn.cursor()
        cur.execute(f'create virtual table sv using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')

        for i in range(100):
            cur.execute('insert into sv (rowid, my_embedding) values (?, ?)', (i, random_vectors[i].tobytes()))

        # Compact.
        cur.execute("insert into sv(command) values('compact')")

        # WAL should be empty after compact.
        wal_count = cur.execute('select count(*) from sv_wal').fetchone()[0]
        assert wal_count == 0

        # Index table should have data.
        index_count = cur.execute('select count(*) from sv_index').fetchone()[0]
        assert index_count > 0

        # Search should still work.
        result = cur.execute('select rowid, distance from sv where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10)).fetchall()
        assert len(result) == 10

        conn.close()

        # Reconnect after compact: should load from snapshot (no WAL replay needed).
        conn = get_file_connection(db_path)
        cur = conn.cursor()
        result = cur.execute('select rowid, distance from sv where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[0].tobytes(), 10)).fetchall()
        assert len(result) == 10
        conn.close()


def test_shadow_table_rebuild(random_vectors):
    """Test that rebuild removes deleted vectors and produces a clean index."""
    with tempfile.TemporaryDirectory() as tempdir:
        db_path = os.path.join(tempdir, 'test.db')

        conn = get_file_connection(db_path)
        cur = conn.cursor()
        cur.execute(f'create virtual table sv using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')

        for i in range(100):
            cur.execute('insert into sv (rowid, my_embedding) values (?, ?)', (i, random_vectors[i].tobytes()))

        # Delete some vectors.
        for i in range(50):
            cur.execute('delete from sv where rowid = ?', (i,))

        # Rebuild: should remove deleted vectors entirely.
        cur.execute("insert into sv(command) values('rebuild')")

        # WAL should be empty.
        wal_count = cur.execute('select count(*) from sv_wal').fetchone()[0]
        assert wal_count == 0

        # Deleted vectors should not be searchable.
        result = cur.execute('select my_embedding from sv where rowid = 0').fetchone()
        assert result is None

        # Remaining vectors should be searchable.
        result = cur.execute('select rowid, distance from sv where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[50].tobytes(), 10)).fetchall()
        assert len(result) == 10
        assert all(r[0] >= 50 for r in result)

        conn.close()


def test_shadow_table_delete_reconnect(random_vectors):
    """Test that deletes persist across reconnect via WAL."""
    with tempfile.TemporaryDirectory() as tempdir:
        db_path = os.path.join(tempdir, 'test.db')

        conn = get_file_connection(db_path)
        cur = conn.cursor()
        cur.execute(f'create virtual table sv using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')

        for i in range(100):
            cur.execute('insert into sv (rowid, my_embedding) values (?, ?)', (i, random_vectors[i].tobytes()))

        cur.execute('delete from sv where rowid = 0')
        conn.close()

        # Reconnect: delete should be replayed from WAL.
        conn = get_file_connection(db_path)
        cur = conn.cursor()
        result = cur.execute('select my_embedding from sv where rowid = 0').fetchone()
        assert result is None

        # Other vectors should still work.
        result = cur.execute('select rowid, distance from sv where knn_search(my_embedding, knn_param(?, ?))', (random_vectors[1].tobytes(), 10)).fetchall()
        assert len(result) == 10

        conn.close()


def test_shadow_table_drop(random_vectors):
    """Test that DROP TABLE removes shadow tables."""
    with tempfile.TemporaryDirectory() as tempdir:
        db_path = os.path.join(tempdir, 'test.db')

        conn = get_file_connection(db_path)
        cur = conn.cursor()
        cur.execute(f'create virtual table sv using vectorlite(my_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')

        for i in range(10):
            cur.execute('insert into sv (rowid, my_embedding) values (?, ?)', (i, random_vectors[i].tobytes()))

        cur.execute('drop table sv')

        # Shadow tables should be gone.
        tables = cur.execute("select name from sqlite_master where type='table'").fetchall()
        table_names = [t[0] for t in tables]
        assert 'sv_index' not in table_names
        assert 'sv_wal' not in table_names

        conn.close()
