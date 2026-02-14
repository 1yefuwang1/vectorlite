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

                
