"""SQL regressions for Rust callbacks, registry ownership, and persistence."""

import json
import os
from pathlib import Path
import sqlite3
import stat
import struct
import subprocess
import sys

import pytest


@pytest.fixture
def extension_path():
    suffix = "dll" if sys.platform == "win32" else "dylib" if sys.platform == "darwin" else "so"
    prefix = "" if sys.platform == "win32" else "lib"
    default = Path(__file__).resolve().parents[1] / "target" / "release" / f"{prefix}vectorlite.{suffix}"
    library = Path(os.environ.get("VECTORLITE_RUST_EXTENSION", default))
    if not library.exists():
        pytest.skip(f"Rust extension is not built: {library}")
    return library.resolve()


@pytest.fixture
def conn(extension_path):
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.enable_load_extension(True)
    connection.load_extension(str(extension_path))
    connection.execute("CREATE VIRTUAL TABLE v USING vectorlite(embedding float32[2], hnsw(max_elements=8))")
    connection.execute("INSERT INTO v(rowid, embedding) VALUES(1, vector_from_json('[1,2]'))")
    yield connection
    connection.close()


@pytest.mark.parametrize(
    "statement, params, message",
    [
        ("INSERT INTO v(operation) VALUES('save')", (), "path must be provided as TEXT"),
        ("INSERT INTO v(operation,path) VALUES('other','ignored')", (), "unknown operation"),
        ("INSERT INTO v(operation,path) VALUES('load',?)", ("/nonexistent/vectorlite/index",), "load failed"),
        ("INSERT INTO v(rowid,embedding) VALUES(2, vector_from_json('[1]'))", (), "Dimension mismatch"),
        ("SELECT rowid FROM v WHERE rowid IN (1,'text')", (), "rowid must be of type INTEGER"),
        ("SELECT rowid FROM v WHERE knn_search(embedding, 1)", (), "knn_param"),
    ],
)
def test_callback_error_leaves_index_usable(conn, statement, params, message):
    with pytest.raises(sqlite3.DatabaseError, match=message):
        conn.execute(statement, params).fetchall()
    assert conn.execute("SELECT vector_to_json(embedding) FROM v WHERE rowid=1").fetchone() == ("[1.0,2.0]",)
    conn.execute("INSERT INTO v(rowid,embedding) VALUES(2,vector_from_json('[3,4]'))")
    assert conn.execute("SELECT rowid FROM v WHERE rowid IN (1,2,3) ORDER BY rowid").fetchall() == [(1,), (2,)]


def test_empty_in_and_duplicate_rowids(conn):
    assert conn.execute("SELECT rowid FROM v WHERE rowid IN (SELECT 1 WHERE 0)").fetchall() == []
    assert conn.execute("SELECT rowid FROM v WHERE rowid IN (1,1,2)").fetchall() == [(1,)]


def test_registry_survives_reparse_rename_and_recreation(conn):
    conn.execute("CREATE TABLE ordinary(value)")
    conn.execute("VACUUM")
    conn.execute("ALTER TABLE v RENAME TO renamed")
    conn.execute("ALTER TABLE ordinary ADD COLUMN extra")
    assert conn.execute("SELECT rowid FROM renamed WHERE rowid=1").fetchall() == [(1,)]
    conn.execute("DROP TABLE renamed")
    conn.execute("CREATE VIRTUAL TABLE renamed USING vectorlite(embedding float32[2], hnsw(max_elements=8))")
    assert conn.execute("SELECT rowid FROM renamed WHERE rowid=1").fetchall() == []


def _save_legacy_index(conn, table, path):
    conn.execute(
        f"INSERT INTO {table}(operation,path) VALUES('save',?)", (str(path),)
    )
    contents = path.read_bytes()
    assert contents[:8] == b"VLTIDX01"
    assert int.from_bytes(contents[24:32], "little") == len(contents) - 32
    # The envelope contains an unmodified native hnswlib payload, using the same
    # serialized layout as the C++ extension's raw saveIndex output.
    path.write_bytes(contents[32:])


@pytest.mark.parametrize("vector_type", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
@pytest.mark.parametrize("empty", [False, True], ids=["populated", "empty"])
def test_legacy_load_and_versioned_resave(conn, tmp_path, vector_type, metric, empty):
    declaration = f"embedding {vector_type}[2] {metric}"
    conn.execute(
        f"CREATE VIRTUAL TABLE source USING vectorlite({declaration}, "
        "hnsw(max_elements=8))"
    )
    if not empty:
        for rowid, vector in [(10, [1, 2]), (11, [3, 1]), (12, [2, 4])]:
            conn.execute(
                "INSERT INTO source(rowid,embedding) VALUES(?,vector_from_json(?))",
                (rowid, json.dumps(vector)),
            )
    expected_vectors = conn.execute(
        "SELECT rowid,embedding FROM source WHERE rowid IN (10,11,12) ORDER BY rowid"
    ).fetchall()
    query = "[1,2]"
    expected_neighbors = conn.execute(
        "SELECT rowid,distance FROM source "
        "WHERE knn_search(embedding,knn_param(vector_from_json(?),8,40))",
        (query,),
    ).fetchall()
    legacy_path = tmp_path / "legacy.bin"
    _save_legacy_index(conn, "source", legacy_path)

    conn.execute(
        f"CREATE VIRTUAL TABLE loaded USING vectorlite({declaration}, "
        "hnsw(max_elements=8))"
    )
    conn.execute(
        "INSERT INTO loaded(rowid,embedding) VALUES(99,vector_from_json('[9,8]'))"
    )
    conn.execute(
        "INSERT INTO loaded(operation,path) VALUES('load',?)", (str(legacy_path),)
    )
    assert conn.execute(
        "SELECT rowid,embedding FROM loaded WHERE rowid IN (10,11,12) ORDER BY rowid"
    ).fetchall() == expected_vectors
    assert conn.execute("SELECT rowid FROM loaded WHERE rowid=99").fetchall() == []
    assert conn.execute(
        "SELECT rowid,distance FROM loaded "
        "WHERE knn_search(embedding,knn_param(vector_from_json(?),8,40))",
        (query,),
    ).fetchall() == expected_neighbors

    upgraded_path = tmp_path / "versioned.bin"
    conn.execute(
        "INSERT INTO loaded(operation,path) VALUES('save',?)", (str(upgraded_path),)
    )
    assert upgraded_path.read_bytes()[:8] == b"VLTIDX01"
    conn.execute(
        f"CREATE VIRTUAL TABLE restored USING vectorlite({declaration}, "
        "hnsw(max_elements=8))"
    )
    conn.execute(
        "INSERT INTO restored(operation,path) VALUES('load',?)", (str(upgraded_path),)
    )
    assert conn.execute(
        "SELECT rowid,embedding FROM restored WHERE rowid IN (10,11,12) ORDER BY rowid"
    ).fetchall() == expected_vectors


@pytest.mark.parametrize(
    "source_space,target_space,expected_vector,expected_distance",
    [
        ("float32[2] l2", "float32[2] ip", [1.0, 2.0], -4.0),
        ("bfloat16[2] l2", "float16[2] l2", [1.875, 2.0], 0.0),
        ("float32[2] l2", "float16[4] l2", [0.0, 1.875, 0.0, 2.0], 0.0),
    ],
)
def test_legacy_uses_declared_schema(
    conn, tmp_path, source_space, target_space, expected_vector, expected_distance
):
    conn.execute(
        f"CREATE VIRTUAL TABLE source USING vectorlite(embedding {source_space}, "
        "hnsw(max_elements=8))"
    )
    conn.execute(
        "INSERT INTO source(rowid,embedding) VALUES(10,vector_from_json('[1,2]'))"
    )
    legacy_path = tmp_path / "legacy.bin"
    _save_legacy_index(conn, "source", legacy_path)
    conn.execute(
        f"CREATE VIRTUAL TABLE loaded USING vectorlite(embedding {target_space}, "
        "hnsw(max_elements=8))"
    )
    conn.execute(
        "INSERT INTO loaded(operation,path) VALUES('load',?)", (str(legacy_path),)
    )
    blob = conn.execute("SELECT embedding FROM loaded WHERE rowid=10").fetchone()[0]
    assert list(struct.unpack(f"<{len(expected_vector)}f", blob)) == expected_vector
    rowid, distance = conn.execute(
        "SELECT rowid,distance FROM loaded "
        "WHERE knn_search(embedding,knn_param(vector_from_json(?),1))",
        (json.dumps(expected_vector),),
    ).fetchone()
    assert rowid == 10
    assert distance == pytest.approx(expected_distance)


@pytest.mark.parametrize(
    "invalid", ["data_size", "short_header", "truncated", "trailing_bytes", "bad_neighbor"]
)
def test_invalid_legacy_load_preserves_live_index(conn, tmp_path, invalid):
    dimension = 3 if invalid == "data_size" else 2
    conn.execute(
        f"CREATE VIRTUAL TABLE source USING vectorlite(embedding float32[{dimension}], "
        "hnsw(max_elements=8))"
    )
    conn.execute(
        "INSERT INTO source(rowid,embedding) VALUES(10,vector_from_json(?))",
        (json.dumps([1] * dimension),),
    )
    legacy_path = tmp_path / "invalid.bin"
    _save_legacy_index(conn, "source", legacy_path)
    payload = bytearray(legacy_path.read_bytes())
    if invalid == "short_header":
        payload = payload[:7]
    elif invalid == "truncated":
        payload = payload[:-1]
    elif invalid == "trailing_bytes":
        payload += b"\x00"
    elif invalid == "bad_neighbor":
        # Native header: ten size_t fields, one int, one tableint, one double.
        data_offset = 10 * struct.calcsize("P") + 16
        struct.pack_into("=I", payload, data_offset, 1)
        struct.pack_into("=I", payload, data_offset + 4, 1)  # count is only one
    legacy_path.write_bytes(payload)

    with pytest.raises(sqlite3.OperationalError, match="load failed"):
        conn.execute(
            "INSERT INTO v(operation,path) VALUES('load',?)", (str(legacy_path),)
        )
    assert conn.execute(
        "SELECT rowid,vector_to_json(embedding) FROM v WHERE rowid IN (1,10)"
    ).fetchall() == [(1, "[1.0,2.0]")]


@pytest.mark.parametrize(
    "source_space,target_space",
    [("bfloat16[2] l2", "float16[2] l2"), ("float32[2] l2", "float32[2] ip")],
)
def test_versioned_descriptor_mismatch_does_not_fall_back(
    conn, tmp_path, source_space, target_space
):
    conn.execute(
        f"CREATE VIRTUAL TABLE source USING vectorlite(embedding {source_space}, "
        "hnsw(max_elements=8))"
    )
    conn.execute(
        "INSERT INTO source(rowid,embedding) VALUES(10,vector_from_json('[1,2]'))"
    )
    versioned_path = tmp_path / "versioned.bin"
    conn.execute(
        "INSERT INTO source(operation,path) VALUES('save',?)", (str(versioned_path),)
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE loaded USING vectorlite(embedding {target_space}, "
        "hnsw(max_elements=8))"
    )
    conn.execute(
        "INSERT INTO loaded(rowid,embedding) VALUES(99,vector_from_json('[9,8]'))"
    )
    with pytest.raises(sqlite3.OperationalError, match="descriptor mismatch"):
        conn.execute(
            "INSERT INTO loaded(operation,path) VALUES('load',?)", (str(versioned_path),)
        )
    assert conn.execute(
        "SELECT rowid FROM loaded WHERE rowid IN (10,99)"
    ).fetchall() == [(99,)]


@pytest.mark.parametrize("allow_replace_deleted", [False, True])
def test_legacy_preserves_configured_capacity_and_deletion_policy(
    conn, tmp_path, allow_replace_deleted
):
    conn.execute(
        "CREATE VIRTUAL TABLE source USING vectorlite(embedding float32[2], "
        "hnsw(max_elements=3))"
    )
    for rowid in (10, 11, 12):
        conn.execute(
            "INSERT INTO source(rowid,embedding) VALUES(?,vector_from_json('[1,2]'))",
            (rowid,),
        )
    legacy_path = tmp_path / "legacy.bin"
    _save_legacy_index(conn, "source", legacy_path)
    replace = str(allow_replace_deleted).lower()
    conn.execute(
        "CREATE VIRTUAL TABLE loaded USING vectorlite(embedding float32[2], "
        f"hnsw(max_elements=4,allow_replace_deleted={replace}))"
    )
    conn.execute(
        "INSERT INTO loaded(operation,path) VALUES('load',?)", (str(legacy_path),)
    )
    insert = "INSERT INTO loaded(rowid,embedding) VALUES(?,vector_from_json('[3,4]'))"
    conn.execute(insert, (13,))  # Uses the receiving table's larger capacity.
    conn.execute("DELETE FROM loaded WHERE rowid=11")
    if allow_replace_deleted:
        conn.execute(insert, (14,))
        expected = [(10,), (12,), (13,), (14,)]
    else:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(insert, (14,))
        expected = [(10,), (12,), (13,)]
    assert conn.execute(
        "SELECT rowid FROM loaded WHERE rowid IN (10,11,12,13,14) ORDER BY rowid"
    ).fetchall() == expected


@pytest.mark.parametrize("file_format", ["versioned", "legacy"])
def test_load_ignores_untrusted_saved_capacity(conn, tmp_path, file_format):
    index_path = tmp_path / "capacity.bin"
    conn.execute(
        "CREATE VIRTUAL TABLE source USING "
        "vectorlite(embedding float32[2], hnsw(max_elements=8))"
    )
    for rowid in (10, 11, 12):
        conn.execute(
            "INSERT INTO source(rowid,embedding) "
            "VALUES(?,vector_from_json('[1,2]'))",
            (rowid,),
        )
    conn.execute(
        "INSERT INTO source(operation,path) VALUES('save',?)",
        (str(index_path),),
    )

    # The raw HNSW payload begins with offsetLevel0_ followed by max_elements_,
    # both native-size words. Exercise the capacity limit with both file formats.
    payload_offset = 32
    if file_format == "legacy":
        index_path.write_bytes(index_path.read_bytes()[32:])
        payload_offset = 0
    word_size = struct.calcsize("P")
    with index_path.open("r+b") as index_file:
        index_file.seek(payload_offset + word_size)
        index_file.write((1_000_000).to_bytes(word_size, sys.byteorder))

    conn.execute(
        "CREATE VIRTUAL TABLE loaded USING "
        "vectorlite(embedding float32[2], hnsw(max_elements=2))"
    )
    conn.execute(
        "INSERT INTO loaded(operation,path) VALUES('load',?)",
        (str(index_path),),
    )
    assert conn.execute(
        "SELECT rowid FROM loaded WHERE rowid IN (10,11,12) ORDER BY rowid"
    ).fetchall() == [(10,), (11,), (12,)]

    # Loading must allocate only max(configured capacity, element count), not
    # the untrusted max_elements_ stored in the payload.
    with pytest.raises(sqlite3.OperationalError):
        conn.execute(
            "INSERT INTO loaded(rowid,embedding) "
            "VALUES(13,vector_from_json('[3,4]'))"
        )
    assert conn.execute(
        "SELECT rowid FROM loaded WHERE rowid IN (10,11,12) ORDER BY rowid"
    ).fetchall() == [(10,), (11,), (12,)]


def test_vtab_is_direct_only(conn, tmp_path):
    index_path = tmp_path / "trigger-save.bin"
    quoted_path = str(index_path).replace("'", "''")
    conn.execute("CREATE TABLE ordinary(value)")
    conn.execute(
        "CREATE TRIGGER save_from_schema AFTER INSERT ON ordinary BEGIN "
        f"INSERT INTO v(operation,path) VALUES('save','{quoted_path}'); "
        "END"
    )

    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("INSERT INTO ordinary VALUES(1)")
    assert not index_path.exists()


@pytest.mark.skipif(os.name == "nt", reason="Unix mode bits are not available on Windows")
def test_save_preserves_existing_unix_mode(conn, tmp_path):
    index_path = tmp_path / "index.bin"
    index_path.write_bytes(b"old contents")
    os.chmod(index_path, 0o640)

    conn.execute(
        "INSERT INTO v(operation,path) VALUES('save',?)",
        (str(index_path),),
    )
    assert stat.S_IMODE(index_path.stat().st_mode) == 0o640

    conn.execute(
        "CREATE VIRTUAL TABLE restored USING "
        "vectorlite(embedding float32[2], hnsw(max_elements=8))"
    )
    conn.execute(
        "INSERT INTO restored(operation,path) VALUES('load',?)",
        (str(index_path),),
    )
    assert conn.execute(
        "SELECT rowid FROM restored WHERE rowid=1"
    ).fetchall() == [(1,)]


def test_native_capacity_error_releases_locks_for_retry(extension_path):
    # Missing C++ exception unwinding can retain a native label lock after the
    # capacity exception. Bound the whole scenario so a regression fails CI
    # instead of leaving its worker blocked in a subsequent insertion.
    script = """
import sqlite3
import sys

conn = sqlite3.connect(':memory:', isolation_level=None)
conn.enable_load_extension(True)
conn.load_extension(sys.argv[1])
conn.execute('CREATE VIRTUAL TABLE v USING vectorlite(embedding float32[2], hnsw(max_elements=2, allow_replace_deleted=true))')
insert = "INSERT INTO v(rowid,embedding) VALUES(?,vector_from_json('[1,2]'))"
conn.execute(insert, (1,))
conn.execute(insert, (2,))
try:
    conn.execute(insert, (3,))
except sqlite3.OperationalError:
    pass
else:
    raise AssertionError('insertion beyond capacity unexpectedly succeeded')
conn.execute('DELETE FROM v WHERE rowid=1')
conn.execute(insert, (3,))
assert conn.execute('SELECT rowid FROM v WHERE rowid IN (1,2,3) ORDER BY rowid').fetchall() == [(2,), (3,)]
conn.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(extension_path)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
