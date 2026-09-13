"""Callback recovery and retained registry ownership for the Rust extension."""

import os
from pathlib import Path
import sqlite3
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
