# Instructions
1. Run `pip3 install -r requirements.txt`. Vectorlite's metadata(rowid) filter feature requires sqlite3 version >= 3.38. Usually, python's built-in sqlite3 module is not new enough. Apsw provides binding to latest sqlite3 releases.
2. If you still want to use the built-in sqlite3 module, run set `USE_BUILTIN_SQLITE3=1`.