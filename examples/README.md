# Instructions
1. Please first run `sh build_release.sh` in project root folder. 
2. Run `pip3 install -r requirements.txt`. Vectorlite's rowid filter feature requires sqlite3 version >= 3.38. Usually, python's built-in sqlite3 module is not new enough. Apsw provides binding to latest sqlite3 releases.
4. Run `python3 knn_search.py`. If you still want to use the built-in sqlite3 module, run `USE_BUILTIN_SQLITE3=1 python3 knn_search.py`.