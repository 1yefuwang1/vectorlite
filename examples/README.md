# Instructions
1. Please first run `sh build_release.sh` in project root folder. Then run `cd examples`
2. (Optionally) Install [apsw](https://github.com/rogerbinns/apsw) if you want to use rowid filter. Vectorlite's rowid filter feature requires sqlite3 version >= 3.38. Usually, python's built-in sqlite3 module is not new enough. Apsw provides binding to latest sqlite3 releases. Please install it using `pip3 install apsw`.
3. Run `python3 knn_search.py` or `USE_APSW=1 python3 knn_search.py`.