# Instructions
1. Please first run `sh build.sh` in project root folder.
2. rowid filter(rowid in (....)) requires sqlite3 version >= 3.38, which is higher than the sqlite3 built-in python sqlite3 module is compiled against.
If you wish to use it, please install apsw using `pip3 install apsw`, and `export USE_APSW=1`.
3. Run `python3 knn_search.py`.