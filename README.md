# Overview
vectorlite is a [Runtime-loadable extension](https://www.sqlite.org/loadext.html) for SQLite that enables fast vector search based on [hnswlib](https://github.com/nmslib/hnswlib). 
It's still in early development and is not considered production ready. There could be breaking changes and bugs.
Examples can be found in examples folder.
# Build Instruction
This project is currently developed on Linux. Compilation on other OS has not been tested.
## Prerequisites
1. CMake >= 3.22
2. A c++ compiler that supports c++17
3. git
## Build
```
git clone --recurse-submodules git@github.com:1yefuwang1/vectorlite.git

sh vcpkg/bootstrap-vcpkg.sh

sh test.sh # for running unit tests
sh build.sh # for debug build
sh build_release.sh # for release build

```
# Acknowledgement
This project is greatly inspired by following projects
- [sqlite-vss](https://github.com/asg017/sqlite-vss)
- [hnsqlite](https://github.com/jiggy-ai/hnsqlite)
- [ChromaDB](https://github.com/chroma-core/chroma)