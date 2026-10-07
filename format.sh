#!/bin/sh
set -eu
repo_root=$(CDPATH= cd "$(dirname "$0")" && pwd)
cd "$repo_root"
cargo fmt --all
clang-format -style=file -i vectorlite/cpp/*.h vectorlite/cpp/*.cpp vectorlite/ops/*.h vectorlite/ops/*.cpp
