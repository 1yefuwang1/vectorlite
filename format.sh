#!/bin/sh
set -eu
repo_root=$(CDPATH= cd "$(dirname "$0")" && pwd)
cd "$repo_root"
cargo fmt --manifest-path rust/Cargo.toml --all
clang-format -style=file -i rust/cpp/*.h rust/cpp/*.cpp vectorlite/ops/*.h vectorlite/ops/*.cpp
