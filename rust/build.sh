#!/bin/sh
# Compatibility build-only shortcut. The root CMake build now produces Rust
# directly; no separate C++ extension or later Cargo deployment is required.
set -eu

repo_root=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"
cmake --preset release
cmake --build build/release --target vectorlite --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

echo "Built and deployed vectorlite from Rust via CMake."
echo "Run sh build_release.sh for Rust, native, and SQL integration tests."
