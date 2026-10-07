#!/bin/sh
# Build the primary Rust extension, native kernel checks, and SQL integration tests.
set -eu

repo_root=$(CDPATH= cd "$(dirname "$0")" && pwd)
cd "$repo_root"

cmake --preset dev
cmake --build build/dev --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
ctest --test-dir build/dev --output-on-failure --no-tests=error
PYTHONPATH="$repo_root/bindings/python" "${PYTHON:-python}" -m pytest \
    --import-mode=importlib bindings/python/vectorlite_py/test vectorlite/tests
