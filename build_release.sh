#!/bin/sh
# Build and validate the production Rust extension.
set -eu

repo_root=$(CDPATH= cd "$(dirname "$0")" && pwd)
cd "$repo_root"

cmake --preset release
cmake --build build/release --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
ctest --test-dir build/release --output-on-failure --no-tests=error
PYTHONPATH="$repo_root/bindings/python" "${PYTHON:-python}" -m pytest \
    --import-mode=importlib bindings/python/vectorlite_py/test vectorlite/tests
