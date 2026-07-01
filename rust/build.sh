#!/bin/sh
# Builds the Rust port of vectorlite and deploys the shared library into the
# Python package so the existing integration tests pick it up.
#
# Works on Linux, macOS and Windows (the latter via Git Bash / MSYS).
#
# Prerequisites: a Rust toolchain, plus the vcpkg-provided headers/libs produced
# by the C++ CMake build (run `sh build.sh` at the repo root once so that
# build/<preset>/vcpkg_installed/<triplet> exists).
#
# No libclang is required: the SQLite FFI bindings are pre-generated and
# committed in the vectorlite-sqlite-sys crate. (Only refreshing them with
# `cargo build -p vectorlite-sqlite-sys --features regenerate` needs libclang.)
set -e

cd "$(dirname "$0")"

cargo build --release

# cdylib artifact name and deployed name vary by platform.
case "$(uname -s)" in
    Linux*)               SRC=libvectorlite.so;    DST=vectorlite.so ;;
    Darwin*)              SRC=libvectorlite.dylib;  DST=vectorlite.dylib ;;
    MINGW*|MSYS*|CYGWIN*) SRC=vectorlite.dll;       DST=vectorlite.dll ;;
    *)                    SRC=libvectorlite.so;     DST=vectorlite.so ;;
esac

cp "target/release/$SRC" "../bindings/python/vectorlite_py/$DST"

echo "Deployed Rust $DST to bindings/python/vectorlite_py/"
echo "Run tests with:"
echo "  PYTHONPATH=bindings/python python -m pytest bindings/python/vectorlite_py/test"
