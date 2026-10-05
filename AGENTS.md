# AGENTS.md

## Project Overview

Vectorlite is a SQLite extension for fast vector search using HNSW. Rust is the main and only extension implementation: the Cargo crate owns the SQLite virtual table, scalar functions, parsers, registry and index policy. C++17 is retained only for hnswlib, Google Highway SIMD operations, and their thin C ABI shim. Distributed Python wheels and npm packages retain the public `vectorlite.so`, `vectorlite.dylib` and `vectorlite.dll` names.

CMake and scikit-build-core remain the source-build and packaging frontends; the `vectorlite` library target invokes Cargo, not a C++ virtual-table implementation.

## Prerequisites

- Latest stable Rust, including rustfmt and Clippy (selected by [rust-toolchain.toml](<rust-toolchain.toml>)).
- C and C++ compilers with C++17 support; MSVC with the dynamic CRT on Windows.
- CMake >= 3.22, Ninja, Git and the vcpkg submodule.
- Python 3.14 or newer with loadable SQLite extensions enabled for integration tests, plus pytest and NumPy.

```bash
# Initialize native dependency tooling once after cloning.
git submodule update --init --recursive
python3 bootstrap_vcpkg.py
python3 -m pip install -r requirements-dev.txt
```

## Build Commands

```bash
# Debug build + CTest (Rust unit/SQLite ABI/native ops tests) + both Python suites
sh build.sh

# Release build + the same tests
sh build_release.sh

# Configure only (also installs native dependencies through vcpkg)
cmake --preset dev
cmake --preset release

# Build only: CMake invokes Cargo and deploys the extension to the Python package
cmake --build build/dev -j8
cmake --build build/release -j8

# Rust + SQLite ABI + retained native ops tests through CTest
ctest --test-dir build/dev --output-on-failure

# Direct Cargo checks after configuring native dependencies
cargo fmt --manifest-path rust/Cargo.toml --all --check
cargo clippy --manifest-path rust/Cargo.toml --locked --workspace --all-targets -- -D warnings
cargo test --manifest-path rust/Cargo.toml --locked --workspace
cargo test --manifest-path rust/Cargo.toml --locked -p vectorlite-sqlite-sys --features abi-check

# Build a wheel through scikit-build-core -> CMake -> Cargo
python3 -m pip wheel .
```

There is no prerequisite C++ extension build and no subsequent Rust redeployment step. Use the root build scripts to run both Python suites against the just-built library. Both Python suites default to the library deployed in the Python package; a missing artifact is an error. `VECTORLITE_RUST_EXTENSION` is an optional override for testing a direct Cargo output, not a requirement for the normal build or installed-wheel tests; see the [contributor guide](<rust/README.md#testing>).

`BUILD_TESTING=ON` enables Rust/SQLite ABI/native ops tests and selects the vcpkg `tests` feature (Google Test and SQLite headers). Wheel builds need only the native runtime dependencies. Ops benchmarks are opt-in with `VECTORLITE_BUILD_BENCHMARKS=ON`, which selects the separate vcpkg benchmark feature:

```bash
cmake --preset release -DVECTORLITE_BUILD_BENCHMARKS=ON
cmake --build build/release --target ops_benchmark -j8
```

## Project Structure

- `rust/src/` — Main extension implementation: SQLite routing/callbacks, parsers, registry, vector conversions and index policy.
- `rust/cpp/` — Thin C ABI over hnswlib and Highway ops; no SQLite virtual-table policy.
- `rust/vectorlite-sqlite-sys/` — Committed SQLite API bindings and native ABI checks.
- `rust/tests/` — Native-dependency discovery tests and Rust-specific SQL regressions.
- `vectorlite/ops/` — Retained native SIMD operations, Google Test tests and Google Benchmark benchmarks.
- `bindings/python/`, `bindings/nodejs/` — Packaging and integration tests for the compiled extension.
- `benchmark/`, `examples/` — Python performance harness and usage examples.
- `vcpkg/` and the root CMake configuration — Native dependency management and Cargo build/installation integration.
- `docs/superpowers/` — Dated design specs/plans. Treat C++ source references there as historical, not current build instructions.

## Key Dependencies

- Rust: bytemuck (checked byte views), serde_json, tempfile, and the vendored vectorlite-sqlite-sys crate.
- Build: Cargo and the cc crate for the native shim/ops compilation.
- Native runtime: hnswlib and Highway; SQLite calls go through the host's loadable-extension API table, not a linked SQLite library.
- Native testing: Google Test and Google Benchmark; SQLite headers are used for ABI checks/regeneration.

## Coding Conventions

### Rust

- Keep SQLite callbacks and raw FFI adapters narrow; parsing, validation and index policy belong in safe Rust helpers.
- Preserve `deny(unsafe_op_in_unsafe_fn)` and the documented-unsafe Clippy checks. Every unsafe operation needs a specific lifetime/layout/ownership justification.
- Validate vector dimensions and buffer lengths before native calls; use the distinct half-storage types rather than interchangeable integer slices.
- Do not let Rust panics or C++ exceptions cross C ABI boundaries. Keep panic-abort library profiles and native exception translation.
- SQLite serializes callbacks per connection. Registry entries use shared ownership so index/space lifetimes survive reparses; do not add unjustified Send/Sync implementations.
- Run rustfmt and Clippy when changing Rust code.

### Retained C++ / Highway

- Google C++ Style Guide, C++17, `#pragma once`, snake_case filenames, PascalCase public functions.
- Native ops live under `vectorlite::ops`; target-specific implementations use `HWY_NAMESPACE` and `namespace hn = hwy::HWY_NAMESPACE`.
- Preserve Highway dynamic dispatch and the generic hnswlib/ops C ABI. Do not move SQLite glue, vector wrappers, parser or registry policy back into C++.

## Testing

- Rust unit tests are colocated with modules; index/persistence tests are in [core_tests.rs](<rust/src/core_tests.rs>).
- SQLite layout/API-prefix checks live in `rust/vectorlite-sqlite-sys/`.
- Retain [ops_test.cpp](<vectorlite/ops/ops_test.cpp>) and [ops_benchmark.cpp](<vectorlite/ops/ops_benchmark.cpp>) with their CMake targets.
- Run both `bindings/python/vectorlite_py/test/` and `rust/tests/` Python suites against the same freshly built library.
- After implementation changes, run the root debug or release build script; it covers CTest and both Python suites.
