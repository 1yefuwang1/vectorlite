# CLAUDE.md

## Project Overview

Vectorlite is a SQLite extension for fast vector search using the HNSW algorithm. Written in C++17 with SIMD acceleration via Google Highway. Distributed as Python wheels and npm packages.

## Build Commands

```bash
# Debug build + tests
sh build.sh
# Equivalent to: cmake --preset dev && cmake --build build/dev -j8 && ctest --test-dir build/dev/vectorlite --output-on-failure && pytest bindings/python/vectorlite_py/test

# Release build + tests
sh build_release.sh

# Configure only
cmake --preset dev        # debug
cmake --preset release    # release

# Build only
cmake --build build/dev -j8
cmake --build build/release -j8

# C++ unit tests only
ctest --test-dir build/dev/vectorlite --output-on-failure

# Python integration tests only
pytest bindings/python/vectorlite_py/test
```

## Project Structure

- `vectorlite/` — Core C++ source (extension entry point, virtual table, vector types, distance functions)
- `vectorlite/ops/` — SIMD operations using Google Highway (distance calculations, quantization)
- `bindings/python/` — Python package wrapping the compiled extension
- `bindings/nodejs/` — Node.js bindings
- `benchmark/` — Performance benchmarks (Python + C++)
- `examples/` — Python usage examples
- `cmake/`, `vcpkg/` — Build infrastructure and dependency management

## Key Dependencies

- **abseil** — Status/StatusOr error handling, string utilities
- **hnswlib** — HNSW index implementation
- **highway** — SIMD abstraction (dynamic dispatch across CPU targets)
- **rapidjson** — JSON parsing for vector serialization
- **sqlite3** — SQLite API
- **gtest** / **benchmark** — Testing and benchmarking frameworks
- **re2** — Regex for input validation

## Coding Conventions

- **Style**: Google C++ Style Guide (enforced by `.clang-format`)
- **C++ standard**: C++17
- **Header guards**: `#pragma once` (no `#ifndef` guards)
- **Naming**:
  - Classes/structs: `PascalCase` (`VirtualTable`, `GenericVector`)
  - Public functions/methods: `PascalCase` (`Distance()`, `FromJSON()`)
  - Member variables: `snake_case_` with trailing underscore (`data_`, `index_`)
  - Local variables: `snake_case`
  - Macros/constants: `SCREAMING_SNAKE_CASE` with `VECTORLITE_` prefix
  - Files: `snake_case.h` / `snake_case.cpp`
  - Type aliases: short names (`Vector`, `BF16Vector`, `VectorView`)
- **Error handling**: `absl::Status` / `absl::StatusOr<T>` (not exceptions)
- **Assertions**: `VECTORLITE_ASSERT()` macro for preconditions
- **Memory**: `std::unique_ptr` for ownership; avoid raw owning pointers
- **Optionals**: `std::optional<T>` with `std::nullopt`
- **Namespace**: `vectorlite` (nested `vectorlite::ops` for SIMD ops)
- **Namespace closing**: `}  // namespace vectorlite`

## SIMD / Highway Patterns

- SIMD code lives in `vectorlite/ops/` using Highway's dynamic dispatch
- All Highway ops prefixed with `hn::` (e.g., `hn::Load`, `hn::Mul`)
- Use `HWY_NAMESPACE` for target-specific code blocks
- Alias: `namespace hn = hwy::HWY_NAMESPACE`

## Testing

- **C++ unit tests**: Google Test, files named `*_test.cpp` in `vectorlite/`
- **C++ benchmarks**: Google Benchmark in `vectorlite/ops/ops_benchmark.cpp`
- **Python integration tests**: pytest in `bindings/python/vectorlite_py/test/`
- Always run both C++ and Python tests after changes: `sh build.sh`
