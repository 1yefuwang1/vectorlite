# vectorlite (Rust port)

A Rust port of the vectorlite SQLite extension. **All virtual-table logic is in
Rust** — the SQLite glue, constraint handling, per-connection index registry,
vector-space/index-option parsing, quantization/normalization *decisions*, the
rowid filter predicate, per-query `ef` handling, the load data-size check and
save/load orchestration. C++ is reached **only via FFI, and only for two things:
hnswlib and the SIMD `ops`** (Google Highway). `unsafe` is confined to the SQLite
FFI boundary and the hnswlib/ops C ABI.

## Architecture

```
┌───────────────────────────── Rust (this crate, cdylib) ─────────────────────────────┐
│ lib.rs            sqlite3_extension_init: register scalar fns + the vtab module       │
│ ffi.rs            sqlite3ext routing (stores sqlite3_api_routines, typed wrappers)     │
│ virtual_table.rs  xCreate/xConnect/xBestIndex/xFilter/xUpdate/xColumn/xRename/...      │
│ scalar.rs         vector_distance / vector_from_json / vector_to_json / knn_* / info   │
│ vector_space.rs   parse "name type[dim] distance"                                     │
│ index_options.rs  parse "hnsw(max_elements=..., M=..., ...)"                          │
│ vector.rs         f32 blob <-> Vec<f32>, JSON (de)serialisation                        │
│ registry.rs       per-connection index registry (survives reparse/vacuum/rename)      │
│ core.rs           vtab policy: encode/decode, ef orchestration, filter, load check     │
│ ops.rs            safe `ops` FFI wrappers + hnswlib distance callbacks (in Rust)        │
│ hnsw.rs           safe hnswlib FFI wrappers + rowid-filter trampoline (in Rust)         │
│ vectorlite-sqlite-sys/  vendored, pre-generated SQLite extension-API bindings          │
└──────────────────────────────────────┬───────────────────────────────────────────────┘
                                        │ C ABI (cpp/core_shim.h) — hnswlib + ops ONLY
┌──────────────────────────────────────▼───────────────────────────────────────────────┐
│ cpp/core_shim.cpp  generic glue: a SpaceInterface adapter around a Rust distance        │
│                    callback, a BaseFilterFunctor adapter around a Rust predicate, thin  │
│                    HierarchicalNSW wrappers, and forwarders to `ops`. No vtab logic.     │
│ vectorlite/ops/ops.cpp  (un-ported) SIMD kernels via Google Highway                    │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

The distance function hnswlib uses is a Rust `extern "C"` callback (`ops.rs`)
that forwards to `ops`; the rowid filter is a Rust predicate invoked through a
trampoline. So the *only* C++ is hnswlib itself, the `ops` kernels, and the
minimal generic adapters needed to expose those two through a C ABI.

## Building

## Building

The build reuses the vcpkg headers/libraries produced by the C++ CMake build, so
run the C++ build once first (it sets up `build/<preset>/vcpkg_installed/<triplet>`):

```sh
sh build.sh            # at the repo root (configures vcpkg, builds C++)
```

Then build and deploy the Rust extension:

| Platform | Build + deploy | Artifact |
|----------|----------------|----------|
| Linux | `sh rust/build.sh` | `vectorlite.so` |
| macOS | `sh rust/build.sh` | `vectorlite.dylib` |
| Windows | `rust\build.ps1` (PowerShell) or `sh rust/build.sh` in Git Bash | `vectorlite.dll` |

`build.rs` is platform-agnostic: it scans `build/*/vcpkg_installed/*/` for the
installed triplet (`x64-linux`, `arm64-osx`, `x64-windows-static-md-release`,
…), uses the right static-archive names (`.a` / `.lib`), and emits the correct
per-linker flags (GNU `--no-gc-sections`, MSVC `/OPT:NOREF`, ld64 no-op). The
only native libraries linked are `hwy` and `sqlite3` — `abseil`, `re2` and
`rapidjson` are not needed because that logic was ported to Rust.

## Notes

- The full SQLite amalgamation is **statically linked** into the library (the
  vcpkg static `sqlite3`, whole-archived and kept past the linker's dead-code
  stripping), mirroring the C++ build (~3.2 MB). The embedded SQLite symbols are
  not exported, so they cannot interpose the host's SQLite; the extension still
  operates on the host connection through the `sqlite3_api_routines` table, per
  the loadable-extension contract.
- The SQLite extension-API bindings are **pre-generated and committed** in the
  `vectorlite-sqlite-sys` crate (`src/bindings.rs`), so the normal build needs
  **no libclang**. Refresh them after a SQLite header change with
  `cargo build -p vectorlite-sqlite-sys --features regenerate` (that step needs
  libclang; if absent, drop a `libclang.so` at `rust/.libclang/` and export
  `LIBCLANG_PATH`).

## Testing

The port passes the existing Python integration suite:

```sh
PYTHONPATH=bindings/python python -m pytest bindings/python/vectorlite_py/test
```
