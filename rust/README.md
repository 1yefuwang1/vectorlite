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

The Rust port supports the **latest stable Rust release**. The repository's
`rust-toolchain.toml` selects `stable` and includes rustfmt and Clippy. Update
your installed stable toolchain before building:

```sh
rustup update stable
```

Build with the committed lockfile (`cargo build --locked --release`) to use the
verified dependency versions. Older Rust releases are not part of the support
policy; code and dependency updates are validated against current stable.

`build.rs` selects a vcpkg triplet matching Cargo's `TARGET`, including the CPU
architecture, operating system and C runtime family. It prefers the matching
CMake preset (`build/dev` for Cargo debug, `build/release` for release), then
requires a unique compatible installation among the remaining build trees and
`vcpkg/installed`. Ambiguous or incompatible installations fail with an error.
To select an installation explicitly, set the full triplet directory:

```sh
export VECTORLITE_VCPKG_TRIPLET_DIR="$PWD/build/release/vcpkg_installed/arm64-osx"
sh rust/build.sh
```

Supported triplet names follow vcpkg's architecture/platform names, optionally
ending in `-release`. Windows MSVC uses static libraries with the dynamic CRT
(`x64-windows-static-md` or `x64-windows-static-md-release`); Linux musl and GNU
libc installations are kept distinct. Cross builds also need a compatible C++
compiler/linker configured for Cargo and `cc`. The deployment scripts target the
native platform and default Cargo output directory.

The C++ shim enables standard exception unwinding on MSVC (`/EHsc`). Native
headers, the Highway archive and vcpkg package metadata are tracked so native
changes invalidate the Rust build. The linked native dependencies are Highway
and the C++ runtime (plus `pthread`/`dl`/`m` on Linux); SQLite is supplied by the
host process.

## Notes

- SQLite is **not** linked into the library. A loadable extension never calls
  SQLite directly — every call goes through the `sqlite3_api_routines` table the
  host passes at load time (the loadable-extension contract) — so the library
  has no undefined SQLite symbols and needs no embedded copy. This keeps the
  artifact small; the
  host process supplies SQLite when it loads the extension.
- The SQLite extension-API bindings are **pre-generated and committed**, so
  normal builds need **no libclang**. The unused `va_list` function-pointer slots
  are private opaque entries; platform-specific varargs types are not exposed.
  To generate and test native target bindings after a SQLite header change:

  ```sh
  cd rust
  cargo test --locked -p vectorlite-sqlite-sys --features regenerate,abi-check
  ```

  This needs libclang (set `LIBCLANG_PATH` when it is not discoverable). It uses
  the same target-aware vcpkg selection and writes `bindings.rs` into Cargo's
  `OUT_DIR`, reported by the build. The generated file is used for that build;
  tracked sources are never changed automatically. Refreshing the committed
  bindings is an explicit reviewed copy from the generated output.
  On macOS, use the Command Line Tools libclang with
  `LIBCLANG_PATH=/Library/Developer/CommandLineTools/usr/lib`; newer Homebrew
  LLVM releases may require a newer bindgen than the pinned generator supports.
- Rust persistence uses a versioned envelope recording dimension, element type,
  distance metric, normalization and native word size/endianness. Loading
  incompatible descriptors or an older raw HNSW file fails before replacing the
  live index. Raw files from earlier Rust/C++ builds must be reopened with the
  matching older extension and their vectors and rowids reinserted into the new
  extension. Successful saves replace the destination atomically using a
  temporary file in the same directory.

## Testing

CI builds and tests the latest stable Rust on Linux, Windows, Apple Silicon
macOS and Intel macOS. Native ABI checks compare the Rust layouts and API
offsets used by the extension against the installed SQLite C headers.

```sh
cd rust
cargo test --locked --workspace
cargo test --locked -p vectorlite-sqlite-sys --features abi-check
```

After deploying the Rust extension with its build script, run the Python suite
from the repository root. The root C++ build script also deploys its own library,
so rerun the Rust build script afterward when testing the Rust port:

```sh
PYTHONPATH=bindings/python python -m pytest bindings/python/vectorlite_py/test rust/tests
```
