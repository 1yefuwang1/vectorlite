# vectorlite Rust implementation

This Cargo crate is the **main and only implementation** of the vectorlite
SQLite extension. It owns the SQLite glue, constraint handling, per-connection
index registry, vector-space/index-option parsing, quantization/normalization
*decisions*, the rowid filter predicate, query-local `ef` overrides, load
validation and save/load orchestration. C++ is reached **only via FFI, and only
for hnswlib and the SIMD `ops`** (Google Highway), through a thin C ABI shim.
The native ops CMake tests and benchmarks are retained; the superseded C++
virtual-table implementation is not built or shipped. `unsafe` is confined to
the SQLite FFI boundary and the hnswlib/ops C ABI.

CMake and scikit-build-core remain the normal build and packaging entrypoints,
and invoke Cargo for the extension library. The Python/npm package names and
public `vectorlite.so` / `vectorlite.dylib` / `vectorlite.dll` filenames remain
unchanged.

## Architecture

The Cargo package/workspace manifest [Cargo.toml](<../Cargo.toml>) and lockfile
[Cargo.lock](<../Cargo.lock>) are at the repository root. The manifest selects
[build.rs](<build.rs>) and [lib.rs](<src/lib.rs>) under this `vectorlite/` source
directory:

```text
repository root/
├── Cargo.toml                  # package/workspace manifest
├── Cargo.lock
└── vectorlite/
    ├── build.rs                # native build script
    ├── src/                    # Rust extension implementation
    ├── tests/                  # build discovery and SQL regressions
    ├── vectorlite-sqlite-sys/   # workspace member / SQLite API bindings
    ├── cpp/                    # core_shim.cpp / core_shim.h
    └── ops/                    # ops.cpp and native tests/benchmarks
```

In the module diagram below, Rust filenames are relative to `vectorlite/src/`;
the sys crate and native paths are relative to `vectorlite/`, with `cpp/` and
`ops/` as siblings.

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
│ ops/ops.cpp       (un-ported) SIMD kernels via Google Highway                         │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

The distance function hnswlib uses is a Rust `extern "C"` callback in
[ops.rs](<src/ops.rs>)
that forwards to `ops`; the rowid filter is a Rust predicate invoked through a
trampoline. So the *only* C++ is hnswlib itself, the `ops` kernels, and the
minimal generic adapters needed to expose those two through a C ABI.

## SQLite-backed DiskANN

The optional `diskann(...)` backend calls Microsoft DiskANN3's pinned Rust graph
algorithms through a private `DataProvider`, while storing vectors, adjacency,
labels, counters, and entrypoints in SQLite shadow tables. Initial support is
float32 squared L2/cosine; HNSW options, types, files, and nontransactional behavior
remain unchanged. See the [SQL/operational guide](<../doc/diskann.md>).

The new modules are [diskann_core.rs](<src/diskann_core.rs>),
[diskann_store.rs](<src/diskann_store.rs>), [atomic_callback.rs](<src/atomic_callback.rs>),
[sqlite.rs](<src/sqlite.rs>), [batch_input.rs](<src/batch_input.rs>), and
[index_error.rs](<src/index_error.rs>). SQLite calls still use only the host API
table; no separate connection or linked SQLite is used. Native pointer batch
INSERT drives spawned DiskANN batch tasks on an operation-private current-thread
Tokio runtime, fixed to one execution thread. The [pinned patch](<../third_party/diskann-0.60.0/vendor/VENDOR_PATCH.md>)
propagates every batch failure and drains tasks before guard completion. Ordinary
single operations retain their first-poll-ready executor. Connection-bound owners
remain non-Send; private scoped tokens check thread, connection, generation, and
reentrancy before accessing borrowed storage. All tasks/runtime and owned chunk
inputs are destroyed before closing the callback scope; prepared plans are reused
only within that operation and finalized before accepting success.

A single virtual-table callback can make multiple successful nested SQL writes
before a later error. SQLite does not necessarily allocate a statement journal
for that single callback. Each DiskANN mutation therefore runs inside a private
ordinary two-row UPDATE carrier, with pre/post guard validation and errors
reported while its scalar action is still inside the statement's rollback
boundary. Do not replace this with bare write-through callbacks or a no-op
trigger: trigger disabling and FAIL/IGNORE behavior can break atomicity.

Authoritative state is SQLite-only; rollback needs no transaction-sized Rust
graph snapshot. Query/prune workspaces are operation-local, records are read
lazily, and projected cursor vectors are captured with their distances. Corrupt
lengths must be gated in SQL before payload materialization, not just checked
after copying a BLOB into Rust.

## Building

Official release and CI platforms are Linux x64, Windows x64 and macOS Apple
Silicon (arm64) only. The macOS arm64 wheel requires macOS 11.0 or newer;
Intel macOS is not supported.

The normal source build uses **CMake -> Cargo**; Python source installs and
wheels use **scikit-build-core -> CMake -> Cargo**. CMake installs native
dependencies through vcpkg, builds the Rust library and deploys it into the
Python package. There is no prerequisite C++ extension build and no separate
Rust deployment step afterward. Source archives omit Git submodules; if the local
vcpkg toolchain is unavailable, CMake fetches and bootstraps the exact revision in
`vcpkg.json`. Source-archive builds therefore require Git and network access.

### Prerequisites

- The **latest stable Rust release**; [rust-toolchain.toml](<../rust-toolchain.toml>)
  selects stable and includes rustfmt and Clippy.
- C and C++17 compilers (MSVC with the dynamic CRT on Windows).
- CMake >= 3.22, Ninja, Git and the vcpkg submodule.
- Python 3.14 or newer with loadable SQLite extensions enabled, pytest and NumPy
  for the integration suites.

From the repository root:

```sh
rustup update stable
git submodule update --init --recursive
python3 bootstrap_vcpkg.py
python3 -m pip install -r requirements-dev.txt

sh build.sh          # Debug build + CTest + both Python suites
sh build_release.sh  # Release build + the same tests

# Build-only iteration:
cmake --preset release
cmake --build build/release -j8

# Python packaging uses the same Rust implementation:
python3 -m pip wheel . --wheel-dir dist
# Or install directly from source:
python3 -m pip install .
```

The public artifacts are `vectorlite.so` on Linux, `vectorlite.dylib` on macOS,
and `vectorlite.dll` on Windows, under `build/<preset>/vectorlite` and deployed
into `bindings/python/vectorlite_py/`.

With `BUILD_TESTING=ON`, CMake selects the vcpkg `tests` feature for Google Test
and SQLite headers, and registers Rust unit/SQLite ABI/native ops tests with
root CTest. Wheel builds need only the runtime native dependencies. The retained
ops microbenchmark is opt-in:

```sh
cmake --preset release -DVECTORLITE_BUILD_BENCHMARKS=ON
cmake --build build/release --target ops_benchmark -j8
```

[build.sh](<build.sh>) and [build.ps1](<build.ps1>) in this directory are
compatibility build-only shortcuts to the CMake release `vectorlite` target.
Use the root scripts for the full build/test cycle.

### Direct Cargo builds

For Rust-only iteration, configure CMake once to install native dependencies,
then invoke Cargo directly from the repository root. Configuration does **not**
require building an old C++ extension:

```sh
cmake --preset release
cargo build --locked --release
```

A direct Cargo build uses its native `libvectorlite.so` / `libvectorlite.dylib`
name on Unix (and `vectorlite.dll` on Windows) in `target/release` by default.
Use the normal CMake build to produce/deploy the public package filename.
Build with the committed lockfile (`--locked`) to use the verified dependency
versions. Older Rust releases are not part of the support policy; code and
dependency updates are validated against current stable.

The normal CMake build passes its exact vcpkg installation, compiler and Cargo
target/output directory to Rust. For direct Cargo builds, [build.rs](<build.rs>)
selects a vcpkg triplet matching Cargo's `TARGET`, including the CPU architecture,
operating system and C runtime family. It prefers the matching CMake preset
(`build/dev` for Cargo debug, `build/release` for release), then requires a unique
compatible installation among the remaining build trees and `vcpkg/installed`.
Ambiguous or incompatible installations fail with an error. To select an
installation explicitly for direct Cargo, set the full triplet directory:

```sh
export VECTORLITE_VCPKG_TRIPLET_DIR="$PWD/build/release/vcpkg_installed/arm64-osx"
cargo build --locked --release
```

Supported triplet names follow vcpkg's architecture/platform names, optionally
ending in `-release`. Windows MSVC uses static libraries with the dynamic CRT
(`x64-windows-static-md` or `x64-windows-static-md-release`); Linux musl and GNU
libc installations are kept distinct. Cross builds also need a compatible
native compiler/linker and an installed Rust target; configure CMake/Cargo for
the same architecture and runtime.

The C++ shim enables standard exception unwinding on MSVC (`/EHsc`). Native
headers, the Highway archive and vcpkg package metadata are tracked so native
changes invalidate the Rust build. The linked native dependencies are Highway
and the C++ runtime (plus `pthread`/`dl`/`m` on Linux); SQLite is supplied by the
host process.

## Notes

- SQLite >= 3.20 is required; rowid lookup/filtering requires SQLite >= 3.38.
  On SQLite >= 3.31, virtual tables are direct-only and cannot be accessed from
  views or triggers. Issue save/load commands directly from application SQL.
- An explicit `knn_param(..., ef)` override is query-local; the previous index
  setting is restored after the search. Queries without an override use the
  default `ef` of 10, including after load.
- SQLite is **not** linked into the library. A loadable extension never calls
  SQLite directly — every call goes through the `sqlite3_api_routines` table the
  host passes at load time (the loadable-extension contract) — so the library
  has no undefined SQLite symbols and needs no embedded copy. This keeps the
  artifact small; the
  host process supplies SQLite when it loads the extension.
- The SQLite extension-API bindings are **pre-generated and committed**, so
  normal builds need **no libclang**. The unused `va_list` function-pointer slots
  are private opaque entries; platform-specific varargs types are not exposed.
  From the repository root, generate and test native target bindings after a
  SQLite header change:

  ```sh
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
- New saves use a versioned envelope recording dimension, element type, distance
  metric, normalization and native word size/endianness. Versioned files must
  match the receiving table's descriptor; an invalid descriptor or payload is
  rejected before replacing the live index.
- Loading also accepts legacy raw HNSW files written by earlier Rust/C++ builds
  or hnswlib. Since these files contain no Vectorlite metadata, the receiving
  virtual table's declared dimension, element type, distance metric and
  normalization policy are authoritative. A matching per-vector byte size is the
  schema compatibility check, so same-width element types or different
  dimensions with the same total byte size are accepted. The existing native
  layout, graph-link and capacity checks still apply, and the raw payload's
  native word size and endianness must be compatible with the current host.
  Loading does not convert or re-normalize the stored vectors or rebuild the
  graph; declare the intended schema when opening a legacy file.
- No export/reinsertion is needed to upgrade a legacy file: load it into a table
  and save again to write the versioned format with that table's descriptor.
  Successful saves replace the destination atomically using a temporary file in
  the same directory. Failed loads leave the current in-memory index unchanged.

  ```sql
  -- Create my_table with the intended schema before loading the raw index.
  INSERT INTO my_table(operation, path) VALUES('load', 'legacy-hnsw.bin');
  INSERT INTO my_table(operation, path) VALUES('save', 'versioned-index.bin');
  ```

## Testing

CI builds and tests the latest stable Rust on Linux x64, Windows x64 and
macOS Apple Silicon (arm64) only. Native ABI checks compare the Rust layouts
and API offsets used by the extension against the installed SQLite C headers.

From the repository root, the normal scripts run CMake's Cargo-backed build,
CTest (Rust unit tests, SQLite ABI checks and native ops tests), and both Python
integration suites against the freshly built library:

```sh
sh build.sh
# Or use the release profile:
sh build_release.sh

# Rerun only the registered Rust/native tests:
ctest --test-dir build/dev --output-on-failure
```

For direct Cargo iteration from the repository root after configuring native
dependencies:

```sh
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets -- -D warnings -D clippy::undocumented_unsafe_blocks -D clippy::missing_safety_doc
cargo test --locked --workspace
cargo test --locked -p vectorlite-sqlite-sys --features abi-check
```

Both Python suites default to the public library in the Python package, using
`vectorlite_py.vectorlite_path()`. A missing artifact fails rather than skipping
Rust regressions. After a normal CMake build/deployment, no override is needed:

```sh
PYTHONPATH=bindings/python python3 -m pytest --import-mode=importlib \
  bindings/python/vectorlite_py/test vectorlite/tests
```

For installed-wheel validation, install the freshly built wheel into the test
interpreter, then use the [wheel test runner](<../scripts/run_wheel_tests.py>)
without a source-tree `PYTHONPATH`:

```sh
python3 scripts/run_wheel_tests.py
```

The runner preloads `vectorlite_py` before pytest collects the nested binding
tests. Even `--import-mode=importlib` can otherwise import the checkout's parent
package and shadow the wheel. It verifies the package's installed-distribution
path and native library, and rejects `VECTORLITE_RUST_EXTENSION` overrides so
both suites test the wheel rather than a local build.

`VECTORLITE_RUST_EXTENSION` remains an optional override for testing a direct
Cargo artifact with the Rust-specific suite. For example, on macOS:

```sh
VECTORLITE_RUST_EXTENSION="$PWD/target/release/libvectorlite.dylib" \
  PYTHONPATH=bindings/python python3 -m pytest vectorlite/tests
```

Use `libvectorlite.so` on Linux or `vectorlite.dll` on Windows. The binding
suite still uses the package library, so keep its deployment current when
running both suites together. No C++ virtual-table artifact is needed.
