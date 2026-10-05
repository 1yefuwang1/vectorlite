# Rust review fixes implementation plan

> Toolchain policy update: the Rust port now supports the latest stable release only. Rust 1.70 validation below records the earlier review work and is no longer an active support commitment. See `rust/README.md` and the root `rust-toolchain.toml`.

> Legacy compatibility update: raw HNSW indexes can now be loaded directly using the receiving vtab's declared schema and a per-vector data-size check, with native structural validation retained. A subsequent save upgrades the file to the versioned format. This supersedes the export/reinsertion migration policy recorded below; see `rust/README.md` for current behavior.

> For agentic workers: implement independent domains with the dispatching-parallel-agents skill, then review the integrated result.

**Goal:** Correct the thirteen findings in the review of baf6715, including scalar L2 overhead and unsafe abstractions.

**Architecture:** Keep SQLite policy in Rust and HNSW/SIMD in the native shim. Make callback values scoped, native ownership explicit, and persistence checked and self-describing.

**Tech stack:** Rust, C++17, SQLite extension API, Highway, hnswlib, pytest.

**Spec:** Code-quality review of `baf6715`, followed by the approved FFI, persistence, performance, half-precision type-safety, and latest-stable toolchain changes.

## Global constraints

- Preserve SQL function and virtual table interfaces and tested numeric behavior.
- Validate native buffer lengths and retain all native owners; no arbitrary unaligned typed blob casts.
- Treat save errors as errors and reject incompatible persisted vector descriptors before replacement.
- Preserve the existing in-memory index when load fails.
- Preserve the pre-existing untracked AGENTS.md. Commit and push only when requested by the user.
- Work in the current checkout; independent workers own disjoint files.

## Task 1: Native core and persistence

Files: `rust/src/{core,hnsw,ops}.rs`, `rust/cpp/core_shim.{cpp,h}`, native/core regression tests.

- [x] Add regression coverage for mismatched safe buffer sizes, surviving external Space drops, failed saves and incompatible same-width formats.
- [x] Retain Space ownership in HNSW and use NonNull handles; check input/output sizes and callback configuration. Restrict byte conversion to numeric plain data and validate core dimensions/layout arithmetic.
- [x] Borrow float32 inputs when normalization is unnecessary; consolidate native result output into one C-compatible buffer.
- [x] Contain every C++ exception and avoid allocation in error handlers and filter construction.
- [x] Implement checked, atomically replaced versioned persistence with explicit legacy compatibility policy and descriptor validation.
- [x] Run focused Rust/core and persistence tests; report commands and compatibility behavior.

## Task 2: Build and portability

Files: `rust/build.rs`, build support, `rust/Cargo.toml`, lockfile, `rust/build.ps1`, sqlite-sys generation files, CI, `rust/README.md`.

- [x] Add /EHsc, deterministic target-aware dependency discovery and header invalidation.
- [x] Align declared MSRV and lockfile/dependencies; verify exact minimum and CI coverage.
- [x] Check PowerShell native command failure before copying; make regeneration target-aware and avoid tracked-source mutation during ordinary feature builds.
- [x] Document configuration overrides, persistence changes coordinated with Task 1, and broaden platform validation.
- [x] Run focused build-support tests and report any unavailable target checks.

## Task 3: SQLite safety and scalar performance

Files: `rust/src/{ffi,lib,scalar,vector,virtual_table,registry}.rs`, SQLite/scalar regressions.

- [x] Replace whole SQLite API table references with supported raw field reads and initial version/entry validation.
- [x] Split the externally mutable vtab header from shared Rust state; return errors before mutating SQLite error fields.
- [x] Introduce scoped callback value/context wrappers and safe scalar implementations. Borrow blob/text data; use checked aligned float views with decoding fallback.
- [x] Propagate IN iteration failure and validate argument-plan lengths.
- [x] Add targeted tests for aligned/misaligned/empty/invalid vector bytes and recoverable callback errors.

## Task 4: Integration and review

- [x] Run fmt, strict Clippy, Rust unit tests, Rust release build, `sh build.sh`, and Python tests explicitly against the Rust library.
- [x] Re-run scalar and KNN benchmarks using the same methodology as the review.
- [x] Request independent review of the integrated diff, resolve actionable findings, and inspect final git diff/status.
- [x] Document test evidence and material compatibility decisions.

## Final validation and compatibility notes

- Rust 1.95.0: rustfmt; strict Clippy (including undocumented unsafe blocks and safety documentation); 39 unit tests and 5 build-discovery tests; native SQLite ABI check; release build.
- Rust 1.70.0: the same 44 unit/discovery tests pass on the integrated source. Native binding regeneration and ABI check also pass with Apple Command Line Tools libclang. A temporary source snapshot was used because this checkout's Git index is not readable by Cargo 1.70's embedded Git library.
- `sh build.sh`: all 55 C++ checks and 112 original Python integration tests pass.
- Python tests explicitly using the Rust release extension: 121 pass (112 original + 9 callback/recovery regressions).
- Independent domain reviews covered native core, SQLite callbacks, and build changes. Follow-up fixes included fallible result-length conversion, discovery-input invalidation, and moving the private load snapshot inside the safe HNSW wrapper.
- Windows/MSVC and PowerShell execution was not available locally. CI now covers Windows including PowerShell deployment and a capacity-error recovery subprocess test, plus Linux, macOS ARM/Intel, and Rust 1.70.
- New saves use a versioned format. Legacy raw HNSW files must be opened using their original extension; export vectors and rowids and reinsert them into the new extension. Native word size/endianness must match. Loading uses temporary disk space. Atomic replacement is not a claim of parent-directory durability across power loss.
- Implementation was validated before the requested commit and push. The existing untracked `AGENTS.md` was preserved.

### Local release benchmark evidence

Same machine and SQLite host, Rust 1.95 before/after, deterministic inputs, warmups, interleaved backends. Scalar/lookup: medians of 7 repetitions of 10,000 calls; insertion/KNN: 1,000 vectors, 500 queries, medians of 3 repetitions, k=10, M=30, construction ef=100. Timings are microseconds; this is a local sample, not a cross-platform performance guarantee.

| Workload | Dimensions | C++ | Rust before | Rust after |
| --- | ---: | ---: | ---: | ---: |
| Scalar L2 | 128 | 0.569 | 0.820 | 0.575 |
| Scalar L2 | 1536 | 0.841 | 2.057 | 0.844 |
| Vector lookup | 128 | 0.906 | 0.841 | 0.671 |
| Vector lookup | 1536 | 2.120 | 1.557 | 0.934 |
| Insert | 128 | 29.548 | 28.491 | 27.975 |
| KNN | 128 | 9.922 | 10.044 | 9.838 |
| Insert | 1536 | 125.109 | 125.257 | 124.028 |
| KNN | 1536 | 53.795 | 56.014 | 53.952 |
