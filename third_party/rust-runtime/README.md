# Selected Rust runtime crate notice inventory

[MANIFEST.json](<MANIFEST.json>) records the conservative union of external
normal dependencies selected for Linux x64, Windows x64/MSVC, and macOS arm64
by the locked default extension build. Build/development-only edges, proc-macro
subtrees, and unused optional branches are excluded. The project/workspace
Apache license and native Highway/hnswlib notices are packaged separately.

Every selected crate is an **unmodified exact-version official crates.io
dependency**. Cargo.lock records the registry archive checksums; there is no
Cargo dependency patch, local path source, vendored DiskANN algorithm/provider
source, fork, or private-helper copy. The current inventory has 66 selected
external runtime crates (18 excluded build/dev/proc-macro/optional packages), is
based on Cargo.lock SHA-256
0e729163e3fd7c377c32da9227caa00f595b0564e73cf0177093823297e103b4, and records
each crate's registry archive checksum plus its Cargo-observed VCS metadata
(including the pinned upstream DiskANN revision where the archive carries
.cargo_vcs_info.json).

Each crate's declared SPDX expression is retained **without selecting or
simplifying an OR alternative**. All license/copyright/notice files supplied in
its published crate archive are copied byte-for-byte. Leading source comments
containing per-file license/copyright notices are additionally reproduced in
SOURCE_NOTICES.txt, with their original source path and hashes in the manifest.
No algorithm or provider source file is bundled: the inventory contains license,
copyright, notice, and source-comment excerpt text only, and no .rs/.c/.h files.
This is intentionally conservative: it does not assert that every source file or
function survives linking. In particular, libm's Sun, FreeBSD, and CORE-MATH
notices are retained alongside its declared MIT license, not relabeled as MIT.

The DiskANN crates' published archives omit their root license/notice files.
Their copies use the previously verified repository revision and upstream text
recorded in [the DiskANN provenance](<../diskann-0.60.0/PROVENANCE.json>).
All six DiskANN 0.60.0 crates (diskann, diskann-providers, diskann-quantization,
diskann-utils, diskann-vector, diskann-wide) are selected from the crates.io
registry; their root license/notice text and Cargo VCS revision are retained
without vendor source or a local patch record. No selected crate currently lacks
collected notice text. This is not a legal review, SPDX interpretation,
binary-reachability audit, or compliance certification. Rust standard-library/
toolchain redistribution licensing is outside this crate inventory; source
notices embedded outside leading comments and unlisted upstream attributions may
require further review. Native dependency notices are a separate inventory.

## Refreshing after a dependency change

The collector uses the existing Cargo cache and does not build anything. It
refuses to overwrite an existing output directory.

    CARGO_HOME=/path/to/cargo/cache python scripts/collect_runtime_licenses.py \
      --output /path/to/new-runtime-notice-bundle

Review the new inventory, declared expressions, missing-notice list, source
provenance, and target-feature changes before replacing committed inputs. The
collection algorithm and official target triples are part of the manifest. CI
packaging checks require the committed lock file hash, source bundle README/
MANIFEST identity, and every bundled notice's SHA-256 to match the current
inventory.

Every release wheel contains this entire versioned union under
vectorlite_py/licenses/rust-runtime, and native npm staging preserves it under
src/licenses/rust-runtime. Shipping the union on each platform avoids dropping
notices when the same extension is packaged through multiple release paths.
