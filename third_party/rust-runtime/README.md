# Selected Rust runtime crate notice inventory

[MANIFEST.json](<MANIFEST.json>) records the conservative union of external
normal dependencies selected for Linux x64, Windows x64/MSVC, and macOS arm64
by the locked default extension build. Build/development-only edges, proc-macro
subtrees, and unused optional branches are excluded. The project/workspace
Apache license and native Highway/hnswlib notices are packaged separately.

Each crate's declared SPDX expression is retained **without selecting or
simplifying an `OR` alternative**. All license/copyright/notice files supplied
in its published crate archive are copied byte-for-byte. Leading source comments
containing per-file license/copyright notices are additionally reproduced in
`SOURCE_NOTICES.txt`, with their original source path and hashes in the manifest.
This is intentionally conservative: it does not assert that every source file or
function survives linking. In particular, libm's Sun, FreeBSD, and CORE-MATH
notices are retained alongside its declared MIT license, not relabeled as MIT.

The DiskANN crates' published archives omit their root license/notice files.
Their copies use the previously verified repository revision and upstream text
recorded in [the DiskANN provenance](<../diskann-0.60.0/PROVENANCE.json>).
The patched `diskann` crate is included even though Cargo reports a local path
source: its upstream archive checksum, baseline revision, modified-file hashes,
and patch description are recorded separately from registry origin. Upstream
notice text remains unmodified. No selected crate currently lacks collected notice text. This is not a legal
review, SPDX interpretation, binary-reachability audit, or compliance certification.
Rust standard-library/toolchain redistribution licensing is outside this crate
inventory; source notices embedded outside leading comments and unlisted upstream
attributions may require further review. Native dependency notices are a separate
inventory.

## Refreshing after a dependency change

The collector uses the existing Cargo cache and does not build anything:

```sh
CARGO_HOME=/path/to/cargo/cache python scripts/collect_runtime_licenses.py \
  --output /path/to/new-runtime-notice-bundle
```

It refuses to overwrite an existing bundle. Review the new inventory, declared
expressions, missing-notice list, source provenance, and target-feature changes
before replacing committed inputs. The collection algorithm and official target
triples are part of the manifest. CI packaging checks require the committed lock
file hash and every bundled notice's SHA-256 to match the current inventory.

Every release wheel contains this entire versioned union under
`vectorlite_py/licenses/rust-runtime`, and native npm staging preserves it under
`src/licenses/rust-runtime`. Shipping the union on each platform avoids dropping
notices when the same extension is packaged through multiple release paths.
