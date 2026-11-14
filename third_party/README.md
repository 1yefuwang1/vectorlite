# Bundled component notices and artifact validation

The extension's own [Apache-2.0 license](<../LICENSE>) is unchanged. The
[DiskANN 0.60.0 license, notice, and provenance](<diskann-0.60.0/PROVENANCE.json>)
are pinned to the release's recorded source revision; its Microsoft and NSG
copyright/permission notices must accompany the compiled components.
The [selected Rust runtime inventory](<rust-runtime/MANIFEST.json>) additionally
records the normal dependency union for all three official targets, with exact
upstream texts and source-comment notices. It preserves declared SPDX alternatives
without choosing one. This is provenance data, not legal review or a compliance
certification; the [inventory scope](<rust-runtime/README.md>) identifies its limits.

CMake installs these files under `vectorlite_py/licenses`, together with the
hnswlib and Highway copyrights from the **selected native installation**. Wheel
staging copies that same directory into each native npm package's `src/licenses`.
The project license is also present at each npm package root. The wrapper package
and native package metadata continue to identify the project's Apache license;
bundled MIT components retain their own notices.

## macOS release floor

The repository-owned `arm64-osx` overlay triplet pins native dependencies to
macOS 11.0. Its contents participate in the native cache ABI and CI cache key.
User-supplied CMake/environment overlays retain priority. A custom overlay or
explicit newer local deployment target does **not** establish macOS 11 support.

CMake checks native archive members before linking. CI checks the final arm64
Mach-O dylib and the built wheel's `macosx_11_0_arm64` tag and binary metadata.
The [artifact checker](<../scripts/check_macos_artifact.py>) can be run directly:

```sh
python scripts/check_macos_artifact.py path/to/vectorlite.dylib path/to/libhwy.a
python scripts/check_macos_artifact.py wheelhouse/*.whl
```

These checks reject missing/ambiguous metadata and objects declaring a newer
minimum OS. They do not test runtime behavior on macOS 11. A successful test run
on a newer macOS host must not be described as a macOS 11 runtime validation.
Use fresh native installation/build roots when changing the triplet or floor;
relinking a previously cached newer-OS object is not a substitute for rebuilding.

## Release payload checks

Wheel checks inspect actual ZIP members and compare the DiskANN notices with
pinned source text. npm tests inspect the selected native package and npm's
actual dry-run pack file list, then exercise both HNSW and transactional,
file-backed DiskANN through the Node SQLite driver. Synthetic unit fixtures test
these guards without loading an extension or claiming large-dataset/RSS results.
