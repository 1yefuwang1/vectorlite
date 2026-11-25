# Upstream provider and SSD-builder selection

Date: 2026-11-12

This note records which upstream implementations the [benchmark plan](<plan.md>)
uses and why. It is not a claim that the implementations have identical storage,
algorithms, numerical results, or durability.

## Exact upstream source

Rust DiskANN **0.60.0**, revision
`97a828a500848018d8be28c3ea6d5a585b07362f`, matching vectorlite's component revision.
The [pinned workspace manifest](<https://github.com/microsoft/DiskANN/blob/97a828a500848018d8be28c3ea6d5a585b07362f/Cargo.toml>)
contains both `diskann-providers` and `diskann-disk`.

Source archive: [GitHub codeload](<https://codeload.github.com/microsoft/DiskANN/tar.gz/97a828a500848018d8be28c3ea6d5a585b07362f>),
SHA-256 `88ea51a9db42d483b330d1be1ea764d0fa8ac7e39876aa9bb02878bd0c1377c1`.
The build helper and independent reviewer each checked all **1,410 regular archived
files** against the extracted tree. No upstream source patch was applied for this
comparison. This differs from vectorlite's recorded vendored first-error/task-draining
patch; successful native builds use the untouched upstream implementation.

## 1. Standard full-precision RAM provider

The provider is `diskann-providers::FullPrecisionProvider<f32>`, created through
`diskann_providers::index::diskann_async::new_index`. This is the type registered
by the upstream standard full-precision graph-index benchmark, and supports graph
and vector save/load. See the
[official construction helper](<https://github.com/microsoft/DiskANN/blob/97a828a500848018d8be28c3ea6d5a585b07362f/diskann-providers/src/index/diskann_async.rs>)
and [benchmark registration](<https://github.com/microsoft/DiskANN/blob/97a828a500848018d8be28c3ea6d5a585b07362f/diskann-benchmark/src/index/benchmarks.rs>).

The repository also has a newer `diskann-inmem` provider. It is a different
implementation and explicitly has no save/load support at this revision; it is
not silently substituted for the standard persisted full-precision baseline.

For the controlled comparison, the driver selects first-vector initialization,
one current-thread runtime, one live-point bootstrap followed by batch-8 inserts,
IBC 8, max minibatch parallelism 1, degree 32 with default slack (physical degree 41),
build list 100 and alpha float32 1.2. These match the SQLite integration where
possible; they are **not a claim that every upstream CLI default is identical**.
The upstream CLI normally permits other roots, larger batches and more concurrency.

Native live IDs are `0..N-1` and the frozen root is ID `N`; SQLite's root is 0 and
its live internal IDs are shifted by 1. Public/query labels remain 0-based in both.
Memory allocation/root setup, graph insertion, save and reload are separately timed.
Memory save/load must preserve exact ordered query IDs and float32 distance bits.
The original index is dropped before reload, avoiding two simultaneously live native
indexes during that check. Query validation is outside the build timer.

## 2. Genuine SSD-format construction

The builder is
[`DiskIndexBuilder`](<https://github.com/microsoft/DiskANN/blob/97a828a500848018d8be28c3ea6d5a585b07362f/diskann-disk/src/build/builder/build.rs>),
not an in-memory index followed only by `save()`.

The driver uses full-precision graph construction (`QuantizationType::FP`) and
includes the entire constructor/build pipeline in its main wall timer:

1. Mandatory disk-search PQ codebook training (256 centers, up to 12 Lloyd iterations).
2. Compress the full dataset for disk-search navigation.
3. Find a sampled medoid and build the graph with sequential single inserts.
4. Static final prune and intermediate graph save.
5. Pack vectors and ordered adjacency into actual 4 KiB disk sectors.
6. Close/flush outputs and remove the temporary in-memory graph artifact.

The 2 GiB build RAM budget permits the official one-shot strategy for these small
inputs; this is not a partitioned or beyond-RAM build measurement. Explicit search
PQ byte counts are 32 for 128D and 96 for 768D (`clamp(dim/4,1,96)`). They are recorded
benchmark choices, not a purported universal upstream default. All 3k/10k input
vectors participate in training because the upstream sample limit is 50,000.

The public API does not expose accurate independent timings for every private
pipeline stage. The driver reports constructor, full build pipeline, and total
ready times rather than inventing a graph-only or PQ-only breakdown. Full-precision
graph building does **not** disable mandatory disk-search PQ training.

Output consists of the genuine `_disk.index`, `_pq_pivots.bin`, and
`_pq_compressed.bin` files. The
[disk-layout writer](<https://github.com/microsoft/DiskANN/blob/97a828a500848018d8be28c3ea6d5a585b07362f/diskann-disk/src/storage/disk_index_writer.rs>)
uses normal writes and flush, without an explicit fsync added by our driver.
SQLite's transaction commit uses its own journal/synchronous contract, so persisted
build times do not establish equal durability guarantees.

## macOS and SIMD constraints

- Relevant upstream linear algebra uses pure-Rust faer, not MKL/OpenBLAS. The
  actual pinned source built on this macOS ARM64 host without installing a BLAS.
- Both default compiler feature flags `neon` and `dotprod` are enabled on this Apple
  target. Native f32 distances use the upstream SIMD backend, not a forced scalar
  reference. Vectorlite uses Highway; no distance-kernel substitution was made.
- The
  [macOS saved-index reader factory](<https://github.com/microsoft/DiskANN/blob/97a828a500848018d8be28c3ea6d5a585b07362f/diskann-disk/src/search/provider/aligned_file_reader/factory/file.rs>)
  deliberately selects a fallback whose
  [reader](<https://github.com/microsoft/DiskANN/blob/97a828a500848018d8be28c3ea6d5a585b07362f/diskann-disk/src/search/provider/aligned_file_reader/reader/storage_provider.rs>)
  loads the complete disk file into a `Vec<u8>`. It validates the saved sectorized
  format and recall, but **does not measure SSD-resident query latency**.
- Linux io_uring / Windows IOCP search behavior and performance were not executed
  on this host. A source build claiming macOS 11 as its deployment target is not
  evidence of actual execution on macOS 11.

## Why this is not a storage-only ratio

The comparison keeps inputs and major graph parameters fixed, but native versus
SQLite builds still differ in distance kernels, runtime lifetime, validation/workspace
checks, transactional behavior and persistence. The SSD builder additionally uses
PQ navigation, a medoid root, single insertions and static final pruning. Report
absolute build times and recall per implementation; do not label their ratio as
pure SQL overhead or a promised speedup from swapping storage alone.

## Stored completeness is not full root reachability

The first campaign stopped before its 768D timing case could be accepted because
a diagnostic counted 9,999 reachable nodes out of 10,001 logical nodes. The frozen
root is included, so this means two live vertices are unreachable—not that the
root was omitted by the counter. This alone does not establish missing insertions.

The pinned [reachability method](<https://github.com/microsoft/DiskANN/blob/97a828a500848018d8be28c3ea6d5a585b07362f/diskann/src/graph/index.rs#L2150-L2180>)
is documented as analysis/sanity instrumentation. Later
[backedge pruning](<https://github.com/microsoft/DiskANN/blob/97a828a500848018d8be28c3ea6d5a585b07362f/diskann/src/graph/index.rs#L2239-L2328>)
can replace a source's outgoing list without protecting a spanning tree or repairing
a target's last incoming edge. Full root reachability is not an enforced insertion
invariant. The official SSD builder only logs reachability in debug builds.

The corrected harness does **not** repair the graph or lower its parameters. It
requires stronger storage checks instead, before build results can be accepted:

- Every live vector and the frozen copy bit-match their expected inputs.
- Every logical node has nonempty adjacency bounded by physical degree, valid
  targets, no self-edges, and no duplicates.
- Post-load exhaustive audits match the post-build audits, in addition to exact
  ordered query IDs/distances and repeatable persisted-file fingerprints.
- Reachable/unreachable counts and unreachable IDs/degrees are published separately
  as quality diagnostics. An extra unused native adjacency capacity slot is not
  treated as an additional logical vector.

This distinction is necessary because payload presence alone is insufficient:
the untouched upstream multi-insert path can log some graph-phase errors without
propagating them. The exhaustive edge audit remains mandatory. The interrupted
campaign is retained and excluded from the final timing campaign; corrections are
reviewed before fresh measurements.
