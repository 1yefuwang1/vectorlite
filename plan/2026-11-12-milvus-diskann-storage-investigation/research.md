# How Milvus stores its DiskANN index

Date: 2026-11-12
Scope: source investigation, not a Milvus build or performance benchmark.

## Main finding

**Milvus's Knowhere-backed DiskANN path produces static, per-segment native binary
index files. It does not store a SQLite-like record for every vector.** The default
SSD node combines uncompressed typed coordinates and adjacency in a fixed-capacity
slot; compact navigation codes live separately and are loaded into RAM. Native
query code turns node IDs into local file offsets arithmetically and issues aligned
local-file reads through the configured disk path. Object storage holds the durable index artifacts; ordinary queries use
reconstructed local files, not object-store requests for each graph expansion.

This is not a discovery that vectors should be binary: vectorlite already stores
binary float32 BLOBs. The differences are addressing, layout, navigation/cache
policy, native build execution and immutable segment lifecycle.

```text
Milvus segment/field data
    -> local raw binary vector matrix
    -> native graph build under a RAM budget
    -> packed graph/vector sectors + navigation model/codes + metadata
    -> index artifacts uploaded as transport slices
    -> QueryNode reconstructs local files
    -> RAM navigation codes / hot-node cache + local SSD sector reads
```

## 1. Exact implementation and scope

| Layer | Pin |
|---|---|
| Milvus | v3.0.6, commit `63100f8f5b4b675a12fe78fc9673b005ee318731` |
| Knowhere | `b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747` |
| Native DiskANN | Vendored C++ tree inside that Knowhere commit; subtree Git SHA `165476e7063dab74add8ead409ac9fb5a054ab15` |

Milvus v3.0.6 was the latest published release returned by the
[release API](<https://api.github.com/repos/milvus-io/milvus/releases/latest>) during
this investigation. Its [tag resolves to the pinned commit](<https://api.github.com/repos/milvus-io/milvus/git/ref/tags/v3.0.6>).
The [Milvus dependency declaration](<https://github.com/milvus-io/milvus/blob/63100f8f5b4b675a12fe78fc9673b005ee318731/internal/core/thirdparty/knowhere/CMakeLists.txt#L17-L19>)
selects Knowhere `b045acd9`. Its native DiskANN is an ordinary vendored directory,
not a separately selected Microsoft Rust crate or Git submodule. The
[pinned native build list](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/cmake/libs/libdiskann.cmake#L14-L41>)
compiles `SSDGraphIndex`, replaceable navigation, PQ and RaBitQ, rather than the
older `PQFlashIndex` often mentioned in older explanations.

The public [build script selects KNOWHERE by default](<https://github.com/milvus-io/milvus/blob/63100f8f5b4b675a12fe78fc9673b005ee318731/scripts/core_build.sh#L97-L105>).
A Cardinal build is a separate qualification: this particular native implementation
[registers as DISKANN_DEPRECATED under KNOWHERE_WITH_CARDINAL](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/src/index/diskann/diskann.cc#L1163-L1169>).
Do not generalize this report to a proprietary/replacement Cardinal engine.
Enabling DiskANN itself is also a build option, not implied by the default engine name.

This is **not the Rust DiskANN 0.60.0 implementation** used in our earlier
[upstream build comparison](<../2026-11-12-upstream-diskann-build-comparison/results.md>).
Those measured times are not Milvus performance measurements.

## 2. Build-time representation versus stored index

Milvus stages a dense vector dataset into a local raw binary file. The simple
non-null dense path writes:

```text
[uint32 number_of_vectors][uint32 dimension][contiguous typed coordinates]
```

The [staging writer](<https://github.com/milvus-io/milvus/blob/63100f8f5b4b675a12fe78fc9673b005ee318731/internal/core/src/index/VectorDiskIndex.cpp#L699-L715>)
writes binary values directly, not textual vectors. Nullable/vector-array fields
have additional validity or offset handling; they are not the simple layout above.

The native builder uses a separate aligned vector matrix and in-memory adjacency
`std::vector<std::vector<unsigned>>` during graph construction
([graph members](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/thirdparty/DiskANN/include/diskann/index.h#L243-L247>)).
Build RAM is configurable; this does not assert every production build fits RAM
without partitioning. Milvus [supplies a build budget based on available memory](<https://github.com/milvus-io/milvus/blob/63100f8f5b4b675a12fe78fc9673b005ee318731/pkg/util/indexparams/index_params.go#L270-L301>).

A temporary `P_mem.index` is an adjacency-list stream with a 24-byte header and
per-node `uint32 degree` followed by `uint32 neighbor IDs`. It is not the final
SSD packing ([temporary graph writer](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/thirdparty/DiskANN/src/index.cpp#L385-L416>)).
The layout pass combines vector payload and adjacency into the final disk file.
Thus, separate build-time vector/graph structures do not imply separate query-time
reads of those two components.

## 3. Concrete binary node/sector layout

For ordinary DiskANN, final `P_disk.index` has a metadata sector followed by
fixed-capacity node slots:

```text
slot = [vector payload][uint32 actual_degree][uint32 neighbors...][unused padding]
```

The node ID is implicit in its position. Edges store 32-bit local node IDs, not
physical byte offsets or global entity primary keys.

Let `B` be vector payload bytes and `Rmax` the serialized graph's maximum degree:

```text
M = B + 4*(Rmax + 1)             # fixed node-slot bytes
S = 4096                        # logical sector bytes

if M <= S:
    nodes_per_sector = floor(S/M)
    sector_offset(i) = (1 + floor(i/nodes_per_sector))*S
    slot_offset(i) = (i mod nodes_per_sector)*M
else:
    sectors_per_node = ceil(M/S)
    sector_offset(i) = (1 + i*sectors_per_node)*S
    slot_offset(i) = 0
```

Sector 0 contains file size, node count, medoid, node length, packing factor, frozen
point metadata and optional reorder-region fields. Large nodes can span multiple
sectors: **not every node fits in one 4 KiB page**. See the
[layout writer](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/thirdparty/DiskANN/src/aux_utils.cpp#L977-L1109>)
and [runtime offset calculations](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/thirdparty/DiskANN/include/diskann/ssd_graph_index.h#L262-L272>).

Illustration only: with FP32 L2 coordinates and a serialized maximum degree of 56,
128D needs `512 + 4*57 = 740` slot bytes (five slots per sector); 768D needs
`3072 + 228 = 3300` bytes (one per sector). Actual packing follows the graph header.
Padding means fixed-slot packing is **not guaranteed to be smaller** than a variable
record format. Its main structural advantage is direct ID-to-offset addressing.
This native binary serialization is not a universal self-describing interchange
format; the correct type/metric/index metadata still matters.

## 4. Navigation encoding is independent of SSD payload encoding

| Choice | Pinned behavior |
|---|---|
| Navigation default | PQ; selectable alternatives are exactly PQ and RABITQ |
| PQ navigation | One byte per subquantizer, 256 centroids; complete codes/model resident in RAM |
| RaBitQ navigation | FP32 input only; a separately serialized model/index |
| Default SSD coordinates | `disk_pq_dims=0`: uncompressed original element type T |
| Optional disk PQ | Nonzero `disk_pq_dims`: PQ-code payload replaces node coordinates, with separate model files |

The [navigation selector](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/src/index/diskann/diskann_config.h#L24-L40>)
is separate from the [disk-PQ option](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/src/index/diskann/diskann_ssd_config.h#L98-L105>).
The [supported-codec set](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/thirdparty/DiskANN/src/navigation_runtime.cpp#L107-L118>)
contains no SQ/SQ8 navigation selector. The
[registration macro](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/include/knowhere/index/index_factory.h#L137-L141>)
explicitly covers FP32, FP16 and BF16; RaBitQ is
[FP32-only](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/thirdparty/DiskANN/include/diskann/rabitq_navigation.h#L19-L24>).
Therefore, "full vectors on SSD" does not always mean FP32, nor does PQ navigation
imply the SSD coordinates themselves are compressed.

Milvus supplies budgets that bare Knowhere does not infer by itself. The checked
[configuration defaults](<https://github.com/milvus-io/milvus/blob/63100f8f5b4b675a12fe78fc9673b005ee318731/configs/milvus.yaml#L1215-L1222>)
are degree 56, build list 100, PQ budget ratio **0.125**, search-cache ratio **0.1**,
with thread/beam ratios also configured. The
[parameter layer](<https://github.com/milvus-io/milvus/blob/63100f8f5b4b675a12fe78fc9673b005ee318731/pkg/util/indexparams/index_params.go#L168-L207>)
can override these via AutoIndex settings. Its
[build budget calculation](<https://github.com/milvus-io/milvus/blob/63100f8f5b4b675a12fe78fc9673b005ee318731/pkg/util/indexparams/index_params.go#L270-L301>)
scales PQ bytes from raw field-data size. For FP32 this corresponds approximately
to `0.125 * 4*d` code bytes/vector before rounding and the native dimension/512-chunk
caps ([PQ sizing](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/thirdparty/DiskANN/src/pq_navigation.cpp#L52-L98>)).
It is not the 32/96-byte choice used in our separate Rust-builder experiment.

For L2, the uncompressed payload preserves the declared typed coordinates. IP and
cosine have metric-specific transformations/norm handling; do not describe every
metric's bytes as untouched original FP32
([preprocessing](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/thirdparty/DiskANN/src/aux_utils.cpp#L1615-L1666>)).

## 5. Index artifacts

For an abstract prefix `P` (Milvus may supply a directory-like prefix):

```text
P_disk.index                 packed node sectors
P_navigation.meta            versioned navigation-codec selection

PQ navigation:
P_pq_compressed.bin          per-node navigation codes
P_pq_pivots.bin              PQ model
..._rearrangement_perm.bin   coordinate permutation, NOT node reordering
..._chunk_offsets.bin
..._centroid.bin

RaBitQ navigation instead:
P_rabitq.index               serialized Faiss model/index/codes
```

Other path-dependent artifacts include medoids/centroids, metric norms, sample
queries, cached-node IDs, embedding-list offsets, nullable validity/id-map data,
and independent disk-PQ pivot sidecars. Do not treat every optional file as present
in every index. The [file collector](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/src/index/diskann/diskann.cc#L389-L423>)
and [PQ loader](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/thirdparty/DiskANN/src/pq_navigation.cpp#L117-L172>)
show the distinction.

The navigation marker uses `KHNAVIDX`, version 1. Missing metadata supports legacy
PQ; malformed present metadata fails rather than silently selecting another codec
([metadata implementation](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/thirdparty/DiskANN/src/navigation_runtime.cpp#L17-L118>)).

Native optional full-precision reorder-region support is not the ordinary Knowhere
build/search path: [Build passes reorder=false](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/src/index/diskann/diskann.cc#L515-L525>),
and [normal search disables that region](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/src/index/diskann/diskann.cc#L992-L995>).
With disk PQ enabled, do not assume automatic exact-coordinate reranking inside
this native path. [AiSAQ has a separate conditional layout](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/thirdparty/DiskANN/src/aux_utils.cpp#L1768-L1816>)
with its own inline/rearranged PQ and mapping artifacts—not the universal DISKANN layout.

## 6. Durable storage and local working files

Milvus creates a `DiskFileManagerImpl` and passes it into Knowhere. After native
construction, Knowhere calls `AddFile()` for the required and present optional
artifacts. Under this manager, `AddFile()` invokes the upload path: large files are
split into configurable **transport slices**, named by file and slice ordinal.
These are not the 4 KiB search sectors
([upload/slicing](<https://github.com/milvus-io/milvus/blob/63100f8f5b4b675a12fe78fc9673b005ee318731/internal/core/src/storage/DiskFileManagerImpl.cpp#L358-L434>)).
The manager uses the configured storage backend; remote is the checked production
[configuration default](<https://github.com/milvus-io/milvus/blob/63100f8f5b4b675a12fe78fc9673b005ee318731/configs/milvus.yaml#L1227>),
while local backends are also possible.

On load, slices are grouped by original filename, numerically sorted, fetched,
decoded and appended to recreate local index files
([reconstruction](<https://github.com/milvus-io/milvus/blob/63100f8f5b4b675a12fe78fc9673b005ee318731/internal/core/src/storage/DiskFileManagerImpl.cpp#L545-L633>)).
Local paths are scoped by build/version/segment/field identity and a local generation
([local prefix](<https://github.com/milvus-io/milvus/blob/63100f8f5b4b675a12fe78fc9673b005ee318731/internal/core/src/storage/DiskFileManagerImpl.cpp#L1580-L1600>)).

For the ordinary DiskANN implementation, `LoadIndexWithStream()` is not overridden
and its [base default is false](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/include/knowhere/index/index_node.h#L973-L976>).
The [selection helper returns all index files for that case](<https://github.com/milvus-io/milvus/blob/63100f8f5b4b675a12fe78fc9673b005ee318731/internal/core/src/index/VectorIndexValidDataUtils.h#L78-L83>),
and Milvus [caches them to disk before Deserialize](<https://github.com/milvus-io/milvus/blob/63100f8f5b4b675a12fe78fc9673b005ee318731/internal/core/src/index/VectorDiskIndex.cpp#L334-L365>).
The [manager's LoadFile callback is a no-op](<https://github.com/milvus-io/milvus/blob/63100f8f5b4b675a12fe78fc9673b005ee318731/internal/core/src/storage/DiskFileManagerImpl.cpp#L212-L215>)
returning true, not a per-node remote GET; the explicit preload did the work.

A subtle timing point: Knowhere Build already reaches the file-manager upload
path. Later [native Serialize is explicitly a no-op](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/src/index/diskann/diskann.cc#L143-L147>),
while [Milvus Upload returns recorded file metadata](<https://github.com/milvus-io/milvus/blob/63100f8f5b4b675a12fe78fc9673b005ee318731/internal/core/src/index/VectorDiskIndex.cpp#L431-L445>). Native local graph-build timing alone therefore does not
represent a full Milvus cluster build including staging and artifact transfer.

## 7. Query-time RAM and actual SSD reads

**RAM:** all navigation codes/model, medoids/centroids, relevant norms, per-thread
scratch, plus a bounded hot-node cache. The cache holds **both neighborhood and
node payload**, not only IDs
([cache loading](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/thirdparty/DiskANN/src/ssd_graph_index.cpp#L315-L377>)).
Selection can use cached-node artifacts, BFS or sample-query visitation. Milvus
supplies its own cache budget and disables prepare warm-up/BFS in the wrapper;
Knowhere's standalone defaults alone do not describe server behavior.

**Local SSD:** uncached nodes' packed payload/adjacency sectors. Knowhere constructs
`LinuxAlignedFileReader` and loads `SSDGraphIndex`
([load path](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/src/index/diskann/diskann.cc#L694-L717>)).
The file opens with **O_DIRECT | O_RDONLY**, and the Linux reader batches libaio
submission/completion
([open flags](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/thirdparty/DiskANN/src/linux_aligned_file_reader.cpp#L146-L152>),
[batch I/O](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/thirdparty/DiskANN/src/linux_aligned_file_reader.cpp#L33-L111>)).
This is actual aligned local-file I/O rather than preloading the complete graph,
normally backed by the deployment's configured SSD path. The source alone does
not prove the physical medium. It differs from the macOS RAM-loading fallback
encountered in our separate Rust-provider benchmark.

A search iteration separates cached nodes from uncached frontier, reads a beam of
uncached nodes, scores expanded-node payloads and scores neighbor candidates using
resident navigation codes
([beam expansion](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/thirdparty/DiskANN/src/ssd_graph_index.cpp#L1231-L1403>)).
Ordinary search waits for the read batch; do not call the entire traversal fully
asynchronous. It can submit one request per frontier ID even when nodes share a
sector. Separate [vector-retrieval code groups by sector and pipelines reads](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/thirdparty/DiskANN/src/ssd_graph_index.cpp#L1691-L1742>).
[CPU prefetch of PQ codes](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/thirdparty/DiskANN/include/diskann/pq_table.h#L18-L44>)
is distinct from SSD read-ahead.

## 8. Inserts, deletes and transactional differences

The disk node's `Train` and `Add` return `not_implemented`; Build rejects an already
populated prefix, and the query file descriptor is read-only
([mutation interface](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/src/index/diskann/diskann.cc#L65-L73>),
[overwrite guard](<https://github.com/zilliztech/knowhere/blob/b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747/src/index/diskann/diskann.cc#L470-L477>)).
This is a bulk-build/read-oriented graph, not in-place SQL INSERT/UPDATE into the
existing disk-node slots.

Milvus has a separate [growing-segment insertion path](<https://github.com/milvus-io/milvus/blob/63100f8f5b4b675a12fe78fc9673b005ee318731/internal/core/src/segcore/SegmentGrowingImpl.cpp#L775-L785>).
For sealed segments, Delete records PK/timestamp deletions rather than mutating the
DiskANN file
([sealed Delete](<https://github.com/milvus-io/milvus/blob/63100f8f5b4b675a12fe78fc9673b005ee318731/internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp#L6717-L6773>)).
The [timestamp-aware delete record](<https://github.com/milvus-io/milvus/blob/63100f8f5b4b675a12fe78fc9673b005ee318731/internal/core/src/segcore/DeletedRecord.h#L227-L277>)
sets query mask bits, and that mask reaches the vector-index search.
Rebuild/replacement is separate from changing this file in place. The complete
coordinator publication, upsert and compaction scheduling protocols were not audited
here; no claim about cluster-wide ACID equivalence is made.

## 9. What transfers to vectorlite—and what does not

| Idea | Relevance and constraint |
|---|---|
| Keep scoring payload with adjacency | Already aligned with our Format-3 node rows. Milvus's final SSD slot co-locates them; its separate build arrays do not justify introducing a join on every SQLite read. |
| Compact local IDs | Native u32 IDs save edge bytes. Our arbitrary public rowids and internal ID range must not be silently narrowed; a lossless, versioned encoding would need its own benchmark. |
| Page/block-oriented fixed slots | Can preserve FP32 and amortize access across several nodes. SQLite still requires pager/B-tree access, and large BLOB overflow chains or reading unused slots can regress performance. |
| RAM hot-node cache | Works with managed read-only index generations. Our SQL writes, triggers and reentrant host BLOB changes require a stronger coherence contract; copying this cache directly is unsafe. |
| Resident compact navigation codes | Changes query approximation/recall and has codebook/model costs. It is not simply a lossless binary repacking optimization. |
| Bulk native build then publish | Avoids per-node database writes during construction. A SQLite-contained bulk-build mode would be a separate bounded-workspace/rollback design, not an automatic replacement for interactive transactional mutations. |
| Aligned direct reads | Native code owns its file format. Direct-reading SQLite database pages would bypass SQLite's pager/WAL/snapshot rules and is not an acceptable shortcut. |

Our [current encoder](<../../vectorlite/src/diskann_store.rs#L354-L360>) already
writes raw binary float32. The earlier
[split-table regression](<../2026-11-10-garnet-provider-optimizations/results.md>)
shows why fewer record bytes do not necessarily mean less wall time: extra lookups
and poorer locality can dominate. The most relevant lossless experiments are
compact IDs, grouped/page-aware reads and carefully bounded block layout—not
conversion from text to binary, and not an unmeasured promise that fixed slots
will reproduce Milvus performance inside SQLite.

## Verification and limits

- Exact Milvus release and Knowhere commit pinned; native DiskANN identified by
  its enclosing commit and vendored path, not an invented separate revision.
- Knowhere source archive SHA-256:
  `6f3fee3264b4a1fda003d15c5968af6a671507244cb8090b969be1b867b8415a`.
- Milvus lifecycle files were retrieved individually from immutable raw URLs with
  recorded file hashes. A larger archive transfer was canceled and is not used as
  complete-source evidence. Cached sources/provenance remain ignored local artifacts.
- Native layout/query path was independently investigated and reviewed by a read-only
  subagent; no blocking native-layer factual corrections remained.
- Checked 48 pinned source citations and their line bounds, compared 16 cited
  Knowhere files against the archive, verified hashes of 16 selected Milvus files,
  and checked 25 local documentation links.
- No Milvus/Knowhere/DiskANN source edits, builds, installations, deployments or
  benchmarks were performed. No speedup, lossless-recall equivalence, or applicability
  to every Milvus distribution/configuration is claimed.
- Source/main branches, versionless docs, optional AiSAQ/Cardinal modes and later
  releases can differ. The report applies to the explicitly scoped pinned path.
