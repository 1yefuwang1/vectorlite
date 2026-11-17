# Garnet DiskANN storage-provider investigation

Date: 2026-11-09

Scope: read-only source investigation and design comparison. This is not an implementation plan, a benchmark of Garnet, or evidence of a new vectorlite speedup. No implementation changes, builds, or benchmarks were performed for this investigation.

Original request:

> There's a garnet(redis in c#) diskann storage provider. Check how storage works for it and see if we can borrow some desgin ideas

## Executive summary

Garnet's DiskANN integration uses **separate namespaced binary key-value records**, **batched callback-based reads**, **callback-local borrowed vector bytes**, **owned pruning working-set reuse**, and **record-level read-modify-write**. It does not use Redis GET/SET or network requests inside graph traversal and does not use the fixed-sector native DiskANN disk-index layout discussed separately.

The most useful ideas for vectorlite are:

1. Bounded multi-ID reads and distance computation while each SQLite row's BLOB lease is live.
2. Reuse of owned prune vectors and allocations within a valid snapshot/write-free interval, instead of destroying the working set on every fill.
3. Seeding from an already-owned, validated batch where stored-data coherence is proved.
4. Separating small mutable graph records from large vector payloads, potentially with fixed adjacency capacity.
5. Removing redundant adjacency reads and reusing bounded buffers.

These are performance hypotheses, not measured gains. Garnet's concurrency, missing-record tolerance, ID reuse and persistence semantics are **not** equivalent to vectorlite's whole-INSERT rollback, corruption checks and snapshot behavior.

## Source revisions and limits

The investigation pinned these upstream source snapshots:

| Repository | Revision | Purpose |
|---|---|---|
| [Microsoft Garnet](<https://github.com/microsoft/garnet/tree/5cc8f3f50aa57ed694dfda928e4467593e073656>) | `5cc8f3f50aa57ed694dfda928e4467593e073656` | C# command, callback, Tsavorite and persistence integration |
| [Microsoft DiskANN](<https://github.com/microsoft/DiskANN/tree/eaeb12578547a84c160909ec639d1a8564cc7216>) | `eaeb12578547a84c160909ec639d1a8564cc7216` | Rust `diskann-garnet` provider, FFI and working sets |

Both the Garnet package reference and the native provider source call the provider package `9.0.0`. That alone does **not** establish that the two main-branch snapshots are a deployed ABI-paired release. Some newer native quantizer variants are absent from the inspected managed enums. Borrow interfaces/designs, not an unverified binary pairing.

Vectorlite's generic indexing dependency is pinned separately to DiskANN `0.60.0`, revision `97a828a500848018d8be28c3ea6d5a585b07362f`; see [component provenance](<../../third_party/diskann-0.60.0/PROVENANCE.json>). Relevant working-set/overlay functionality already exists in that pinned dependency, but API and ownership compatibility still need checking before implementation.

Garnet Vector Sets are a preview feature. Published benchmark numbers, default quantization and settings are not a matched comparison with vectorlite. In particular, the published/default Q8/M16/EF200 configuration is not our float32/R32/build-list-100 setup.

## 1. Architecture: embedded callbacks, not network storage

```text
Client's RESP command
    -> Garnet StorageSession / vector-set operations
    -> native Rust DiskANN algorithm
    -> read / write / delete / RMW function-pointer callbacks
    -> typed C# Tsavorite sessions
    -> hash-indexed hybrid-log records, memory and disk
```

The native provider is Rust. The server and storage bridge are C#. The graph algorithm calls the embedded Tsavorite API directly; there are no per-neighbor loopback Redis GET/SET operations.

The managed callback implementation explicitly describes its data as part of a Vector Set operation, **not a RESP command**: [VectorSessionFunctions](<https://github.com/microsoft/garnet/blob/5cc8f3f50aa57ed694dfda928e4467593e073656/libs/server/Storage/Functions/VectorStore/VectorSessionFunctions.cs#L14-L17>).

At the inspected Garnet revision, string/object/unified/vector sessions are all created from the same store with a shared session ID override. The vector session operates raw namespaced bytes, not ordinary managed Redis object instances. Do not assume an older architecture with a separate SpanByte main store versus object store: [StorageSession](<https://github.com/microsoft/garnet/blob/5cc8f3f50aa57ed694dfda928e4467593e073656/libs/server/Storage/Session/StorageSession.cs#L113-L146>).

Tsavorite's physical record includes record metadata, namespace/key information, and inline value bytes or references to overflow data. `PinnedSpanByte` is a borrow descriptor, not the persistent node-record schema. The precise physical header is version-specific; do not reconstruct it from an older description of the SpanByte struct.

## 2. Keys, namespaces and binary values

A native context is a `u64`; its least significant three bits distinguish term namespaces. The same internal node ID can key a full vector and its adjacency value without combining them into one record.

| Term tag | Term | Key | Value |
|---:|---|---|---|
| 0 | Full vector | Internal `u32` ID | Raw `dimension * sizeof(T)` bytes, no per-vector dimension/header |
| 1 | Neighbors | Internal `u32` ID | Fixed `(max_degree + 1) * 4` bytes |
| 2 | Quantized vector | Internal `u32` ID | Separate quantizer-specific binary representation |
| 3 | Attributes | Internal `u32` ID | Separate variable-length attribute bytes |
| 4 | Metadata | Internal metadata keys | FSM, quantizer and import/lifecycle state |
| 5 | IntMap | External byte-string ID | Internal `u32` ID |
| 6 | ExtMap | Internal `u32` ID | External byte-string ID |

**IntMap and ExtMap names must not be inverted.** Graph algorithms work on compact internal IDs; external IDs are arbitrary byte strings and are resolved for user-facing operations/results.

Read-call keys are four-byte length-prefixed: an internal-key batch is `[4u32, id, 4u32, id, ...]`. The prefix `4` is the **key length**, not a vector or adjacency header. Write/delete/RMW callbacks receive unprefixed key bytes and a separate key length.

The provider stores raw native POD bytes under a little-endian managed-host assumption. Metadata identifiers described as ASCII magic keys in documentation are constructed numerically in source; do not infer their literal serialized spelling without checking the endian conversion.

Sources: [callback/key/term implementation](<https://github.com/microsoft/DiskANN/blob/eaeb12578547a84c160909ec639d1a8564cc7216/diskann-garnet/src/garnet.rs>), [data design](<https://github.com/microsoft/DiskANN/blob/eaeb12578547a84c160909ec639d1a8564cc7216/diskann-garnet/docs/data-design.md>), [provider ID mappings](<https://github.com/microsoft/DiskANN/blob/eaeb12578547a84c160909ec639d1a8564cc7216/diskann-garnet/src/provider.rs#L1760-L1800>).

### Fixed-capacity neighbor format

```text
u32 neighbor slots [0 .. max_degree)
u32 logical count  [max_degree]          <- FINAL word
```

The count is not a prefix. Only slots below the count are active. For storage capacity 16, the value is 68 bytes; for capacity 32, it is 132 bytes.

This stabilizes allocation and update size. However, Garnet's max-degree argument is a hard storage cap; its FFI derives a target degree using the graph slack factor. It is not automatically equivalent to vectorlite's configured pruned degree. Do not reduce vectorlite's existing physical slack or drop edges to copy this format.

Sources: [neighbor reads/replacement/append](<https://github.com/microsoft/DiskANN/blob/eaeb12578547a84c160909ec639d1a8564cc7216/diskann-garnet/src/provider.rs#L1642-L1757>), [import size validation](<https://github.com/microsoft/DiskANN/blob/eaeb12578547a84c160909ec639d1a8564cc7216/diskann-garnet/src/provider.rs#L1280-L1355>), [degree configuration](<https://github.com/microsoft/DiskANN/blob/eaeb12578547a84c160909ec639d1a8564cc7216/diskann-garnet/src/lib.rs#L345-L357>).

### Allocation and root

The start point defaults to `u32::MAX`, is excluded from normal results/FSM accounting, and has specially cached vector/neighbor terms. The FSM uses 8 KiB blocks covering 65,536 IDs and maintains an available-ID queue. **Freed IDs are reused**, unlike vectorlite's monotonic, never-reused internal IDs. That allocator is not directly borrowable under our current policy: [FSM implementation](<https://github.com/microsoft/DiskANN/blob/eaeb12578547a84c160909ec639d1a8564cc7216/diskann-garnet/src/fsm.rs>).

## 3. Read path and buffer ownership

For each expanded graph node:

1. Copy its fixed-sized neighbor value into reusable scratch.
2. Filter neighbor IDs already visited by the search.
3. Submit one multi-key read for that node's remaining vectors.
4. Calculate distances directly on bytes lent to the read callback.

This is a batch per expanded adjacency list, not a single giant whole-beam/full-graph read and not a persistent general-purpose vector cache: [search accessor](<https://github.com/microsoft/DiskANN/blob/eaeb12578547a84c160909ec639d1a8564cc7216/diskann-garnet/src/provider.rs#L2022-L2231>).

The C# batch path uses `ReadWithPrefetch`, carries result indices and callback context, and drains pending operations with `CompletePending(wait: true)` before returning to native. A value-size hint plus record overhead helps avoid undersized first disk reads: [managed batch callbacks](<https://github.com/microsoft/garnet/blob/5cc8f3f50aa57ed694dfda928e4467593e073656/libs/server/Resp/Vector/VectorManager.Callbacks.cs#L29-L231>).

The managed reader passes `ValueSpan` directly to the native callback. Inline values are already pinned; an overflow byte array is temporarily pinned and unpinned in `finally`. Ordinary native callbacks suppress the GC transition; attribute callbacks do not because filtering can reenter managed code.

**The pointer is valid only for the callback.** This is zero-copy at that boundary, not universal disk-to-CPU zero-copy. Other helpers and pruning explicitly copy data. SQLite column BLOB pointers must similarly never outlive step/reset/finalization or an invalidating conversion, and must not be treated as writable buffers.

There is also an alignment caution: native helpers document eight-byte alignment, while the inspected C# assertions establish four-byte alignment for relevant records. A universal eight-byte allocator guarantee was not independently proved. SQLite must use checked alignment/endian handling and an owned-buffer fallback rather than copying these unchecked cast assumptions.

Sources: [managed reader and pin lifetime](<https://github.com/microsoft/garnet/blob/5cc8f3f50aa57ed694dfda928e4467593e073656/libs/server/Storage/Functions/VectorStore/VectorSessionFunctions.cs#L35-L119>), [native callback types/helpers](<https://github.com/microsoft/DiskANN/blob/eaeb12578547a84c160909ec639d1a8564cc7216/diskann-garnet/src/garnet.rs>), [FFI contract](<https://github.com/microsoft/DiskANN/blob/eaeb12578547a84c160909ec639d1a8564cc7216/diskann-garnet/docs/ffi-design.rs>).

## 4. Pruning retains owned payloads across fills

Garnet's prune accessor owns `workingset::Map<u32, Box<[u8]>>`. It calls `prepare()` before each fill, reuses existing entries, batch-reads and copies only misses, and returns a borrowing view for pruning. The generic algorithm reuses the same accessor through a logical operation's backedge prunes.

This is reuse of **actual vector payloads**, not only allocation capacity. It is operation/accessor-local, not an index-wide persistent graph cache.

The eviction mechanism is generation tagging, not engine-page pinning and not LRU:

- Increment the generation for a new fill.
- Mark requested existing entries as belonging to the new generation.
- Evict unrequested old entries as needed.
- Fetch only missing payloads.

The configured capacity is a retention target, **not a hard entry/byte maximum**: a single requested fill can exceed it. Hash buckets also retain capacity. Vectorlite must continue charging payloads, container overhead and active pinned views against the real workspace budget and must not silently truncate a fill to fit.

Sources: [Garnet prune working set](<https://github.com/microsoft/DiskANN/blob/eaeb12578547a84c160909ec639d1a8564cc7216/diskann-garnet/src/provider.rs#L2596-L2749>), [generation/eviction implementation](<https://github.com/microsoft/DiskANN/blob/eaeb12578547a84c160909ec639d1a8564cc7216/diskann/src/graph/workingset/map.rs#L155-L275>).

### Concrete local gap

Our [PruneAccessor fill](<../../vectorlite/src/diskann_core.rs#L995-L1019>) replaces its entire vector HashMap on every fill. The [prune-accessor constructor](<../../vectorlite/src/diskann_core.rs#L1055-L1064>) ignores the upstream capacity hint. Working-set reuse and owned-batch overlay support already exist in our [vendored DiskANN 0.60 map](<../../third_party/diskann-0.60.0/vendor/src/graph/workingset/map.rs#L43-L93>).

**Coherence caveat:** Garnet retains vector copies across neighbor RMWs, but our SQL fault triggers can modify vectors during any DML. Our nominally immutable vector rows are immutable only under the supported regular API. Safe initial reuse must be restricted to a proved write-free interval, with invalidation/revalidation around writes, unless a separately verified coherence mechanism exists. Backedge prunes are usually separated by writes, so blindly clearing after each write can eliminate much of the advertised cross-prune reuse. This is a conditional opportunity, not a promised speedup.

## 5. Mutation, concurrency and persistence

### Record-level mutation

Write callbacks copy an entire value using Upsert. Neighbor replacement uses full-value RMW; append modifies the stored neighbor buffer, deduplicates IDs and updates the trailing count. The start-point cache is updated inside the RMW callback for coherence.

Tsavorite can mutate a record in its mutable in-memory log region. Read-only/read-cache/disk records require copying or appending a new mutable version; pending completion rechecks for newer versions. Fixed-sized values therefore help but do not mean every update is copy-free.

The generic graph algorithm still reads adjacency before deciding append versus prune. Garnet RMW avoids a second independent Rust-side adjacency read; it does not eliminate the underlying RMW lookup or the generic first read.

Sources: [managed write/RMW callbacks](<https://github.com/microsoft/garnet/blob/5cc8f3f50aa57ed694dfda928e4467593e073656/libs/server/Resp/Vector/VectorManager.Callbacks.cs#L361-L447>), [in-place/copy update functions](<https://github.com/microsoft/garnet/blob/5cc8f3f50aa57ed694dfda928e4467593e073656/libs/server/Storage/Functions/VectorStore/VectorSessionFunctions.cs#L244-L428>), [Tsavorite RMW paths](<https://github.com/microsoft/garnet/blob/5cc8f3f50aa57ed694dfda928e4467593e073656/libs/storage/Tsavorite/cs/src/core/Index/Tsavorite/Implementation/InternalRMW.cs#L67-L247>).

### Threading and memory protection

A thread-static active session routes callbacks through the calling C# thread. Native DiskANN must remain on that thread within an invocation, while independent Garnet sessions/commands may invoke the index concurrently. Existing insertions use a shared per-set lifecycle lock plus an exclusive hashed same-element sublock; individual KV operations have record/hash protection.

Epoch protection keeps borrowed storage memory alive and coordinates versions/reclamation. It is not a SQL transaction or a graph-wide snapshot. Similarly, `CompletePending(wait: true)` means complete the pending storage operation, not commit it durably.

Sources: [callback thread binding](<https://github.com/microsoft/garnet/blob/5cc8f3f50aa57ed694dfda928e4467593e073656/libs/server/Resp/Vector/VectorManager.Callbacks.cs#L235-L250>), [lifecycle/element locking](<https://github.com/microsoft/garnet/blob/5cc8f3f50aa57ed694dfda928e4467593e073656/libs/server/Resp/Vector/VectorManager.Locking.cs#L98-L129>).

### Persistence is not SQLite-style whole-operation rollback

Standard checkpoints persist actual stored terms and the small index stub. Native handles are invalid across recovery/eviction and are lazily rebuilt over those records, not by reconstructing the graph through all VADDs. Context-reservation recovery has reconciliation/cleanup paths for partially published multi-record state.

Synthetic AOF entries are logged after successful vector commands. Consecutive operations can replay in parallel; replay may produce a different graph topology. A log-injection failure after a native mutation does not generally undo every preceding graph write. Native guards compensate particular element terms, but they are not an enclosing graph/vector/metadata statement journal.

The provider's own design explicitly says **index operations are not atomic, although individual term accesses are**. Missing vectors during concurrent search/deletion can be transient and ignored. Concurrent updates may cause vector/attribute or candidate/result skew. Imports use an availability/finalization lifecycle; failed terms or finalization can leave changed records and an unavailable/failed index rather than restoring a prior graph.

Do not adopt these semantics for vectorlite. Preserve SQLite's existing journal carrier, original errors, snapshot anchors and all-or-nothing INSERT behavior.

Sources: [native concurrency/failure model](<https://github.com/microsoft/DiskANN/blob/eaeb12578547a84c160909ec639d1a8564cc7216/diskann-garnet/docs/data-design.md#concurrency-issues>), [synthetic replication](<https://github.com/microsoft/garnet/blob/5cc8f3f50aa57ed694dfda928e4467593e073656/libs/server/Resp/Vector/VectorManager.Replication.cs#L86-L108>), [command mutation/logging order](<https://github.com/microsoft/garnet/blob/5cc8f3f50aa57ed694dfda928e4467593e073656/libs/server/Storage/Session/MainStore/VectorStoreOps.cs#L405-L445>), [import failure/finalization](<https://github.com/microsoft/garnet/blob/5cc8f3f50aa57ed694dfda928e4467593e073656/libs/server/Resp/Vector/VectorManager.Import.cs#L86-L184>), [official vector-set documentation](<https://microsoft.github.io/garnet/docs/commands/vector-sets>).

## 6. Borrowable ideas and tradeoffs

All expected performance effects below are hypotheses. Source inspection alone does not establish vectorlite's CPU bottleneck or an achievable speedup.

| Idea | Why it may help | Required SQLite safeguards / costs |
|---|---|---|
| **Bounded multi-ID reads + callback-local distance** | Fewer SQL crossings, temporary vector allocations and copies; Garnet batches unseen vectors per adjacency list. | Same connection/thread; preserve requested IDs/order and every referenced-node/payload check; detect missing rows instead of silently skipping; consume BLOBs before row invalidation; checked alignment/endian fallback. |
| **Owned prune working-set reuse** | Avoid re-fetch/decode/allocation of vectors needed by several prunes; local fill currently clears everything. | Hard byte accounting and eviction of unneeded/unpinned entries; only valid reuse intervals; SQL trigger mutation visibility; no persistent semantic cache or leaked leases. Cross-write gains need an independently proved coherence rule. |
| **Owned-batch overlay seed** | Newly inserted canonical batch vectors already exist in bounded owned buffers; avoid repeated storage reads when valid. | This is generic upstream overlay functionality, not something the inspected Garnet strategy currently implements. Our seed is currently empty. Validate stored-data coherence and invalidate/revalidate after fault-trigger mutations; never retain caller pointers or alter intra-batch candidates/order. |
| **Split hot graph records from large vector payloads** | Neighbor updates no longer operate on vector-containing SQL records; smaller topology rows can improve page locality and reduce copying/journaling. | Requires versioned shadow-layout/migration design and atomic publication; adds lookups/joins; keep all current identity and payload checks. Do not assert SQLite rewrites every overflow page without measuring. |
| **Fixed adjacency capacity and a logical count** | Stable record/buffer size can reduce resize/allocation churn and support efficient append. | Padding may increase page/WAL space; keep u64 IDs and existing physical slack unless an explicit checked codec redesign is approved. Fixed BLOB size is not a journaling exemption. |
| **A better append/RMW interface and pooled buffers** | Our append re-reads adjacency already retrieved by the generic algorithm; reusable buffers avoid repeated capacity loss. | Preserve exact append/prune decisions, edge order, validation and same-connection write visibility. Never mutate SQLite column memory. Incremental BLOB I/O requires explicit rollback tests inside the current carrier boundary. |

Local reference points:

- [Search expansion and per-ID distance reads](<../../vectorlite/src/diskann_core.rs#L753-L801>).
- [Prune fill destroys its map](<../../vectorlite/src/diskann_core.rs#L995-L1019>).
- [Current empty batch seed](<../../vectorlite/src/diskann_core.rs#L1083-L1126>).
- [Current append adjacency re-read](<../../vectorlite/src/diskann_core.rs#L960-L985>).
- [SQLite-backed store and row decoding](<../../vectorlite/src/diskann_store.rs>).
- [Generic working-set and batch-overlay support already vendored](<../../third_party/diskann-0.60.0/vendor/src/graph/workingset/map.rs>).

Garnet's engine additionally treats record roles differently for read promotion: neighbors, quantized vectors, maps and metadata can be promoted; full vectors and attributes are not promoted by that path. Do not copy this policy blindly: Garnet's quantized/rerank workload differs from our unquantized float32 build. An owned prune working set is also different from an engine read cache: [role-specific policy](<https://github.com/microsoft/garnet/blob/5cc8f3f50aa57ed694dfda928e4467593e073656/libs/server/Resp/Vector/VectorManager.Callbacks.cs#L327-L357>).

## 7. Ideas and assumptions not to copy

- Reusable u32 internal IDs or a silently imposed 32-bit lifetime limit; keep vectorlite's current public int64 and monotonic non-reuse policy.
- Quantization, reduced degree, changed candidate policy or fewer searches as recall-neutral changes.
- Ignoring missing vectors, failed neighbor reads or unreadable external-ID mappings. Vectorlite deliberately fails closed rather than treating corruption/I/O as absence.
- Thread-pool calls into SQLite or unjustified Send/Sync declarations.
- Borrowed pointers escaping the storage callback or mutable access to SQLite-owned column bytes.
- Using scratch bytes preceding a native length-prefixed key with arbitrary SQLite buffers.
- Calling an entry-capacity hint a hard memory limit.
- Treating epoch protection, a per-key lock, import FINISH, checkpoints or synthetic AOF as graph-wide SQL rollback/snapshot guarantees.
- Comparing Garnet's published hardware/quantization/settings directly with our local 128D float32 results.

## 8. Recommended next experiments, not implemented

Start with access shape and working-set lifecycle while keeping the persisted SQLite format unchanged:

1. Profile release-build wall time and allocations to separate SQL/record access, decoding/copies, pruning/distance computation and context-check costs. Keep instrumentation separate from the reported timing run.
2. Test bounded multi-ID access and row-local distance computation with unchanged float32 math and graph iteration order.
3. Test prune allocation/payload reuse inside validated no-write phases. Measure repeated IDs by phase; do not assume that reuse survives our mandatory write invalidation. Any broader reuse needs an explicit coherence proof and fault-trigger tests.
4. Evaluate owned-batch seeding only with checked stored/input coherence.
5. If copying or graph-record update traffic is substantial, prototype split graph/vector rows and fixed adjacency storage as a separate migration-capable design.

Use the existing [matched native A/B runner](<../../benchmark/diskann_identity_benchmark.py>) and [completed identity-cache plan](<../2026-11-09-diskann-identity-cache/plan.md>) as methodology references, not a promise of another 2x gain. Keep degree 32, build list 100, alpha 1.2, float32 L2/cosine, query windows, input hashes and work budgets fixed. Compare exact graph/ordered-query output and recall; rerun transaction/savepoint, corruption, trigger mutation, pointer/thread/runtime cleanup and memory-admission tests. Do not silently truncate a fill, weaken validation or raise budgets to manufacture a result.

The prior checked-identity cache's measured 1.15–1.30x gain is retained in the [local measured report](<../../build/benchmarks/identity-cache-20261108/matched-release/results.md>). That link refers to ignored local artifacts, not data committed with this research note. It does not establish the performance value of any idea above.

## Conclusion

Garnet supplies concrete evidence that the provider boundary can be more efficient without selecting fewer ANN candidates: separate binary terms, batched borrowed reads, payload-aware working sets and fixed-sized record updates. The strongest immediately visible local gap is the lifetime of our pruning buffers and vector reads, not the fact that our payloads are SQLite BLOBs. Preserving vectorlite's stronger transaction, validation and lifetime guarantees is a prerequisite to borrowing those designs.
