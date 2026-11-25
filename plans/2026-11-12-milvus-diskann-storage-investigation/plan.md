# Milvus DiskANN storage investigation

Supporting document: [source-cited storage research](<research.md>).

Date: 2026-11-12
Status: Complete. Pinned-source research, native-layer review and citation/link verification are recorded in the report.

## Request

Investigate how Milvus stores and accesses its DiskANN index, especially whether
its binary layout offers useful alternatives to vectorlite's SQLite node rows and
binary float32 vector BLOBs. This is read-only research, not implementation or a
Milvus performance benchmark.

## Source scope

- Milvus latest published release observed during investigation: **v3.0.6**, pinned
  to commit `63100f8f5b4b675a12fe78fc9673b005ee318731`.
- Follow that release's actual Knowhere dependency: short pin `b045acd9`, resolved
  to `b045acd9f3b2f3cbe8a6aca9b3e0246f4291b747`.
- Follow Knowhere's actual DiskANN fork/revision; do not substitute the Microsoft
  Rust 0.60.0 implementation used in the earlier [build comparison](<../2026-11-12-upstream-diskann-build-comparison/results.md>).
- Treat versionless documentation and search summaries as context. Source findings
  must cite immutable revision URLs, distinguishing default and optional paths.

## Questions

1. How are raw vectors staged for build, and where does graph construction occur?
2. What binary artifacts are produced: graph/vector sectors, navigation codes,
   codebooks, metadata, optional refinement or cache files?
3. How do durable object storage, local build directories and QueryNode working
   disks relate? Are artifacts split for transport and reconstructed locally?
4. Which data are in RAM and which are read from local SSD during search? What
   batching, direct I/O, cache or prefetch behavior actually appears in the pinned code?
5. How do immutable sealed segments, new inserts, deletion masks and compaction
   differ from our SQLite-contained transactional INSERT/UPDATE/DELETE model?
6. Which layout ideas could preserve float32 precision inside SQLite, and which
   require quantization, immutable builds or a different durability contract?

## Delivery

- Write a co-located source-cited research report with a concrete file/layout and
  build→upload→load→search explanation.
- Keep downloaded source archives/provenance in ignored local cache; do not change
  Milvus, Knowhere, DiskANN or vectorlite production code.
- Record uncertainties and optional modes; no unmeasured speedup or quality claim.
- Check local report links and source revision/path references.
- Do not commit or push unless requested.

## Progress

- [x] Pin Milvus release and its Knowhere dependency.
- [x] Pin the transitive DiskANN implementation and inspect its layout.
- [x] Trace Milvus's storage and segment lifecycle.
- [x] Assess transferable SQLite ideas and their limitations.
- [x] Complete the cited report and link validation.
