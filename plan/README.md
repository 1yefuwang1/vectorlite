# DiskANN plans

Each dated bundle keeps its main plan and supporting research/results together:

```text
plan/<YYYY-MM-DD>-<plan-slug>/
├── plan.md
├── research.md                 # when present
├── results.md                  # when present
└── <other-supporting-docs>.md
```

Keep source code in its existing source directories and large raw benchmark data,
binaries, downloads and validation logs in ignored local artifact directories.
Supporting documents retain their original dates and historical conclusions.

Use the [current contributor guide](<../vectorlite/README.md>) for current build
instructions; these dated documents are not evidence of a new verification run.
Other historical plans remain in [docs/plans](<../docs/plans/README.md>).

| Plan | Supporting documents |
|---|---|
| [2026-11-09 — DiskANN identity cache](<2026-11-09-diskann-identity-cache/plan.md>) | Completion evidence is recorded in the plan |
| [2026-11-10 — Garnet-inspired provider optimizations](<2026-11-10-garnet-provider-optimizations/plan.md>) | [Research](<2026-11-10-garnet-provider-optimizations/research.md>), [results/post-mortem](<2026-11-10-garnet-provider-optimizations/results.md>) |
| [2026-11-11 — SQLite-native DiskANN optimizations](<2026-11-11-sqlite-native-diskann-optimizations/plan.md>) | [Results](<2026-11-11-sqlite-native-diskann-optimizations/results.md>), [distance-kernel comparison](<2026-11-11-sqlite-native-diskann-optimizations/distance-kernel-comparison.md>) |
| [2026-11-12 — Upstream DiskANN build comparison](<2026-11-12-upstream-diskann-build-comparison/plan.md>) | [Research](<2026-11-12-upstream-diskann-build-comparison/research.md>), [results](<2026-11-12-upstream-diskann-build-comparison/results.md>) |
| [2026-11-12 — Milvus DiskANN storage investigation](<2026-11-12-milvus-diskann-storage-investigation/plan.md>) | [Research](<2026-11-12-milvus-diskann-storage-investigation/research.md>) |
