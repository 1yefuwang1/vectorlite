# Plans

Keep each plan and its supporting documents together:

```text
docs/plans/<YYYY-MM-DD>-<plan-slug>/
├── plan.md
├── design.md                  # when there is a separate design
├── research.md                # when there is supporting research
├── results.md                 # when there is an implementation/benchmark report
└── <other-supporting-docs>.md
```

Use the plan's date for the directory. Supporting documents retain their own dates
in their contents; they do not need separate date-prefixed filenames. Link related
documents from the plan, and use relative links between bundles when one plan
builds on another. Keep source code in its existing source directories and large
raw benchmark/validation artifacts in the ignored build directory, not in plan bundles.

These documents include historical designs and measurements. Their recorded source
paths, toolchain versions, validation counts, and benchmark results are not current
build instructions or evidence of a new run. Use the
[current contributor guide](<../../vectorlite/README.md>) for building and testing.

| Plan | Supporting documents |
|---|---|
| [2026-06-14 — Explicit index persistence](<2026-06-14-explicit-index-persistence/plan.md>) | [Design](<2026-06-14-explicit-index-persistence/design.md>) |
| [2026-06-14 — Index registry survives schema reparse](<2026-06-14-index-registry-survive-reparse/plan.md>) | [Design](<2026-06-14-index-registry-survive-reparse/design.md>) |
| [2026-06-22 — Python integration regression suite](<2026-06-22-python-integration-regression-suite/plan.md>) | [Design](<2026-06-22-python-integration-regression-suite/design.md>) |
| [2026-09-13 — Rust review fixes](<2026-09-13-rust-review-fixes/plan.md>) | — |
| [2026-11-09 — DiskANN identity cache](<2026-11-09-diskann-identity-cache/plan.md>) | Completion evidence is recorded in the plan |
| [2026-11-10 — Garnet-inspired provider optimizations](<2026-11-10-garnet-provider-optimizations/plan.md>) | [Research](<2026-11-10-garnet-provider-optimizations/research.md>), [results/post-mortem](<2026-11-10-garnet-provider-optimizations/results.md>) |
| [2026-11-11 — SQLite-native DiskANN optimizations](<2026-11-11-sqlite-native-diskann-optimizations/plan.md>) | [Results](<2026-11-11-sqlite-native-diskann-optimizations/results.md>), [distance-kernel comparison](<2026-11-11-sqlite-native-diskann-optimizations/distance-kernel-comparison.md>) |
| [2026-11-12 — Upstream DiskANN build comparison](<2026-11-12-upstream-diskann-build-comparison/plan.md>) | [Research](<2026-11-12-upstream-diskann-build-comparison/research.md>), [results](<2026-11-12-upstream-diskann-build-comparison/results.md>) |
