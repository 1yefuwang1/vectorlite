# Plans

DiskANN-related bundles are in the root [plan directory](<../../plan/README.md>).
This directory retains the other historical plans.

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
