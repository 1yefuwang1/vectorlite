# Pinned crate provenance and local patch

## Baseline

- Package: `diskann` 0.60.0, exact cached crates.io source copied from
  `.cache/cargo-diskann/registry/src/index.crates.io-1949cf8c6b5b557f/diskann-0.60.0`.
- Upstream repository: https://github.com/microsoft/DiskANN
- Package VCS baseline: `97a828a500848018d8be28c3ea6d5a585b07362f`, package directory `diskann`.
- `UPSTREAM_SHA256.json` records SHA-256 for every baseline file before any
  local patch. It is an audit record, not a regenerated hash of patched files.
- Registry cache was not modified. Existing upstream copyright/license headers
  are retained. The release LICENSE and NOTICE remain in the parent directory.

## Local source patch

Only `src/graph/index.rs` differs from the source baseline:

1. `set_elements` keeps the first observed error and joins every remaining
   element task before returning. Returned insertion guards are dropped on
   failure rather than completed.
2. Candidate generation retains the main-task failure, drains all spawned
   candidate tasks, includes task/join failures, and returns before graph
   assignment on any failure.
3. Bootstrap similarly drains all task handles and propagates the first
   main/task/join failure rather than accepting partial replacement edges.
4. Backedge updates join every handle, propagate the first task/join error,
   and do not complete insertion guards unless the full batch succeeds.
5. Batch error documentation describes these strict semantics. Error handling
   does not depend on the `tracing` feature or logging macros.

The spawning algorithm, public batch API, upstream dependency manifest, and
license headers are unchanged. This is error propagation and task-lifetime
repair, not a serial/unspawned substitute or provider rollback implementation.
The provider owner remains responsible for statement-wide rollback of partial
writes on any error.

The adapter executes the retained spawning algorithm on a fresh, private Tokio
current-thread runtime inside its callback-thread store scope. It fixes batch
parallelism at one and admits a batch of at most 32 vectors. Resource admission
and runtime/context rejection are adapter responsibilities, not new upstream
contracts. Re-audit scheduler/container estimates and these joins when upgrading
this pinned source; an unexpected pure future suspension is not a timeout or
background-shutdown escape hatch.
