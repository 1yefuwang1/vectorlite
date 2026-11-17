//! Matched full-precision benchmark using the unmodified upstream in-memory provider.
//!
//! This deliberately uses `diskann-providers`, the implementation registered by
//! upstream's `graph-index-full-precision-f32` benchmark. `diskann-inmem` is a
//! different, newer provider and has no save/load support at the pinned revision.

use std::{num::NonZeroUsize, sync::Arc, time::Instant};

use anyhow::{ensure, Context, Result};
use diskann::{
    graph::{
        config::{Builder, IntraBatchCandidates, MaxDegree, PruneKind},
        search::Knn,
        search_output_buffer::IdDistance,
        AdjacencyList, DiskANNIndex,
    },
    provider::DefaultContext,
};
use diskann_providers::{
    index::diskann_async,
    model::{
        configuration::IndexConfiguration,
        graph::provider::async_::{
            common::{FullPrecision, NoDeletes},
            inmem::{DefaultProviderParameters, FullPrecisionProvider, SetStartPoints},
        },
    },
    storage::{AsyncIndexMetadata, FileStorageProvider, LoadWith, SaveWith},
};
use diskann_utils::views::rowmajor;
use diskann_vector::distance::Metric;
use serde_json::{json, Value};

use crate::{Config, Dataset};

type NativeIndex = DiskANNIndex<FullPrecisionProvider<f32>>;

const K: usize = 10;
const SEARCH_WINDOWS: [usize; 3] = [10, 50, 100];

#[derive(Debug, PartialEq, Eq)]
struct UnreachableNode {
    id: u32,
    in_degree: usize,
    out_degree: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct ProviderAudit {
    validated_live_vectors: usize,
    validated_nodes: usize,
    reachable_nodes: usize,
    unreachable_ids: Vec<u32>,
    unreachable_degrees: Vec<UnreachableNode>,
    edge_count: usize,
    min_out_degree: usize,
    max_out_degree: usize,
    // Diagnostic only, not a cryptographic substitute for structural validation.
    adjacency_fingerprint: u64,
}

impl ProviderAudit {
    fn as_json(&self) -> Value {
        let unreachable_degrees = self
            .unreachable_degrees
            .iter()
            .map(|node| {
                json!({
                    "id": node.id,
                    "in_degree": node.in_degree,
                    "out_degree": node.out_degree,
                })
            })
            .collect::<Vec<_>>();
        json!({
            "validated_live_vectors": self.validated_live_vectors,
            "validated_nodes": self.validated_nodes,
            "reachable_nodes": self.reachable_nodes,
            "unreachable_nodes": self.unreachable_ids.len(),
            "unreachable_ids": self.unreachable_ids,
            "unreachable_degrees": unreachable_degrees,
            "edge_count": self.edge_count,
            "min_out_degree": self.min_out_degree,
            "max_out_degree": self.max_out_degree,
            "adjacency_fingerprint_fnv1a64": format!("{:016x}", self.adjacency_fingerprint),
            "root_included_in_reachable_nodes": true,
            "reachability_is_quality_diagnostic": true,
        })
    }
}

/// Check stored data independently of graph navigability.
///
/// This is only called after all insert tasks have completed, or after a fresh
/// load, with no writes concurrent with or subsequent to the audit. It checks
/// every logical node: all N live vectors plus the frozen root at N. The pinned
/// provider overallocates one extra unused adjacency slot; that storage capacity
/// is not another live node and is not included in the logical-node audit.
fn audit_provider(
    index: &NativeIndex,
    config: &Config,
    data: &Dataset,
    physical_degree: usize,
) -> Result<ProviderAudit> {
    let total = config
        .count
        .checked_add(1)
        .context("native node count overflow")?;
    let provider = index.provider();
    ensure!(
        provider.capacity() == config.count,
        "native live capacity mismatch"
    );
    ensure!(
        provider.total_points() == total,
        "native logical node count mismatch"
    );
    ensure!(
        provider.num_start_points() == 1,
        "native frozen point count mismatch"
    );
    ensure!(
        provider.starting_points()? == [config.count as u32],
        "unexpected native root convention"
    );
    let store = &provider.base_vectors;
    ensure!(
        store.total() == total,
        "native stored vector count mismatch"
    );
    ensure!(
        store.dim() == config.dim,
        "native stored vector dimension mismatch"
    );

    for id in 0..total {
        let expected = if id == config.count {
            &data.vectors[..config.dim]
        } else {
            &data.vectors[id * config.dim..(id + 1) * config.dim]
        };
        // SAFETY: store.total() was checked to equal N+1 and id is in 0..N+1.
        // This driver is the only owner issuing mutations; every successful
        // multi_insert has joined all of its set/search/backedge tasks before
        // returning (max_minibatch_par is 1). No insert/delete tasks remain here,
        // no writes follow the audit, and loaded indexes have no writer tasks.
        // Each borrowed slice is consumed before proceeding to the next ID.
        let stored = unsafe { store.get_vector_sync(id) };
        ensure!(
            stored.len() == config.dim,
            "native vector {id} has wrong length"
        );
        for (coordinate, (&actual, &expected)) in stored.iter().zip(expected).enumerate() {
            ensure!(
                actual.is_finite(),
                "native vector {id} contains nonfinite value"
            );
            ensure!(
                actual.to_bits() == expected.to_bits(),
                "native stored vector {id} differs from input at coordinate {coordinate}"
            );
        }
    }

    let neighbors = provider.neighbors();
    let mut list = AdjacencyList::with_capacity(physical_degree);
    let mut incoming = vec![0usize; total];
    let mut outgoing = vec![0usize; total];
    let mut edge_count = 0usize;
    let mut fingerprint = 0xcbf29ce484222325u64;
    let mut hash_word = |word: u64| {
        for byte in word.to_le_bytes() {
            fingerprint = (fingerprint ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    };
    for (id, out_degree) in outgoing.iter_mut().enumerate() {
        neighbors.get_neighbors_sync(id, &mut list)?;
        // With N>=10, no deletes, one initialized root and positive degree,
        // successful insertion always has at least one valid non-self candidate.
        // An empty list here may indicate an incomplete graph phase, not merely
        // a graph that has lost incoming reachability through later pruning.
        ensure!(
            !list.is_empty() && list.len() <= physical_degree,
            "native adjacency {id} has invalid degree {} (expected 1..={physical_degree})",
            list.len()
        );
        *out_degree = list.len();
        edge_count = edge_count
            .checked_add(list.len())
            .context("native edge count overflow")?;
        hash_word(id as u64);
        hash_word(list.len() as u64);
        let mut seen_neighbors = Vec::with_capacity(list.len());
        for neighbor in list.iter().copied() {
            ensure!(
                (neighbor as usize) < total,
                "native adjacency {id} has invalid target {neighbor}"
            );
            ensure!(
                neighbor as usize != id,
                "native adjacency {id} has a self-loop"
            );
            ensure!(
                !seen_neighbors.contains(&neighbor),
                "native adjacency {id} has duplicate target {neighbor}"
            );
            seen_neighbors.push(neighbor);
            incoming[neighbor as usize] += 1;
            hash_word(u64::from(neighbor));
        }
    }

    // Like upstream count_reachable_nodes, traverse directed outgoing edges and
    // count the root itself. Connectivity is not an insertion invariant: a
    // later RobustPrune can discard a node's last incoming reachable edge.
    let mut reached = vec![false; total];
    reached[config.count] = true;
    let mut pending = vec![config.count as u32];
    let mut reachable_nodes = 0usize;
    while let Some(id) = pending.pop() {
        reachable_nodes += 1;
        neighbors.get_neighbors_sync(id as usize, &mut list)?;
        for neighbor in list.iter().copied() {
            // Every target was range-checked above, and there are no mutations
            // between the structural scan and this traversal.
            if !reached[neighbor as usize] {
                reached[neighbor as usize] = true;
                pending.push(neighbor);
            }
        }
    }
    let unreachable_ids = reached
        .iter()
        .enumerate()
        .filter_map(|(id, reached)| (!reached).then_some(id as u32))
        .collect::<Vec<_>>();
    let unreachable_degrees = unreachable_ids
        .iter()
        .map(|&id| UnreachableNode {
            id,
            in_degree: incoming[id as usize],
            out_degree: outgoing[id as usize],
        })
        .collect();
    Ok(ProviderAudit {
        validated_live_vectors: config.count,
        validated_nodes: total,
        reachable_nodes,
        unreachable_ids,
        unreachable_degrees,
        edge_count,
        min_out_degree: outgoing
            .iter()
            .copied()
            .min()
            .context("native graph is empty")?,
        max_out_degree: outgoing
            .iter()
            .copied()
            .max()
            .context("native graph is empty")?,
        adjacency_fingerprint: fingerprint,
    })
}

fn query(
    runtime: &tokio::runtime::Runtime,
    index: &NativeIndex,
    vector: &[f32],
    search_l: usize,
    count: usize,
    ids: &mut [u32],
    distances: &mut [f32],
) -> Result<()> {
    ids.fill(u32::MAX);
    distances.fill(f32::NAN);
    let mut output = IdDistance::new(ids, distances);
    let stats = runtime.block_on(index.search(
        Knn::new_default(search_l)?,
        &FullPrecision,
        &DefaultContext,
        vector,
        &mut output,
    ))?;
    ensure!(
        stats.result_count as usize == K,
        "native query returned {} rows, expected {K}",
        stats.result_count
    );
    for (position, (&id, &distance)) in ids.iter().zip(distances.iter()).enumerate() {
        ensure!(
            (id as usize) < count,
            "native query returned invalid/frozen ID {id}"
        );
        ensure!(
            distance.is_finite(),
            "native query returned nonfinite distance"
        );
        ensure!(
            !ids[..position].contains(&id),
            "native query returned duplicate ID {id}"
        );
        if position != 0 {
            ensure!(
                distances[position - 1] <= distance,
                "native query distances are not ordered"
            );
        }
    }
    Ok(())
}

/// Build, query, save, and reload the standard upstream full-precision L2 index.
///
/// Input generation and file reading happen in the caller, outside these timings.
/// No distance-kernel substitution, final graph prune, insertion shuffle, or
/// automatic worker-count selection is performed here.
pub fn run(config: &Config, data: &Dataset) -> Result<Value> {
    ensure!(
        config.count >= K,
        "native benchmark requires at least {K} vectors"
    );
    ensure!(config.dim != 0, "vector dimension must be nonzero");
    ensure!(config.workers != 0, "worker count must be nonzero");
    ensure!(
        (1..=32).contains(&config.batch_size),
        "matched batch size must be between 1 and 32"
    );
    ensure!(
        config.count < u32::MAX as usize,
        "native provider IDs exceed u32"
    );
    let vector_elements = config
        .count
        .checked_mul(config.dim)
        .context("dataset shape overflow")?;
    ensure!(
        data.vectors.len() == vector_elements,
        "vector data does not match count/dimension"
    );
    ensure!(
        !data.queries.is_empty() && data.queries.len().is_multiple_of(config.dim),
        "query data has an invalid shape"
    );
    ensure!(
        data.vectors
            .iter()
            .chain(data.queries.iter())
            .all(|value| value.is_finite()),
        "input contains a nonfinite value"
    );
    let query_count = data.queries.len() / config.dim;

    let ready_start = Instant::now();
    let graph_config = Builder::new_with(
        config.degree,
        MaxDegree::default_slack(),
        config.build_l,
        PruneKind::TriangleInequality,
        |builder| {
            builder
                .alpha(config.alpha)
                .max_minibatch_par(1)
                .intra_batch_candidates(IntraBatchCandidates::new(8));
        },
    )
    .build()?;
    let physical_degree = graph_config.max_degree_u32().get();
    let runtime = if config.workers == 1 {
        tokio::runtime::Builder::new_current_thread().build()?
    } else {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(config.workers)
            .build()?
    };
    let params =
        DefaultProviderParameters::simple(config.count, config.dim, Metric::L2, physical_degree);
    let index = diskann_async::new_index::<f32, _>(graph_config.clone(), params, NoDeletes)?;
    // SQLite freezes a copy of the first live input at internal ID 0. Upstream
    // reserves ID N for the frozen copy and uses live identity IDs 0..N.
    index
        .provider()
        .set_start_points(std::iter::once(&data.vectors[..config.dim]))?;
    let provider_setup_seconds = ready_start.elapsed().as_secs_f64();

    let insert_start = Instant::now();
    let mut inserted = 0usize;
    runtime.block_on(async {
        // Match SQLite's bootstrap even when the remaining ingestion is batched.
        index
            .insert(
                &FullPrecision,
                &DefaultContext,
                &0u32,
                &data.vectors[..config.dim],
            )
            .await?;
        inserted += 1;
        if config.batch_size == 1 {
            for id in 1..config.count {
                let begin = id * config.dim;
                index
                    .insert(
                        &FullPrecision,
                        &DefaultContext,
                        &(id as u32),
                        &data.vectors[begin..begin + config.dim],
                    )
                    .await?;
                inserted += 1;
            }
        } else {
            for begin in (1..config.count).step_by(config.batch_size) {
                let end = (begin + config.batch_size).min(config.count);
                // Copy only one bounded chunk, as in upstream's MultiInsert
                // benchmark runner and SQLite's owned native-batch chunks.
                let vectors = rowmajor::Owned::try_from_data(
                    data.vectors[begin * config.dim..end * config.dim]
                        .to_vec()
                        .into_boxed_slice(),
                    end - begin,
                    config.dim,
                )
                .map_err(|error| anyhow::anyhow!("invalid native batch matrix: {error}"))?;
                let ids: Arc<[u32]> = (begin..end).map(|id| id as u32).collect();
                index
                    .multi_insert::<_, rowmajor::Owned<f32>>(
                        FullPrecision,
                        &DefaultContext,
                        Arc::new(vectors),
                        ids,
                    )
                    .await?;
                inserted += end - begin;
            }
        }
        Ok::<(), anyhow::Error>(())
    })?;
    let graph_insert_seconds = insert_start.elapsed().as_secs_f64();
    let total_ready_seconds = ready_start.elapsed().as_secs_f64();
    ensure!(
        inserted == config.count,
        "native build did not insert the full dataset"
    );

    // Validate every stored payload and adjacency, not just root reachability.
    // These checks remain outside provider/build/ready and query timings.
    let validation_start = Instant::now();
    let audit = audit_provider(&index, config, data, physical_degree as usize)?;
    let graph_validation_seconds = validation_start.elapsed().as_secs_f64();

    let mut searches = Vec::new();
    let mut expected = Vec::new();
    let mut ids = vec![u32::MAX; K];
    let mut distances = vec![f32::NAN; K];
    for search_l in SEARCH_WINDOWS {
        // One complete warm-up pass per window. It is not part of reported QPS.
        for vector in data.queries.chunks_exact(config.dim) {
            query(
                &runtime,
                &index,
                vector,
                search_l,
                config.count,
                &mut ids,
                &mut distances,
            )?;
        }
        let search_start = Instant::now();
        let mut results = Vec::with_capacity(query_count);
        let mut window_expected = Vec::with_capacity(query_count);
        for (query_index, vector) in data.queries.chunks_exact(config.dim).enumerate() {
            query(
                &runtime,
                &index,
                vector,
                search_l,
                config.count,
                &mut ids,
                &mut distances,
            )?;
            window_expected.push((
                ids.clone(),
                distances
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            ));
            // Construct JSON only after the timed query pass below.
            results.push((query_index, ids.clone(), distances.clone()));
        }
        let elapsed_seconds = search_start.elapsed().as_secs_f64();
        let results = results.into_iter().map(|(query_index, ids, distances)| {
            let distance_bits = distances.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
            json!({"query_index": query_index, "ids": ids, "distances": distances, "distance_bits": distance_bits})
        }).collect::<Vec<_>>();
        searches.push(json!({
            "search_l": search_l,
            "k": K,
            "query_count": query_count,
            "warmup_passes": 1,
            "elapsed_seconds": elapsed_seconds,
            "qps": query_count as f64 / elapsed_seconds,
            "timing_scope": "serial runtime block_on search, output validation and owned result capture; excludes JSON serialization",
            "results": results,
        }));
        expected.push(window_expected);
    }

    std::fs::create_dir_all(&config.output_dir)?;
    let graph_path = config.output_dir.join("native-memory.index");
    let graph_prefix = graph_path
        .to_str()
        .context("native index path is not UTF-8")?;
    let metadata = AsyncIndexMetadata::new(graph_prefix);
    let save_start = Instant::now();
    runtime.block_on(index.save_with(&FileStorageProvider, &metadata))?;
    let save_seconds = save_start.elapsed().as_secs_f64();
    let data_path = metadata.data_path();
    let graph_bytes = std::fs::metadata(&graph_path)?.len();
    let data_bytes = std::fs::metadata(&data_path)?.len();
    // Do not inflate process peak RSS by retaining two complete native indexes
    // during the persistence roundtrip. Expected query outputs are owned above.
    drop(index);

    let load_config = IndexConfiguration::new(
        Metric::L2,
        config.dim,
        config.count,
        NonZeroUsize::new(1).context("invalid frozen point count")?,
        config.workers,
        graph_config,
    )
    .with_pseudo_rng_from_seed(config.seed);
    let load_start = Instant::now();
    let reloaded = runtime.block_on(NativeIndex::load_with(
        &FileStorageProvider,
        &(graph_prefix, load_config),
    ))?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let load_validation_start = Instant::now();
    let reloaded_audit = audit_provider(&reloaded, config, data, physical_degree as usize)?;
    ensure!(
        reloaded_audit == audit,
        "native save/load changed structural audit or adjacency fingerprint: before {audit:?}, after {reloaded_audit:?}"
    );
    for (window, search_l) in SEARCH_WINDOWS.into_iter().enumerate() {
        for (query_index, vector) in data.queries.chunks_exact(config.dim).enumerate() {
            query(
                &runtime,
                &reloaded,
                vector,
                search_l,
                config.count,
                &mut ids,
                &mut distances,
            )?;
            ensure!(
                ids == expected[window][query_index].0,
                "native save/load changed ordered IDs"
            );
            ensure!(
                distances
                    .iter()
                    .map(|value| value.to_bits())
                    .eq(expected[window][query_index].1.iter().copied()),
                "native save/load changed distance bits"
            );
        }
    }
    let load_validation_seconds = load_validation_start.elapsed().as_secs_f64();

    Ok(json!({
        "mode": "memory",
        "provider": "diskann_providers::model::graph::provider::async_::inmem::FullPrecisionProvider<f32>",
        "upstream_revision": "97a828a500848018d8be28c3ea6d5a585b07362f",
        "upstream_version": "0.60.0",
        "upstream_patches": [],
        "metric": "squared_l2",
        "kernel": "unmodified upstream diskann-vector f32 distance; target defaults retained",
        "config": {
            "count": config.count, "dim": config.dim,
            "degree": config.degree, "physical_max_degree": physical_degree,
            "build_l": config.build_l, "alpha": config.alpha, "seed": config.seed,
            "seed_scope": "input generation metadata; no insertion shuffle or graph RNG",
            "workers": config.workers,
            "runtime": if config.workers == 1 { "current_thread" } else { "multi_thread" },
            "batch_size": config.batch_size, "max_minibatch_par": 1,
            "intra_batch_candidates": 8, "backedge_ratio": 1.0,
            "final_prune": false,
            "prefetch_lookahead": null, "prefetch_cache_line_level": null,
            "start_point_strategy": "first_vector",
            "frozen_id": config.count, "live_ids": "0..count",
            "first_live_insert": "single", "frozen_point_count": 1,
        },
        "vectors_inserted": inserted,
        "query_count": query_count,
        "validated_live_vectors": audit.validated_live_vectors,
        "validated_nodes": audit.validated_nodes,
        "reachable_nodes": audit.reachable_nodes,
        "unreachable_nodes": audit.unreachable_ids.len(),
        "unreachable_ids": audit.unreachable_ids,
        "unreachable_degrees": audit.as_json()["unreachable_degrees"],
        "storage_validation": {
            "post_build": audit.as_json(),
            "post_load": reloaded_audit.as_json(),
            "roundtrip_audit_matches": true,
            "scope": "all logical vector payloads bit-match input including frozen root; all logical adjacency lists nonempty, bounded, in-range, no self-loops or duplicates; reachability diagnostic only",
            "adjacency_fingerprint_scope": "noncryptographic FNV-1a diagnostic, checked before/after persistence; no graph repair",
        },
        "timings_seconds": {
            "provider_setup": provider_setup_seconds,
            "graph_insert": graph_insert_seconds,
            "total_ready": total_ready_seconds,
            "graph_validation": graph_validation_seconds,
            "save": save_seconds,
            "load": load_seconds,
            "load_validation": load_validation_seconds,
        },
        "searches": searches,
        "persistence": {
            "graph_path": graph_path, "data_path": data_path,
            "graph_bytes": graph_bytes, "data_bytes": data_bytes,
            "roundtrip_exact": true,
            "timing_scope": "upstream save/load API wall time, no explicit fsync",
        },
    }))
}
