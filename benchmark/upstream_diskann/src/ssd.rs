//! Full official SSD-format build pipeline; saved-index searches validate quality only.
//! macOS's upstream search reader loads the whole disk file into memory.

use std::{collections::HashSet, fs, time::Instant};

use anyhow::{ensure, Context, Result};
use diskann::{graph::config as graph_config, utils::ONE};
use diskann_disk::{
    build::builder::build::DiskIndexBuilder,
    data_model::{AdHoc, CachingStrategy},
    disk_index_build_parameter::{MemoryBudget, NumPQChunks, DISK_SECTOR_LEN},
    search::{
        provider::{
            disk_provider::DiskIndexSearcher,
            disk_vertex_provider_factory::DiskVertexProviderFactory,
        },
        search_mode::SearchMode,
    },
    storage::{disk_index_reader::DiskIndexReader, DiskIndexWriter},
    DiskIndexBuildParameters, QuantizationType,
};
use diskann_providers::{
    model::IndexConfiguration,
    storage::{
        get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file, FileStorageProvider,
    },
};
use diskann_vector::distance::Metric;
use serde_json::{json, Value};

use crate::{Config, Dataset};

pub fn run(config: &Config, data: &Dataset) -> Result<Value> {
    ensure!(
        config.count >= 256,
        "SSD fixture requires at least 256 diverse training rows"
    );
    ensure!(
        data.vectors.len() == config.count * config.dim,
        "base shape mismatch"
    );
    let input = config
        .vectors
        .to_str()
        .context("non-UTF8 input path")?
        .to_owned();
    let prefix = config
        .output_dir
        .join("ssd")
        .to_str()
        .context("non-UTF8 output path")?
        .to_owned();
    // These are explicit benchmark parameters, not a purported universal default.
    let pq_chunks = (config.dim / 4).clamp(1, 96);
    let provider = FileStorageProvider;
    let all_start = Instant::now();
    let cfg = graph_config::Builder::new_with(
        config.degree,
        graph_config::MaxDegree::default_slack(),
        config.build_l,
        Metric::L2.into(),
        |builder| {
            builder.alpha(config.alpha);
            builder.saturate_after_prune(true);
        },
    )
    .build()?;
    let physical_degree = cfg.max_degree_u32().get();
    let index_cfg = IndexConfiguration::new(
        Metric::L2,
        config.dim,
        config.count,
        ONE,
        config.workers,
        cfg,
    )
    .with_pseudo_rng_from_seed(config.seed);
    let params = DiskIndexBuildParameters::new(
        MemoryBudget::try_from_gb(2.0)?,
        QuantizationType::FP,
        NumPQChunks::new_with(pq_chunks, config.dim)?,
    );
    let writer = DiskIndexWriter::new(input, prefix.clone(), None, DISK_SECTOR_LEN)?;
    let constructor_start = Instant::now();
    let mut builder = DiskIndexBuilder::<AdHoc<f32>, _>::new(&provider, params, index_cfg, writer)?;
    let constructor_seconds = constructor_start.elapsed().as_secs_f64();
    let pipeline_start = Instant::now();
    builder.build()?;
    let pipeline_seconds = pipeline_start.elapsed().as_secs_f64();
    let all_seconds = all_start.elapsed().as_secs_f64();
    drop(builder);

    let mut files = Vec::new();
    let mut total_bytes = 0u64;
    for path in [
        get_pq_pivot_file(&prefix),
        get_compressed_pq_file(&prefix),
        get_disk_index_file(&prefix),
    ] {
        let bytes = fs::metadata(&path)
            .with_context(|| format!("missing SSD output {path}"))?
            .len();
        ensure!(bytes > 0, "empty SSD output {path}");
        total_bytes += bytes;
        files.push(json!({"path":path,"bytes":bytes}));
    }
    let load_start = Instant::now();
    let reader = DiskIndexReader::new(
        get_pq_pivot_file(&prefix),
        get_compressed_pq_file(&prefix),
        &provider,
    )?;
    ensure!(
        reader.get_num_points() == config.count,
        "persisted count mismatch"
    );
    let factory = DiskVertexProviderFactory::from_disk_index_path(
        get_disk_index_file(&prefix),
        CachingStrategy::None,
    )?;
    let searcher = DiskIndexSearcher::<AdHoc<f32>, _>::new(
        config.workers,
        usize::MAX,
        &reader,
        factory,
        Metric::L2,
        None,
    )?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let validation_start = Instant::now();
    let mut searches = Vec::new();
    for search_l in [10u32, 50, 100] {
        let mut results = Vec::new();
        for (query_index, query) in data.queries.chunks_exact(config.dim).enumerate() {
            let result = searcher.search(query, 10, search_l, Some(4), SearchMode::graph())?;
            let count = result.stats.result_count as usize;
            ensure!(
                count == 10 && count <= result.results.len(),
                "query {query_index} L{search_l} returned {count} rows"
            );
            let mut ids = Vec::new();
            let mut distances = Vec::new();
            let mut distance_bits = Vec::new();
            let mut seen = HashSet::new();
            for item in result.results.iter().take(count) {
                ensure!(
                    (item.vertex_id as usize) < config.count,
                    "out-of-range SSD result"
                );
                ensure!(item.distance.is_finite(), "nonfinite SSD result");
                ensure!(seen.insert(item.vertex_id), "duplicate SSD result");
                ids.push(item.vertex_id);
                distances.push(item.distance);
                distance_bits.push(item.distance.to_bits());
            }
            ensure!(
                distances.windows(2).all(|pair| pair[0] <= pair[1]),
                "unordered SSD distances"
            );
            results.push(json!({"query_index":query_index,"ids":ids,"distances":distances,"distance_bits":distance_bits}));
        }
        searches.push(json!({"search_l":search_l,"k":10,"beam_width":4,"results":results}));
    }
    let validation_seconds = validation_start.elapsed().as_secs_f64();
    Ok(json!({
        "provider":"official_diskann_ssd_format_builder",
        "build_quantization":"FP",
        "kernel":"unmodified upstream diskann-vector float32",
        "timings_seconds":{
            "constructor":constructor_seconds,"build_pipeline":pipeline_seconds,"total_ready":all_seconds,
            "load":load_seconds,"query_validation":validation_seconds,
        },
        "separate_stage_seconds":null,
        "stages_included":["search_PQ_train_256_centers_12_Lloyd_reps","search_PQ_compress_all_vectors",
            "medoid_and_sequential_single_insert_graph","final_prune","intermediate_graph_save",
            "sectorized_disk_layout_write","intermediate_cleanup"],
        "config":{
            "degree":config.degree,"physical_max_degree":physical_degree,"build_l":config.build_l,
            "alpha":config.alpha,"seed":config.seed,"workers":config.workers,
            "search_pq_chunks":pq_chunks,"search_pq_budget_selection":"explicit_dimension_div4_capped96_not_upstream_default",
            "pq_training_sample_probability":"min(1,50000/N)",
            "pq_training_rows":if config.count<=50000 {Some(config.count)} else {None},
            "build_ram_budget_gib":2.0,"requested_batch_size_not_used_by_ssd":config.batch_size,
            "actual_insert_mode":"single_insert_per_vector_no_multi_insert",
            "start_point":"sampled_medoid_frozen_root","saturate_after_prune":true,"final_prune":true,
        },
        "durability":"buffer_flush_and_close_no_explicit_fsync",
        "files":files,"total_output_bytes":total_bytes,
        "search_validation_only":true,
        "search_reader":if cfg!(target_os="macos") {
            "official_full_file_into_Vec_fallback_not_disk_resident_search"
        } else {"official_platform_aligned_reader"},
        "searches":searches,
    }))
}
