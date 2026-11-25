//! Standalone benchmark driver for the unmodified pinned upstream DiskANN providers.
//! Its generated Cargo manifest is outside vectorlite's production workspace.
#![deny(unsafe_op_in_unsafe_fn)]

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use serde_json::json;

mod memory;
mod ssd;

pub struct Config {
    pub count: usize,
    pub dim: usize,
    pub degree: usize,
    pub build_l: usize,
    pub alpha: f32,
    pub seed: u64,
    pub workers: usize,
    pub batch_size: usize,
    pub vectors: PathBuf,
    pub query_vectors: PathBuf,
    pub output_dir: PathBuf,
}

pub struct Dataset {
    pub vectors: Vec<f32>,
    pub queries: Vec<f32>,
}

fn read_bigann(path: &Path, expected_count: Option<usize>, dim: usize) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(bytes.len() >= 8, "input has no BigANN header");
    let count = u32::from_le_bytes(bytes[..4].try_into()?) as usize;
    let actual_dim = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
    ensure!(
        count > 0 && actual_dim == dim,
        "input dimensions do not match"
    );
    if let Some(expected) = expected_count {
        ensure!(count == expected, "input count does not match");
    }
    let elements = count.checked_mul(dim).context("input shape overflow")?;
    let length = elements
        .checked_mul(4)
        .and_then(|n| n.checked_add(8))
        .context("input byte overflow")?;
    ensure!(bytes.len() == length, "input payload has wrong length");
    let mut values = Vec::with_capacity(elements);
    for chunk in bytes[8..].as_chunks::<4>().0 {
        let value = f32::from_le_bytes(*chunk);
        ensure!(value.is_finite(), "non-finite input");
        values.push(value);
    }
    Ok(values)
}

fn main() -> Result<()> {
    let mut options = BTreeMap::new();
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        ensure!(key.starts_with("--"), "expected --option, got {key}");
        let value = args
            .next()
            .with_context(|| format!("missing value for {key}"))?;
        ensure!(
            options.insert(key.clone(), value).is_none(),
            "duplicate option {key}"
        );
    }
    fn take(options: &mut BTreeMap<String, String>, key: &str) -> Result<String> {
        options
            .remove(key)
            .with_context(|| format!("required option {key}"))
    }
    let mode = take(&mut options, "--mode")?;
    let output = PathBuf::from(take(&mut options, "--output")?);
    let config = Config {
        count: take(&mut options, "--count")?.parse()?,
        dim: take(&mut options, "--dimension")?.parse()?,
        degree: options
            .remove("--degree")
            .unwrap_or_else(|| "32".into())
            .parse()?,
        build_l: options
            .remove("--build-list")
            .unwrap_or_else(|| "100".into())
            .parse()?,
        alpha: options
            .remove("--alpha")
            .unwrap_or_else(|| "1.2".into())
            .parse()?,
        seed: options
            .remove("--seed")
            .unwrap_or_else(|| "42".into())
            .parse()?,
        workers: options
            .remove("--workers")
            .unwrap_or_else(|| "1".into())
            .parse()?,
        batch_size: options
            .remove("--batch-size")
            .unwrap_or_else(|| "8".into())
            .parse()?,
        vectors: PathBuf::from(take(&mut options, "--vectors")?),
        query_vectors: PathBuf::from(take(&mut options, "--queries")?),
        output_dir: PathBuf::from(take(&mut options, "--output-dir")?),
    };
    ensure!(options.is_empty(), "unknown options: {options:?}");
    ensure!(
        matches!(mode.as_str(), "memory" | "ssd"),
        "mode must be memory or ssd"
    );
    ensure!(
        (10..=100_000).contains(&config.count),
        "count must be 10..100000"
    );
    ensure!(
        (1..=3000).contains(&config.dim),
        "dimension must be 1..3000"
    );
    ensure!(
        config.workers > 0 && config.workers <= 64,
        "workers must be 1..64"
    );
    ensure!(
        (1..=32).contains(&config.batch_size),
        "batch-size must be 1..32"
    );
    ensure!(
        config.degree >= 2 && config.build_l >= config.degree,
        "invalid graph settings"
    );
    ensure!(
        config.alpha.is_finite() && config.alpha >= 1.0,
        "invalid alpha"
    );
    let input_start = Instant::now();
    let data = Dataset {
        vectors: read_bigann(&config.vectors, Some(config.count), config.dim)?,
        queries: read_bigann(&config.query_vectors, None, config.dim)?,
    };
    let input_seconds = input_start.elapsed().as_secs_f64();
    ensure!(data.queries.len() / config.dim <= 1000, "too many queries");
    // Refuse existing results and index directories before doing any benchmark work.
    let mut destination = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)?;
    fs::create_dir(&config.output_dir)?;
    let result = match mode.as_str() {
        "memory" => memory::run(&config, &data)?,
        "ssd" => ssd::run(&config, &data)?,
        _ => bail!("unreachable mode"),
    };
    let report = json!({
        "schema_version": 1,
        "status": "complete",
        "mode": mode,
        "count": config.count,
        "dimension": config.dim,
        "query_count": data.queries.len() / config.dim,
        "metric": "l2_squared",
        "upstream_version": "0.60.0",
        "upstream_revision": "97a828a500848018d8be28c3ea6d5a585b07362f",
        "architecture": std::env::consts::ARCH,
        "os": std::env::consts::OS,
        "compiled_neon": cfg!(target_feature = "neon"),
        "compiled_dotprod": cfg!(target_feature = "dotprod"),
        "input_load_seconds_excluded": input_seconds,
        "requested_config": {
            "degree":config.degree,"build_list_size":config.build_l,"alpha":config.alpha,
            "seed":config.seed,"workers":config.workers,"batch_size":config.batch_size,
        },
        "result": result,
    });
    serde_json::to_writer_pretty(&mut destination, &report)?;
    destination.write_all(b"\n")?;
    destination.sync_all()?;
    Ok(())
}
