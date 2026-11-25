//! Owned-RAM, full-precision native DiskANN OneShot construction.
//!
//! This module has no database, callback, registry, or filesystem dependency. The SQL
//! owner must finish its readers and qualify its connection *before* calling `build`.
//! Only checked owned input enters the coordinator; only immutable owned topology
//! leaves it. Durable identities/fencing and retention are the caller's responsibility.
//!
//! The orchestration uses public, unmodified DiskANN 0.60.0 APIs. Scalar insertion
//! and final pruning do not spawn tasks in that version. An independent, joined
//! coordinator owns a private current-thread Tokio runtime, so an ambient host runtime
//! is never entered or nested. Native allocations/panics may still abort the process;
//! memory admission is an estimate, not an allocator or process-RSS guarantee.
#![deny(unsafe_op_in_unsafe_fn)]

use std::{
    collections::HashSet,
    fmt,
    mem::size_of,
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, RwLock,
    },
    time::{Duration, Instant},
};

use diskann::{
    graph::{
        config::{Builder, MaxDegree, PruneKind},
        glue::{InsertStrategy, PruneStrategy},
        AdjacencyList, DiskANNIndex,
    },
    provider::DefaultContext,
};
use diskann_providers::{
    model::graph::provider::async_::{
        common::{FullPrecision, NoDeletes, NoStore},
        inmem::{
            CreateFullPrecision, DefaultProvider, DefaultProviderParameters, FullPrecisionProvider,
            SetStartPoints,
        },
    },
    utils::{
        create_rnd_from_optional_seed, find_medoid_with_sampling, MAX_MEDOID_SAMPLE_SIZE,
        READ_WRITE_BLOCK_SIZE,
    },
};
use diskann_vector::distance::Metric;
use sha2::{Digest, Sha256};

#[path = "diskann_build/memory_reader.rs"]
mod memory_reader;
use memory_reader::MemoryStorage;

pub type BuildResult<T> = Result<T, BuildError>;
/// Requested stack for the independent native coordinator; admission includes it.
pub const COORDINATOR_STACK_BYTES: usize = 2 * 1024 * 1024;
type NativeProvider = FullPrecisionProvider<f32>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    InvalidInput(String),
    InvalidConfig(String),
    Limit(String),
    Allocation,
    InputRead(String),
    Native {
        stage: &'static str,
        message: String,
    },
    InvalidTopology(String),
    Cancelled,
    Coordinator(String),
    NativePanic,
    Teardown,
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(s)
            | Self::InvalidConfig(s)
            | Self::Limit(s)
            | Self::InputRead(s)
            | Self::InvalidTopology(s)
            | Self::Coordinator(s) => f.write_str(s),
            Self::Native { stage, message } => write!(f, "DiskANN {stage}: {message}"),
            Self::Allocation => f.write_str("DiskANN owned build allocation failed"),
            Self::Cancelled => f.write_str("DiskANN owned build cancelled"),
            Self::NativePanic => {
                f.write_str("DiskANN native coordinator panicked; no candidate returned")
            }
            Self::Teardown => {
                f.write_str("DiskANN native runtime still had live tasks; no candidate returned")
            }
        }
    }
}
impl std::error::Error for BuildError {}

fn native_error(stage: &'static str, error: impl fmt::Display) -> BuildError {
    BuildError::Native {
        stage,
        message: error.to_string(),
    }
}
fn overflow() -> BuildError {
    BuildError::Limit("DiskANN owned build size/ID arithmetic overflow".into())
}
fn add(a: usize, b: usize) -> BuildResult<usize> {
    a.checked_add(b).ok_or_else(overflow)
}
fn mul(a: usize, b: usize) -> BuildResult<usize> {
    a.checked_mul(b)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(overflow)
}
fn aligned_bytes(bytes: usize) -> BuildResult<usize> {
    mul(add(bytes, 63)? / 64, 64)
}
fn reserved_vec<T>(capacity: usize) -> BuildResult<Vec<T>> {
    mul(capacity, size_of::<T>())?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(capacity)
        .map_err(|_| BuildError::Allocation)?;
    Ok(result)
}

/// There is deliberately no inner-product variant. Cosine input is already encoded
/// by canonical extension policy, including its finite zero-vector representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildMetric {
    L2Squared,
    CosineNormalized,
}
impl BuildMetric {
    fn native(self) -> Metric {
        match self {
            Self::L2Squared => Metric::L2,
            Self::CosineNormalized => Metric::CosineNormalized,
        }
    }
    fn tag(self) -> u8 {
        match self {
            Self::L2Squared => 0,
            Self::CosineNormalized => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BuildConfig {
    pub dimension: usize,
    pub metric: BuildMetric,
    pub max_degree: usize,
    pub construction_l: usize,
    pub alpha: f32,
    pub seed: u64,
    /// Exactly one is supported. Zero/auto and parallel construction are rejected.
    pub workers: usize,
}
impl BuildConfig {
    fn native_config(&self) -> BuildResult<diskann::graph::Config> {
        if self.dimension == 0 || self.dimension > u32::MAX as usize {
            return Err(BuildError::InvalidConfig(
                "invalid DiskANN build dimension".into(),
            ));
        }
        mul(self.dimension, size_of::<f32>())?;
        if !(2..=10_000).contains(&self.max_degree) {
            return Err(BuildError::InvalidConfig(
                "DiskANN degree must be between 2 and 10000".into(),
            ));
        }
        if self.construction_l == 0 || self.construction_l > u32::MAX as usize {
            return Err(BuildError::InvalidConfig(
                "DiskANN build list size must be positive and at most u32::MAX".into(),
            ));
        }
        if !self.alpha.is_finite() || self.alpha < 1.0 {
            return Err(BuildError::InvalidConfig(
                "DiskANN alpha must be finite and at least 1".into(),
            ));
        }
        if self.workers != 1 {
            return Err(BuildError::InvalidConfig(
                "DiskANN owned build requires exactly one native worker".into(),
            ));
        }
        if !cfg!(target_endian = "little") {
            return Err(BuildError::InvalidConfig(
                "native DiskANN build requires a little-endian platform".into(),
            ));
        }
        // Full-precision OneShot uses saturation after pruning. Both approved
        // metrics map to native TriangleInequality, via the public metric mapping.
        Builder::new_with(
            self.max_degree,
            MaxDegree::default_slack(),
            self.construction_l,
            PruneKind::from_metric(self.metric.native()),
            |b| {
                b.alpha(self.alpha);
                b.max_minibatch_par(1);
                b.saturate_after_prune(true);
            },
        )
        .build()
        .map_err(|e| native_error("configuration", e))
    }
    pub fn digest(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"vectorlite/native-oneshot/0.60.0/config/v1");
        hash.update((self.dimension as u64).to_le_bytes());
        hash.update([self.metric.tag()]);
        hash.update((self.max_degree as u64).to_le_bytes());
        hash.update((self.construction_l as u64).to_le_bytes());
        hash.update(self.alpha.to_bits().to_le_bytes());
        hash.update(self.seed.to_le_bytes());
        hash.update((self.workers as u64).to_le_bytes());
        // Fixed OneShot protocol: saturation, native default slack, occlusion
        // cap750, full backedges, one frozen point, native default prefetch.
        hash.update(b"saturate=true/slack=1.3/occlusion=750/backedges=R/frozen=1/prefetch=default");
        hash.finalize().into()
    }
}

/// Budget shared with operation-local SQL buffers and other retained candidates.
/// The caller supplies those coexistence bytes explicitly; this module never reads
/// query/cache limits or treats this admission as a hard memory/RSS cap.
#[derive(Debug, Clone, Copy)]
pub struct BuildLimits {
    pub memory_budget_bytes: usize,
    pub external_reservation_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryAdmission {
    pub input_bytes: usize,
    pub provider_vector_bytes: usize,
    pub provider_graph_bytes: usize,
    pub native_workspace_bytes: usize,
    pub retained_upper_bytes: usize,
    pub native_coordinator_stack_bytes: usize,
    pub runtime_and_allocator_allowance_bytes: usize,
    pub external_reservation_bytes: usize,
    pub peak_estimate_bytes: usize,
}

/// Check N, the frozen node N, and the native exclusive end N+1 before allocation.
/// The provider's neighbor implementation allocates an additional unused row;
/// that N+2 allocation is separately admitted (it is not an extra native node).
fn checked_population(population: usize) -> BuildResult<(u32, u32)> {
    let n = u32::try_from(population).map_err(|_| overflow())?;
    let end = if n == 0 {
        0
    } else {
        n.checked_add(1).ok_or_else(overflow)?
    };
    add(population, 2)?;
    Ok((n, end))
}

/// Admission before materialization, assuming an exact-size owned input Vec.
/// `build` rechecks the actual Vec capacity, so spare capacity cannot bypass it.
pub fn admit(
    population: usize,
    config: &BuildConfig,
    limits: BuildLimits,
) -> BuildResult<MemoryAdmission> {
    let native = config.native_config()?;
    checked_population(population)?;
    let input_bytes = mul(mul(population, config.dimension)?, size_of::<f32>())?;
    admit_capacity(population, config, limits, input_bytes, &native)
}
fn admit_capacity(
    population: usize,
    config: &BuildConfig,
    limits: BuildLimits,
    input_bytes: usize,
    native: &diskann::graph::Config,
) -> BuildResult<MemoryAdmission> {
    if limits.memory_budget_bytes == 0 || limits.memory_budget_bytes > isize::MAX as usize {
        return Err(BuildError::Limit(
            "DiskANN build memory budget must be positive and at most isize::MAX".into(),
        ));
    }
    let (_, end) = checked_population(population)?;
    let nodes = end as usize;
    let vector_bytes = mul(config.dimension, 4)?;
    let retained_upper_bytes = add(
        size_of::<OwnedTopology>(),
        add(
            mul(add(nodes, 1)?, 8)?,
            add(
                mul(mul(nodes, config.max_degree)?, 4)?,
                if nodes == 0 { 0 } else { vector_bytes },
            )?,
        )?,
    )?;
    // 0.60.0 aligned stores pad every row to 64 bytes and add 60 bytes per allocation.
    let (provider_vector_bytes, provider_graph_bytes, native_workspace_bytes, runtime_bytes) =
        if nodes == 0 {
            (0, 0, 0, 0)
        } else {
            let vectors = add(
                add(mul(nodes, aligned_bytes(vector_bytes)?)?, 60)?,
                add(mul(nodes.div_ceil(16), size_of::<Mutex<()>>())?, 1024)?,
            )?;
            let graph_rows = add(nodes, 1)?;
            let physical_degree = native.max_degree().get();
            let graph = add(
                add(
                    mul(
                        graph_rows,
                        aligned_bytes(mul(add(physical_degree, 1)?, 4)?)?,
                    )?,
                    60,
                )?,
                add(mul(graph_rows, add(size_of::<RwLock<()>>(), 64)?)?, 1024)?,
            )?;
            // Scalar traversal can visit every node. Also admit the native initial
            // degree*L reserves, geometric Vec/hash growth, queue/prune/validation
            // buffers and the public medoid helper's 64 MiB dense-reader buffer.
            // The 750 occlusion cap is NOT used as a bound on total visited nodes.
            let initial = mul(mul(physical_degree, add(config.construction_l, 1)?)?, 2)?;
            let visited = mul(initial.max(nodes), 256)?;
            let mut workspace = READ_WRITE_BLOCK_SIZE;
            for bytes in [
                visited,
                mul(add(config.construction_l, 2)?, 64)?,
                mul(physical_degree, 256)?,
                mul(native.max_occlusion_size().get(), 128)?,
                mul(vector_bytes, 6)?,
                mul(nodes, 8)?,
            ] {
                workspace = add(workspace, bytes)?;
            }
            // Explicit coordinator stack plus conservative runtime/container/allocator
            // allowance. Native infallible allocations still are not NOMEM-safe.
            (vectors, graph, workspace, 6 * 1024 * 1024)
        };
    let stack_bytes = if nodes == 0 {
        0
    } else {
        COORDINATOR_STACK_BYTES
    };
    let mut peak = 0;
    for bytes in [
        input_bytes,
        provider_vector_bytes,
        provider_graph_bytes,
        native_workspace_bytes,
        retained_upper_bytes,
        stack_bytes,
        runtime_bytes,
        limits.external_reservation_bytes,
    ] {
        peak = add(peak, bytes)?;
    }
    if peak > isize::MAX as usize || peak > limits.memory_budget_bytes {
        return Err(BuildError::Limit(format!("DiskANN owned build admission requires {peak} bytes; budget is {} bytes (estimate, not RSS cap)", limits.memory_budget_bytes)));
    }
    Ok(MemoryAdmission {
        input_bytes,
        provider_vector_bytes,
        provider_graph_bytes,
        native_workspace_bytes,
        retained_upper_bytes,
        native_coordinator_stack_bytes: stack_bytes,
        runtime_and_allocator_allowance_bytes: runtime_bytes,
        external_reservation_bytes: limits.external_reservation_bytes,
        peak_estimate_bytes: peak,
    })
}

/// Checked *canonical* float32 snapshot, densely ordered as native IDs 0..N-1.
/// This constructor does not normalize again. The caller must have applied the
/// existing canonical cosine encoder exactly once, including its zero policy.
/// Revalidation preserves the existing metric-specific finite/norm restrictions;
/// it deliberately does not invent a new unit-norm tolerance or numeric policy.
#[derive(Debug)]
pub struct CheckedSnapshot {
    population: usize,
    dimension: usize,
    metric: BuildMetric,
    values: Vec<f32>,
    digest: [u8; 32],
}
impl CheckedSnapshot {
    pub fn new(
        population: usize,
        dimension: usize,
        metric: BuildMetric,
        values: Vec<f32>,
    ) -> BuildResult<Self> {
        checked_population(population)?;
        if dimension == 0 || dimension > u32::MAX as usize {
            return Err(BuildError::InvalidInput(
                "invalid DiskANN snapshot dimension".into(),
            ));
        }
        let length = mul(population, dimension)?;
        mul(length, 4)?;
        if values.len() != length {
            return Err(BuildError::InvalidInput(
                "DiskANN snapshot length does not match its declared population/dimension".into(),
            ));
        }
        let mut hash = Sha256::new();
        hash.update(b"vectorlite/native-oneshot/0.60.0/snapshot/v1");
        hash.update((population as u64).to_le_bytes());
        hash.update((dimension as u64).to_le_bytes());
        hash.update([metric.tag()]);
        for vector in values.chunks_exact(dimension) {
            validate_numeric(vector, metric)?;
            for value in vector {
                hash.update(value.to_bits().to_le_bytes());
            }
        }
        Ok(Self {
            population,
            dimension,
            metric,
            values,
            digest: hash.finalize().into(),
        })
    }
    pub fn population(&self) -> usize {
        self.population
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
}
fn validate_numeric(vector: &[f32], metric: BuildMetric) -> BuildResult<()> {
    if vector.iter().any(|v| !v.is_finite()) {
        return Err(BuildError::InvalidInput(
            "DiskANN snapshot contains a non-finite float32 value".into(),
        ));
    }
    let norm: f64 = vector.iter().map(|&v| f64::from(v).powi(2)).sum();
    let maximum = match metric {
        BuildMetric::L2Squared => f64::from(f32::MAX) / 4.0,
        BuildMetric::CosineNormalized => f64::from(f32::MAX),
    };
    if !norm.is_finite() || norm > maximum {
        return Err(BuildError::InvalidInput(match metric {
            BuildMetric::L2Squared => "DiskANN squared L2 requires squared norm <= f32::MAX/4 to avoid distance overflow",
            BuildMetric::CosineNormalized => "DiskANN cosine norm would overflow float32",
        }.into()));
    }
    Ok(())
}

/// Cooperative cancellation only; it cannot preempt an arbitrary native operation.
#[derive(Debug, Clone, Default)]
pub struct Cancellation(Arc<AtomicBool>);
impl Cancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    fn check(&self) -> BuildResult<()> {
        if self.is_cancelled() {
            Err(BuildError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct BuildStats {
    pub native_workers: usize,
    pub native_coordinator_threads: usize,
    pub inserted_vectors: u32,
    pub pruned_nodes: u32,
    pub extracted_nodes: u32,
    pub reachable_nodes: u32,
    pub edge_count: usize,
    pub min_degree: usize,
    pub max_degree: usize,
    pub medoid_read_bytes: u64,
    pub medoid_full_scan_vectors: usize,
    pub medoid_readers_closed: usize,
    pub native_alive_tasks_at_teardown: usize,
    pub coordinator_joined: bool,
    pub retained_bytes: usize,
    pub medoid_time: Duration,
    pub insert_time: Duration,
    pub prune_time: Duration,
    pub extract_time: Duration,
    pub total_time: Duration,
}

/// Immutable CSR in native order, including frozen root N for nonempty inputs.
/// Ordinary vectors stay in the caller's durable snapshot; only the frozen vector
/// is retained here. No provider, runtime, task, input Vec or borrowed pointer escapes.
#[derive(Debug)]
pub struct OwnedTopology {
    config: BuildConfig,
    population: u32,
    root: Option<u32>,
    medoid_ordinal: Option<u32>,
    root_vector: Vec<f32>,
    offsets: Vec<u64>,
    neighbors: Vec<u32>,
    snapshot_digest: [u8; 32],
    config_digest: [u8; 32],
    topology_digest: [u8; 32],
    admission: MemoryAdmission,
    stats: BuildStats,
}
impl OwnedTopology {
    pub fn config(&self) -> &BuildConfig {
        &self.config
    }
    pub fn population(&self) -> u32 {
        self.population
    }
    pub fn node_count(&self) -> usize {
        self.offsets.len() - 1
    }
    pub fn root(&self) -> Option<u32> {
        self.root
    }
    pub fn medoid_ordinal(&self) -> Option<u32> {
        self.medoid_ordinal
    }
    pub fn root_vector(&self) -> &[f32] {
        &self.root_vector
    }
    pub fn offsets(&self) -> &[u64] {
        &self.offsets
    }
    pub fn adjacency(&self) -> &[u32] {
        &self.neighbors
    }
    pub fn neighbors(&self, id: u32) -> Option<&[u32]> {
        let id = id as usize;
        let start = usize::try_from(*self.offsets.get(id)?).ok()?;
        let end = usize::try_from(*self.offsets.get(id.checked_add(1)?)?).ok()?;
        self.neighbors.get(start..end)
    }
    pub fn snapshot_digest(&self) -> [u8; 32] {
        self.snapshot_digest
    }
    pub fn config_digest(&self) -> [u8; 32] {
        self.config_digest
    }
    pub fn topology_digest(&self) -> [u8; 32] {
        self.topology_digest
    }
    pub fn admission(&self) -> MemoryAdmission {
        self.admission
    }
    pub fn stats(&self) -> &BuildStats {
        &self.stats
    }

    /// Full, fallible shape/hash validation for qualification and bounded-import
    /// owners. Production publication must not hide this O(N+E) scan in a short lock.
    pub fn validate(&self) -> BuildResult<()> {
        let (_, end) = checked_population(self.population as usize)?;
        let nodes = end as usize;
        if self.offsets.len() != add(nodes, 1)?
            || self.offsets.first() != Some(&0)
            || self.offsets.last().copied() != Some(self.neighbors.len() as u64)
        {
            return Err(BuildError::InvalidTopology(
                "invalid DiskANN CSR shape".into(),
            ));
        }
        if nodes == 0 {
            if self.root.is_some()
                || self.medoid_ordinal.is_some()
                || !self.root_vector.is_empty()
                || !self.neighbors.is_empty()
            {
                return Err(BuildError::InvalidTopology(
                    "invalid empty DiskANN topology".into(),
                ));
            }
        } else {
            if self.root != Some(self.population)
                || self.medoid_ordinal.is_none_or(|id| id >= self.population)
                || self.root_vector.len() != self.config.dimension
            {
                return Err(BuildError::InvalidTopology(
                    "invalid native DiskANN root/medoid".into(),
                ));
            }
            validate_numeric(&self.root_vector, self.config.metric)?;
        }
        if nodes != 0 {
            let mut unique = HashSet::new();
            unique
                .try_reserve(self.config.max_degree)
                .map_err(|_| BuildError::Allocation)?;
            for id in 0..end {
                let list = self.neighbors(id).ok_or_else(|| {
                    BuildError::InvalidTopology("invalid DiskANN CSR offsets".into())
                })?;
                validate_neighbors(id, end, self.config.max_degree, list, &mut unique)?;
            }
        }
        if self.config_digest != self.config.digest()
            || self.topology_digest != self.calculate_topology_digest()
        {
            return Err(BuildError::InvalidTopology(
                "DiskANN topology/config digest mismatch".into(),
            ));
        }
        Ok(())
    }
    fn calculate_topology_digest(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"vectorlite/native-oneshot/0.60.0/topology/v1");
        hash.update(self.snapshot_digest);
        hash.update(self.config_digest);
        hash.update(self.population.to_le_bytes());
        hash.update([u8::from(self.root.is_some())]);
        if let Some(root) = self.root {
            hash.update(root.to_le_bytes());
        }
        if let Some(medoid) = self.medoid_ordinal {
            hash.update(medoid.to_le_bytes());
        }
        for value in &self.root_vector {
            hash.update(value.to_bits().to_le_bytes());
        }
        for offset in &self.offsets {
            hash.update(offset.to_le_bytes());
        }
        for neighbor in &self.neighbors {
            hash.update(neighbor.to_le_bytes());
        }
        hash.finalize().into()
    }
    fn retained_bytes(&self) -> BuildResult<usize> {
        add(
            size_of::<Self>(),
            add(
                mul(self.root_vector.capacity(), 4)?,
                add(
                    mul(self.offsets.capacity(), 8)?,
                    mul(self.neighbors.capacity(), 4)?,
                )?,
            )?,
        )
    }
}
fn validate_neighbors(
    id: u32,
    end: u32,
    max_degree: usize,
    list: &[u32],
    unique: &mut HashSet<u32>,
) -> BuildResult<()> {
    if list.len() > max_degree {
        return Err(BuildError::InvalidTopology(
            "DiskANN native degree exceeds final pruned degree".into(),
        ));
    }
    unique.clear();
    for &neighbor in list {
        if neighbor >= end || neighbor == id || !unique.insert(neighbor) {
            return Err(BuildError::InvalidTopology(
                "DiskANN native adjacency contains an out-of-range/self/duplicate edge".into(),
            ));
        }
    }
    Ok(())
}
fn reachable(topology: &OwnedTopology) -> BuildResult<u32> {
    let Some(root) = topology.root else {
        return Ok(0);
    };
    let mut seen = reserved_vec::<bool>(topology.node_count())?;
    seen.resize(topology.node_count(), false);
    let mut queue = reserved_vec::<u32>(topology.node_count())?;
    seen[root as usize] = true;
    queue.push(root);
    let mut cursor = 0;
    while cursor < queue.len() {
        for &neighbor in topology
            .neighbors(queue[cursor])
            .ok_or_else(|| BuildError::InvalidTopology("invalid reachability input".into()))?
        {
            if !seen[neighbor as usize] {
                seen[neighbor as usize] = true;
                queue.push(neighbor);
            }
        }
        cursor += 1;
    }
    u32::try_from(queue.len()).map_err(|_| overflow())
}

/// Construct one global native OneShot graph. The call is synchronous but safe to
/// invoke from an ambient Tokio runtime: computation and runtime destruction happen
/// on an independent coordinator, which is always joined before any return.
pub fn build(
    snapshot: CheckedSnapshot,
    config: BuildConfig,
    limits: BuildLimits,
    cancellation: Cancellation,
) -> BuildResult<OwnedTopology> {
    build_with_strategy(snapshot, config, limits, cancellation, FullPrecision)
}
fn build_with_strategy<S>(
    snapshot: CheckedSnapshot,
    config: BuildConfig,
    limits: BuildLimits,
    cancellation: Cancellation,
    strategy: S,
) -> BuildResult<OwnedTopology>
where
    S: for<'a> InsertStrategy<'a, NativeProvider, &'a [f32]> + PruneStrategy<NativeProvider>,
{
    let started = Instant::now();
    let native_config = config.native_config()?;
    if snapshot.dimension != config.dimension || snapshot.metric != config.metric {
        return Err(BuildError::InvalidInput(
            "DiskANN snapshot/configuration mismatch".into(),
        ));
    }
    let actual_input_bytes = mul(snapshot.values.capacity(), 4)?;
    let admission = admit_capacity(
        snapshot.population,
        &config,
        limits,
        actual_input_bytes,
        &native_config,
    )?;
    cancellation.check()?;
    // No runtime is created for empty input. It is an explicit empty generation,
    // not a fabricated native medoid or a call to a constructor with max_points=0.
    if snapshot.population == 0 {
        let mut offsets = reserved_vec(1)?;
        offsets.push(0);
        let mut result = OwnedTopology {
            config_digest: config.digest(),
            config,
            population: 0,
            root: None,
            medoid_ordinal: None,
            root_vector: Vec::new(),
            offsets,
            neighbors: Vec::new(),
            snapshot_digest: snapshot.digest,
            topology_digest: [0; 32],
            admission,
            stats: BuildStats::default(),
        };
        result.topology_digest = result.calculate_topology_digest();
        result.stats.retained_bytes = result.retained_bytes()?;
        result.stats.total_time = started.elapsed();
        result.validate()?;
        return Ok(result);
    }
    let thread = std::thread::Builder::new()
        .name("vectorlite-diskann-build".into())
        .stack_size(COORDINATOR_STACK_BYTES)
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .map_err(|e| {
                    BuildError::Coordinator(format!("DiskANN private runtime creation failed: {e}"))
                })?;
            let mut result = runtime.block_on(compute(
                snapshot,
                config,
                native_config,
                admission,
                cancellation,
                strategy,
            ));
            let live = runtime.metrics().num_alive_tasks();
            // Scalar insert/prune (unmodified 0.60.0) spawn no tasks. Never accept
            // escaped tasks if that pinned contract changes or a test violates it.
            drop(runtime);
            if live != 0 {
                return Err(BuildError::Teardown);
            }
            if let Ok(topology) = &mut result {
                topology.stats.native_alive_tasks_at_teardown = live;
            }
            result
        })
        .map_err(|e| {
            BuildError::Coordinator(format!("DiskANN coordinator creation failed: {e}"))
        })?;
    let mut result = thread.join().map_err(|_| BuildError::NativePanic)??;
    result.stats.coordinator_joined = true;
    result.stats.total_time = started.elapsed();
    Ok(result)
}
async fn compute<S>(
    snapshot: CheckedSnapshot,
    config: BuildConfig,
    native_config: diskann::graph::Config,
    admission: MemoryAdmission,
    cancellation: Cancellation,
    strategy: S,
) -> BuildResult<OwnedTopology>
where
    S: for<'a> InsertStrategy<'a, NativeProvider, &'a [f32]> + PruneStrategy<NativeProvider>,
{
    let (population, end) = checked_population(snapshot.population)?;
    cancellation.check()?;
    let provider: NativeProvider = DefaultProvider::new_empty(
        DefaultProviderParameters::simple(
            snapshot.population,
            config.dimension,
            config.metric.native(),
            native_config.max_degree_u32().get(),
        ),
        CreateFullPrecision::<f32>::new(config.dimension, None),
        NoStore,
        NoDeletes,
    )
    .map_err(|e| native_error("constructor", e))?;
    if provider.capacity() != snapshot.population
        || provider.total_points() != end as usize
        || provider.num_start_points() != 1
        || provider
            .starting_points()
            .map_err(|e| native_error("root", e))?
            != [population]
    {
        return Err(BuildError::InvalidTopology(
            "native DiskANN population/frozen endpoints mismatch".into(),
        ));
    }
    let index = DiskANNIndex::new(native_config, provider, NonZeroUsize::new(1));
    let context = DefaultContext;
    let input = Arc::new(snapshot.values);
    let storage = MemoryStorage::new(
        input.clone(),
        population,
        config.dimension,
        cancellation.clone(),
    )?;
    let medoid_start = Instant::now();
    cancellation.check()?;
    let mut rng = create_rnd_from_optional_seed(Some(config.seed));
    let native_medoid = find_medoid_with_sampling::<f32, _>(
        MemoryStorage::KEY,
        &storage,
        MAX_MEDOID_SAMPLE_SIZE,
        &mut rng,
    );
    // The public helper internally has an error-as-EOF iterator. A first-error
    // latch and exact full-scan/closed-reader postconditions make it fail closed.
    let io = storage.finish(native_medoid.is_ok())?;
    let (medoid, medoid_id) = native_medoid.map_err(|e| native_error("medoid", e))?;
    cancellation.check()?;
    if medoid_id >= population as usize || medoid.len() != config.dimension {
        return Err(BuildError::InvalidTopology(
            "native DiskANN medoid cardinality/dimension mismatch".into(),
        ));
    }
    let source = &input[medoid_id * config.dimension..(medoid_id + 1) * config.dimension];
    if !same_bits(&medoid, source) {
        return Err(BuildError::InvalidTopology(
            "native DiskANN medoid is not the declared snapshot vector".into(),
        ));
    }
    let mut root_vector = reserved_vec(config.dimension)?;
    root_vector.extend_from_slice(&medoid);
    index
        .provider()
        .set_start_points(std::iter::once(medoid.as_slice()))
        .map_err(|e| native_error("set-start-point", e))?;
    let mut stats = BuildStats {
        native_workers: 1,
        native_coordinator_threads: 1,
        medoid_time: medoid_start.elapsed(),
        medoid_read_bytes: io.bytes_read,
        medoid_full_scan_vectors: io.full_scan_vectors,
        medoid_readers_closed: io.readers_closed,
        ..BuildStats::default()
    };
    drop(storage);
    drop(medoid);

    let insert_start = Instant::now();
    for id in 0..population {
        cancellation.check()?;
        let start = id as usize * config.dimension;
        let vector = &input[start..start + config.dimension];
        index
            .insert(&strategy, &context, &id, vector)
            .await
            .map_err(|e| native_error("scalar-insert", e))?;
        stats.inserted_vectors = stats.inserted_vectors.checked_add(1).ok_or_else(overflow)?;
    }
    stats.insert_time = insert_start.elapsed();
    if stats.inserted_vectors != population {
        return Err(BuildError::InvalidTopology(
            "native DiskANN inserted population mismatch".into(),
        ));
    }
    cancellation.check()?;
    let prune_start = Instant::now();
    index
        .prune_range(&strategy, &context, 0..end)
        .await
        .map_err(|e| native_error("final-prune", e))?;
    stats.prune_time = prune_start.elapsed();
    stats.pruned_nodes = end;
    cancellation.check()?;

    let extract_start = Instant::now();
    let mut offsets = reserved_vec(add(end as usize, 1)?)?;
    offsets.push(0);
    let mut neighbors = reserved_vec(mul(end as usize, config.max_degree)?)?;
    let mut list = AdjacencyList::with_capacity(config.max_degree);
    let mut unique = HashSet::new();
    unique
        .try_reserve(config.max_degree)
        .map_err(|_| BuildError::Allocation)?;
    stats.min_degree = config.max_degree;
    for id in 0..end {
        cancellation.check()?;
        index
            .provider()
            .neighbors()
            .get_neighbors_sync(id as usize, &mut list)
            .map_err(|e| native_error("neighbor-extraction", e))?;
        validate_neighbors(id, end, config.max_degree, &list, &mut unique)?;
        // SAFETY: The native provider/index is owned only by this coordinator,
        // every scalar insert/prune is fully awaited, no task is spawned by those
        // pinned APIs, and id is in the checked 0..N+1 range. There is no concurrent
        // mutable access while this borrowed logical vector is inspected.
        let stored = unsafe { index.provider().base_vectors.get_vector_sync(id as usize) };
        let expected = if id == population {
            root_vector.as_slice()
        } else {
            let start = id as usize * config.dimension;
            &input[start..start + config.dimension]
        };
        if !same_bits(stored, expected) {
            return Err(BuildError::InvalidTopology(
                "native DiskANN vector bytes/local ordinal mismatch".into(),
            ));
        }
        neighbors.extend_from_slice(&list);
        offsets.push(u64::try_from(neighbors.len()).map_err(|_| overflow())?);
        stats.min_degree = stats.min_degree.min(list.len());
        stats.max_degree = stats.max_degree.max(list.len());
        stats.extracted_nodes += 1;
    }
    stats.edge_count = neighbors.len();
    drop(unique);
    drop(list);
    // Destroy native input/provider/scratch before returning the owned candidate.
    drop(index);
    drop(input);
    let mut result = OwnedTopology {
        config_digest: config.digest(),
        config,
        population,
        root: Some(population),
        medoid_ordinal: Some(u32::try_from(medoid_id).map_err(|_| overflow())?),
        root_vector,
        offsets,
        neighbors,
        snapshot_digest: snapshot.digest,
        topology_digest: [0; 32],
        admission,
        stats,
    };
    result.topology_digest = result.calculate_topology_digest();
    result.validate()?;
    result.stats.reachable_nodes = reachable(&result)?;
    result.stats.retained_bytes = result.retained_bytes()?;
    if result.stats.retained_bytes > admission.retained_upper_bytes {
        return Err(BuildError::Limit(
            "DiskANN retained allocation exceeded admitted output estimate".into(),
        ));
    }
    result.stats.extract_time = extract_start.elapsed();
    cancellation.check()?;
    Ok(result)
}
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

#[cfg(test)]
#[path = "diskann_build/tests.rs"]
mod tests;
