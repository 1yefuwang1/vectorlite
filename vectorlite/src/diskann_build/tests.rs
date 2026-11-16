use super::*;
use diskann_providers::storage::StorageReadProvider;
use memory_reader::ReaderFault;
use std::{
    io::{Read, Seek, SeekFrom},
    sync::atomic::AtomicUsize,
};

fn config(metric: BuildMetric, dimension: usize) -> BuildConfig {
    BuildConfig {
        dimension,
        metric,
        max_degree: 32,
        construction_l: 100,
        alpha: 1.2,
        seed: 42,
        workers: 1,
    }
}
fn limits() -> BuildLimits {
    BuildLimits {
        memory_budget_bytes: 512 * 1024 * 1024,
        external_reservation_bytes: 4 * 1024 * 1024,
    }
}
fn toy(metric: BuildMetric) -> CheckedSnapshot {
    CheckedSnapshot::new(3, 2, metric, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap()
}
fn toy_output() -> OwnedTopology {
    build(
        toy(BuildMetric::L2Squared),
        config(BuildMetric::L2Squared, 2),
        limits(),
        Cancellation::default(),
    )
    .unwrap()
}

#[test]
fn initial_native_toy_api_smoke() {
    let output = toy_output();
    output.validate().unwrap();
    assert_eq!(output.root(), Some(3));
    assert_eq!(output.node_count(), 4);
    assert_eq!(output.stats().inserted_vectors, 3);
    assert_eq!(output.stats().pruned_nodes, 4);
    assert_eq!(output.stats().medoid_full_scan_vectors, 3);
    assert_eq!(output.stats().medoid_readers_closed, 3);
    assert_eq!(output.stats().native_alive_tasks_at_teardown, 0);
    assert!(output.stats().coordinator_joined);
    println!("native toy: {output:?}");
}

#[test]
fn explicit_empty_has_no_native_constructor_medoid_or_runtime() {
    for metric in [BuildMetric::L2Squared, BuildMetric::CosineNormalized] {
        let cfg = config(metric, 2);
        let input = CheckedSnapshot::new(0, 2, metric, Vec::new()).unwrap();
        let output = build_with_strategy(
            input,
            cfg,
            limits(),
            Cancellation::default(),
            FaultStrategy::new(FaultMode::Panic),
        )
        .unwrap();
        output.validate().unwrap();
        assert_eq!(output.population(), 0);
        assert_eq!(output.root(), None);
        assert_eq!(output.medoid_ordinal(), None);
        assert_eq!(output.node_count(), 0);
        assert_eq!(output.offsets(), [0]);
        assert!(output.root_vector().is_empty() && output.adjacency().is_empty());
        assert_eq!(output.stats().inserted_vectors, 0);
        assert_eq!(output.stats().medoid_readers_closed, 0);
        assert!(!output.stats().coordinator_joined);
        assert_eq!(output.admission().provider_vector_bytes, 0);
        assert_eq!(output.admission().provider_graph_bytes, 0);
        assert_eq!(output.admission().native_workspace_bytes, 0);
        assert_eq!(output.admission().runtime_and_allocator_allowance_bytes, 0);
    }
}

#[test]
fn singleton_native_medoid_root_and_signed_zero_bytes() {
    for metric in [BuildMetric::L2Squared, BuildMetric::CosineNormalized] {
        for vector in [vec![0.0, -0.0], vec![1.0, -0.0]] {
            let output = build(
                CheckedSnapshot::new(1, 2, metric, vector.clone()).unwrap(),
                config(metric, 2),
                limits(),
                Cancellation::default(),
            )
            .unwrap();
            output.validate().unwrap();
            assert_eq!(output.root(), Some(1));
            assert_eq!(output.medoid_ordinal(), Some(0));
            assert!(same_bits(output.root_vector(), &vector));
            assert_eq!(output.node_count(), 2);
            assert_eq!(output.stats().inserted_vectors, 1);
            assert_eq!(output.stats().pruned_nodes, 2);
            assert_eq!(output.stats().medoid_full_scan_vectors, 1);
            assert_eq!(output.stats().medoid_readers_closed, 3);
            assert!(output.stats().coordinator_joined);
            println!(
                "singleton {metric:?}: offsets={:?}, edges={:?}",
                output.offsets(),
                output.adjacency()
            );
        }
    }
}

#[test]
fn zero_duplicate_small_populations_repeat_native_shape_and_order() {
    for metric in [BuildMetric::L2Squared, BuildMetric::CosineNormalized] {
        for population in [2, 4, 17] {
            let data = vec![0.0; population * 2];
            let mut previous = None;
            for _ in 0..2 {
                let output = build(
                    CheckedSnapshot::new(population, 2, metric, data.clone()).unwrap(),
                    config(metric, 2),
                    limits(),
                    Cancellation::default(),
                )
                .unwrap();
                output.validate().unwrap();
                assert_eq!(output.root(), Some(population as u32));
                assert_eq!(output.medoid_ordinal(), Some(0));
                assert_eq!(output.stats().inserted_vectors as usize, population);
                assert_eq!(output.stats().extracted_nodes as usize, population + 1);
                if let Some(digest) = previous {
                    assert_eq!(digest, output.topology_digest());
                }
                previous = Some(output.topology_digest());
            }
        }
    }
}

#[test]
fn checked_input_rejects_nonfinite_dimension_population_and_norm_errors() {
    for metric in [BuildMetric::L2Squared, BuildMetric::CosineNormalized] {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
            assert!(CheckedSnapshot::new(1, 1, metric, vec![value]).is_err());
        }
        assert!(CheckedSnapshot::new(1, 0, metric, vec![]).is_err());
        assert!(CheckedSnapshot::new(1, 2, metric, vec![1.0]).is_err());
        assert!(CheckedSnapshot::new(0, 2, metric, vec![0.0]).is_err());
        assert!(CheckedSnapshot::new(2, 1, metric, vec![0.0]).is_err());
    }
    assert!(CheckedSnapshot::new(1, 1, BuildMetric::L2Squared, vec![1.0e19]).is_err());
    assert!(CheckedSnapshot::new(1, 1, BuildMetric::CosineNormalized, vec![1.0e19]).is_ok());
    assert!(checked_population(u32::MAX as usize).is_err());
    assert!(checked_population(usize::MAX).is_err());
    assert_eq!(
        checked_population(u32::MAX as usize - 1).unwrap(),
        (u32::MAX - 1, u32::MAX)
    );
    // The public native start-point helper itself rejects the overflowing end;
    // never attempt the huge allocating provider constructor to observe this error.
    assert!(
        diskann_providers::model::graph::provider::async_::common::StartPoints::new(
            u32::MAX,
            NonZeroUsize::new(1).unwrap()
        )
        .is_err()
    );
}

#[test]
fn accepted_extreme_l2_values_use_real_native_medoid_and_graph() {
    let data = vec![8.0e18_f32, -8.0e18_f32, 0.0];
    let output = build(
        CheckedSnapshot::new(3, 1, BuildMetric::L2Squared, data).unwrap(),
        config(BuildMetric::L2Squared, 1),
        limits(),
        Cancellation::default(),
    )
    .unwrap();
    output.validate().unwrap();
    assert_eq!(output.medoid_ordinal(), Some(2));
    assert_eq!(output.stats().inserted_vectors, 3);
}

#[test]
fn descriptor_validation_precedes_native_allocation() {
    let mut cfg = config(BuildMetric::L2Squared, 2);
    for workers in [0, 2, usize::MAX] {
        cfg.workers = workers;
        assert!(matches!(
            admit(3, &cfg, limits()),
            Err(BuildError::InvalidConfig(_))
        ));
    }
    cfg.workers = 1;
    for alpha in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.9] {
        cfg.alpha = alpha;
        assert!(matches!(
            admit(3, &cfg, limits()),
            Err(BuildError::InvalidConfig(_))
        ));
    }
    cfg.alpha = 1.2;
    for degree in [0, 1, 10_001, usize::MAX] {
        cfg.max_degree = degree;
        assert!(admit(3, &cfg, limits()).is_err());
    }
    cfg.max_degree = 32;
    cfg.construction_l = 0;
    assert!(admit(3, &cfg, limits()).is_err());
    cfg.construction_l = u32::MAX as usize + 1;
    assert!(admit(3, &cfg, limits()).is_err());
    cfg.construction_l = 100;
    cfg.dimension = 0;
    assert!(admit(3, &cfg, limits()).is_err());
    cfg.dimension = u32::MAX as usize + 1;
    assert!(admit(3, &cfg, limits()).is_err());
}

#[test]
fn admission_accounts_alignment_native_extra_row_and_dense_medoid_overlap() {
    let cfg = config(BuildMetric::L2Squared, 17);
    let admission = admit(3000, &cfg, limits()).unwrap();
    assert_eq!(admission.input_bytes, 3000 * 17 * 4);
    assert_eq!(
        admission.provider_vector_bytes,
        3001 * 128 + 60 + 3001_usize.div_ceil(16) * size_of::<Mutex<()>>() + 1024
    );
    // Native R32 default slack gives physical41, graph rows have 42 u32s,
    // padded192 bytes. DefaultProvider passes N+1 then neighbors adds another1.
    assert_eq!(
        admission.provider_graph_bytes,
        3002 * 192 + 60 + 3002 * (size_of::<RwLock<()>>() + 64) + 1024
    );
    assert!(admission.native_workspace_bytes >= 64 * 1024 * 1024);
    assert_eq!(
        admission.peak_estimate_bytes,
        admission.input_bytes
            + admission.provider_vector_bytes
            + admission.provider_graph_bytes
            + admission.native_workspace_bytes
            + admission.retained_upper_bytes
            + admission.native_coordinator_stack_bytes
            + admission.runtime_and_allocator_allowance_bytes
            + admission.external_reservation_bytes
    );
    let mut low = limits();
    low.memory_budget_bytes = admission.peak_estimate_bytes - 1;
    assert!(matches!(admit(3000, &cfg, low), Err(BuildError::Limit(_))));
    low.memory_budget_bytes = admission.peak_estimate_bytes;
    assert_eq!(admit(3000, &cfg, low).unwrap(), admission);
    low.external_reservation_bytes += 1;
    assert!(admit(3000, &cfg, low).is_err());
    low.memory_budget_bytes = 0;
    assert!(admit(0, &cfg, low).is_err());
    assert!(admit(usize::MAX, &cfg, limits()).is_err());
    println!("memory admission17D/3k: {admission:?}");
}

#[test]
fn actual_spare_input_capacity_cannot_bypass_admission() {
    let cfg = config(BuildMetric::L2Squared, 2);
    let mut exact = limits();
    exact.memory_budget_bytes = admit(3, &cfg, exact).unwrap().peak_estimate_bytes;
    let mut data = Vec::with_capacity(1000);
    data.extend_from_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    let input = CheckedSnapshot::new(3, 2, cfg.metric, data).unwrap();
    assert!(matches!(
        build(input, cfg, exact, Cancellation::default()),
        Err(BuildError::Limit(_))
    ));
}

#[test]
fn exact_admitted_budget_builds_and_excess_external_reservation_rejects() {
    let cfg = config(BuildMetric::L2Squared, 2);
    let mut exact = limits();
    exact.memory_budget_bytes = admit(3, &cfg, exact).unwrap().peak_estimate_bytes;
    let output = build(toy(cfg.metric), cfg.clone(), exact, Cancellation::default()).unwrap();
    assert!(output.stats().retained_bytes <= output.admission().retained_upper_bytes);
    exact.external_reservation_bytes += 1;
    assert!(matches!(
        build(toy(cfg.metric), cfg, exact, Cancellation::default()),
        Err(BuildError::Limit(_))
    ));
}

#[test]
fn snapshot_config_mismatch_is_fail_closed() {
    assert!(matches!(
        build(
            toy(BuildMetric::L2Squared),
            config(BuildMetric::CosineNormalized, 2),
            limits(),
            Cancellation::default()
        ),
        Err(BuildError::InvalidInput(_))
    ));
    assert!(matches!(
        build(
            toy(BuildMetric::L2Squared),
            config(BuildMetric::L2Squared, 3),
            limits(),
            Cancellation::default()
        ),
        Err(BuildError::InvalidInput(_))
    ));
}

#[test]
fn cancellation_before_native_constructor_and_between_completed_inserts() {
    let cancel = Cancellation::default();
    cancel.cancel();
    assert!(matches!(
        build(
            toy(BuildMetric::L2Squared),
            config(BuildMetric::L2Squared, 2),
            limits(),
            cancel
        ),
        Err(BuildError::Cancelled)
    ));
    let cancel = Cancellation::default();
    let mut strategy = FaultStrategy::new(FaultMode::CancelAt(0));
    strategy.cancellation = Some(cancel.clone());
    assert!(matches!(
        build_with_strategy(
            toy(BuildMetric::L2Squared),
            config(BuildMetric::L2Squared, 2),
            limits(),
            cancel,
            strategy
        ),
        Err(BuildError::Cancelled)
    ));
}

#[derive(Clone, Copy)]
enum FaultMode {
    InsertAt(usize),
    Prune,
    Panic,
    CancelAt(usize),
}
struct FaultStrategy {
    mode: FaultMode,
    calls: AtomicUsize,
    dropped: Arc<AtomicUsize>,
    cancellation: Option<Cancellation>,
}
impl FaultStrategy {
    fn new(mode: FaultMode) -> Self {
        Self {
            mode,
            calls: AtomicUsize::new(0),
            dropped: Arc::new(AtomicUsize::new(0)),
            cancellation: None,
        }
    }
}
impl Drop for FaultStrategy {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}
impl<'a> InsertStrategy<'a, NativeProvider, &'a [f32]> for FaultStrategy {
    type SearchAccessor =
        <FullPrecision as InsertStrategy<'a, NativeProvider, &'a [f32]>>::SearchAccessor;
    type SearchAccessorError = diskann::ANNError;
    type PruneStrategy = FullPrecision;
    fn insert_search_accessor(
        &'a self,
        provider: &'a NativeProvider,
        context: &'a DefaultContext,
        vector: &'a [f32],
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        let ordinal = self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            FaultMode::InsertAt(fail) if ordinal == fail => {
                return Err(diskann::ANNError::message(
                    "injected public native insert accessor error",
                ))
            }
            FaultMode::Panic => panic!("injected native accessor panic (unwind test profile only)"),
            FaultMode::CancelAt(cancel) if ordinal == cancel => {
                self.cancellation.as_ref().unwrap().cancel()
            }
            _ => {}
        }
        <FullPrecision as InsertStrategy<'a, NativeProvider, &'a [f32]>>::insert_search_accessor(
            &FullPrecision,
            provider,
            context,
            vector,
        )
        .map_err(|error| diskann::ANNError::message(error.to_string()))
    }
    fn prune_strategy(&self) -> Self::PruneStrategy {
        FullPrecision
    }
}
impl PruneStrategy<NativeProvider> for FaultStrategy {
    type PruneAccessor<'a> = <FullPrecision as PruneStrategy<NativeProvider>>::PruneAccessor<'a>;
    type PruneAccessorError = diskann::ANNError;
    fn prune_accessor<'a>(
        &'a self,
        provider: &'a NativeProvider,
        context: &'a DefaultContext,
        capacity: usize,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        if matches!(self.mode, FaultMode::Prune) {
            return Err(diskann::ANNError::message(
                "injected public native final prune accessor error",
            ));
        }
        <FullPrecision as PruneStrategy<NativeProvider>>::prune_accessor(
            &FullPrecision,
            provider,
            context,
            capacity,
        )
        .map_err(|error| diskann::ANNError::message(error.to_string()))
    }
}

#[test]
fn real_native_scalar_and_final_prune_errors_return_no_candidate_and_join() {
    for mode in [
        FaultMode::InsertAt(0),
        FaultMode::InsertAt(2),
        FaultMode::Prune,
    ] {
        let strategy = FaultStrategy::new(mode);
        let dropped = strategy.dropped.clone();
        let error = build_with_strategy(
            toy(BuildMetric::L2Squared),
            config(BuildMetric::L2Squared, 2),
            limits(),
            Cancellation::default(),
            strategy,
        )
        .unwrap_err();
        let expected = if matches!(mode, FaultMode::Prune) {
            "final-prune"
        } else {
            "scalar-insert"
        };
        assert!(
            matches!(&error, BuildError::Native { stage, message } if *stage == expected && message.contains("injected public native"))
        );
        assert_eq!(
            dropped.load(Ordering::SeqCst),
            1,
            "owned strategy destroyed before caller return"
        );
        toy_output().validate().unwrap();
        println!("propagated native error: {error}");
    }
}

#[test]
fn coordinator_unwind_join_destroys_owned_work_and_returns_no_candidate() {
    let strategy = FaultStrategy::new(FaultMode::Panic);
    let dropped = strategy.dropped.clone();
    assert!(matches!(
        build_with_strategy(
            toy(BuildMetric::L2Squared),
            config(BuildMetric::L2Squared, 2),
            limits(),
            Cancellation::default(),
            strategy
        ),
        Err(BuildError::NativePanic)
    ));
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert!(toy_output().stats().coordinator_joined);
}

#[test]
fn ambient_current_thread_and_multithread_tokio_remain_safe_and_joined() {
    for multithread in [false, true] {
        let rt = if multithread {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .build()
                .unwrap()
        } else {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
        };
        rt.block_on(async {
            assert!(tokio::runtime::Handle::try_current().is_ok());
            let output = toy_output();
            assert!(output.stats().coordinator_joined);
            assert_eq!(output.stats().native_alive_tasks_at_teardown, 0);
            assert!(tokio::runtime::Handle::try_current().is_ok());
        });
        assert_eq!(rt.metrics().num_alive_tasks(), 0);
        drop(rt);
    }
}

fn memory_storage() -> MemoryStorage {
    MemoryStorage::new(
        Arc::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
        3,
        2,
        Cancellation::default(),
    )
    .unwrap()
}
fn native_medoid_with_audit(storage: &MemoryStorage) -> BuildResult<(Vec<f32>, usize)> {
    let native = find_medoid_with_sampling::<f32, _>(
        MemoryStorage::KEY,
        storage,
        MAX_MEDOID_SAMPLE_SIZE,
        &mut create_rnd_from_optional_seed(Some(42)),
    );
    storage.finish(native.is_ok())?;
    native.map_err(|error| native_error("medoid", error))
}

#[test]
fn checked_memory_read_seek_preserves_exact_le_bytes_and_rejects_outside_seeks() {
    let storage = memory_storage();
    assert!(storage.exists(MemoryStorage::KEY));
    assert!(!storage.exists("arbitrary-file-path"));
    assert_eq!(storage.get_length(MemoryStorage::KEY).unwrap(), 32);
    let mut reader = storage.open_reader(MemoryStorage::KEY).unwrap();
    let mut bytes = Vec::new();
    for width in [1, 3, 5, 23] {
        let mut chunk = vec![0; width];
        reader.read_exact(&mut chunk).unwrap();
        bytes.extend(chunk);
    }
    let mut expected = Vec::new();
    expected.extend_from_slice(&3_u32.to_le_bytes());
    expected.extend_from_slice(&2_u32.to_le_bytes());
    for value in [0_f32, 0.0, 1.0, 0.0, 0.0, 1.0] {
        expected.extend_from_slice(&value.to_le_bytes());
    }
    assert_eq!(bytes, expected);
    reader.seek(SeekFrom::Start(17)).unwrap();
    let mut mid = [0; 3];
    reader.read_exact(&mut mid).unwrap();
    assert_eq!(mid, expected[17..20]);
    assert!(reader.seek(SeekFrom::Start(u64::MAX)).is_err());
    assert!(reader.seek(SeekFrom::End(1)).is_err());
    assert!(reader.seek(SeekFrom::Current(i64::MIN)).is_err());
    drop(reader);
    assert!(
        matches!(storage.finish(false), Err(BuildError::InputRead(message)) if message.contains("seek outside"))
    );
}

#[test]
fn public_medoid_memory_bridge_requires_all_readers_closed_and_exact_scan() {
    let storage = memory_storage();
    let mut reader = storage.open_reader(MemoryStorage::KEY).unwrap();
    let mut header = [0; 8];
    reader.read_exact(&mut header).unwrap();
    assert!(
        matches!(storage.finish(false), Err(BuildError::InputRead(message)) if message.contains("escaped"))
    );
    drop(reader);
    assert!(
        matches!(storage.finish(true), Err(BuildError::InputRead(message)) if message.contains("cardinality"))
    );
    let complete = memory_storage();
    let (root, ordinal) = native_medoid_with_audit(&complete).unwrap();
    assert_eq!(ordinal, 0);
    assert_eq!(root, [0.0, 0.0]);
    let stats = complete.finish(true).unwrap();
    assert_eq!(stats.full_scan_vectors, 3);
    assert_eq!(stats.readers_closed, 3);
    assert_eq!(stats.bytes_read, 72);
}

#[test]
fn public_medoid_read_open_and_length_errors_fail_closed() {
    for fault in [
        ReaderFault::Open(0),
        ReaderFault::Open(1),
        ReaderFault::Open(2),
        ReaderFault::Length,
        ReaderFault::Read {
            reader: 0,
            at: 0,
            eof: false,
        },
        ReaderFault::Read {
            reader: 1,
            at: 8,
            eof: false,
        },
        ReaderFault::Read {
            reader: 2,
            at: 8,
            eof: false,
        },
    ] {
        let mut storage = memory_storage();
        storage.set_fault(fault);
        assert!(
            native_medoid_with_audit(&storage).is_err(),
            "fault={fault:?}"
        );
    }
}

#[test]
fn swallowed_native_iterator_read_errors_and_eof_are_recovered_by_latch() {
    for at in [16, 24, 28] {
        for eof in [false, true] {
            let mut storage = memory_storage();
            storage.set_fault(ReaderFault::Read { reader: 2, at, eof });
            // The unmodified public helper actually succeeds using a truncated
            // nearest-vector scan. It is not sufficient to trust its Result.
            let native = find_medoid_with_sampling::<f32, _>(
                MemoryStorage::KEY,
                &storage,
                MAX_MEDOID_SAMPLE_SIZE,
                &mut create_rnd_from_optional_seed(Some(42)),
            );
            assert!(
                native.is_ok(),
                "qualify the upstream swallowed-error hazard"
            );
            assert!(
                matches!(storage.finish(true), Err(BuildError::InputRead(message)) if message.contains("injected medoid read failure"))
            );
        }
    }
}

#[test]
fn medoid_sparse_sampling_seek_errors_and_exact_full_scan_are_qualified() {
    let population = 100_001;
    let values: Vec<f32> = (0..population).map(|id| id as f32).collect();
    let good = MemoryStorage::new(
        Arc::new(values.clone()),
        population,
        1,
        Cancellation::default(),
    )
    .unwrap();
    let (root, ordinal) = native_medoid_with_audit(&good).unwrap();
    assert_eq!(root[0], ordinal as f32);
    assert_eq!(
        good.finish(true).unwrap().full_scan_vectors,
        population as usize
    );
    let mut bad =
        MemoryStorage::new(Arc::new(values), population, 1, Cancellation::default()).unwrap();
    bad.set_fault(ReaderFault::Seek(1));
    assert!(
        matches!(native_medoid_with_audit(&bad), Err(BuildError::InputRead(message)) if message.contains("seek failure"))
    );
}

#[test]
fn medoid_first_error_latch_keeps_original_error_over_cancellation() {
    let cancel = Cancellation::default();
    let storage = MemoryStorage::new(Arc::new(vec![1.0]), 1, 1, cancel.clone()).unwrap();
    assert!(storage.open_reader("not-the-owned-object").is_err());
    cancel.cancel();
    let mut reader = storage.open_reader(MemoryStorage::KEY).unwrap();
    assert!(reader.read(&mut [0; 1]).is_err());
    drop(reader);
    assert!(
        matches!(storage.finish(false), Err(BuildError::InputRead(message)) if message.contains("unknown native medoid"))
    );
}

#[test]
fn native_full_precision_dimension_error_is_propagated_without_partial_success() {
    let cfg = config(BuildMetric::L2Squared, 2);
    let native_cfg = cfg.native_config().unwrap();
    let provider: NativeProvider = DefaultProvider::new_empty(
        DefaultProviderParameters::simple(1, 2, Metric::L2, native_cfg.max_degree_u32().get()),
        CreateFullPrecision::<f32>::new(2, None),
        NoStore,
        NoDeletes,
    )
    .unwrap();
    provider
        .set_start_points(std::iter::once(&[0.0_f32, 0.0][..]))
        .unwrap();
    let index = DiskANNIndex::new(native_cfg, provider, NonZeroUsize::new(1));
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(async {
        let error = index
            .insert(&FullPrecision, &DefaultContext, &0, &[1.0_f32][..])
            .await
            .unwrap_err();
        assert!(error.to_string().to_ascii_lowercase().contains("dimension"));
        index
            .insert(&FullPrecision, &DefaultContext, &0, &[1.0_f32, 0.0][..])
            .await
            .unwrap();
        index
            .prune_range(&FullPrecision, &DefaultContext, 0..2)
            .await
            .unwrap();
    });
    assert_eq!(rt.metrics().num_alive_tasks(), 0);
    drop(index);
    drop(rt);
}

#[test]
fn cancelled_medoid_reads_fail_terminally_instead_of_retrying_interrupted() {
    let cancel = Cancellation::default();
    let storage = MemoryStorage::new(Arc::new(vec![1.0, 0.0]), 1, 2, cancel.clone()).unwrap();
    cancel.cancel();
    assert!(matches!(
        native_medoid_with_audit(&storage),
        Err(BuildError::Cancelled)
    ));
}

#[test]
fn repeated_native_error_then_success_has_no_escaped_tasks_or_state() {
    let digest = toy_output().topology_digest();
    for _ in 0..5 {
        let strategy = FaultStrategy::new(FaultMode::Prune);
        assert!(build_with_strategy(
            toy(BuildMetric::L2Squared),
            config(BuildMetric::L2Squared, 2),
            limits(),
            Cancellation::default(),
            strategy
        )
        .is_err());
        let result = toy_output();
        assert_eq!(result.topology_digest(), digest);
        assert!(result.stats().coordinator_joined);
        assert_eq!(result.stats().native_alive_tasks_at_teardown, 0);
    }
}

#[test]
fn immutable_output_validator_rejects_corrupt_csr_root_and_digests() {
    for fault in 0..8 {
        let mut output = toy_output();
        match fault {
            0 => output.neighbors[0] = 0,
            1 => output.neighbors[0] = 4,
            2 => output.neighbors[1] = output.neighbors[0],
            3 => *output.offsets.last_mut().unwrap() += 1,
            4 => output.root = None,
            5 => output.medoid_ordinal = Some(3),
            6 => output.root_vector[0] = f32::NAN,
            7 => output.topology_digest[0] ^= 1,
            _ => unreachable!(),
        }
        assert!(output.validate().is_err(), "corruption={fault}");
    }
}
