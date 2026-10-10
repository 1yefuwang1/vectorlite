use super::*;
use diskann::graph::glue::SearchPostProcess;
use diskann::provider::{Guard, SetElement};

#[derive(Clone)]
struct Node {
    rowid: Option<u64>,
    state: NodeState,
    vector: Vec<f32>,
    neighbors: Vec<u64>,
}
#[derive(Clone, Default)]
struct FakeStore {
    nodes: HashMap<u64, Node>,
    mapping: HashMap<u64, u64>,
    next: u64,
    fail: Option<&'static str>,
    calls: Vec<&'static str>,
    fail_at: Option<(&'static str, usize)>,
    owner_thread: Option<ThreadId>,
    // Deliberately !Send/!Sync, like a borrowed SQLite connection owner.
    _local: Rc<()>,
}
impl FakeStore {
    fn call(&mut self, name: &'static str) -> StoreResult<()> {
        assert_eq!(
            *self.owner_thread.get_or_insert(thread::current().id()),
            thread::current().id()
        );
        self.calls.push(name);
        if self.fail == Some(name)
            || self.fail_at.is_some_and(|(method, nth)| {
                method == name && self.calls.iter().filter(|call| **call == name).count() == nth
            })
        {
            return Err(IndexError::with_code(
                778,
                format!("injected SQLite I/O: {name}"),
            ));
        }
        Ok(())
    }
}
impl GraphStore for FakeStore {
    fn allocate(&mut self, rowid: u64, vector: &[f32]) -> StoreResult<u64> {
        self.call("allocate")?;
        if self.mapping.contains_key(&rowid) {
            return Err(IndexError::new("duplicate rowid"));
        }
        self.nodes.entry(0).or_insert_with(|| Node {
            rowid: None,
            state: NodeState::Frozen,
            vector: vector.to_vec(),
            neighbors: vec![],
        });
        self.next += 1;
        let id = self.next;
        self.nodes.insert(
            id,
            Node {
                rowid: Some(rowid),
                state: NodeState::Live,
                vector: vector.to_vec(),
                neighbors: vec![],
            },
        );
        self.mapping.insert(rowid, id);
        Ok(id)
    }
    fn internal_id(&mut self, rowid: u64) -> StoreResult<u64> {
        self.call("internal_id")?;
        self.mapping
            .get(&rowid)
            .copied()
            .ok_or_else(|| IndexError::new("missing rowid"))
    }
    fn rowid(&mut self, id: u64) -> StoreResult<Option<u64>> {
        self.call("rowid")?;
        self.nodes
            .get(&id)
            .map(|n| n.rowid)
            .ok_or_else(|| IndexError::new("missing node"))
    }
    fn state(&mut self, id: u64) -> StoreResult<NodeState> {
        self.call("state")?;
        self.nodes
            .get(&id)
            .map(|n| n.state)
            .ok_or_else(|| IndexError::new("missing state"))
    }
    fn vector(&mut self, id: u64) -> StoreResult<Vec<f32>> {
        self.call("vector")?;
        self.nodes
            .get(&id)
            .map(|n| n.vector.clone())
            .ok_or_else(|| IndexError::new("missing vector"))
    }
    fn neighbors(&mut self, id: u64) -> StoreResult<Vec<u64>> {
        self.call("neighbors")?;
        self.nodes
            .get(&id)
            .map(|n| n.neighbors.clone())
            .ok_or_else(|| IndexError::new("missing edges"))
    }
    fn set_neighbors(&mut self, id: u64, neighbors: &[u64]) -> StoreResult<()> {
        self.call("set_neighbors")?;
        self.nodes
            .get_mut(&id)
            .ok_or_else(|| IndexError::new("missing node"))?
            .neighbors = neighbors.to_vec();
        Ok(())
    }
    fn mark_delete(&mut self, rowid: u64) -> StoreResult<()> {
        self.call("mark_delete")?;
        let id = self
            .mapping
            .remove(&rowid)
            .ok_or_else(|| IndexError::new("missing delete rowid"))?;
        let node = self.nodes.get_mut(&id).unwrap();
        node.rowid = None;
        node.state = NodeState::Deleted;
        Ok(())
    }
    fn start_points(&mut self) -> StoreResult<Vec<u64>> {
        self.call("start_points")?;
        Ok(if self.nodes.is_empty() {
            vec![]
        } else {
            vec![0]
        })
    }
}
#[test]
fn deleted_one_way_bridge_keeps_live_nodes_reachable() {
    let mut store = FakeStore::default();
    let bridge = store.allocate(2, &[2.0, 0.0]).unwrap();
    let live = store.allocate(3, &[3.0, 0.0]).unwrap();
    // Legal directed topology: the entrypoint is an incoming neighbor that the
    // bridge does not point back to. OneHop repair would miss that incoming edge.
    store.set_neighbors(0, &[bridge]).unwrap();
    store.set_neighbors(bridge, &[live]).unwrap();
    store.set_neighbors(live, &[bridge]).unwrap();
    delete(&mut store, &config(), 2).unwrap();
    assert_eq!(store.nodes[&bridge].neighbors, vec![live]);
    assert_eq!(store.nodes[&bridge].state, NodeState::Deleted);
    assert_eq!(
        search(&mut store, &config(), &[3.0, 0.0], 1, None, None).unwrap(),
        vec![SearchResult::new(0.0, 3)]
    );
}

fn config() -> GraphConfig {
    GraphConfig {
        dimension: 2,
        distance: DistanceType::L2,
        max_degree: 8,
        construction_l: 32,
        search_l: 32,
        alpha: 1.2,
        limits: ResourceLimits::default(),
    }
}
fn populated() -> FakeStore {
    let mut store = FakeStore::default();
    for rowid in 1..=20 {
        insert(&mut store, &config(), rowid, &[rowid as f32, 0.0]).unwrap();
    }
    store.calls.clear();
    store
}
#[test]
fn real_graph_search_filter_delete_and_reinsert() {
    let mut store = populated();
    let results = search(&mut store, &config(), &[7.0, 0.0], 3, None, None).unwrap();
    assert_eq!(results[0], SearchResult::new(0.0, 7));
    let filter = HashSet::from([17, 19]);
    let results = search(&mut store, &config(), &[7.0, 0.0], 2, None, Some(&filter)).unwrap();
    assert_eq!(
        results.iter().map(|r| r.rowid).collect::<Vec<_>>(),
        vec![17, 19]
    );
    let old = store.mapping[&7];
    delete(&mut store, &config(), 7).unwrap();
    assert_eq!(store.nodes[&old].state, NodeState::Deleted);
    assert!(search(&mut store, &config(), &[7.0, 0.0], 20, None, None)
        .unwrap()
        .iter()
        .all(|r| r.rowid != 7));
    insert(&mut store, &config(), 7, &[70.0, 0.0]).unwrap();
    assert_ne!(store.mapping[&7], old);
    // Retired vectors/topology remain available until the SQL owner rebuilds
    // the whole live graph, rather than pruning unknown directed bridge edges.
    assert!(!store.nodes[&old].neighbors.is_empty());
    assert_eq!(store.nodes[&old].vector, vec![7.0, 0.0]);
}
#[test]
fn rejected_live_nodes_still_bridge_filtered_search() {
    let mut store = FakeStore::default();
    for rowid in 1..=3 {
        store.allocate(rowid, &[rowid as f32, 0.0]).unwrap();
    }
    store.set_neighbors(0, &[1]).unwrap();
    store.set_neighbors(1, &[2]).unwrap();
    store.set_neighbors(2, &[3]).unwrap();
    let results = search(
        &mut store,
        &config(),
        &[3.0, 0.0],
        1,
        None,
        Some(&HashSet::from([3])),
    )
    .unwrap();
    assert_eq!(results, vec![SearchResult::new(0.0, 3)]);
}
#[test]
fn empty_full_delete_and_cosine_match_existing_policy() {
    let mut cfg = config();
    cfg.distance = DistanceType::Cosine;
    let mut store = FakeStore::default();
    assert!(search(&mut store, &cfg, &[1.0, 0.0], 1, None, None)
        .unwrap()
        .is_empty());
    insert(&mut store, &cfg, 1, &[1.0, 0.0]).unwrap();
    insert(&mut store, &cfg, 2, &[0.0, 1.0]).unwrap();
    insert(&mut store, &cfg, 3, &[0.0, 0.0]).unwrap();
    let results = search(&mut store, &cfg, &[1.0, 0.0], 3, None, None).unwrap();
    assert_eq!(results[0], SearchResult::new(0.0, 1));
    assert!(results[1..].iter().all(|r| r.distance == 1.0));
    for rowid in 1..=3 {
        delete(&mut store, &cfg, rowid).unwrap();
    }
    assert!(search(&mut store, &cfg, &[1.0, 0.0], 3, None, None)
        .unwrap()
        .is_empty());
    insert(&mut store, &cfg, 4, &[1.0, 0.0]).unwrap();
    assert_eq!(
        search(&mut store, &cfg, &[1.0, 0.0], 1, None, None).unwrap()[0].rowid,
        4
    );
}
#[test]
fn every_sql_failure_stays_fatal_and_requires_whole_statement_rollback() {
    for method in [
        "allocate",
        "start_points",
        "vector",
        "neighbors",
        "set_neighbors",
    ] {
        let mut store = populated();
        let before = store.clone();
        store.fail = Some(method);
        let error = insert(&mut store, &config(), 21, &[21.0, 0.0]).unwrap_err();
        assert_eq!(error.code, 778, "{method}: {error}");
        assert!(error.message.contains(method));
        // Fake models the caller's whole-statement restoration; no graph guard
        // claims to undo adjacency writes. SQLite rollback is tested by store.
        store = before;
        assert!(!store.mapping.contains_key(&21));
        assert_eq!(
            search(&mut store, &config(), &[7.0, 0.0], 1, None, None).unwrap()[0].rowid,
            7
        );
    }
    // Soft retirement deliberately performs no approximate topology repair.
    // SQL errors inside the store's mapping/counter writes surface through this
    // single checked boundary; SQL regressions inject failures in those writes.
    {
        let method = "mark_delete";
        let mut store = populated();
        let before = store.clone();
        store.fail = Some(method);
        let error = delete(&mut store, &config(), 7).unwrap_err();
        assert_eq!(error.code, 778, "{method}: {error}");
        store = before;
        assert!(store.mapping.contains_key(&7));
    }
    let mut store = populated();
    store.fail = Some("rowid");
    assert_eq!(
        search(&mut store, &config(), &[7.0, 0.0], 1, None, None)
            .unwrap_err()
            .code,
        778
    );
}
#[test]
fn visits_and_vector_bytes_have_hard_limits() {
    let mut store = populated();
    let mut cfg = config();
    cfg.limits.max_visits = 33;
    assert!(search(&mut store, &cfg, &[7.0, 0.0], 3, None, None)
        .unwrap_err()
        .message
        .contains("limit"));
    let mut cfg = config();
    cfg.limits.max_cached_vectors = 1;
    assert!(search(&mut store, &cfg, &[7.0, 0.0], 1, None, None)
        .unwrap_err()
        .message
        .contains("working-set"));
    let mut cfg = config();
    cfg.limits.max_vector_bytes = 4096;
    store.calls.clear();
    let error = search(&mut store, &cfg, &[7.0, 0.0], 1, None, None).unwrap_err();
    assert!(error.message.contains("scratch"));
    assert_eq!(error.code, crate::ffi::SQLITE_TOOBIG as i32);
    assert!(
        store.calls.is_empty(),
        "scratch is admitted before any store callbacks"
    );
    let mut cfg = config();
    cfg.limits.max_vector_bytes = 16 * 1024 * 1024;
    store.calls.clear();
    assert!(search(&mut store, &cfg, &[7.0, 0.0], 1, Some(50_000), None)
        .unwrap_err()
        .message
        .contains("scratch"));
    assert!(store.calls.is_empty());
}
#[test]
fn overflowing_distance_latches_errors_in_pruning_and_search() {
    let mut store = populated();
    let error = insert(&mut store, &config(), 21, &[f32::MAX, 0.0]).unwrap_err();
    assert!(error.message.contains("non-finite"));
    let mut store = populated();
    let error = search(&mut store, &config(), &[f32::MAX, 0.0], 1, None, None).unwrap_err();
    assert!(error.message.contains("non-finite"));
    operation(&mut store, &config(), |_, context| {
        assert_eq!(
            Computer(context.clone()).evaluate_similarity(&[f32::MAX, 0.0], &[-f32::MAX, 0.0]),
            f32::MAX
        );
        Ok(())
    })
    .unwrap_err();
}
#[test]
fn leases_release_bytes_on_success_and_fetch_error() {
    let mut store = populated();
    operation(&mut store, &config(), |_, context| {
        let base = context.lock().bytes;
        let vector = context.fetch_vector(1)?;
        assert_eq!(context.lock().bytes, base + 8 + vector_metadata_bytes());
        drop(vector);
        assert_eq!(context.lock().bytes, base);
        assert_eq!(context.lock().vectors, 0);
        Ok(())
    })
    .unwrap();
    store.fail = Some("vector");
    operation(&mut store, &config(), |_, context| {
        let base = context.lock().bytes;
        assert!(context.fetch_vector(1).is_err());
        assert_eq!(context.lock().bytes, base);
        assert_eq!(context.lock().vectors, 0);
        Ok(())
    })
    .unwrap_err();
}
#[test]
fn wrong_thread_stale_tokens_and_reentrancy_fail_closed() {
    let mut store = FakeStore::default();
    let cfg = config();
    let mut slot = StoreSlot { store: &mut store };
    let scope = StoreScope::enter(&mut slot, &cfg, 10).unwrap();
    let context = scope.context.clone();
    let moved = context.clone();
    let error = thread::spawn(move || with_store(&moved, |_| Ok(())))
        .join()
        .unwrap()
        .unwrap_err();
    assert!(error.message.contains("wrong thread"));
    drop(scope);
    let stale = OperationContext {
        state: Arc::new(Mutex::new(OperationState::default())),
        ..context
    };
    assert!(with_store(&stale, |_| Ok(()))
        .unwrap_err()
        .message
        .contains("token"));
    let mut slot = StoreSlot { store: &mut store };
    let scope = StoreScope::enter(&mut slot, &cfg, 10).unwrap();
    let nested_context = scope.context.clone();
    let error =
        with_store(&scope.context, |_| with_store(&nested_context, |_| Ok(()))).unwrap_err();
    assert!(error.message.contains("reentrant"));
    drop(scope);
    let mut second = FakeStore::default();
    let mut slot = StoreSlot { store: &mut store };
    let scope = StoreScope::enter(&mut slot, &cfg, 10).unwrap();
    let mut other = StoreSlot { store: &mut second };
    assert!(StoreScope::enter(&mut other, &cfg, 10)
        .err()
        .unwrap()
        .message
        .contains("nested"));
    drop(scope);
    assert!(ACTIVE.with(|active| active.get().is_none()));
}
#[test]
fn unexpected_pending_cancels_inside_the_live_scope() {
    struct PendingGuard {
        context: OperationContext,
        dropped: Arc<std::sync::atomic::AtomicBool>,
    }
    impl Future for PendingGuard {
        type Output = ();
        fn poll(self: std::pin::Pin<&mut Self>, _: &mut TaskContext<'_>) -> Poll<()> {
            Poll::Pending
        }
    }
    impl Drop for PendingGuard {
        fn drop(&mut self) {
            assert!(with_store(&self.context, |store| store.start_points()).is_ok());
            self.dropped.store(true, Ordering::Relaxed);
        }
    }
    let mut store = FakeStore::default();
    let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let error = operation(&mut store, &config(), |_, context| {
        drive(PendingGuard {
            context: context.clone(),
            dropped: dropped.clone(),
        })
    })
    .unwrap_err();
    assert!(error.message.contains("suspended"));
    assert!(dropped.load(Ordering::Relaxed));
    assert!(ACTIVE.with(|active| active.get().is_none()));
    operation(&mut store, &config(), |_, _| Ok(())).unwrap();
}
#[test]
fn guard_drop_marks_failure_without_accessing_sql() {
    let mut store = FakeStore::default();
    let error = operation(&mut store, &config(), |_, context| {
        let guard = drive(Provider.set_element(context, &1, &[1.0, 0.0]))?.map_err(index_error)?;
        drop(guard);
        Ok(())
    })
    .unwrap_err();
    assert!(error.message.contains("rollback"));
    assert_eq!(store.calls, vec!["allocate"]);
    let mut store = FakeStore::default();
    operation(&mut store, &config(), |_, context| {
        let guard = drive(Provider.set_element(context, &1, &[1.0, 0.0]))?.map_err(index_error)?;
        drive(guard.complete())?;
        Ok(())
    })
    .unwrap();
}
#[test]
fn cached_accessor_cannot_continue_after_scope_expiry() {
    let mut store = populated();
    let mut accessor: Accessor<'static> = operation(&mut store, &config(), |_, context| {
        let mut accessor = Accessor::new(context, &[7.0, 0.0], None).map_err(index_error)?;
        accessor.distance(7).map_err(index_error)?;
        Ok(accessor)
    })
    .unwrap();
    assert!(accessor
        .distance(7)
        .unwrap_err()
        .to_string()
        .contains("token"));
    assert!(ACTIVE.with(|active| active.get().is_none()));
}
#[test]
fn borrowed_connection_owner_does_not_need_send() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Provider>();
    send_sync::<OperationContext>();
    send_sync::<Accessor<'static>>();
    send_sync::<PruneAccessor>();
    send_sync::<InsertGuard>();
}
#[test]
fn postprocessor_counts_only_successful_copies() {
    let mut store = populated();
    operation(&mut store, &config(), |_, context| {
        let mut accessor = Accessor::new(context, &[7.0, 0.0], None).map_err(index_error)?;
        let candidates = [
            Neighbor::new(0, 0.0),
            Neighbor::new(7, 0.0),
            Neighbor::new(8, 1.0),
        ];
        let mut ids = [0u64; 1];
        let mut distances = [0.0; 1];
        let mut output =
            diskann::graph::search_output_buffer::IdDistance::new(&mut ids, &mut distances);
        assert_eq!(
            drive(Output.post_process(
                &mut accessor,
                &[7.0, 0.0],
                candidates.into_iter(),
                &mut output
            ))?
            .map_err(index_error)?,
            1
        );
        assert_eq!(
            drive(Output.post_process(
                &mut accessor,
                &[7.0, 0.0],
                candidates.into_iter(),
                &mut output
            ))?
            .map_err(index_error)?,
            0
        );
        assert_eq!(ids, [7]);
        let mut ids = [];
        let mut distances = [];
        let mut empty =
            diskann::graph::search_output_buffer::IdDistance::new(&mut ids, &mut distances);
        assert_eq!(
            drive(Output.post_process(
                &mut accessor,
                &[7.0, 0.0],
                candidates.into_iter(),
                &mut empty
            ))?
            .map_err(index_error)?,
            0
        );
        Ok(())
    })
    .unwrap();
}
#[test]
fn input_validation_never_mutates_storage() {
    let mut store = FakeStore::default();
    assert!(insert(&mut store, &config(), 1, &[f32::NAN, 0.0]).is_err());
    assert!(insert(&mut store, &config(), 1, &[1.0]).is_err());
    assert!(insert(&mut store, &config(), u64::MAX, &[1.0, 0.0]).is_err());
    assert!(store.calls.is_empty());
    assert_eq!(config().core_config().unwrap().max_degree().get(), 10);
}

fn reachable(store: &FakeStore) -> HashSet<u64> {
    let mut seen = HashSet::new();
    let mut work = vec![0];
    while let Some(id) = work.pop() {
        if seen.insert(id) {
            work.extend_from_slice(&store.nodes[&id].neighbors);
        }
    }
    seen
}

#[test]
fn bounded_batches_l2_cosine_reachability_and_canonical_input() {
    for distance in [DistanceType::L2, DistanceType::Cosine] {
        for width in [1usize, 8, 32] {
            let mut cfg = config();
            cfg.distance = distance;
            cfg.search_l = 128;
            let mut store = FakeStore::default();
            insert(&mut store, &cfg, 1, &[1.0, 0.0]).unwrap();
            for chunk in 0..2 {
                let ids: Vec<u64> = (0..width).map(|i| 2 + (chunk * width + i) as u64).collect();
                let encoded: Vec<f32> = ids
                    .iter()
                    .flat_map(|id| {
                        if distance == DistanceType::Cosine {
                            let angle = (*id as f32) * 0.04;
                            [angle.cos(), angle.sin()]
                        } else {
                            [*id as f32, (*id % 3) as f32]
                        }
                    })
                    .collect();
                insert_batch(&mut store, &cfg, ids.clone(), encoded.clone()).unwrap();
                let seen = reachable(&store);
                for (i, id) in ids.iter().enumerate() {
                    let internal = store.mapping[id];
                    assert!(
                        seen.contains(&internal),
                        "{distance:?} width={width} chunk={chunk} id={id}"
                    );
                    assert_eq!(store.nodes[&internal].vector, encoded[i * 2..i * 2 + 2]);
                    assert!(store.nodes[&internal].vector.iter().all(|x| x.is_finite()));
                }
                assert!(ACTIVE.with(|active| active.get().is_none()));
            }
            let query = store.nodes[&store.mapping[&2]].vector.clone();
            assert_eq!(
                search(&mut store, &cfg, &query, 1, None, None).unwrap()[0].rowid,
                2
            );
        }
    }
}

#[test]
fn batch_input_and_workspace_admission_precede_store_callbacks() {
    let mut store = FakeStore::default();
    insert_batch(&mut store, &config(), vec![], vec![]).unwrap();
    for (ids, values) in [
        (vec![1], vec![f32::NAN, 0.0]),
        (vec![1], vec![1.0]),
        (vec![1, 1], vec![0.0; 4]),
        (vec![u64::MAX], vec![0.0; 2]),
        ((1..=33).collect(), vec![0.0; 66]),
    ] {
        assert!(insert_batch(&mut store, &config(), ids, values).is_err());
    }
    let mut cfg = config();
    cfg.limits.max_vector_bytes = 4096;
    assert!(insert_batch(&mut store, &cfg, vec![1], vec![0.0; 2])
        .unwrap_err()
        .message
        .contains("scratch"));
    // Capacity, not just length, participates in admission.
    let mut values = Vec::with_capacity(1024 * 1024);
    values.extend_from_slice(&[0.0, 0.0]);
    cfg.limits.max_vector_bytes = config().scratch_bytes(10).unwrap() + 2 * 1024 * 1024;
    assert!(insert_batch(&mut store, &cfg, vec![1], values).is_err());
    assert!(store.calls.is_empty());
}

#[test]
fn visible_ambient_runtime_and_nested_scope_rejected_without_writes() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let mut store = FakeStore::default();
    {
        let _entered = runtime.enter();
        assert!(check_batch_execution_context()
            .unwrap_err()
            .message
            .contains("ambient"));
        assert!(insert_batch(&mut store, &config(), vec![1], vec![0.0; 2]).is_err());
    }
    runtime.block_on(async {
        assert!(check_batch_execution_context().is_err());
        assert!(insert_batch(&mut store, &config(), vec![1], vec![0.0; 2]).is_err());
    });
    assert!(store.calls.is_empty());
    operation(&mut store, &config(), |_, _| {
        assert!(check_batch_execution_context()
            .unwrap_err()
            .message
            .contains("nested"));
        Ok(())
    })
    .unwrap();
    assert!(check_batch_execution_context().is_ok());
}

#[test]
fn batch_late_sql_failure_keeps_original_error_and_drains() {
    let original = populated();
    let mut success = original.clone();
    insert_batch(
        &mut success,
        &config(),
        vec![21, 22, 23],
        vec![21.0, 0.0, 22.0, 0.0, 23.0, 0.0],
    )
    .unwrap();
    for method in [
        "allocate",
        "start_points",
        "vector",
        "neighbors",
        "set_neighbors",
    ] {
        let count = success.calls.iter().filter(|call| **call == method).count();
        assert!(count > 0);
        for nth in [1, count] {
            let mut store = original.clone();
            store.fail_at = Some((method, nth));
            let error = insert_batch(
                &mut store,
                &config(),
                vec![21, 22, 23],
                vec![21.0, 0.0, 22.0, 0.0, 23.0, 0.0],
            )
            .unwrap_err();
            assert_eq!(error.code, 778, "{method} nth={nth}: {error}");
            assert!(error.message.contains(method));
            assert!(ACTIVE.with(|active| active.get().is_none()));
            // SQL owner, not guards, owns rollback; model restoration here.
            store = original.clone();
            insert_batch(&mut store, &config(), vec![24], vec![24.0, 0.0]).unwrap();
        }
    }
}

#[test]
fn batch_visits_are_shared_and_failure_is_bounded() {
    let mut store = populated();
    let mut cfg = config();
    cfg.limits.max_visits = 64;
    let error = insert_batch(
        &mut store,
        &cfg,
        (21..=52).collect(),
        (21..=52).flat_map(|id| [id as f32, 0.0]).collect(),
    )
    .unwrap_err();
    assert_eq!(error.code, crate::ffi::SQLITE_TOOBIG as i32);
    assert!(error.message.contains("visit"));
    assert!(store.calls.len() < 1000);
    assert!(ACTIVE.with(|active| active.get().is_none()));
}

#[test]
fn private_runtime_destroys_unpolled_tasks_guards_and_vectors_inside_scope() {
    use diskann::provider::ExecutionContext;
    struct Capture(OperationContext);
    impl Drop for Capture {
        fn drop(&mut self) {
            assert_eq!(thread::current().id(), self.0.thread);
            assert!(ACTIVE.with(|active| active
                .get()
                .is_some_and(|frame| frame.token == self.0.token)));
        }
    }
    let mut store = FakeStore::default();
    operation(&mut store, &config(), |_, context| {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let guard = drive(Provider.set_element(context, &1, &[1.0, 0.0]))?.map_err(index_error)?;
        let vector = context.copy_vector(&[1.0, 0.0])?;
        let capture = Capture(context.clone());
        let entered = runtime.enter();
        let handle = tokio::spawn(context.wrap_spawn(async move {
            let _owned = (guard, vector, capture);
            std::future::pending::<()>().await;
        }));
        assert_eq!(context.lock().tasks, 1);
        drop(handle); // detached but still owned by this private runtime
        drop(entered);
        drop(runtime); // cancels even unpolled task, while ACTIVE is live
        assert_eq!(context.lock().tasks, 0);
        assert_eq!(context.lock().guards, 0);
        assert_eq!(context.lock().vectors, 0);
        assert!(context.lock().incomplete_insertion);
        Ok(())
    })
    .unwrap_err();
    assert!(ACTIVE.with(|active| active.get().is_none()));
}

#[derive(Clone)]
struct FailingBatchStrategy {
    phase: &'static str,
    seed_calls: Arc<std::sync::atomic::AtomicUsize>,
}
impl glue::PruneStrategy<Provider> for FailingBatchStrategy {
    type PruneAccessor<'a> = PruneAccessor;
    type PruneAccessorError = ANNError;
    fn prune_accessor<'a>(
        &'a self,
        provider: &'a Provider,
        context: &'a OperationContext,
        capacity: usize,
    ) -> ANNResult<PruneAccessor> {
        // Candidate uses seeded accessor; bootstrap uses unseeded capacity750,
        // outgoing assignment uses capacity0.
        if (self.phase == "bootstrap" && capacity != 0)
            || (self.phase == "outgoing" && capacity == 0)
        {
            return Err(ANNError::message(format!("pure {} failure", self.phase)));
        }
        glue::PruneStrategy::prune_accessor(&Strategy, provider, context, capacity)
    }
}
impl<'a> glue::InsertStrategy<'a, Provider, &'a [f32]> for FailingBatchStrategy {
    type SearchAccessorError = ANNError;
    type SearchAccessor = Accessor<'a>;
    type PruneStrategy = Self;
    fn insert_search_accessor(
        &'a self,
        _: &'a Provider,
        context: &'a OperationContext,
        query: &'a [f32],
    ) -> ANNResult<Accessor<'a>> {
        Accessor::new(context, query, None)
    }
    fn prune_strategy(&self) -> Self {
        self.clone()
    }
}
impl glue::MultiInsertStrategy<Provider, FlatBatch> for FailingBatchStrategy {
    type Seed = ();
    type FinishError = ANNError;
    type PruneStrategy = Self;
    type InsertStrategy = Self;
    fn insert_strategy(&self) -> Self {
        self.clone()
    }
    fn finish<Itr>(
        &self,
        _: &Provider,
        _: &OperationContext,
        _: &Arc<FlatBatch>,
        _: Itr,
    ) -> impl Future<Output = ANNResult<()>> + Send
    where
        Itr: ExactSizeIterator<Item = u64> + Send,
    {
        ready(if self.phase == "finish" {
            Err(ANNError::message("pure finish failure"))
        } else {
            Ok(())
        })
    }
    fn seeded_prune_accessor<'a>(
        &'a self,
        provider: &'a Provider,
        context: &'a OperationContext,
        _: &'a (),
        capacity: usize,
    ) -> ANNResult<PruneAccessor> {
        let call = self.seed_calls.fetch_add(1, Ordering::Relaxed);
        if (self.phase == "candidate" && call == 0) || (self.phase == "backedge" && call >= 4) {
            return Err(ANNError::message(format!("pure {} failure", self.phase)));
        }
        glue::PruneStrategy::prune_accessor(&Strategy, provider, context, capacity)
    }
}

#[test]
fn upstream_pure_phase_errors_are_not_success_or_guard_fallback() {
    for phase in ["finish", "candidate", "bootstrap", "outgoing", "backedge"] {
        let mut store = FakeStore::default();
        insert(&mut store, &config(), 1, &[1.0, 0.0]).unwrap();
        let cfg = config();
        let core = Builder::new_with(
            8,
            MaxDegree::default_slack(),
            32,
            PruneKind::TriangleInequality,
            |b| {
                b.alpha(1.2)
                    .max_minibatch_par(4)
                    .intra_batch_candidates(IntraBatchCandidates::None);
            },
        )
        .build()
        .unwrap();
        let mut slot = StoreSlot { store: &mut store };
        let scope = StoreScope::enter(&mut slot, &cfg, 10).unwrap();
        let context = scope.context.clone();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let index = Arc::new(DiskANNIndex::new(core, Provider, None));
        let strategy = FailingBatchStrategy {
            phase,
            seed_calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        };
        let result = runtime.block_on(index.multi_insert::<FailingBatchStrategy, FlatBatch>(
            strategy,
            &context,
            Arc::new(FlatBatch {
                encoded: (2..=17).flat_map(|i| [i as f32, 0.0]).collect(),
                dimension: 2,
            }),
            (2..=17).collect::<Vec<u64>>().into(),
        ));
        drop(runtime);
        drop(index);
        assert_eq!(context.lock().tasks, 0, "{phase}");
        assert_eq!(context.lock().guards, 0, "{phase}");
        assert_eq!(context.lock().vectors, 0, "{phase}");
        assert!(context.lock().incomplete_insertion, "{phase}");
        assert!(
            context.lock().fatal.is_none(),
            "{phase} is not a SQL/store failure"
        );
        let error = finish_operation(&context, result.map_err(index_error), true).unwrap_err();
        assert!(
            error.message.contains(&format!("pure {phase} failure")),
            "{error}"
        );
        drop(scope);
        assert!(ACTIVE.with(|active| active.get().is_none()));
    }
}
