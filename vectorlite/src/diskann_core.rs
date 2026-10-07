//! Connection-local DiskANN adapter.
#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::future::{ready, Future};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context as TaskContext, Poll, Waker};
use std::thread::{self, ThreadId};

use diskann::graph::config::{Builder, MaxDegree, PruneKind};
use diskann::graph::glue::{self, FilteredAccessor, HybridPredicate, SearchAccessor};
use diskann::graph::search::{InlineFilterSearch, Knn};
use diskann::graph::{workingset, AdjacencyList, DiskANNIndex};
use diskann::neighbor::Neighbor;
use diskann::provider::{
    self, DataProvider, ElementStatus, HasId, NeighborAccessor, NeighborAccessorMut,
};
use diskann::{ANNError, ANNResult};
use diskann_utils::Reborrow;
use diskann_vector::DistanceFunction;

use crate::core::SearchResult;
use crate::index_error::IndexError;
use crate::ops;
use crate::vector_space::DistanceType;

type StoreResult<T> = Result<T, IndexError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NodeState {
    Live,
    Deleted,
    Frozen,
}

/// No Send/Sync requirement is imposed on the host's SQLite owner. The store
/// checks blob/adjacency lengths before allocating snapshots. allocate
/// bootstraps frozen0; deletion retires rowids but retains node vectors.
pub(crate) trait GraphStore {
    fn allocate(&mut self, rowid: u64, vector: &[f32]) -> StoreResult<u64>;
    fn internal_id(&mut self, rowid: u64) -> StoreResult<u64>;
    fn rowid(&mut self, id: u64) -> StoreResult<Option<u64>>;
    fn state(&mut self, id: u64) -> StoreResult<NodeState>;
    fn vector(&mut self, id: u64) -> StoreResult<Vec<f32>>;
    fn neighbors(&mut self, id: u64) -> StoreResult<Vec<u64>>;
    fn set_neighbors(&mut self, id: u64, neighbors: &[u64]) -> StoreResult<()>;
    fn mark_delete(&mut self, rowid: u64) -> StoreResult<()>;
    fn start_points(&mut self) -> StoreResult<Vec<u64>>;
}

#[derive(Clone, Debug)]
pub(crate) struct ResourceLimits {
    pub(crate) max_visits: usize,
    /// Simultaneous owned snapshots, including queries and prune vectors.
    pub(crate) max_cached_vectors: usize,
    /// Remaining whole-core workspace after the SQL owner reserves its buffers.
    /// Upfront scratch admission and live vector payload/container leases both
    /// consume this budget; SQLite's own page cache is outside it.
    pub(crate) max_vector_bytes: usize,
}
impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_visits: 100_000,
            max_cached_vectors: 2048,
            max_vector_bytes: 64 * 1024 * 1024,
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) struct GraphConfig {
    pub(crate) dimension: usize,
    pub(crate) distance: DistanceType,
    /// Logical pruned target; physical degree is floor(target * 1.3).
    pub(crate) max_degree: usize,
    pub(crate) construction_l: usize,
    pub(crate) search_l: usize,
    pub(crate) alpha: f32,
    pub(crate) limits: ResourceLimits,
}
fn limit(message: &str) -> IndexError {
    IndexError::with_code(crate::ffi::SQLITE_TOOBIG as i32, message)
}
fn allocation_error() -> IndexError {
    IndexError::with_code(
        crate::ffi::SQLITE_NOMEM as i32,
        "DiskANN snapshot allocation failed",
    )
}
impl GraphConfig {
    fn core_config(&self) -> StoreResult<diskann::graph::Config> {
        let bytes = self
            .dimension
            .checked_mul(4)
            .filter(|_| self.dimension != 0)
            .ok_or_else(|| IndexError::new("invalid DiskANN vector dimension"))?;
        if !matches!(self.distance, DistanceType::L2 | DistanceType::Cosine) {
            return Err(IndexError::new("DiskANN supports only float32 L2/cosine"));
        }
        if !self.alpha.is_finite() || self.alpha < 1.0 {
            return Err(IndexError::new("DiskANN alpha must be finite and >= 1"));
        }
        if self.search_l == 0
            || self.limits.max_visits == 0
            || self.limits.max_cached_vectors == 0
            || self.limits.max_vector_bytes < bytes
            || self.construction_l > self.limits.max_visits
            || self.search_l > self.limits.max_visits
            || self.max_degree > self.limits.max_visits
        {
            return Err(limit("invalid DiskANN resource/search limits"));
        }
        Builder::new_with(
            self.max_degree,
            MaxDegree::default_slack(),
            self.construction_l,
            PruneKind::TriangleInequality,
            |b| {
                b.alpha(self.alpha);
            },
        )
        .build()
        .map_err(|e| IndexError::new(e.to_string()))
    }
    fn scratch_bytes(&self, physical_degree: usize) -> StoreResult<usize> {
        // Pinned 0.60: fixed best queue + visited HashSet; insert additionally
        // records visited Neighbors, inline search collects matching Neighbors.
        // Account geometric Vec/hash capacities, their initial degree*L reserve,
        // default750 prune states, and OneHop's degree-squared topology work.
        // Upgrades must re-audit these bounds; no bulk/paged/resize APIs are used.
        let search_l = self
            .search_l
            .max(self.construction_l)
            .checked_add(1)
            .ok_or_else(|| limit("DiskANN scratch estimate overflow"))?;
        let visits = self
            .limits
            .max_visits
            .checked_add(physical_degree)
            .and_then(|n| n.checked_mul(128));
        let initial = physical_degree
            .checked_mul(search_l)
            .and_then(|n| n.checked_mul(256));
        let topology = physical_degree
            .checked_mul(physical_degree)
            .and_then(|n| n.checked_mul(128));
        let bytes = visits
            .and_then(|a| initial.and_then(|b| a.checked_add(b)))
            .and_then(|a| topology.and_then(|b| a.checked_add(b)))
            .and_then(|a| a.checked_add(750 * 128 + 8192))
            .ok_or_else(|| limit("DiskANN scratch estimate overflow"))?;
        let minimum_vector = self
            .dimension
            .checked_mul(4)
            .and_then(|n| n.checked_add(vector_metadata_bytes()))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| limit("DiskANN vector estimate overflow"))?;
        if bytes
            .checked_add(minimum_vector)
            .is_none_or(|n| n > self.limits.max_vector_bytes)
        {
            return Err(limit("DiskANN algorithm scratch exceeds workspace limit"));
        }
        Ok(bytes)
    }
    fn validate_vector(&self, vector: &[f32]) -> StoreResult<()> {
        if vector.len() != self.dimension || vector.iter().any(|x| !x.is_finite()) {
            return Err(IndexError::new(
                "DiskANN vector has wrong dimension or non-finite values",
            ));
        }
        Ok(())
    }
}
#[derive(Default)]
struct OperationState {
    fatal: Option<IndexError>,
    visits: usize,
    vectors: usize,
    bytes: usize,
}
#[derive(Clone)]
struct OperationContext {
    token: u64,
    thread: ThreadId,
    config: GraphConfig,
    physical_degree: usize,
    state: Arc<Mutex<OperationState>>,
}
impl provider::ExecutionContext for OperationContext {}
impl OperationContext {
    fn lock(&self) -> MutexGuard<'_, OperationState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
    fn fail(&self, error: IndexError) -> IndexError {
        self.lock()
            .fatal
            .get_or_insert_with(|| error.clone())
            .clone()
    }
    fn check(&self) -> StoreResult<()> {
        if thread::current().id() != self.thread {
            return Err(self.fail(IndexError::new(
                "DiskANN operation used on the wrong thread",
            )));
        }
        if let Some(error) = self.lock().fatal.clone() {
            return Err(error);
        }
        if !ACTIVE.with(|active| active.get().is_some_and(|frame| frame.token == self.token)) {
            return Err(self.fail(IndexError::new(
                "expired or mismatched DiskANN operation token",
            )));
        }
        Ok(())
    }
    fn visit(&self) -> StoreResult<()> {
        self.check()?;
        let mut state = self.lock();
        if state.visits >= self.config.limits.max_visits {
            let error = limit("DiskANN visit limit exceeded");
            state.fatal.get_or_insert_with(|| error.clone());
            return Err(error);
        }
        state.visits += 1;
        Ok(())
    }
    fn reserve_vector(&self) -> StoreResult<VectorLease> {
        self.check()?;
        let bytes = self.config.dimension * 4 + vector_metadata_bytes();
        let mut state = self.lock();
        if state.vectors >= self.config.limits.max_cached_vectors
            || bytes
                > self
                    .config
                    .limits
                    .max_vector_bytes
                    .saturating_sub(state.bytes)
        {
            let error = limit("DiskANN vector working-set limit exceeded");
            state.fatal.get_or_insert_with(|| error.clone());
            return Err(error);
        }
        state.vectors += 1;
        state.bytes += bytes;
        Ok(VectorLease {
            context: self.clone(),
            bytes,
        })
    }
    fn copy_vector(&self, source: &[f32]) -> StoreResult<OwnedVector> {
        self.config
            .validate_vector(source)
            .map_err(|e| self.fail(e))?;
        let lease = self.reserve_vector()?;
        let mut data = Vec::new();
        data.try_reserve_exact(source.len())
            .map_err(|_| self.fail(allocation_error()))?;
        data.extend_from_slice(source);
        Ok(OwnedVector {
            data,
            _lease: lease,
        })
    }
    fn fetch_vector(&self, id: u64) -> StoreResult<OwnedVector> {
        self.visit()?;
        let lease = self.reserve_vector()?;
        let data = with_store(self, |store| store.vector(id))?;
        self.config
            .validate_vector(&data)
            .map_err(|e| self.fail(e))?;
        Ok(OwnedVector {
            data,
            _lease: lease,
        })
    }
    fn neighbors(&self, id: u64) -> StoreResult<Vec<u64>> {
        let neighbors = with_store(self, |store| store.neighbors(id))?;
        if neighbors.len() > self.physical_degree
            || neighbors
                .iter()
                .enumerate()
                .any(|(i, id)| neighbors[..i].contains(id))
        {
            return Err(self.fail(IndexError::new("corrupt DiskANN adjacency list")));
        }
        Ok(neighbors)
    }
}
struct VectorLease {
    context: OperationContext,
    bytes: usize,
}
impl Drop for VectorLease {
    fn drop(&mut self) {
        let mut state = self.context.lock();
        state.vectors = state.vectors.saturating_sub(1);
        state.bytes = state.bytes.saturating_sub(self.bytes);
    }
}
struct OwnedVector {
    data: Vec<f32>,
    _lease: VectorLease,
}
fn vector_metadata_bytes() -> usize {
    // Allow up to four hash buckets per live item (rounding/load factor),
    // inline OwnedVector/lease/context storage, and allocator metadata.
    std::mem::size_of::<OwnedVector>() * 4 + 128
}
impl<'a> Reborrow<'a> for OwnedVector {
    type Target = &'a [f32];
    fn reborrow(&'a self) -> Self::Target {
        &self.data
    }
}
struct StoreSlot<'a> {
    store: &'a mut dyn GraphStore,
}
#[derive(Clone, Copy)]
struct ActiveStore {
    token: u64,
    pointer: NonNull<()>,
    busy: bool,
}
thread_local! { static ACTIVE: Cell<Option<ActiveStore>> = const { Cell::new(None) }; }
static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);
/// Lifetime retains the exclusive stack-slot borrow until TLS removal;
/// Rc's marker makes the lease !Send/!Sync. Tokens never contain its pointer.
struct StoreScope<'slot> {
    context: OperationContext,
    _slot: PhantomData<&'slot mut ()>,
    _local: PhantomData<Rc<()>>,
}
impl<'slot> StoreScope<'slot> {
    fn enter(
        slot: &'slot mut StoreSlot<'_>,
        config: &GraphConfig,
        physical_degree: usize,
    ) -> StoreResult<Self> {
        let token = NEXT_TOKEN
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| IndexError::new("DiskANN operation token exhausted"))?;
        let context = OperationContext {
            token,
            thread: thread::current().id(),
            config: config.clone(),
            physical_degree,
            state: Arc::new(Mutex::new(OperationState {
                bytes: config.scratch_bytes(physical_degree)?,
                ..OperationState::default()
            })),
        };
        ACTIVE.with(|active| {
            if active.get().is_some() {
                return Err(IndexError::new("nested DiskANN operation is not supported"));
            }
            active.set(Some(ActiveStore {
                token,
                pointer: NonNull::from(slot).cast(),
                busy: false,
            }));
            Ok(())
        })?;
        Ok(Self {
            context,
            _slot: PhantomData,
            _local: PhantomData,
        })
    }
}
impl Drop for StoreScope<'_> {
    fn drop(&mut self) {
        ACTIVE.with(|active| {
            if active.get().is_some_and(|a| a.token == self.context.token) {
                active.set(None);
            }
        });
    }
}
struct BusyScope(u64);
impl Drop for BusyScope {
    fn drop(&mut self) {
        ACTIVE.with(|active| {
            if let Some(mut a) = active.get().filter(|a| a.token == self.0) {
                a.busy = false;
                active.set(Some(a));
            }
        });
    }
}
fn with_store<T>(
    context: &OperationContext,
    f: impl FnOnce(&mut dyn GraphStore) -> StoreResult<T>,
) -> StoreResult<T> {
    context.check()?;
    let pointer = ACTIVE
        .with(|active| {
            let mut frame = active
                .get()
                .filter(|a| a.token == context.token)
                .ok_or_else(|| IndexError::new("expired or mismatched DiskANN operation token"))?;
            if frame.busy {
                return Err(IndexError::new("reentrant DiskANN store access"));
            }
            frame.busy = true;
            active.set(Some(frame));
            Ok(frame.pointer)
        })
        .map_err(|e| context.fail(e))?;
    let _busy = BusyScope(context.token);
    // SAFETY: StoreScope borrows the live stack slot exclusively and prevents
    // movement until TLS removal. Exact token and callback thread were checked;
    // busy excludes recursive mutable aliases. No other code touches slot/store
    // during this closure, and no borrowed data can escape the closure bound.
    let slot = unsafe { &mut *pointer.cast::<StoreSlot<'_>>().as_ptr() };
    f(slot.store).map_err(|e| context.fail(e))
}
fn drive<F: Future>(future: F) -> StoreResult<F::Output> {
    let mut context = TaskContext::from_waker(Waker::noop());
    let mut future = std::pin::pin!(future);
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => Ok(value),
        Poll::Pending => Err(IndexError::new(
            "DiskANN single-operation future unexpectedly suspended",
        )),
    }
}
fn operation<T>(
    store: &mut dyn GraphStore,
    config: &GraphConfig,
    f: impl FnOnce(&DiskANNIndex<Provider>, &OperationContext) -> StoreResult<T>,
) -> StoreResult<T> {
    let core_config = config.core_config()?;
    let physical_degree = core_config.max_degree().get();
    let mut slot = StoreSlot { store };
    let scope = StoreScope::enter(&mut slot, config, physical_degree)?;
    let index = DiskANNIndex::new(core_config, Provider, None);
    let result = f(&index, &scope.context);
    let fatal = scope.context.lock().fatal.clone();
    match fatal {
        Some(error) => Err(error),
        None => result,
    }
}
fn ann(error: IndexError) -> ANNError {
    ANNError::message(error.to_string())
}
fn index_error(error: ANNError) -> IndexError {
    IndexError::new(error.to_string())
}
struct Provider;
struct InsertGuard {
    context: OperationContext,
    id: u64,
    complete: bool,
}
impl Drop for InsertGuard {
    fn drop(&mut self) {
        if !self.complete {
            self.context.fail(IndexError::new(
                "DiskANN insertion did not complete; rollback required",
            ));
        }
    }
}
impl provider::Guard for InsertGuard {
    type Id = u64;
    fn id(&self) -> u64 {
        self.id
    }
    fn complete(mut self) -> impl Future<Output = ()> + Send {
        // Completion cannot return errors in the upstream trait; keep scope
        // violations fatal in the shared operation state instead.
        let _ = self.context.check();
        self.complete = true;
        ready(())
    }
}
impl DataProvider for Provider {
    type Context = OperationContext;
    type InternalId = u64;
    type ExternalId = u64;
    type Error = ANNError;
    type Guard = InsertGuard;
    fn to_internal_id(&self, context: &OperationContext, id: &u64) -> ANNResult<u64> {
        with_store(context, |store| store.internal_id(*id)).map_err(ann)
    }
    fn to_external_id(&self, context: &OperationContext, id: u64) -> ANNResult<u64> {
        with_store(context, |store| store.rowid(id))
            .map_err(ann)?
            .ok_or_else(|| ANNError::message("node has no live rowid"))
    }
}
impl provider::SetElement<&[f32]> for Provider {
    type SetError = ANNError;
    fn set_element(
        &self,
        context: &OperationContext,
        rowid: &u64,
        vector: &[f32],
    ) -> impl Future<Output = ANNResult<InsertGuard>> + Send {
        ready(
            with_store(context, |store| store.allocate(*rowid, vector))
                .map(|id| InsertGuard {
                    context: context.clone(),
                    id,
                    complete: false,
                })
                .map_err(ann),
        )
    }
}
impl provider::Delete for Provider {
    fn delete(
        &self,
        context: &OperationContext,
        rowid: &u64,
    ) -> impl Future<Output = ANNResult<()>> + Send {
        ready(with_store(context, |store| store.mark_delete(*rowid)).map_err(ann))
    }
    fn release(
        &self,
        context: &OperationContext,
        _: u64,
    ) -> impl Future<Output = ANNResult<()>> + Send {
        ready(context.check().map_err(ann))
    }
    fn status_by_internal_id(
        &self,
        context: &OperationContext,
        id: u64,
    ) -> impl Future<Output = ANNResult<ElementStatus>> + Send {
        ready(
            with_store(context, |store| store.state(id))
                .map(|state| match state {
                    NodeState::Deleted => ElementStatus::Deleted,
                    NodeState::Live | NodeState::Frozen => ElementStatus::Valid,
                })
                .map_err(ann),
        )
    }
    fn status_by_external_id(
        &self,
        context: &OperationContext,
        rowid: &u64,
    ) -> impl Future<Output = ANNResult<ElementStatus>> + Send {
        ready(
            with_store(context, |store| {
                let id = store.internal_id(*rowid)?;
                store.state(id)
            })
            .map(|state| {
                if state == NodeState::Deleted {
                    ElementStatus::Deleted
                } else {
                    ElementStatus::Valid
                }
            })
            .map_err(ann),
        )
    }
}
#[derive(Clone)]
struct Computer(OperationContext);
impl DistanceFunction<&[f32], &[f32], f32> for Computer {
    fn evaluate_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if self.0.check().is_err() {
            return f32::MAX;
        }
        let distance = match self.0.config.distance {
            DistanceType::L2 => ops::l2_sq_f32(a, b),
            DistanceType::Cosine | DistanceType::InnerProduct => ops::ip_dist_f32(a, b),
        };
        if !distance.is_finite() {
            self.0.fail(IndexError::new("non-finite DiskANN distance"));
            // The trait cannot fail. Keep numerics defined and abort via latch.
            f32::MAX
        } else {
            distance
        }
    }
}
struct Accessor<'a> {
    context: OperationContext,
    query: OwnedVector,
    starts: Vec<u64>,
    filter: Option<&'a HashSet<u64>>,
    cache: HashMap<u64, OwnedVector>,
}
impl<'a> Accessor<'a> {
    fn new(
        context: &OperationContext,
        query: &[f32],
        filter: Option<&'a HashSet<u64>>,
    ) -> ANNResult<Self> {
        let starts = with_store(context, |store| store.start_points()).map_err(ann)?;
        if starts.as_slice() != [0] && !starts.is_empty() {
            return Err(ann(
                context.fail(IndexError::new("invalid frozen DiskANN entry point"))
            ));
        }
        let query = context.copy_vector(query).map_err(ann)?;
        Ok(Self {
            context: context.clone(),
            query,
            starts,
            filter,
            cache: HashMap::new(),
        })
    }
    fn distance(&mut self, id: u64) -> ANNResult<f32> {
        if !self.cache.contains_key(&id) {
            let cap = self
                .context
                .config
                .limits
                .max_cached_vectors
                .min(32)
                .saturating_sub(1);
            if cap == 0 || self.cache.len() >= cap {
                // Release hash buckets together with their metadata leases.
                self.cache = HashMap::new();
            }
            let value = self.context.fetch_vector(id).map_err(ann)?;
            self.cache
                .try_reserve(1)
                .map_err(|_| ann(self.context.fail(allocation_error())))?;
            self.cache.insert(id, value);
        }
        let distance = Computer(self.context.clone())
            .evaluate_similarity(&self.query.data, &self.cache[&id].data);
        self.context.check().map_err(ann)?;
        Ok(distance)
    }
    fn decision(&self, id: u64) -> ANNResult<glue::Decision<u64>> {
        let rowid = with_store(&self.context, |store| store.rowid(id)).map_err(ann)?;
        Ok(
            if rowid.is_some_and(|rowid| self.filter.is_none_or(|filter| filter.contains(&rowid))) {
                glue::Decision::accept(id)
            } else {
                glue::Decision::reject(id)
            },
        )
    }
    fn expand<I, P, F>(&mut self, ids: I, mut predicate: P, mut callback: F) -> ANNResult<()>
    where
        I: Iterator<Item = u64>,
        P: HybridPredicate<u64>,
        F: FnMut(u64, f32),
    {
        for id in ids {
            for neighbor in self.context.neighbors(id).map_err(ann)? {
                if predicate.eval_mut(&neighbor) {
                    self.context.visit().map_err(ann)?;
                    callback(neighbor, self.distance(neighbor)?);
                }
            }
        }
        Ok(())
    }
}
impl HasId for Accessor<'_> {
    type Id = u64;
}
impl SearchAccessor for Accessor<'_> {
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<u64>>> + Send {
        ready(Ok(self.starts.clone()))
    }
    fn start_point_distances<F>(
        &mut self,
        mut callback: F,
    ) -> impl Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(u64, f32) + Send,
    {
        ready((|| {
            for id in self.starts.clone() {
                self.context.visit().map_err(ann)?;
                callback(id, self.distance(id)?);
            }
            Ok(())
        })())
    }
    fn expand_beam<I, P, F>(
        &mut self,
        ids: I,
        predicate: P,
        callback: F,
    ) -> impl Future<Output = ANNResult<()>> + Send
    where
        I: Iterator<Item = u64> + Send,
        P: HybridPredicate<u64> + Send + Sync,
        F: FnMut(u64, f32) + Send,
    {
        ready(self.expand(ids, predicate, callback))
    }
}
impl FilteredAccessor for Accessor<'_> {
    fn start_point_distances<F>(
        &mut self,
        mut callback: F,
    ) -> impl Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(glue::Decision<u64>, f32) + Send,
    {
        ready((|| {
            for id in self.starts.clone() {
                self.context.visit().map_err(ann)?;
                callback(self.decision(id)?, self.distance(id)?);
            }
            Ok(())
        })())
    }
    fn num_starting_points(&self) -> impl Future<Output = ANNResult<usize>> + Send {
        ready(Ok(self.starts.len()))
    }
    fn expand_beam_filtered<I, P, F>(
        &mut self,
        ids: I,
        mut predicate: P,
        mut callback: F,
    ) -> impl Future<Output = ANNResult<()>> + Send
    where
        I: Iterator<Item = u64> + Send,
        P: HybridPredicate<u64> + Send + Sync,
        F: FnMut(glue::Decision<u64>, f32) + Send,
    {
        ready((|| {
            for id in ids {
                for neighbor in self.context.neighbors(id).map_err(ann)? {
                    // All nodes see the visited predicate; rejected nodes navigate too.
                    if predicate.eval_mut(&neighbor) {
                        self.context.visit().map_err(ann)?;
                        callback(self.decision(neighbor)?, self.distance(neighbor)?);
                    }
                }
            }
            Ok(())
        })())
    }
    fn expand_beam_accept_only<I, P, F>(
        &mut self,
        ids: I,
        mut predicate: P,
        mut callback: F,
    ) -> impl Future<Output = ANNResult<()>> + Send
    where
        I: Iterator<Item = u64> + Send,
        P: glue::Predicate<u64> + glue::PredicateMut<glue::Accept<u64>> + Send + Sync,
        F: FnMut(glue::Accept<u64>, f32) + Send,
    {
        ready((|| {
            for id in ids {
                for neighbor in self.context.neighbors(id).map_err(ann)? {
                    if predicate.eval(&neighbor) {
                        if let glue::Decision::Accept(accept) = self.decision(neighbor)? {
                            if predicate.eval_mut(&accept) {
                                self.context.visit().map_err(ann)?;
                                callback(accept, self.distance(neighbor)?);
                            }
                        }
                    }
                }
            }
            Ok(())
        })())
    }
}
struct PruneAccessor {
    context: OperationContext,
    vectors: HashMap<u64, OwnedVector>,
}
struct VectorView<'a>(&'a HashMap<u64, OwnedVector>);
impl workingset::View<u64> for VectorView<'_> {
    type ElementRef<'a> = &'a [f32];
    type Element<'a>
        = &'a [f32]
    where
        Self: 'a;
    fn get(&self, id: u64) -> Option<Self::Element<'_>> {
        self.0.get(&id).map(|v| v.data.as_slice())
    }
}
impl HasId for PruneAccessor {
    type Id = u64;
}
impl NeighborAccessor for PruneAccessor {
    fn get_neighbors(
        &mut self,
        id: u64,
        neighbors: &mut AdjacencyList<u64>,
    ) -> impl Future<Output = ANNResult<()>> + Send {
        ready(
            self.context
                .neighbors(id)
                .map(|values| {
                    *neighbors = AdjacencyList::from_iter_untrusted(values);
                })
                .map_err(ann),
        )
    }
}
impl NeighborAccessorMut for PruneAccessor {
    fn set_neighbors(
        &mut self,
        id: u64,
        neighbors: &[u64],
    ) -> impl Future<Output = ANNResult<()>> + Send {
        let result = if neighbors.len() > self.context.physical_degree {
            Err(self
                .context
                .fail(limit("DiskANN adjacency exceeds physical degree")))
        } else {
            with_store(&self.context, |store| store.set_neighbors(id, neighbors))
        };
        ready(result.map_err(ann))
    }
    fn append_vector(
        &mut self,
        id: u64,
        neighbors: &[u64],
    ) -> impl Future<Output = ANNResult<()>> + Send {
        ready(
            (|| {
                let mut current = self.context.neighbors(id)?;
                for neighbor in neighbors {
                    if !current.contains(neighbor) {
                        if current.len() == self.context.physical_degree {
                            return Err(self
                                .context
                                .fail(limit("DiskANN append exceeds physical degree")));
                        }
                        current
                            .try_reserve(1)
                            .map_err(|_| self.context.fail(allocation_error()))?;
                        current.push(*neighbor);
                    }
                }
                with_store(&self.context, |store| store.set_neighbors(id, &current))
            })()
            .map_err(ann),
        )
    }
}
impl glue::PruneAccessor for PruneAccessor {
    type Neighbors<'a> = provider::Neighbors<'a, Self>;
    type ElementRef<'a> = &'a [f32];
    type View<'a> = VectorView<'a>;
    type Distance<'a> = Computer;
    fn neighbors(&mut self) -> Self::Neighbors<'_> {
        provider::Neighbors(self)
    }
    fn fill<I>(
        &mut self,
        ids: I,
    ) -> impl Future<Output = ANNResult<(Self::View<'_>, Self::Distance<'_>)>> + Send
    where
        I: ExactSizeIterator<Item = u64> + Clone + Send + Sync,
    {
        // Do not retain empty buckets after dropping payload/metadata leases.
        self.vectors = HashMap::new();
        for id in ids {
            if !self.vectors.contains_key(&id) {
                let value = match self.context.fetch_vector(id) {
                    Ok(value) => value,
                    Err(error) => return ready(Err(ann(error))),
                };
                if self.vectors.try_reserve(1).is_err() {
                    return ready(Err(ann(self.context.fail(allocation_error()))));
                }
                self.vectors.insert(id, value);
            }
        }
        ready(Ok((
            VectorView(&self.vectors),
            Computer(self.context.clone()),
        )))
    }
}
#[derive(Clone, Copy)]
struct Strategy;
#[derive(Clone, Copy)]
struct QueryStrategy<'a> {
    filter: Option<&'a HashSet<u64>>,
}
impl<'a> glue::SearchStrategy<'a, Provider, &'a [f32]> for Strategy {
    type SearchAccessorError = ANNError;
    type SearchAccessor = Accessor<'a>;
    fn search_accessor(
        &'a self,
        _: &'a Provider,
        context: &'a OperationContext,
        query: &'a [f32],
    ) -> ANNResult<Accessor<'a>> {
        Accessor::new(context, query, None)
    }
}
impl<'a, 'filter> glue::SearchStrategy<'a, Provider, &'a [f32]> for QueryStrategy<'filter> {
    type SearchAccessorError = ANNError;
    type SearchAccessor = Accessor<'filter>;
    fn search_accessor(
        &'a self,
        _: &'a Provider,
        context: &'a OperationContext,
        query: &'a [f32],
    ) -> ANNResult<Accessor<'filter>> {
        Accessor::new(context, query, self.filter)
    }
}
impl glue::PruneStrategy<Provider> for Strategy {
    type PruneAccessor<'a> = PruneAccessor;
    type PruneAccessorError = ANNError;
    fn prune_accessor<'a>(
        &'a self,
        _: &'a Provider,
        context: &'a OperationContext,
        _: usize,
    ) -> ANNResult<PruneAccessor> {
        Ok(PruneAccessor {
            context: context.clone(),
            vectors: HashMap::new(),
        })
    }
}
impl<'a> glue::InsertStrategy<'a, Provider, &'a [f32]> for Strategy {
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
        *self
    }
}
struct Output;
impl glue::SearchPostProcess<Accessor<'_>, &[f32]> for Output {
    type Error = ANNError;
    fn post_process<I, B>(
        &self,
        accessor: &mut Accessor<'_>,
        _: &[f32],
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = ANNResult<usize>> + Send
    where
        I: Iterator<Item = Neighbor<u64>> + Send,
        B: diskann::graph::SearchOutputBuffer<u64> + Send + ?Sized,
    {
        ready((|| {
            let before = output.current_len();
            for neighbor in candidates {
                if output.size_hint() == Some(0) {
                    break;
                }
                if let glue::Decision::Accept(_) = accessor.decision(*neighbor.id())? {
                    if output.push(neighbor).is_full() {
                        break;
                    }
                }
            }
            Ok(output.current_len() - before)
        })())
    }
}
impl glue::InplaceDeleteStrategy<Provider> for Strategy {
    type DeleteElement<'a> = &'a [f32];
    type DeleteElementGuard = OwnedVector;
    type DeleteElementError = ANNError;
    type PruneStrategy = Self;
    type DeleteSearchAccessor<'a> = Accessor<'a>;
    type SearchPostProcessor = Output;
    type SearchStrategy = Self;
    fn prune_strategy(&self) -> Self {
        *self
    }
    fn search_strategy(&self) -> Self {
        *self
    }
    fn search_post_processor(&self) -> Output {
        Output
    }
    fn get_delete_element<'a>(
        &'a self,
        _: &'a Provider,
        context: &'a OperationContext,
        id: u64,
    ) -> impl Future<Output = ANNResult<OwnedVector>> + Send {
        ready(context.fetch_vector(id).map_err(ann))
    }
}
/// Inputs are already encoded/normalized by extension policy.
pub(crate) fn insert(
    store: &mut dyn GraphStore,
    config: &GraphConfig,
    rowid: u64,
    vector: &[f32],
) -> StoreResult<()> {
    config.validate_vector(vector)?;
    if rowid > i64::MAX as u64 {
        return Err(IndexError::new(
            "DiskANN rowid exceeds SQLite INTEGER range",
        ));
    }
    operation(store, config, |index, context| {
        drive(index.insert(&Strategy, context, &rowid, vector))?.map_err(index_error)
    })
}
pub(crate) fn delete(
    store: &mut dyn GraphStore,
    config: &GraphConfig,
    rowid: u64,
) -> StoreResult<()> {
    if rowid > i64::MAX as u64 {
        return Err(IndexError::new(
            "DiskANN rowid exceeds SQLite INTEGER range",
        ));
    }
    // A directed graph may have incoming edges that OneHop repair cannot find.
    // Clearing a retired node's outgoing topology can disconnect live nodes from
    // the frozen entrypoint. Keep the vector AND adjacency as navigation bridges
    // until the SQLite owner transactionally rebuilds the live graph.
    operation(store, config, |_index, context| {
        with_store(context, |store| store.mark_delete(rowid))
    })
}
pub(crate) fn search(
    store: &mut dyn GraphStore,
    config: &GraphConfig,
    query: &[f32],
    k: usize,
    search_l: Option<usize>,
    filter: Option<&HashSet<u64>>,
) -> StoreResult<Vec<SearchResult>> {
    config.validate_vector(query)?;
    if k == 0 {
        return Ok(Vec::new());
    }
    let l = search_l.unwrap_or(config.search_l).max(k);
    if l > config.limits.max_visits {
        return Err(limit("DiskANN search exceeds visit limit"));
    }
    let mut admitted = config.clone();
    admitted.search_l = l;
    operation(store, &admitted, |index, context| {
        let strategy = QueryStrategy { filter };
        let mut neighbors: Vec<Neighbor<u64>> = Vec::new();
        let capacity = l
            .checked_add(1)
            .ok_or_else(|| limit("DiskANN result capacity overflow"))?;
        neighbors
            .try_reserve_exact(capacity)
            .map_err(|_| context.fail(allocation_error()))?;
        let knn = Knn::new(l, None).map_err(|e| IndexError::new(e.to_string()))?;
        // Retained tombstones and frozen nodes navigate but never consume the
        // live-result stream, even when there is no explicit SQL rowid filter.
        // Filtering only a completed global best-L list could hide every live
        // candidate behind nearby retired embeddings after sustained updates.
        drive(index.search_with(
            InlineFilterSearch::new(knn, None),
            &strategy,
            Output,
            context,
            query,
            &mut neighbors,
        ))?
        .map_err(index_error)?;
        let mut results = Vec::new();
        results
            .try_reserve_exact(k.min(neighbors.len()))
            .map_err(|_| context.fail(allocation_error()))?;
        for neighbor in neighbors.into_iter().take(k) {
            let rowid = with_store(context, |store| store.rowid(*neighbor.id()))?
                .ok_or_else(|| IndexError::new("DiskANN result lost its live rowid"))?;
            results.push(SearchResult::new(*neighbor.distance(), rowid));
        }
        Ok(results)
    })
}
#[cfg(test)]
#[path = "diskann_core/tests.rs"]
mod tests;
