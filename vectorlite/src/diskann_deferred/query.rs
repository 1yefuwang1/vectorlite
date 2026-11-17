//! Operation-local, read-only search of one published Format4 generation.
//! The metadata cursor anchors every read and all accepted projected bytes.
//! Native ID zero is ordinary data; the only frozen start point is explicit N.
use std::mem::size_of;

use crate::core::{SearchFilter, SearchResult};
use crate::diskann_core::{
    self, GraphConfig, GraphStore, NodeState, ResourceLimits, StoredVector, VectorBatchAction,
};
use crate::diskann_store::codec::{
    blob_projection, bounded_blob, checked_id, checked_product, corrupt, decode_ids, nonnegative,
    normalize_input, reserve, too_big, with_decoded_vector,
};
use crate::diskann_store::{DiskAnnTable, Generation, GraphNodeRecord, QueryRows};
use crate::ffi;
use crate::index_error::IndexError;
use crate::ops;
use crate::sqlite::{Statement, Step};
use crate::vector_space::DistanceType;

type Result<T> = std::result::Result<T, IndexError>;

/// Upper bound on the aggregate MEMUSED retained by one operation's eager reader
/// set. The set is prepared before GraphConfig exists, so the retained statements
/// and the core's copied allowance are disjoint by construction.
///
/// MEMUSED is an approximate, process-state-sensitive measure: the pinned SQLite
/// 3.51.2 reports roughly 68-80 KiB for the five shapes once a deferred build has
/// run in the process (about 8 prepares' worth fits easily in a fresh process at
/// the older 64 KiB cap, but that cap then silently demotes the projection-capable
/// shape and every projected result re-prepares). This cap is therefore a reserve
/// bound with room for that measurement variance rather than a tight fit. The
/// allowance still subtracts the core floor and the reprepare headroom from
/// cache_bytes before applying the cap, so tight budgets keep their partial/none
/// admission behaviour.
const MAX_READER_BYTES: usize = 256 * 1024;
/// Reserved above the admitted readers for SQLite transparent reprepare growth
/// and for the bytes released when a reader is demoted mid-operation.
const REPREPARE_HEADROOM: usize = 16 * 1024;
/// Number of distinct per-operation reader shapes (see ReaderShape).
const READER_SHAPES: usize = 5;

// --- debug-only adapter instrumentation ---
//
// The counters are monotonic process-wide totals. A native test host measures a
// single query by reading before and after it. Release builds compile this
// module out, so no vectorlite_test_ symbol may appear in a release artifact.
#[cfg(debug_assertions)]
mod instrumentation {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    /// Effective search list of the most recent knn operation.
    static LAST_SEARCH_L: AtomicU64 = AtomicU64::new(0);
    /// Completed reader executions since process start.
    static READER_EXECUTIONS: AtomicU64 = AtomicU64::new(0);
    /// with_vectors_batch override invocations since process start.
    static BATCH_CALLS: AtomicU64 = AtomicU64::new(0);
    /// One-shot request used by Debug-only contract tests to force demotion on
    /// the next query; consumed at operation start.
    static FORCE_DEMOTE: AtomicBool = AtomicBool::new(false);
    /// Reader shapes whose next admission prepare attempt must be forced to fail,
    /// one bit per `ReaderShape::ALL` index.
    static FAIL_READER_PREPARE: AtomicU64 = AtomicU64::new(0);

    pub(super) fn set_search_l(l: usize) {
        LAST_SEARCH_L.store(u64::try_from(l).unwrap_or(u64::MAX), Ordering::Relaxed);
    }
    pub(super) fn note_reader_execution() {
        READER_EXECUTIONS.fetch_add(1, Ordering::Relaxed);
    }
    pub(super) fn note_batch_call() {
        BATCH_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    pub(super) fn take_force_demote() -> bool {
        FORCE_DEMOTE.swap(false, Ordering::Relaxed)
    }
    /// Consumes one armed forced-failure bit for `shape`.
    pub(super) fn take_failed_prepare(shape_mask: u64) -> bool {
        let mut current = FAIL_READER_PREPARE.load(Ordering::Relaxed);
        loop {
            if current & shape_mask == 0 {
                return false;
            }
            match FAIL_READER_PREPARE.compare_exchange_weak(
                current,
                current & !shape_mask,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(observed) => current = observed,
            }
        }
    }

    /// Reports the effective search list used by the most recent query.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_last_search_l() -> u64 {
        LAST_SEARCH_L.load(Ordering::Relaxed)
    }
    /// Reports the monotonic count of reused-reader executions.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_reader_executions() -> u64 {
        READER_EXECUTIONS.load(Ordering::Relaxed)
    }
    /// Reports the monotonic count of with_vectors_batch override calls.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_batch_calls() -> u64 {
        BATCH_CALLS.load(Ordering::Relaxed)
    }
    /// Requests demotion of every admitted reader after its first completed read.
    ///
    /// The hook is one-shot, applies to the next knn operation initiated after the
    /// call (consumed at operation start), and is Debug builds only. It exists so
    /// contract tests can exercise the demotion path without corrupting the
    /// retained readers; results and visit charges must stay identical to the
    /// non-demoted path and an empty or reader-less operation consumes the request
    /// without demoting anything.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_force_demote() {
        FORCE_DEMOTE.store(true, Ordering::Relaxed);
    }
    /// Forces the next admission prepare attempt of each selected reader shape to
    /// fail, so a query silently falls back to scalar readers.
    ///
    /// `mask` selects shapes by `ReaderShape::ALL` bit index: bit 0 Graph, bit 1
    /// Identity, bit 2 Combined, bit 3 ExactUnfiltered, bit 4 ExactPoint. Each
    /// selected shape fails exactly one attempt and then behaves normally. The
    /// failure is a real `sqlite3_prepare_v2` dispatch with invalid SQL, so
    /// `vectorlite_test_prepare_attempts` counts it while the successful-prepare
    /// counter does not. Debug builds only; release artifacts export none of this.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_fail_reader_prepare(mask: u64) {
        FAIL_READER_PREPARE.fetch_or(mask, Ordering::Relaxed);
    }
}

fn record_search_l(l: usize) {
    #[cfg(debug_assertions)]
    instrumentation::set_search_l(l);
    #[cfg(not(debug_assertions))]
    let _ = l;
}
fn note_reader_execution() {
    #[cfg(debug_assertions)]
    instrumentation::note_reader_execution();
}
fn note_batch_call() {
    #[cfg(debug_assertions)]
    instrumentation::note_batch_call();
}
fn take_force_demote() -> bool {
    #[cfg(debug_assertions)]
    {
        instrumentation::take_force_demote()
    }
    #[cfg(not(debug_assertions))]
    {
        false
    }
}
fn take_failed_reader_prepare(shape: ReaderShape) -> bool {
    #[cfg(debug_assertions)]
    {
        instrumentation::take_failed_prepare(shape.mask())
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = shape;
        false
    }
}

fn unsupported() -> IndexError {
    IndexError::with_code(
        ffi::SQLITE_READONLY as i32,
        "DiskANN published query adapter is read-only",
    )
}

// Explicitly finalize on both success and failure. A primary SQL/read error is
// never replaced by a later cleanup failure. No auxiliary statement escapes.
fn finish<T>(statement: Statement<'_>, result: Result<T>) -> Result<T> {
    let cleanup = statement.finish();
    match result {
        Err(error) => Err(error),
        Ok(value) => cleanup.map(|()| value),
    }
}

/// Core's admitted effective search list: override.unwrap_or(search_l).max(k).
fn effective_l(override_l: Option<usize>, search_list_size: usize, count: usize) -> usize {
    override_l.unwrap_or(search_list_size).max(count)
}

/// Route-aware retained-reader allowance.
///
/// The graph route must leave the whole core floor plus the reprepare headroom
/// for GraphConfig; the exact route has no core admission and only reserves the
/// headroom. The allowance never exceeds MAX_READER_BYTES, and a head budget
/// that cannot cover the reservation yields zero (scalar-only operation).
fn reader_allowance(head: usize, floor: Option<usize>) -> usize {
    head.saturating_sub(floor.unwrap_or(0))
        .saturating_sub(REPREPARE_HEADROOM)
        .min(MAX_READER_BYTES)
}

/// The only workspace value handed to GraphConfig.
///
/// Admitted readers and their headroom are subtracted from HEAD's owner head so
/// the copied core allowance and the retained statements stay disjoint. With no
/// admitted reader the operation runs HEAD's scalar path, so HEAD's exact value
/// is passed and no headroom is withheld (preserving every HEAD admission
/// outcome at every cache_bytes).
fn core_bytes(head: usize, admitted: usize) -> usize {
    if admitted == 0 {
        head
    } else {
        head.saturating_sub(admitted)
            .saturating_sub(REPREPARE_HEADROOM)
    }
}

/// Sequential, priority-ordered reader admission.
///
/// A shape whose measurement is unavailable (prepare failure or unavailable
/// MEMUSED) is silently scalar-only and does not end admission. The first shape
/// whose measured size does not fit ends admission: lower-priority shapes are not
/// prepared, keeping "unused shapes are never prepared" true.
#[derive(Debug)]
struct ReaderAllowance {
    remaining: usize,
    closed: bool,
}
impl ReaderAllowance {
    fn new(allowance: usize) -> Self {
        Self {
            remaining: allowance,
            closed: false,
        }
    }
    /// Whether admission has ended, so no lower-priority shape may be prepared.
    fn closed(&self) -> bool {
        self.closed
    }
    /// Returns the measured bytes when the offered shape is admitted.
    fn offer(&mut self, measured: Option<usize>) -> Option<usize> {
        if self.closed {
            return None;
        }
        match measured {
            None => None,
            Some(bytes) if bytes <= self.remaining => {
                self.remaining -= bytes;
                Some(bytes)
            }
            Some(_) => {
                self.closed = true;
                None
            }
        }
    }
}

/// One reused read statement shape, in admission priority order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReaderShape {
    /// Single-table graph_nodes row with the frozen-root-aware projection.
    Graph,
    /// Three-table identity join without the vector payload.
    Identity,
    /// Three-table identity join with the vector payload (state-aware).
    Combined,
    /// eligible_sql(point=false, vector=true) streamed to Done.
    ExactUnfiltered,
    /// eligible_sql(point=true, vector=true) shared by point scans, IN probes
    /// and projection.
    ExactPoint,
}
impl ReaderShape {
    const ALL: [Self; READER_SHAPES] = [
        Self::Graph,
        Self::Identity,
        Self::Combined,
        Self::ExactUnfiltered,
        Self::ExactPoint,
    ];
    fn index(self) -> usize {
        self as usize
    }
    /// Debug-only hook bit selecting this shape.
    #[cfg(debug_assertions)]
    fn mask(self) -> u64 {
        1u64 << self.index()
    }
    /// SQL text for this shape. None means the shape cannot be represented for
    /// this table, which silently keeps it on the scalar path.
    fn sql(self, table: &DiskAnnTable) -> Option<String> {
        match self {
            Self::Graph => graph_sql(table),
            Self::Identity => Some(node_sql(table, false)),
            Self::Combined => Some(node_sql(table, true)),
            Self::ExactUnfiltered => Some(eligible_sql(table, false, true)),
            Self::ExactPoint => Some(eligible_sql(table, true, true)),
        }
    }
}

/// Route's reader shapes in admission priority order; shapes never repeat.
#[derive(Clone, Copy, Debug)]
struct ReaderPlan {
    shapes: [ReaderShape; READER_SHAPES],
    len: usize,
}
impl ReaderPlan {
    fn new() -> Self {
        Self {
            shapes: [ReaderShape::Graph; READER_SHAPES],
            len: 0,
        }
    }
    fn push(&mut self, shape: ReaderShape) {
        if self.len < READER_SHAPES && !self.as_slice().contains(&shape) {
            self.shapes[self.len] = shape;
            self.len += 1;
        }
    }
    fn as_slice(&self) -> &[ReaderShape] {
        &self.shapes[..self.len]
    }
}

/// Shapes used by a graph-route search, highest expected value first. The exact
/// fallback shape is filter-aware and the projection reader is only planned when
/// an accepted row is projected.
fn graph_reader_plan(filter: &SearchFilter<'_>, project: bool) -> ReaderPlan {
    let mut plan = ReaderPlan::new();
    plan.push(ReaderShape::Graph);
    plan.push(ReaderShape::Identity);
    plan.push(ReaderShape::Combined);
    match filter {
        SearchFilter::None => plan.push(ReaderShape::ExactUnfiltered),
        SearchFilter::Equals(_) | SearchFilter::In(_) => plan.push(ReaderShape::ExactPoint),
    }
    if project {
        plan.push(ReaderShape::ExactPoint);
    }
    plan
}

/// Shapes used by an exact route. There is no core call, so exactly one stream
/// shape is always needed; projection reuses the point shape.
fn exact_reader_plan(filter: &SearchFilter<'_>, project: bool) -> ReaderPlan {
    let mut plan = ReaderPlan::new();
    match filter {
        SearchFilter::None => {
            plan.push(ReaderShape::ExactUnfiltered);
            if project {
                plan.push(ReaderShape::ExactPoint);
            }
        }
        SearchFilter::Equals(_) | SearchFilter::In(_) => plan.push(ReaderShape::ExactPoint),
    }
    plan
}

/// Deterministic demotion order: largest retained measurement first, ties by
/// shape priority. None entries are not retained and are skipped.
fn demotion_order(measured: &[Option<usize>]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..measured.len())
        .filter(|&index| measured[index].is_some())
        .collect();
    order.sort_by(|&a, &b| {
        measured[b]
            .unwrap_or(0)
            .cmp(&measured[a].unwrap_or(0))
            .then(a.cmp(&b))
    });
    order
}

/// Whether retained-reader growth has left the admitted envelope by more than the
/// reprepare headroom, which triggers mark-and-drain demotion.
fn demotion_needed(high_water: usize, admitted_total: usize) -> bool {
    high_water > admitted_total.saturating_add(REPREPARE_HEADROOM)
}

/// Per-operation reused readers. Every retained statement is finalized before
/// knn returns; a shape falls back to HEAD's scalar prepare/finalize path once
/// it is not retained (never admitted, or demoted).
struct Readers {
    statements: [Option<Statement<'static>>; READER_SHAPES],
    /// Latest MEMUSED for each retained shape (None when not retained).
    measurements: [Option<usize>; READER_SHAPES],
    /// Sum of the admission measurements for the retained shapes.
    admitted_total: usize,
    /// Highest retained aggregate observed since the last demotion.
    high_water: usize,
    /// One-shot Debug demotion request captured at operation start; always false
    /// in release builds and never read while a reader is on a row.
    force_demote: bool,
}
impl Readers {
    fn new() -> Self {
        Self {
            statements: std::array::from_fn(|_| None),
            measurements: [None; READER_SHAPES],
            admitted_total: 0,
            high_water: 0,
            force_demote: false,
        }
    }
    /// Captures the one-shot Debug demotion request for this operation, so the
    /// next query after the hook deterministically exercises demotion once.
    fn arm_force_demote(&mut self, force: bool) {
        self.force_demote = force;
    }
    /// Takes the retained statement for one logical read. The slot stays empty
    /// until the read has drained to Done and the statement is restored, so a
    /// partially consumed reader can never be handed out twice.
    fn take(&mut self, shape: ReaderShape) -> Option<Statement<'static>> {
        self.statements[shape.index()].take()
    }
    fn restore(&mut self, shape: ReaderShape, statement: Statement<'static>) {
        self.statements[shape.index()] = Some(statement);
    }
    fn retain(&mut self, shape: ReaderShape, statement: Statement<'static>, bytes: usize) {
        let index = shape.index();
        self.statements[index] = Some(statement);
        self.measurements[index] = Some(bytes);
        self.admitted_total = self.admitted_total.saturating_add(bytes);
        self.high_water = self.admitted_total;
    }
    fn admitted_total(&self) -> usize {
        self.admitted_total
    }
    fn aggregate(&self) -> usize {
        ReaderShape::ALL
            .iter()
            .map(|shape| {
                let index = shape.index();
                if self.statements[index].is_some() {
                    self.measurements[index].unwrap_or(0)
                } else {
                    0
                }
            })
            .sum()
    }
    /// Samples the reader that just completed and demotes largest-first once the
    /// retained aggregate outgrows the admission envelope. Called only after the
    /// read drained to Done, so finalization never happens on a live row.
    fn note_read(&mut self, shape: ReaderShape) {
        let index = shape.index();
        let Some(bytes) = self.statements[index]
            .as_ref()
            .and_then(Statement::memory_used)
        else {
            return;
        };
        self.measurements[index] = Some(bytes);
        self.high_water = self.high_water.max(self.aggregate());
        if self.force_demote {
            self.force_demote = false;
            self.demote_all();
            return;
        }
        if demotion_needed(self.high_water, self.admitted_total) {
            self.demote_to_admitted();
        }
    }
    /// Finalizes one retained reader; demotion is an accounting decision and never
    /// replaces a primary error with a cleanup error.
    fn finalize(&mut self, shape: ReaderShape) {
        let index = shape.index();
        if let Some(statement) = self.statements[index].take() {
            self.measurements[index] = None;
            let _ = statement.finish();
        }
    }
    fn demote_all(&mut self) {
        for shape in ReaderShape::ALL {
            self.finalize(shape);
        }
        self.high_water = 0;
    }
    fn demote_to_admitted(&mut self) {
        for index in demotion_order(&self.measurements) {
            if self.aggregate() <= self.admitted_total {
                break;
            }
            self.finalize(ReaderShape::ALL[index]);
        }
        self.high_water = self.aggregate();
    }
    /// Finalizes every remaining reader, returning the first cleanup error.
    fn finish_all(&mut self) -> Result<()> {
        let mut cleanup = Ok(());
        for shape in ReaderShape::ALL {
            let index = shape.index();
            if let Some(statement) = self.statements[index].take() {
                self.measurements[index] = None;
                if let Err(error) = statement.finish() {
                    if cleanup.is_ok() {
                        cleanup = Err(error);
                    }
                }
            }
        }
        self.high_water = 0;
        cleanup
    }
}

#[derive(Debug)]
struct Budget {
    visits: usize,
    max_visits: usize,
    /// HEAD's unchanged owner reservation.
    owner: usize,
    cache_bytes: usize,
}
impl Budget {
    fn admit(
        table: &DiskAnnTable,
        count: usize,
        filter: &SearchFilter<'_>,
        project: bool,
    ) -> Result<Self> {
        // Reserve SQL names/VMs, metadata, owner decode/payload buffers, both
        // result representations during final materialization and filter storage.
        // Core receives only the remainder, not a second full cache_bytes budget.
        let names = [
            &table.names().meta,
            &table.names().entries,
            &table.names().current,
            &table.names().generations,
            &table.names().graph_nodes,
        ]
        .iter()
        .try_fold(0usize, |sum, name| {
            sum.checked_add(name.len())
                .ok_or_else(|| too_big("DiskANN query identifier overflow"))
        })?;
        let filter_bytes = match filter {
            SearchFilter::In(ids) => checked_product(ids.capacity(), 64, "query filter")?,
            _ => 0,
        };
        let per_result = size_of::<SearchResult>()
            .checked_mul(2)
            .and_then(|n| {
                n.checked_add(if project {
                    table.vector_bytes() + size_of::<Vec<u8>>()
                } else {
                    0
                })
            })
            .ok_or_else(|| too_big("DiskANN query result overflow"))?;
        let owner = checked_product(table.vector_bytes(), 5, "query vector buffers")?
            .checked_add(checked_product(
                table.options().physical_degree(),
                24,
                "query adjacency",
            )?)
            .and_then(|n| {
                checked_product(names, 32, "query SQL names")
                    .ok()
                    .and_then(|s| n.checked_add(s))
            })
            .and_then(|n| n.checked_add(64 * 1024))
            .and_then(|n| n.checked_add(filter_bytes))
            .and_then(|n| {
                checked_product(count, per_result, "query results")
                    .ok()
                    .and_then(|s| n.checked_add(s))
            })
            .ok_or_else(|| too_big("DiskANN query workspace overflow"))?;
        let cache_bytes = table.options().cache_bytes;
        cache_bytes
            .checked_sub(owner)
            .ok_or_else(|| too_big("DiskANN query workspace exceeds cache_bytes"))?;
        Ok(Self {
            visits: 0,
            max_visits: table.options().max_visits,
            owner,
            cache_bytes,
        })
    }
    /// HEAD's owner remainder before any retained reader is subtracted.
    fn head_bytes(&self) -> usize {
        self.cache_bytes - self.owner
    }
    fn reader_allowance(&self, floor: Option<usize>) -> usize {
        reader_allowance(self.head_bytes(), floor)
    }
    fn core_bytes(&self, admitted: usize) -> usize {
        core_bytes(self.head_bytes(), admitted)
    }
    fn visit(&mut self) -> Result<()> {
        if self.visits >= self.max_visits {
            return Err(too_big("DiskANN query exceeded max_visits"));
        }
        self.visits += 1;
        Ok(())
    }
    fn remaining(&self) -> usize {
        self.max_visits - self.visits
    }
}

fn node_sql(table: &DiskAnnTable, vector: bool) -> String {
    let names = table.names();
    let payload = if vector {
        format!(
            ",{}",
            blob_projection(
                "CASE WHEN n.state=2 THEN n.frozen_vector ELSE e.vector END",
                table.vector_bytes(),
                true
            )
        )
    } else {
        String::new()
    };
    format!("SELECT n.entry_id,n.state,n.frozen_vector IS NULL,e.public_rowid,e.created_revision,e.retired_revision,c.entry_id{payload} FROM {} n LEFT JOIN {} e ON e.entry_id=n.entry_id LEFT JOIN {} c ON c.public_rowid=e.public_rowid WHERE n.generation_id=?1 AND n.native_local_id=?2", names.graph_nodes, names.entries, names.current)
}

fn eligible_sql(table: &DiskAnnTable, point: bool, vector: bool) -> String {
    let names = table.names();
    let payload = if vector {
        format!(
            ",{}",
            blob_projection("e.vector", table.vector_bytes(), true)
        )
    } else {
        String::new()
    };
    // current joins the GRAPH version, never the latest canonical payload by
    // rowid alone. LEFT entries makes a broken current entry a corruption.
    format!("SELECT n.native_local_id,n.entry_id,e.public_rowid,e.created_revision,e.retired_revision,c.public_rowid,n.frozen_vector IS NULL{payload} FROM {} n JOIN {} c ON c.entry_id=n.entry_id LEFT JOIN {} e ON e.entry_id=n.entry_id WHERE n.generation_id=?1 AND n.state=0{} ORDER BY n.native_local_id", names.graph_nodes, names.current, names.entries, if point { " AND c.public_rowid=?2" } else { "" })
}

/// Byte-identical to the single-table statement used by read_graph_node.
fn graph_sql(table: &DiskAnnTable) -> Option<String> {
    let names = table.names();
    let adjacency_bytes =
        checked_product(table.options().physical_degree(), 8, "query adjacency").ok()?;
    Some(format!("SELECT entry_id,state,{},{},frozen_vector IS NULL FROM {} WHERE generation_id=?1 AND native_local_id=?2",blob_projection("frozen_vector",table.vector_bytes(),true),blob_projection("neighbors",adjacency_bytes,false),names.graph_nodes))
}

/// Moves a retained reader through one reuse cycle: reset, clear bindings, bind.
fn reset_reader(statement: &mut Statement<'static>) -> Result<()> {
    statement.reset()?;
    statement.clear_bindings()
}

fn bind_node(generation: &Generation, statement: &mut Statement<'static>, id: u64) -> Result<()> {
    if generation.total_count == 0 || id > generation.total_count {
        return Err(corrupt("DiskANN query local ID exceeds generation"));
    }
    statement.bind_i64(1, checked_id(generation.generation_id)?)?;
    statement.bind_i64(2, checked_id(id)?)
}

// Read immutable identity and current membership before accepting top-k.
// Missing/corrupt metadata is never interpreted as a deleted embedding.
fn classify(
    generation: &Generation,
    revision: u64,
    statement: &Statement<'_>,
    id: u64,
) -> Result<(NodeState, Option<u64>)> {
    let role = nonnegative(statement, 1, "query node role")?;
    if Some(id) == generation.root_local_id {
        if role != 2
            || !statement.column_is_null(0)?
            || nonnegative(statement, 2, "root vector NULL")? != 0
        {
            return Err(corrupt("DiskANN explicit frozen root role invalid"));
        }
        return Ok((NodeState::Frozen, None));
    }
    if role != 0
        || id >= generation.total_count
        || nonnegative(statement, 0, "query entry ID")? == 0
        || nonnegative(statement, 2, "data frozen vector NULL")? != 1
    {
        return Err(corrupt("DiskANN query node role/version invalid"));
    }
    let rowid = nonnegative(statement, 3, "query original rowid")?;
    let created = nonnegative(statement, 4, "query creation revision")?;
    if created == 0 || created > generation.source_revision {
        return Err(corrupt("DiskANN graph entry lies outside frozen snapshot"));
    }
    let retired = if statement.column_is_null(5)? {
        None
    } else {
        Some(nonnegative(statement, 5, "query retirement revision")?)
    };
    if retired.is_some_and(|r| r <= created || r > revision) {
        return Err(corrupt("DiskANN query immutable interval invalid"));
    }
    let current = if statement.column_is_null(6)? {
        None
    } else {
        Some(nonnegative(statement, 6, "query current entry ID")?)
    };
    if current == Some(0) {
        return Err(corrupt("DiskANN query current entry is zero"));
    }
    if current == Some(nonnegative(statement, 0, "query entry ID")?) {
        if retired.is_some() {
            return Err(corrupt("DiskANN current graph entry is retired"));
        }
        Ok((NodeState::Live, Some(rowid)))
    } else {
        Ok((NodeState::Deleted, None))
    }
}

fn eligible_identity(generation: &Generation, statement: &Statement<'_>) -> Result<(u64, u64)> {
    let id = nonnegative(statement, 0, "eligible local ID")?;
    let entry = nonnegative(statement, 1, "eligible entry ID")?;
    let rowid = nonnegative(statement, 2, "eligible original rowid")?;
    let created = nonnegative(statement, 3, "eligible creation revision")?;
    if id >= generation.total_count
        || entry == 0
        || created == 0
        || created > generation.source_revision
        || !statement.column_is_null(4)?
        || nonnegative(statement, 6, "eligible frozen vector NULL")? != 1
        || rowid != nonnegative(statement, 5, "eligible current rowid")?
    {
        return Err(corrupt(
            "DiskANN eligible graph/current/entry identity invalid",
        ));
    }
    Ok((id, rowid))
}

fn optional_id(statement: &Statement<'_>, column: usize, name: &str) -> Result<Option<u64>> {
    if statement.column_is_null(column)? {
        Ok(None)
    } else {
        Ok(Some(nonnegative(statement, column, name)?))
    }
}

/// HEAD's scalar identity read, used when the identity reader is not retained.
fn identity_scalar(
    table: &DiskAnnTable,
    generation: &Generation,
    revision: u64,
    id: u64,
) -> Result<(NodeState, Option<u64>)> {
    let mut statement = table.connection().prepare(&node_sql(table, false))?;
    let outcome = (|| {
        bind_node(generation, &mut statement, id)?;
        if statement.step()? != Step::Row {
            return Err(corrupt("DiskANN referenced query node/entry missing"));
        }
        let identity = classify(generation, revision, &statement, id)?;
        if statement.step()? != Step::Done {
            return Err(corrupt("DiskANN query node identity not unique"));
        }
        Ok(identity)
    })();
    finish(statement, outcome)
}

/// One retained identity-reader execution. Charges nothing: visit accounting for
/// rowid resolution stays in rowid(), and neighbour-target validation is free.
fn identity_read(
    generation: &Generation,
    revision: u64,
    statement: &mut Statement<'static>,
    id: u64,
) -> Result<(NodeState, Option<u64>)> {
    reset_reader(statement)?;
    bind_node(generation, statement, id)?;
    if statement.step()? != Step::Row {
        return Err(corrupt("DiskANN referenced query node/entry missing"));
    }
    let identity = classify(generation, revision, statement, id)?;
    if statement.step()? != Step::Done {
        return Err(corrupt("DiskANN query node identity not unique"));
    }
    Ok(identity)
}

/// HEAD's scalar combined read, used when the combined reader is not retained.
fn vector_scalar(
    table: &DiskAnnTable,
    generation: &Generation,
    revision: u64,
    id: u64,
    decode: &mut Vec<f32>,
    action: &mut dyn FnMut(StoredVector<'_>) -> Result<()>,
) -> Result<()> {
    let mut statement = table.connection().prepare(&node_sql(table, true))?;
    let outcome = (|| {
        bind_node(generation, &mut statement, id)?;
        if statement.step()? != Step::Row {
            return Err(corrupt("DiskANN referenced vector node/entry missing"));
        }
        classify(generation, revision, &statement, id)?;
        check_blob(&statement, 7, 8, table.vector_bytes())?;
        statement.with_column_blob(8, |bytes| {
            with_decoded_vector(bytes, table.space().dim, decode, action)
        })?;
        if statement.step()? != Step::Done {
            return Err(corrupt("DiskANN query vector node not unique"));
        }
        Ok(())
    })();
    finish(statement, outcome)
}

/// One retained combined-reader execution. The borrowed checked proof lives only
/// in the synchronous column callback; action is pure scoring/copying.
fn vector_read(
    table: &DiskAnnTable,
    generation: &Generation,
    revision: u64,
    statement: &mut Statement<'static>,
    id: u64,
    decode: &mut Vec<f32>,
    action: &mut dyn FnMut(StoredVector<'_>) -> Result<()>,
) -> Result<()> {
    reset_reader(statement)?;
    bind_node(generation, statement, id)?;
    if statement.step()? != Step::Row {
        return Err(corrupt("DiskANN referenced vector node/entry missing"));
    }
    classify(generation, revision, statement, id)?;
    check_blob(statement, 7, 8, table.vector_bytes())?;
    statement.with_column_blob(8, |bytes| {
        with_decoded_vector(bytes, table.space().dim, decode, action)
    })?;
    if statement.step()? != Step::Done {
        return Err(corrupt("DiskANN query vector node not unique"));
    }
    Ok(())
}

/// One retained graph/adjacency reader execution. Validation is byte-identical
/// to DiskAnnTable::read_graph_node, which stays the scalar fallback.
fn graph_read(
    table: &DiskAnnTable,
    generation: &Generation,
    statement: &mut Statement<'static>,
    id: u64,
) -> Result<GraphNodeRecord> {
    if id > generation.total_count || generation.total_count == 0 {
        return Err(corrupt("DiskANN local node ID exceeds generation"));
    }
    reset_reader(statement)?;
    statement.bind_i64(1, checked_id(generation.generation_id)?)?;
    statement.bind_i64(2, checked_id(id)?)?;
    if statement.step()? != Step::Row {
        return Err(corrupt("DiskANN referenced graph node is missing"));
    }
    let entry_id = optional_id(statement, 0, "graph entry ID")?;
    let state = nonnegative(statement, 1, "graph state")?;
    let frozen = state == 2;
    if (frozen && (generation.root_local_id != Some(id) || entry_id.is_some()))
        || (!frozen
            && (state != 0
                || id >= generation.total_count
                || entry_id.is_none_or(|v| v == 0)
                || nonnegative(statement, 6, "frozen NULL flag")? != 1))
    {
        return Err(corrupt("DiskANN native root/node version role mismatch"));
    }
    let frozen_vector = if frozen {
        let bytes = bounded_blob(statement, 2, 3, table.vector_bytes(), true, "frozen vector")?;
        with_decoded_vector(&bytes, table.space().dim, &mut Vec::new(), &mut |_| Ok(()))?;
        Some(bytes)
    } else {
        None
    };
    let adjacency_bytes = checked_product(table.options().physical_degree(), 8, "query adjacency")?;
    let neighbors = decode_ids(
        &bounded_blob(statement, 4, 5, adjacency_bytes, false, "neighbors")?,
        table.options().physical_degree(),
        Some(id),
    )?;
    if neighbors.iter().any(|n| *n > generation.total_count) {
        return Err(corrupt(
            "DiskANN adjacency exceeds generation local ID range",
        ));
    }
    if statement.step()? != Step::Done {
        return Err(corrupt("DiskANN graph node is not unique"));
    }
    Ok(GraphNodeRecord {
        native_local_id: id,
        entry_id,
        frozen_vector,
        neighbors,
    })
}

type QueryOutcome = (Vec<SearchResult>, Option<Vec<Vec<u8>>>);

struct PublishedGraph<'a> {
    table: &'a DiskAnnTable,
    generation: Generation,
    revision: u64,
    budget: Budget,
    readers: Readers,
    decode: Vec<f32>,
}

impl PublishedGraph<'_> {
    /// Prepares and measures only the route's needed readers, in priority order.
    ///
    /// Preparation is attempted only while the envelope is open: the first shape
    /// whose measured size does not fit closes admission, and no lower-priority
    /// shape is built or prepared after that, so "unused shapes are never
    /// prepared" holds. A shape whose SQL cannot be built, whose prepare fails
    /// (including the Debug-only forced failure), or whose MEMUSED is unavailable
    /// is silently scalar-only and does not end admission. Admission never raises
    /// a setup error; a rejected local statement is finalized by Drop and its
    /// shape keeps HEAD's scalar prepare/finalize path.
    fn admit_readers(&mut self, plan: &ReaderPlan, allowance: usize) {
        let mut cursor = ReaderAllowance::new(allowance);
        for &shape in plan.as_slice() {
            if cursor.closed() {
                break;
            }
            let Some(sql) = shape.sql(self.table) else {
                continue;
            };
            let Ok(statement) = self.prepare_reader(shape, &sql) else {
                continue;
            };
            if let Some(bytes) = cursor.offer(statement.memory_used()) {
                self.readers.retain(shape, statement, bytes);
            }
        }
    }
    /// Prepares one reader shape, honoring the Debug-only forced-failure hook.
    ///
    /// The hook selects shapes by `ReaderShape::mask` and fails only the next
    /// attempt for each selected shape. A forced failure is a real prepare
    /// dispatch with invalid SQL, so the attempt counter sees it while the
    /// successful-prepare counter does not, and the shape silently falls back to
    /// the scalar path exactly like any other admission prepare failure.
    fn prepare_reader(&self, shape: ReaderShape, sql: &str) -> Result<Statement<'static>> {
        if take_failed_reader_prepare(shape) {
            return self
                .table
                .connection()
                .prepare(&format!("{sql} forced_reader_prepare_failure"));
        }
        self.table.connection().prepare(sql)
    }
    /// Restores a completed reader and samples it, or finalizes the affected
    /// reader on a failed read while preserving the primary error.
    fn finish_reader<T>(
        &mut self,
        shape: ReaderShape,
        statement: Statement<'static>,
        outcome: Result<T>,
    ) -> Result<T> {
        match outcome {
            Ok(value) => {
                self.readers.restore(shape, statement);
                self.readers.note_read(shape);
                Ok(value)
            }
            Err(error) => {
                let _ = statement.finish();
                Err(error)
            }
        }
    }
    /// Finalizes every retained reader before knn returns. A primary error is
    /// never replaced by a cleanup error.
    fn finish_readers<T>(&mut self, result: Result<T>) -> Result<T> {
        let cleanup = self.readers.finish_all();
        match result {
            Err(error) => Err(error),
            Ok(value) => cleanup.map(|()| value),
        }
    }
    /// Identity-only read: the reused three-table reader when retained, else
    /// HEAD's per-read statement. Used for rowid resolution, state checks,
    /// start-point validation and zero-charge neighbour-target validation.
    fn identity(&mut self, id: u64) -> Result<(NodeState, Option<u64>)> {
        let Some(mut statement) = self.readers.take(ReaderShape::Identity) else {
            return identity_scalar(self.table, &self.generation, self.revision, id);
        };
        note_reader_execution();
        let outcome = identity_read(&self.generation, self.revision, &mut statement, id);
        self.finish_reader(ReaderShape::Identity, statement, outcome)
    }
    /// Graph/adjacency read: the reused single-table reader when retained, else
    /// HEAD's per-expansion read_graph_node.
    fn graph_record(&mut self, id: u64) -> Result<GraphNodeRecord> {
        let Some(mut statement) = self.readers.take(ReaderShape::Graph) else {
            return self.table.read_graph_node(&self.generation, id);
        };
        note_reader_execution();
        let outcome = graph_read(self.table, &self.generation, &mut statement, id);
        self.finish_reader(ReaderShape::Graph, statement, outcome)
    }
    fn exact(
        &mut self,
        query: &[f32],
        k: usize,
        filter: &SearchFilter<'_>,
    ) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();
        reserve(&mut results, k)?;
        match filter {
            SearchFilter::Equals(rowid) => self.exact_scan(query, k, Some(*rowid), &mut results)?,
            SearchFilter::In(ids) => {
                for &rowid in *ids {
                    self.exact_scan(query, k, Some(rowid), &mut results)?;
                }
            }
            SearchFilter::None => self.exact_scan(query, k, None, &mut results)?,
        }
        Ok(results)
    }
    /// Exact stream scan: the unfiltered reader for a full stream, the point
    /// reader for one rowid probe (including IN loops and misses).
    fn exact_scan(
        &mut self,
        query: &[f32],
        k: usize,
        rowid: Option<u64>,
        results: &mut Vec<SearchResult>,
    ) -> Result<()> {
        if rowid.is_some_and(|r| r > i64::MAX as u64) {
            return Ok(());
        }
        let shape = if rowid.is_some() {
            ReaderShape::ExactPoint
        } else {
            ReaderShape::ExactUnfiltered
        };
        let Some(mut statement) = self.readers.take(shape) else {
            let mut statement = self.table.connection().prepare(&eligible_sql(
                self.table,
                rowid.is_some(),
                true,
            ))?;
            let outcome = self.exact_stream(&mut statement, query, k, rowid, results);
            return finish(statement, outcome);
        };
        note_reader_execution();
        let outcome = self.exact_stream(&mut statement, query, k, rowid, results);
        self.finish_reader(shape, statement, outcome)
    }
    /// Shared exact-stream body for both the retained reader and HEAD's fallback.
    fn exact_stream(
        &mut self,
        statement: &mut Statement<'static>,
        query: &[f32],
        k: usize,
        rowid: Option<u64>,
        results: &mut Vec<SearchResult>,
    ) -> Result<()> {
        reset_reader(statement)?;
        statement.bind_i64(1, checked_id(self.generation.generation_id)?)?;
        if let Some(rowid) = rowid {
            statement.bind_i64(2, checked_id(rowid)?)?;
        }
        // Charge point misses too: a huge IN set cannot evade the visit bound.
        if rowid.is_some() {
            self.budget.visit()?;
        }
        let mut count = 0usize;
        while statement.step()? == Step::Row {
            if rowid.is_none() {
                self.budget.visit()?;
            }
            count += 1;
            if count
                > usize::try_from(self.generation.eligible_count)
                    .map_err(|_| too_big("DiskANN eligible population overflow"))?
                || (rowid.is_some() && count > 1)
            {
                return Err(corrupt("DiskANN eligible count/rowid uniqueness mismatch"));
            }
            let (_, label) = eligible_identity(&self.generation, statement)?;
            check_blob(statement, 7, 8, self.table.vector_bytes())?;
            let metric = self.table.space().distance_type;
            let dimension = self.table.space().dim;
            let mut distance = 0.0;
            statement.with_column_blob(8, |bytes| {
                with_decoded_vector(bytes, dimension, &mut self.decode, &mut |vector| {
                    let vector = vector
                        .validate(dimension)
                        .map_err(|_| corrupt("DiskANN eligible vector invalid"))?;
                    distance = score(metric, query, vector)?;
                    Ok(())
                })
            })?;
            insert_best(results, SearchResult::new(distance, label), k);
        }
        if rowid.is_none() && count as u64 != self.generation.eligible_count {
            return Err(corrupt("DiskANN published eligible count mismatch"));
        }
        Ok(())
    }
    /// Accepted-row projection: one point-reader execution per projected row.
    fn project(&mut self, rowid: u64) -> Result<Vec<u8>> {
        self.budget.visit()?;
        let Some(mut statement) = self.readers.take(ReaderShape::ExactPoint) else {
            let mut statement = self
                .table
                .connection()
                .prepare(&eligible_sql(self.table, true, true))?;
            let outcome = self.project_stream(&mut statement, rowid);
            return finish(statement, outcome);
        };
        note_reader_execution();
        let outcome = self.project_stream(&mut statement, rowid);
        self.finish_reader(ReaderShape::ExactPoint, statement, outcome)
    }
    /// Shared accepted-row point read for both the retained reader and HEAD's
    /// fallback; returns the owned projection bytes.
    fn project_stream(
        &mut self,
        statement: &mut Statement<'static>,
        rowid: u64,
    ) -> Result<Vec<u8>> {
        reset_reader(statement)?;
        statement.bind_i64(1, checked_id(self.generation.generation_id)?)?;
        statement.bind_i64(2, checked_id(rowid)?)?;
        if statement.step()? != Step::Row {
            return Err(corrupt("DiskANN accepted immutable projection missing"));
        }
        eligible_identity(&self.generation, statement)?;
        let bytes = bounded_blob(
            statement,
            7,
            8,
            self.table.vector_bytes(),
            true,
            "accepted vector",
        )?;
        with_decoded_vector(
            &bytes,
            self.table.space().dim,
            &mut self.decode,
            &mut |_| Ok(()),
        )?;
        if statement.step()? != Step::Done {
            return Err(corrupt("DiskANN accepted rowid not unique"));
        }
        Ok(bytes)
    }

    /// One chunk of HEAD's default with_vectors_batch loop: a rowid read and a
    /// vector read, each charging one visit before its validation.
    fn scalar_batch_slot(
        &mut self,
        index: usize,
        id: u64,
        need_rowid: bool,
        action: &mut VectorBatchAction<'_>,
    ) -> Result<()> {
        let rowid = if need_rowid { self.rowid(id)? } else { None };
        self.with_vector(id, &mut |vector| action(index, id, rowid, vector))
    }

    /// Combined-reader batch slot: one execution per id replaces HEAD's identity
    /// plus vector pair. The rowid charge and identity validation still precede
    /// the vector charge and payload validation, so totals match HEAD exactly.
    fn batch_slot(
        &mut self,
        index: usize,
        id: u64,
        need_rowid: bool,
        action: &mut VectorBatchAction<'_>,
    ) -> Result<()> {
        let Some(mut statement) = self.readers.take(ReaderShape::Combined) else {
            return self.scalar_batch_slot(index, id, need_rowid, action);
        };
        note_reader_execution();
        let outcome = (|| {
            reset_reader(&mut statement)?;
            bind_node(&self.generation, &mut statement, id)?;
            if statement.step()? != Step::Row {
                return Err(corrupt("DiskANN referenced vector node/entry missing"));
            }
            let identity = if need_rowid {
                self.budget.visit()?;
                classify(&self.generation, self.revision, &statement, id)?
            } else {
                classify(&self.generation, self.revision, &statement, id)?
            };
            self.budget.visit()?;
            check_blob(&statement, 7, 8, self.table.vector_bytes())?;
            let dimension = self.table.space().dim;
            let decode = &mut self.decode;
            statement.with_column_blob(8, |bytes| {
                let mut deliver = |vector: StoredVector<'_>| {
                    deliver_batch_vector(action, index, id, need_rowid, identity.1, vector)
                };
                with_decoded_vector(bytes, dimension, decode, &mut deliver)
            })?;
            if statement.step()? != Step::Done {
                return Err(corrupt("DiskANN query vector node not unique"));
            }
            Ok(())
        })();
        self.finish_reader(ReaderShape::Combined, statement, outcome)
    }
}

/// Delivers one combined-reader vector to a batch action slot with exactly the
/// rowid semantics of diskann_core's default loop: the resolved identity rowid is
/// passed only in `need_rowid` mode and `None` is passed in vector-only mode.
fn deliver_batch_vector(
    action: &mut VectorBatchAction<'_>,
    index: usize,
    id: u64,
    need_rowid: bool,
    identity_rowid: Option<u64>,
    vector: StoredVector<'_>,
) -> Result<()> {
    let rowid = if need_rowid { identity_rowid } else { None };
    action(index, id, rowid, vector)
}

fn check_blob(
    statement: &Statement<'_>,
    length: usize,
    payload: usize,
    expected: usize,
) -> Result<()> {
    if nonnegative(statement, length, "query vector length")? != expected as u64
        || statement.column_type(payload)? != ffi::SQLITE_BLOB as i32
    {
        return Err(corrupt(
            "DiskANN query vector has an invalid byte length/type",
        ));
    }
    Ok(())
}
fn score(metric: DistanceType, query: &[f32], vector: &[f32]) -> Result<f32> {
    // Dimension/finite proof precedes these unchecked native SIMD operations.
    let distance = match metric {
        DistanceType::L2 => ops::l2_sq_f32(query, vector),
        DistanceType::Cosine => ops::ip_dist_f32(query, vector),
        DistanceType::InnerProduct => {
            return Err(IndexError::new("DiskANN does not support inner product"))
        }
    };
    if distance.is_finite() {
        Ok(distance)
    } else {
        Err(corrupt("DiskANN non-finite query distance"))
    }
}
fn insert_best(results: &mut Vec<SearchResult>, result: SearchResult, k: usize) {
    let position = results
        .binary_search_by(|other| {
            other
                .distance
                .total_cmp(&result.distance)
                .then(other.rowid.cmp(&result.rowid))
        })
        .unwrap_or_else(|p| p);
    if position < k {
        if results.len() == k {
            results.pop();
        }
        results.insert(position, result);
    }
}

impl GraphStore for PublishedGraph<'_> {
    fn allocate(&mut self, _: u64, _: &[f32]) -> Result<u64> {
        Err(unsupported())
    }
    fn internal_id(&mut self, rowid: u64) -> Result<u64> {
        let mut statement = self
            .table
            .connection()
            .prepare(&eligible_sql(self.table, true, false))?;
        let outcome = (|| {
            statement.bind_i64(1, checked_id(self.generation.generation_id)?)?;
            statement.bind_i64(2, checked_id(rowid)?)?;
            if statement.step()? != Step::Row {
                return Err(IndexError::new("DiskANN published rowid absent"));
            }
            let (id, _) = eligible_identity(&self.generation, &statement)?;
            if statement.step()? != Step::Done {
                return Err(corrupt("DiskANN published rowid not unique"));
            }
            Ok(id)
        })();
        finish(statement, outcome)
    }
    fn rowid(&mut self, id: u64) -> Result<Option<u64>> {
        // InlineFilterSearch requests rowid for every expansion, including
        // distance-cache hits. Together with with_vector fetch charges this
        // conservatively covers core visits in the SAME fallback budget.
        self.budget.visit()?;
        self.identity(id).map(|(_, rowid)| rowid)
    }
    fn state(&mut self, id: u64) -> Result<NodeState> {
        self.identity(id).map(|(state, _)| state)
    }
    fn vector(&mut self, id: u64) -> Result<Vec<f32>> {
        let dimension = self.table.space().dim;
        let mut output = Vec::new();
        reserve(&mut output, dimension)?;
        self.with_vector(id, &mut |vector| {
            output.extend_from_slice(
                vector
                    .validate(dimension)
                    .map_err(|_| corrupt("DiskANN query vector invalid"))?,
            );
            Ok(())
        })?;
        Ok(output)
    }
    fn remaining_vector_visits(&self) -> usize {
        self.budget.remaining()
    }
    fn with_vector(
        &mut self,
        id: u64,
        action: &mut dyn FnMut(StoredVector<'_>) -> Result<()>,
    ) -> Result<()> {
        self.budget.visit()?;
        let Some(mut statement) = self.readers.take(ReaderShape::Combined) else {
            return vector_scalar(
                self.table,
                &self.generation,
                self.revision,
                id,
                &mut self.decode,
                action,
            );
        };
        note_reader_execution();
        let outcome = vector_read(
            self.table,
            &self.generation,
            self.revision,
            &mut statement,
            id,
            &mut self.decode,
            action,
        );
        self.finish_reader(ReaderShape::Combined, statement, outcome)
    }
    fn with_vectors_batch(
        &mut self,
        ids: &[u64],
        need_rowid: bool,
        action: &mut VectorBatchAction<'_>,
    ) -> Result<()> {
        note_batch_call();
        for (index, &id) in ids.iter().enumerate() {
            self.batch_slot(index, id, need_rowid, action)?;
        }
        Ok(())
    }
    fn neighbors(&mut self, id: u64) -> Result<Vec<u64>> {
        self.budget.visit()?;
        let node = self.graph_record(id)?;
        // Range/order/duplicates/self are checked by the frozen accessor. Check
        // each bounded target identity too: a dangling edge is corruption, not EOF.
        // These target validations are zero-charge on the reused identity reader.
        for &neighbor in &node.neighbors {
            self.identity(neighbor)?;
        }
        Ok(node.neighbors)
    }
    #[cfg(test)]
    fn set_neighbors(&mut self, _: u64, _: &[u64]) -> Result<()> {
        Err(unsupported())
    }
    fn mark_delete(&mut self, _: u64) -> Result<()> {
        Err(unsupported())
    }
    fn start_points(&mut self) -> Result<Vec<u64>> {
        let root = self
            .generation
            .root_local_id
            .ok_or_else(|| corrupt("DiskANN nonempty generation has no frozen root"))?;
        if self.identity(root)?.0 != NodeState::Frozen {
            return Err(corrupt("DiskANN start point is not frozen"));
        }
        Ok(vec![root])
    }
}

/// Whole-core admission floor for the route's effective list. None means the
/// probe configuration is invalid; admission then keeps the scalar path so core
/// reports exactly the error HEAD would report.
fn workspace_floor(table: &DiskAnnTable, budget: &Budget, effective_l: usize) -> Result<usize> {
    let options = table.options();
    GraphConfig {
        dimension: table.space().dim,
        distance: table.space().distance_type,
        max_degree: options.degree,
        construction_l: options.build_list_size,
        search_l: options.search_list_size,
        alpha: options.alpha,
        limits: ResourceLimits {
            max_visits: options.max_visits,
            max_cached_vectors: 2048,
            max_vector_bytes: budget.head_bytes(),
        },
    }
    .workspace_floor(effective_l)
}

pub(crate) fn knn(
    table: &DiskAnnTable,
    query: &[f32],
    k: usize,
    override_l: Option<usize>,
    filter: SearchFilter<'_>,
    project_vector: bool,
) -> Result<QueryRows> {
    if override_l.is_some_and(|l| l == 0 || l > table.options().max_visits) {
        return Err(IndexError::new(
            "DiskANN search list override must be positive and no greater than max_visits",
        ));
    }
    if query.len() != table.space().dim {
        return Err(IndexError::new("DiskANN vector dimension mismatch"));
    }
    // Capture the one-shot Debug demotion request for exactly this operation, so
    // a host hook applies to the next query even when that query takes an early
    // route (where nothing is demoted). Always false in release builds.
    let force_demote = take_force_demote();
    let mut normalized = Vec::new();
    reserve(&mut normalized, query.len())?;
    normalized.extend_from_slice(query);
    normalize_input(
        &mut normalized,
        table.space().dim,
        table.space().distance_type,
    )?;
    let (metadata, anchor) = table.read_metadata()?;
    let Some(id) = metadata.published_generation else {
        Budget::admit(table, 0, &filter, project_vector)?;
        return Ok(QueryRows {
            results: Vec::new(),
            vectors: project_vector.then(Vec::new),
            anchor,
        });
    };
    let generation = table.read_generation(id)?;
    if generation.state != 1 || generation.source_revision > metadata.revision {
        return Err(corrupt(
            "DiskANN published generation identity/revision invalid",
        ));
    }
    if let Some(root) = generation.root_local_id {
        table.read_graph_node(&generation, root)?;
    }
    let eligible = usize::try_from(generation.eligible_count)
        .map_err(|_| too_big("DiskANN eligible population overflow"))?;
    let count = k.min(eligible).min(match &filter {
        SearchFilter::Equals(_) => 1,
        SearchFilter::In(ids) => ids.len(),
        SearchFilter::None => usize::MAX,
    });
    let budget = Budget::admit(table, count, &filter, project_vector)?;
    if count == 0 {
        return Ok(QueryRows {
            results: Vec::new(),
            vectors: project_vector.then(Vec::new),
            anchor,
        });
    }
    let l = effective_l(override_l, table.options().search_list_size, count);
    record_search_l(l);
    let filtered_small = match &filter {
        SearchFilter::Equals(_) => true,
        SearchFilter::In(ids) => ids.len() <= l,
        SearchFilter::None => false,
    };
    let exact_route = eligible <= l || count >= eligible || filtered_small;
    let mut readers = Readers::new();
    readers.arm_force_demote(force_demote);
    let mut graph = PublishedGraph {
        table,
        generation,
        revision: metadata.revision,
        budget,
        readers,
        decode: Vec::new(),
    };
    let outcome = (|| -> Result<QueryOutcome> {
        let mut results = if exact_route {
            // No core admission: the exact route only has to keep the owner, the
            // exact/shared readers and the headroom inside cache_bytes.
            let plan = exact_reader_plan(&filter, project_vector);
            let allowance = graph.budget.reader_allowance(None);
            graph.admit_readers(&plan, allowance);
            graph.exact(&normalized, count, &filter)?
        } else {
            // Eager route-aware envelope: the whole core floor and the reprepare
            // headroom are reserved before GraphConfig copies its allowance.
            let floor = workspace_floor(table, &graph.budget, l).ok();
            let plan = graph_reader_plan(&filter, project_vector);
            let allowance = floor.map_or(0, |floor| graph.budget.reader_allowance(Some(floor)));
            graph.admit_readers(&plan, allowance);
            let options = table.options();
            let config = GraphConfig {
                dimension: table.space().dim,
                distance: table.space().distance_type,
                max_degree: options.degree,
                construction_l: options.build_list_size,
                search_l: options.search_list_size,
                alpha: options.alpha,
                limits: ResourceLimits {
                    max_visits: options.max_visits,
                    max_cached_vectors: 2048,
                    max_vector_bytes: graph.budget.core_bytes(graph.readers.admitted_total()),
                },
            };
            let set = match &filter {
                SearchFilter::In(ids) => Some(*ids),
                _ => None,
            };
            let found =
                diskann_core::search(&mut graph, &config, &normalized, count, override_l, set)?;
            if found.len() < count {
                // Same owner budget; never reset visits or search raw current rows.
                graph.exact(&normalized, count, &filter)?
            } else {
                found
            }
        };
        results.sort_unstable_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then(a.rowid.cmp(&b.rowid))
        });
        let vectors = if project_vector {
            let mut vectors = Vec::new();
            reserve(&mut vectors, results.len())?;
            for result in &results {
                vectors.push(graph.project(result.rowid)?);
            }
            Some(vectors)
        } else {
            None
        };
        Ok((results, vectors))
    })();
    let (results, vectors) = graph.finish_readers(outcome)?;
    Ok(QueryRows {
        results,
        vectors,
        anchor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn bounded_topk_has_deterministic_public_label_ties() {
        let mut results = Vec::with_capacity(2);
        for rowid in [8, 0, i64::MAX as u64, 3] {
            insert_best(&mut results, SearchResult::new(1.0, rowid), 2);
        }
        assert_eq!(results.iter().map(|r| r.rowid).collect::<Vec<_>>(), [0, 3]);
        insert_best(&mut results, SearchResult::new(0.0, 7), 2);
        assert_eq!(results.iter().map(|r| r.rowid).collect::<Vec<_>>(), [7, 0]);
    }
    #[test]
    fn zero_k_does_not_insert() {
        let mut results = Vec::new();
        insert_best(&mut results, SearchResult::new(0.0, 0), 0);
        assert!(results.is_empty());
    }

    #[test]
    fn effective_list_uses_override_then_search_list_and_never_below_k() {
        assert_eq!(effective_l(Some(10), 64, 10), 10);
        assert_eq!(effective_l(None, 64, 10), 64);
        assert_eq!(effective_l(None, 4, 16), 16);
        assert_eq!(effective_l(Some(8), 4, 16), 16);
        assert_eq!(effective_l(Some(100), 64, 10), 100);
        assert_eq!(effective_l(None, 64, 0), 64);
    }

    #[test]
    fn reader_allowance_saturates_and_reserves_floor_and_headroom() {
        assert_eq!(reader_allowance(0, Some(0)), 0);
        assert_eq!(reader_allowance(0, None), 0);
        // A head budget below the floor reservation yields zero (scalar-only).
        assert_eq!(reader_allowance(10_000, Some(20_000)), 0);
        // Exactly the reservation fits nothing.
        assert_eq!(reader_allowance(REPREPARE_HEADROOM, None), 0);
        assert_eq!(reader_allowance(REPREPARE_HEADROOM, Some(0)), 0);
        // One spare byte is the whole allowance, never more than MAX_READER_BYTES.
        assert_eq!(reader_allowance(REPREPARE_HEADROOM + 1, None), 1);
        assert_eq!(reader_allowance(REPREPARE_HEADROOM + 100, Some(30)), 70);
        assert_eq!(
            reader_allowance(REPREPARE_HEADROOM + MAX_READER_BYTES * 4, Some(0)),
            MAX_READER_BYTES
        );
    }

    #[test]
    fn reader_cap_is_a_reserve_bound_that_still_admits_partial_or_none() {
        // No allowance ever exceeds the reserve bound, whatever the budget or floor.
        for head in [
            0usize,
            1,
            REPREPARE_HEADROOM,
            100_000,
            1_000_000,
            64 * 1024 * 1024,
        ] {
            for floor in [None, Some(0usize), Some(4_096), Some(head)] {
                assert!(reader_allowance(head, floor) <= MAX_READER_BYTES);
            }
        }
        // A budget that can cover the whole set admits it in full; the cap binds
        // only after that, independently of additional head room.
        let full = REPREPARE_HEADROOM + MAX_READER_BYTES;
        assert_eq!(reader_allowance(full, Some(0)), MAX_READER_BYTES);
        assert_eq!(
            reader_allowance(full + 4 * MAX_READER_BYTES, Some(0)),
            MAX_READER_BYTES
        );
        // A tight budget still yields the partial fallback: shapes are admitted in
        // priority order and the first shape that does not fit ends admission.
        let mut partial =
            ReaderAllowance::new(reader_allowance(REPREPARE_HEADROOM + 32 * 1024, Some(0)));
        assert_eq!(partial.offer(Some(20 * 1024)), Some(20 * 1024));
        assert_eq!(partial.offer(Some(20 * 1024)), None);
        assert_eq!(partial.offer(Some(1)), None);
        // Below the reprepare reservation nothing is admitted at all.
        let mut none = ReaderAllowance::new(reader_allowance(REPREPARE_HEADROOM, None));
        assert_eq!(none.offer(Some(1)), None);
    }

    #[test]
    fn core_bytes_keeps_the_headroom_only_when_a_reader_is_admitted() {
        let head = 1_000_000;
        // No reader: HEAD's exact value is passed so every HEAD admission
        // outcome is preserved at every cache_bytes.
        assert_eq!(core_bytes(head, 0), head);
        assert_eq!(core_bytes(head, 1), head - 1 - REPREPARE_HEADROOM);
        assert_eq!(
            core_bytes(head, MAX_READER_BYTES),
            head - MAX_READER_BYTES - REPREPARE_HEADROOM
        );
    }

    #[test]
    fn disjointness_holds_at_existing_budget_boundaries() {
        // owner values around the recorded fixtures; the head is cache - owner.
        for &(cache_bytes, owner) in &[
            (131_072usize, 66_000usize),
            (131_072, 131_072),
            (64 * 1024 * 1024, 84_000),
            (24_576, 9_000),
            (4_096, 4_000),
        ] {
            let head = cache_bytes - owner;
            for &floor in &[0usize, 1, 4_096, head / 2, head] {
                let allowance = reader_allowance(head, Some(floor));
                assert!(
                    allowance
                        <= head
                            .saturating_sub(floor)
                            .saturating_sub(REPREPARE_HEADROOM)
                );
                // The exact route reserve: owner + readers + headroom <= cache_bytes
                // whenever a reader is admitted; otherwise the HEAD scalar path only
                // enforces HEAD's own owner <= cache_bytes check.
                let exact_allowance = reader_allowance(head, None);
                if exact_allowance > 0 {
                    assert!(owner + exact_allowance + REPREPARE_HEADROOM <= cache_bytes);
                    assert!(owner + exact_allowance + REPREPARE_HEADROOM <= head + owner);
                }
                // The graph route: admitted readers + headroom + core <= head, so
                // core never receives less than the floor it was reserved for.
                if allowance > 0 {
                    let core = core_bytes(head, allowance);
                    assert!(core >= floor);
                    // owner + admitted readers + headroom + core_bytes <= cache_bytes
                    assert_eq!(owner + allowance + REPREPARE_HEADROOM + core, cache_bytes);
                } else {
                    // No admitted reader keeps HEAD's scalar admission value.
                    assert_eq!(core_bytes(head, 0), head);
                }
            }
        }
    }

    #[test]
    fn allowance_cursor_is_priority_ordered_and_partial() {
        let mut cursor = ReaderAllowance::new(1_000);
        assert!(!cursor.closed());
        assert_eq!(cursor.offer(Some(200)), Some(200));
        assert!(!cursor.closed());
        assert_eq!(cursor.offer(Some(801)), None);
        // The first shape that does not fit ends admission, so admit_readers
        // stops before building or preparing any lower-priority shape.
        assert!(cursor.closed());
        assert_eq!(cursor.offer(Some(1)), None);
        assert!(cursor.closed());
    }

    #[test]
    fn allowance_cursor_skips_unavailable_measurements_without_closing() {
        let mut cursor = ReaderAllowance::new(1_000);
        assert_eq!(cursor.offer(None), None);
        assert_eq!(cursor.offer(Some(1_000)), Some(1_000));
        assert_eq!(cursor.offer(None), None);
    }

    #[test]
    fn zero_allowance_admits_no_reader_and_keeps_head_core_budget() {
        let mut cursor = ReaderAllowance::new(0);
        assert_eq!(cursor.offer(Some(1)), None);
        let mut readers = Readers::new();
        assert!(readers.take(ReaderShape::Graph).is_none());
        assert_eq!(readers.admitted_total(), 0);
        assert_eq!(core_bytes(1_000_000, readers.admitted_total()), 1_000_000);
    }

    #[test]
    fn batch_delivery_passes_a_rowid_only_in_need_rowid_mode() {
        let mut calls: Vec<(usize, u64, Option<u64>)> = Vec::new();
        let mut action = |index: usize, id: u64, rowid: Option<u64>, _vector: StoredVector<'_>| {
            calls.push((index, id, rowid));
            Ok(())
        };
        let data = [0.0f32, 1.0];
        // diskann_core's default loop passes None in vector-only mode, so the
        // combined reader must withhold a resolved identity rowid there too.
        deliver_batch_vector(
            &mut action,
            0,
            7,
            false,
            Some(70),
            StoredVector::Unvalidated(&data),
        )
        .unwrap();
        deliver_batch_vector(
            &mut action,
            1,
            8,
            true,
            Some(80),
            StoredVector::Unvalidated(&data),
        )
        .unwrap();
        deliver_batch_vector(
            &mut action,
            2,
            9,
            false,
            None,
            StoredVector::Unvalidated(&data),
        )
        .unwrap();
        deliver_batch_vector(
            &mut action,
            3,
            10,
            true,
            None,
            StoredVector::Unvalidated(&data),
        )
        .unwrap();
        assert_eq!(
            calls,
            [(0, 7, None), (1, 8, Some(80)), (2, 9, None), (3, 10, None)]
        );
    }

    #[test]
    fn demotion_prefers_the_largest_retained_readers() {
        // Largest first; ties keep shape priority (lower index first).
        assert_eq!(
            demotion_order(&[Some(10), Some(20), None, Some(20), None]),
            vec![1, 3, 0]
        );
        // Aggregate headroom boundary: exactly the headroom is still inside.
        let admitted = 10_000;
        assert!(!demotion_needed(admitted + REPREPARE_HEADROOM, admitted));
        assert!(demotion_needed(admitted + REPREPARE_HEADROOM + 1, admitted));
        // Partial drain restores the admitted envelope.
        let measured = [Some(6_000), Some(6_000), Some(2_000), None, None];
        let mut aggregate: usize = 14_000;
        let mut drained = Vec::new();
        for index in demotion_order(&measured) {
            if aggregate <= admitted {
                break;
            }
            aggregate -= measured[index].unwrap_or(0);
            drained.push(index);
        }
        assert_eq!(drained, vec![0]);
        assert_eq!(aggregate, 8_000);
    }

    #[test]
    fn graph_plan_is_filter_aware_and_never_repeats_a_shape() {
        let plan = graph_reader_plan(&SearchFilter::None, false);
        assert_eq!(
            plan.as_slice(),
            &[
                ReaderShape::Graph,
                ReaderShape::Identity,
                ReaderShape::Combined,
                ReaderShape::ExactUnfiltered
            ]
        );
        let plan = graph_reader_plan(&SearchFilter::None, true);
        assert_eq!(
            plan.as_slice(),
            &[
                ReaderShape::Graph,
                ReaderShape::Identity,
                ReaderShape::Combined,
                ReaderShape::ExactUnfiltered,
                ReaderShape::ExactPoint
            ]
        );
        // The point shape is both the filtered fallback and the projection
        // reader; it is planned once.
        let plan = graph_reader_plan(&SearchFilter::Equals(7), true);
        assert_eq!(
            plan.as_slice(),
            &[
                ReaderShape::Graph,
                ReaderShape::Identity,
                ReaderShape::Combined,
                ReaderShape::ExactPoint
            ]
        );
        let ids = HashSet::from([1u64, 2]);
        let plan = graph_reader_plan(&SearchFilter::In(&ids), false);
        assert_eq!(
            plan.as_slice(),
            &[
                ReaderShape::Graph,
                ReaderShape::Identity,
                ReaderShape::Combined,
                ReaderShape::ExactPoint
            ]
        );
    }

    #[test]
    fn exact_plan_needs_exactly_one_stream_shape_plus_projection() {
        assert_eq!(
            exact_reader_plan(&SearchFilter::None, false).as_slice(),
            &[ReaderShape::ExactUnfiltered]
        );
        assert_eq!(
            exact_reader_plan(&SearchFilter::None, true).as_slice(),
            &[ReaderShape::ExactUnfiltered, ReaderShape::ExactPoint]
        );
        let ids = HashSet::from([1u64, 2]);
        assert_eq!(
            exact_reader_plan(&SearchFilter::In(&ids), false).as_slice(),
            &[ReaderShape::ExactPoint]
        );
        assert_eq!(
            exact_reader_plan(&SearchFilter::Equals(3), true).as_slice(),
            &[ReaderShape::ExactPoint]
        );
    }
}
