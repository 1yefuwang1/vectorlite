//! Operation-local, read-only search of one published Format5 generation.
//! The metadata cursor anchors every read and all accepted projected bytes.
//! Native ID zero is ordinary data; the only frozen start point is explicit N.
use std::mem::size_of;

use crate::core::{SearchFilter, SearchResult};
use crate::diskann_core::{
    self, GraphConfig, GraphStore, NodeState, ResourceLimits, StoredVector, VectorBatchAction,
};
use crate::diskann_store::codec::{
    blob_projection, bounded_blob, checked_id, checked_product, corrupt, decode_slots_streamed,
    nonnegative, normalize_input, reserve, too_big, with_decoded_vector,
};
use crate::diskann_store::{DiskAnnTable, Generation, QueryRows, StorageNames};
use crate::ffi;
use crate::index_error::IndexError;
use crate::ops;
use crate::sqlite::{qualified_name, Statement, Step};
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
/// Bytes charged for one carry ring slot: an INTEGER-only native local id plus
/// an optional public rowid. The carry is operation-scoped and caches no
/// payload, only these two INTEGER columns.
const CARRY_SLOT_BYTES: usize = size_of::<(u64, Option<u64>)>();
/// Fixed container slack charged with the ring (Vec header plus cursor state).
const CARRY_CONTAINER_BYTES: usize = size_of::<CarryRing>();
/// Bytes charged per recorded adjacency identity pair: the native local id and
/// the entry id the slot froze at import.
const SLOT_PAIR_BYTES: usize = size_of::<(u64, u64)>();
/// Fixed container slack charged with the pair cache (Vec header).
const SLOT_PAIR_CONTAINER_BYTES: usize = size_of::<Vec<(u64, u64)>>();

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
    /// One-shot request used by Debug-only contract tests to disable the carry
    /// for the next query, exercising the D6 tight-budget path exactly; consumed
    /// at operation start.
    static FORCE_CARRY_OFF: AtomicBool = AtomicBool::new(false);
    /// One-shot request used by Debug-only contract tests to force the pair cache
    /// off for the next query, exercising the tight-budget miss path exactly;
    /// consumed at operation start.
    static FORCE_PAIR_CACHE_OFF: AtomicBool = AtomicBool::new(false);
    /// Reader shapes whose next admission prepare attempt must be forced to fail,
    /// one bit per ReaderShape::ALL index.
    static FAIL_READER_PREPARE: AtomicU64 = AtomicU64::new(0);
    /// Three-table identity executions. Format-served routes must report zero;
    /// the D3 fallback's executions are counted here and in FALLBACK_3TABLE.
    static IDENTITY_3TABLE: AtomicU64 = AtomicU64::new(0);
    /// Every three-table execution performed by the D3 fallback route.
    static FALLBACK_3TABLE: AtomicU64 = AtomicU64::new(0);
    /// NodeExpand single-table reader/scalar executions.
    static NODE_EXPAND: AtomicU64 = AtomicU64::new(0);
    /// NodeScore single-table reader/scalar executions.
    static NODE_SCORE: AtomicU64 = AtomicU64::new(0);
    /// NodeIdentity single-table reader/scalar executions.
    static NODE_IDENTITY: AtomicU64 = AtomicU64::new(0);
    /// Fixed non-traversal statement executions (C3 setup reads).
    static SETUP_READS: AtomicU64 = AtomicU64::new(0);
    /// Operations that admitted the retired-entry set through the reader cursor.
    static RETIRED_ADMISSIONS: AtomicU64 = AtomicU64::new(0);
    /// Operations whose retired set could not be admitted (whole-query fallback).
    static RETIRED_FALLBACKS: AtomicU64 = AtomicU64::new(0);
    /// Retired-entry set size resolved by the most recent graph-route query
    /// (zero when no member exists, the count when the set fell back).
    static RETIRED_SIZE: AtomicU64 = AtomicU64::new(0);
    /// Post-processing decision reads (charged rowid() resolutions).
    static DECISION_READS: AtomicU64 = AtomicU64::new(0);
    /// Neighbour-expansion reads (one per neighbors() call).
    static EXPANSIONS: AtomicU64 = AtomicU64::new(0);
    /// Scored candidate reads (one per single-table score execution).
    static SCORED_CANDIDATES: AtomicU64 = AtomicU64::new(0);
    /// Projection reads: one ExactPoint execution per projected result row.
    static PROJECTION_READS: AtomicU64 = AtomicU64::new(0);
    /// Resolved-rowid carry counters: consulted hits, real charged reads on a
    /// miss, admitted operations and the bytes of the last admitted shape.
    static CARRY_HITS: AtomicU64 = AtomicU64::new(0);
    static CARRY_MISSES: AtomicU64 = AtomicU64::new(0);
    static CARRY_ADMISSIONS: AtomicU64 = AtomicU64::new(0);
    static CARRY_BYTES: AtomicU64 = AtomicU64::new(0);
    /// Pair-cache counters: scored ids resolved from the admitted expansion
    /// cache, scored ids that fell back to one NodeIdentity read, and operations
    /// that ran with the cache disabled (declined or forced off).
    static PAIR_HITS: AtomicU64 = AtomicU64::new(0);
    static PAIR_MISSES: AtomicU64 = AtomicU64::new(0);
    static PAIR_DISABLED_OPERATIONS: AtomicU64 = AtomicU64::new(0);
    /// Cumulative owner visits across completed queries.
    static OWNER_VISITS: AtomicU64 = AtomicU64::new(0);
    /// Owner visits of the most recently completed query.
    static LAST_QUERY_VISITS: AtomicU64 = AtomicU64::new(0);

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
    pub(super) fn take_force_carry_off() -> bool {
        FORCE_CARRY_OFF.swap(false, Ordering::Relaxed)
    }
    pub(super) fn take_force_pair_cache_off() -> bool {
        FORCE_PAIR_CACHE_OFF.swap(false, Ordering::Relaxed)
    }
    pub(super) fn note_carry_admission(bytes: usize) {
        CARRY_ADMISSIONS.fetch_add(1, Ordering::Relaxed);
        CARRY_BYTES.store(u64::try_from(bytes).unwrap_or(u64::MAX), Ordering::Relaxed);
    }
    /// Consumes one armed forced-failure bit for a shape.
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
    pub(super) fn note_verb(verb: super::Verb) {
        let counter = match verb {
            super::Verb::Identity3Table => &IDENTITY_3TABLE,
            super::Verb::Fallback3Table => &FALLBACK_3TABLE,
            super::Verb::NodeExpand => &NODE_EXPAND,
            super::Verb::NodeScore => &NODE_SCORE,
            super::Verb::NodeIdentity => &NODE_IDENTITY,
            super::Verb::Setup => &SETUP_READS,
            super::Verb::RetiredAdmission => &RETIRED_ADMISSIONS,
            super::Verb::RetiredFallback => &RETIRED_FALLBACKS,
            super::Verb::Decision => &DECISION_READS,
            super::Verb::Expansion => &EXPANSIONS,
            super::Verb::Scored => &SCORED_CANDIDATES,
            super::Verb::Projection => &PROJECTION_READS,
            super::Verb::CarryHit => &CARRY_HITS,
            super::Verb::CarryMiss => &CARRY_MISSES,
            super::Verb::PairHit => &PAIR_HITS,
            super::Verb::PairMiss => &PAIR_MISSES,
            super::Verb::PairDisabled => &PAIR_DISABLED_OPERATIONS,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }
    pub(super) fn note_retired_size(size: u64) {
        RETIRED_SIZE.store(size, Ordering::Relaxed);
    }
    pub(super) fn record_owner_visits(visits: usize) {
        let visits = u64::try_from(visits).unwrap_or(u64::MAX);
        OWNER_VISITS.fetch_add(visits, Ordering::Relaxed);
        LAST_QUERY_VISITS.store(visits, Ordering::Relaxed);
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
    /// Reports three-table identity executions (zero on format-served routes).
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_identity_3table_executions() -> u64 {
        IDENTITY_3TABLE.load(Ordering::Relaxed)
    }
    /// Reports every three-table execution performed by the D3 fallback route.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_fallback_3table_executions() -> u64 {
        FALLBACK_3TABLE.load(Ordering::Relaxed)
    }
    /// Reports single-table NodeExpand executions.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_node_expand_executions() -> u64 {
        NODE_EXPAND.load(Ordering::Relaxed)
    }
    /// Reports single-table NodeScore executions.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_node_score_executions() -> u64 {
        NODE_SCORE.load(Ordering::Relaxed)
    }
    /// Reports single-table NodeIdentity executions.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_node_identity_executions() -> u64 {
        NODE_IDENTITY.load(Ordering::Relaxed)
    }
    /// Reports fixed non-traversal setup reads.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_setup_reads() -> u64 {
        SETUP_READS.load(Ordering::Relaxed)
    }
    /// Reports retired-set reader admissions since process start.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_retired_admissions() -> u64 {
        RETIRED_ADMISSIONS.load(Ordering::Relaxed)
    }
    /// Reports whole-query retired-set fallbacks since process start.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_retired_fallbacks() -> u64 {
        RETIRED_FALLBACKS.load(Ordering::Relaxed)
    }
    /// Reports the size of the most recently admitted retired-entry set.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_retired_size() -> u64 {
        RETIRED_SIZE.load(Ordering::Relaxed)
    }
    /// Reports post-processing decision reads.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_decision_reads() -> u64 {
        DECISION_READS.load(Ordering::Relaxed)
    }
    /// Reports neighbour-expansion reads.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_expansions() -> u64 {
        EXPANSIONS.load(Ordering::Relaxed)
    }
    /// Reports scored candidate reads.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_scored_candidates() -> u64 {
        SCORED_CANDIDATES.load(Ordering::Relaxed)
    }
    /// Reports projection reads (one ExactPoint execution per projected row).
    ///
    /// A public knn_search() query always references the vector column, so the
    /// adapter always requests its vector payload: this is the per-result
    /// projection term the C3 identity must include alongside setup, expansion,
    /// scored-candidate and decision reads. Debug builds only; release artifacts
    /// export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_projection_reads() -> u64 {
        PROJECTION_READS.load(Ordering::Relaxed)
    }
    /// Reports carry hits (consulted resolutions that needed no SQL).
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_carry_hits() -> u64 {
        CARRY_HITS.load(Ordering::Relaxed)
    }
    /// Reports carry misses (real charged NodeIdentity reads).
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_carry_misses() -> u64 {
        CARRY_MISSES.load(Ordering::Relaxed)
    }
    /// Reports operations that admitted the carry through the reader cursor.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_carry_admissions() -> u64 {
        CARRY_ADMISSIONS.load(Ordering::Relaxed)
    }
    /// Reports the charged bytes of the most recently admitted carry shape.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_carry_bytes() -> u64 {
        CARRY_BYTES.load(Ordering::Relaxed)
    }
    /// Reports scored ids resolved from the admitted expansion pair cache.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_pair_cache_hits() -> u64 {
        PAIR_HITS.load(Ordering::Relaxed)
    }
    /// Reports scored ids that fell back to one NodeIdentity read because the
    /// pair cache had no entry for them (or was disabled for the operation).
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_pair_cache_misses() -> u64 {
        PAIR_MISSES.load(Ordering::Relaxed)
    }
    /// Reports operations that ran with the pair cache disabled: the D6
    /// allowance declined its fixed shape at a tight budget, or the Debug hook
    /// forced it off. Scoring then used the one-read-per-id fallback.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_pair_cache_disabled_operations() -> u64 {
        PAIR_DISABLED_OPERATIONS.load(Ordering::Relaxed)
    }
    /// Reports cumulative owner visits across completed queries.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_owner_visit_total() -> u64 {
        OWNER_VISITS.load(Ordering::Relaxed)
    }
    /// Reports owner visits of the most recently completed query.
    ///
    /// Debug builds only; release artifacts export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_last_query_visits() -> u64 {
        LAST_QUERY_VISITS.load(Ordering::Relaxed)
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
    /// Disables the resolved-rowid carry for the next knn operation.
    ///
    /// The hook is one-shot, consumed at operation start, and Debug builds only.
    /// It exercises exactly the D6 tight-budget path (carry not admitted, final
    /// loop falls back to charged NodeIdentity reads) without changing the
    /// operation's budget, results, or visit charges. Release artifacts export
    /// no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_force_carry_off() {
        FORCE_CARRY_OFF.store(true, Ordering::Relaxed);
    }
    /// Forces the pair cache off for the next knn operation.
    ///
    /// The hook is one-shot, consumed at operation start, and Debug builds only.
    /// It exercises exactly the D6 tight-budget path (pair cache not admitted, so
    /// every scored id falls back to one NodeIdentity read) without changing the
    /// operation's budget, results, or visit charges. Release artifacts export no
    /// instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_force_pair_cache_off() {
        FORCE_PAIR_CACHE_OFF.store(true, Ordering::Relaxed);
    }
    /// Forces the next admission prepare attempt of each selected reader shape to
    /// fail, so a query silently falls back to scalar readers.
    ///
    /// mask selects shapes by ReaderShape::ALL bit index: bit 0 NodeExpand, bit 1
    /// NodeScore, bit 2 NodeIdentity, bit 3 ExactUnfiltered, bit 4 ExactPoint.
    /// Each selected shape fails exactly one attempt and then behaves normally.
    /// The failure is a real sqlite3_prepare_v2 dispatch with invalid SQL, so
    /// vectorlite_test_prepare_attempts counts it while the successful-prepare
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
fn take_force_carry_off() -> bool {
    #[cfg(debug_assertions)]
    {
        instrumentation::take_force_carry_off()
    }
    #[cfg(not(debug_assertions))]
    {
        false
    }
}
fn take_force_pair_cache_off() -> bool {
    #[cfg(debug_assertions)]
    {
        instrumentation::take_force_pair_cache_off()
    }
    #[cfg(not(debug_assertions))]
    {
        false
    }
}
fn note_carry_admission(bytes: usize) {
    #[cfg(debug_assertions)]
    instrumentation::note_carry_admission(bytes);
    #[cfg(not(debug_assertions))]
    let _ = bytes;
}

/// Debug-only counter selector shared by the query adapter call sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verb {
    Identity3Table,
    Fallback3Table,
    NodeExpand,
    NodeScore,
    NodeIdentity,
    Setup,
    RetiredAdmission,
    RetiredFallback,
    // Noted only from Debug-only phase instrumentation, so it is never
    // constructed by a release build.
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    Decision,
    Expansion,
    Scored,
    Projection,
    CarryHit,
    CarryMiss,
    PairHit,
    PairMiss,
    /// Noted once per operation that ran with the pair cache disabled.
    PairDisabled,
}

/// Records one Debug-only counter event; a no-op in release builds.
fn note(verb: Verb) {
    #[cfg(debug_assertions)]
    instrumentation::note_verb(verb);
    #[cfg(not(debug_assertions))]
    let _ = verb;
}
fn note_retired_size(size: u64) {
    #[cfg(debug_assertions)]
    instrumentation::note_retired_size(size);
    #[cfg(not(debug_assertions))]
    let _ = size;
}
fn record_owner_visits(visits: usize) {
    #[cfg(debug_assertions)]
    instrumentation::record_owner_visits(visits);
    #[cfg(not(debug_assertions))]
    let _ = visits;
}

/// Instrumentation phase of one graph-route operation. It attributes rowid()
/// resolutions to setup, expansion, post-processing or the post-processing tail
/// without changing any charge or result.
#[cfg(debug_assertions)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Setup,
    Traversal,
    PostProcess,
    Final,
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
    /// Offers the operation's last, lowest-value shape (the carry).
    ///
    /// A non-fitting measurement declines WITHOUT closing admission, so the
    /// carry's size (which grows with the effective search list) can never end
    /// admission for anything prepared after it. A declined or already closed
    /// cursor simply keeps the shape disabled for this operation.
    fn offer_non_closing(&mut self, measured: Option<usize>) -> Option<usize> {
        if self.closed {
            return None;
        }
        let bytes = measured?;
        if bytes > self.remaining {
            return None;
        }
        self.remaining -= bytes;
        Some(bytes)
    }
}

/// One reused read statement shape, in admission priority order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReaderShape {
    /// Single-table published node: identity columns plus neighbors, no vector.
    NodeExpand,
    /// Single-table canonical entries point read by frozen slot entry id: the
    /// identity columns of the entries row plus its vector, no adjacency. The
    /// frozen root's node-row payload has no retained shape of its own.
    NodeScore,
    /// Single-table published node: identity columns only.
    NodeIdentity,
    /// Indexed two-table eligible stream without a point filter.
    ExactUnfiltered,
    /// Indexed two-table point shape shared by point probes, IN probes and
    /// accepted-row projection.
    ExactPoint,
}
impl ReaderShape {
    const ALL: [Self; READER_SHAPES] = [
        Self::NodeExpand,
        Self::NodeScore,
        Self::NodeIdentity,
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
            Self::NodeExpand => node_expand_sql(table),
            Self::NodeScore => Some(node_score_sql(table)),
            Self::NodeIdentity => Some(node_identity_sql(table)),
            Self::ExactUnfiltered => Some(exact_sql(table, false, true)),
            Self::ExactPoint => Some(exact_sql(table, true, true)),
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
            shapes: [ReaderShape::NodeExpand; READER_SHAPES],
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
/// shape is filter-aware and the projection reader is only planned when an
/// accepted row is projected.
fn graph_reader_plan(filter: &SearchFilter<'_>, project: bool) -> ReaderPlan {
    let mut plan = ReaderPlan::new();
    plan.push(ReaderShape::NodeExpand);
    plan.push(ReaderShape::NodeScore);
    plan.push(ReaderShape::NodeIdentity);
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
    retained_total: usize,
    /// Non-demotable admitted bytes (the retired-set shape). The load statement
    /// itself drains and finalizes, but its measured bytes and the owned id set
    /// stay recorded so demotion and the copied core allowance never forget them.
    pinned_total: usize,
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
            retained_total: 0,
            pinned_total: 0,
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
        self.retained_total = self.retained_total.saturating_add(bytes);
        self.high_water = self.retained_total;
    }
    /// Records a non-demotable admitted shape (the retired-set load).
    fn pin(&mut self, bytes: usize) {
        self.pinned_total = self.pinned_total.saturating_add(bytes);
    }
    /// Every admitted byte: pinned shapes plus the retained reader set.
    fn admitted_total(&self) -> usize {
        self.retained_total.saturating_add(self.pinned_total)
    }
    /// Non-demotable bytes only, for admission arithmetic tests.
    #[cfg(test)]
    fn pinned_total(&self) -> usize {
        self.pinned_total
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
        if demotion_needed(self.high_water, self.retained_total) {
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
            if self.aggregate() <= self.retained_total {
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

/// Operation-local ring of resolved rowids.
///
/// Only post-processing decision reads populate it; every `rowid()` caller
/// consults it first, so pre-post-processing calls can never wrap the ring.
/// The ring is fixed-capacity `effective L + 1`, overwrites the oldest slot when
/// full, holds INTEGER-only `(native_local_id, Option<rowid>)` entries, and is
/// dropped at operation end: it is not a store-level row-content cache.
struct CarryRing {
    slots: Vec<(u64, Option<u64>)>,
    /// Fixed logical capacity; the allocated Vec may report more.
    capacity: usize,
    /// Next slot to overwrite once the ring is full.
    next: usize,
}
impl CarryRing {
    fn with_capacity(capacity: usize) -> Result<Self> {
        let mut slots = Vec::new();
        reserve(&mut slots, capacity)?;
        Ok(Self {
            slots,
            capacity,
            next: 0,
        })
    }
    /// `Some(rowid)` on a hit and `None` when the id is not carried. The carried
    /// rowid is itself optional: retired nodes and the frozen root map to None.
    fn get(&self, id: u64) -> Option<Option<u64>> {
        self.slots
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, rowid)| *rowid)
    }
    /// Inserts one resolution, overwriting the oldest slot once the ring is full.
    fn insert(&mut self, id: u64, rowid: Option<u64>) {
        if self.capacity == 0 {
            return;
        }
        if self.slots.len() < self.capacity {
            self.slots.push((id, rowid));
        } else {
            self.slots[self.next] = (id, rowid);
            self.next = (self.next + 1) % self.capacity;
        }
    }
}

/// Ring capacity: at most `effective L + 1` entries. post_process supplies at
/// most `take(L)` candidates, so its own decision stream never wraps the ring
/// and the accepted top-k are guaranteed hits in the final loop.
fn carry_capacity(effective_l: usize) -> Result<usize> {
    effective_l
        .checked_add(1)
        .ok_or_else(|| too_big("DiskANN carry capacity overflow"))
}

/// Charged bytes of the carry shape: one fixed slot per ring entry plus the
/// container slack. Independent of runtime measurements, so it is unit-pinned
/// and the T0 boundary constants apply unchanged.
fn carry_bytes(effective_l: usize) -> usize {
    effective_l
        .saturating_add(1)
        .saturating_mul(CARRY_SLOT_BYTES)
        .saturating_add(CARRY_CONTAINER_BYTES)
}

/// Consults the carry. Returning None means a miss and therefore a real read;
/// a disabled carry (tight budget) is a permanent miss for this operation.
fn carry_lookup(carry: &Option<CarryRing>, id: u64) -> Option<Option<u64>> {
    carry.as_ref().and_then(|carry| carry.get(id))
}

/// Whether a rowid() call may record its resolution: only the decision reads
/// performed inside Output::post_process populate the carry.
fn carry_populates(in_post_process: bool) -> bool {
    in_post_process
}

/// Frozen slot identities from the most recent expansion.
///
/// The pinned upstream search scores one expanded node's neighbours before it
/// expands the next node, so the pairs NodeExpand decoded are exactly the ids the
/// adapter is asked to score next. The cache is operation-scoped, INTEGER-only,
/// holds at most `physical_degree` pairs (the bound the streamed decoder
/// validates against), and never replaces a read: an absent pair costs one
/// NodeIdentity read for the entry id, so correctness never depends on the
/// estimate or on admission. It is filled directly from the streaming adjacency
/// decode; no intermediate pair buffer exists.
#[derive(Debug)]
struct SlotPairs {
    pairs: Vec<(u64, u64)>,
    capacity: usize,
}
impl SlotPairs {
    fn with_capacity(capacity: usize) -> Result<Self> {
        let mut pairs = Vec::new();
        reserve(&mut pairs, capacity)?;
        Ok(Self { pairs, capacity })
    }
    /// Starts one expansion's slot stream, discarding the previous expansion's
    /// identities before the first streamed pair arrives.
    fn begin(&mut self) {
        self.pairs.clear();
    }
    /// Records one streamed pair. Bounded by the physical degree, so at most one
    /// expansion's slots are ever retained even if a corrupt blob slipped through.
    fn push(&mut self, local: u64, entry: u64) {
        if self.pairs.len() < self.capacity {
            self.pairs.push((local, entry));
        }
    }
    /// The frozen entry id for one native local id, when this expansion owns it.
    fn entry_of(&self, id: u64) -> Option<u64> {
        self.pairs
            .iter()
            .find(|(local, _)| *local == id)
            .map(|(_, entry)| *entry)
    }
    /// Retained pair count, for the admission/decline unit tests.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.pairs.len()
    }
}

/// Charged bytes of the pair cache: one fixed 16-byte pair per adjacency slot
/// plus the fixed container slack.
///
/// The cache is charged explicitly here, never folded into the D6 owner proof
/// (which covers only the streamed decode: 16-byte blob + 8-byte id per slot).
/// It is offered through the same reader cursor as the retired shape, node
/// readers and carry, last and with a non-closing decline. The bound is the
/// physical degree the decoder validates against, so the cache can never hold
/// more than one expansion's slots.
fn slot_pair_bytes(physical_degree: usize) -> usize {
    physical_degree
        .saturating_mul(SLOT_PAIR_BYTES)
        .saturating_add(SLOT_PAIR_CONTAINER_BYTES)
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
        // The adjacency term is exactly the streamed expansion decode's
        // simultaneous footprint: a 16-byte slot blob copy plus the 8-byte decoded
        // native id, 24 B per physical-degree slot. The pair cache is NOT part of
        // this owner proof; it is a separate shape charged through the reader
        // cursor and simply disabled when a tight budget cannot cover it.
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

// --- statement shapes -------------------------------------------------------
//
// Format 5 makes the published node row self-contained, so every graph-route
// read is a single-table point lookup. The exact routes keep an indexed
// two-table plan and the D3 fallback keeps HEAD's three-table join shape with
// Format-5 role semantics.

/// Single-table identity columns carried by every node-row reader shape.
const NODE_IDENTITY_COLUMNS: &str = "entry_id,public_rowid,state";

/// NodeIdentity: identity columns only, the smallest prefix read.
fn node_identity_sql(table: &DiskAnnTable) -> String {
    format!(
        "SELECT {NODE_IDENTITY_COLUMNS} FROM {} WHERE generation_id=?1 AND native_local_id=?2",
        table.names().graph_nodes
    )
}

/// NodeScore: the canonical entries row for one frozen slot identity.
///
/// A live candidate's vector is read from `diskann_entries` by the entry id its
/// adjacency slot carries, so the single-table node row stays payload-free. The
/// row also carries the creation/retirement interval columns, which the scored-row
/// classification validates on every read (D10 row 4): interval bounds first, then
/// exact agreement between `retired_revision` and the retire-maintained set.
fn node_score_sql(table: &DiskAnnTable) -> String {
    let vector = blob_projection("vector", table.vector_bytes(), true);
    format!(
        "SELECT entry_id,public_rowid,created_revision,retired_revision,{vector} FROM {} WHERE entry_id=?1",
        table.names().entries
    )
}

/// RootScore: the frozen root's identity columns and inline vector.
///
/// Only the explicit start point still carries a payload on its node row; every
/// other native id is scored from the canonical entries table.
fn root_score_sql(table: &DiskAnnTable) -> String {
    let vector = blob_projection("vector", table.vector_bytes(), true);
    format!(
        "SELECT {NODE_IDENTITY_COLUMNS},{vector} FROM {} WHERE generation_id=?1 AND native_local_id=?2",
        table.names().graph_nodes
    )
}

/// NodeExpand: identity columns and the adjacency, never the vector payload.
fn node_expand_sql(table: &DiskAnnTable) -> Option<String> {
    let adjacency =
        checked_product(table.options().physical_degree(), 16, "query adjacency").ok()?;
    let neighbors = blob_projection("neighbors", adjacency, false);
    Some(format!(
        "SELECT {NODE_IDENTITY_COLUMNS},{neighbors} FROM {} WHERE generation_id=?1 AND native_local_id=?2",
        table.names().graph_nodes
    ))
}

/// Three-table fallback with HEAD's join shape. Current membership still decides
/// eligibility because the fallback runs precisely when the retire-maintained set
/// could not be admitted, and the vector payload reverts to the canonical
/// `e.vector` (Format-4 semantics). The row projects the entries interval columns
/// so the restored retirement-consistency checks run on this route too (D10 row
/// 4), and the `n.vector IS NULL` flag reports the root-only node payload shape so
/// role validation can reject a mis-shaped row.
fn fallback_sql(table: &DiskAnnTable, vector: bool) -> String {
    fallback_sql_for(table.names(), table.vector_bytes(), vector)
}

/// Shape-preserving builder for `fallback_sql`, shared with the unit test that
/// pins the restored retirement columns (D10 row 4).
fn fallback_sql_for(names: &StorageNames, vector_bytes: usize, vector: bool) -> String {
    let payload = if vector {
        // Exactly HEAD's role-dependent payload: the frozen root has no entries
        // row, so its node-row copy is the only source; every live node reverts
        // to the canonical entries vector.
        format!(
            ",{}",
            blob_projection(
                "CASE WHEN n.state=2 THEN n.vector ELSE e.vector END",
                vector_bytes,
                true
            )
        )
    } else {
        String::new()
    };
    format!(
        "SELECT n.entry_id,n.state,n.public_rowid,e.public_rowid,e.created_revision,e.retired_revision,c.entry_id,(n.vector IS NULL){payload} FROM {} n LEFT JOIN {} e ON e.entry_id=n.entry_id LEFT JOIN {} c ON c.public_rowid=e.public_rowid WHERE n.generation_id=?1 AND n.native_local_id=?2",
        names.graph_nodes, names.entries, names.current
    )
}

/// Indexed exact plan: current's primary key plus the existing
/// UNIQUE(generation_id, entry_id) index on the published nodes. Eligibility is
/// current membership exactly as HEAD, with HEAD's `eligible_sql` interval columns
/// restored: the published rowid, `e.created_revision`, `e.retired_revision` and
/// the current rowid are projected on both shapes, and the vector payload reverts
/// to the canonical entries row (Format-4 semantics), joined by the same entry id.
///
/// The `entries` LEFT JOIN is unconditional — even for `internal_id`'s
/// `vector = false` shape — so the validation column indices are identical on
/// both shapes and a missing entries row is corruption, never a silently shorter
/// row.
fn exact_sql(table: &DiskAnnTable, point: bool, vector: bool) -> String {
    exact_sql_for(table.names(), table.vector_bytes(), point, vector)
}

/// Shape-preserving builder for `exact_sql`, shared with the unit test that pins
/// the unconditional entries join and the restored retirement columns.
fn exact_sql_for(names: &StorageNames, vector_bytes: usize, point: bool, vector: bool) -> String {
    let payload = if vector {
        format!(",{}", blob_projection("e.vector", vector_bytes, true))
    } else {
        String::new()
    };
    format!(
        "SELECT n.native_local_id,n.entry_id,n.public_rowid,e.created_revision,e.retired_revision,c.public_rowid{payload} FROM {} n JOIN {} c ON c.entry_id=n.entry_id LEFT JOIN {} e ON e.entry_id=n.entry_id WHERE n.generation_id=?1 AND n.state=0{} ORDER BY n.native_local_id",
        names.graph_nodes,
        names.current,
        names.entries,
        if point { " AND c.public_rowid=?2" } else { "" }
    )
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

fn optional_id(statement: &Statement<'_>, column: usize, name: &str) -> Result<Option<u64>> {
    if statement.column_is_null(column)? {
        Ok(None)
    } else {
        Ok(Some(nonnegative(statement, column, name)?))
    }
}

/// Role/identity validation shared by the single-table node shapes.
///
/// Eligibility is retire-maintained membership, never a per-read join. The entry
/// id is returned alongside the class because the lean layout scores a live
/// candidate from the canonical entries row instead of an inline payload.
fn node_identity_of(
    generation: &Generation,
    retired: &Retired,
    statement: &Statement<'_>,
    id: u64,
) -> Result<(NodeState, Option<u64>, Option<u64>)> {
    let state = nonnegative(statement, 2, "query node state")?;
    let entry = optional_id(statement, 0, "query node entry ID")?;
    let rowid = optional_id(statement, 1, "query node rowid")?;
    if Some(id) == generation.root_local_id {
        if state != 2 || entry.is_some() || rowid.is_some() {
            return Err(corrupt("DiskANN explicit frozen root role invalid"));
        }
        return Ok((NodeState::Frozen, None, None));
    }
    if state != 0
        || id >= generation.total_count
        || entry.is_none_or(|value| value == 0)
        || rowid.is_none()
    {
        return Err(corrupt("DiskANN query node role/version invalid"));
    }
    let entry = entry.ok_or_else(|| corrupt("DiskANN query node entry ID missing"))?;
    let rowid = rowid.ok_or_else(|| corrupt("DiskANN query node rowid missing"))?;
    if retired.contains(entry) {
        Ok((NodeState::Deleted, None, Some(entry)))
    } else {
        Ok((NodeState::Live, Some(rowid), Some(entry)))
    }
}

/// Vector payload guard plus decode for a node-row reader positioned on its row.
fn node_vector_of(
    table: &DiskAnnTable,
    statement: &mut Statement<'_>,
    length: usize,
    payload: usize,
    decode: &mut Vec<f32>,
    action: &mut dyn FnMut(StoredVector<'_>) -> Result<()>,
) -> Result<()> {
    check_blob(statement, length, payload, table.vector_bytes())?;
    statement.with_column_blob(payload, |bytes| {
        with_decoded_vector(bytes, table.space().dim, decode, action)
    })?;
    Ok(())
}

/// Streamed adjacency decode for a NodeExpand row.
///
/// The row also carries identity columns, so the node role is validated first: a
/// missing/dangling or role-shifted node is corruption, never EOF. Every valid
/// target is emitted unchanged and in slot order; retired bridges and the frozen
/// root are never filtered. The streamed decoder hands each validated pair
/// straight to the operation's bounded pair cache when it was admitted, so the
/// only transient buffers are the copied slot blob (16 B per slot) and the
/// traversal ids (8 B per slot): exactly the D6 owner adjacency term.
fn node_expand_of(
    table: &DiskAnnTable,
    generation: &Generation,
    retired: &Retired,
    statement: &Statement<'_>,
    id: u64,
    ids: &mut Vec<u64>,
    pairs: &mut Option<SlotPairs>,
) -> Result<()> {
    // The row also carries identity columns; validate the node role so a
    // missing/dangling or role-shifted node is corruption, never EOF.
    node_identity_of(generation, retired, statement, id)?;
    let adjacency = checked_product(table.options().physical_degree(), 16, "query adjacency")?;
    let bytes = bounded_blob(statement, 3, 4, adjacency, false, "neighbors")?;
    if let Some(pairs) = pairs.as_mut() {
        pairs.begin();
    }
    decode_slots_streamed(
        &bytes,
        table.options().physical_degree(),
        Some(id),
        generation.total_count,
        ids,
        |local, entry| {
            if let Some(pairs) = pairs.as_mut() {
                pairs.push(local, entry);
            }
            Ok(())
        },
    )?;
    if ids.iter().any(|&local| local > generation.total_count) {
        return Err(corrupt(
            "DiskANN adjacency exceeds generation local ID range",
        ));
    }
    Ok(())
}

/// Fallback-route classification: HEAD's `classify` asymmetry, kept exactly.
///
/// The fallback runs when the retire-maintained set could not be admitted, so it
/// decides eligibility from `current` and validates the interval columns instead
/// of set membership (D10 row 4):
/// - creation lies inside the frozen snapshot;
/// - when a retirement revision is present it must follow creation and must not
///   exceed the anchored metadata revision;
/// - a still-current entry carrying ANY retirement revision is corruption (HEAD's
///   original rule; the retire-direction tamper is query-visible here too);
/// - a non-current entry is a `Deleted` bridge WITHOUT requiring a revision: a
///   shadow-deleted `current` row is excluded, not corrupt (D10 row 3), which is
///   exactly HEAD's asymmetry.
fn fallback_retirement_state(
    created: u64,
    retired_revision: Option<u64>,
    current: bool,
    snapshot_revision: u64,
    revision: u64,
) -> Result<NodeState> {
    if created == 0 || created > snapshot_revision {
        return Err(corrupt("DiskANN graph entry lies outside frozen snapshot"));
    }
    if let Some(retired) = retired_revision {
        if retired <= created || retired > revision {
            return Err(corrupt("DiskANN query immutable interval invalid"));
        }
        if current {
            return Err(corrupt("DiskANN current graph entry is retired"));
        }
    }
    Ok(if current {
        NodeState::Live
    } else {
        NodeState::Deleted
    })
}

/// Format-5 role classification for the three-table fallback. The published
/// columns decide the rowid and role; `current` membership plus HEAD's classify
/// asymmetry decide eligibility, with the retirement interval validated on every
/// scored row.
fn fallback_identity_of(
    generation: &Generation,
    revision: u64,
    statement: &Statement<'_>,
    id: u64,
) -> Result<(NodeState, Option<u64>)> {
    let state = nonnegative(statement, 1, "query node state")?;
    let entry = optional_id(statement, 0, "query node entry ID")?;
    let rowid = optional_id(statement, 2, "query node rowid")?;
    // Lean layout: only the frozen root carries an inline vector, so the node's
    // own vector IS NULL flag must be one for a live node and zero for the root.
    let vector_null = nonnegative(statement, 7, "query vector NULL flag")?;
    if Some(id) == generation.root_local_id {
        if state != 2 || entry.is_some() || rowid.is_some() || vector_null != 0 {
            return Err(corrupt("DiskANN explicit frozen root role invalid"));
        }
        return Ok((NodeState::Frozen, None));
    }
    if state != 0
        || id >= generation.total_count
        || entry.is_none_or(|value| value == 0)
        || rowid.is_none()
        || vector_null != 1
    {
        return Err(corrupt("DiskANN query node role/version invalid"));
    }
    // The entries row stays a validation-only join, exactly as HEAD reads it.
    if optional_id(statement, 3, "query original rowid")?.is_none() {
        return Err(corrupt("DiskANN graph entry is missing"));
    }
    let created = nonnegative(statement, 4, "query creation revision")?;
    let retired_revision = optional_id(statement, 5, "query retirement revision")?;
    let current = optional_id(statement, 6, "query current entry ID")?;
    if current == Some(0) {
        return Err(corrupt("DiskANN query current entry is zero"));
    }
    match fallback_retirement_state(
        created,
        retired_revision,
        current == entry,
        generation.source_revision,
        revision,
    )? {
        NodeState::Live => Ok((NodeState::Live, rowid)),
        _ => Ok((NodeState::Deleted, None)),
    }
}

/// HEAD's `eligible_identity` interval rule, restored for the exact and
/// projection routes (T6'/D10 row 4): creation lies inside the frozen snapshot and
/// the entry carries no retirement revision, because an eligible published row can
/// never be retired while a `current` row still points at it.
fn eligible_entry_interval(created: u64, retired_is_null: bool, snapshot_revision: u64) -> bool {
    created != 0 && created <= snapshot_revision && retired_is_null
}

/// Identity validation for the indexed exact shapes, with HEAD's interval checks
/// restored. The published rowid stays the result label; it must agree with the
/// row `current` serves for this entry, exactly as HEAD compared its entries and
/// current rowids.
fn exact_identity(generation: &Generation, statement: &Statement<'_>) -> Result<(u64, u64)> {
    let id = nonnegative(statement, 0, "eligible local ID")?;
    let entry = nonnegative(statement, 1, "eligible entry ID")?;
    let rowid = nonnegative(statement, 2, "eligible original rowid")?;
    let created = nonnegative(statement, 3, "eligible creation revision")?;
    if id >= generation.total_count
        || entry == 0
        || !eligible_entry_interval(
            created,
            statement.column_is_null(4)?,
            generation.source_revision,
        )
        || rowid != nonnegative(statement, 5, "eligible current rowid")?
    {
        return Err(corrupt(
            "DiskANN eligible graph/current/entry identity invalid",
        ));
    }
    Ok((id, rowid))
}

/// One scalar single-table node read: prepare, bind, consume exactly one row,
/// and drain to DONE. Used only when the retained reader for the shape is absent.
fn scalar_node_row<T>(
    table: &DiskAnnTable,
    sql: &str,
    generation: &Generation,
    id: u64,
    missing: &'static str,
    not_unique: &'static str,
    action: impl FnOnce(&mut Statement<'static>) -> Result<T>,
) -> Result<T> {
    let mut statement = table.connection().prepare(sql)?;
    let outcome = (|| {
        bind_node(generation, &mut statement, id)?;
        if statement.step()? != Step::Row {
            return Err(corrupt(missing));
        }
        let value = action(&mut statement)?;
        if statement.step()? != Step::Done {
            return Err(corrupt(not_unique));
        }
        Ok(value)
    })();
    finish(statement, outcome)
}

fn node_identity_scalar(
    table: &DiskAnnTable,
    generation: &Generation,
    retired: &Retired,
    id: u64,
) -> Result<(NodeState, Option<u64>, Option<u64>)> {
    let sql = node_identity_sql(table);
    scalar_node_row(
        table,
        &sql,
        generation,
        id,
        "DiskANN referenced query node is missing",
        "DiskANN query node identity not unique",
        |statement| node_identity_of(generation, retired, statement, id),
    )
}

/// One retained NodeIdentity-reader execution. Charges nothing: the caller owns
/// the visit accounting.
fn node_identity_read(
    generation: &Generation,
    retired: &Retired,
    statement: &mut Statement<'static>,
    id: u64,
) -> Result<(NodeState, Option<u64>, Option<u64>)> {
    reset_reader(statement)?;
    bind_node(generation, statement, id)?;
    if statement.step()? != Step::Row {
        return Err(corrupt("DiskANN referenced query node is missing"));
    }
    let identity = node_identity_of(generation, retired, statement, id)?;
    if statement.step()? != Step::Done {
        return Err(corrupt("DiskANN query node identity not unique"));
    }
    Ok(identity)
}

/// The frozen root is the only node whose vector is still read from its node
/// row: it has no canonical entries row and no adjacency slot identity.
fn root_score_scalar(
    table: &DiskAnnTable,
    generation: &Generation,
    retired: &Retired,
    id: u64,
    decode: &mut Vec<f32>,
    action: &mut dyn FnMut(StoredVector<'_>) -> Result<()>,
) -> Result<()> {
    let sql = root_score_sql(table);
    scalar_node_row(
        table,
        &sql,
        generation,
        id,
        "DiskANN referenced root node is missing",
        "DiskANN root node not unique",
        move |statement| {
            node_identity_of(generation, retired, statement, id)?;
            node_vector_of(table, statement, 3, 4, decode, action)
        },
    )
}

/// Binds one canonical entry id on an entries-point statement.
fn bind_entry(statement: &mut Statement<'static>, entry: u64) -> Result<()> {
    statement.bind_i64(1, checked_id(entry)?)
}

/// Fast-route scored-row classification: interval bounds first, then exact
/// agreement between `entries.retired_revision` and the retire-maintained set.
///
/// D10 row 4 / C1(ii), both directions: a retirement revision present while the
/// entry is not in the retire-maintained set is corruption (one rule subsuming
/// both the `<= source_revision` and the `> source_revision` bookkeeping-bypassing
/// shadow edits), and a set member whose revision is NULL is corruption (the
/// symmetric un-retire rule). A set member carrying a validated revision is a
/// `Deleted` navigation bridge: `neighbors()` still returns it, but a search can
/// never return it. Intervals are validated before membership, so a malformed
/// interval on a set member is corruption too.
fn fast_retirement_state(
    created: u64,
    retired_revision: Option<u64>,
    in_retired_set: bool,
    snapshot_revision: u64,
    revision: u64,
) -> Result<NodeState> {
    if created == 0 || created > snapshot_revision {
        return Err(corrupt("DiskANN graph entry lies outside frozen snapshot"));
    }
    if retired_revision.is_some_and(|value| value <= created || value > revision) {
        return Err(corrupt("DiskANN query immutable interval invalid"));
    }
    match (retired_revision, in_retired_set) {
        (Some(_), true) => Ok(NodeState::Deleted),
        (None, false) => Ok(NodeState::Live),
        _ => Err(corrupt("DiskANN query retirement bookkeeping disagreement")),
    }
}

/// Reads and classifies one scored canonical entries row.
fn entry_score_of(
    generation: &Generation,
    retired: &Retired,
    revision: u64,
    statement: &Statement<'_>,
    entry: u64,
) -> Result<(NodeState, u64)> {
    let row_entry = nonnegative(statement, 0, "query entry ID")?;
    let rowid = nonnegative(statement, 1, "query original rowid")?;
    let created = nonnegative(statement, 2, "query creation revision")?;
    let retired_revision = optional_id(statement, 3, "query retirement revision")?;
    if row_entry != entry {
        return Err(corrupt("DiskANN query entry identity mismatch"));
    }
    let state = fast_retirement_state(
        created,
        retired_revision,
        retired.contains(entry),
        generation.source_revision,
        revision,
    )?;
    Ok((state, rowid))
}

/// Validates the positioned entries row, decodes its vector and delivers the
/// resolved class, public rowid and payload to `action`.
///
/// The anchored retirement context (set + metadata revision) travels with the
/// reader exactly as HEAD threaded its metadata revision; keeping it explicit
/// avoids hiding route state in the graph object at this layer.
#[allow(clippy::too_many_arguments)]
fn entry_row_deliver(
    table: &DiskAnnTable,
    generation: &Generation,
    retired: &Retired,
    revision: u64,
    statement: &mut Statement<'_>,
    entry: u64,
    decode: &mut Vec<f32>,
    action: &mut dyn FnMut(NodeState, u64, StoredVector<'_>) -> Result<()>,
) -> Result<()> {
    let (state, rowid) = entry_score_of(generation, retired, revision, statement, entry)?;
    check_blob(statement, 4, 5, table.vector_bytes())?;
    let dimension = table.space().dim;
    statement.with_column_blob(5, |bytes| {
        with_decoded_vector(bytes, dimension, decode, &mut |vector| {
            action(state, rowid, vector)
        })
    })?;
    Ok(())
}

/// One scalar scoring read of the canonical entries row. Used only when the
/// retained entries reader for the shape is absent.
fn entry_score_scalar(
    table: &DiskAnnTable,
    generation: &Generation,
    retired: &Retired,
    revision: u64,
    entry: u64,
    decode: &mut Vec<f32>,
    action: &mut dyn FnMut(NodeState, u64, StoredVector<'_>) -> Result<()>,
) -> Result<()> {
    let sql = node_score_sql(table);
    let mut statement = table.connection().prepare(&sql)?;
    let outcome = (|| {
        bind_entry(&mut statement, entry)?;
        if statement.step()? != Step::Row {
            return Err(corrupt("DiskANN referenced graph entry is missing"));
        }
        entry_row_deliver(
            table,
            generation,
            retired,
            revision,
            &mut statement,
            entry,
            decode,
            action,
        )?;
        if statement.step()? != Step::Done {
            return Err(corrupt("DiskANN query graph entry not unique"));
        }
        Ok(())
    })();
    finish(statement, outcome)
}

/// One retained NodeScore-reader execution over the canonical entries row.
/// Charges nothing: the caller owns the visit accounting.
///
/// The anchored retirement context (set + metadata revision) travels with the
/// reader exactly as HEAD threaded its metadata revision.
#[allow(clippy::too_many_arguments)]
fn entry_score_read(
    table: &DiskAnnTable,
    generation: &Generation,
    retired: &Retired,
    revision: u64,
    statement: &mut Statement<'static>,
    entry: u64,
    decode: &mut Vec<f32>,
    action: &mut dyn FnMut(NodeState, u64, StoredVector<'_>) -> Result<()>,
) -> Result<()> {
    reset_reader(statement)?;
    bind_entry(statement, entry)?;
    if statement.step()? != Step::Row {
        return Err(corrupt("DiskANN referenced graph entry is missing"));
    }
    entry_row_deliver(
        table, generation, retired, revision, statement, entry, decode, action,
    )?;
    if statement.step()? != Step::Done {
        return Err(corrupt("DiskANN query graph entry not unique"));
    }
    Ok(())
}

fn node_expand_scalar(
    table: &DiskAnnTable,
    generation: &Generation,
    retired: &Retired,
    id: u64,
    ids: &mut Vec<u64>,
    pairs: &mut Option<SlotPairs>,
) -> Result<()> {
    let Some(sql) = node_expand_sql(table) else {
        return Err(too_big("DiskANN query adjacency size overflow"));
    };
    scalar_node_row(
        table,
        &sql,
        generation,
        id,
        "DiskANN referenced graph node is missing",
        "DiskANN graph node is not unique",
        |statement| node_expand_of(table, generation, retired, statement, id, ids, pairs),
    )
}

/// One retained NodeExpand-reader execution. Charges nothing: neighbors() owns
/// the single expansion visit.
fn node_expand_read(
    table: &DiskAnnTable,
    generation: &Generation,
    retired: &Retired,
    statement: &mut Statement<'static>,
    id: u64,
    ids: &mut Vec<u64>,
    pairs: &mut Option<SlotPairs>,
) -> Result<()> {
    reset_reader(statement)?;
    bind_node(generation, statement, id)?;
    if statement.step()? != Step::Row {
        return Err(corrupt("DiskANN referenced graph node is missing"));
    }
    node_expand_of(table, generation, retired, statement, id, ids, pairs)?;
    if statement.step()? != Step::Done {
        return Err(corrupt("DiskANN graph node is not unique"));
    }
    Ok(())
}

fn fallback_identity_scalar(
    table: &DiskAnnTable,
    generation: &Generation,
    revision: u64,
    id: u64,
) -> Result<(NodeState, Option<u64>)> {
    let sql = fallback_sql(table, false);
    scalar_node_row(
        table,
        &sql,
        generation,
        id,
        "DiskANN referenced query node/entry missing",
        "DiskANN query node identity not unique",
        |statement| fallback_identity_of(generation, revision, statement, id),
    )
}

fn fallback_vector_scalar(
    table: &DiskAnnTable,
    generation: &Generation,
    revision: u64,
    id: u64,
    decode: &mut Vec<f32>,
    action: &mut dyn FnMut(StoredVector<'_>) -> Result<()>,
) -> Result<()> {
    let sql = fallback_sql(table, true);
    scalar_node_row(
        table,
        &sql,
        generation,
        id,
        "DiskANN referenced vector node/entry missing",
        "DiskANN query vector node not unique",
        move |statement| {
            fallback_identity_of(generation, revision, statement, id)?;
            node_vector_of(table, statement, 8, 9, decode, action)
        },
    )
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

/// Delivers one combined-reader vector to a batch action slot with exactly the
/// rowid semantics of diskann_core's default loop: the resolved identity rowid is
/// passed only in need_rowid mode and None is passed in vector-only mode.
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

type QueryOutcome = (Vec<SearchResult>, Option<Vec<Vec<u8>>>);

/// The published generation's retired-entry identities, resolved once at
/// operation start inside the metadata anchor's snapshot.
#[derive(Debug)]
enum Retired {
    /// The invariant count was zero: every state-0 node is live, no reads.
    Empty,
    /// Sorted entry IDs admitted through the reader allowance; bytes are pinned.
    Admitted { ids: Vec<u64>, bytes: usize },
    /// The set could not be admitted: the whole query uses the three-table path.
    Unavailable,
}
impl Retired {
    fn contains(&self, entry: u64) -> bool {
        match self {
            Self::Empty => false,
            Self::Admitted { ids, .. } => ids.binary_search(&entry).is_ok(),
            Self::Unavailable => false,
        }
    }
    fn is_unavailable(&self) -> bool {
        matches!(self, Self::Unavailable)
    }
}

struct PublishedGraph<'a> {
    table: &'a DiskAnnTable,
    generation: Generation,
    /// Anchored metadata revision of this operation. A scored row's retirement
    /// revision must never exceed it (D4/D10 row 4 interval bound).
    revision: u64,
    budget: Budget,
    readers: Readers,
    decode: Vec<f32>,
    retired: Retired,
    /// True when the retired set could not be admitted: every traversal read
    /// uses the three-table scalar path so eligibility never weakens.
    fallback: bool,
    /// Resolved-rowid carry; None when the D6 allowance could not cover it.
    carry: Option<CarryRing>,
    /// Frozen adjacency identities from the most recent expansion; None when the
    /// fixed pair-cache shape was not admitted (every lookup then reads once).
    pairs: Option<SlotPairs>,
    /// Real (release-active) post-processing boundary. Only decision reads taken
    /// while this is set populate the carry.
    in_post_process: bool,
    #[cfg(debug_assertions)]
    phase: Phase,
}

impl PublishedGraph<'_> {
    /// Prepares and measures only the route's needed readers against the shared
    /// admission cursor, in priority order.
    ///
    /// Preparation is attempted only while the envelope is open: the first shape
    /// whose measured size does not fit closes admission, and no lower-priority
    /// shape is built or prepared after that, so "unused shapes are never
    /// prepared" holds. A shape whose SQL cannot be built, whose prepare fails
    /// (including the Debug-only forced failure), or whose MEMUSED is unavailable
    /// is silently scalar-only and does not end admission. Admission never raises
    /// a setup error; a rejected local statement is finalized by Drop and its
    /// shape keeps HEAD's scalar prepare/finalize path.
    fn admit_readers(&mut self, plan: &ReaderPlan, cursor: &mut ReaderAllowance) {
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
    /// Offers the carry as the operation's last admitted shape (D6 order:
    /// retired shape, node readers, then the carry) with non-closing decline.
    ///
    /// The ring's fixed (L + 1) slots plus container slack are pinned
    /// non-demotably, exactly like the retired Vec's bytes. When the cursor
    /// declines (tight budgets) or the Debug-only hook disables it, the
    /// operation keeps the final loop on ordinary charged NodeIdentity reads.
    fn admit_carry(
        &mut self,
        effective_l: usize,
        forced_off: bool,
        cursor: &mut ReaderAllowance,
    ) -> Result<()> {
        if forced_off {
            return Ok(());
        }
        let capacity = carry_capacity(effective_l)?;
        let Some(admitted) = cursor.offer_non_closing(Some(carry_bytes(effective_l))) else {
            return Ok(());
        };
        note_carry_admission(admitted);
        self.readers.pin(admitted);
        self.carry = Some(CarryRing::with_capacity(capacity)?);
        Ok(())
    }
    /// Offers the fixed pair-cache shape after the carry (D6 value order:
    /// retired shape, node readers, carry, then pairs), with the same
    /// non-closing decline so it can never end admission of an existing shape.
    ///
    /// The cache is charged explicitly and pinned non-demotably: one 16-byte pair
    /// per physical-degree slot plus the fixed container slack
    /// (`slot_pair_bytes`). When the cursor declines it (tight budgets) or the
    /// Debug hook forces it off, the cache is disabled for this operation and
    /// scoring falls back to one NodeIdentity read per scored id at the unchanged
    /// visit charge (see `entry_of`); results, order and owner totals are
    /// unaffected by that choice.
    fn admit_pairs(&mut self, forced_off: bool, cursor: &mut ReaderAllowance) -> Result<()> {
        let physical = self.table.options().physical_degree();
        if !forced_off {
            if let Some(admitted) = cursor.offer_non_closing(Some(slot_pair_bytes(physical))) {
                let cache = SlotPairs::with_capacity(physical)?;
                self.readers.pin(admitted);
                self.pairs = Some(cache);
                return Ok(());
            }
        }
        note(Verb::PairDisabled);
        Ok(())
    }
    /// Prepares one reader shape, honoring the Debug-only forced-failure hook.
    ///
    /// The hook selects shapes by ReaderShape::mask and fails only the next
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
    /// Switches the instrumentation phase to traversal on the first expansion.
    fn enter_traversal(&mut self) {
        #[cfg(debug_assertions)]
        {
            if self.phase == Phase::Setup {
                self.phase = Phase::Traversal;
            }
        }
    }
    /// Attributes one charged rowid() resolution to its instrumentation phase.
    /// This never changes a visit, a statement, or a result. Carry hits/misses
    /// are recorded separately by rowid() itself.
    fn note_rowid_phase(&self) {
        #[cfg(debug_assertions)]
        match self.phase {
            Phase::Setup => note(Verb::Setup),
            Phase::Traversal => {}
            Phase::PostProcess => note(Verb::Decision),
            Phase::Final => {}
        }
    }
    /// Identity-only read that also reports the frozen entry id: retire-maintained
    /// eligibility on a format-served route, current-based classification on the
    /// D3 fallback (whose join has no slot identity to report).
    fn identity_roles(&mut self, id: u64) -> Result<(NodeState, Option<u64>, Option<u64>)> {
        if self.fallback {
            note(Verb::Identity3Table);
            note(Verb::Fallback3Table);
            return fallback_identity_scalar(self.table, &self.generation, self.revision, id)
                .map(|(state, rowid)| (state, rowid, None));
        }
        note(Verb::NodeIdentity);
        let Some(mut statement) = self.readers.take(ReaderShape::NodeIdentity) else {
            return node_identity_scalar(self.table, &self.generation, &self.retired, id);
        };
        note_reader_execution();
        let outcome = node_identity_read(&self.generation, &self.retired, &mut statement, id);
        self.finish_reader(ReaderShape::NodeIdentity, statement, outcome)
    }
    /// Identity-only read without the entry id, exactly as HEAD charges it.
    fn identity(&mut self, id: u64) -> Result<(NodeState, Option<u64>)> {
        self.identity_roles(id)
            .map(|(state, rowid, _)| (state, rowid))
    }
    /// Resolves one native local id to the entry id its adjacency slot froze.
    ///
    /// The streaming expansion decode fills the bounded pair cache directly, so
    /// the scoring that follows a normal expansion is a hit. An id with no cached
    /// pair (the start point, a cache-hit path, or a tight-budget operation whose
    /// pair cache was declined or forced off) costs one NodeIdentity read to
    /// obtain the entry id, still inside the same single charged score visit: the
    /// scored candidate's rowid and vector charges are already the D5 map's
    /// charges, and this identity read replaces HEAD's per-target identity
    /// validation without adding a visit. Owner-visit totals are therefore
    /// identical with the pair cache admitted or disabled.
    fn entry_of(&mut self, id: u64) -> Result<u64> {
        if let Some(entry) = self.pairs.as_ref().and_then(|pairs| pairs.entry_of(id)) {
            if entry != 0 {
                note(Verb::PairHit);
                return Ok(entry);
            }
        }
        note(Verb::PairMiss);
        let (_state, _rowid, entry) = self.identity_roles(id)?;
        entry.ok_or_else(|| corrupt("DiskANN query node entry ID missing"))
    }
    /// Scored-candidate vector read: the frozen root still reads its node row,
    /// every live candidate reads the canonical entries row by slot entry id, and
    /// the D3 fallback keeps HEAD's three-table projection.
    fn score_vector(
        &mut self,
        id: u64,
        action: &mut dyn FnMut(StoredVector<'_>) -> Result<()>,
    ) -> Result<()> {
        self.budget.visit()?;
        if self.fallback {
            note(Verb::Fallback3Table);
            return fallback_vector_scalar(
                self.table,
                &self.generation,
                self.revision,
                id,
                &mut self.decode,
                action,
            );
        }
        note(Verb::NodeScore);
        note(Verb::Scored);
        if Some(id) == self.generation.root_local_id {
            return root_score_scalar(
                self.table,
                &self.generation,
                &self.retired,
                id,
                &mut self.decode,
                action,
            );
        }
        let entry = self.entry_of(id)?;
        let mut deliver = |_state: NodeState, _rowid: u64, vector: StoredVector<'_>| action(vector);
        let Some(mut statement) = self.readers.take(ReaderShape::NodeScore) else {
            return entry_score_scalar(
                self.table,
                &self.generation,
                &self.retired,
                self.revision,
                entry,
                &mut self.decode,
                &mut deliver,
            );
        };
        note_reader_execution();
        let outcome = entry_score_read(
            self.table,
            &self.generation,
            &self.retired,
            self.revision,
            &mut statement,
            entry,
            &mut self.decode,
            &mut deliver,
        );
        self.finish_reader(ReaderShape::NodeScore, statement, outcome)
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

    /// Batch slot: one canonical entries-point read per scored candidate replaces
    /// HEAD's identity plus vector pair. The rowid charge and the eligibility
    /// decide still precede the vector charge and payload validation, so visit
    /// totals match HEAD; a missing retained reader keeps HEAD's two-statement
    /// scalar path unchanged.
    fn batch_slot(
        &mut self,
        index: usize,
        id: u64,
        need_rowid: bool,
        action: &mut VectorBatchAction<'_>,
    ) -> Result<()> {
        // The fallback route and the frozen root (which has no entries row) keep
        // HEAD's two-statement scalar path: a charged rowid read plus a charged
        // vector read, so visit totals never depend on the entries reader.
        if self.fallback || Some(id) == self.generation.root_local_id {
            return self.scalar_batch_slot(index, id, need_rowid, action);
        }
        let Some(mut statement) = self.readers.take(ReaderShape::NodeScore) else {
            return self.scalar_batch_slot(index, id, need_rowid, action);
        };
        note_reader_execution();
        note(Verb::NodeScore);
        note(Verb::Scored);
        let outcome = (|| {
            let entry = self.entry_of(id)?;
            if need_rowid {
                self.budget.visit()?;
            }
            self.budget.visit()?;
            let mut deliver = |state: NodeState, rowid: u64, vector: StoredVector<'_>| {
                let resolved = if state == NodeState::Live {
                    Some(rowid)
                } else {
                    None
                };
                deliver_batch_vector(action, index, id, need_rowid, resolved, vector)
            };
            entry_score_read(
                self.table,
                &self.generation,
                &self.retired,
                self.revision,
                &mut statement,
                entry,
                &mut self.decode,
                &mut deliver,
            )
        })();
        self.finish_reader(ReaderShape::NodeScore, statement, outcome)
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
            let mut statement =
                self.table
                    .connection()
                    .prepare(&exact_sql(self.table, rowid.is_some(), true))?;
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
            let (_, label) = exact_identity(&self.generation, statement)?;
            check_blob(statement, 6, 7, self.table.vector_bytes())?;
            let metric = self.table.space().distance_type;
            let dimension = self.table.space().dim;
            let mut distance = 0.0;
            statement.with_column_blob(7, |bytes| {
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
        note(Verb::Projection);
        let Some(mut statement) = self.readers.take(ReaderShape::ExactPoint) else {
            let mut statement = self
                .table
                .connection()
                .prepare(&exact_sql(self.table, true, true))?;
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
        exact_identity(&self.generation, statement)?;
        let bytes = bounded_blob(
            statement,
            6,
            7,
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
}

// --- operation-start eligibility, fallback, and mutation detector ----------

/// One indexed count of this generation's retired entries. Equality against
/// total_count - eligible_count is the D3 invariant over existing columns.
fn retired_count(table: &DiskAnnTable, generation: &Generation) -> Result<u64> {
    let mut statement = table.connection().prepare(&format!(
        "SELECT count(*) FROM {} WHERE generation_id=?1",
        table.names().generation_retired
    ))?;
    let outcome = (|| {
        statement.bind_i64(1, checked_id(generation.generation_id)?)?;
        if statement.step()? != Step::Row {
            return Err(corrupt("DiskANN retired count is missing"));
        }
        let count = nonnegative(&statement, 0, "retired count")?;
        if statement.step()? != Step::Done {
            return Err(corrupt("DiskANN retired count is not unique"));
        }
        let expected = generation
            .total_count
            .checked_sub(generation.eligible_count)
            .ok_or_else(|| corrupt("DiskANN eligible count exceeds total count"))?;
        if count != expected {
            return Err(corrupt("DiskANN generation retired count mismatch"));
        }
        Ok(count)
    })();
    let count = finish(statement, outcome)?;
    note(Verb::Setup);
    Ok(count)
}

/// Enforces the admitted retired-set population and ordering before one loaded
/// row is appended.
///
/// Admission charged exactly `expected * 8` bytes for the owned id Vec, so the Vec
/// may never grow past `expected`: a reentrant writer that makes the load
/// statement return an extra row for this generation fails closed here instead of
/// materializing bytes outside the admitted shape. The bound is checked before the
/// ordering rules so no rejected row can be appended either way.
fn push_retired_entry(
    ids: &mut Vec<u64>,
    expected: usize,
    entry: u64,
    previous: &mut u64,
) -> Result<()> {
    if ids.len() >= expected {
        return Err(corrupt(
            "DiskANN retired entry set grew beyond its admitted population",
        ));
    }
    if entry == 0 || entry <= *previous {
        return Err(corrupt("DiskANN retired entry set is not ordered"));
    }
    *previous = entry;
    ids.push(entry);
    Ok(())
}

/// Admits, loads and validates the published generation's retired-entry set.
///
/// Zero means no membership storage and no further eligibility reads. A set that
/// cannot be admitted makes the whole query take the three-table path, so no
/// eligibility rule weakens at any budget. The load/validation charges zero
/// visits; its measured statement bytes plus count * 8 are pinned non-demotably.
/// The per-row bound in `push_retired_entry` keeps the owned Vec inside that
/// admitted byte count even if a reentrant callback grows the set mid-load.
fn load_retired(
    table: &DiskAnnTable,
    generation: &Generation,
    cursor: &mut ReaderAllowance,
) -> Result<Retired> {
    let count = retired_count(table, generation)?;
    note_retired_size(count);
    if count == 0 {
        return Ok(Retired::Empty);
    }
    let mut statement = table.connection().prepare(&format!(
        "SELECT entry_id FROM {} WHERE generation_id=?1 ORDER BY entry_id",
        table.names().generation_retired
    ))?;
    let measured = usize::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(size_of::<u64>()))
        .and_then(|bytes| {
            statement
                .memory_used()
                .and_then(|held| held.checked_add(bytes))
        });
    let Some(measured) = measured else {
        let _ = statement.finish();
        note(Verb::RetiredFallback);
        return Ok(Retired::Unavailable);
    };
    let Some(admitted) = cursor.offer(Some(measured)) else {
        let _ = statement.finish();
        note(Verb::RetiredFallback);
        return Ok(Retired::Unavailable);
    };
    note(Verb::RetiredAdmission);
    let outcome = (|| {
        statement.bind_i64(1, checked_id(generation.generation_id)?)?;
        let capacity =
            usize::try_from(count).map_err(|_| too_big("DiskANN retired entry count overflow"))?;
        let mut ids = Vec::new();
        reserve(&mut ids, capacity)?;
        let mut previous = 0u64;
        while statement.step()? == Step::Row {
            let entry = nonnegative(&statement, 0, "retired entry ID")?;
            push_retired_entry(&mut ids, capacity, entry, &mut previous)?;
        }
        if ids.len() != capacity {
            return Err(corrupt("DiskANN retired entry count changed"));
        }
        Ok(ids)
    })();
    let ids = finish(statement, outcome)?;
    note(Verb::Setup);
    Ok(Retired::Admitted {
        ids,
        bytes: admitted,
    })
}

/// D3 shadow-schema stamp: table count and rootpage sum over the nine shadow
/// names only, so benign unrelated DDL never moves it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ShadowStamp {
    tables: u64,
    rootpage_sum: u64,
}

fn shadow_stamp(table: &DiskAnnTable) -> Result<ShadowStamp> {
    let names = table.names();
    let bases = names.shadow_bases();
    let catalog = qualified_name(&names.schema, "sqlite_schema")?;
    let placeholders = (1..=bases.len())
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(",");
    let mut statement = table.connection().prepare(&format!(
        "SELECT count(*),sum(rootpage) FROM {catalog} WHERE type='table' AND name COLLATE NOCASE IN ({placeholders})"
    ))?;
    let outcome = (|| {
        for (index, base) in bases.iter().enumerate() {
            statement.bind_text(index + 1, base)?;
        }
        if statement.step()? != Step::Row {
            return Err(corrupt("DiskANN shadow schema stamp is missing"));
        }
        let tables = nonnegative(&statement, 0, "shadow table count")?;
        let rootpage_sum = if statement.column_is_null(1)? {
            0
        } else {
            nonnegative(&statement, 1, "shadow rootpage sum")?
        };
        if statement.step()? != Step::Done {
            return Err(corrupt("DiskANN shadow schema stamp is not unique"));
        }
        Ok(ShadowStamp {
            tables,
            rootpage_sum,
        })
    })();
    let stamp = finish(statement, outcome)?;
    note(Verb::Setup);
    Ok(stamp)
}

/// Whether either detector stamp moved between capture and verification.
fn detector_moved(
    before: ShadowStamp,
    after: ShadowStamp,
    changes_before: i64,
    changes_after: i64,
) -> bool {
    changes_before != changes_after || before != after
}

/// D3 two-stamp mid-operation mutation detector.
///
/// Both stamps are captured immediately after the metadata anchor opens and
/// before any other operation read, and verified immediately before knn returns
/// Ok. Any move fails closed with SQLITE_BUSY on every route.
#[derive(Clone, Copy, Debug)]
struct MutationGuard {
    changes: i64,
    stamp: ShadowStamp,
}
impl MutationGuard {
    fn capture(table: &DiskAnnTable) -> Result<Self> {
        // The counter is a pure in-memory field read taken FIRST so its window
        // also covers the schema-stamp statement's own execution.
        let changes = table.connection().total_changes();
        let stamp = shadow_stamp(table)?;
        Ok(Self { changes, stamp })
    }
    fn verify(self, table: &DiskAnnTable) -> Result<()> {
        let stamp = shadow_stamp(table)?;
        let changes = table.connection().total_changes();
        if detector_moved(self.stamp, stamp, self.changes, changes) {
            return Err(IndexError::with_code(
                ffi::SQLITE_BUSY as i32,
                "DiskANN query observed a concurrent mutation",
            ));
        }
        Ok(())
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
            .prepare(&exact_sql(self.table, true, false))?;
        let outcome = (|| {
            statement.bind_i64(1, checked_id(self.generation.generation_id)?)?;
            statement.bind_i64(2, checked_id(rowid)?)?;
            if statement.step()? != Step::Row {
                return Err(IndexError::new("DiskANN published rowid absent"));
            }
            let (id, _) = exact_identity(&self.generation, &statement)?;
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
        // conservatively covers core visits in the SAME fallback budget. The
        // charge is unconditional, so a carry hit costs exactly as much as
        // HEAD's charged NodeIdentity read and visit totals stay identical.
        self.budget.visit()?;
        self.note_rowid_phase();
        // Every caller consults the carry first; only a post-processing decision
        // read may populate it, so pre-post-processing calls can never wrap the
        // ring. A miss (including every call when the carry is disabled) performs
        // the same charged NodeIdentity read HEAD performs.
        if let Some(rowid) = carry_lookup(&self.carry, id) {
            note(Verb::CarryHit);
            return Ok(rowid);
        }
        note(Verb::CarryMiss);
        let populate = carry_populates(self.in_post_process);
        let rowid = self.identity(id).map(|(_, rowid)| rowid)?;
        if populate {
            if let Some(carry) = &mut self.carry {
                carry.insert(id, rowid);
            }
        }
        Ok(rowid)
    }
    fn state(&mut self, id: u64) -> Result<NodeState> {
        // The query adapter only checks the frozen start point's state.
        note(Verb::Setup);
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
        self.score_vector(id, action)
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
        self.enter_traversal();
        note(Verb::Expansion);
        note(Verb::NodeExpand);
        // Every valid target is returned unchanged and in slot order: retired
        // bridge nodes and the frozen root are navigation structure and are never
        // filtered out. The same single pass fills this expansion's pair cache
        // when it was admitted, and the pinned search scores one expansion's
        // neighbours before expanding the next node, so those frozen identities
        // answer the scoring that follows.
        let mut ids = Vec::new();
        match self.readers.take(ReaderShape::NodeExpand) {
            Some(mut statement) => {
                note_reader_execution();
                let outcome = node_expand_read(
                    self.table,
                    &self.generation,
                    &self.retired,
                    &mut statement,
                    id,
                    &mut ids,
                    &mut self.pairs,
                );
                self.finish_reader(ReaderShape::NodeExpand, statement, outcome)?;
            }
            None => node_expand_scalar(
                self.table,
                &self.generation,
                &self.retired,
                id,
                &mut ids,
                &mut self.pairs,
            )?,
        }
        Ok(ids)
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
        note(Verb::Setup);
        Ok(vec![root])
    }
    fn begin_post_process(&mut self) {
        // Release-active: only decision reads inside this boundary populate the
        // carry. The Debug phase value only drives instrumentation counters.
        self.in_post_process = true;
        #[cfg(debug_assertions)]
        {
            self.phase = Phase::PostProcess;
        }
    }
    fn end_post_process(&mut self) {
        self.in_post_process = false;
        #[cfg(debug_assertions)]
        {
            self.phase = Phase::Final;
        }
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
    // Capture the one-shot Debug requests for exactly this operation, so a host
    // hook applies to the next query even when that query takes an early route
    // (where nothing is demoted or carried). Always false in release builds.
    let force_demote = take_force_demote();
    let force_carry_off = take_force_carry_off();
    let force_pair_cache_off = take_force_pair_cache_off();
    let mut normalized = Vec::new();
    reserve(&mut normalized, query.len())?;
    normalized.extend_from_slice(query);
    normalize_input(
        &mut normalized,
        table.space().dim,
        table.space().distance_type,
    )?;
    let (metadata, anchor) = table.read_metadata()?;
    note(Verb::Setup);
    let Some(id) = metadata.published_generation else {
        Budget::admit(table, 0, &filter, project_vector)?;
        return Ok(QueryRows {
            results: Vec::new(),
            vectors: project_vector.then(Vec::new),
            anchor,
        });
    };
    // D3: capture both detector stamps immediately after the metadata anchor and
    // before any other operation read. All later reads stay on the anchor's
    // snapshot; both stamps are verified before this function returns Ok.
    let guard = MutationGuard::capture(table)?;
    let result = (|| -> Result<QueryRows> {
        let generation = table.read_generation(id)?;
        note(Verb::Setup);
        if generation.state != 1 || generation.source_revision > metadata.revision {
            return Err(corrupt(
                "DiskANN published generation identity/revision invalid",
            ));
        }
        if let Some(root) = generation.root_local_id {
            table.read_graph_node(&generation, root)?;
            note(Verb::Setup);
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
            record_owner_visits(budget.visits);
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
            retired: Retired::Empty,
            fallback: false,
            carry: None,
            pairs: None,
            in_post_process: false,
            #[cfg(debug_assertions)]
            phase: Phase::Setup,
        };
        let outcome = (|| -> Result<QueryOutcome> {
            let mut results = if exact_route {
                // No core admission: the exact route only has to keep the owner,
                // the exact/shared readers and the headroom inside cache_bytes.
                // There is no core call and no final top-k loop, so no carry.
                let plan = exact_reader_plan(&filter, project_vector);
                let mut cursor = ReaderAllowance::new(graph.budget.reader_allowance(None));
                graph.admit_readers(&plan, &mut cursor);
                graph.exact(&normalized, count, &filter)?
            } else {
                // Eager route-aware envelope: the whole core floor and the
                // reprepare headroom are reserved before GraphConfig copies its
                // allowance. D6 value order: the retired shape first, then the
                // node readers, then the carry; every shape shares one cursor.
                // When the retired shape cannot be admitted the whole query keeps
                // HEAD's three-table eligibility path.
                let floor = workspace_floor(table, &graph.budget, l).ok();
                let allowance = floor.map_or(0, |floor| graph.budget.reader_allowance(Some(floor)));
                let mut cursor = ReaderAllowance::new(allowance);
                let retired = load_retired(table, &graph.generation, &mut cursor)?;
                if retired.is_unavailable() {
                    graph.fallback = true;
                } else {
                    if let Retired::Admitted { bytes, .. } = &retired {
                        graph.readers.pin(*bytes);
                    }
                    graph.retired = retired;
                    let plan = graph_reader_plan(&filter, project_vector);
                    graph.admit_readers(&plan, &mut cursor);
                    // The carry is offered after the node readers and its
                    // decline never closes admission; admitted bytes join the
                    // non-demotable slice. The fixed pair cache is offered last
                    // with the same non-closing decline and its own explicit
                    // charge; when it is declined or forced off, scoring falls
                    // back to one NodeIdentity read per scored id.
                    graph.admit_carry(l, force_carry_off, &mut cursor)?;
                    graph.admit_pairs(force_pair_cache_off, &mut cursor)?;
                }
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
        record_owner_visits(graph.budget.visits);
        Ok(QueryRows {
            results,
            vectors,
            anchor,
        })
    })();
    let rows = result?;
    // Both stamps were captured immediately after the anchor opened; any move
    // observed now fails closed on every route and before any row is returned.
    guard.verify(table)?;
    Ok(rows)
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
    fn retired_shape_is_offered_before_the_node_readers() {
        // The retired shape is offered first; the remaining envelope then covers
        // the node readers in priority order.
        let mut cursor = ReaderAllowance::new(30_000);
        assert_eq!(cursor.offer(Some(20_000)), Some(20_000));
        assert_eq!(cursor.offer(Some(9_000)), Some(9_000));
        assert_eq!(cursor.offer(Some(1_001)), None);
        assert!(cursor.closed());
        // A retired shape that does not fit closes admission: the whole query
        // falls back to the three-table path and no node reader is admitted.
        let mut cursor = ReaderAllowance::new(1_000);
        assert_eq!(cursor.offer(Some(1_001)), None);
        assert!(cursor.closed());
        assert_eq!(cursor.offer(Some(1)), None);
    }

    #[test]
    fn pinned_retired_shape_is_charged_and_never_demoted() {
        let head = 1_000_000;
        let mut readers = Readers::new();
        readers.pin(4_096);
        assert_eq!(readers.pinned_total(), 4_096);
        assert_eq!(readers.admitted_total(), 4_096);
        // Demotion only finalizes retained statements; the pinned set stays.
        readers.demote_all();
        assert_eq!(readers.pinned_total(), 4_096);
        assert_eq!(readers.admitted_total(), 4_096);
        // The pinned bytes reduce the copied core allowance by exactly their own
        // size plus the unchanged reprepare headroom.
        assert_eq!(
            core_bytes(head, readers.admitted_total()),
            head - 4_096 - REPREPARE_HEADROOM
        );
    }

    #[test]
    fn retired_set_membership_is_sorted_and_zero_set_means_live() {
        let empty = Retired::Empty;
        assert!(!empty.contains(7));
        assert!(!empty.is_unavailable());
        let admitted = Retired::Admitted {
            ids: vec![2, 5, 9],
            bytes: 24,
        };
        assert!(admitted.contains(2));
        assert!(admitted.contains(5));
        assert!(admitted.contains(9));
        assert!(!admitted.contains(6));
        // The fallback route never consults membership; it must report
        // unavailability so the whole query switches paths.
        assert!(Retired::Unavailable.is_unavailable());
        assert!(!Retired::Unavailable.contains(5));
    }

    #[test]
    fn detector_reports_either_stamp_move() {
        let stamp = ShadowStamp {
            tables: 9,
            rootpage_sum: 42,
        };
        assert!(!detector_moved(stamp, stamp, 7, 7));
        // A same-connection DML moves only the change counter.
        assert!(detector_moved(stamp, stamp, 7, 8));
        // A relocating DDL moves only the shadow-schema stamp.
        assert!(detector_moved(
            stamp,
            ShadowStamp {
                tables: 9,
                rootpage_sum: 43
            },
            7,
            7
        ));
        // A rename away drops the matched table count.
        assert!(detector_moved(
            stamp,
            ShadowStamp {
                tables: 8,
                rootpage_sum: 42
            },
            7,
            7
        ));
    }

    #[test]
    fn zero_allowance_admits_no_reader_and_keeps_head_core_budget() {
        let mut cursor = ReaderAllowance::new(0);
        assert_eq!(cursor.offer(Some(1)), None);
        let mut readers = Readers::new();
        assert!(readers.take(ReaderShape::NodeExpand).is_none());
        assert!(readers.take(ReaderShape::NodeScore).is_none());
        assert!(readers.take(ReaderShape::NodeIdentity).is_none());
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
                ReaderShape::NodeExpand,
                ReaderShape::NodeScore,
                ReaderShape::NodeIdentity,
                ReaderShape::ExactUnfiltered
            ]
        );
        let plan = graph_reader_plan(&SearchFilter::None, true);
        assert_eq!(
            plan.as_slice(),
            &[
                ReaderShape::NodeExpand,
                ReaderShape::NodeScore,
                ReaderShape::NodeIdentity,
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
                ReaderShape::NodeExpand,
                ReaderShape::NodeScore,
                ReaderShape::NodeIdentity,
                ReaderShape::ExactPoint
            ]
        );
        let ids = HashSet::from([1u64, 2]);
        let plan = graph_reader_plan(&SearchFilter::In(&ids), false);
        assert_eq!(
            plan.as_slice(),
            &[
                ReaderShape::NodeExpand,
                ReaderShape::NodeScore,
                ReaderShape::NodeIdentity,
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

    // --- resolved-rowid carry (T5) -----------------------------------------

    /// T0-recorded fixed budget constant for the small-budget fixture
    /// (degree=4, build_list_size=4, search_list_size=4, max_visits=4096, dim=2,
    /// 16 published rows; binding shape "filter"). T6 asserts the database-level
    /// carry-on/off outcome at one byte below/at/above this value; the constant
    /// and the pure carry arithmetic are pinned here and never re-derived.
    const T0_SMALL_BUDGET_MIN_ADMITTED_CACHE_BYTES: usize = 711_832;

    /// The T0 small-budget fixture's core workspace floor, computed with the same
    /// formulas the query path uses (no reader measurements involved).
    fn small_budget_fixture_floor() -> usize {
        let options = crate::index_options::DiskAnnOptions::parse(
            "diskann(degree=4,build_list_size=4,search_list_size=4,max_visits=4096)",
        )
        .expect("T0 small-budget options");
        GraphConfig {
            dimension: 2,
            distance: DistanceType::L2,
            max_degree: options.degree,
            construction_l: options.build_list_size,
            search_l: options.search_list_size,
            alpha: options.alpha,
            limits: ResourceLimits {
                max_visits: options.max_visits,
                max_cached_vectors: 2048,
                max_vector_bytes: 64 * 1024 * 1024,
            },
        }
        .workspace_floor(4)
        .expect("T0 small-budget floor")
    }

    #[test]
    fn carry_ring_wraps_and_overwrites_the_oldest_slot() {
        let mut carry = CarryRing::with_capacity(3).unwrap();
        carry.insert(10, Some(100));
        carry.insert(11, Some(110));
        carry.insert(12, None);
        assert_eq!(carry.get(11), Some(Some(110)));
        carry.insert(13, Some(130));
        // The oldest slot is overwritten; the ring never grows past capacity.
        assert_eq!(carry.get(10), None);
        assert_eq!(carry.get(13), Some(Some(130)));
        assert_eq!(carry.get(12), Some(None));
        assert_eq!(carry.slots.len(), 3);
    }

    #[test]
    fn carry_ring_preserves_optional_rowids_and_misses() {
        let mut carry = CarryRing::with_capacity(2).unwrap();
        carry.insert(7, None);
        carry.insert(8, Some(80));
        // Retired nodes and the frozen root resolve to a carried None rowid; that
        // is a hit, not a miss, and must stay distinguishable from one.
        assert_eq!(carry.get(7), Some(None));
        assert_eq!(carry.get(8), Some(Some(80)));
        assert_eq!(carry.get(9), None);
        // A zero-capacity ring is never admitted, but it must not panic.
        let mut empty = CarryRing::with_capacity(0).unwrap();
        empty.insert(1, Some(1));
        assert_eq!(empty.get(1), None);
    }

    #[test]
    fn carry_capacity_and_bytes_follow_the_effective_search_list() {
        for l in [1usize, 4, 10, 100, 65_536] {
            assert_eq!(carry_capacity(l).unwrap(), l + 1);
            assert_eq!(
                carry_bytes(l),
                (l + 1) * CARRY_SLOT_BYTES + CARRY_CONTAINER_BYTES
            );
        }
        // The ring holds one INTEGER id plus an optional INTEGER rowid, and the
        // fixed container slack stays inside the reprepare headroom.
        let slot = CARRY_SLOT_BYTES;
        let container = CARRY_CONTAINER_BYTES;
        assert_eq!(slot, size_of::<(u64, Option<u64>)>());
        assert!(container <= 64);
        assert!(container < REPREPARE_HEADROOM);
        // A capacity that cannot be represented is an error, never a wrap; the
        // byte charge saturates so the offer stays well formed at extremes.
        assert!(carry_capacity(usize::MAX).is_err());
        assert_eq!(
            carry_bytes(usize::MAX),
            usize::MAX
                .saturating_mul(CARRY_SLOT_BYTES)
                .saturating_add(CARRY_CONTAINER_BYTES)
        );
    }

    #[test]
    fn carry_consult_never_populates_and_only_post_processing_populates() {
        let mut carry = Some(CarryRing::with_capacity(1).unwrap());
        assert_eq!(carry_lookup(&carry, 5), None);
        // Setup, traversal, and final-loop callers consult but never populate.
        assert!(!carry_populates(false));
        // Only the post-processing decision read records its resolution.
        assert!(carry_populates(true));
        carry.as_mut().unwrap().insert(5, Some(50));
        assert_eq!(carry_lookup(&carry, 5), Some(Some(50)));
        // A disabled carry is a permanent miss: the caller must read for real.
        assert_eq!(carry_lookup(&None, 5), None);
    }

    #[test]
    fn carry_is_offered_last_and_declines_without_closing_admission() {
        // D6 value order: retired shape, node readers, then the carry.
        let mut cursor = ReaderAllowance::new(10_000);
        assert_eq!(cursor.offer(Some(4_000)), Some(4_000));
        assert_eq!(cursor.offer(Some(5_000)), Some(5_000));
        assert_eq!(cursor.offer_non_closing(Some(1_001)), None);
        assert!(!cursor.closed(), "a carry decline must not close admission");
        assert_eq!(cursor.offer_non_closing(Some(1_000)), Some(1_000));
        assert_eq!(cursor.remaining, 0);
        // A cursor already closed by a reader still declines the carry.
        let mut closed = ReaderAllowance::new(10);
        assert_eq!(closed.offer(Some(11)), None);
        assert!(closed.closed());
        assert_eq!(closed.offer_non_closing(Some(1)), None);
        assert!(closed.closed());
    }

    #[test]
    fn carry_bytes_are_pinned_non_demotable_and_reduce_core_bytes() {
        let head = 1_000_000;
        let carry = carry_bytes(10);
        let mut readers = Readers::new();
        readers.pin(carry);
        assert_eq!(readers.admitted_total(), carry);
        readers.demote_all();
        assert_eq!(
            readers.admitted_total(),
            carry,
            "pinned carry bytes are never demoted"
        );
        assert_eq!(
            core_bytes(head, readers.admitted_total()),
            head - carry - REPREPARE_HEADROOM
        );
        // The retired Vec and the carry share the non-demotable slice.
        let retired = 4_096usize;
        let mut both = Readers::new();
        both.pin(retired);
        both.pin(carry);
        assert_eq!(both.admitted_total(), retired + carry);
        both.demote_all();
        assert_eq!(both.admitted_total(), retired + carry);
    }

    #[test]
    fn carry_admission_keeps_the_core_floor_reserved() {
        // The retired shape, node readers, and carry all consume the same
        // cursor, so the copied core allowance can never drop below the floor
        // the envelope reserved, and one admitted carry always costs exactly its
        // charged bytes.
        let head = 900_000;
        let floor = 500_000;
        let allowance = reader_allowance(head, Some(floor));
        for carry in [0usize, carry_bytes(1), carry_bytes(10), carry_bytes(100)] {
            let readers = allowance.saturating_sub(carry);
            let mut cursor = ReaderAllowance::new(allowance);
            assert_eq!(cursor.offer(Some(readers)), Some(readers));
            assert_eq!(cursor.offer_non_closing(Some(carry)), Some(carry));
            let admitted = readers + carry;
            assert!(admitted <= allowance);
            let core = core_bytes(head, admitted);
            assert_eq!(core, head - admitted - REPREPARE_HEADROOM);
            assert!(
                core >= floor,
                "core allowance fell below the reserved floor"
            );
        }
    }

    #[test]
    fn t0_small_budget_carry_boundary_is_pinned_for_t6() {
        const MIN: usize = T0_SMALL_BUDGET_MIN_ADMITTED_CACHE_BYTES;
        let floor = small_budget_fixture_floor();
        assert!(floor > 0);
        assert!(
            floor < MIN,
            "the fixture's core floor fits inside the recorded minimum budget"
        );
        // The fixture always runs an effective L of 4, so the carry is five
        // slots: a few hundred charged bytes, well below the reprepare headroom.
        let carry = carry_bytes(4);
        assert_eq!(carry, 5 * CARRY_SLOT_BYTES + CARRY_CONTAINER_BYTES);
        assert!(carry < 256);
        assert!(carry < REPREPARE_HEADROOM);
        // Carry-on/carry-off boundary: with a reader set that exactly exhausts
        // the allowance (the meaning of the recorded T0 minimum) the carry is
        // disabled; it is enabled only once the spare envelope covers its whole
        // shape, and the decline never closes admission. The database-level
        // outcome at MIN - 1 / MIN / MIN + 1 is measured in T6 because the
        // owner reserve depends on runtime HashSet capacity.
        let readers = 70_000usize;
        let mut off = ReaderAllowance::new(readers);
        assert_eq!(off.offer(Some(readers)), Some(readers));
        assert_eq!(off.offer_non_closing(Some(carry)), None);
        assert!(!off.closed());
        let mut one_short = ReaderAllowance::new(readers + carry - 1);
        assert_eq!(one_short.offer(Some(readers)), Some(readers));
        assert_eq!(one_short.offer_non_closing(Some(carry)), None);
        assert!(!one_short.closed());
        let mut on = ReaderAllowance::new(readers + carry);
        assert_eq!(on.offer(Some(readers)), Some(readers));
        assert_eq!(on.offer_non_closing(Some(carry)), Some(carry));
    }

    // --- streamed adjacency and pair-cache workspace (D6) -------------------

    #[test]
    fn maximum_adjacency_workspace_matches_the_owner_term_and_declines_cleanly() {
        // The largest legal configuration is degree=10_000 => physical=13_000.
        // D6's owner adjacency term is exactly the streamed decode's simultaneous
        // footprint: the 16-byte slot blob copy plus the 8-byte decoded native id.
        let physical = 13_000usize;
        assert_eq!(physical * 16 + physical * size_of::<u64>(), physical * 24);
        // A maximal valid adjacency decodes in one pass with exactly one exact id
        // reservation and no pair buffer, so its transient bytes equal the owner
        // term and fit the unchanged reservation without any 64 KiB slack.
        let mut bytes = Vec::new();
        for id in 0..physical as u64 {
            bytes.extend_from_slice(&id.to_le_bytes());
            bytes.extend_from_slice(&(id + 1).to_le_bytes());
        }
        assert_eq!(bytes.len(), physical * 16);
        let mut ids = Vec::new();
        let mut streamed_pairs = 0usize;
        decode_slots_streamed(&bytes, physical, None, physical as u64, &mut ids, |_, _| {
            streamed_pairs += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(ids.len(), physical);
        assert_eq!(ids.capacity(), physical);
        assert_eq!(streamed_pairs, physical);
        assert_eq!(
            bytes.len() + ids.capacity() * size_of::<u64>(),
            physical * 24
        );
        // The pair cache is a separate explicit charge (16 B per slot plus fixed
        // container slack) and is never folded into the owner proof.
        let pair = slot_pair_bytes(physical);
        assert_eq!(pair, physical * SLOT_PAIR_BYTES + SLOT_PAIR_CONTAINER_BYTES);
        assert!(pair < physical * 24);
        // One admitted graph-route envelope: retired shape, readers, carry and
        // pair cache share the cursor; when the spare bytes cover the whole pair
        // shape it is admitted and the copied core allowance still holds the
        // floor, and one byte less declines the cache without closing admission
        // or shrinking the core floor.
        let head = 1_500_000usize;
        let floor = 700_000usize;
        let allowance = reader_allowance(head, Some(floor));
        let carry = carry_bytes(10);
        let readers = allowance - carry - pair;
        let mut cursor = ReaderAllowance::new(allowance);
        assert_eq!(cursor.offer(Some(readers)), Some(readers));
        assert_eq!(cursor.offer_non_closing(Some(carry)), Some(carry));
        assert_eq!(cursor.offer_non_closing(Some(pair)), Some(pair));
        let admitted = readers + carry + pair;
        assert_eq!(admitted, allowance);
        assert_eq!(
            core_bytes(head, admitted),
            head - admitted - REPREPARE_HEADROOM
        );
        assert!(core_bytes(head, admitted) >= floor);
        let mut tight = ReaderAllowance::new(allowance);
        let tight_readers = allowance - carry - pair + 1;
        assert_eq!(tight.offer(Some(tight_readers)), Some(tight_readers));
        assert_eq!(tight.offer_non_closing(Some(carry)), Some(carry));
        assert_eq!(tight.offer_non_closing(Some(pair)), None);
        assert!(
            !tight.closed(),
            "a pair-cache decline must not close admission"
        );
        assert!(core_bytes(head, tight_readers + carry) >= floor);
    }

    #[test]
    fn pair_cache_admission_is_last_and_declines_cleanly_at_tight_allowances() {
        let physical = 41usize;
        let pair = slot_pair_bytes(physical);
        assert_eq!(SLOT_PAIR_BYTES, size_of::<(u64, u64)>());
        assert_eq!(SLOT_PAIR_BYTES, 16);
        assert_eq!(pair, physical * 16 + SLOT_PAIR_CONTAINER_BYTES);
        // Container slack stays far below the reprepare headroom, like the carry.
        let container = SLOT_PAIR_CONTAINER_BYTES;
        assert!(container < REPREPARE_HEADROOM);
        // Exactly the charge admits; one byte short declines without closing the
        // cursor; a cursor already closed by a reader still declines.
        let mut exact = ReaderAllowance::new(pair);
        assert_eq!(exact.offer_non_closing(Some(pair)), Some(pair));
        let mut short = ReaderAllowance::new(pair - 1);
        assert_eq!(short.offer_non_closing(Some(pair)), None);
        assert!(!short.closed());
        let mut closed = ReaderAllowance::new(10);
        assert_eq!(closed.offer(Some(11)), None);
        assert!(closed.closed());
        assert_eq!(closed.offer_non_closing(Some(pair)), None);
        // D6 value order: the pair cache is offered after the retired shape, the
        // node readers and the carry, and its decline never closes admission.
        let mut cursor = ReaderAllowance::new(10_000);
        assert_eq!(cursor.offer(Some(4_000)), Some(4_000));
        assert_eq!(cursor.offer(Some(1_000)), Some(1_000));
        assert_eq!(cursor.offer_non_closing(Some(500)), Some(500));
        assert_eq!(cursor.offer_non_closing(Some(4_501)), None);
        assert!(!cursor.closed());
        // Explicitly charged bytes are pinned non-demotably like the retired set
        // and the carry, and reduce the copied core allowance by exactly their
        // own size plus the unchanged reprepare headroom.
        let mut readers = Readers::new();
        readers.pin(pair);
        assert_eq!(readers.admitted_total(), pair);
        readers.demote_all();
        assert_eq!(readers.admitted_total(), pair);
        let head = 1_000_000;
        assert_eq!(core_bytes(head, pair), head - pair - REPREPARE_HEADROOM);
        // The cache is bounded by the physical degree the decoder validated, and
        // a new expansion replaces the previous expansion's identities.
        let mut pairs = SlotPairs::with_capacity(2).unwrap();
        pairs.begin();
        pairs.push(1, 11);
        pairs.push(2, 22);
        pairs.push(3, 33);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs.entry_of(1), Some(11));
        assert_eq!(pairs.entry_of(3), None);
        pairs.begin();
        assert_eq!(pairs.len(), 0);
        assert_eq!(pairs.entry_of(1), None);
    }

    // --- retirement consistency (C1(ii), D10 row 4, T6') -------------------

    fn assert_corrupt<T: std::fmt::Debug>(result: Result<T>) -> IndexError {
        let error = result.expect_err("expected SQLITE_CORRUPT");
        assert_eq!(error.code, ffi::SQLITE_CORRUPT as i32);
        error
    }

    /// Fast route (D10 row 4, both directions). Snapshot 10, anchored revision 12:
    /// supported post-publish churn carries revisions above the snapshot, so set
    /// membership -- not the revision bound -- separates a booked bridge from an
    /// unbooked tamper.
    #[test]
    fn fast_retirement_requires_exact_membership_symmetry() {
        const SNAPSHOT: u64 = 10;
        const REVISION: u64 = 12;
        // Control: live rows and a correctly booked bridge, including one retired
        // after the frozen snapshot (supported post-publish churn).
        assert_eq!(
            fast_retirement_state(3, None, false, SNAPSHOT, REVISION).unwrap(),
            NodeState::Live
        );
        assert_eq!(
            fast_retirement_state(3, Some(SNAPSHOT + 1), true, SNAPSHOT, REVISION).unwrap(),
            NodeState::Deleted
        );
        assert_eq!(
            fast_retirement_state(3, Some(4), true, SNAPSHOT, REVISION).unwrap(),
            NodeState::Deleted
        );
        // Retire direction: a revision present while the entry is NOT in the set is
        // corruption -- one rule covering both the <= source_revision edit (4) and
        // the > source_revision edits (11, 12).
        for retired in [Some(4u64), Some(SNAPSHOT + 1), Some(REVISION)] {
            let error =
                assert_corrupt(fast_retirement_state(3, retired, false, SNAPSHOT, REVISION));
            assert!(error
                .message
                .contains("retirement bookkeeping disagreement"));
        }
        // Un-retire direction: a set member whose revision was cleared.
        let error = assert_corrupt(fast_retirement_state(3, None, true, SNAPSHOT, REVISION));
        assert!(error
            .message
            .contains("retirement bookkeeping disagreement"));
        // Interval validation runs first and includes set members: creation outside
        // the snapshot, ordering, and revisions beyond the anchored metadata
        // revision fail before membership decides.
        for (created, retired) in [
            (0u64, None),
            (SNAPSHOT + 1, None),
            (3, Some(3)),
            (3, Some(REVISION + 1)),
        ] {
            assert_corrupt(fast_retirement_state(
                created, retired, true, SNAPSHOT, REVISION,
            ));
            assert_corrupt(fast_retirement_state(
                created, retired, false, SNAPSHOT, REVISION,
            ));
        }
    }

    /// Fallback route keeps HEAD's exact classify asymmetry: intervals first, then
    /// current membership; no revision is required on the non-current side.
    #[test]
    fn fallback_retirement_keeps_head_classify_asymmetry() {
        const SNAPSHOT: u64 = 10;
        const REVISION: u64 = 12;
        // Live control.
        assert_eq!(
            fallback_retirement_state(3, None, true, SNAPSHOT, REVISION).unwrap(),
            NodeState::Live
        );
        // A still-current entry carrying ANY retirement revision is corruption.
        for retired in [Some(4u64), Some(SNAPSHOT + 1), Some(REVISION)] {
            let error = assert_corrupt(fallback_retirement_state(
                3, retired, true, SNAPSHOT, REVISION,
            ));
            assert!(error.message.contains("current graph entry is retired"));
        }
        // Non-current entries are Deleted bridges: a shadow-deleted current row
        // without a revision is excluded, not corruption (D10 row 3), so no
        // revision is required on this side.
        assert_eq!(
            fallback_retirement_state(3, None, false, SNAPSHOT, REVISION).unwrap(),
            NodeState::Deleted
        );
        assert_eq!(
            fallback_retirement_state(3, Some(SNAPSHOT + 1), false, SNAPSHOT, REVISION).unwrap(),
            NodeState::Deleted
        );
        // A present revision is still interval-validated on the non-current side.
        for retired in [Some(0u64), Some(3), Some(REVISION + 1)] {
            let error = assert_corrupt(fallback_retirement_state(
                3, retired, false, SNAPSHOT, REVISION,
            ));
            assert!(error.message.contains("immutable interval invalid"));
        }
        // Creation must lie inside the frozen snapshot on both sides.
        for current in [true, false] {
            assert_corrupt(fallback_retirement_state(
                0, None, current, SNAPSHOT, REVISION,
            ));
            assert_corrupt(fallback_retirement_state(
                SNAPSHOT + 1,
                None,
                current,
                SNAPSHOT,
                REVISION,
            ));
        }
    }

    /// Exact/projection routes restore HEAD's eligible_identity interval checks.
    #[test]
    fn exact_routes_require_unretired_entries_inside_the_snapshot() {
        const SNAPSHOT: u64 = 10;
        for created in [1u64, 5, SNAPSHOT] {
            assert!(eligible_entry_interval(created, true, SNAPSHOT));
        }
        // Creation outside the frozen snapshot is corruption, never exclusion.
        for created in [0u64, SNAPSHOT + 1, u64::MAX] {
            assert!(!eligible_entry_interval(created, true, SNAPSHOT));
        }
        // A retired entry can never be served by an exact/projection shape.
        for created in [1u64, SNAPSHOT] {
            assert!(!eligible_entry_interval(created, false, SNAPSHOT));
        }
        assert!(!eligible_entry_interval(0, false, SNAPSHOT));
        // Exact-shape note (T6'): the SQL-level route split for the un-retire
        // variant B (ExactPoint serves; ExactUnfiltered fails its population /
        // eligible_count guard exactly as HEAD) is pinned by the Python suite,
        // because this crate never opens a SQLite connection from a unit test.
        // This predicate is the shared rule both exact shapes apply per row.
    }

    /// The restored retirement checks read fixed column positions, so pin the SQL
    /// shapes: the exact/projection plan joins `entries` unconditionally (both
    /// `vector = true` and `internal_id`'s `vector = false`) and projects the
    /// interval columns in the same order; the fallback projects the entries
    /// interval and current columns before the `n.vector IS NULL` role flag.
    #[test]
    fn scored_row_sql_shapes_pin_the_restored_retirement_columns() {
        let names = test_storage_names();
        for vector in [false, true] {
            for point in [false, true] {
                let sql = exact_sql_for(&names, 8, point, vector);
                assert!(sql.contains("LEFT JOIN v_diskann_entries e ON e.entry_id=n.entry_id"));
                assert!(sql.contains(
                    "SELECT n.native_local_id,n.entry_id,n.public_rowid,e.created_revision,e.retired_revision,c.public_rowid"
                ));
                assert_eq!(sql.contains("e.vector"), vector);
                assert_eq!(sql.contains(" AND c.public_rowid=?2"), point);
            }
        }
        let identity = fallback_sql_for(&names, 8, false);
        let payload = fallback_sql_for(&names, 8, true);
        for sql in [&identity, &payload] {
            assert!(sql.contains(
                "e.public_rowid,e.created_revision,e.retired_revision,c.entry_id,(n.vector IS NULL)"
            ));
        }
        assert!(payload.contains("CASE WHEN n.state=2 THEN n.vector ELSE e.vector END"));
        assert!(!identity.contains("CASE WHEN"));
    }

    fn test_storage_names() -> StorageNames {
        StorageNames {
            schema: "main".to_owned(),
            table: "v".to_owned(),
            meta: "v_diskann_meta".to_owned(),
            entries: "v_diskann_entries".to_owned(),
            current: "v_diskann_current".to_owned(),
            generations: "v_diskann_generations".to_owned(),
            graph_nodes: "v_diskann_graph_nodes".to_owned(),
            generation_retired: "v_diskann_generation_retired".to_owned(),
            jobs: "v_diskann_jobs".to_owned(),
            snapshot_map: "v_diskann_snapshot_map".to_owned(),
            txn: "v_diskann_txn".to_owned(),
        }
    }

    // --- retired-set load bound (reentrant growth) --------------------------

    /// Admission charges exactly `count * 8` bytes for the retired id Vec, so the
    /// loader must refuse an extra row before materializing it. This exercises the
    /// per-row bound directly: the Vec keeps its exact admitted capacity and never
    /// grows past the admitted population.
    #[test]
    fn retired_load_never_materializes_more_than_the_admitted_population() {
        const ADMITTED: usize = 3;
        let mut ids = Vec::new();
        reserve(&mut ids, ADMITTED).unwrap();
        assert_eq!(ids.capacity(), ADMITTED);
        let mut previous = 0u64;
        for entry in [2u64, 5, 9] {
            push_retired_entry(&mut ids, ADMITTED, entry, &mut previous).unwrap();
        }
        assert_eq!(ids, [2, 5, 9]);
        assert_eq!(ids.capacity(), ADMITTED);
        // A reentrant growth row must fail closed before it is appended.
        let error = push_retired_entry(&mut ids, ADMITTED, 11, &mut previous).unwrap_err();
        assert_eq!(error.code, ffi::SQLITE_CORRUPT as i32);
        assert!(error.message.contains("admitted population"));
        assert_eq!(ids.len(), ADMITTED);
        assert_eq!(ids.capacity(), ADMITTED);
        assert_eq!(previous, 9);
        // Ordering violations also fail before the append while the admitted
        // population still has room.
        let mut ordered = Vec::new();
        reserve(&mut ordered, ADMITTED).unwrap();
        let mut previous = 2u64;
        for bad in [0u64, 2] {
            let error = push_retired_entry(&mut ordered, ADMITTED, bad, &mut previous).unwrap_err();
            assert!(error.message.contains("not ordered"));
            assert!(ordered.is_empty());
            assert_eq!(ordered.capacity(), ADMITTED);
        }
        // A population that ends short of the admitted count is caught by the
        // loader's final length comparison, never by the per-row bound.
        let mut short = Vec::new();
        reserve(&mut short, ADMITTED).unwrap();
        let mut previous = 0u64;
        push_retired_entry(&mut short, ADMITTED, 1, &mut previous).unwrap();
        assert!(short.len() < ADMITTED);
    }
}
