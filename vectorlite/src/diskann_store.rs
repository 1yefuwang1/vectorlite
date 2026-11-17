//! SQLite-authoritative deferred DiskANN canonical/version storage (Format4).
//! Mutations remain inside the ordinary two-row journal carrier. No constructor,
//! graph traversal, runtime, background work, recovery write, or GC is performed
//! by canonical DML or open. All names refer to the borrowed host connection.
use crate::batch_input::BatchView;
use crate::core::{SearchFilter, SearchResult};
use crate::diskann_core::{StoredVector, ValidatedVector, VectorValidationError};
use crate::ffi;
use crate::index_error::IndexError;
use crate::index_options::DiskAnnOptions;
use crate::ops;
use crate::sqlite::{qualified_name, quote_identifier, Connection, Statement, Step};
use crate::vector_space::{DistanceType, NamedVectorSpace, VectorType};
use std::cell::RefCell;
use std::mem::size_of;
type Result<T> = std::result::Result<T, IndexError>;
const FORMAT_VERSION: i64 = 4;
const MAX_DESCRIPTOR_BYTES: usize = 16 * 1024;
const MAX_ID: u64 = i64::MAX as u64;
const MAX_INSERT_BATCH: usize = 32;
/// Upper bound on the SQLite VM memory one table's retained write statements may
/// hold, matching the query reader cap.
const MAX_WRITER_BYTES: usize = 256 * 1024;
/// Reserve above the admitted writers for SQLite's transparent reprepare growth
/// after DDL, which rewrites a retained statement's compiled program in place.
const WRITER_REPREPARE_HEADROOM: usize = 16 * 1024;
#[path = "diskann_deferred/codec.rs"]
pub(crate) mod codec;
use codec::*;
#[path = "diskann_deferred/import.rs"]
pub(crate) mod import;
#[path = "diskann_deferred/jobs.rs"]
pub(crate) mod jobs;
#[path = "diskann_deferred/lifecycle.rs"]
pub(crate) mod lifecycle;
#[path = "diskann_deferred/schema.rs"]
mod schema;

/// Qualified names are immutable across callbacks; rollback cannot poison them.
#[derive(Clone, Debug)]
pub(crate) struct StorageNames {
    pub(crate) schema: String,
    pub(crate) table: String,
    pub(crate) meta: String,
    pub(crate) entries: String,
    pub(crate) current: String,
    pub(crate) generations: String,
    pub(crate) graph_nodes: String,
    pub(crate) jobs: String,
    pub(crate) snapshot_map: String,
    pub(crate) txn: String,
}
impl StorageNames {
    pub(crate) fn base(&self, suffix: &str) -> String {
        format!("{}_{}", self.table, suffix)
    }
    pub(crate) fn qualified(&self, suffix: &str) -> Result<String> {
        qualified_name(&self.schema, &self.base(suffix))
    }
    fn new(schema: &str, table: &str) -> Result<Self> {
        let name = |suffix| qualified_name(schema, &format!("{table}_{suffix}"));
        Ok(Self {
            schema: schema.into(),
            table: table.into(),
            meta: name("diskann_meta")?,
            entries: name("diskann_entries")?,
            current: name("diskann_current")?,
            generations: name("diskann_generations")?,
            graph_nodes: name("diskann_graph_nodes")?,
            jobs: name("diskann_jobs")?,
            snapshot_map: name("diskann_snapshot_map")?,
            txn: name("diskann_txn")?,
        })
    }
}

pub(crate) struct DiskAnnTable {
    connection: Connection<'static>,
    names: StorageNames,
    space: NamedVectorSpace,
    options: DiskAnnOptions,
    descriptor: serde_json::Value,
    vector_bytes: usize,
    adjacency_bytes: usize,
    /// Non-Send retained writer statements, finalized in Drop before the
    /// borrowed host connection can close.
    writer: RefCell<WriterStatements>,
}
impl Drop for DiskAnnTable {
    fn drop(&mut self) {
        // D5: retained statements are finalized here, on the callback thread,
        // before the table (and its connection borrow) goes away. A deferred
        // finalize error is unreportable at this point and a statement is only
        // ever retained after it drained to DONE, so no step error can be lost.
        self.writer.get_mut().evict();
    }
}
pub(crate) struct QueryRows {
    pub(crate) results: Vec<SearchResult>,
    pub(crate) vectors: Option<Vec<Vec<u8>>>,
    pub(crate) anchor: Statement<'static>,
}
/// IDs are committed non-reused allocators; rolled-back IDs never escape SQL.
#[derive(Clone, Debug)]
pub(crate) struct Metadata {
    pub(crate) instance_id: [u8; 16],
    pub(crate) revision: u64,
    pub(crate) next_entry_id: u64,
    pub(crate) next_job_id: u64,
    pub(crate) next_generation_id: u64,
    /// Committed fences are never reused; only T3 controls allocate this value.
    pub(crate) next_fence: u64,
    /// Last examined historical entry; zero resets the bounded lifecycle scan.
    pub(crate) cleanup_entry_id: u64,
    pub(crate) canonical_count: u64,
    pub(crate) retired_count: u64,
    pub(crate) published_generation: Option<u64>,
    pub(crate) candidate_generation: Option<u64>,
    pub(crate) active_job: Option<u64>,
}
#[derive(Debug)]
pub(crate) struct EntryRecord {
    pub(crate) entry_id: u64,
    pub(crate) public_rowid: u64,
    pub(crate) created_revision: u64,
    pub(crate) retired_revision: Option<u64>,
    pub(crate) vector: Vec<u8>,
}
#[derive(Debug)]
pub(crate) struct Generation {
    pub(crate) generation_id: u64,
    pub(crate) job_id: u64,
    pub(crate) state: u64,
    pub(crate) source_revision: u64,
    pub(crate) source_hash: Option<[u8; 32]>,
    pub(crate) config_hash: [u8; 32],
    pub(crate) total_count: u64,
    pub(crate) eligible_count: u64,
    pub(crate) root_local_id: Option<u64>,
}
/// Graph payloads retain native local IDs, including LIVE zero and frozen N.
#[allow(dead_code)] // T5/T6 consume these checked operation-local records.
#[derive(Debug)]
pub(crate) struct GraphNodeRecord {
    pub(crate) native_local_id: u64,
    pub(crate) entry_id: Option<u64>,
    pub(crate) frozen_vector: Option<Vec<u8>>,
    pub(crate) neighbors: Vec<u64>,
}
/// Durable progress is authoritative; no state here survives a callback.
#[allow(dead_code)] // Snapshot/control registration is integrated by the lead.
#[derive(Debug)]
pub(crate) struct Job {
    pub(crate) job_id: u64,
    pub(crate) generation_id: u64,
    pub(crate) parent_generation: Option<u64>,
    pub(crate) cutoff_revision: u64,
    pub(crate) cutoff_entry_id: u64,
    /// Opaque resumable SHA chain; T3 defines seed/progress, not storage.
    pub(crate) snapshot_chain: [u8; 32],
    pub(crate) fence: u64,
    pub(crate) owner_nonce: Option<[u8; 16]>,
    pub(crate) phase: u64,
    pub(crate) expected_count: u64,
    pub(crate) scan_entry_id: u64,
    pub(crate) captured_count: u64,
    pub(crate) import_cursor: u64,
    pub(crate) validation_cursor: u64,
    pub(crate) cleanup_cursor: u64,
    pub(crate) source_hash: Option<[u8; 32]>,
    pub(crate) config_hash: [u8; 32],
    pub(crate) candidate_hash: Option<[u8; 32]>,
    pub(crate) failure_code: Option<i64>,
}
fn optional_id(s: &Statement<'_>, column: usize, name: &str) -> Result<Option<u64>> {
    if s.column_is_null(column)? {
        Ok(None)
    } else {
        Ok(Some(nonnegative(s, column, name)?))
    }
}
/// Drains one statement to DONE. A statement that never reaches DONE is an
/// error, so a retained slot can never be reset from a half-executed statement.
fn drain(statement: &mut Statement<'_>) -> Result<()> {
    while statement.step()? == Step::Row {}
    Ok(())
}
fn done(mut statement: Statement<'_>) -> Result<()> {
    drain(&mut statement)?;
    statement.finish()
}

/// Stable slot numbering for the retained per-row write statements (D5).
///
/// The five catalog pragma shapes already bind the layout name as a parameter, so
/// one retained statement per shape serves all eight layouts: SQLite replaces a
/// per-layout compile with an argument. Each slot has exactly one SQL text,
/// derived only from immutable per-table state (qualified names, vector bytes,
/// physical degree, schema), so a retained statement is never reused for a
/// different statement.
const SLOT_CATALOG_KIND: usize = 0;
const SLOT_CATALOG_COLUMNS: usize = 1;
const SLOT_CATALOG_INDEX_LIST: usize = 2;
const SLOT_CATALOG_INDEX_INFO: usize = 3;
const SLOT_CATALOG_FOREIGN_KEYS: usize = 4;
const SLOT_GUARD_INDEX_LIST: usize = 5;
const SLOT_GUARD_FOREIGN_KEYS: usize = 6;
const SLOT_GUARD_ROWS: usize = 7;
const SLOT_GUARD_TRIGGER: usize = 8;
const SLOT_GUARD_TEMP_TRIGGER: usize = 9;
const SLOT_JOURNAL_MODE: usize = 10;
const SLOT_FORMAT: usize = 11;
const SLOT_METADATA: usize = 12;
const SLOT_GENERATION: usize = 13;
const SLOT_GRAPH_NODE: usize = 14;
const SLOT_JOB: usize = 15;
const SLOT_LIVE_ID: usize = 16;
const SLOT_INSERT_ENTRY: usize = 17;
const SLOT_INSERT_CURRENT: usize = 18;
const SLOT_UPDATE_METADATA: usize = 19;
const SLOT_RETIRE_NODE: usize = 20;
const SLOT_RETIRE_ELIGIBLE: usize = 21;
const SLOT_RETIRE_ENTRY: usize = 22;
const SLOT_RETIRE_CURRENT: usize = 23;

/// Whether a slot may be retained across uses.
///
/// Only read-only statements are retained. Retaining a DML statement and
/// re-executing it from inside the journal-carrier callback makes SQLite's
/// debug build abort in `sqlite3BtreeInsert` ("BTCF_ValidNKey", the
/// BTREE_SAVEPOSITION branch) as soon as the carrier resumes its own
/// one-pass UPDATE: the nested write leaves the outer cursor's cached key
/// stale. Release builds compile that assertion out, so this is not merely a
/// debug-build inconvenience - it is a state inconsistency we do not want to
/// ship. The measured cost of the four write statements is small next to the
/// read paths (catalog, guard, journal, format, metadata, identity, live-id),
/// which stay retained.
const fn retains_statement(slot: usize) -> bool {
    slot <= SLOT_LIVE_ID
}
/// Number of retained writer slots; every slot is lazy and may stay empty.
const WRITER_SLOTS: usize = 24;

/// Proof that the frontend's pre-write live-id probe already observed a public
/// rowid absent in this callback, so insertion must not probe a second time.
///
/// The token is produced only where that probe runs (the virtual-table insert
/// branch) and consumed only by insert(), which keeps the single probe before
/// every write.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LiveIdProbe(());

impl LiveIdProbe {
    /// Records the frontend's completed live-id probe.
    ///
    /// Callers must have executed this table's live-id probe for the inserted
    /// rowid earlier in the same callback, before any write.
    pub(crate) fn after_frontend_probe() -> Self {
        Self(())
    }
}

/// Space a retained writer set may occupy: cache_bytes minus the canonical
/// workspace reservation and the reprepare headroom, capped by MAX_WRITER_BYTES.
fn writer_allowance(cache_bytes: usize, floor: usize, reservation: usize) -> usize {
    cache_bytes
        .saturating_sub(floor)
        .saturating_sub(reservation)
        .saturating_sub(WRITER_REPREPARE_HEADROOM)
        .min(MAX_WRITER_BYTES)
}

/// One retained statement plus the budget evidence that admitted it.
struct Retained {
    statement: Statement<'static>,
    /// MEMUSED at admission, refreshed after any transparent reprepare.
    bytes: usize,
    /// REPREPARE counter sampled at that measurement.
    reprepares: i32,
}

/// Admission state of a table's retained writer set.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Admission {
    /// Measured statements may still be retained.
    Open,
    /// The measured envelope is spent; this set keeps what it proved and
    /// prepares everything else per use until a new budget re-opens admission.
    Full,
    /// SQLite reported no MEMUSED for a statement, so nothing is ever retained
    /// for this table: exactly today's prepare/finalize behaviour.
    Unavailable,
}

/// Per-table retained statement set for the per-row write path (D5-D7).
///
/// Every retained statement binds copied values only: no scoped pointer, no
/// borrowed sqlite3_value, and no callback lease. A statement is taken out of
/// its slot while it runs and is only put back after it drained to DONE, so a
/// statement left on ROW or in FAILED is finalized instead of reused.
struct WriterStatements {
    slots: [Option<Retained>; WRITER_SLOTS],
    /// Workspace reservation the allowance was computed against.
    reservation: usize,
    /// Largest aggregate MEMUSED this set may retain.
    allowance: usize,
    admission: Admission,
    /// Only the per-row write path enables retention.
    armed: bool,
    /// Set while a maintenance/control operation owns the budget. Retention is
    /// refused for the whole operation, including the carrier's own pre-guard
    /// arm, so the set cannot be re-admitted underneath it.
    suppressed: bool,
}

impl WriterStatements {
    fn new() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
            reservation: 0,
            allowance: 0,
            admission: Admission::Open,
            armed: false,
            suppressed: false,
        }
    }
    fn take(&mut self, slot: usize) -> Option<Retained> {
        self.slots.get_mut(slot).and_then(Option::take)
    }
    /// Summed measurements of the retained statements, so a transparent
    /// reprepare is accounted at its new size rather than its admission size.
    fn aggregate(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .map(|retained| retained.bytes)
            .sum()
    }
    /// Drops the retained set and closes retention for the rest of this callback
    /// sequence, so a query or maintenance operation started after the drop
    /// cannot re-admit statements against the budget it just claimed. The next
    /// write-path arm re-opens it.
    fn release(&mut self) {
        self.evict();
        self.armed = false;
        if self.admission == Admission::Full {
            self.admission = Admission::Open;
        }
    }
    /// Finalizes every retained statement. Eviction is a budget decision, so a
    /// cleanup error never replaces a primary result.
    fn evict(&mut self) {
        for slot in &mut self.slots {
            if let Some(retained) = slot.take() {
                let _ = retained.statement.finish();
            }
        }
    }
    /// Re-derives the envelope for a callback's workspace reservation.
    ///
    /// A tighter reservation evicts the set only when its measured footprint no
    /// longer fits; a looser or unchanged one keeps it. A permanently
    /// unavailable measurement is never re-opened.
    fn arm(&mut self, reservation: usize, allowance: usize) {
        // A suppressed set stays empty: the operation that suppressed it owns
        // the budget, including through any carrier pre-guard that calls here.
        if self.suppressed {
            self.evict();
            self.armed = false;
            return;
        }
        if !self.armed || self.reservation != reservation {
            if self.aggregate() > allowance {
                self.evict();
            }
            if self.admission == Admission::Full {
                self.admission = Admission::Open;
            }
            self.reservation = reservation;
            self.allowance = allowance;
        }
        self.armed = true;
        if self.admission == Admission::Open && allowance == 0 {
            self.admission = Admission::Full;
        }
    }
    /// Takes ownership of a completed statement when the measured budget admits
    /// it; otherwise finalizes it and keeps today's per-use behaviour.
    fn admit(&mut self, slot: usize, statement: Statement<'static>) {
        if !retains_statement(slot) || !self.armed || self.admission != Admission::Open {
            let _ = statement.finish();
            return;
        }
        let Some(bytes) = statement.memory_used() else {
            // D7: unavailable MEMUSED means scalar fallback, not a free statement.
            self.admission = Admission::Unavailable;
            self.evict();
            let _ = statement.finish();
            return;
        };
        if self.aggregate().saturating_add(bytes) > self.allowance {
            // Admission ends at the first statement that does not fit; the
            // already admitted slots keep their measured proof.
            self.admission = Admission::Full;
            let _ = statement.finish();
            return;
        }
        let reprepares = statement.reprepares();
        if let Some(target) = self.slots.get_mut(slot) {
            *target = Some(Retained {
                statement,
                bytes,
                reprepares,
            });
        }
    }
    /// Re-measures a statement whose compiled program was transparently rebuilt.
    ///
    /// Growth beyond the admitted envelope demotes the whole set to today's
    /// per-use behaviour rather than letting an estimate drift.
    fn settle(&mut self, slot: usize, mut retained: Retained) {
        let reprepares = retained.statement.reprepares();
        if reprepares != retained.reprepares {
            let Some(bytes) = retained.statement.memory_used() else {
                self.admission = Admission::Unavailable;
                let _ = retained.statement.finish();
                self.evict();
                return;
            };
            retained.bytes = bytes;
            retained.reprepares = reprepares;
        }
        if let Some(target) = self.slots.get_mut(slot) {
            *target = Some(retained);
        }
        if self.aggregate() > self.allowance {
            self.admission = Admission::Full;
            self.evict();
        }
    }
}

/// Restores writer retention when a maintenance or control action finishes.
///
/// Dropping this guard does not re-admit anything: the next per-row write arms
/// the set again from the budget it then holds.
pub(crate) struct WriterSuppression<'a>(&'a DiskAnnTable);
impl Drop for WriterSuppression<'_> {
    fn drop(&mut self) {
        self.0.restore_writer_retention();
    }
}

/// The one live-id projection read by the frontend probe and the write path.
fn live_id_sql(table: &DiskAnnTable) -> String {
    format!("SELECT c.entry_id,e.public_rowid,e.created_revision,e.retired_revision FROM {} c LEFT JOIN {} e ON e.entry_id=c.entry_id WHERE c.public_rowid=?1",table.names.current,table.names.entries)
}
/// The one durable job projection read by scalar and retained callers alike.
fn job_sql(table: &DiskAnnTable) -> String {
    format!("SELECT generation_id,parent_generation,cutoff_revision,fence,owner_nonce IS NULL,{},phase,expected_count,scan_entry_id,captured_count,import_cursor,validation_cursor,cleanup_cursor,source_hash IS NULL,{}, {},candidate_hash IS NULL,{},failure_code,cutoff_entry_id,{} FROM {} WHERE job_id=?1",blob_projection("owner_nonce",16,true),blob_projection("source_hash",32,true),blob_projection("config_hash",32,true),blob_projection("candidate_hash",32,true),blob_projection("snapshot_chain",32,true),table.names.jobs)
}
/// The one graph-node projection read by scalar and retained callers alike.
fn graph_node_sql(table: &DiskAnnTable) -> String {
    format!("SELECT entry_id,state,{},{},frozen_vector IS NULL FROM {} WHERE generation_id=?1 AND native_local_id=?2",blob_projection("frozen_vector",table.vector_bytes,true),blob_projection("neighbors",table.adjacency_bytes,false),table.names.graph_nodes)
}
/// The one generation projection read by both scalar and retained callers.
fn generation_sql(table: &DiskAnnTable) -> String {
    format!("SELECT job_id,state,source_revision,{},{},total_count,eligible_count,root_local_id,source_hash IS NULL FROM {} WHERE generation_id=?1",blob_projection("source_hash",32,true),blob_projection("config_hash",32,true),table.names.generations)
}
/// The one metadata projection read by both the anchor and retained readers.
fn metadata_sql(table: &DiskAnnTable) -> String {
    format!("SELECT format_version,{},{},revision,next_entry_id,next_job_id,next_generation_id,canonical_count,retired_count,published_generation,candidate_generation,active_job,(SELECT count(*) FROM (SELECT singleton FROM {} LIMIT 2)),next_fence,cleanup_entry_id FROM {} WHERE singleton=1",blob_projection("instance_id",16,true),blob_projection("descriptor",MAX_DESCRIPTOR_BYTES,false),table.names.meta,table.names.meta)
}
/// Moves a retained statement through one reuse cycle: reset, clear, bind.
fn resume_writer(statement: &mut Statement<'static>) -> Result<()> {
    statement.reset()?;
    statement.clear_bindings()
}

impl DiskAnnTable {
    pub(crate) fn open(
        connection: Connection<'static>,
        schema: &str,
        table: &str,
        space: NamedVectorSpace,
        options: DiskAnnOptions,
        is_create: bool,
    ) -> Result<Self> {
        options.validate()?;
        if space.dim == 0
            || space.vector_type != VectorType::Float32
            || !matches!(space.distance_type, DistanceType::L2 | DistanceType::Cosine)
        {
            return Err(IndexError::new(
                "DiskANN supports only positive-dimension float32 squared L2 or cosine",
            ));
        }
        let vector_bytes = checked_product(space.dim, 4, "vector")?;
        if vector_bytes > i32::MAX as usize {
            return Err(too_big("DiskANN vector exceeds SQLite BLOB length limit"));
        }
        let adjacency_bytes = checked_product(options.physical_degree(), 8, "adjacency")?;
        let descriptor = descriptor(&space, &options);
        let handle = Self {
            connection,
            names: StorageNames::new(schema, table)?,
            space,
            options,
            descriptor,
            vector_bytes,
            adjacency_bytes,
            writer: RefCell::new(WriterStatements::new()),
        };
        handle.check_workspace(0)?;
        if is_create {
            handle.require_rollback_journal()?;
            handle.create_schema()?;
        }
        // Format is checked first: an old database never gets a mutation or migration.
        handle.check_format()?;
        handle.validate_schema()?;
        let (metadata, anchor) = handle.read_metadata()?;
        handle.verify_guard()?;
        handle.validate_generation_identities(&metadata)?;
        anchor.finish()?;
        Ok(handle)
    }
    pub(crate) fn space(&self) -> &NamedVectorSpace {
        &self.space
    }
    pub(crate) fn options(&self) -> &DiskAnnOptions {
        &self.options
    }
    // Integration seams are read-only, host-thread-bound, and never cache progress.
    #[allow(dead_code)]
    pub(crate) fn names(&self) -> &StorageNames {
        &self.names
    }
    #[allow(dead_code)]
    pub(crate) fn connection(&self) -> &Connection<'static> {
        &self.connection
    }
    pub(crate) fn vector_bytes(&self) -> usize {
        self.vector_bytes
    }
    /// Canonical workspace reservation, before any caller-specific request.
    fn workspace_floor(&self) -> Result<usize> {
        let names = schema::LAYOUTS.iter().try_fold(0usize, |sum, l| {
            sum.checked_add(self.names.base(l.suffix).len())
                .ok_or_else(|| too_big("DiskANN SQL identifier workspace overflow"))
        })?;
        checked_product(self.vector_bytes, 3, "canonical buffers")?
            .checked_add(checked_product(names, 16, "SQL names")?)
            .and_then(|v| v.checked_add(24 * 1024))
            .ok_or_else(|| too_big("DiskANN canonical workspace overflow"))
    }
    fn check_workspace(&self, extra: usize) -> Result<()> {
        let minimum = self
            .workspace_floor()?
            .checked_add(extra)
            .ok_or_else(|| too_big("DiskANN canonical workspace overflow"))?;
        if minimum > self.options.cache_bytes {
            return Err(too_big("DiskANN canonical workspace exceeds cache_bytes"));
        }
        Ok(())
    }
    /// Enables and budgets retention for the current per-row write callback.
    ///
    /// Called only from the write path (verify_writable_guard and the native
    /// batch), so read-only operations - open, control status, maintenance steps
    /// and queries - keep preparing their statements exactly as they do today.
    fn arm_writer(&self, reservation: usize) -> Result<()> {
        let allowance = writer_allowance(
            self.options.cache_bytes,
            self.workspace_floor()?,
            reservation,
        );
        self.writer.borrow_mut().arm(reservation, allowance);
        Ok(())
    }
    /// Drops the retained set before an operation that needs the same budget.
    ///
    /// Query reads, accepted-row projection and live-id lookups that follow it
    /// then prepare per use, exactly as before retention existed.
    pub(crate) fn release_writer(&self) {
        self.writer.borrow_mut().release();
    }
    /// Suppresses retention for the duration of a maintenance or control action.
    ///
    /// The carrier's pre-guard calls `arm_writer`, so a one-shot release before
    /// `atomic_write` would be undone immediately. Suppression spans preparation,
    /// the carrier and the action, and the guard restores the previous state when
    /// it drops, so the next per-row write can retain again.
    pub(crate) fn suppress_writer_retention(&self) -> WriterSuppression<'_> {
        {
            let mut writer = self.writer.borrow_mut();
            writer.suppressed = true;
            writer.release();
        }
        WriterSuppression(self)
    }
    fn restore_writer_retention(&self) {
        self.writer.borrow_mut().suppressed = false;
    }
    /// Runs one logical statement use against a retained slot.
    ///
    /// A retained statement is reused only from DONE: it is reset and cleared
    /// before the action, and put back only after the action succeeded (which
    /// means it drained to DONE). Any other outcome finalizes it and propagates
    /// the original SQL error unchanged, so nothing is ever retried. The SQL text
    /// is built only when a statement actually has to be prepared.
    fn with_writer<T>(
        &self,
        slot: usize,
        sql: impl FnOnce() -> String,
        action: impl FnOnce(&mut Statement<'static>) -> Result<T>,
    ) -> Result<T> {
        let cached = self
            .writer
            .borrow_mut()
            .take(slot)
            .filter(|_| retains_statement(slot));
        let Some(mut retained) = cached else {
            let mut statement = self.connection.prepare(&sql())?;
            return match action(&mut statement) {
                Ok(value) => {
                    self.writer.borrow_mut().admit(slot, statement);
                    Ok(value)
                }
                Err(error) => {
                    let _ = statement.finish();
                    Err(error)
                }
            };
        };
        if let Err(error) = resume_writer(&mut retained.statement) {
            // A statement left on ROW or in FAILED is never reused.
            let _ = retained.statement.finish();
            return Err(error);
        }

        match action(&mut retained.statement) {
            Ok(value) => {
                self.writer.borrow_mut().settle(slot, retained);
                Ok(value)
            }
            Err(error) => {
                let _ = retained.statement.finish();
                Err(error)
            }
        }
    }
    fn check_format(&self) -> Result<()> {
        self.with_writer(
            SLOT_FORMAT,
            || {
                format!(
                    "SELECT format_version FROM {} WHERE singleton=1",
                    self.names.meta
                )
            },
            |s| {
                if s.step()? != Step::Row {
                    return Err(corrupt("DiskANN singleton metadata is missing"));
                }
                if nonnegative(s, 0, "format version")? != FORMAT_VERSION as u64 {
                    return Err(corrupt(
                    "DiskANN format version is unsupported; Format4 requires recreation (no migration)",
                ));
                }
                if s.step()? != Step::Done {
                    return Err(corrupt("DiskANN singleton metadata is not unique"));
                }
                Ok(())
            },
        )
    }
    /// Metadata reader whose cursor the caller keeps open as a read anchor.
    ///
    /// Snapshot-holding callers (the cursor anchor, status) keep this fresh,
    /// owned statement; the per-row write path uses read_metadata_drained.
    pub(crate) fn read_metadata(&self) -> Result<(Metadata, Statement<'static>)> {
        let mut anchor = self.connection.prepare(&metadata_sql(self))?;
        if anchor.step()? != Step::Row {
            return Err(corrupt("DiskANN metadata missing"));
        }
        let metadata = self.metadata_from_row(&anchor)?;
        Ok((metadata, anchor))
    }
    /// Retained metadata reader for the write path (D5b).
    ///
    /// It is always drained to DONE before the slot is reset, so it never holds
    /// a read cursor across rows the way the anchor statement does.
    fn read_metadata_drained(&self) -> Result<Metadata> {
        self.with_writer(
            SLOT_METADATA,
            || metadata_sql(self),
            |anchor| {
                if anchor.step()? != Step::Row {
                    return Err(corrupt("DiskANN metadata missing"));
                }
                let metadata = self.metadata_from_row(anchor)?;
                // The singleton count column already proved uniqueness; the
                // drain only returns the statement to a resettable state.
                drain(anchor)?;
                Ok(metadata)
            },
        )
    }
    fn metadata_from_row(&self, anchor: &Statement<'_>) -> Result<Metadata> {
        if nonnegative(anchor, 0, "format version")? != 4 {
            return Err(corrupt("DiskANN format version is unsupported"));
        }
        if nonnegative(anchor, 14, "metadata count")? != 1 {
            return Err(corrupt("DiskANN metadata singleton count"));
        }
        let instance_id = bounded_blob(anchor, 1, 2, 16, true, "instance ID")?
            .try_into()
            .map_err(|_| corrupt("DiskANN instance ID"))?;
        validate_descriptor(
            &bounded_blob(anchor, 3, 4, MAX_DESCRIPTOR_BYTES, false, "descriptor")?,
            &self.descriptor,
        )?;
        let metadata = Metadata {
            instance_id,
            revision: nonnegative(anchor, 5, "revision")?,
            next_entry_id: nonnegative(anchor, 6, "next entry")?,
            next_job_id: nonnegative(anchor, 7, "next job")?,
            next_generation_id: nonnegative(anchor, 8, "next generation")?,
            next_fence: nonnegative(anchor, 15, "next fence")?,
            cleanup_entry_id: nonnegative(anchor, 16, "cleanup entry ID")?,
            canonical_count: nonnegative(anchor, 9, "canonical count")?,
            retired_count: nonnegative(anchor, 10, "retired count")?,
            published_generation: optional_id(anchor, 11, "published generation")?,
            candidate_generation: optional_id(anchor, 12, "candidate generation")?,
            active_job: optional_id(anchor, 13, "active job")?,
        };
        if metadata.next_entry_id == 0
            || metadata.next_job_id == 0
            || metadata.next_generation_id == 0
            || metadata.next_fence == 0
            || metadata.cleanup_entry_id >= metadata.next_entry_id
            || metadata
                .canonical_count
                .checked_add(metadata.retired_count)
                .is_none_or(|n| n >= metadata.next_entry_id)
            || metadata.candidate_generation.is_some() != metadata.active_job.is_some()
            || metadata
                .published_generation
                .is_some_and(|g| g == 0 || g >= metadata.next_generation_id)
            || metadata.candidate_generation.is_some_and(|g| {
                g == 0
                    || g >= metadata.next_generation_id
                    || Some(g) == metadata.published_generation
            })
            || metadata
                .active_job
                .is_some_and(|j| j == 0 || j >= metadata.next_job_id)
        {
            return Err(corrupt(
                "DiskANN metadata counters/identities are inconsistent",
            ));
        }
        Ok(metadata)
    }
    fn validate_generation_identities(&self, m: &Metadata) -> Result<()> {
        for (id, state) in [(m.published_generation, 1), (m.candidate_generation, 0)] {
            if let Some(id) = id {
                let g = self.read_generation(id)?;
                if g.state != state
                    || g.source_revision > m.revision
                    || (state == 0 && Some(g.job_id) != m.active_job)
                {
                    return Err(corrupt("DiskANN generation identity/state mismatch"));
                }
                if state == 1 {
                    if let Some(root) = g.root_local_id {
                        self.read_graph_node(&g, root)?;
                    }
                }
            }
        }
        if let Some(id) = m.active_job {
            let j = self.read_job(id)?;
            if Some(j.generation_id) != m.candidate_generation
                || j.cutoff_revision > m.revision
                || j.cutoff_entry_id >= m.next_entry_id
                || j.fence >= m.next_fence
            {
                return Err(corrupt("DiskANN active job binding/cutoff mismatch"));
            }
        }
        Ok(())
    }
    pub(crate) fn read_generation(&self, id: u64) -> Result<Generation> {
        self.with_writer(
            SLOT_GENERATION,
            || generation_sql(self),
            |s| {
                s.bind_i64(1, checked_id(id)?)?;
                self.generation_row(id, s)
            },
        )
    }
    fn generation_row(&self, id: u64, s: &mut Statement<'static>) -> Result<Generation> {
        if s.step()? != Step::Row {
            return Err(corrupt("DiskANN referenced generation is missing"));
        }
        let g = Generation {
            generation_id: id,
            job_id: nonnegative(s, 0, "generation job")?,
            state: nonnegative(s, 1, "generation state")?,
            source_revision: nonnegative(s, 2, "source revision")?,
            source_hash: if nonnegative(s, 10, "source hash NULL")? == 1 {
                None
            } else {
                Some(
                    bounded_blob(s, 3, 4, 32, true, "source hash")?
                        .try_into()
                        .map_err(|_| corrupt("source hash"))?,
                )
            },
            config_hash: bounded_blob(s, 5, 6, 32, true, "config hash")?
                .try_into()
                .map_err(|_| corrupt("config hash"))?,
            total_count: nonnegative(s, 7, "total count")?,
            eligible_count: nonnegative(s, 8, "eligible count")?,
            root_local_id: optional_id(s, 9, "frozen root")?,
        };
        if g.job_id == 0
            || g.state > 2
            || (g.state == 1 && g.source_hash.is_none())
            || g.total_count > u64::from(u32::MAX - 1)
            || g.eligible_count > g.total_count
            || if g.total_count == 0 {
                g.root_local_id.is_some()
            } else {
                g.root_local_id != Some(g.total_count)
            }
        {
            return Err(corrupt("DiskANN generation root/counts invalid"));
        }
        if s.step()? != Step::Done {
            return Err(corrupt("DiskANN generation not unique"));
        }
        Ok(g)
    }
    /// Point reader only. It never traverses or accepts results; T6 must match
    /// current(public_rowid)->entry_id BEFORE accepting every result version.
    pub(crate) fn read_graph_node(&self, g: &Generation, id: u64) -> Result<GraphNodeRecord> {
        if id > g.total_count || g.total_count == 0 {
            return Err(corrupt("DiskANN local node ID exceeds generation"));
        }
        self.with_writer(
            SLOT_GRAPH_NODE,
            || graph_node_sql(self),
            |s| {
                s.bind_i64(1, checked_id(g.generation_id)?)?;
                s.bind_i64(2, checked_id(id)?)?;
                self.graph_node_row(g, id, s)
            },
        )
    }
    fn graph_node_row(
        &self,
        g: &Generation,
        id: u64,
        s: &mut Statement<'static>,
    ) -> Result<GraphNodeRecord> {
        if s.step()? != Step::Row {
            return Err(corrupt("DiskANN referenced graph node is missing"));
        }
        let entry_id = optional_id(s, 0, "graph entry ID")?;
        let state = nonnegative(s, 1, "graph state")?;
        let frozen = state == 2;
        if (frozen && (g.root_local_id != Some(id) || entry_id.is_some()))
            || (!frozen
                && (state != 0
                    || id >= g.total_count
                    || entry_id.is_none_or(|v| v == 0)
                    || nonnegative(s, 6, "frozen NULL flag")? != 1))
        {
            return Err(corrupt("DiskANN native root/node version role mismatch"));
        }
        let frozen_vector = if frozen {
            let bytes = bounded_blob(s, 2, 3, self.vector_bytes, true, "frozen vector")?;
            with_decoded_vector(&bytes, self.space.dim, &mut Vec::new(), &mut |_| Ok(()))?;
            Some(bytes)
        } else {
            None
        };
        let neighbors = decode_ids(
            &bounded_blob(s, 4, 5, self.adjacency_bytes, false, "neighbors")?,
            self.options.physical_degree(),
            Some(id),
        )?;
        if neighbors.iter().any(|n| *n > g.total_count) {
            return Err(corrupt(
                "DiskANN adjacency exceeds generation local ID range",
            ));
        }
        if s.step()? != Step::Done {
            return Err(corrupt("DiskANN graph node is not unique"));
        }
        Ok(GraphNodeRecord {
            native_local_id: id,
            entry_id,
            frozen_vector,
            neighbors,
        })
    }
    pub(crate) fn read_job(&self, id: u64) -> Result<Job> {
        self.with_writer(
            SLOT_JOB,
            || job_sql(self),
            |s| {
                s.bind_i64(1, checked_id(id)?)?;
                self.job_row(id, s)
            },
        )
    }
    fn job_row(&self, id: u64, s: &mut Statement<'static>) -> Result<Job> {
        if s.step()? != Step::Row {
            return Err(corrupt("DiskANN active job is missing"));
        }
        let j = Job {
            job_id: id,
            generation_id: nonnegative(s, 0, "job generation")?,
            parent_generation: optional_id(s, 1, "job parent")?,
            cutoff_revision: nonnegative(s, 2, "cutoff revision")?,
            cutoff_entry_id: nonnegative(s, 23, "cutoff entry ID")?,
            snapshot_chain: bounded_blob(s, 24, 25, 32, true, "snapshot chain")?
                .try_into()
                .map_err(|_| corrupt("DiskANN snapshot chain"))?,
            fence: nonnegative(s, 3, "job fence")?,
            owner_nonce: if nonnegative(s, 4, "owner NULL")? == 1 {
                None
            } else {
                Some(
                    bounded_blob(s, 5, 6, 16, true, "owner nonce")?
                        .try_into()
                        .map_err(|_| corrupt("owner nonce"))?,
                )
            },
            phase: nonnegative(s, 7, "job phase")?,
            expected_count: nonnegative(s, 8, "expected count")?,
            scan_entry_id: nonnegative(s, 9, "scan cursor")?,
            captured_count: nonnegative(s, 10, "captured count")?,
            import_cursor: nonnegative(s, 11, "import cursor")?,
            validation_cursor: nonnegative(s, 12, "validation cursor")?,
            cleanup_cursor: nonnegative(s, 13, "cleanup cursor")?,
            source_hash: if nonnegative(s, 14, "source NULL")? == 1 {
                None
            } else {
                Some(
                    bounded_blob(s, 15, 16, 32, true, "job source hash")?
                        .try_into()
                        .map_err(|_| corrupt("source hash"))?,
                )
            },
            config_hash: bounded_blob(s, 17, 18, 32, true, "job config hash")?
                .try_into()
                .map_err(|_| corrupt("config hash"))?,
            candidate_hash: if nonnegative(s, 19, "candidate NULL")? == 1 {
                None
            } else {
                Some(
                    bounded_blob(s, 20, 21, 32, true, "candidate hash")?
                        .try_into()
                        .map_err(|_| corrupt("candidate hash"))?,
                )
            },
            failure_code: if s.column_is_null(22)? {
                None
            } else {
                if s.column_type(22)? != ffi::SQLITE_INTEGER as i32 {
                    return Err(corrupt("DiskANN failure code not integer"));
                }
                Some(s.column_i64(22)?)
            },
        };
        if j.job_id == 0
            || j.generation_id == 0
            || j.parent_generation == Some(0)
            || j.fence == 0
            || j.phase > 6
            || j.expected_count > u64::from(u32::MAX - 1)
            || j.expected_count > j.cutoff_entry_id
            || j.scan_entry_id > j.cutoff_entry_id
            || j.captured_count > j.expected_count
            || j.import_cursor > j.expected_count + 1
            || j.validation_cursor > j.expected_count + 1
        {
            return Err(corrupt("DiskANN durable job counters/identity invalid"));
        }
        if s.step()? != Step::Done {
            return Err(corrupt("DiskANN job not unique"));
        }
        Ok(j)
    }
    pub(crate) fn read_entry(&self, id: u64) -> Result<EntryRecord> {
        let mut s = self.connection.prepare(&format!(
            "SELECT public_rowid,created_revision,retired_revision,{} FROM {} WHERE entry_id=?1",
            blob_projection("vector", self.vector_bytes, true),
            self.names.entries
        ))?;
        s.bind_i64(1, checked_id(id)?)?;
        if s.step()? != Step::Row {
            return Err(corrupt("DiskANN immutable entry missing"));
        }
        let e = EntryRecord {
            entry_id: id,
            public_rowid: nonnegative(&s, 0, "original rowid")?,
            created_revision: nonnegative(&s, 1, "created revision")?,
            retired_revision: optional_id(&s, 2, "retired revision")?,
            vector: bounded_blob(&s, 3, 4, self.vector_bytes, true, "vector")?,
        };
        if e.entry_id == 0
            || e.created_revision == 0
            || e.retired_revision.is_some_and(|r| r <= e.created_revision)
        {
            return Err(corrupt("DiskANN immutable entry interval invalid"));
        }
        // Checks finite values without normalizing already-canonical bytes again.
        with_decoded_vector(&e.vector, self.space.dim, &mut Vec::new(), &mut |_| Ok(()))?;
        if s.step()? != Step::Done {
            return Err(corrupt("DiskANN entry not unique"));
        }
        s.finish()?;
        Ok(e)
    }
    fn lookup_live_id(&self, rowid: u64) -> Result<Option<u64>> {
        if rowid > MAX_ID {
            return Ok(None);
        }
        self.with_writer(
            SLOT_LIVE_ID,
            || live_id_sql(self),
            |s| {
                s.bind_i64(1, checked_id(rowid)?)?;
                if s.step()? == Step::Done {
                    return Ok(None);
                }
                let id = nonnegative(s, 0, "current entry ID")?;
                if id == 0
                    || nonnegative(s, 1, "original public rowid")? != rowid
                    || nonnegative(s, 2, "created revision")? == 0
                    || !s.column_is_null(3)?
                {
                    return Err(corrupt(
                        "DiskANN current mapping references invalid version",
                    ));
                }
                if s.step()? != Step::Done {
                    return Err(corrupt("DiskANN current mapping not unique"));
                }
                Ok(Some(id))
            },
        )
    }
    pub(crate) fn contains(&self, rowid: u64) -> Result<bool> {
        Ok(self.lookup_live_id(rowid)?.is_some())
    }
    fn live_blob(&self, rowid: u64) -> Result<Option<Vec<u8>>> {
        self.lookup_live_id(rowid)?
            .map(|id| {
                self.read_entry(id).and_then(|e| {
                    if e.public_rowid == rowid {
                        Ok(e.vector)
                    } else {
                        Err(corrupt(
                            "DiskANN projected immutable version has a different public rowid",
                        ))
                    }
                })
            })
            .transpose()
    }
    fn encode_input(&self, vector: &[f32]) -> Result<Vec<u8>> {
        if vector.len() != self.space.dim {
            return Err(IndexError::new("DiskANN vector dimension mismatch"));
        }
        let mut owned = Vec::new();
        reserve(&mut owned, vector.len())?;
        owned.extend_from_slice(vector);
        normalize_input(&mut owned, self.space.dim, self.space.distance_type)?;
        encode_vector(&owned)
    }
    fn expect_one_change(&self, operation: &str) -> Result<()> {
        if self.connection.changes() != 1 {
            return Err(corrupt(format!(
                "DiskANN {operation} did not affect exactly one row"
            )));
        }
        Ok(())
    }
    fn write_metadata(&self, m: &Metadata) -> Result<()> {
        self.with_writer(
            SLOT_UPDATE_METADATA,
            || {
                format!(
                    "UPDATE {} SET revision=?1,next_entry_id=?2,canonical_count=?3,retired_count=?4 WHERE singleton=1",
                    self.names.meta
                )
            },
            |s| {
                for (slot, value) in [
                    m.revision,
                    m.next_entry_id,
                    m.canonical_count,
                    m.retired_count,
                ]
                .into_iter()
                .enumerate()
                {
                    s.bind_i64(slot + 1, checked_id(value)?)?;
                }
                drain(s)
            },
        )?;
        self.expect_one_change("canonical metadata")
    }
    /// Inserts one immutable version plus its current mapping.
    ///
    /// A caller that already ran this table's live-id probe for the same rowid in
    /// this callback passes its proof, so exactly one probe precedes the write.
    fn insert_encoded(
        &self,
        rowid: u64,
        bytes: &[u8],
        m: &mut Metadata,
        probed: Option<LiveIdProbe>,
    ) -> Result<()> {
        checked_id(rowid)?;
        if probed.is_none() && self.contains(rowid)? {
            return Err(IndexError::with_code(
                ffi::SQLITE_CONSTRAINT as i32,
                "DiskANN rowid already exists",
            ));
        }
        let id = m.next_entry_id;
        m.next_entry_id = checked_increment(id, "entry ID")?;
        m.canonical_count = checked_increment(m.canonical_count, "canonical count")?;
        self.with_writer(
            SLOT_INSERT_ENTRY,
            || {
                format!(
                    "INSERT INTO {} VALUES(?1,?2,?3,NULL,?4)",
                    self.names.entries
                )
            },
            |s| {
                s.bind_i64(1, checked_id(id)?)?;
                s.bind_i64(2, checked_id(rowid)?)?;
                s.bind_i64(3, checked_id(m.revision)?)?;
                s.bind_blob(4, bytes)?;
                drain(s)
            },
        )?;
        self.expect_one_change("immutable entry insertion")?;
        self.with_writer(
            SLOT_INSERT_CURRENT,
            || format!("INSERT INTO {} VALUES(?1,?2)", self.names.current),
            |s| {
                s.bind_i64(1, checked_id(rowid)?)?;
                s.bind_i64(2, checked_id(id)?)?;
                drain(s)
            },
        )?;
        self.expect_one_change("current insertion")
    }
    /// Inserts one row. The frontend's live-id probe is the single probe for this
    /// row: it runs in the same callback, before any write, and reports the rowid
    /// conflict with its own message and SQLITE_CONSTRAINT code.
    pub(crate) fn insert(&self, rowid: u64, vector: &[f32], probed: LiveIdProbe) -> Result<()> {
        self.check_workspace(0)?;
        let bytes = self.encode_input(vector)?;
        let mut m = self.read_metadata_drained()?;
        m.revision = checked_increment(m.revision, "revision")?;
        self.insert_encoded(rowid, &bytes, &mut m, Some(probed))?;
        self.write_metadata(&m)
    }
    /// The tagged descriptor is borrowed only during this callback. Each bounded
    /// chunk is owned; all chunks share the caller's ONE ordinary journal carrier.
    pub(crate) fn insert_batch(&self, batch: &BatchView<'_>, batch_size: usize) -> Result<()> {
        if !(1..=MAX_INSERT_BATCH).contains(&batch_size) || batch.dimension() != self.space.dim {
            return Err(IndexError::new("invalid DiskANN batch size or dimension"));
        }
        if batch.len() == 0 {
            return Ok(());
        }
        let chunk_bytes = checked_product(batch_size, self.vector_bytes, "batch payload")?
            .checked_add(checked_product(
                batch_size,
                size_of::<(u64, Vec<u8>)>(),
                "batch rows",
            )?)
            .ok_or_else(|| too_big("DiskANN batch workspace overflow"))?;
        self.check_workspace(chunk_bytes)?;
        // The chunk reserve is the budget in force for the whole batch, so the
        // retained set is re-bounded (or evicted) against it once here.
        self.arm_writer(chunk_bytes)?;
        let mut m = self.read_metadata_drained()?;
        let count =
            u64::try_from(batch.len()).map_err(|_| too_big("DiskANN batch population overflow"))?;
        for (base, name) in [
            (m.next_entry_id, "entry IDs"),
            (m.canonical_count, "canonical count"),
        ] {
            if base.checked_add(count).is_none_or(|n| n > MAX_ID) {
                return Err(too_big(format!("DiskANN batch {name} exhausted")));
            }
        }
        m.revision = checked_increment(m.revision, "revision")?;
        for start in (0..batch.len()).step_by(batch_size) {
            let end = start + batch_size.min(batch.len() - start);
            let mut chunk = Vec::new();
            reserve(&mut chunk, end - start)?;
            for index in start..end {
                let rowid = batch.rowid(index)?;
                checked_id(rowid)?;
                let bytes = self.encode_input(batch.vector(index))?;
                chunk.push((rowid, bytes));
            }
            for (rowid, bytes) in chunk {
                self.insert_encoded(rowid, &bytes, &mut m, None)?;
            }
        }
        self.write_metadata(&m)
    }
    /// Only two reverse-map point lookups are examined, regardless of historical
    /// generations. Import adds eligibility iff current still matches; retirement
    /// removes it iff that version has already been imported. No scan at publish.
    fn retire(&self, rowid: u64, id: u64, m: &mut Metadata) -> Result<()> {
        m.canonical_count = m
            .canonical_count
            .checked_sub(1)
            .ok_or_else(|| corrupt("DiskANN canonical count underflow"))?;
        m.retired_count = checked_increment(m.retired_count, "retired count")?;
        for generation in [m.published_generation, m.candidate_generation]
            .into_iter()
            .flatten()
        {
            let exists = self.with_writer(
                SLOT_RETIRE_NODE,
                || {
                    format!(
                        "SELECT native_local_id,state FROM {} WHERE generation_id=?1 AND entry_id=?2",
                        self.names.graph_nodes
                    )
                },
                |lookup| {
                    lookup.bind_i64(1, checked_id(generation)?)?;
                    lookup.bind_i64(2, checked_id(id)?)?;
                    let exists = lookup.step()? == Step::Row;
                    if exists {
                        nonnegative(lookup, 0, "native local ID")?;
                        if nonnegative(lookup, 1, "graph node state")? != 0
                            || lookup.step()? != Step::Done
                        {
                            return Err(corrupt("DiskANN reverse version mapping invalid"));
                        }
                    }
                    Ok(exists)
                },
            )?;
            if exists {
                let g = self.read_generation(generation)?;
                let count = g
                    .eligible_count
                    .checked_sub(1)
                    .ok_or_else(|| corrupt("DiskANN eligible count underflow"))?;
                self.with_writer(
                    SLOT_RETIRE_ELIGIBLE,
                    || {
                        format!(
                            "UPDATE {} SET eligible_count=?1 WHERE generation_id=?2",
                            self.names.generations
                        )
                    },
                    |s| {
                        s.bind_i64(1, checked_id(count)?)?;
                        s.bind_i64(2, checked_id(generation)?)?;
                        drain(s)
                    },
                )?;
                self.expect_one_change("generation eligibility")?;
            }
        }
        self.with_writer(
            SLOT_RETIRE_ENTRY,
            || {
                format!(
                    "UPDATE {} SET retired_revision=?1 WHERE entry_id=?2 AND public_rowid=?3 AND retired_revision IS NULL",
                    self.names.entries
                )
            },
            |s| {
                s.bind_i64(1, checked_id(m.revision)?)?;
                s.bind_i64(2, checked_id(id)?)?;
                s.bind_i64(3, checked_id(rowid)?)?;
                drain(s)
            },
        )?;
        self.expect_one_change("immutable version retirement")?;
        self.with_writer(
            SLOT_RETIRE_CURRENT,
            || {
                format!(
                    "DELETE FROM {} WHERE public_rowid=?1 AND entry_id=?2",
                    self.names.current
                )
            },
            |s| {
                s.bind_i64(1, checked_id(rowid)?)?;
                s.bind_i64(2, checked_id(id)?)?;
                drain(s)
            },
        )?;
        self.expect_one_change("current retirement")
    }
    pub(crate) fn update(&self, old_rowid: u64, new_rowid: u64, vector: &[f32]) -> Result<()> {
        checked_id(old_rowid)?;
        checked_id(new_rowid)?;
        if old_rowid != new_rowid {
            return Err(IndexError::new("DiskANN rowid cannot be changed by UPDATE"));
        }
        let bytes = self.encode_input(vector)?;
        let mut m = self.read_metadata_drained()?;
        m.revision = checked_increment(m.revision, "revision")?;
        checked_increment(m.next_entry_id, "entry ID")?;
        checked_increment(m.retired_count, "retired count")?;
        let id = self
            .lookup_live_id(old_rowid)?
            .ok_or_else(|| IndexError::new("DiskANN update rowid is absent"))?;
        self.retire(old_rowid, id, &mut m)?;
        self.insert_encoded(new_rowid, &bytes, &mut m, None)?;
        self.write_metadata(&m)
    }
    pub(crate) fn mark_delete(&self, rowid: u64) -> Result<()> {
        checked_id(rowid)?;
        let mut m = self.read_metadata_drained()?;
        m.revision = checked_increment(m.revision, "revision")?;
        let id = self
            .lookup_live_id(rowid)?
            .ok_or_else(|| IndexError::new("DiskANN rowid is absent"))?;
        self.retire(rowid, id, &mut m)?;
        self.write_metadata(&m)
    }
    pub(crate) fn knn(
        &self,
        query: &[f32],
        k: usize,
        override_l: Option<usize>,
        filter: SearchFilter<'_>,
        project_vector: bool,
    ) -> Result<QueryRows> {
        // D7: the query operation owns cache_bytes for its own retained readers,
        // so the write set releases its share first.
        self.release_writer();
        crate::diskann_query::knn(self, query, k, override_l, filter, project_vector)
    }
    fn result_budget(&self, count: usize, project: bool, filter: &SearchFilter<'_>) -> Result<()> {
        let per = size_of::<SearchResult>()
            .checked_add(if project {
                self.vector_bytes + size_of::<Vec<u8>>()
            } else {
                0
            })
            .ok_or_else(|| too_big("DiskANN result workspace overflow"))?;
        let filter_bytes = if let SearchFilter::In(ids) = filter {
            checked_product(ids.capacity(), 64, "filter")?
        } else {
            0
        };
        let extra = checked_product(count, per, "result")?
            .checked_add(filter_bytes)
            .ok_or_else(|| too_big("DiskANN query workspace overflow"))?;
        self.check_workspace(extra)
    }
    pub(crate) fn select_rowids(
        &self,
        filter: SearchFilter<'_>,
        project_vector: bool,
    ) -> Result<QueryRows> {
        // D7: an exact scan reads per use; the write set must not hold budget
        // while it runs.
        self.release_writer();
        let (m, anchor) = self.read_metadata()?;
        let count = match &filter {
            SearchFilter::None => {
                return Err(IndexError::new(
                    "DiskANN requires a KNN or rowid constraint",
                ))
            }
            SearchFilter::Equals(_) => 1,
            SearchFilter::In(ids) => ids.len(),
        }
        .min(
            usize::try_from(m.canonical_count)
                .map_err(|_| too_big("DiskANN canonical population overflow"))?,
        );
        self.result_budget(count, project_vector, &filter)?;
        let mut results = Vec::new();
        reserve(&mut results, count)?;
        let mut vectors = project_vector.then(Vec::new);
        if let Some(v) = &mut vectors {
            reserve(v, count)?;
        }
        let mut visits = 0usize;
        let mut select = |rowid| -> Result<()> {
            visits = visits
                .checked_add(1)
                .ok_or_else(|| too_big("DiskANN visit overflow"))?;
            if visits > self.options.max_visits {
                return Err(too_big("DiskANN operation exceeded max_visits"));
            }
            let blob = if project_vector {
                self.live_blob(rowid)?
            } else {
                None
            };
            let exists = if project_vector {
                blob.is_some()
            } else {
                self.contains(rowid)?
            };
            if exists {
                if results.len() >= count {
                    return Err(corrupt(
                        "DiskANN canonical count smaller than selected population",
                    ));
                }
                results.push(SearchResult::new(0.0, rowid));
                if let Some(v) = &mut vectors {
                    v.push(blob.ok_or_else(|| corrupt("DiskANN projection missing"))?);
                }
            }
            Ok(())
        };
        match filter {
            SearchFilter::Equals(id) => select(id)?,
            SearchFilter::In(ids) => {
                for &id in ids {
                    select(id)?;
                }
            }
            SearchFilter::None => unreachable!(),
        };
        Ok(QueryRows {
            results,
            vectors,
            anchor,
        })
    }
    pub(crate) fn destroy(&self) -> Result<()> {
        self.require_rollback_journal()?;
        self.validate_schema()?;
        let (m, a) = self.read_metadata()?;
        a.finish()?;
        if m.active_job.is_some() {
            return Err(IndexError::with_code(
                ffi::SQLITE_BUSY as i32,
                "DiskANN active job must be cancelled before DROP",
            ));
        }
        for l in schema::LAYOUTS.iter().rev() {
            self.connection
                .execute(&format!("DROP TABLE {}", self.names.qualified(l.suffix)?))?;
        }
        Ok(())
    }
    /// Reject active ownership rather than silently invalidating an in-flight job.
    /// Cancel first, rename, then prepare/restart with a fresh table-bound attempt.
    pub(crate) fn rename(&self, new_name: &str) -> Result<()> {
        self.require_rollback_journal()?;
        self.validate_schema()?;
        let (m, a) = self.read_metadata()?;
        a.finish()?;
        if m.active_job.is_some() {
            return Err(IndexError::with_code(
                ffi::SQLITE_BUSY as i32,
                "DiskANN active job must be cancelled before RENAME",
            ));
        }
        quote_identifier(&self.names.schema)?;
        for l in schema::LAYOUTS {
            self.connection.execute(&format!(
                "ALTER TABLE {} RENAME TO {}",
                self.names.qualified(l.suffix)?,
                quote_identifier(&format!("{new_name}_{}", l.suffix))?
            ))?;
        }
        Ok(())
    }
    /// Runs the complete xUpdate action inside an ordinary SQLite statement's
    /// rollback boundary, without opening or controlling any transaction here.
    pub(crate) fn atomic_write<T>(&self, action: impl FnOnce() -> Result<T>) -> Result<T> {
        crate::atomic_callback::with_action(
            self.connection.raw_handle(),
            || {
                let value = action()?;
                // Graph-table triggers could damage the carrier during action.
                // Reject that damage while the scalar is still executing, so
                // SQLite rolls it and every prior nested write back together.
                self.verify_writable_guard()?;
                Ok(value)
            },
            || self.verify_writable_guard(),
            |invocation| {
                let mut statement = self.connection.prepare(&format!(
                    "UPDATE OR ABORT {} SET value=CASE singleton WHEN 1 THEN {}(?1,?2) ELSE 0 END",
                    self.names.txn,
                    crate::atomic_callback::FUNCTION_NAME
                ))?;
                // SAFETY: with_action leases this private invocation frame until
                // execute returns. The static NUL-terminated tag lives forever;
                // done consumes/finalizes the statement before the frame lease
                // ends, and SQLite receives no destructor/ownership transfer.
                unsafe {
                    statement.bind_pointer(
                        1,
                        invocation.pointer,
                        crate::atomic_callback::POINTER_TAG.as_ptr().cast(),
                    )?;
                }
                statement.bind_i64(2, invocation.token)?;
                done(statement)?;
                // Pre/post scalar validation makes two rows invariant; this is
                // diagnostic only, not the action's rollback/error boundary.
                if self.connection.changes() != 2 {
                    return Err(corrupt(
                        "DiskANN atomic carrier did not update exactly two rows",
                    ));
                }
                Ok(())
            },
        )
    }

    /// Reads the single pragma_table_list row describing one shadow table.
    ///
    /// The per-caller predicates stay with the caller; this reproduces the
    /// shared "exactly one row of integer counters" read, including the caller's
    /// own corruption message for a missing or non-unique row.
    fn pragma_table_row(
        &self,
        table: &str,
        names: [&str; 3],
        message: &'static str,
    ) -> Result<(String, u64, u64, u64)> {
        self.with_writer(
            SLOT_CATALOG_KIND,
            || {
                String::from(
                    "SELECT type,ncol,wr,strict FROM pragma_table_list(?1) WHERE schema=?2 COLLATE NOCASE",
                )
            },
            |kind| {
                kind.bind_text(1, table)?;
                kind.bind_text(2, &self.names.schema)?;
                if kind.step()? != Step::Row {
                    return Err(corrupt(message));
                }
                let name = kind.column_text(0)?;
                let columns = nonnegative(kind, 1, names[0])?;
                let without_rowid = nonnegative(kind, 2, names[1])?;
                let strict = nonnegative(kind, 3, names[2])?;
                if kind.step()? != Step::Done {
                    return Err(corrupt(message));
                }
                Ok((name, columns, without_rowid, strict))
            },
        )
    }
    fn validate_columns(
        &self,
        table: &str,
        expected: &[(&str, &str, u64, u64)],
        slot: usize,
    ) -> Result<()> {
        self.with_writer(
            slot,
            || {
                String::from(
                    "SELECT cid,length(name),name,length(type),type,\"notnull\",pk,hidden FROM pragma_table_xinfo(?1,?2) ORDER BY cid",
                )
            },
            |columns| {
                columns.bind_text(1, table)?;
                columns.bind_text(2, &self.names.schema)?;
                for (index, &(name, kind, not_null, primary_key)) in expected.iter().enumerate() {
                    if columns.step()? != Step::Row
                        || nonnegative(columns, 0, "column position")? != index as u64
                        || nonnegative(columns, 1, "column name length")? != name.len() as u64
                        || nonnegative(columns, 3, "column type length")? != kind.len() as u64
                        || columns.column_text(2)? != name
                        || !columns.column_text(4)?.eq_ignore_ascii_case(kind)
                        || nonnegative(columns, 5, "not null flag")? != not_null
                        || nonnegative(columns, 6, "primary key flag")? != primary_key
                        || nonnegative(columns, 7, "hidden column flag")? != 0
                    {
                        return Err(corrupt("DiskANN shadow-table schema is incompatible"));
                    }
                }
                if columns.step()? != Step::Done {
                    return Err(corrupt(
                        "DiskANN shadow-table schema has additional columns",
                    ));
                }
                Ok(())
            },
        )
    }

    fn require_rollback_journal(&self) -> Result<()> {
        // The schema identifier is validated on every call, even when the
        // retained statement is reused, so no check depends on retention.
        let sql = format!(
            "PRAGMA {}.journal_mode",
            quote_identifier(&self.names.schema)?
        );
        let name = self.with_writer(
            SLOT_JOURNAL_MODE,
            || sql,
            |mode| {
                if mode.step()? != Step::Row {
                    return Err(corrupt("DiskANN could not read SQLite journal_mode"));
                }
                let name = mode.column_text(0)?;
                if mode.step()? != Step::Done {
                    return Err(corrupt("DiskANN journal_mode returned unexpected rows"));
                }
                Ok(name)
            },
        )?;
        validate_journal_mode(&name)
    }
    fn verify_writable_guard(&self) -> Result<()> {
        // The per-row write path is the only caller, so retention is armed here
        // and never by open, control, maintenance or query reads.
        self.arm_writer(0)?;
        // OFF disables SQLite rollback itself, including nested carrier writes.
        // Refuse it before action; never silently change the host's settings.
        self.require_rollback_journal()?;
        self.check_format()?;
        self.validate_schema()?;
        let m = self.read_metadata_drained()?;
        self.validate_generation_identities(&m)?;
        self.verify_guard()
    }

    fn verify_guard(&self) -> Result<()> {
        // A full-table UPDATE over these two immutable rows forces SQLite's
        // carrier to journal every nested graph write. There are intentionally
        // no triggers: ENABLE_TRIGGER=off, FAIL, and IGNORE cannot remove or
        // weaken the statement's rollback boundary.
        let table = self.names.base("diskann_txn");
        let (kind, columns, without_rowid, strict) = self.pragma_table_row(
            &table,
            [
                "atomic guard column count",
                "atomic guard WITHOUT ROWID flag",
                "atomic guard STRICT flag",
            ],
            "DiskANN atomic guard must be an ordinary rowid shadow table",
        )?;
        if !matches!(kind.as_str(), "table" | "shadow")
            || columns != 2
            || without_rowid != 0
            || strict != 0
        {
            return Err(corrupt(
                "DiskANN atomic guard must be an ordinary rowid shadow table",
            ));
        }
        self.validate_columns(
            &table,
            &[("singleton", "INTEGER", 0, 1), ("value", "INTEGER", 1, 0)],
            SLOT_CATALOG_COLUMNS,
        )?;
        self.guard_relations(SLOT_GUARD_INDEX_LIST, "pragma_index_list")?;
        self.guard_relations(SLOT_GUARD_FOREIGN_KEYS, "pragma_foreign_key_list")?;
        self.with_writer(
            SLOT_GUARD_ROWS,
            || {
                format!(
                    "SELECT singleton,value FROM {} ORDER BY singleton LIMIT 3",
                    self.names.txn
                )
            },
            |rows| {
                for singleton in [1, 2] {
                    if rows.step()? != Step::Row
                        || nonnegative(rows, 0, "atomic guard singleton")? != singleton
                        || nonnegative(rows, 1, "atomic guard value")? != 0
                    {
                        return Err(corrupt(
                            "DiskANN atomic guard must contain its two unchanged rows",
                        ));
                    }
                }
                if rows.step()? != Step::Done {
                    return Err(corrupt("DiskANN atomic guard contains unexpected rows"));
                }
                Ok(())
            },
        )?;

        // TEMP triggers can target a non-TEMP table. Conservatively reject the
        // matching base name in both schemas before EVERY carrier preparation.
        self.reject_guard_triggers(&self.names.schema, SLOT_GUARD_TRIGGER)?;
        if !self.names.schema.eq_ignore_ascii_case("temp") {
            self.reject_guard_triggers("temp", SLOT_GUARD_TEMP_TRIGGER)?;
        }
        Ok(())
    }

    fn guard_relations(&self, slot: usize, pragma: &'static str) -> Result<()> {
        self.with_writer(
            slot,
            || format!("SELECT 1 FROM {pragma}(?1,?2) LIMIT 1"),
            |relations| {
                relations.bind_text(1, &self.names.base("diskann_txn"))?;
                relations.bind_text(2, &self.names.schema)?;
                if relations.step()? != Step::Done {
                    return Err(corrupt(
                        "DiskANN atomic guard cannot have indices or foreign keys",
                    ));
                }
                Ok(())
            },
        )
    }

    fn reject_guard_triggers(&self, schema: &str, slot: usize) -> Result<()> {
        let sql = qualified_name(schema, "sqlite_schema")?;
        self.with_writer(
            slot,
            || {
                format!(
                    "SELECT 1 FROM {sql} WHERE type='trigger' AND tbl_name=?1 COLLATE NOCASE LIMIT 1"
                )
            },
            |trigger| {
                trigger.bind_text(1, &self.names.base("diskann_txn"))?;
                if trigger.step()? != Step::Done {
                    return Err(corrupt(
                        "DiskANN atomic guard cannot have schema or TEMP triggers",
                    ));
                }
                Ok(())
            },
        )
    }
}
