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
use std::mem::size_of;
type Result<T> = std::result::Result<T, IndexError>;
const FORMAT_VERSION: i64 = 4;
const MAX_DESCRIPTOR_BYTES: usize = 16 * 1024;
const MAX_ID: u64 = i64::MAX as u64;
const MAX_INSERT_BATCH: usize = 32;
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
fn done(mut statement: Statement<'_>) -> Result<()> {
    while statement.step()? == Step::Row {}
    statement.finish()
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
    fn check_workspace(&self, extra: usize) -> Result<()> {
        let names = schema::LAYOUTS.iter().try_fold(0usize, |sum, l| {
            sum.checked_add(self.names.base(l.suffix).len())
                .ok_or_else(|| too_big("DiskANN SQL identifier workspace overflow"))
        })?;
        let minimum = checked_product(self.vector_bytes, 3, "canonical buffers")?
            .checked_add(checked_product(names, 16, "SQL names")?)
            .and_then(|v| v.checked_add(24 * 1024))
            .and_then(|v| v.checked_add(extra))
            .ok_or_else(|| too_big("DiskANN canonical workspace overflow"))?;
        if minimum > self.options.cache_bytes {
            return Err(too_big("DiskANN canonical workspace exceeds cache_bytes"));
        }
        Ok(())
    }
    fn check_format(&self) -> Result<()> {
        let mut s = self.connection.prepare(&format!(
            "SELECT format_version FROM {} WHERE singleton=1",
            self.names.meta
        ))?;
        if s.step()? != Step::Row {
            return Err(corrupt("DiskANN singleton metadata is missing"));
        }
        if nonnegative(&s, 0, "format version")? != FORMAT_VERSION as u64 {
            return Err(corrupt(
                "DiskANN format version is unsupported; Format4 requires recreation (no migration)",
            ));
        }
        if s.step()? != Step::Done {
            return Err(corrupt("DiskANN singleton metadata is not unique"));
        }
        s.finish()
    }
    pub(crate) fn read_metadata(&self) -> Result<(Metadata, Statement<'static>)> {
        let mut anchor=self.connection.prepare(&format!("SELECT format_version,{},{},revision,next_entry_id,next_job_id,next_generation_id,canonical_count,retired_count,published_generation,candidate_generation,active_job,(SELECT count(*) FROM (SELECT singleton FROM {} LIMIT 2)),next_fence,cleanup_entry_id FROM {} WHERE singleton=1",blob_projection("instance_id",16,true),blob_projection("descriptor",MAX_DESCRIPTOR_BYTES,false),self.names.meta,self.names.meta))?;
        if anchor.step()? != Step::Row {
            return Err(corrupt("DiskANN metadata missing"));
        }
        if nonnegative(&anchor, 0, "format version")? != 4 {
            return Err(corrupt("DiskANN format version is unsupported"));
        }
        if nonnegative(&anchor, 14, "metadata count")? != 1 {
            return Err(corrupt("DiskANN metadata singleton count"));
        }
        let instance_id = bounded_blob(&anchor, 1, 2, 16, true, "instance ID")?
            .try_into()
            .map_err(|_| corrupt("DiskANN instance ID"))?;
        validate_descriptor(
            &bounded_blob(&anchor, 3, 4, MAX_DESCRIPTOR_BYTES, false, "descriptor")?,
            &self.descriptor,
        )?;
        let metadata = Metadata {
            instance_id,
            revision: nonnegative(&anchor, 5, "revision")?,
            next_entry_id: nonnegative(&anchor, 6, "next entry")?,
            next_job_id: nonnegative(&anchor, 7, "next job")?,
            next_generation_id: nonnegative(&anchor, 8, "next generation")?,
            next_fence: nonnegative(&anchor, 15, "next fence")?,
            cleanup_entry_id: nonnegative(&anchor, 16, "cleanup entry ID")?,
            canonical_count: nonnegative(&anchor, 9, "canonical count")?,
            retired_count: nonnegative(&anchor, 10, "retired count")?,
            published_generation: optional_id(&anchor, 11, "published generation")?,
            candidate_generation: optional_id(&anchor, 12, "candidate generation")?,
            active_job: optional_id(&anchor, 13, "active job")?,
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
        Ok((metadata, anchor))
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
        let mut s=self.connection.prepare(&format!("SELECT job_id,state,source_revision,{},{},total_count,eligible_count,root_local_id,source_hash IS NULL FROM {} WHERE generation_id=?1",blob_projection("source_hash",32,true),blob_projection("config_hash",32,true),self.names.generations))?;
        s.bind_i64(1, checked_id(id)?)?;
        if s.step()? != Step::Row {
            return Err(corrupt("DiskANN referenced generation is missing"));
        }
        let g = Generation {
            generation_id: id,
            job_id: nonnegative(&s, 0, "generation job")?,
            state: nonnegative(&s, 1, "generation state")?,
            source_revision: nonnegative(&s, 2, "source revision")?,
            source_hash: if nonnegative(&s, 10, "source hash NULL")? == 1 {
                None
            } else {
                Some(
                    bounded_blob(&s, 3, 4, 32, true, "source hash")?
                        .try_into()
                        .map_err(|_| corrupt("source hash"))?,
                )
            },
            config_hash: bounded_blob(&s, 5, 6, 32, true, "config hash")?
                .try_into()
                .map_err(|_| corrupt("config hash"))?,
            total_count: nonnegative(&s, 7, "total count")?,
            eligible_count: nonnegative(&s, 8, "eligible count")?,
            root_local_id: optional_id(&s, 9, "frozen root")?,
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
        s.finish()?;
        Ok(g)
    }
    /// Point reader only. It never traverses or accepts results; T6 must match
    /// current(public_rowid)->entry_id BEFORE accepting every result version.
    pub(crate) fn read_graph_node(&self, g: &Generation, id: u64) -> Result<GraphNodeRecord> {
        if id > g.total_count || g.total_count == 0 {
            return Err(corrupt("DiskANN local node ID exceeds generation"));
        }
        let mut s=self.connection.prepare(&format!("SELECT entry_id,state,{},{},frozen_vector IS NULL FROM {} WHERE generation_id=?1 AND native_local_id=?2",blob_projection("frozen_vector",self.vector_bytes,true),blob_projection("neighbors",self.adjacency_bytes,false),self.names.graph_nodes))?;
        s.bind_i64(1, checked_id(g.generation_id)?)?;
        s.bind_i64(2, checked_id(id)?)?;
        if s.step()? != Step::Row {
            return Err(corrupt("DiskANN referenced graph node is missing"));
        }
        let entry_id = optional_id(&s, 0, "graph entry ID")?;
        let state = nonnegative(&s, 1, "graph state")?;
        let frozen = state == 2;
        if (frozen && (g.root_local_id != Some(id) || entry_id.is_some()))
            || (!frozen
                && (state != 0
                    || id >= g.total_count
                    || entry_id.is_none_or(|v| v == 0)
                    || nonnegative(&s, 6, "frozen NULL flag")? != 1))
        {
            return Err(corrupt("DiskANN native root/node version role mismatch"));
        }
        let frozen_vector = if frozen {
            let bytes = bounded_blob(&s, 2, 3, self.vector_bytes, true, "frozen vector")?;
            with_decoded_vector(&bytes, self.space.dim, &mut Vec::new(), &mut |_| Ok(()))?;
            Some(bytes)
        } else {
            None
        };
        let neighbors = decode_ids(
            &bounded_blob(&s, 4, 5, self.adjacency_bytes, false, "neighbors")?,
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
        s.finish()?;
        Ok(GraphNodeRecord {
            native_local_id: id,
            entry_id,
            frozen_vector,
            neighbors,
        })
    }
    pub(crate) fn read_job(&self, id: u64) -> Result<Job> {
        let mut s=self.connection.prepare(&format!("SELECT generation_id,parent_generation,cutoff_revision,fence,owner_nonce IS NULL,{},phase,expected_count,scan_entry_id,captured_count,import_cursor,validation_cursor,cleanup_cursor,source_hash IS NULL,{}, {},candidate_hash IS NULL,{},failure_code,cutoff_entry_id,{} FROM {} WHERE job_id=?1",blob_projection("owner_nonce",16,true),blob_projection("source_hash",32,true),blob_projection("config_hash",32,true),blob_projection("candidate_hash",32,true),blob_projection("snapshot_chain",32,true),self.names.jobs))?;
        s.bind_i64(1, checked_id(id)?)?;
        if s.step()? != Step::Row {
            return Err(corrupt("DiskANN active job is missing"));
        }
        let j = Job {
            job_id: id,
            generation_id: nonnegative(&s, 0, "job generation")?,
            parent_generation: optional_id(&s, 1, "job parent")?,
            cutoff_revision: nonnegative(&s, 2, "cutoff revision")?,
            cutoff_entry_id: nonnegative(&s, 23, "cutoff entry ID")?,
            snapshot_chain: bounded_blob(&s, 24, 25, 32, true, "snapshot chain")?
                .try_into()
                .map_err(|_| corrupt("DiskANN snapshot chain"))?,
            fence: nonnegative(&s, 3, "job fence")?,
            owner_nonce: if nonnegative(&s, 4, "owner NULL")? == 1 {
                None
            } else {
                Some(
                    bounded_blob(&s, 5, 6, 16, true, "owner nonce")?
                        .try_into()
                        .map_err(|_| corrupt("owner nonce"))?,
                )
            },
            phase: nonnegative(&s, 7, "job phase")?,
            expected_count: nonnegative(&s, 8, "expected count")?,
            scan_entry_id: nonnegative(&s, 9, "scan cursor")?,
            captured_count: nonnegative(&s, 10, "captured count")?,
            import_cursor: nonnegative(&s, 11, "import cursor")?,
            validation_cursor: nonnegative(&s, 12, "validation cursor")?,
            cleanup_cursor: nonnegative(&s, 13, "cleanup cursor")?,
            source_hash: if nonnegative(&s, 14, "source NULL")? == 1 {
                None
            } else {
                Some(
                    bounded_blob(&s, 15, 16, 32, true, "job source hash")?
                        .try_into()
                        .map_err(|_| corrupt("source hash"))?,
                )
            },
            config_hash: bounded_blob(&s, 17, 18, 32, true, "job config hash")?
                .try_into()
                .map_err(|_| corrupt("config hash"))?,
            candidate_hash: if nonnegative(&s, 19, "candidate NULL")? == 1 {
                None
            } else {
                Some(
                    bounded_blob(&s, 20, 21, 32, true, "candidate hash")?
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
        s.finish()?;
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
        let mut s=self.connection.prepare(&format!("SELECT c.entry_id,e.public_rowid,e.created_revision,e.retired_revision FROM {} c LEFT JOIN {} e ON e.entry_id=c.entry_id WHERE c.public_rowid=?1",self.names.current,self.names.entries))?;
        s.bind_i64(1, checked_id(rowid)?)?;
        if s.step()? == Step::Done {
            s.finish()?;
            return Ok(None);
        }
        let id = nonnegative(&s, 0, "current entry ID")?;
        if id == 0
            || nonnegative(&s, 1, "original public rowid")? != rowid
            || nonnegative(&s, 2, "created revision")? == 0
            || !s.column_is_null(3)?
        {
            return Err(corrupt(
                "DiskANN current mapping references invalid version",
            ));
        }
        if s.step()? != Step::Done {
            return Err(corrupt("DiskANN current mapping not unique"));
        }
        s.finish()?;
        Ok(Some(id))
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
        let mut s=self.connection.prepare(&format!("UPDATE {} SET revision=?1,next_entry_id=?2,canonical_count=?3,retired_count=?4 WHERE singleton=1",self.names.meta))?;
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
        done(s)?;
        self.expect_one_change("canonical metadata")
    }
    fn insert_encoded(&self, rowid: u64, bytes: &[u8], m: &mut Metadata) -> Result<()> {
        checked_id(rowid)?;
        if self.contains(rowid)? {
            return Err(IndexError::with_code(
                ffi::SQLITE_CONSTRAINT as i32,
                "DiskANN rowid already exists",
            ));
        }
        let id = m.next_entry_id;
        m.next_entry_id = checked_increment(id, "entry ID")?;
        m.canonical_count = checked_increment(m.canonical_count, "canonical count")?;
        let mut s = self.connection.prepare(&format!(
            "INSERT INTO {} VALUES(?1,?2,?3,NULL,?4)",
            self.names.entries
        ))?;
        s.bind_i64(1, checked_id(id)?)?;
        s.bind_i64(2, checked_id(rowid)?)?;
        s.bind_i64(3, checked_id(m.revision)?)?;
        s.bind_blob(4, bytes)?;
        done(s)?;
        self.expect_one_change("immutable entry insertion")?;
        let mut s = self
            .connection
            .prepare(&format!("INSERT INTO {} VALUES(?1,?2)", self.names.current))?;
        s.bind_i64(1, checked_id(rowid)?)?;
        s.bind_i64(2, checked_id(id)?)?;
        done(s)?;
        self.expect_one_change("current insertion")
    }
    pub(crate) fn insert(&self, rowid: u64, vector: &[f32]) -> Result<()> {
        self.check_workspace(0)?;
        let bytes = self.encode_input(vector)?;
        let (mut m, anchor) = self.read_metadata()?;
        m.revision = checked_increment(m.revision, "revision")?;
        anchor.finish()?;
        self.insert_encoded(rowid, &bytes, &mut m)?;
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
        let (mut m, anchor) = self.read_metadata()?;
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
        anchor.finish()?;
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
                self.insert_encoded(rowid, &bytes, &mut m)?;
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
            let mut lookup = self.connection.prepare(&format!(
                "SELECT native_local_id,state FROM {} WHERE generation_id=?1 AND entry_id=?2",
                self.names.graph_nodes
            ))?;
            lookup.bind_i64(1, checked_id(generation)?)?;
            lookup.bind_i64(2, checked_id(id)?)?;
            let exists = lookup.step()? == Step::Row;
            if exists {
                nonnegative(&lookup, 0, "native local ID")?;
                if nonnegative(&lookup, 1, "graph node state")? != 0 || lookup.step()? != Step::Done
                {
                    return Err(corrupt("DiskANN reverse version mapping invalid"));
                }
            }
            lookup.finish()?;
            if exists {
                let g = self.read_generation(generation)?;
                let count = g
                    .eligible_count
                    .checked_sub(1)
                    .ok_or_else(|| corrupt("DiskANN eligible count underflow"))?;
                let mut s = self.connection.prepare(&format!(
                    "UPDATE {} SET eligible_count=?1 WHERE generation_id=?2",
                    self.names.generations
                ))?;
                s.bind_i64(1, checked_id(count)?)?;
                s.bind_i64(2, checked_id(generation)?)?;
                done(s)?;
                self.expect_one_change("generation eligibility")?;
            }
        }
        let mut s=self.connection.prepare(&format!("UPDATE {} SET retired_revision=?1 WHERE entry_id=?2 AND public_rowid=?3 AND retired_revision IS NULL",self.names.entries))?;
        s.bind_i64(1, checked_id(m.revision)?)?;
        s.bind_i64(2, checked_id(id)?)?;
        s.bind_i64(3, checked_id(rowid)?)?;
        done(s)?;
        self.expect_one_change("immutable version retirement")?;
        let mut s = self.connection.prepare(&format!(
            "DELETE FROM {} WHERE public_rowid=?1 AND entry_id=?2",
            self.names.current
        ))?;
        s.bind_i64(1, checked_id(rowid)?)?;
        s.bind_i64(2, checked_id(id)?)?;
        done(s)?;
        self.expect_one_change("current retirement")
    }
    pub(crate) fn update(&self, old_rowid: u64, new_rowid: u64, vector: &[f32]) -> Result<()> {
        checked_id(old_rowid)?;
        checked_id(new_rowid)?;
        if old_rowid != new_rowid {
            return Err(IndexError::new("DiskANN rowid cannot be changed by UPDATE"));
        }
        let bytes = self.encode_input(vector)?;
        let (mut m, anchor) = self.read_metadata()?;
        m.revision = checked_increment(m.revision, "revision")?;
        checked_increment(m.next_entry_id, "entry ID")?;
        checked_increment(m.retired_count, "retired count")?;
        anchor.finish()?;
        let id = self
            .lookup_live_id(old_rowid)?
            .ok_or_else(|| IndexError::new("DiskANN update rowid is absent"))?;
        self.retire(old_rowid, id, &mut m)?;
        self.insert_encoded(new_rowid, &bytes, &mut m)?;
        self.write_metadata(&m)
    }
    pub(crate) fn mark_delete(&self, rowid: u64) -> Result<()> {
        checked_id(rowid)?;
        let (mut m, anchor) = self.read_metadata()?;
        m.revision = checked_increment(m.revision, "revision")?;
        anchor.finish()?;
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

    fn validate_columns(&self, table: &str, expected: &[(&str, &str, u64, u64)]) -> Result<()> {
        let mut columns = self.connection.prepare(
            "SELECT cid,length(name),name,length(type),type,\"notnull\",pk,hidden FROM pragma_table_xinfo(?1,?2) ORDER BY cid"
        )?;
        columns.bind_text(1, table)?;
        columns.bind_text(2, &self.names.schema)?;
        for (index, &(name, kind, not_null, primary_key)) in expected.iter().enumerate() {
            if columns.step()? != Step::Row
                || nonnegative(&columns, 0, "column position")? != index as u64
                || nonnegative(&columns, 1, "column name length")? != name.len() as u64
                || nonnegative(&columns, 3, "column type length")? != kind.len() as u64
                || columns.column_text(2)? != name
                || !columns.column_text(4)?.eq_ignore_ascii_case(kind)
                || nonnegative(&columns, 5, "not null flag")? != not_null
                || nonnegative(&columns, 6, "primary key flag")? != primary_key
                || nonnegative(&columns, 7, "hidden column flag")? != 0
            {
                return Err(corrupt("DiskANN shadow-table schema is incompatible"));
            }
        }
        if columns.step()? != Step::Done {
            return Err(corrupt(
                "DiskANN shadow-table schema has additional columns",
            ));
        }
        columns.finish()
    }

    fn require_rollback_journal(&self) -> Result<()> {
        let mut mode = self.connection.prepare(&format!(
            "PRAGMA {}.journal_mode",
            quote_identifier(&self.names.schema)?
        ))?;
        if mode.step()? != Step::Row {
            return Err(corrupt("DiskANN could not read SQLite journal_mode"));
        }
        let name = mode.column_text(0)?;
        if mode.step()? != Step::Done {
            return Err(corrupt("DiskANN journal_mode returned unexpected rows"));
        }
        mode.finish()?;
        validate_journal_mode(&name)
    }
    fn verify_writable_guard(&self) -> Result<()> {
        // OFF disables SQLite rollback itself, including nested carrier writes.
        // Refuse it before action; never silently change the host's settings.
        self.require_rollback_journal()?;
        self.check_format()?;
        self.validate_schema()?;
        let (m, a) = self.read_metadata()?;
        self.validate_generation_identities(&m)?;
        a.finish()?;
        self.verify_guard()
    }

    fn verify_guard(&self) -> Result<()> {
        // A full-table UPDATE over these two immutable rows forces SQLite's
        // carrier to journal every nested graph write. There are intentionally
        // no triggers: ENABLE_TRIGGER=off, FAIL, and IGNORE cannot remove or
        // weaken the statement's rollback boundary.
        let mut kind = self.connection.prepare(
            "SELECT type,ncol,wr,strict FROM pragma_table_list(?1) WHERE schema=?2 COLLATE NOCASE",
        )?;
        kind.bind_text(1, &self.names.base("diskann_txn"))?;
        kind.bind_text(2, &self.names.schema)?;
        if kind.step()? != Step::Row
            || !matches!(kind.column_text(0)?.as_str(), "table" | "shadow")
            || nonnegative(&kind, 1, "atomic guard column count")? != 2
            || nonnegative(&kind, 2, "atomic guard WITHOUT ROWID flag")? != 0
            || nonnegative(&kind, 3, "atomic guard STRICT flag")? != 0
            || kind.step()? != Step::Done
        {
            return Err(corrupt(
                "DiskANN atomic guard must be an ordinary rowid shadow table",
            ));
        }
        kind.finish()?;
        self.validate_columns(
            &self.names.base("diskann_txn"),
            &[("singleton", "INTEGER", 0, 1), ("value", "INTEGER", 1, 0)],
        )?;
        for pragma in ["pragma_index_list", "pragma_foreign_key_list"] {
            let mut relations = self
                .connection
                .prepare(&format!("SELECT 1 FROM {pragma}(?1,?2) LIMIT 1"))?;
            relations.bind_text(1, &self.names.base("diskann_txn"))?;
            relations.bind_text(2, &self.names.schema)?;
            if relations.step()? != Step::Done {
                return Err(corrupt(
                    "DiskANN atomic guard cannot have indices or foreign keys",
                ));
            }
            relations.finish()?;
        }
        let mut rows = self.connection.prepare(&format!(
            "SELECT singleton,value FROM {} ORDER BY singleton LIMIT 3",
            self.names.txn
        ))?;
        for singleton in [1, 2] {
            if rows.step()? != Step::Row
                || nonnegative(&rows, 0, "atomic guard singleton")? != singleton
                || nonnegative(&rows, 1, "atomic guard value")? != 0
            {
                return Err(corrupt(
                    "DiskANN atomic guard must contain its two unchanged rows",
                ));
            }
        }
        if rows.step()? != Step::Done {
            return Err(corrupt("DiskANN atomic guard contains unexpected rows"));
        }
        rows.finish()?;

        // TEMP triggers can target a non-TEMP table. Conservatively reject the
        // matching base name in both schemas before EVERY carrier preparation.
        self.reject_guard_triggers(&self.names.schema)?;
        if !self.names.schema.eq_ignore_ascii_case("temp") {
            self.reject_guard_triggers("temp")?;
        }
        Ok(())
    }

    fn reject_guard_triggers(&self, schema: &str) -> Result<()> {
        let mut trigger = self.connection.prepare(&format!(
            "SELECT 1 FROM {} WHERE type='trigger' AND tbl_name=?1 COLLATE NOCASE LIMIT 1",
            qualified_name(schema, "sqlite_schema")?
        ))?;
        trigger.bind_text(1, &self.names.base("diskann_txn"))?;
        if trigger.step()? != Step::Done {
            return Err(corrupt(
                "DiskANN atomic guard cannot have schema or TEMP triggers",
            ));
        }
        trigger.finish()
    }
}
