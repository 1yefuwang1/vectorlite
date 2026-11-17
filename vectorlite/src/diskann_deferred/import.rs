//! Immutable candidate chunk import and progressive SQL validation.
//! Prefetch candidate copies BEFORE entering the caller's SQL carrier.
use super::jobs::{self, BuildBinding, StepOptions, BUILDING, COMPLETED, VALIDATED, VALIDATING};
use super::*;
use crate::diskann_worker::{
    CandidateChunk, CandidateHandle, CandidateManifest, CandidateSessions, ChunkNode,
};

fn worker_error(e: crate::diskann_worker::WorkerError) -> IndexError {
    use crate::diskann_worker::WorkerErrorKind;
    let message = e.to_string();
    match e.kind() {
        WorkerErrorKind::Allocation => IndexError::with_code(ffi::SQLITE_NOMEM as i32, message),
        WorkerErrorKind::BudgetExceeded
        | WorkerErrorKind::CapacityExhausted
        | WorkerErrorKind::TooManyIndexes => too_big(message),
        _ => jobs::stale(&message),
    }
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
/// Only this constructor admits an immutable candidate; graph records are private.
pub(crate) struct CandidateInput {
    binding: BuildBinding,
    manifest: CandidateManifest,
    chunk: CandidateChunk,
    root: Vec<u8>,
}
impl CandidateInput {
    /// Call outside atomic_write. Identity is rooted in the private worker session,
    /// not SQL caller-supplied graph data or token text alone.
    pub(crate) fn copy(
        sessions: &CandidateSessions,
        handle: &CandidateHandle,
        binding: &BuildBinding,
        raw_digest: [u8; 32],
        cursor: u64,
        opts: StepOptions,
    ) -> Result<Self> {
        let identity = handle.identity();
        if identity.connection() != sessions.connection()
            || identity.connection().as_str() != format!("conn:{}", hex(&binding.owner_nonce))
            || identity.binding().index().as_str() != format!("idx:{}", hex(&binding.instance_id))
            || identity.job().as_str() != format!("job:{}", binding.job_id)
            || identity.fence().as_str() != format!("fence:{}", binding.fence)
            || identity.input_digest().as_bytes() != &raw_digest
            || identity.config_digest().as_bytes() != &binding.config_hash
        {
            return Err(jobs::stale("candidate session identity mismatch"));
        }
        let manifest = sessions.manifest(handle).map_err(worker_error)?;
        if manifest.input_digest.as_bytes() != &raw_digest
            || manifest.config_digest.as_bytes() != &binding.config_hash
        {
            return Err(jobs::stale("native candidate digest mismatch"));
        }
        let per = (manifest.max_degree as usize)
            .checked_mul(8)
            .and_then(|n| n.checked_add(manifest.dimension as usize * 4 + 96))
            .ok_or_else(|| too_big("candidate chunk bytes"))?;
        if opts.max_bytes < per {
            return Err(too_big("candidate chunk budget below one node"));
        }
        let limit = (opts.max_records as usize).min(opts.max_bytes / per) as u32;
        let start = u32::try_from(cursor).map_err(|_| too_big("candidate cursor"))?;
        let chunk = sessions
            .copy_chunk(handle, start, limit)
            .map_err(worker_error)?;
        let root = if chunk.nodes.iter().any(|n| Some(n.ordinal) == manifest.root) {
            let floats = sessions.root_vector(handle).map_err(worker_error)?;
            let mut bytes = Vec::new();
            reserve(&mut bytes, floats.len() * 4)?;
            for v in floats {
                if !v.is_finite() {
                    return Err(corrupt("DiskANN candidate frozen vector"));
                }
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes
        } else {
            Vec::new()
        };
        Ok(Self {
            binding: binding.clone(),
            manifest,
            chunk,
            root,
        })
    }
}
fn verify(table: &DiskAnnTable, input: &CandidateInput, cursor: u64) -> Result<Job> {
    let j = jobs::bound(table, &input.binding)?;
    let m = &input.manifest;
    let nodes = if j.expected_count == 0 {
        0
    } else {
        j.expected_count + 1
    };
    let config = jobs::build_config(table);
    if u64::from(m.population) != j.expected_count
        || u64::from(m.node_count) != nodes
        || m.root.map(u64::from)
            != (if j.expected_count == 0 {
                None
            } else {
                Some(j.expected_count)
            })
        || m.dimension as usize != table.space().dim
        || m.metric != config.metric
        || m.max_degree as usize != config.max_degree
        || m.construction_l as usize != config.construction_l
        || input.chunk.start as u64 != cursor
    {
        return Err(jobs::stale("candidate population/config/cursor"));
    }
    if j.candidate_hash
        .is_some_and(|h| &h != m.topology_digest.as_bytes())
    {
        return Err(jobs::stale("another native topology already bound"));
    }
    if input.chunk.nodes.is_empty() && cursor < nodes {
        return Err(corrupt("DiskANN early candidate chunk exhaustion"));
    }
    Ok(j)
}
/// The two graph-node row shapes. Each kind has its own statement so the column
/// that must stay NULL for that kind is SQL text and never an unbound parameter:
/// sqlite3_reset keeps the previous iteration's bindings, so a shared statement
/// would let the frozen root inherit the live node's entry_id and violate the
/// (state, entry_id) CHECK. Every parameter below is bound on every iteration.
#[derive(Clone, Copy)]
enum NodeKind {
    Live,
    FrozenRoot,
}
impl NodeKind {
    fn sql(self, names: &StorageNames) -> String {
        match self {
            Self::Live => format!(
                "INSERT INTO {} VALUES(?1,?2,?3,0,NULL,?4)",
                names.graph_nodes
            ),
            Self::FrozenRoot => {
                format!(
                    "INSERT INTO {} VALUES(?1,?2,NULL,2,?3,?4)",
                    names.graph_nodes
                )
            }
        }
    }
}
fn map_sql(names: &StorageNames) -> String {
    format!(
        "SELECT entry_id,public_rowid FROM {} WHERE job_id=?1 AND ordinal=?2",
        names.snapshot_map
    )
}
/// Verify the stored map and immutable source version against the trusted candidate.
fn map_node(
    table: &DiskAnnTable,
    j: &Job,
    node: &ChunkNode,
    mapped: &mut jobs::LoopStatement,
) -> Result<Option<(u64, u64)>> {
    if node.ordinal as u64 == j.expected_count && j.expected_count > 0 {
        if node.entry.is_some() {
            return Err(corrupt("DiskANN frozen node source mapping"));
        }
        return Ok(None);
    }
    if node.ordinal as u64 >= j.expected_count {
        return Err(corrupt("DiskANN native ordinal range"));
    }
    let source = node
        .entry
        .ok_or_else(|| corrupt("DiskANN live node missing snapshot identity"))?;
    let s = mapped.begin(table.connection())?;
    s.bind_i64(1, checked_id(j.job_id)?)?;
    s.bind_i64(2, i64::from(node.ordinal))?;
    if s.step()? != Step::Row {
        return Err(corrupt("DiskANN imported snapshot map missing"));
    }
    let entry = nonnegative(s, 0, "mapped entry")?;
    let rowid = nonnegative(s, 1, "mapped rowid")?;
    if s.step()? != Step::Done {
        return Err(corrupt("DiskANN snapshot ordinal not unique"));
    }
    let e = table.read_entry(entry)?;
    if source.entry != entry
        || source.version != e.created_revision
        || rowid != e.public_rowid
        || entry > j.cutoff_entry_id
        || e.created_revision > j.cutoff_revision
        || e.retired_revision.is_some_and(|r| r <= j.cutoff_revision)
    {
        return Err(corrupt("DiskANN candidate/source map mismatch"));
    }
    Ok(Some((entry, rowid)))
}
fn adjacency(table: &DiskAnnTable, j: &Job, node: &ChunkNode) -> Result<Vec<u8>> {
    let mut ids = Vec::new();
    reserve(&mut ids, node.neighbors.len())?;
    for &id in &node.neighbors {
        if u64::from(id) > j.expected_count {
            return Err(corrupt("DiskANN candidate adjacency range"));
        }
        ids.push(u64::from(id));
    }
    if node.offset_end.checked_sub(node.offset_start) != Some(ids.len() as u64) {
        return Err(corrupt("DiskANN candidate CSR offsets"));
    }
    encode_ids(
        &ids,
        table.options().physical_degree(),
        Some(u64::from(node.ordinal)),
    )
}
fn advance(
    table: &DiskAnnTable,
    j: &Job,
    input: &CandidateInput,
    next: u64,
    validation: bool,
) -> Result<()> {
    let total = u64::from(input.manifest.node_count);
    if next > total {
        return Err(corrupt("DiskANN chunk exceeds native population"));
    }
    let column = if validation {
        "validation_cursor"
    } else {
        "import_cursor"
    };
    let old = if validation {
        j.validation_cursor
    } else {
        j.import_cursor
    };
    let phase = if next == total {
        if validation {
            VALIDATED
        } else {
            VALIDATING
        }
    } else {
        j.phase
    };
    let mut s=table.connection().prepare(&format!("UPDATE {} SET {column}=?1,phase=?2,candidate_hash=?3 WHERE job_id=?4 AND fence=?5 AND owner_nonce=?6 AND {column}=?7 AND phase=?8",table.names().jobs))?;
    s.bind_i64(1, checked_id(next)?)?;
    s.bind_i64(2, phase as i64)?;
    s.bind_blob(3, input.manifest.topology_digest.as_bytes())?;
    s.bind_i64(4, checked_id(j.job_id)?)?;
    s.bind_i64(5, checked_id(j.fence)?)?;
    s.bind_blob(6, &input.binding.owner_nonce)?;
    s.bind_i64(7, checked_id(old)?)?;
    s.bind_i64(8, j.phase as i64)?;
    done(s)?;
    jobs::changed(table, "candidate progress")
}
pub(crate) fn import_step(table: &DiskAnnTable, input: &CandidateInput) -> Result<Job> {
    let j = verify(table, input, input.chunk.start as u64)?;
    if j.phase != BUILDING || j.import_cursor != input.chunk.start as u64 {
        return Err(jobs::stale("import phase/cursor"));
    }
    let mut eligible = 0;
    let mut next = j.import_cursor;
    // One statement per node kind for the whole bounded chunk. Each is prepared at
    // most once per step call, on the first node of its kind, and every iteration
    // starts with reset + clear_bindings inside this step call.
    let mut mapped = jobs::LoopStatement::new(map_sql(table.names()));
    let mut live = jobs::LoopStatement::new(NodeKind::Live.sql(table.names()));
    let mut frozen = jobs::LoopStatement::new(NodeKind::FrozenRoot.sql(table.names()));
    for node in &input.chunk.nodes {
        if u64::from(node.ordinal) != next {
            return Err(corrupt("DiskANN import order"));
        }
        let mapped_row = map_node(table, &j, node, &mut mapped)?;
        let adj = adjacency(table, &j, node)?;
        if let Some((entry, rowid)) = mapped_row {
            let statement = live.begin(table.connection())?;
            statement.bind_i64(1, checked_id(j.generation_id)?)?;
            statement.bind_i64(2, i64::from(node.ordinal))?;
            statement.bind_i64(3, checked_id(entry)?)?;
            statement.bind_blob(4, &adj)?;
            // The live-id probe runs before the row is written, in the order the
            // per-row statement it replaces used.
            let eligible_now = table.lookup_live_id(rowid)? == Some(entry);
            jobs::drain(statement)?;
            if eligible_now {
                eligible += 1;
            }
        } else {
            if input.root.len() != table.vector_bytes() {
                return Err(corrupt("DiskANN frozen root length"));
            }
            let statement = frozen.begin(table.connection())?;
            statement.bind_i64(1, checked_id(j.generation_id)?)?;
            statement.bind_i64(2, i64::from(node.ordinal))?;
            statement.bind_blob(3, &input.root)?;
            statement.bind_blob(4, &adj)?;
            jobs::drain(statement)?;
        }
        jobs::changed(table, "candidate node insert")?;
        next += 1;
    }
    mapped.finish()?;
    live.finish()?;
    frozen.finish()?;
    if eligible > 0 {
        let mut s=table.connection().prepare(&format!("UPDATE {} SET eligible_count=eligible_count+?1 WHERE generation_id=?2 AND state=0 AND eligible_count+?1<=total_count",table.names().generations))?;
        s.bind_i64(1, eligible)?;
        s.bind_i64(2, checked_id(j.generation_id)?)?;
        done(s)?;
        jobs::changed(table, "candidate eligible increment")?;
    }
    advance(table, &j, input, next, false)?;
    table.read_job(j.job_id)
}
/// Progressive equality to the already validated native immutable candidate.
/// No O(N+E) native validation or whole-SQL digest scan under writer lock.
pub(crate) fn validate_step(table: &DiskAnnTable, input: &CandidateInput) -> Result<Job> {
    let j = verify(table, input, input.chunk.start as u64)?;
    if j.phase != VALIDATING
        || j.validation_cursor != input.chunk.start as u64
        || j.import_cursor != u64::from(input.manifest.node_count)
    {
        return Err(jobs::stale("validation phase/cursor"));
    }
    let g = table.read_generation(j.generation_id)?;
    let mut next = j.validation_cursor;
    // One statement for the whole bounded chunk, prepared at most once per call.
    let mut mapped_statement = jobs::LoopStatement::new(map_sql(table.names()));
    for node in &input.chunk.nodes {
        if u64::from(node.ordinal) != next {
            return Err(corrupt("DiskANN validation order"));
        }
        let mapped = map_node(table, &j, node, &mut mapped_statement)?;
        let expected = adjacency(table, &j, node)?;
        let stored = table.read_graph_node(&g, u64::from(node.ordinal))?;
        let actual = encode_ids(
            &stored.neighbors,
            table.options().physical_degree(),
            Some(u64::from(node.ordinal)),
        )?;
        if stored.native_local_id != u64::from(node.ordinal)
            || stored.entry_id != mapped.map(|p| p.0)
            || actual != expected
            || (mapped.is_none() && stored.frozen_vector.as_deref() != Some(input.root.as_slice()))
            || (mapped.is_some() && stored.frozen_vector.is_some())
        {
            return Err(corrupt(
                "DiskANN stored topology differs from immutable candidate",
            ));
        }
        next += 1;
    }
    mapped_statement.finish()?;
    if next == u64::from(input.manifest.node_count) {
        // Indexed range seeks detect extra corrupt rows without COUNT/full scan.
        for (name, column, id) in [
            (
                &table.names().graph_nodes,
                "native_local_id",
                j.generation_id,
            ),
            (&table.names().snapshot_map, "ordinal", j.job_id),
        ] {
            let key = if column == "ordinal" {
                "job_id"
            } else {
                "generation_id"
            };
            let mut s = table.connection().prepare(&format!(
                "SELECT 1 FROM {name} WHERE {key}=?1 AND {column}>=?2 LIMIT 1"
            ))?;
            s.bind_i64(1, checked_id(id)?)?;
            s.bind_i64(
                2,
                checked_id(if column == "ordinal" {
                    j.expected_count
                } else {
                    next
                })?,
            )?;
            if s.step()? != Step::Done {
                return Err(corrupt("DiskANN extra snapshot/topology rows"));
            }
            s.finish()?;
        }
    }
    advance(table, &j, input, next, true)?;
    table.read_job(j.job_id)
}
/// Short fenced metadata transition. Success inside BEGIN is NOT a COMMIT.
/// Root may release RAM only after this helper returns, never as a SQL effect.
pub(crate) fn publish(
    table: &DiskAnnTable,
    b: &BuildBinding,
    candidate_hash: [u8; 32],
) -> Result<Job> {
    let j = jobs::bound(table, b)?;
    let total = if j.expected_count == 0 {
        0
    } else {
        j.expected_count + 1
    };
    if j.phase != VALIDATED
        || j.captured_count != j.expected_count
        || j.import_cursor != total
        || j.validation_cursor != total
        || j.candidate_hash != Some(candidate_hash)
    {
        return Err(jobs::stale(
            "publication requires fully validated candidate",
        ));
    }
    if let Some(old) = j.parent_generation {
        let mut s = table.connection().prepare(&format!(
            "UPDATE {} SET state=2 WHERE generation_id=?1 AND state=1",
            table.names().generations
        ))?;
        s.bind_i64(1, checked_id(old)?)?;
        done(s)?;
        jobs::changed(table, "parent generation transition")?;
    }
    let mut s=table.connection().prepare(&format!("UPDATE {} SET state=1 WHERE generation_id=?1 AND job_id=?2 AND state=0 AND source_hash=?3 AND config_hash=?4",table.names().generations))?;
    s.bind_i64(1, checked_id(j.generation_id)?)?;
    s.bind_i64(2, checked_id(j.job_id)?)?;
    s.bind_blob(3, &b.source_hash)?;
    s.bind_blob(4, &b.config_hash)?;
    done(s)?;
    jobs::changed(table, "publish candidate")?;
    let mut s=table.connection().prepare(&format!("UPDATE {} SET phase=5 WHERE job_id=?1 AND fence=?2 AND owner_nonce=?3 AND phase=4 AND candidate_hash=?4",table.names().jobs))?;
    s.bind_i64(1, checked_id(j.job_id)?)?;
    s.bind_i64(2, checked_id(j.fence)?)?;
    s.bind_blob(3, &b.owner_nonce)?;
    s.bind_blob(4, &candidate_hash)?;
    done(s)?;
    jobs::changed(table, "complete job")?;
    let mut s=table.connection().prepare(&format!("UPDATE {} SET published_generation=?1,candidate_generation=NULL,active_job=NULL WHERE singleton=1 AND active_job=?2 AND candidate_generation=?1 AND published_generation IS ?3",table.names().meta))?;
    s.bind_i64(1, checked_id(j.generation_id)?)?;
    s.bind_i64(2, checked_id(j.job_id)?)?;
    if let Some(p) = j.parent_generation {
        s.bind_i64(3, checked_id(p)?)?;
    } else {
        /* Fresh prepared parameters default to SQL NULL; statement is never reused. */
    }
    done(s)?;
    jobs::changed(table, "publication metadata fence")?;
    let complete = table.read_job(j.job_id)?;
    debug_assert_eq!(complete.phase, COMPLETED);
    Ok(complete)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskann_worker::{RebuildCause, WorkerError};

    /// Renders one `VALUES` row the way SQLite sees it: `?N` takes the parameter
    /// bound for this iteration, `NULL` is SQL NULL and an integer stays literal.
    /// Every declared slot must be bound and no parameter may be left over, which
    /// is what makes an iteration independent of the previous one.
    fn modeled_row(sql: &str, parameters: &[Option<&str>]) -> Vec<Option<String>> {
        let marker = " VALUES(";
        let start = sql.find(marker).expect("statement writes a VALUES row") + marker.len();
        let list = sql[start..sql.len() - 1].trim();
        let tokens: Vec<&str> = list.split(',').map(str::trim).collect();
        let mut bound = 0usize;
        let row = tokens
            .iter()
            .map(|token| match token.strip_prefix('?') {
                Some(index) => {
                    let index: usize = index.parse().expect("numeric SQLite parameter index");
                    bound = bound.max(index);
                    parameters
                        .get(index - 1)
                        .copied()
                        .flatten()
                        .map(str::to_string)
                }
                None if *token == "NULL" => None,
                None => Some((*token).to_string()),
            })
            .collect();
        assert_eq!(bound, parameters.len(), "every slot is bound exactly once");
        row
    }
    fn rendered(row: &[Option<String>]) -> Vec<Option<&str>> {
        row.iter().map(|cell| cell.as_deref()).collect()
    }

    #[test]
    fn live_node_then_frozen_root_inside_one_reuse_cycle_keeps_entry_id_null() {
        let names = StorageNames::new("main", "v").unwrap();
        // The persisted column order of the shadow table this statement writes.
        let columns: Vec<&str> = super::super::schema::LAYOUTS
            .iter()
            .find(|layout| layout.suffix == "diskann_graph_nodes")
            .unwrap()
            .columns
            .iter()
            .map(|column| column.0)
            .collect();
        assert_eq!(
            columns,
            [
                "generation_id",
                "native_local_id",
                "entry_id",
                "state",
                "frozen_vector",
                "neighbors"
            ]
        );
        // Row 0 of one bounded chunk: a live node. Every slot of its statement is
        // bound on this iteration.
        let live = modeled_row(
            &NodeKind::Live.sql(&names),
            &[Some("9"), Some("0"), Some("11"), Some("adj")],
        );
        assert_eq!(
            rendered(&live),
            [
                Some("9"),
                Some("0"),
                Some("11"),
                Some("0"),
                None,
                Some("adj")
            ]
        );
        // Row 1 of the same chunk: the frozen root. `?3` (entry_id) is not a
        // parameter of this statement at all, so a reset that preserved the live
        // node's bindings still cannot give the root an entry_id; `?5`
        // (frozen_vector) likewise cannot be inherited by a later live node.
        let root = modeled_row(
            &NodeKind::FrozenRoot.sql(&names),
            &[Some("9"), Some("1"), Some("root"), Some("adj")],
        );
        assert_eq!(
            rendered(&root),
            [
                Some("9"),
                Some("1"),
                None,
                Some("2"),
                Some("root"),
                Some("adj")
            ]
        );
        // Column 2 is entry_id and column 4 is frozen_vector in both kinds.
        assert_eq!(live[2].as_deref(), Some("11"));
        assert_eq!(root[2], None);
        assert_eq!(live[4], None);
        assert_eq!(root[4].as_deref(), Some("root"));
    }

    #[test]
    fn graph_node_statements_have_one_shape_per_kind() {
        let names = StorageNames::new("main", "v").unwrap();
        assert_eq!(
            NodeKind::Live.sql(&names),
            "INSERT INTO \"main\".\"v_diskann_graph_nodes\" VALUES(?1,?2,?3,0,NULL,?4)"
        );
        assert_eq!(
            NodeKind::FrozenRoot.sql(&names),
            "INSERT INTO \"main\".\"v_diskann_graph_nodes\" VALUES(?1,?2,NULL,2,?3,?4)"
        );
        assert_eq!(
            map_sql(&names),
            "SELECT entry_id,public_rowid FROM \"main\".\"v_diskann_snapshot_map\" WHERE job_id=?1 AND ordinal=?2"
        );
    }

    #[test]
    fn worker_copy_errors_preserve_resource_and_stale_codes() {
        assert_eq!(
            worker_error(WorkerError::Allocation {
                what: "owned chunk"
            })
            .code,
            ffi::SQLITE_NOMEM as i32
        );
        assert_eq!(
            worker_error(WorkerError::BudgetExceeded {
                requested: 2,
                limit: 1
            })
            .code,
            ffi::SQLITE_TOOBIG as i32
        );
        assert_eq!(
            worker_error(WorkerError::CapacityExhausted {
                what: "candidate ordinal"
            })
            .code,
            ffi::SQLITE_TOOBIG as i32
        );
        assert_eq!(
            worker_error(WorkerError::RebuildNeeded {
                cause: RebuildCause::LostCandidate
            })
            .code,
            ffi::SQLITE_BUSY as i32
        );
    }
}
