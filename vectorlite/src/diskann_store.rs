//! SQLite-contained, write-through DiskANN storage.
//!
//! There is deliberately no persistent Rust graph cache and no independent
//! connection or transaction here. SQLite owns every durable mutation and its
//! rollback. Mutating callbacks enter `atomic_write`: its ordinary two-row
//! carrier statement also journals nested graph SQL when an outer single-row
//! virtual-table write would otherwise lack that rollback boundary. No triggers
//! or transaction-control SQL are used. The virtual table supplies a borrowed
//! `Connection<'static>`: that
//! erased lifetime is valid only because SQLite destroys the virtual table and
//! all of its cursors before closing the host connection. Cursor-owned anchor
//! statements are finalized by the parent before a shadow table is destroyed.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::mem::size_of;

use crate::batch_input::BatchView;
use crate::core::{SearchFilter, SearchResult};
use crate::diskann_core::{self, GraphConfig, GraphStore, NodeState, ResourceLimits};
use crate::ffi;
use crate::index_error::IndexError;
use crate::index_options::DiskAnnOptions;
use crate::ops;
use crate::sqlite::{qualified_name, quote_identifier, Connection, Statement, Step};
use crate::vector_space::{DistanceType, NamedVectorSpace, VectorType};

type Result<T> = std::result::Result<T, IndexError>;
type MaterializedRows = (Vec<SearchResult>, Option<Vec<Vec<u8>>>);
const FORMAT_VERSION: i64 = 3;
const ATOMIC_GUARD_WORKSPACE: usize = 1024;
const MAX_DESCRIPTOR_BYTES: usize = 16 * 1024;
const EXACT_FILTER_LIMIT: usize = 1024;
const CONSOLIDATE_BATCH: usize = 128;
const MAX_ID: u64 = i64::MAX as u64;
const MAX_INSERT_BATCH: usize = 32;
const STATEMENT_CACHE_WORKSPACE: usize = 16 * 1024;

/// A lightweight per-vtab handle. None of its fields depend on a transaction.
pub(crate) struct DiskAnnTable {
    connection: Connection<'static>,
    schema: String,
    meta_name: String,
    nodes_name: String,
    txn_table: String,
    txn_name: String,
    rebuild_table: String,
    rebuild_name: String,
    space: NamedVectorSpace,
    options: DiskAnnOptions,
    descriptor: serde_json::Value,
    vector_bytes: usize,
    adjacency_bytes: usize,
}

/// Results and projected vectors were read under `anchor`'s SQLite snapshot.
/// Keep the anchor at Row until the cursor is closed or filtered again.
pub(crate) struct QueryRows {
    pub(crate) results: Vec<SearchResult>,
    pub(crate) vectors: Option<Vec<Vec<u8>>>,
    pub(crate) anchor: Statement<'static>,
}

struct Metadata {
    revision: u64,
    next_node_id: u64,
    live_count: u64,
    deleted_count: u64,
    entrypoints: Vec<u64>,
}

/// Every instance exists for only one callback. All graph operations issue SQL
/// against the same host connection, including while a cursor anchor is active.
struct SqliteGraphStore<'a> {
    table: &'a DiskAnnTable,
    // Counts vector visits across multiple upstream scopes (UPDATE and exact
    // fallback), not just one core invocation. Maintenance resets per node.
    visits: usize,
    // Fixed, lazy prepared plans only; neither graph data nor metadata is cached.
    // Bindings are cleared after every use, and all plans finalize in this callback.
    statements: [Option<Statement<'static>>; 11],
}

#[derive(Clone, Copy)]
enum GraphStatement {
    Identity,
    Vector,
    Neighbors,
    SetNeighbors,
    Lookup,
    Metadata,
    Frozen,
    InsertNode,
    AllocateMetadata,
    DeleteNode,
    DeleteMetadata,
}

fn corrupt(message: impl Into<String>) -> IndexError {
    IndexError::with_code(ffi::SQLITE_CORRUPT as i32, message)
}

fn too_big(message: impl Into<String>) -> IndexError {
    IndexError::with_code(ffi::SQLITE_TOOBIG as i32, message)
}

fn no_memory() -> IndexError {
    IndexError::with_code(
        ffi::SQLITE_NOMEM as i32,
        "cannot allocate DiskANN workspace",
    )
}

fn reserve<T>(values: &mut Vec<T>, count: usize) -> Result<()> {
    values.try_reserve_exact(count).map_err(|_| no_memory())
}

fn checked_id(id: u64) -> Result<i64> {
    i64::try_from(id).map_err(|_| {
        IndexError::with_code(
            ffi::SQLITE_RANGE as i32,
            "DiskANN row/node ID exceeds i64::MAX",
        )
    })
}

fn nonnegative(statement: &Statement<'_>, column: usize, name: &str) -> Result<u64> {
    // Type errors in an on-disk format are corruption, not user coercions.
    if statement.column_type(column)? != ffi::SQLITE_INTEGER as i32 {
        return Err(corrupt(format!("DiskANN {name} is not an INTEGER")));
    }
    u64::try_from(statement.column_i64(column)?)
        .map_err(|_| corrupt(format!("DiskANN {name} is negative")))
}

fn checked_increment(value: u64, name: &str) -> Result<u64> {
    value
        .checked_add(1)
        .filter(|&value| value <= MAX_ID)
        .ok_or_else(|| too_big(format!("DiskANN {name} exhausted")))
}

fn batch_preflight(metadata: &Metadata, count: usize) -> Result<(u64, u64, u64)> {
    let count = u64::try_from(count).map_err(|_| too_big("DiskANN batch population overflow"))?;
    let add = |value: u64, name: &str| {
        value
            .checked_add(count)
            .filter(|&value| value <= MAX_ID)
            .ok_or_else(|| too_big(format!("DiskANN batch {name} exhausted")))
    };
    Ok((
        add(metadata.next_node_id, "next node ID")?,
        add(metadata.live_count, "live count")?,
        checked_increment(metadata.revision, "revision")?,
    ))
}

fn rebuild_preflight(metadata: &Metadata) -> Result<u64> {
    metadata
        .next_node_id
        .checked_add(metadata.live_count)
        .filter(|&next| next <= MAX_ID)
        .ok_or_else(|| too_big("DiskANN rebuild would exhaust never-reused node IDs"))?;
    metadata
        .live_count
        .checked_add(metadata.deleted_count)
        .filter(|&count| count <= MAX_ID)
        .ok_or_else(|| too_big("DiskANN rebuild retired population overflow"))
}

fn checked_product(a: usize, b: usize, name: &str) -> Result<usize> {
    a.checked_mul(b)
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| too_big(format!("DiskANN {name} size overflow")))
}

fn blob_projection(column: &str, maximum: usize, exact: bool) -> String {
    // Identifiers here are fixed implementation column names, never user input.
    // SQLite's length(BLOB)/typeof opcodes read only record header metadata;
    // CASE prevents its payload Column opcode from copying oversized corrupt
    // BLOBs or TEXT before the Rust wrapper can inspect the length field.
    let comparison = if exact { "=" } else { "<=" };
    format!(
        "CASE WHEN typeof({column})='blob' THEN length({column}) ELSE -1 END,CASE WHEN typeof({column})='blob' AND length({column}){comparison}{maximum} THEN {column} ELSE NULL END"
    )
}

fn bounded_blob(
    statement: &Statement<'_>,
    length_column: usize,
    blob_column: usize,
    maximum: usize,
    exact: bool,
    name: &str,
) -> Result<Vec<u8>> {
    let length = nonnegative(statement, length_column, name)?;
    if length > maximum as u64 || (exact && length != maximum as u64) {
        return Err(corrupt(format!(
            "DiskANN {name} has an invalid byte length"
        )));
    }
    if statement.column_type(blob_column)? != ffi::SQLITE_BLOB as i32 {
        return Err(corrupt(format!("DiskANN {name} is not a BLOB")));
    }
    // SQL length() is checked BEFORE the wrapper copies any BLOB bytes.
    let bytes = statement.column_blob_owned(blob_column)?;
    if bytes.len() as u64 != length {
        return Err(corrupt(format!("DiskANN {name} byte length changed")));
    }
    Ok(bytes)
}

fn encode_ids(ids: &[u64], maximum: usize, owner: Option<u64>) -> Result<Vec<u8>> {
    validate_ids(ids, maximum, owner)?;
    let bytes = checked_product(ids.len(), 8, "adjacency")?;
    let mut output = Vec::new();
    reserve(&mut output, bytes)?;
    for &id in ids {
        output.extend_from_slice(&id.to_le_bytes());
    }
    Ok(output)
}

fn validate_ids(ids: &[u64], maximum: usize, owner: Option<u64>) -> Result<()> {
    if ids.len() > maximum {
        return Err(corrupt("DiskANN adjacency exceeds physical degree"));
    }
    // Degree is bounded by the options; no graph-sized duplicate set is needed.
    for (index, &id) in ids.iter().enumerate() {
        if id > MAX_ID || owner == Some(id) || ids[..index].contains(&id) {
            return Err(corrupt(
                "DiskANN adjacency contains an invalid, duplicate, or self ID",
            ));
        }
    }
    Ok(())
}

fn decode_ids(bytes: &[u8], maximum: usize, owner: Option<u64>) -> Result<Vec<u64>> {
    if !bytes.len().is_multiple_of(8) || bytes.len() / 8 > maximum {
        return Err(corrupt("DiskANN adjacency has an invalid byte length"));
    }
    let mut output = Vec::new();
    reserve(&mut output, bytes.len() / 8)?;
    for chunk in bytes.as_chunks::<8>().0 {
        output.push(u64::from_le_bytes(*chunk));
    }
    validate_ids(&output, maximum, owner)?;
    Ok(output)
}

fn encode_vector(vector: &[f32]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reserve(&mut bytes, checked_product(vector.len(), 4, "vector")?)?;
    for &value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn decode_vector(bytes: &[u8], dimension: usize) -> Result<Vec<f32>> {
    if bytes.len() != checked_product(dimension, 4, "vector")? {
        return Err(corrupt("DiskANN vector has an invalid byte length"));
    }
    let mut vector = Vec::new();
    reserve(&mut vector, dimension)?;
    for chunk in bytes.as_chunks::<4>().0 {
        let value = f32::from_le_bytes(*chunk);
        if !value.is_finite() {
            return Err(corrupt("DiskANN stored vector contains a non-finite value"));
        }
        vector.push(value);
    }
    Ok(vector)
}

fn validate_numeric_range(vector: &[f32], metric: DistanceType) -> Result<f64> {
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(IndexError::new(
            "DiskANN vector contains a non-finite value",
        ));
    }
    let squared_norm: f64 = vector.iter().map(|&value| f64::from(value).powi(2)).sum();
    let (maximum, message) = match metric {
        // The triangle inequality gives ||a-b||² <= 4 * max(||a||²,||b||²).
        // This DiskANN-only restriction makes *all* accepted L2 pairs safe before
        // graph writes, rather than discovering overflow after allocation.
        DistanceType::L2 => (
            f64::from(f32::MAX) / 4.0,
            "DiskANN squared L2 requires squared norm <= f32::MAX/4 to avoid distance overflow",
        ),
        DistanceType::Cosine => (
            f64::from(f32::MAX),
            "DiskANN cosine norm would overflow float32",
        ),
        DistanceType::InnerProduct => {
            return Err(IndexError::new("DiskANN does not support inner product"))
        }
    };
    if !squared_norm.is_finite() || squared_norm > maximum {
        return Err(IndexError::new(message));
    }
    Ok(squared_norm)
}

fn normalize_input(vector: &mut [f32], dimension: usize, metric: DistanceType) -> Result<()> {
    if vector.len() != dimension {
        return Err(IndexError::new(format!(
            "dimension mismatch: expected {dimension}, got {}",
            vector.len()
        )));
    }
    let squared_norm = validate_numeric_range(vector, metric)?;
    if metric == DistanceType::Cosine {
        if !ops::ip_dist_f32(vector, vector).is_finite() {
            return Err(IndexError::new(
                "DiskANN cosine norm would overflow float32",
            ));
        }
        ops::normalize_f32(vector);
        if vector.iter().any(|value| !value.is_finite())
            || (squared_norm != 0.0 && vector.iter().all(|&value| value == 0.0))
        {
            return Err(IndexError::new(
                "DiskANN cosine normalization overflowed float32",
            ));
        }
    }
    Ok(())
}

fn validate_journal_mode(mode: &str) -> Result<()> {
    match mode {
        "delete" | "truncate" | "persist" | "memory" | "wal" => Ok(()),
        "off" => Err(IndexError::new(
            "DiskANN mutations require rollback journaling; journal_mode=OFF is not supported",
        )),
        _ => Err(corrupt(
            "DiskANN encountered an unsupported SQLite journal_mode",
        )),
    }
}

fn descriptor(space: &NamedVectorSpace, options: &DiskAnnOptions) -> serde_json::Value {
    serde_json::json!({
        "format_version": FORMAT_VERSION,
        "backend": "diskann",
        "dimension": space.dim,
        "vector_type": "float32",
        "metric": if space.distance_type == DistanceType::Cosine { "cosine" } else { "l2_squared" },
        "normalize": space.distance_type == DistanceType::Cosine,
        "adapter_version": "0.60",
        "atomic_write_protocol": "guard-two-row-v1",
        "deletion_policy": "retained-topology-v1",
        "maintenance": "streamed-rebuild-v1",
        "options": {
            "degree": options.degree,
            "build_list_size": options.build_list_size,
            "search_list_size": options.search_list_size,
            "alpha": options.alpha,
            "cache_bytes": options.cache_bytes,
            "max_visits": options.max_visits,
        }
    })
}

fn validate_descriptor(bytes: &[u8], expected: &serde_json::Value) -> Result<()> {
    if bytes.len() > MAX_DESCRIPTOR_BYTES {
        return Err(corrupt("DiskANN descriptor is too large"));
    }
    let actual: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|_| corrupt("DiskANN descriptor is invalid JSON"))?;
    if &actual != expected {
        return Err(corrupt(
            "DiskANN descriptor does not match the virtual-table declaration",
        ));
    }
    Ok(())
}

fn node_identity(
    statement: &Statement<'_>,
    state_column: usize,
    rowid_column: usize,
) -> Result<(NodeState, Option<u64>)> {
    let state = match nonnegative(statement, state_column, "node state")? {
        0 => NodeState::Live,
        1 => NodeState::Deleted,
        2 => NodeState::Frozen,
        _ => return Err(corrupt("DiskANN node has an unknown state")),
    };
    let rowid = if statement.column_is_null(rowid_column)? {
        None
    } else {
        Some(nonnegative(statement, rowid_column, "public rowid")?)
    };
    if (state == NodeState::Live) != rowid.is_some() {
        return Err(corrupt("DiskANN node state/public rowid are inconsistent"));
    }
    Ok((state, rowid))
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
        if space.vector_type != VectorType::Float32
            || !matches!(space.distance_type, DistanceType::L2 | DistanceType::Cosine)
        {
            return Err(IndexError::new(
                "DiskANN supports only float32 squared L2 or cosine",
            ));
        }
        if space.dim == 0 {
            return Err(IndexError::new(
                "DiskANN dimension must be greater than zero",
            ));
        }
        let vector_bytes = checked_product(space.dim, 4, "vector")?;
        if vector_bytes > i32::MAX as usize {
            return Err(too_big("DiskANN vector exceeds SQLite's BLOB length limit"));
        }
        let adjacency_bytes = checked_product(options.physical_degree(), 8, "adjacency")?;
        // Provider BLOB/decode buffers, one keyset batch, and at least two core
        // vector snapshots must fit before any SQL graph write.
        let minimum = checked_product(vector_bytes, 5, "minimum vector workspace")?
            .checked_add(checked_product(
                adjacency_bytes,
                3,
                "minimum adjacency workspace",
            )?)
            .and_then(|bytes| bytes.checked_add(CONSOLIDATE_BATCH * size_of::<u64>()))
            .ok_or_else(|| too_big("DiskANN minimum workspace overflow"))?;
        if minimum > options.cache_bytes {
            return Err(too_big(
                "DiskANN cache_bytes cannot hold the minimum graph workspace",
            ));
        }
        let expected = descriptor(&space, &options);
        let txn_table = format!("{table}_diskann_txn");
        let rebuild_table = format!("{table}_diskann_rebuild");
        let handle = Self {
            connection,
            schema: schema.to_owned(),
            meta_name: qualified_name(schema, &format!("{table}_diskann_meta"))?,
            nodes_name: qualified_name(schema, &format!("{table}_diskann_nodes"))?,
            txn_name: qualified_name(schema, &txn_table)?,
            txn_table,
            rebuild_name: qualified_name(schema, &rebuild_table)?,
            rebuild_table,
            space,
            options,
            descriptor: expected,
            vector_bytes,
            adjacency_bytes,
        };
        let atomic_minimum = handle
            .transient_workspace()?
            .checked_add(checked_product(
                handle.vector_bytes,
                2,
                "minimum core vectors",
            )?)
            .ok_or_else(|| too_big("DiskANN atomic workspace overflow"))?;
        if atomic_minimum > handle.options.cache_bytes {
            return Err(too_big(
                "DiskANN cache_bytes cannot hold the atomic graph workspace",
            ));
        }
        if is_create {
            handle.require_rollback_journal()?;
            handle.create_schema()?;
        }
        // xConnect never creates tables or scans graph rows. Schema inspection
        // is bounded by the small fixed shadow-table layouts.
        handle.validate_schema(
            &format!("{table}_diskann_meta"),
            &format!("{table}_diskann_nodes"),
        )?;
        let (metadata, anchor) = handle.read_metadata()?;
        handle.verify_guard()?;
        handle.validate_root(&metadata)?;
        drop(metadata);
        anchor.finish()?;
        handle
            .connection
            .prepare(&format!(
                "SELECT node_id,public_rowid,state,{},{} FROM {} LIMIT 0",
                blob_projection("vector", handle.vector_bytes, true),
                blob_projection("neighbors", handle.adjacency_bytes, false),
                handle.nodes_name
            ))?
            .finish()?;
        Ok(handle)
    }

    pub(crate) fn space(&self) -> &NamedVectorSpace {
        &self.space
    }
    pub(crate) fn options(&self) -> &DiskAnnOptions {
        &self.options
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
                    self.txn_name,
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

    fn validate_schema(&self, meta_table: &str, nodes_table: &str) -> Result<()> {
        self.validate_columns(
            meta_table,
            &[
                ("singleton", "INTEGER", 0, 1),
                ("format_version", "INTEGER", 1, 0),
                ("instance_id", "BLOB", 1, 0),
                ("descriptor", "BLOB", 1, 0),
                ("revision", "INTEGER", 1, 0),
                ("next_node_id", "INTEGER", 1, 0),
                ("live_count", "INTEGER", 1, 0),
                ("deleted_count", "INTEGER", 1, 0),
                ("entrypoints", "BLOB", 1, 0),
            ],
        )?;
        self.validate_columns(
            nodes_table,
            &[
                ("node_id", "INTEGER", 0, 1),
                ("public_rowid", "INTEGER", 0, 0),
                ("state", "INTEGER", 1, 0),
                ("vector", "BLOB", 1, 0),
                ("neighbors", "BLOB", 1, 0),
            ],
        )?;
        let mut indexes = self.connection.prepare(
            "SELECT length(name),name,\"unique\",origin,partial FROM pragma_index_list(?1,?2)",
        )?;
        indexes.bind_text(1, nodes_table)?;
        indexes.bind_text(2, &self.schema)?;
        if indexes.step()? != Step::Row
            || nonnegative(&indexes, 0, "index name length")? > (nodes_table.len() + 64) as u64
            || nonnegative(&indexes, 2, "index uniqueness")? != 1
            || indexes.column_text(3)? != "u"
            || nonnegative(&indexes, 4, "partial index flag")? != 0
        {
            return Err(corrupt(
                "DiskANN public rowid requires its automatic UNIQUE index",
            ));
        }
        let index_name = indexes.column_text(1)?;
        if indexes.step()? != Step::Done {
            return Err(corrupt("DiskANN nodes have an unexpected additional index"));
        }
        indexes.finish()?;
        let mut columns = self
            .connection
            .prepare("SELECT seqno,cid,length(name),name FROM pragma_index_info(?1,?2)")?;
        columns.bind_text(1, &index_name)?;
        columns.bind_text(2, &self.schema)?;
        if columns.step()? != Step::Row
            || nonnegative(&columns, 0, "index sequence")? != 0
            || nonnegative(&columns, 1, "index column")? != 1
            || nonnegative(&columns, 2, "index column name length")? != 12
            || columns.column_text(3)? != "public_rowid"
            || columns.step()? != Step::Done
        {
            return Err(corrupt(
                "DiskANN automatic index does not uniquely map public rowids",
            ));
        }
        columns.finish()?;
        Ok(())
    }

    fn validate_columns(&self, table: &str, expected: &[(&str, &str, u64, u64)]) -> Result<()> {
        let mut columns = self.connection.prepare(
            "SELECT cid,length(name),name,length(type),type,\"notnull\",pk,hidden FROM pragma_table_xinfo(?1,?2) ORDER BY cid"
        )?;
        columns.bind_text(1, table)?;
        columns.bind_text(2, &self.schema)?;
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
            quote_identifier(&self.schema)?
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
        kind.bind_text(1, &self.txn_table)?;
        kind.bind_text(2, &self.schema)?;
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
            &self.txn_table,
            &[("singleton", "INTEGER", 0, 1), ("value", "INTEGER", 1, 0)],
        )?;
        for pragma in ["pragma_index_list", "pragma_foreign_key_list"] {
            let mut relations = self
                .connection
                .prepare(&format!("SELECT 1 FROM {pragma}(?1,?2) LIMIT 1"))?;
            relations.bind_text(1, &self.txn_table)?;
            relations.bind_text(2, &self.schema)?;
            if relations.step()? != Step::Done {
                return Err(corrupt(
                    "DiskANN atomic guard cannot have indices or foreign keys",
                ));
            }
            relations.finish()?;
        }
        let mut rows = self.connection.prepare(&format!(
            "SELECT singleton,value FROM {} ORDER BY singleton LIMIT 3",
            self.txn_name
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
        self.verify_rebuild_shadow()?;
        // TEMP triggers can target a non-TEMP table. Conservatively reject the
        // matching base name in both schemas before EVERY carrier preparation.
        self.reject_guard_triggers(&self.schema)?;
        if !self.schema.eq_ignore_ascii_case("temp") {
            self.reject_guard_triggers("temp")?;
        }
        Ok(())
    }

    fn verify_rebuild_shadow(&self) -> Result<()> {
        let mut kind = self.connection.prepare(
            "SELECT type,ncol,wr,strict FROM pragma_table_list(?1) WHERE schema=?2 COLLATE NOCASE",
        )?;
        kind.bind_text(1, &self.rebuild_table)?;
        kind.bind_text(2, &self.schema)?;
        if kind.step()? != Step::Row
            || !matches!(kind.column_text(0)?.as_str(), "table" | "shadow")
            || nonnegative(&kind, 1, "rebuild column count")? != 2
            || nonnegative(&kind, 2, "rebuild WITHOUT ROWID flag")? != 0
            || nonnegative(&kind, 3, "rebuild STRICT flag")? != 0
            || kind.step()? != Step::Done
        {
            return Err(corrupt(
                "DiskANN rebuild staging must be an ordinary rowid shadow table",
            ));
        }
        kind.finish()?;
        self.validate_columns(
            &self.rebuild_table,
            &[("public_rowid", "INTEGER", 0, 1), ("vector", "BLOB", 1, 0)],
        )?;
        self.verify_rebuild_empty()
    }

    fn verify_rebuild_empty(&self) -> Result<()> {
        let mut staging = self
            .connection
            .prepare(&format!("SELECT 1 FROM {} LIMIT 1", self.rebuild_name))?;
        if staging.step()? != Step::Done {
            return Err(corrupt(
                "DiskANN rebuild staging must be empty outside consolidation",
            ));
        }
        staging.finish()
    }

    fn reject_guard_triggers(&self, schema: &str) -> Result<()> {
        let mut trigger = self.connection.prepare(&format!(
            "SELECT 1 FROM {} WHERE type='trigger' AND tbl_name=?1 COLLATE NOCASE LIMIT 1",
            qualified_name(schema, "sqlite_schema")?
        ))?;
        trigger.bind_text(1, &self.txn_table)?;
        if trigger.step()? != Step::Done {
            return Err(corrupt(
                "DiskANN atomic guard cannot have schema or TEMP triggers",
            ));
        }
        trigger.finish()
    }

    fn create_schema(&self) -> Result<()> {
        self.connection.execute(&format!(
            "CREATE TABLE {} (singleton INTEGER PRIMARY KEY CHECK(singleton=1),format_version INTEGER NOT NULL,instance_id BLOB NOT NULL CHECK(length(instance_id)=16),descriptor BLOB NOT NULL,revision INTEGER NOT NULL CHECK(revision>=0),next_node_id INTEGER NOT NULL CHECK(next_node_id>=1),live_count INTEGER NOT NULL CHECK(live_count>=0),deleted_count INTEGER NOT NULL CHECK(deleted_count>=0),entrypoints BLOB NOT NULL)",
            self.meta_name
        ))?;
        self.connection.execute(&format!(
            "CREATE TABLE {} (node_id INTEGER PRIMARY KEY CHECK(node_id>=0),public_rowid INTEGER UNIQUE CHECK(public_rowid>=0),state INTEGER NOT NULL CHECK(state IN (0,1,2)),vector BLOB NOT NULL CHECK(length(vector)={}),neighbors BLOB NOT NULL CHECK(length(neighbors)<={} AND length(neighbors)%8=0),CHECK((state=0 AND public_rowid IS NOT NULL) OR (state IN (1,2) AND public_rowid IS NULL)))",
            self.nodes_name, self.vector_bytes, self.adjacency_bytes
        ))?;
        self.connection.execute(&format!(
            "CREATE TABLE {} (singleton INTEGER PRIMARY KEY CHECK(singleton IN (1,2)),value INTEGER NOT NULL CHECK(value=0))",
            self.txn_name
        ))?;
        self.connection.execute(&format!(
            "INSERT INTO {} (singleton,value) VALUES (1,0),(2,0)",
            self.txn_name
        ))?;
        if self.connection.changes() != 2 {
            return Err(corrupt(
                "DiskANN atomic guard initialization did not affect two rows",
            ));
        }
        self.connection.execute(&format!(
            "CREATE TABLE {} (public_rowid INTEGER PRIMARY KEY CHECK(public_rowid>=0),vector BLOB NOT NULL CHECK(length(vector)={}))",
            self.rebuild_name, self.vector_bytes
        ))?;
        let bytes = serde_json::to_vec(&self.descriptor)
            .map_err(|error| IndexError::new(error.to_string()))?;
        if bytes.len() > MAX_DESCRIPTOR_BYTES {
            return Err(too_big("DiskANN descriptor is too large"));
        }
        let mut statement = self.connection.prepare(&format!(
            "INSERT INTO {} (singleton,format_version,instance_id,descriptor,revision,next_node_id,live_count,deleted_count,entrypoints) VALUES (1,{FORMAT_VERSION},?1,?2,0,1,0,0,?3)", self.meta_name
        ))?;
        statement.bind_blob(1, &self.connection.random_instance_id())?;
        statement.bind_blob(2, &bytes)?;
        statement.bind_blob(3, &[])?;
        done(statement)?;
        self.expect_one_change("metadata initialization")
    }

    fn read_metadata(&self) -> Result<(Metadata, Statement<'static>)> {
        let mut anchor = self.connection.prepare(&format!(
            "SELECT format_version,{instance},{descriptor},revision,next_node_id,live_count,deleted_count,{entrypoints},(SELECT count(*) FROM (SELECT singleton FROM {table} LIMIT 2)) FROM {table} WHERE singleton=1",
            instance = blob_projection("instance_id", 16, true),
            descriptor = blob_projection("descriptor", MAX_DESCRIPTOR_BYTES, false),
            entrypoints = blob_projection("entrypoints", self.adjacency_bytes, false),
            table = self.meta_name
        ))?;
        if anchor.step()? != Step::Row {
            return Err(corrupt("DiskANN singleton metadata is missing"));
        }
        let metadata = self.metadata_row(&anchor)?;
        Ok((metadata, anchor))
    }

    fn metadata_row(&self, anchor: &Statement<'_>) -> Result<Metadata> {
        if nonnegative(anchor, 11, "metadata row count")? != 1 {
            return Err(corrupt(
                "DiskANN metadata must contain exactly one singleton row",
            ));
        }
        if nonnegative(anchor, 0, "format version")? != FORMAT_VERSION as u64 {
            return Err(corrupt("DiskANN format version is unsupported"));
        }
        bounded_blob(anchor, 1, 2, 16, true, "instance ID")?;
        let bytes = bounded_blob(anchor, 3, 4, MAX_DESCRIPTOR_BYTES, false, "descriptor")?;
        validate_descriptor(&bytes, &self.descriptor)?;
        let metadata = Metadata {
            revision: nonnegative(anchor, 5, "revision")?,
            next_node_id: nonnegative(anchor, 6, "next node ID")?,
            live_count: nonnegative(anchor, 7, "live count")?,
            deleted_count: nonnegative(anchor, 8, "deleted count")?,
            entrypoints: decode_ids(
                &bounded_blob(anchor, 9, 10, self.adjacency_bytes, false, "entrypoints")?,
                self.options.physical_degree(),
                None,
            )?,
        };
        if metadata.next_node_id == 0
            || metadata
                .live_count
                .checked_add(metadata.deleted_count)
                .is_none_or(|count| count >= metadata.next_node_id)
            || (metadata.entrypoints.is_empty()
                && (metadata.next_node_id != 1
                    || metadata.live_count != 0
                    || metadata.deleted_count != 0))
            || (!metadata.entrypoints.is_empty() && metadata.entrypoints != [0])
        {
            return Err(corrupt(
                "DiskANN metadata counters/entrypoints are inconsistent",
            ));
        }
        Ok(metadata)
    }

    fn validate_root(&self, metadata: &Metadata) -> Result<()> {
        if metadata.entrypoints.is_empty() {
            return Ok(());
        }
        // Open validates only the permanent root and its bounded adjacency,
        // never the graph population. Keep the metadata anchor at Row so these
        // checks use exactly the snapshot whose descriptor was validated.
        let mut store = SqliteGraphStore::new(self);
        if store.state(0)? != NodeState::Frozen {
            return Err(corrupt("DiskANN entrypoint is not frozen"));
        }
        drop(store.vector(0)?); // Exact size and finite LE f32 payload.
        drop(store.neighbors(0)?); // Codec plus existence of at most R neighbors.
        store.finish()
    }

    fn bump_revision(&self) -> Result<()> {
        let (metadata, anchor) = self.read_metadata()?;
        let revision = checked_increment(metadata.revision, "revision")?;
        anchor.finish()?;
        let mut statement = self.connection.prepare(&format!(
            "UPDATE {} SET revision=?1 WHERE singleton=1",
            self.meta_name
        ))?;
        statement.bind_i64(1, checked_id(revision)?)?;
        done(statement)?;
        self.expect_one_change("metadata revision")
    }

    fn expect_one_change(&self, operation: &str) -> Result<()> {
        if self.connection.changes() != 1 {
            return Err(corrupt(format!(
                "DiskANN {operation} did not affect exactly one row"
            )));
        }
        Ok(())
    }

    fn encode_input(&self, vector: &[f32]) -> Result<Vec<f32>> {
        if vector.len() != self.space.dim {
            return Err(IndexError::new(format!(
                "dimension mismatch: expected {}, got {}",
                self.space.dim,
                vector.len()
            )));
        }
        let mut encoded = Vec::new();
        reserve(&mut encoded, vector.len())?;
        encoded.extend_from_slice(vector);
        normalize_input(&mut encoded, self.space.dim, self.space.distance_type)?;
        Ok(encoded)
    }

    fn graph_config(&self, available_bytes: usize) -> Result<GraphConfig> {
        let max_cached_vectors = available_bytes / self.vector_bytes;
        if max_cached_vectors < 2 {
            return Err(too_big("DiskANN operation exceeds cache_bytes"));
        }
        Ok(GraphConfig {
            dimension: self.space.dim,
            distance: self.space.distance_type,
            max_degree: self.options.degree,
            construction_l: self.options.build_list_size,
            search_l: self.options.search_list_size,
            alpha: self.options.alpha,
            limits: ResourceLimits {
                max_visits: self.options.max_visits,
                max_cached_vectors,
                max_vector_bytes: available_bytes,
            },
        })
    }

    fn transient_workspace(&self) -> Result<usize> {
        // Eight fixed plans contain nodes_name and four occurrences contain
        // meta_name. Charge identifier-sized SQL/CString/compiler copies too;
        // a fixed allowance alone undercounts unusually long legal identifiers.
        let plan_names = checked_product(self.nodes_name.len(), 8, "cached node SQL names")?
            .checked_add(checked_product(
                self.meta_name.len(),
                4,
                "cached metadata SQL names",
            )?)
            .ok_or_else(|| too_big("DiskANN cached SQL name workspace overflow"))?;
        let plan_bytes = checked_product(plan_names, 8, "cached SQL name copies")?
            .checked_add(STATEMENT_CACHE_WORKSPACE)
            .ok_or_else(|| too_big("DiskANN cached SQL workspace overflow"))?;
        let guard_bytes = checked_product(self.txn_name.len(), 3, "atomic SQL workspace")?
            .checked_add(ATOMIC_GUARD_WORKSPACE)
            .ok_or_else(|| too_big("DiskANN atomic scope workspace overflow"))?;
        checked_product(self.vector_bytes, 3, "vector workspace")?
            .checked_add(checked_product(
                self.adjacency_bytes,
                3,
                "adjacency workspace",
            )?)
            .and_then(|bytes| bytes.checked_add(CONSOLIDATE_BATCH * size_of::<u64>()))
            .and_then(|bytes| bytes.checked_add(guard_bytes))
            .and_then(|bytes| bytes.checked_add(plan_bytes))
            .ok_or_else(|| too_big("DiskANN transient workspace overflow"))
    }

    fn mutation_config(&self) -> Result<GraphConfig> {
        let reserved = self.transient_workspace()?;
        self.graph_config(
            self.options
                .cache_bytes
                .checked_sub(reserved)
                .ok_or_else(|| too_big("DiskANN mutation exceeds cache_bytes"))?,
        )
    }

    pub(crate) fn contains(&self, rowid: u64) -> Result<bool> {
        Ok(self.lookup_live_id(rowid)?.is_some())
    }

    fn lookup_live_id(&self, rowid: u64) -> Result<Option<u64>> {
        if rowid > MAX_ID {
            return Ok(None);
        }
        let mut statement = self.connection.prepare(&format!(
            "SELECT node_id,state,public_rowid FROM {} WHERE public_rowid=?1",
            self.nodes_name
        ))?;
        statement.bind_i64(1, checked_id(rowid)?)?;
        if statement.step()? == Step::Done {
            statement.finish()?;
            return Ok(None);
        }
        let id = nonnegative(&statement, 0, "node ID")?;
        let (state, stored_rowid) = node_identity(&statement, 1, 2)?;
        if id == 0 || state != NodeState::Live || stored_rowid != Some(rowid) {
            return Err(corrupt("DiskANN public rowid refers to a non-live node"));
        }
        if statement.step()? != Step::Done {
            return Err(corrupt("DiskANN public rowid is not unique"));
        }
        statement.finish()?;
        Ok(Some(id))
    }

    fn live_blob(&self, rowid: u64) -> Result<Option<Vec<u8>>> {
        if rowid > MAX_ID {
            return Ok(None);
        }
        let mut statement = self.connection.prepare(&format!(
            "SELECT node_id,state,public_rowid,{} FROM {} WHERE public_rowid=?1",
            blob_projection("vector", self.vector_bytes, true),
            self.nodes_name
        ))?;
        statement.bind_i64(1, checked_id(rowid)?)?;
        if statement.step()? == Step::Done {
            statement.finish()?;
            return Ok(None);
        }
        let id = nonnegative(&statement, 0, "node ID")?;
        let (state, stored_rowid) = node_identity(&statement, 1, 2)?;
        if id == 0 || state != NodeState::Live || stored_rowid != Some(rowid) {
            return Err(corrupt("DiskANN public rowid refers to a non-live node"));
        }
        let bytes = bounded_blob(&statement, 3, 4, self.vector_bytes, true, "vector")?;
        if statement.step()? != Step::Done {
            return Err(corrupt("DiskANN public rowid is not unique"));
        }
        statement.finish()?;
        Ok(Some(bytes))
    }

    pub(crate) fn insert(&self, rowid: u64, vector: &[f32]) -> Result<()> {
        checked_id(rowid)?;
        let encoded = self.encode_input(vector)?;
        let config = self.mutation_config()?;
        // Validate revision before graph writes; late errors are still rolled
        // back by the parent callback's SQLite statement/transaction.
        let (metadata, anchor) = self.read_metadata()?;
        checked_increment(metadata.revision, "revision")?;
        anchor.finish()?;
        let mut store = SqliteGraphStore::new(self);
        diskann_core::insert(&mut store, &config, rowid, &encoded)?;
        store.finish()?;
        self.bump_revision()
    }

    /// Runs every chunk inside the caller's ONE atomic_write action. No semantic
    /// state survives this callback, and any late error aborts the whole INSERT.
    pub(crate) fn insert_batch(&self, batch: &BatchView<'_>, batch_size: usize) -> Result<()> {
        diskann_core::check_batch_execution_context()?;
        if !(1..=MAX_INSERT_BATCH).contains(&batch_size) || batch.dimension() != self.space.dim {
            return Err(IndexError::new("invalid DiskANN batch size or dimension"));
        }
        if batch.len() == 0 {
            return Ok(());
        }
        let config = self.mutation_config()?;
        let (metadata, anchor) = self.read_metadata()?;
        let (expected_next, expected_live, expected_revision) =
            batch_preflight(&metadata, batch.len())?;
        self.validate_root(&metadata)?;
        anchor.finish()?;
        let mut store = SqliteGraphStore::new(self);
        let mut offset = 0;
        if metadata.live_count == 0 {
            // Existing single insertion owns frozen0 bootstrap/canonical storage.
            // Do not normalize this point a second time inside allocate/core.
            let rowid = batch.rowid(0)?;
            checked_id(rowid)?;
            let encoded = self.encode_input(batch.vector(0))?;
            diskann_core::insert(&mut store, &config, rowid, &encoded)?;
            offset = 1;
        }
        while offset < batch.len() {
            let count = batch_size.min(batch.len() - offset);
            let matrix_len = checked_product(count, self.space.dim, "batch matrix")?;
            let input_bytes = checked_product(matrix_len, size_of::<f32>(), "batch matrix")?
                .checked_add(checked_product(count, size_of::<u64>(), "batch rowids")?)
                .ok_or_else(|| too_big("DiskANN batch input workspace overflow"))?;
            if input_bytes > config.limits.max_vector_bytes {
                return Err(too_big("DiskANN batch input exceeds cache_bytes"));
            }
            let mut rowids = Vec::new();
            let mut encoded = Vec::new();
            reserve(&mut rowids, count)?;
            reserve(&mut encoded, matrix_len)?;
            for index in offset..offset + count {
                let rowid = batch.rowid(index)?;
                checked_id(rowid)?;
                if rowids.contains(&rowid) || store.lookup_live_id(rowid)?.is_some() {
                    return Err(IndexError::with_code(
                        ffi::SQLITE_CONSTRAINT as i32,
                        "DiskANN rowid already exists in batch or graph",
                    ));
                }
                rowids.push(rowid);
                let start = encoded.len();
                encoded.extend_from_slice(batch.vector(index));
                normalize_input(
                    &mut encoded[start..],
                    self.space.dim,
                    self.space.distance_type,
                )?;
            }
            // One max_visits allowance across all tasks in this true chunk.
            // Core owns and counts rowids, encoded matrix, and algorithm scratch.
            store.visits = 0;
            diskann_core::insert_batch(&mut store, &config, rowids, encoded)?;
            offset += count;
        }
        let final_metadata = store.read_metadata()?;
        if final_metadata.next_node_id != expected_next
            || final_metadata.live_count != expected_live
            || final_metadata.deleted_count != metadata.deleted_count
            || final_metadata.revision != metadata.revision
        {
            return Err(corrupt(
                "DiskANN batch allocation counters changed unexpectedly",
            ));
        }
        store.finish()?;
        self.bump_revision()?;
        let (final_metadata, anchor) = self.read_metadata()?;
        if final_metadata.revision != expected_revision {
            return Err(corrupt("DiskANN batch revision was not updated"));
        }
        anchor.finish()
    }

    pub(crate) fn update(&self, old_rowid: u64, new_rowid: u64, vector: &[f32]) -> Result<()> {
        checked_id(old_rowid)?;
        checked_id(new_rowid)?;
        if old_rowid != new_rowid {
            return Err(IndexError::new("DiskANN rowid cannot be changed by UPDATE"));
        }
        let encoded = self.encode_input(vector)?;
        let config = self.mutation_config()?;
        let (metadata, anchor) = self.read_metadata()?;
        checked_increment(metadata.revision, "revision")?;
        // Preflight counts/ID exhaustion before the first delete write.
        checked_increment(metadata.next_node_id, "next node ID")?;
        checked_increment(metadata.deleted_count, "deleted count")?;
        anchor.finish()?;
        if self.lookup_live_id(old_rowid)?.is_none() {
            return Err(IndexError::new("DiskANN update rowid is absent"));
        }
        if new_rowid != old_rowid && self.contains(new_rowid)? {
            return Err(IndexError::with_code(
                ffi::SQLITE_CONSTRAINT as i32,
                "DiskANN rowid already exists",
            ));
        }
        let mut store = SqliteGraphStore::new(self);
        diskann_core::delete(&mut store, &config, old_rowid)?;
        diskann_core::insert(&mut store, &config, new_rowid, &encoded)?;
        store.finish()?;
        self.bump_revision()
    }

    pub(crate) fn mark_delete(&self, rowid: u64) -> Result<()> {
        checked_id(rowid)?;
        let (metadata, anchor) = self.read_metadata()?;
        checked_increment(metadata.revision, "revision")?;
        anchor.finish()?;
        let config = self.mutation_config()?;
        let mut store = SqliteGraphStore::new(self);
        diskann_core::delete(&mut store, &config, rowid)?;
        store.finish()?;
        self.bump_revision()
    }
}

fn done(mut statement: Statement<'_>) -> Result<()> {
    // Our DML has no RETURNING clause, but PRAGMA count_changes=ON emits a
    // count Row before completion. Drain it rather than manufacturing a late
    // error whose finalize could otherwise commit successful nested writes.
    while statement.step()? == Step::Row {}
    statement.finish()
}

impl GraphStore for SqliteGraphStore<'_> {
    fn allocate(&mut self, rowid: u64, encoded: &[f32]) -> Result<u64> {
        checked_id(rowid)?;
        let table = self.table;
        if encoded.len() != table.space.dim || encoded.iter().any(|value| !value.is_finite()) {
            return Err(IndexError::new(
                "DiskANN allocation requires an already encoded finite vector",
            ));
        }
        if self.lookup_live_id(rowid)?.is_some() {
            return Err(IndexError::with_code(
                ffi::SQLITE_CONSTRAINT as i32,
                "DiskANN rowid already exists",
            ));
        }
        let bytes = encode_vector(encoded)?;
        let metadata = self.read_metadata()?;
        let next_node_id = checked_increment(metadata.next_node_id, "next node ID")?;
        let live_count = checked_increment(metadata.live_count, "live count")?;
        if metadata.entrypoints.is_empty() {
            self.with_statement(GraphStatement::Frozen, |statement| {
                statement.bind_blob(1, &bytes)?;
                statement.bind_blob(2, &[])
            })?;
            table.expect_one_change("frozen node insertion")?;
        } else if self.state(0)? != NodeState::Frozen {
            return Err(corrupt(
                "DiskANN frozen entrypoint is missing or not frozen",
            ));
        }
        let id = metadata.next_node_id;
        self.with_statement(GraphStatement::InsertNode, |statement| {
            statement.bind_i64(1, checked_id(id)?)?;
            statement.bind_i64(2, checked_id(rowid)?)?;
            statement.bind_blob(3, &bytes)?;
            statement.bind_blob(4, &[])
        })?;
        table.expect_one_change("live node insertion")?;
        self.with_statement(GraphStatement::AllocateMetadata, |statement| {
            statement.bind_i64(1, checked_id(next_node_id)?)?;
            statement.bind_i64(2, checked_id(live_count)?)?;
            statement.bind_blob(3, &0u64.to_le_bytes())
        })?;
        table.expect_one_change("allocation metadata")?;
        Ok(id)
    }

    fn internal_id(&mut self, rowid: u64) -> Result<u64> {
        self.lookup_live_id(rowid)?
            .ok_or_else(|| IndexError::new("DiskANN rowid is absent"))
    }
    fn rowid(&mut self, id: u64) -> Result<Option<u64>> {
        Ok(self.identity(id)?.1)
    }
    fn state(&mut self, id: u64) -> Result<NodeState> {
        Ok(self.identity(id)?.0)
    }

    fn vector(&mut self, id: u64) -> Result<Vec<f32>> {
        let table = self.table;
        table.visit(&mut self.visits)?;
        self.with_statement(GraphStatement::Vector, |statement| {
            statement.bind_i64(1, checked_id(id)?)?;
            if statement.step()? != Step::Row {
                return Err(corrupt("DiskANN referenced node is missing"));
            }
            let (state, _) = node_identity(statement, 0, 1)?;
            if (id == 0) != (state == NodeState::Frozen) {
                return Err(corrupt("DiskANN frozen node ID/state are inconsistent"));
            }
            let bytes = bounded_blob(statement, 2, 3, table.vector_bytes, true, "vector")?;
            decode_vector(&bytes, table.space.dim)
        })
    }

    fn neighbors(&mut self, id: u64) -> Result<Vec<u64>> {
        let table = self.table;
        let neighbors = self.with_statement(GraphStatement::Neighbors, |statement| {
            statement.bind_i64(1, checked_id(id)?)?;
            if statement.step()? != Step::Row {
                return Err(corrupt("DiskANN referenced node is missing"));
            }
            let (state, _) = node_identity(statement, 0, 1)?;
            if (id == 0) != (state == NodeState::Frozen) {
                return Err(corrupt("DiskANN frozen node ID/state are inconsistent"));
            }
            let bytes = bounded_blob(statement, 2, 3, table.adjacency_bytes, false, "neighbors")?;
            decode_ids(&bytes, table.options.physical_degree(), Some(id))
        })?;
        // Deleted nodes remain legal bridges. Check EVERY referenced identity;
        // statement reuse removes reprepare costs, not corruption checks.
        for &neighbor in &neighbors {
            self.identity(neighbor)?;
        }
        Ok(neighbors)
    }

    fn set_neighbors(&mut self, id: u64, neighbors: &[u64]) -> Result<()> {
        checked_id(id)?;
        self.identity(id)?;
        let table = self.table;
        let bytes = encode_ids(neighbors, table.options.physical_degree(), Some(id))?;
        for &neighbor in neighbors {
            self.identity(neighbor)?;
        }
        self.with_statement(GraphStatement::SetNeighbors, |statement| {
            statement.bind_blob(1, &bytes)?;
            statement.bind_i64(2, checked_id(id)?)
        })?;
        table.expect_one_change("neighbor update")
    }

    fn mark_delete(&mut self, rowid: u64) -> Result<()> {
        let id = self.internal_id(rowid)?;
        let table = self.table;
        let metadata = self.read_metadata()?;
        let live_count = metadata
            .live_count
            .checked_sub(1)
            .ok_or_else(|| corrupt("DiskANN live count underflow"))?;
        let deleted_count = checked_increment(metadata.deleted_count, "deleted count")?;
        self.with_statement(GraphStatement::DeleteNode, |statement| {
            statement.bind_i64(1, checked_id(id)?)
        })?;
        table.expect_one_change("delete node")?;
        self.with_statement(GraphStatement::DeleteMetadata, |statement| {
            statement.bind_i64(1, checked_id(live_count)?)?;
            statement.bind_i64(2, checked_id(deleted_count)?)
        })?;
        table.expect_one_change("delete metadata")
    }

    fn start_points(&mut self) -> Result<Vec<u64>> {
        let metadata = self.read_metadata()?;
        for &id in &metadata.entrypoints {
            if self.state(id)? != NodeState::Frozen {
                return Err(corrupt("DiskANN entrypoint is not frozen"));
            }
        }
        Ok(metadata.entrypoints)
    }
}

impl<'a> SqliteGraphStore<'a> {
    fn new(table: &'a DiskAnnTable) -> Self {
        Self {
            table,
            visits: 0,
            statements: std::array::from_fn(|_| None),
        }
    }

    fn sql(&self, kind: GraphStatement) -> String {
        let table = self.table;
        match kind {
            GraphStatement::Identity => format!("SELECT state,public_rowid FROM {} WHERE node_id=?1", table.nodes_name),
            GraphStatement::Vector => format!("SELECT state,public_rowid,{} FROM {} WHERE node_id=?1", blob_projection("vector", table.vector_bytes, true), table.nodes_name),
            GraphStatement::Neighbors => format!("SELECT state,public_rowid,{} FROM {} WHERE node_id=?1", blob_projection("neighbors", table.adjacency_bytes, false), table.nodes_name),
            GraphStatement::SetNeighbors => format!("UPDATE {} SET neighbors=?1 WHERE node_id=?2", table.nodes_name),
            GraphStatement::Lookup => format!("SELECT node_id,state,public_rowid FROM {} WHERE public_rowid=?1", table.nodes_name),
            GraphStatement::Metadata => format!(
                "SELECT format_version,{instance},{descriptor},revision,next_node_id,live_count,deleted_count,{entrypoints},(SELECT count(*) FROM (SELECT singleton FROM {table} LIMIT 2)) FROM {table} WHERE singleton=1",
                instance=blob_projection("instance_id",16,true), descriptor=blob_projection("descriptor",MAX_DESCRIPTOR_BYTES,false),
                entrypoints=blob_projection("entrypoints",table.adjacency_bytes,false), table=table.meta_name),
            GraphStatement::Frozen => format!("INSERT INTO {} (node_id,public_rowid,state,vector,neighbors) VALUES (0,NULL,2,?1,?2)", table.nodes_name),
            GraphStatement::InsertNode => format!("INSERT INTO {} (node_id,public_rowid,state,vector,neighbors) VALUES (?1,?2,0,?3,?4)", table.nodes_name),
            GraphStatement::AllocateMetadata => format!("UPDATE {} SET next_node_id=?1,live_count=?2,entrypoints=?3 WHERE singleton=1",table.meta_name),
            GraphStatement::DeleteNode => format!("UPDATE {} SET state=1,public_rowid=NULL WHERE node_id=?1 AND state=0",table.nodes_name),
            GraphStatement::DeleteMetadata => format!("UPDATE {} SET live_count=?1,deleted_count=?2 WHERE singleton=1",table.meta_name),
        }
    }

    fn with_statement<T>(
        &mut self,
        kind: GraphStatement,
        action: impl FnOnce(&mut Statement<'static>) -> Result<T>,
    ) -> Result<T> {
        let slot = kind as usize;
        let mut statement = match self.statements[slot].take() {
            Some(statement) => statement,
            None => self.table.connection.prepare(&self.sql(kind))?,
        };
        // Copy the primary error BEFORE reset/finalize can overwrite SQLite's
        // extended error. An unsuccessful plan is never returned to the cache.
        let result = (|| {
            let value = action(&mut statement)?;
            while statement.step()? == Step::Row {} // includes count_changes
            statement.reset()?;
            statement.clear_bindings()?; // release transient BLOBs between calls
            Ok(value)
        })();
        match result {
            Ok(value) => {
                self.statements[slot] = Some(statement);
                Ok(value)
            }
            Err(error) => {
                let _ = statement.finish();
                Err(error)
            }
        }
    }

    fn finish(mut self) -> Result<()> {
        // Finalize ALL statements, but report the first cleanup failure inside
        // the carrier action rather than silently ignoring deferred errors.
        let mut error = None;
        for slot in &mut self.statements {
            if let Some(statement) = slot.take() {
                if let Err(failure) = statement.finish() {
                    error.get_or_insert(failure);
                }
            }
        }
        error.map_or(Ok(()), Err)
    }

    fn identity(&mut self, id: u64) -> Result<(NodeState, Option<u64>)> {
        self.with_statement(GraphStatement::Identity, |statement| {
            statement.bind_i64(1, checked_id(id)?)?;
            if statement.step()? != Step::Row {
                return Err(corrupt("DiskANN referenced node is missing"));
            }
            let identity = node_identity(statement, 0, 1)?;
            if (id == 0) != (identity.0 == NodeState::Frozen) {
                return Err(corrupt("DiskANN frozen node ID/state are inconsistent"));
            }
            Ok(identity)
        })
    }

    fn lookup_live_id(&mut self, rowid: u64) -> Result<Option<u64>> {
        if rowid > MAX_ID {
            return Ok(None);
        }
        self.with_statement(GraphStatement::Lookup, |statement| {
            statement.bind_i64(1, checked_id(rowid)?)?;
            if statement.step()? == Step::Done {
                return Ok(None);
            }
            let id = nonnegative(statement, 0, "node ID")?;
            let (state, stored_rowid) = node_identity(statement, 1, 2)?;
            if id == 0 || state != NodeState::Live || stored_rowid != Some(rowid) {
                return Err(corrupt("DiskANN public rowid refers to a non-live node"));
            }
            if statement.step()? != Step::Done {
                return Err(corrupt("DiskANN public rowid is not unique"));
            }
            Ok(Some(id))
        })
    }

    fn read_metadata(&mut self) -> Result<Metadata> {
        let table = self.table;
        self.with_statement(GraphStatement::Metadata, |statement| {
            if statement.step()? != Step::Row {
                return Err(corrupt("DiskANN singleton metadata is missing"));
            }
            table.metadata_row(statement)
        })
    }
}

// Query and lifecycle methods follow below; no retained semantic state is added.

#[derive(Debug)]
struct Candidate {
    result: SearchResult,
    vector: Option<Vec<u8>>,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.result
            .distance
            .total_cmp(&other.result.distance)
            .then_with(|| self.result.rowid.cmp(&other.result.rowid))
    }
}

impl DiskAnnTable {
    fn result_budget(
        &self,
        count: usize,
        project_vector: bool,
        topk: bool,
        filter: &SearchFilter<'_>,
    ) -> Result<usize> {
        let per_result = size_of::<SearchResult>()
            .checked_add(if topk { size_of::<Candidate>() } else { 0 })
            .and_then(|bytes| {
                bytes.checked_add(if project_vector {
                    self.vector_bytes + size_of::<Vec<u8>>()
                } else {
                    0
                })
            })
            .ok_or_else(|| too_big("DiskANN result size overflow"))?;
        let results = checked_product(count, per_result, "results")?;
        let transient = self.transient_workspace()?;
        let filter_bytes = match filter {
            SearchFilter::In(rowids) => checked_product(rowids.capacity(), 64, "filter")?,
            SearchFilter::None | SearchFilter::Equals(_) => 0,
        };
        let reserved = results
            .checked_add(transient)
            .and_then(|bytes| bytes.checked_add(filter_bytes))
            .ok_or_else(|| too_big("DiskANN query workspace overflow"))?;
        let available = self
            .options
            .cache_bytes
            .checked_sub(reserved)
            .ok_or_else(|| too_big("DiskANN results exceed cache_bytes"))?;
        if available < self.vector_bytes * 2 {
            return Err(too_big(
                "DiskANN results leave insufficient graph workspace",
            ));
        }
        Ok(available)
    }

    pub(crate) fn knn(
        &self,
        query: &[f32],
        k: usize,
        override_l: Option<usize>,
        filter: SearchFilter<'_>,
        project_vector: bool,
    ) -> Result<QueryRows> {
        let query = self.encode_input(query)?;
        if override_l.is_some_and(|limit| limit == 0 || limit > self.options.max_visits) {
            return Err(IndexError::new(
                "DiskANN search list override must be positive and no greater than max_visits",
            ));
        }
        let (metadata, anchor) = self.read_metadata()?;
        let live_count = usize::try_from(metadata.live_count)
            .map_err(|_| too_big("DiskANN live population exceeds addressable memory"))?;
        let cardinality = match &filter {
            SearchFilter::None => live_count,
            SearchFilter::Equals(_) => 1,
            SearchFilter::In(rowids) => rowids.len(),
        };
        let k = k.min(live_count).min(cardinality);
        let available = self.result_budget(k, project_vector, true, &filter)?;
        let mut store = SqliteGraphStore::new(self);
        let (results, vectors) = if k == 0 {
            (Vec::new(), project_vector.then(Vec::new))
        } else if k >= live_count
            || !matches!(&filter, SearchFilter::None) && cardinality <= EXACT_FILTER_LIMIT
        {
            self.exact_topk(&query, k, &filter, project_vector, store.visits)?
        } else {
            let config = self.graph_config(available)?;
            let graph_filter = match &filter {
                SearchFilter::None => None,
                SearchFilter::In(rowids) => Some(*rowids),
                SearchFilter::Equals(_) => {
                    return Err(corrupt(
                        "DiskANN single-row filter unexpectedly reached graph search",
                    ));
                }
            };
            let mut results =
                diskann_core::search(&mut store, &config, &query, k, override_l, graph_filter)?;
            if results.len() > k
                || results
                    .iter()
                    .any(|row| !row.distance.is_finite() || row.rowid > MAX_ID)
            {
                return Err(corrupt("DiskANN graph returned invalid search results"));
            }
            // Detect duplicate public IDs without an additional result-sized set
            // or a quadratic pairwise check.
            results.sort_unstable_by_key(|row| row.rowid);
            if results
                .windows(2)
                .any(|pair| pair[0].rowid == pair[1].rowid)
            {
                return Err(corrupt("DiskANN graph returned duplicate rows"));
            }
            if graph_filter.is_some() && results.len() < k {
                // Do not retain approximate results alongside the exact heap.
                drop(results);
                self.exact_topk(&query, k, &filter, project_vector, store.visits)?
            } else {
                results.sort_unstable_by(|a, b| {
                    a.distance
                        .total_cmp(&b.distance)
                        .then_with(|| a.rowid.cmp(&b.rowid))
                });
                let mut vectors = project_vector.then(Vec::new);
                if let Some(vectors) = &mut vectors {
                    reserve(vectors, results.len())?;
                }
                for result in &results {
                    if graph_filter.is_some_and(|allowed| !allowed.contains(&result.rowid)) {
                        return Err(corrupt("DiskANN graph returned a row outside the filter"));
                    }
                    if let Some(vectors) = &mut vectors {
                        let blob = self.live_blob(result.rowid)?.ok_or_else(|| {
                            corrupt("DiskANN result row is absent from its snapshot")
                        })?;
                        // Validate even projections; SQLite may contain a malformed
                        // finite/length payload independently of graph adjacency.
                        decode_vector(&blob, self.space.dim)?;
                        vectors.push(blob);
                    }
                }
                (results, vectors)
            }
        };
        store.finish()?;
        Ok(QueryRows {
            results,
            vectors,
            anchor,
        })
    }

    fn distance(&self, query: &[f32], vector: &[f32]) -> Result<f32> {
        let distance = match self.space.distance_type {
            DistanceType::L2 => ops::l2_sq_f32(query, vector),
            DistanceType::Cosine => ops::ip_dist_f32(query, vector),
            DistanceType::InnerProduct => {
                return Err(corrupt("DiskANN descriptor has an unsupported metric"))
            }
        };
        if !distance.is_finite() {
            return Err(corrupt("DiskANN distance calculation overflowed float32"));
        }
        Ok(distance)
    }

    fn exact_topk(
        &self,
        query: &[f32],
        k: usize,
        filter: &SearchFilter<'_>,
        project_vector: bool,
        prior_visits: usize,
    ) -> Result<MaterializedRows> {
        let mut heap = BinaryHeap::new();
        heap.try_reserve_exact(k).map_err(|_| no_memory())?;
        let mut visits = prior_visits;
        let mut consider = |rowid: u64, blob: Vec<u8>| -> Result<()> {
            let vector = decode_vector(&blob, self.space.dim)?;
            let distance = self.distance(query, &vector)?;
            let candidate = Candidate {
                result: SearchResult::new(distance, rowid),
                vector: project_vector.then_some(blob),
            };
            if heap.len() < k {
                heap.push(candidate);
            } else if heap.peek().is_some_and(|worst| candidate < *worst) {
                // Pop first so heap and projected vectors never hold k+1 rows.
                heap.pop();
                heap.push(candidate);
            }
            Ok(())
        };
        match filter {
            SearchFilter::None => {
                let mut statement = self.connection.prepare(&format!(
                    "SELECT node_id,state,public_rowid,{} FROM {} WHERE state=0 ORDER BY node_id",
                    blob_projection("vector", self.vector_bytes, true),
                    self.nodes_name
                ))?;
                while statement.step()? == Step::Row {
                    self.visit(&mut visits)?;
                    let id = nonnegative(&statement, 0, "node ID")?;
                    let (state, rowid) = node_identity(&statement, 1, 2)?;
                    if id == 0 || state != NodeState::Live {
                        return Err(corrupt(
                            "DiskANN exact scan encountered an invalid live node",
                        ));
                    }
                    let blob = bounded_blob(&statement, 3, 4, self.vector_bytes, true, "vector")?;
                    consider(
                        rowid.ok_or_else(|| corrupt("DiskANN live rowid is missing"))?,
                        blob,
                    )?;
                }
                statement.finish()?;
            }
            SearchFilter::Equals(rowid) => {
                self.visit(&mut visits)?;
                if let Some(blob) = self.live_blob(*rowid)? {
                    consider(*rowid, blob)?;
                }
            }
            SearchFilter::In(rowids) => {
                for &rowid in *rowids {
                    self.visit(&mut visits)?;
                    if let Some(blob) = self.live_blob(rowid)? {
                        consider(rowid, blob)?;
                    }
                }
            }
        }
        let candidates = heap.into_sorted_vec();
        let mut results = Vec::new();
        reserve(&mut results, candidates.len())?;
        let mut vectors = project_vector.then(Vec::new);
        if let Some(vectors) = &mut vectors {
            reserve(vectors, candidates.len())?;
        }
        for candidate in candidates {
            results.push(candidate.result);
            if let Some(vectors) = &mut vectors {
                vectors.push(
                    candidate
                        .vector
                        .ok_or_else(|| corrupt("DiskANN projection is missing"))?,
                );
            }
        }
        Ok((results, vectors))
    }

    fn visit(&self, visits: &mut usize) -> Result<()> {
        *visits = visits
            .checked_add(1)
            .ok_or_else(|| too_big("DiskANN visit count overflow"))?;
        if *visits > self.options.max_visits {
            return Err(too_big("DiskANN operation exceeded max_visits"));
        }
        Ok(())
    }

    pub(crate) fn select_rowids(
        &self,
        filter: SearchFilter<'_>,
        project_vector: bool,
    ) -> Result<QueryRows> {
        let (metadata, anchor) = self.read_metadata()?;
        let count = match &filter {
            SearchFilter::None => {
                return Err(IndexError::new(
                    "DiskANN requires a KNN or rowid constraint",
                ))
            }
            SearchFilter::Equals(_) => 1,
            SearchFilter::In(rowids) => rowids.len(),
        }
        .min(
            usize::try_from(metadata.live_count)
                .map_err(|_| too_big("DiskANN live population exceeds addressable memory"))?,
        );
        self.result_budget(count, project_vector, false, &filter)?;
        let mut results = Vec::new();
        reserve(&mut results, count)?;
        let mut vectors = project_vector.then(Vec::new);
        if let Some(vectors) = &mut vectors {
            reserve(vectors, count)?;
        }
        let mut visits = 0;
        let mut select = |rowid: u64| -> Result<()> {
            self.visit(&mut visits)?;
            // A projected vector is materialized during selection, never looked
            // up later by xColumn after the distance snapshot could change.
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
                        "DiskANN live count is smaller than selected population",
                    ));
                }
                results.push(SearchResult::new(0.0, rowid));
                if let Some(vectors) = &mut vectors {
                    let blob = blob.ok_or_else(|| corrupt("DiskANN projection is missing"))?;
                    decode_vector(&blob, self.space.dim)?;
                    vectors.push(blob);
                }
            }
            Ok(())
        };
        match filter {
            SearchFilter::Equals(rowid) => select(rowid)?,
            SearchFilter::In(rowids) => {
                for &rowid in rowids {
                    select(rowid)?;
                }
            }
            SearchFilter::None => {
                return Err(IndexError::new("DiskANN requires a rowid constraint"))
            }
        }
        // Distance zero matches HNSW's direct-rowid path. Input IN iteration
        // order is deliberately not an SQL ordering promise.
        Ok(QueryRows {
            results,
            vectors,
            anchor,
        })
    }

    /// The parent calls this inside atomic_write from control INSERT's xUpdate.
    /// Tombstones retain their topology until this explicit maintenance command
    /// rebuilds a fresh live graph from SQLite-staged, already encoded vectors.
    /// The carrier rolls back the old graph AND staging on any late failure.
    pub(crate) fn consolidate(&self) -> Result<()> {
        self.verify_rebuild_shadow()?;
        let (metadata, anchor) = self.read_metadata()?;
        let revision = checked_increment(metadata.revision, "revision")?;
        anchor.finish()?;
        if metadata.deleted_count == 0 {
            return self.write_consolidation_revision(revision);
        }
        let retired_count = rebuild_preflight(&metadata)?;
        let config = self.mutation_config()?;
        self.require_count(
            &format!("SELECT count(*) FROM {} WHERE state=0", self.nodes_name),
            metadata.live_count,
            "live rebuild population",
        )?;
        self.require_count(
            &format!("SELECT count(*) FROM {} WHERE state=1", self.nodes_name),
            metadata.deleted_count,
            "deleted rebuild population",
        )?;
        self.require_no_row(&format!(
            "SELECT 1 FROM {} WHERE state=0 AND (node_id=0 OR typeof(public_rowid)!='integer' OR public_rowid<0) LIMIT 1",
            self.nodes_name
        ), "rebuild encountered an invalid live identity")?;

        // SQLite owns the O(N) staging space. CASE uses only record headers for
        // bad type/length, so malformed payloads cannot materialize before bounds.
        // NULL then fails staging's NOT NULL constraint inside the carrier.
        self.connection.execute(&format!(
            "INSERT INTO {} (public_rowid,vector) SELECT public_rowid,CASE WHEN typeof(vector)='blob' AND length(vector)={} THEN vector ELSE NULL END FROM {} WHERE state=0",
            self.rebuild_name, self.vector_bytes, self.nodes_name
        ))?;
        self.require_count(
            &format!("SELECT count(*) FROM {}", self.rebuild_name),
            metadata.live_count,
            "staged rebuild population",
        )?;
        // Validate finite encoded payloads one at a time before retiring any old
        // mappings. In particular cosine vectors are NOT normalized again.
        self.for_each_staged_vector(|_, _| Ok(()))?;
        let mut store = SqliteGraphStore::new(self);
        if store.state(0)? != NodeState::Frozen {
            return Err(corrupt("DiskANN rebuild has no permanent frozen root"));
        }

        self.connection.execute(&format!(
            "UPDATE {} SET state=1,public_rowid=NULL WHERE state=0",
            self.nodes_name
        ))?;
        self.require_count(
            &format!("SELECT count(*) FROM {} WHERE state=0", self.nodes_name),
            0,
            "retired live mappings",
        )?;
        self.require_count(
            &format!("SELECT count(*) FROM {} WHERE state=1", self.nodes_name),
            retired_count,
            "retired rebuild population",
        )?;
        // A fresh graph must not navigate any old live/deleted component. Keep
        // frozen0's encoded vector but remove ALL old adjacency, including its
        // outgoing edges. Old vectors remain present until the new graph verifies.
        self.connection
            .execute(&format!("UPDATE {} SET neighbors=X''", self.nodes_name))?;
        self.require_no_row(&format!(
            "SELECT 1 FROM {} WHERE CASE WHEN typeof(neighbors)='blob' THEN length(neighbors)!=0 ELSE 1 END LIMIT 1",
            self.nodes_name
        ), "rebuild could not clear every old adjacency")?;
        let mut reset = self.connection.prepare(&format!(
            "UPDATE {} SET live_count=0,deleted_count=?1 WHERE singleton=1",
            self.meta_name
        ))?;
        reset.bind_i64(1, checked_id(retired_count)?)?;
        done(reset)?;
        self.expect_one_change("rebuild metadata reset")?;

        self.for_each_staged_vector(|rowid, encoded| {
            // Explicit maintenance is O(N) disk I/O and rebuild CPU, not an O(N)
            // Rust graph. Each insertion independently respects traversal budgets.
            store.visits = 0;
            diskann_core::insert(&mut store, &config, rowid, encoded)
        })?;
        let (rebuilt, anchor) = self.read_metadata()?;
        if rebuilt.live_count != metadata.live_count
            || rebuilt.deleted_count != retired_count
            || rebuilt.next_node_id != metadata.next_node_id + metadata.live_count
        {
            return Err(corrupt(
                "DiskANN rebuilt counters do not match the staged population",
            ));
        }
        anchor.finish()?;
        self.require_count(
            &format!("SELECT count(*) FROM {} WHERE state=0", self.nodes_name),
            metadata.live_count,
            "rebuilt live population",
        )?;
        self.require_no_row(&format!(
            "SELECT 1 FROM {staging} AS staged LEFT JOIN {nodes} AS node ON node.public_rowid=staged.public_rowid AND node.state=0 WHERE node.node_id IS NULL OR CASE WHEN typeof(node.vector)='blob' AND length(node.vector)={bytes} AND typeof(staged.vector)='blob' AND length(staged.vector)={bytes} THEN node.vector!=staged.vector ELSE 1 END LIMIT 1",
            staging=self.rebuild_name, nodes=self.nodes_name, bytes=self.vector_bytes
        ), "rebuilt vectors do not exactly match encoded staging")?;
        self.for_each_live_node(|id| {
            for neighbor in store.neighbors(id)? {
                if store.state(neighbor)? == NodeState::Deleted {
                    return Err(corrupt("DiskANN rebuilt graph references an old tombstone"));
                }
            }
            Ok(())
        })?;
        self.require_no_row(&format!(
            "SELECT 1 FROM {} WHERE state=1 AND CASE WHEN typeof(neighbors)='blob' THEN length(neighbors)!=0 ELSE 1 END LIMIT 1",
            self.nodes_name
        ), "rebuild left nonempty tombstone adjacency")?;
        self.connection
            .execute(&format!("DELETE FROM {} WHERE state=1", self.nodes_name))?;
        self.require_no_row(
            &format!("SELECT 1 FROM {} WHERE state=1 LIMIT 1", self.nodes_name),
            "rebuild could not remove every old tombstone",
        )?;
        self.connection
            .execute(&format!("DELETE FROM {}", self.rebuild_name))?;
        self.verify_rebuild_empty()?;
        store.finish()?;
        self.write_consolidation_revision(revision)
    }

    fn write_consolidation_revision(&self, revision: u64) -> Result<()> {
        let mut statement = self.connection.prepare(&format!(
            "UPDATE {} SET deleted_count=0,revision=?1 WHERE singleton=1",
            self.meta_name
        ))?;
        statement.bind_i64(1, checked_id(revision)?)?;
        done(statement)?;
        self.expect_one_change("consolidation metadata")
    }

    fn require_count(&self, sql: &str, expected: u64, name: &str) -> Result<()> {
        // COUNT is an SQLite i64. changes() cannot validate a maintenance-sized
        // population because its host API returns only i32.
        let mut count = self.connection.prepare(sql)?;
        if count.step()? != Step::Row
            || nonnegative(&count, 0, name)? != expected
            || count.step()? != Step::Done
        {
            return Err(corrupt(format!("DiskANN {name} does not match metadata")));
        }
        count.finish()
    }

    fn require_no_row(&self, sql: &str, message: &str) -> Result<()> {
        let mut check = self.connection.prepare(sql)?;
        if check.step()? != Step::Done {
            return Err(corrupt(format!("DiskANN {message}")));
        }
        check.finish()
    }

    fn for_each_staged_vector(
        &self,
        mut action: impl FnMut(u64, &[f32]) -> Result<()>,
    ) -> Result<()> {
        let mut last = -1i64;
        loop {
            let mut row = self.connection.prepare(&format!(
                "SELECT public_rowid,{} FROM {} WHERE public_rowid>?1 ORDER BY public_rowid LIMIT 1",
                blob_projection("vector", self.vector_bytes, true), self.rebuild_name
            ))?;
            row.bind_i64(1, last)?;
            if row.step()? == Step::Done {
                row.finish()?;
                return Ok(());
            }
            let rowid = nonnegative(&row, 0, "staged public rowid")?;
            let blob = bounded_blob(&row, 1, 2, self.vector_bytes, true, "staged vector")?;
            let encoded = decode_vector(&blob, self.space.dim)?;
            drop(blob);
            row.finish()?;
            action(rowid, &encoded)?;
            last = checked_id(rowid)?;
        }
    }

    fn for_each_live_node(&self, mut action: impl FnMut(u64) -> Result<()>) -> Result<()> {
        let mut last = -1i64;
        loop {
            // Finish each keyset statement before modifying its source table.
            // At most 128 IDs, not the graph, are held in Rust.
            let mut statement = self.connection.prepare(&format!(
                "SELECT node_id FROM {} WHERE node_id>?1 AND state IN (0,2) ORDER BY node_id LIMIT {}", self.nodes_name, CONSOLIDATE_BATCH
            ))?;
            statement.bind_i64(1, last)?;
            let mut batch = Vec::new();
            reserve(&mut batch, CONSOLIDATE_BATCH)?;
            while statement.step()? == Step::Row {
                batch.push(nonnegative(&statement, 0, "node ID")?);
            }
            statement.finish()?;
            if batch.is_empty() {
                return Ok(());
            }
            for id in batch {
                action(id)?;
                last = checked_id(id)?;
            }
        }
    }

    /// The caller must finalize all cursor anchor statements first.
    pub(crate) fn destroy(&self) -> Result<()> {
        self.require_rollback_journal()?;
        self.connection
            .execute(&format!("DROP TABLE {}", self.rebuild_name))?;
        self.connection
            .execute(&format!("DROP TABLE {}", self.txn_name))?;
        self.connection
            .execute(&format!("DROP TABLE {}", self.nodes_name))?;
        self.connection
            .execute(&format!("DROP TABLE {}", self.meta_name))
    }

    /// SQLite reparses the virtual-table declaration after xRename. Do not
    /// mutate this handle's names: a rolled-back rename must not poison it.
    pub(crate) fn rename(&self, new_name: &str) -> Result<()> {
        self.require_rollback_journal()?;
        let new_meta = quote_identifier(&format!("{new_name}_diskann_meta"))?;
        let new_nodes = quote_identifier(&format!("{new_name}_diskann_nodes"))?;
        let new_txn = quote_identifier(&format!("{new_name}_diskann_txn"))?;
        let new_rebuild = quote_identifier(&format!("{new_name}_diskann_rebuild"))?;
        // Validate schema independently, even though ALTER's destination name
        // must be unqualified in SQLite syntax.
        quote_identifier(&self.schema)?;
        self.connection.execute(&format!(
            "ALTER TABLE {} RENAME TO {}",
            self.meta_name, new_meta
        ))?;
        self.connection.execute(&format!(
            "ALTER TABLE {} RENAME TO {}",
            self.nodes_name, new_nodes
        ))?;
        self.connection.execute(&format!(
            "ALTER TABLE {} RENAME TO {}",
            self.txn_name, new_txn
        ))?;
        self.connection.execute(&format!(
            "ALTER TABLE {} RENAME TO {}",
            self.rebuild_name, new_rebuild
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn space(metric: DistanceType) -> NamedVectorSpace {
        NamedVectorSpace {
            vector_name: "embedding".into(),
            dim: 3,
            distance_type: metric,
            vector_type: VectorType::Float32,
        }
    }

    #[test]
    fn payload_projection_rejects_wrong_types_and_bounds_before_copying() {
        let vector = blob_projection("vector", 12, true);
        assert!(vector.contains("THEN length(vector) ELSE -1 END"));
        assert!(vector.contains("typeof(vector)='blob' AND length(vector)=12"));
        assert!(vector.ends_with("THEN vector ELSE NULL END"));
        let neighbors = blob_projection("neighbors", 328, false);
        assert!(neighbors.contains("length(neighbors)<=328"));
        assert!(neighbors.ends_with("THEN neighbors ELSE NULL END"));
    }

    #[test]
    fn vector_codec_is_little_endian_and_finite() {
        let vector = [1.25, -2.5, 0.0];
        let bytes = encode_vector(&vector).unwrap();
        assert_eq!(&bytes[..4], &1.25f32.to_le_bytes());
        assert_eq!(decode_vector(&bytes, 3).unwrap(), vector);
        assert!(decode_vector(&bytes[..11], 3).is_err());
        assert!(decode_vector(&f32::NAN.to_le_bytes(), 1).is_err());
        assert!(decode_vector(&f32::INFINITY.to_le_bytes(), 1).is_err());
    }

    #[test]
    fn adjacency_codec_checks_degree_duplicates_self_and_signed_range() {
        let bytes = encode_ids(&[0, 17, MAX_ID], 3, Some(2)).unwrap();
        assert_eq!(decode_ids(&bytes, 3, Some(2)).unwrap(), [0, 17, MAX_ID]);
        assert!(decode_ids(&bytes, 2, None).is_err());
        assert!(decode_ids(&bytes[..23], 3, None).is_err());
        assert!(encode_ids(&[1, 1], 3, None).is_err());
        assert!(encode_ids(&[3], 3, Some(3)).is_err());
        assert!(encode_ids(&[MAX_ID + 1], 3, None).is_err());
        assert!(decode_ids(&encode_ids(&[0, 1], 3, None).unwrap(), 3, Some(1)).is_err());
    }

    #[test]
    fn descriptor_is_owned_versioned_and_options_sensitive() {
        let options = DiskAnnOptions::default();
        let expected = descriptor(&space(DistanceType::Cosine), &options);
        let bytes = serde_json::to_vec(&expected).unwrap();
        validate_descriptor(&bytes, &expected).unwrap();
        assert!(
            validate_descriptor(&bytes, &descriptor(&space(DistanceType::L2), &options)).is_err()
        );
        let mut changed = options.clone();
        changed.cache_bytes *= 2;
        assert!(
            validate_descriptor(&bytes, &descriptor(&space(DistanceType::Cosine), &changed))
                .is_err()
        );
        assert!(validate_descriptor(b"not-json", &expected).is_err());
        assert!(validate_descriptor(&vec![b' '; MAX_DESCRIPTOR_BYTES + 1], &expected).is_err());
        assert_eq!(expected["adapter_version"], "0.60");
        assert_eq!(expected["format_version"], 3);
        assert_eq!(expected["atomic_write_protocol"], "guard-two-row-v1");
        assert_eq!(expected["deletion_policy"], "retained-topology-v1");
        assert_eq!(expected["maintenance"], "streamed-rebuild-v1");
        assert_eq!(expected["normalize"], true);
        let mut obsolete = expected.clone();
        obsolete["format_version"] = serde_json::json!(1);
        assert!(validate_descriptor(&serde_json::to_vec(&obsolete).unwrap(), &expected).is_err());
        obsolete = expected.clone();
        obsolete["format_version"] = serde_json::json!(2);
        assert!(validate_descriptor(&serde_json::to_vec(&obsolete).unwrap(), &expected).is_err());
        obsolete = expected.clone();
        obsolete["maintenance"] = serde_json::json!("drop-deleted-neighbors");
        assert!(validate_descriptor(&serde_json::to_vec(&obsolete).unwrap(), &expected).is_err());
        obsolete = expected.clone();
        obsolete["atomic_write_protocol"] = serde_json::json!("guard-v1");
        assert!(validate_descriptor(&serde_json::to_vec(&obsolete).unwrap(), &expected).is_err());
    }

    #[test]
    fn prewrite_numeric_ranges_are_finite_and_metric_specific() {
        assert_eq!(
            validate_numeric_range(&[0.0, 0.0], DistanceType::Cosine).unwrap(),
            0.0
        );
        assert!(validate_numeric_range(&[f32::NAN], DistanceType::L2).is_err());
        assert!(validate_numeric_range(&[f32::INFINITY], DistanceType::Cosine).is_err());
        assert!(validate_numeric_range(&[f32::MAX], DistanceType::Cosine).is_err());
        let large = [1.0e19f32];
        assert!(validate_numeric_range(&large, DistanceType::Cosine).is_ok());
        let error = validate_numeric_range(&large, DistanceType::L2).unwrap_err();
        assert!(error.message.contains("squared norm <= f32::MAX/4"));
        assert!(validate_numeric_range(&[8.0e18f32], DistanceType::L2).is_ok());
        assert!(validate_numeric_range(&[1.0], DistanceType::InnerProduct).is_err());
    }

    #[test]
    fn journal_policy_requires_rollback_without_changing_settings() {
        for mode in ["delete", "truncate", "persist", "memory", "wal"] {
            validate_journal_mode(mode).unwrap();
        }
        let error = validate_journal_mode("off").unwrap_err();
        assert!(error.message.contains("journal_mode=OFF"));
        assert!(validate_journal_mode("unknown").is_err());
    }

    #[test]
    fn streamed_rebuild_preflight_never_reuses_or_overflows_node_ids() {
        let mut metadata = Metadata {
            revision: 0,
            next_node_id: 12,
            live_count: 5,
            deleted_count: 6,
            entrypoints: vec![0],
        };
        assert_eq!(rebuild_preflight(&metadata).unwrap(), 11);
        assert_eq!(metadata.next_node_id, 12);
        metadata.next_node_id = MAX_ID - 5;
        rebuild_preflight(&metadata).unwrap();
        metadata.next_node_id += 1;
        assert!(rebuild_preflight(&metadata).is_err());
        metadata.next_node_id = 1;
        metadata.live_count = 0;
        assert_eq!(rebuild_preflight(&metadata).unwrap(), 6);
        metadata.deleted_count = MAX_ID;
        metadata.live_count = 1;
        assert!(rebuild_preflight(&metadata).is_err());
    }

    #[test]
    fn batch_preflight_checks_all_counters_before_writes() {
        let mut metadata = Metadata {
            revision: 7,
            next_node_id: 10,
            live_count: 5,
            deleted_count: 3,
            entrypoints: vec![0],
        };
        assert_eq!(batch_preflight(&metadata, 32).unwrap(), (42, 37, 8));
        metadata.next_node_id = MAX_ID - 32;
        batch_preflight(&metadata, 32).unwrap();
        assert!(batch_preflight(&metadata, 33).is_err());
        metadata.next_node_id = 10;
        metadata.live_count = MAX_ID;
        assert!(batch_preflight(&metadata, 1).is_err());
        metadata.live_count = 5;
        metadata.revision = MAX_ID;
        assert!(batch_preflight(&metadata, 1).is_err());
    }

    #[test]
    fn input_normalization_matches_single_and_keeps_zero_finite() {
        let mut l2 = [1.0, 2.0, 3.0];
        normalize_input(&mut l2, 3, DistanceType::L2).unwrap();
        assert_eq!(l2, [1.0, 2.0, 3.0]);
        let mut cosine = [3.0, 4.0, 0.0];
        let mut expected = cosine;
        ops::normalize_f32(&mut expected);
        normalize_input(&mut cosine, 3, DistanceType::Cosine).unwrap();
        assert_eq!(cosine, expected);
        let mut zero = [0.0; 3];
        normalize_input(&mut zero, 3, DistanceType::Cosine).unwrap();
        assert!(zero.iter().all(|value| value.is_finite()));
        assert!(normalize_input(&mut l2, 2, DistanceType::L2).is_err());
        assert!(normalize_input(&mut [f32::NAN], 1, DistanceType::L2).is_err());
        assert!(normalize_input(&mut [f32::MAX], 1, DistanceType::Cosine).is_err());
    }

    #[test]
    fn checked_size_and_id_boundaries() {
        assert_eq!(checked_id(MAX_ID).unwrap(), i64::MAX);
        assert!(checked_id(MAX_ID + 1).is_err());
        assert!(checked_increment(MAX_ID, "ID").is_err());
        assert!(checked_product(usize::MAX, 4, "vector").is_err());
        assert_eq!(checked_increment(0, "revision").unwrap(), 1);
    }

    #[test]
    fn bounded_heap_retains_deterministic_distance_then_rowid_order() {
        let mut heap = BinaryHeap::new();
        for (distance, rowid) in [(2.0, 8), (1.0, 7), (1.0, 3), (0.5, 9)] {
            let candidate = Candidate {
                result: SearchResult::new(distance, rowid),
                vector: None,
            };
            if heap.len() < 2 {
                heap.push(candidate);
            } else if candidate < *heap.peek().unwrap() {
                heap.pop();
                heap.push(candidate);
            }
        }
        let rows: Vec<_> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|candidate| candidate.result.rowid)
            .collect();
        assert_eq!(rows, [9, 3]);
    }
}
