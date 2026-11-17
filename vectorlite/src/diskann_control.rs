//! Public deferred frontend. Candidate copying precedes the original journal carrier;
//! native construction is reachable only through a standalone transaction-free scalar.
use crate::diskann_build::{self as native, BuildError, BuildLimits, CheckedSnapshot};
use crate::diskann_build_guard::{BuildGuard, OWNED_WORKER_STACK_BYTES};
use crate::diskann_registration::{hex, worker_error, Registration, TargetMetadata};
use crate::diskann_store::{import, jobs, lifecycle};
use crate::diskann_store::{DiskAnnTable, Job};
use crate::diskann_target::{parse_declaration, MAX_DECLARATION_BYTES};
use crate::diskann_worker::{
    AttemptIdentity, AttemptToken, BeginOutcome, Digest32, FenceToken, JobId, SnapshotMeta,
    SnapshotNode,
};
use crate::ffi::{self, Value};
use crate::index_error::IndexError;
use crate::sqlite::{quote_identifier, Connection, Step};
use serde::de::{self, MapAccess, Visitor};
use serde::Deserialize;
use serde_json::{json, Value as Json};
use std::collections::BTreeMap;
use std::os::raw::{c_int, c_void};
use std::sync::Arc;
type Result<T> = std::result::Result<T, IndexError>;
const MAX_JSON_BYTES: usize = 4096;
const MAX_NAME_BYTES: usize = 1024;
const DEFAULT_STEP_BYTES: usize = 1024 * 1024;
fn invalid(s: impl Into<String>) -> IndexError {
    IndexError::new(s)
}
fn too_big(s: &str) -> IndexError {
    IndexError::with_code(ffi::SQLITE_TOOBIG as i32, s)
}
fn native_error(e: BuildError) -> IndexError {
    let code = match &e {
        BuildError::Limit(_) => ffi::SQLITE_TOOBIG,
        BuildError::Allocation => ffi::SQLITE_NOMEM,
        BuildError::Cancelled => ffi::SQLITE_ABORT,
        BuildError::InvalidTopology(_) => ffi::SQLITE_CORRUPT,
        _ => ffi::SQLITE_ERROR,
    };
    IndexError::with_code(code as i32, format!("DiskANN native build: {e}"))
}
fn bounded_text<'a>(value: &'a mut Value<'_>, max: usize, label: &str) -> Result<&'a str> {
    if value.kind() != ffi::SQLITE_TEXT as i32 {
        return Err(invalid(format!("{label} must be TEXT")));
    }
    // SAFETY: protected TEXT is not coerced; length is checked before obtaining its view.
    let n = unsafe { ffi::value_bytes(value.as_ptr()) };
    if n < 0 || n as usize > max {
        return Err(too_big("DiskANN frontend text byte bound"));
    }
    value.text().map_err(invalid)
}
#[derive(Default)]
struct Object(BTreeMap<String, Json>);
impl<'de> Deserialize<'de> for Object {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Object;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a strict JSON object")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut m: M,
            ) -> std::result::Result<Object, M::Error> {
                let mut fields = BTreeMap::new();
                while let Some((key, value)) = m.next_entry::<String, Json>()? {
                    if fields.insert(key, value).is_some() {
                        return Err(de::Error::custom("duplicate JSON key"));
                    }
                }
                Ok(Object(fields))
            }
        }
        d.deserialize_map(V)
    }
}
impl Object {
    fn parse(text: &str, allowed: &[&str]) -> Result<Self> {
        if text.len() > MAX_JSON_BYTES {
            return Err(too_big("DiskANN JSON byte bound"));
        }
        let obj: Self =
            serde_json::from_str(text).map_err(|e| invalid(format!("DiskANN strict JSON: {e}")))?;
        if let Some(k) = obj.0.keys().find(|k| !allowed.contains(&k.as_str())) {
            return Err(invalid(format!("unknown DiskANN JSON key: {k}")));
        }
        Ok(obj)
    }
    fn integer(&self, k: &str, default: Option<u64>) -> Result<u64> {
        let n = match self.0.get(k) {
            Some(v) => v
                .as_u64()
                .ok_or_else(|| invalid(format!("{k} must be an integer")))?,
            None => default.ok_or_else(|| invalid(format!("required DiskANN JSON key: {k}")))?,
        };
        if n > i64::MAX as u64 {
            return Err(too_big("DiskANN integer exceeds SQLite i64"));
        }
        Ok(n)
    }
    fn id(&self, k: &str) -> Result<u64> {
        let n = self.integer(k, None)?;
        if n == 0 {
            return Err(invalid(format!("{k} must be positive")));
        }
        Ok(n)
    }
    fn token(&self) -> Result<&str> {
        self.0
            .get("candidate_token")
            .and_then(Json::as_str)
            .ok_or_else(|| invalid("candidate_token must be TEXT JSON string"))
    }
    fn force(&self) -> Result<bool> {
        match self.0.get("force") {
            None => Ok(false),
            Some(Json::Bool(v)) => Ok(*v),
            _ => Err(invalid("force must be boolean")),
        }
    }
    fn steps(&self, table: &DiskAnnTable) -> Result<jobs::StepOptions> {
        let records = self.integer("max_records", Some(256))?;
        let bytes = self.integer("max_bytes", Some(DEFAULT_STEP_BYTES as u64))?;
        let records = u32::try_from(records).map_err(|_| too_big("max_records"))?;
        let bytes = usize::try_from(bytes).map_err(|_| too_big("max_bytes"))?;
        if bytes > table.options().cache_bytes {
            return Err(too_big("max_bytes exceeds DiskANN owner workspace"));
        }
        jobs::StepOptions::new(records, bytes)
    }
}
fn binding(
    table: &DiskAnnTable,
    registration: &Registration,
    job: u64,
    fence: u64,
) -> Result<jobs::BuildBinding> {
    let j = jobs::active(table, job)?;
    if j.fence != fence || j.owner_nonce != Some(registration.owner) {
        return Err(jobs::stale("current owner/fence required"));
    }
    let b = jobs::BuildBinding {
        instance_id: jobs::metadata(table)?.instance_id,
        job_id: job,
        generation_id: j.generation_id,
        fence,
        owner_nonce: registration.owner,
        source_hash: j
            .source_hash
            .ok_or_else(|| invalid("sealed source required"))?,
        config_hash: j.config_hash,
    };
    jobs::bound(table, &b)?;
    Ok(b)
}
fn fence_check(table: &DiskAnnTable, job: u64, fence: u64, owner: &Registration) -> Result<Job> {
    let j = jobs::active(table, job)?;
    if j.fence != fence || j.owner_nonce.is_some_and(|n| n != owner.owner) {
        return Err(jobs::stale("control current owner/fence required"));
    }
    Ok(j)
}
/// Resolves only after exact declaration grammar validation; no other module/view is executed.
fn resolve(
    connection: Connection<'static>,
    registration: &Arc<Registration>,
    schema: &str,
    name: &str,
) -> Result<(DiskAnnTable, TargetMetadata)> {
    if schema.is_empty() || name.is_empty() || schema.contains('\0') || name.contains('\0') {
        return Err(invalid("nonempty schema/table without NUL required"));
    }
    let mut s=connection.prepare(&format!("SELECT name,typeof(sql),length(CAST(sql AS BLOB)),CASE WHEN typeof(sql)='text' AND length(CAST(sql AS BLOB))<=?2 THEN sql END FROM {}.sqlite_schema WHERE name=?1 COLLATE NOCASE AND type='table' LIMIT 2",quote_identifier(schema)?))?;
    s.bind_text(1, name)?;
    s.bind_i64(2, MAX_DECLARATION_BYTES as i64)?;
    let result = (|| {
        if s.step()? != Step::Row {
            return Err(invalid("DiskANN target table not found"));
        }
        let name = s.column_text(0)?;
        if s.column_text(1)? != "text" {
            return Err(invalid("target declaration is not TEXT"));
        }
        let len = s.column_i64(2)?;
        if len < 0 || len as usize > MAX_DECLARATION_BYTES {
            return Err(too_big("target declaration byte bound"));
        }
        let ddl = s.column_text(3)?;
        if s.step()? != Step::Done {
            return Err(invalid("ambiguous target declaration"));
        }
        let parsed = parse_declaration(&ddl, schema, &name, MAX_DECLARATION_BYTES)
            .map_err(|e| invalid(format!("DiskANN target: {e}")))?;
        Ok((name, parsed))
    })();
    let finish = s.finish();
    let (name, parsed) = result?;
    finish?;
    // Force this proved vectorlite table's xConnect; every reader, including the vtab cursor, closes.
    connection.execute(&format!(
        "SELECT rowid FROM {}.{} WHERE rowid=-1",
        quote_identifier(schema)?,
        quote_identifier(&name)?
    ))?;
    let target = registration.target(schema, &name)?;
    if target.space.vector_name != parsed.vector_space.vector_name
        || target.space.dim != parsed.vector_space.dim
        || target.space.distance_type != parsed.vector_space.distance_type
        || target.space.vector_type != parsed.vector_space.vector_type
        || target.options != parsed.diskann
    {
        return Err(invalid("DiskANN live declaration mismatch"));
    }
    let table = DiskAnnTable::open(
        connection,
        schema,
        &name,
        parsed.vector_space,
        parsed.diskann,
        false,
    )?;
    if jobs::metadata(&table)?.instance_id != target.instance {
        return Err(jobs::stale("current index UUID differs from live lease"));
    }
    registration.verify_target(&target)?;
    Ok((table, target))
}
fn job_json(j: &Job) -> Json {
    json!({"job_id":j.job_id,"generation_id":j.generation_id,"parent_generation":j.parent_generation,
    "phase":j.phase,"fence":j.fence,"owner_nonce":j.owner_nonce.map(|n|hex(&n)),"cutoff_revision":j.cutoff_revision,
    "cutoff_entry_id":j.cutoff_entry_id,"expected_count":j.expected_count,"captured_count":j.captured_count,
    "scan_entry_id":j.scan_entry_id,"import_cursor":j.import_cursor,"validation_cursor":j.validation_cursor,"cleanup_cursor":j.cleanup_cursor,
    "source_hash":j.source_hash.map(|h|hex(&h)),"config_hash":hex(&j.config_hash),"candidate_hash":j.candidate_hash.map(|h|hex(&h)),
    "snapshot_chain":hex(&j.snapshot_chain),"failure_code":j.failure_code})
}
fn status(table: &DiskAnnTable) -> Result<String> {
    let (_, anchor) = table.read_metadata()?;
    let result = (|| {
        let bounded = jobs::status(table)?;
        let m = bounded.metadata;
        let published = m
            .published_generation
            .map(|g| table.read_generation(g))
            .transpose()?;
        let current_job = bounded.job;
        let eligible = published.as_ref().map_or(0, |g| g.eligible_count);
        if eligible > m.canonical_count {
            return Err(IndexError::with_code(
                ffi::SQLITE_CORRUPT as i32,
                "eligible exceeds canonical count",
            ));
        }
        Ok(json!({"format":4,"index_uuid":format!("idx:{}",hex(&m.instance_id)),"canonical_count":m.canonical_count,
            "pending_count":m.canonical_count-eligible,"eligible_count":eligible,"retired_count":m.retired_count,"revision":m.revision,
            "published_generation":m.published_generation,"published_count":published.as_ref().map_or(0,|g|g.total_count),
            "published_source_count":published.as_ref().map_or(0,|g|g.total_count),"published_source_hash":published.as_ref().and_then(|g|g.source_hash).map(|h|hex(&h)),
            "candidate_generation":m.candidate_generation,"active_job":m.active_job,"current_job":current_job.as_ref().map(job_json),
            "phase":current_job.as_ref().map(|j|j.phase),"fence":current_job.as_ref().map(|j|j.fence),"failure":current_job.as_ref().and_then(|j|j.failure_code),
            "cleanup_entry_id":m.cleanup_entry_id}).to_string())
    })();
    let finish = anchor.finish();
    let value = result?;
    finish?;
    Ok(value)
}
fn manifest_json(
    registration: &Registration,
    sessions: &crate::diskann_worker::CandidateSessions,
    handle: &crate::diskann_worker::CandidateHandle,
) -> Result<String> {
    let m = sessions.manifest(handle).map_err(worker_error)?;
    let statistics = registration
        .statistics(handle)?
        .map(|s| serde_json::from_str::<Json>(&s))
        .transpose()
        .map_err(|e| invalid(e.to_string()))?;
    Ok(json!({"candidate_token":handle.token().as_str(),"population":m.population,"node_count":m.node_count,"root":m.root,
        "medoid_ordinal":m.medoid_ordinal,"edge_count":m.edge_count,"input_hash":hex(m.input_digest.as_bytes()),
        "config_hash":hex(m.config_digest.as_bytes()),"candidate_hash":hex(m.topology_digest.as_bytes()),"retained_bytes":m.retained_bytes,"stats":statistics}).to_string())
}
fn build(
    connection: Connection<'static>,
    registration: &Arc<Registration>,
    schema: &str,
    name: &str,
    args: &Object,
) -> Result<String> {
    let guard = BuildGuard::enter(&connection)?;
    let job = args.id("job_id")?;
    let fence = args.id("fence")?;
    if args.integer("workers", Some(1))? != 1 {
        return Err(invalid("DiskANN workers must be 1"));
    }
    let budget =
        usize::try_from(args.id("memory_budget_bytes")?).map_err(|_| too_big("build budget"))?;
    let (table, target) = resolve(connection, registration, schema, name)?;
    let b = binding(&table, registration, job, fence)?;
    let existing_sessions = registration.existing_sessions()?;
    if let Some(handle) = registration.ready(&target.binding, job, fence)? {
        if handle.identity().config_digest().as_bytes() != &b.config_hash {
            return Err(jobs::stale("ready config"));
        }
        guard.recheck()?;
        let sessions = registration.sessions(None)?;
        return manifest_json(registration, &sessions, &handle);
    }
    let j = jobs::bound(&table, &b)?;
    if j.phase != jobs::BUILDING || j.import_cursor != 0 || j.candidate_hash.is_some() {
        return Err(jobs::stale("missing candidate requires fresh attempt"));
    }
    let n = usize::try_from(j.expected_count).map_err(|_| too_big("population"))?;
    let frontend_metadata = registration.metadata_reservation()?;
    let metadata_bytes = n
        .checked_mul(std::mem::size_of::<SnapshotNode>())
        .and_then(|v| v.checked_add(frontend_metadata))
        .ok_or_else(|| too_big("snapshot metadata"))?;
    let retained = existing_sessions
        .as_ref()
        .map(|s| s.retained_bytes())
        .transpose()
        .map_err(worker_error)?
        .unwrap_or(0);
    let retained = usize::try_from(retained).map_err(|_| too_big("retained candidates"))?;
    let external = metadata_bytes
        .checked_add(OWNED_WORKER_STACK_BYTES)
        .and_then(|v| v.checked_add(retained))
        .ok_or_else(|| too_big("build coexistence reservation"))?;
    let limits = BuildLimits {
        memory_budget_bytes: budget,
        external_reservation_bytes: external,
    };
    let config = jobs::build_config(&table);
    native::admit(n, &config, limits).map_err(native_error)?;
    let sessions = registration.sessions(Some(budget as u64))?;
    let materialized = guard.materialize(|readers| {
        jobs::materialize_snapshot(&table, &b, readers, budget.saturating_sub(external))
    })?;
    if materialized.source_hash != b.source_hash || materialized.config.digest() != b.config_hash {
        return Err(jobs::stale("materialized logical source/config"));
    }
    let checked = CheckedSnapshot::new(n, config.dimension, config.metric, materialized.vectors)
        .map_err(native_error)?;
    let n = checked.population();
    let identity = AttemptIdentity::new(
        registration.connection_nonce().clone(),
        target.binding.clone(),
        JobId::parse(&format!("job:{job}")).map_err(worker_error)?,
        FenceToken::parse(&format!("fence:{fence}")).map_err(worker_error)?,
        AttemptToken::parse(&format!("attempt:{job}-{fence}")).map_err(worker_error)?,
        Digest32::from_bytes(checked.digest()),
        Digest32::from_bytes(config.digest()),
    );
    let ticket = match sessions.begin(identity.clone()).map_err(worker_error)? {
        BeginOutcome::Replay(h) => return manifest_json(registration, &sessions, &h),
        BeginOutcome::Proceed(t) => t,
    };
    if let Err(e) = registration.starting(&identity, job, fence) {
        let _ = sessions.abandon(ticket);
        return Err(e);
    }
    let cancellation = ticket.cancellation();
    let result = guard.run_owned(checked, move |input| {
        native::build(input, config, limits, cancellation).map_err(native_error)
    });
    let topology = match result {
        Ok(t) => t,
        Err(e) => {
            let _ = sessions.abandon(ticket);
            return Err(e);
        }
    };
    // Native topology validation is compute-only, outside EVERY writer carrier.
    if let Err(e) = guard.recheck() {
        let _ = sessions.abandon(ticket);
        return Err(e);
    }
    let meta = match SnapshotMeta::new(n as u32, materialized.nodes).map_err(worker_error) {
        Ok(meta) => meta,
        Err(e) => {
            let _ = sessions.abandon(ticket);
            return Err(e);
        }
    };
    let stats = topology.stats();
    let statistics=json!({"native_workers":stats.native_workers,"native_coordinator_threads":stats.native_coordinator_threads,
        "inserted_vectors":stats.inserted_vectors,"pruned_nodes":stats.pruned_nodes,"extracted_nodes":stats.extracted_nodes,
        "reachable_nodes":stats.reachable_nodes,"edge_count":stats.edge_count,
        "medoid_read_bytes":stats.medoid_read_bytes,"medoid_full_scan_vectors":stats.medoid_full_scan_vectors,
        "medoid_readers_closed":stats.medoid_readers_closed,"native_alive_tasks_at_teardown":stats.native_alive_tasks_at_teardown,
        "coordinator_joined":stats.coordinator_joined,"medoid_seconds":stats.medoid_time.as_secs_f64(),
        "insert_seconds":stats.insert_time.as_secs_f64(),"prune_seconds":stats.prune_time.as_secs_f64(),
        "extract_seconds":stats.extract_time.as_secs_f64(),"total_seconds":stats.total_time.as_secs_f64(),
        "peak_estimate_bytes":topology.admission().peak_estimate_bytes,"external_reservation_bytes":external}).to_string();
    let handle = sessions
        .complete(ticket, topology, meta)
        .map_err(worker_error)?;
    if let Err(e) = guard
        .recheck()
        .and_then(|()| registration.completed(handle.clone(), statistics))
    {
        let _ = sessions.release(&handle);
        return Err(e);
    }
    manifest_json(registration, &sessions, &handle)
}
/// The prepared action contains only bounded owned candidate input and pure authority.
pub(crate) struct PreparedAction {
    kind: ActionKind,
    target: TargetMetadata,
    job: u64,
    fence: u64,
}
enum ActionKind {
    Prepare,
    Snapshot(jobs::StepOptions),
    Claim,
    Import(import::CandidateInput),
    Validate(import::CandidateInput),
    Publish(jobs::BuildBinding, [u8; 32]),
    Cancel(bool),
    Restart(bool),
    Cleanup(jobs::StepOptions),
}
pub(crate) enum ActionEffect {
    None,
    Claim(u64, u64),
    Forget,
}
/// Names accepted by [`prepare_action`] that drive maintenance and therefore
/// claim the connection's DiskANN workspace. `insert_batch` is deliberately
/// absent: it re-arms the writer set against its own chunk reservation instead.
///
/// This list mirrors the match arms in [`prepare_action`]; keep them identical.
pub(crate) fn is_maintenance_operation(operation: &str) -> bool {
    matches!(
        operation,
        "index_prepare"
            | "consolidate"
            | "index_snapshot_step"
            | "index_claim_build"
            | "index_import_step"
            | "index_validate_step"
            | "index_publish"
            | "index_cancel"
            | "index_restart"
            | "index_cleanup_step"
    )
}

pub(crate) fn prepare_action(
    table: &DiskAnnTable,
    registration: &Arc<Registration>,
    schema: &str,
    name: &str,
    operation: &str,
    path: &str,
) -> Result<PreparedAction> {
    let allowed: &[&str] = match operation {
        "index_prepare" | "consolidate" => &[],
        // NOTE: the accepted control names are mirrored by
        // `is_maintenance_operation` below; keep the two lists identical.
        "index_snapshot_step" => &["job_id", "fence", "max_records", "max_bytes"],
        "index_claim_build" => &["job_id", "fence"],
        "index_import_step" | "index_validate_step" => &[
            "job_id",
            "fence",
            "candidate_token",
            "max_records",
            "max_bytes",
        ],
        "index_publish" => &["job_id", "fence", "candidate_token"],
        "index_cancel" | "index_restart" => &["job_id", "fence", "force"],
        "index_cleanup_step" => &["max_records", "max_bytes"],
        _ => return Err(invalid("unknown DiskANN control operation")),
    };
    let obj = Object::parse(path, allowed)?;
    let target = registration.target(schema, name)?;
    registration.verify_target(&target)?;
    if jobs::metadata(table)?.instance_id != target.instance {
        return Err(jobs::stale("control target UUID"));
    }
    let has_job = !matches!(
        operation,
        "index_prepare" | "consolidate" | "index_cleanup_step"
    );
    let (job, fence) = if has_job {
        (obj.id("job_id")?, obj.id("fence")?)
    } else {
        (0, 0)
    };
    if has_job && jobs::job(table, job)?.fence != fence {
        return Err(jobs::stale("prepared action fence"));
    }
    let kind = match operation {
        "index_prepare" | "consolidate" => ActionKind::Prepare,
        "index_snapshot_step" => ActionKind::Snapshot(obj.steps(table)?),
        "index_claim_build" => ActionKind::Claim,
        "index_import_step" | "index_validate_step" | "index_publish" => {
            let b = binding(table, registration, job, fence)?;
            let handle = registration.candidate(&target.binding, job, fence, obj.token()?)?;
            let sessions = registration.sessions(None)?;
            if operation == "index_publish" {
                let m = sessions.manifest(&handle).map_err(worker_error)?;
                ActionKind::Publish(b, *m.topology_digest.as_bytes())
            } else {
                let j = jobs::bound(table, &b)?;
                let cursor = if operation == "index_import_step" {
                    j.import_cursor
                } else {
                    j.validation_cursor
                };
                let input = import::CandidateInput::copy(
                    &sessions,
                    &handle,
                    &b,
                    *handle.identity().input_digest().as_bytes(),
                    cursor,
                    obj.steps(table)?,
                )?;
                if operation == "index_import_step" {
                    ActionKind::Import(input)
                } else {
                    ActionKind::Validate(input)
                }
            }
        }
        "index_cancel" => ActionKind::Cancel(obj.force()?),
        "index_restart" => ActionKind::Restart(obj.force()?),
        "index_cleanup_step" => ActionKind::Cleanup(obj.steps(table)?),
        _ => unreachable!(),
    };
    Ok(PreparedAction {
        kind,
        target,
        job,
        fence,
    })
}
pub(crate) fn execute_action(
    table: &DiskAnnTable,
    registration: &Arc<Registration>,
    a: &PreparedAction,
) -> Result<ActionEffect> {
    registration.verify_target(&a.target)?;
    if jobs::metadata(table)?.instance_id != a.target.instance {
        return Err(jobs::stale("control target instance changed"));
    }
    let owner = Some(registration.owner);
    match &a.kind {
        ActionKind::Prepare => {
            jobs::prepare(table)?;
        }
        ActionKind::Snapshot(opts) => {
            fence_check(table, a.job, a.fence, registration)?;
            jobs::snapshot_step(table, a.job, *opts)?;
        }
        ActionKind::Claim => {
            fence_check(table, a.job, a.fence, registration)?;
            jobs::claim_build(table, a.job, registration.owner)?;
            return Ok(ActionEffect::Claim(a.job, a.fence));
        }
        ActionKind::Import(input) => {
            fence_check(table, a.job, a.fence, registration)?;
            import::import_step(table, input)?;
        }
        ActionKind::Validate(input) => {
            fence_check(table, a.job, a.fence, registration)?;
            import::validate_step(table, input)?;
        }
        ActionKind::Publish(b, hash) => {
            fence_check(table, a.job, a.fence, registration)?;
            import::publish(table, b, *hash)?;
        }
        ActionKind::Cancel(force) => {
            let j = jobs::active(table, a.job)?;
            lifecycle::cancel(
                table,
                a.job,
                a.fence,
                if j.owner_nonce.is_none() { None } else { owner },
                *force,
                None,
            )?;
            return Ok(ActionEffect::Forget);
        }
        ActionKind::Restart(force) => {
            let j = jobs::active(table, a.job)?;
            lifecycle::restart(
                table,
                a.job,
                a.fence,
                if j.owner_nonce.is_none() { None } else { owner },
                *force,
            )?;
            return Ok(ActionEffect::Forget);
        }
        ActionKind::Cleanup(opts) => {
            lifecycle::cleanup_step(table, *opts)?;
            if jobs::metadata(table)?.active_job.is_none() {
                return Ok(ActionEffect::Forget);
            }
        }
    }
    Ok(ActionEffect::None)
}
pub(crate) fn after_action(
    registration: &Registration,
    a: &PreparedAction,
    e: ActionEffect,
) -> Result<()> {
    match e {
        ActionEffect::None => Ok(()),
        ActionEffect::Claim(job, fence) => registration.claimed(&a.target.binding, job, fence),
        ActionEffect::Forget => registration.forget(&a.target.binding),
    }
}
/// Ownership transferred to SQLite by create_function_v2, including registration failure.
pub(crate) unsafe extern "C" fn registration_destroy(p: *mut c_void) {
    if !p.is_null() {
        // SAFETY: SQLite invokes this once for the unique boxed Arc supplied as pApp.
        unsafe { drop(Box::from_raw(p.cast::<Arc<Registration>>())) };
    }
}
unsafe fn scalar_call(
    ctx: *mut ffi::sqlite3_context,
    argc: c_int,
    argv: *mut *mut ffi::sqlite3_value,
    is_build: bool,
) {
    let result = (|| {
        // SAFETY: the initialized host API reports its actual supported prefix.
        if unsafe { ffi::libversion_number() } < 3038000 {
            return Err(invalid("DiskANN requires SQLite 3.38.0 or newer"));
        }
        if argc != if is_build { 3 } else { 2 } {
            return Err(invalid("invalid DiskANN scalar argument count"));
        }
        // SAFETY: SQLite supplies protected argv and our boxed Arc pApp for this callback.
        let values = unsafe { ffi::arguments(argc, argv) };
        let schema = bounded_text(&mut values[0], MAX_NAME_BYTES, "schema")?.to_owned();
        let name = bounded_text(&mut values[1], MAX_NAME_BYTES, "table")?.to_owned();
        // SAFETY: these callback APIs return the current host connection and the registration's live pApp.
        let (db, p) = unsafe { (ffi::context_db_handle(ctx), ffi::user_data(ctx)) };
        if p.is_null() {
            return Err(invalid("DiskANN registration unavailable"));
        }
        // SAFETY: pApp is exactly the boxed Arc transferred during this scalar's registration.
        let registration = unsafe { &*p.cast::<Arc<Registration>>() }.clone();
        // SAFETY: the erased store connection is borrowed only on this callback thread and dropped before return;
        // no connection, table, statement or registry reference enters the owned native closure.
        let connection: Connection<'static> = unsafe { Connection::borrow(db) }?;
        if is_build {
            let text = bounded_text(&mut values[2], MAX_JSON_BYTES, "build options")?;
            let obj = Object::parse(text, &["job_id", "fence", "memory_budget_bytes", "workers"])?;
            build(connection, &registration, &schema, &name, &obj)
        } else {
            let (table, _) = resolve(connection, &registration, &schema, &name)?;
            status(&table)
        }
    })();
    match result {
        Ok(text) => {
            // SAFETY: the result helper copies UTF-8 to the live callback context.
            unsafe { ffi::result_text(ctx, &text) };
        }
        Err(e) => {
            // SAFETY: the helpers copy error text and preserve the typed SQLite code.
            unsafe {
                ffi::result_error(ctx, &e.message);
                ffi::result_error_code(ctx, e.code)
            };
        }
    }
}
pub(crate) unsafe extern "C" fn index_status(
    ctx: *mut ffi::sqlite3_context,
    argc: c_int,
    argv: *mut *mut ffi::sqlite3_value,
) {
    // SAFETY: forwards the current SQLite scalar callback contract.
    unsafe { scalar_call(ctx, argc, argv, false) }
}
pub(crate) unsafe extern "C" fn index_build(
    ctx: *mut ffi::sqlite3_context,
    argc: c_int,
    argv: *mut *mut ffi::sqlite3_value,
) {
    // SAFETY: forwards the current SQLite scalar callback contract.
    unsafe { scalar_call(ctx, argc, argv, true) }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_error_codes_distinguish_budget_allocation_cancellation() {
        assert_eq!(
            native_error(BuildError::Limit("budget".into())).code,
            ffi::SQLITE_TOOBIG as i32
        );
        assert_eq!(
            native_error(BuildError::Allocation).code,
            ffi::SQLITE_NOMEM as i32
        );
        assert_eq!(
            native_error(BuildError::Cancelled).code,
            ffi::SQLITE_ABORT as i32
        );
        assert_eq!(
            native_error(BuildError::InvalidTopology("bad".into())).code,
            ffi::SQLITE_CORRUPT as i32
        );
        assert_eq!(
            native_error(BuildError::InvalidInput("bad".into())).code,
            ffi::SQLITE_ERROR as i32
        );
    }
    #[test]
    fn strict_json_types_keys_and_duplicates() {
        for bad in ["[]", "null", "{\"x\":1}", "{\"job_id\":1,\"job_id\":2}"] {
            assert!(Object::parse(bad, &["job_id"]).is_err());
        }
        for bad in ["true", "1.0", "\"1\"", "-1", "null"] {
            let text = format!("{{\"job_id\":{bad}}}");
            assert!(Object::parse(&text, &["job_id"])
                .unwrap()
                .id("job_id")
                .is_err());
        }
        assert_eq!(
            Object::parse("{\"job_id\":1}", &["job_id"])
                .unwrap()
                .id("job_id")
                .unwrap(),
            1
        );
    }
}
