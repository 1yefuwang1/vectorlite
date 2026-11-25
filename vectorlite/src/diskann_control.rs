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
use crate::sqlite::{quote_identifier, Connection, Statement, Step};
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
        Ok(json!({"format":5,"index_uuid":format!("idx:{}",hex(&m.instance_id)),"canonical_count":m.canonical_count,
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
    // SAFETY: forwards the scalar result contract of this callback.
    unsafe { scalar_result(ctx, result) };
}
/// Reports one scalar outcome with the original typed SQLite code.
unsafe fn scalar_result(ctx: *mut ffi::sqlite3_context, result: Result<String>) {
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
/// D7 migration scalar. It never resolves a vtab connection; the upgrade reads
/// the schema catalog and rewrites the derived shadow tables in one savepoint.
unsafe fn upgrade_scalar_call(
    ctx: *mut ffi::sqlite3_context,
    argc: c_int,
    argv: *mut *mut ffi::sqlite3_value,
) {
    let result = (|| {
        // SAFETY: the initialized host API reports its actual supported prefix.
        if unsafe { ffi::libversion_number() } < 3038000 {
            return Err(invalid("DiskANN requires SQLite 3.38.0 or newer"));
        }
        if argc != 2 {
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
        // SAFETY: the erased store connection is borrowed only on this callback thread and dropped before return.
        let connection: Connection<'static> = unsafe { Connection::borrow(db) }?;
        upgrade::run(connection, &registration, &schema, &name)
    })();
    // SAFETY: forwards the scalar result contract of this callback.
    unsafe { scalar_result(ctx, result) }
}
pub(crate) unsafe extern "C" fn index_upgrade(
    ctx: *mut ffi::sqlite3_context,
    argc: c_int,
    argv: *mut *mut ffi::sqlite3_value,
) {
    // SAFETY: forwards the current SQLite scalar callback contract.
    unsafe { upgrade_scalar_call(ctx, argc, argv) }
}
/// D7 migration surface: promote one canonical Format-4 deferred index to
/// Format 5 inside one savepoint transaction.
///
/// The scalar deliberately does not resolve a virtual-table connection: a
/// Format-4 declaration cannot xConnect against this build, because the format
/// check rejects it before any mutation. Instead the upgrade reads the table
/// declaration from the schema catalog with the same strict grammar as the
/// control path, proves the carried catalog is exactly the frozen Format-4
/// shape, and then rewrites only the derived tables. entries and current are
/// never written.
mod upgrade {
    use super::*;
    use crate::diskann_store::codec::{
        self, bounded_blob, checked_id, checked_increment, checked_product, corrupt, nonnegative,
    };
    use crate::diskann_store::{
        format5_create_table_sql, Metadata, StorageNames, FORMAT_VERSION, MAX_DESCRIPTOR_BYTES,
    };
    use crate::diskann_target::DeclarationTarget;

    /// Frozen Format-4 shadow layouts. The migration surface owns the historical
    /// definitions: they are never used to create data, only to prove that the
    /// carried catalog is exactly the Format-4 shape before the transaction
    /// starts. Their byte text is the Format-4 build's own DDL, which SQLite
    /// stores with the schema qualifier stripped.
    struct LegacyLayout {
        suffix: &'static str,
        definition: &'static str,
        without_rowid: bool,
        unique: &'static [&'static [&'static str]],
    }

    const LEGACY_LAYOUTS: &[LegacyLayout] = &[
        LegacyLayout {
            suffix: "diskann_meta",
            without_rowid: false,
            unique: &[],
            definition: "singleton INTEGER PRIMARY KEY CHECK(singleton=1),format_version INTEGER NOT NULL CHECK(format_version=4),instance_id BLOB NOT NULL CHECK(length(instance_id)=16),descriptor BLOB NOT NULL,revision INTEGER NOT NULL CHECK(revision>=0),next_entry_id INTEGER NOT NULL CHECK(next_entry_id>=1),next_job_id INTEGER NOT NULL CHECK(next_job_id>=1),next_generation_id INTEGER NOT NULL CHECK(next_generation_id>=1),canonical_count INTEGER NOT NULL CHECK(canonical_count>=0),retired_count INTEGER NOT NULL CHECK(retired_count>=0),published_generation INTEGER CHECK(published_generation>=1),candidate_generation INTEGER CHECK(candidate_generation>=1),active_job INTEGER CHECK(active_job>=1),next_fence INTEGER NOT NULL CHECK(next_fence>=1),cleanup_entry_id INTEGER NOT NULL CHECK(cleanup_entry_id>=0 AND cleanup_entry_id<next_entry_id)",
        },
        LegacyLayout {
            suffix: "diskann_entries",
            without_rowid: false,
            unique: &[],
            definition: "entry_id INTEGER PRIMARY KEY CHECK(entry_id>=1),public_rowid INTEGER NOT NULL CHECK(public_rowid>=0),created_revision INTEGER NOT NULL CHECK(created_revision>=1),retired_revision INTEGER CHECK(retired_revision>created_revision),vector BLOB NOT NULL CHECK(length(vector)={vector_bytes})",
        },
        LegacyLayout {
            suffix: "diskann_current",
            without_rowid: false,
            unique: &[&["entry_id"]],
            definition: "public_rowid INTEGER PRIMARY KEY CHECK(public_rowid>=0),entry_id INTEGER NOT NULL UNIQUE CHECK(entry_id>=1)",
        },
        LegacyLayout {
            suffix: "diskann_generations",
            without_rowid: false,
            unique: &[],
            definition: "generation_id INTEGER PRIMARY KEY CHECK(generation_id>=1),job_id INTEGER NOT NULL CHECK(job_id>=1),state INTEGER NOT NULL CHECK(state IN (0,1,2)),source_revision INTEGER NOT NULL CHECK(source_revision>=0),source_hash BLOB CHECK(source_hash IS NULL OR length(source_hash)=32),config_hash BLOB NOT NULL CHECK(length(config_hash)=32),total_count INTEGER NOT NULL CHECK(total_count>=0 AND total_count<=4294967294),eligible_count INTEGER NOT NULL CHECK(eligible_count>=0 AND eligible_count<=total_count),root_local_id INTEGER,CHECK(state!=1 OR source_hash IS NOT NULL),CHECK((total_count=0 AND root_local_id IS NULL) OR (total_count>0 AND root_local_id=total_count))",
        },
        LegacyLayout {
            suffix: "diskann_graph_nodes",
            without_rowid: true,
            unique: &[
                &["generation_id", "native_local_id"],
                &["generation_id", "entry_id"],
            ],
            definition: "generation_id INTEGER NOT NULL CHECK(generation_id>=1),native_local_id INTEGER NOT NULL CHECK(native_local_id>=0 AND native_local_id<=4294967294),entry_id INTEGER CHECK(entry_id>=1),state INTEGER NOT NULL CHECK(state IN (0,2)),frozen_vector BLOB,neighbors BLOB NOT NULL CHECK(length(neighbors)<={adjacency_bytes} AND length(neighbors)%8=0),PRIMARY KEY(generation_id,native_local_id),UNIQUE(generation_id,entry_id),CHECK((state=0 AND entry_id IS NOT NULL AND frozen_vector IS NULL) OR (state=2 AND entry_id IS NULL AND typeof(frozen_vector)='blob' AND length(frozen_vector)={vector_bytes}))",
        },
        LegacyLayout {
            suffix: "diskann_jobs",
            without_rowid: false,
            unique: &[],
            definition: "job_id INTEGER PRIMARY KEY CHECK(job_id>=1),generation_id INTEGER NOT NULL CHECK(generation_id>=1),parent_generation INTEGER CHECK(parent_generation>=1),cutoff_revision INTEGER NOT NULL CHECK(cutoff_revision>=0),fence INTEGER NOT NULL CHECK(fence>=1),owner_nonce BLOB CHECK(owner_nonce IS NULL OR length(owner_nonce)=16),phase INTEGER NOT NULL CHECK(phase BETWEEN 0 AND 6),expected_count INTEGER NOT NULL CHECK(expected_count>=0 AND expected_count<=4294967294),scan_entry_id INTEGER NOT NULL CHECK(scan_entry_id>=0),captured_count INTEGER NOT NULL CHECK(captured_count>=0 AND captured_count<=expected_count),import_cursor INTEGER NOT NULL CHECK(import_cursor>=0),validation_cursor INTEGER NOT NULL CHECK(validation_cursor>=0),cleanup_cursor INTEGER NOT NULL CHECK(cleanup_cursor>=0),source_hash BLOB CHECK(source_hash IS NULL OR length(source_hash)=32),config_hash BLOB NOT NULL CHECK(length(config_hash)=32),candidate_hash BLOB CHECK(candidate_hash IS NULL OR length(candidate_hash)=32),failure_code INTEGER,cutoff_entry_id INTEGER NOT NULL CHECK(cutoff_entry_id>=0),snapshot_chain BLOB NOT NULL CHECK(typeof(snapshot_chain)='blob' AND length(snapshot_chain)=32),CHECK(scan_entry_id<=cutoff_entry_id AND expected_count<=cutoff_entry_id)",
        },
        LegacyLayout {
            suffix: "diskann_snapshot_map",
            without_rowid: true,
            unique: &[
                &["job_id", "ordinal"],
                &["job_id", "entry_id"],
            ],
            definition: "job_id INTEGER NOT NULL CHECK(job_id>=1),ordinal INTEGER NOT NULL CHECK(ordinal>=0 AND ordinal<=4294967293),entry_id INTEGER NOT NULL CHECK(entry_id>=1),public_rowid INTEGER NOT NULL CHECK(public_rowid>=0),PRIMARY KEY(job_id,ordinal),UNIQUE(job_id,entry_id)",
        },
        LegacyLayout {
            suffix: "diskann_txn",
            without_rowid: false,
            unique: &[],
            definition: "singleton INTEGER PRIMARY KEY CHECK(singleton IN (1,2)),value INTEGER NOT NULL CHECK(value=0)",
        },
    ];

    const LEGACY_FORMAT_VERSION: u64 = 4;
    /// Format-4 neighbour slots were a bare 8-byte native id.
    const LEGACY_SLOT_BYTES: usize = 8;
    const SAVEPOINT: &str = "vectorlite_diskann_format5_upgrade";

    /// The target declaration text validated before the transaction starts. The
    /// SAVEPOINT body re-reads it under the write reservation and refuses the
    /// upgrade if it changed, so a supported writer racing validation can never
    /// make the upgrade rewrite the derived tables from a stale declaration.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct DeclarationSnapshot {
        name: String,
        sql: String,
    }

    /// Everything preserved from the pre-transaction validation that the
    /// SAVEPOINT body later consumes. The re-read under the write reservation must
    /// equal this snapshot exactly; a supported INSERT commits a new revision and
    /// counters, so any difference means an intervening writer.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct PreservedSnapshot {
        metadata: Metadata,
        descriptor: Vec<u8>,
        next_fence: u64,
        declaration: DeclarationSnapshot,
    }

    /// Everything validated before the transaction starts. The descriptor is the
    /// Format-5 descriptor derived by parsing the carried Format-4 descriptor,
    /// never a synthesized value.
    struct UpgradePlan {
        names: StorageNames,
        metadata: Metadata,
        descriptor: Vec<u8>,
        vector_bytes: usize,
        legacy_adjacency_bytes: usize,
        adjacency_bytes: usize,
        next_fence: u64,
        declaration: DeclarationSnapshot,
    }

    impl UpgradePlan {
        /// The exact values the transaction will consume from the validated plan.
        fn preserved(&self) -> PreservedSnapshot {
            PreservedSnapshot {
                metadata: self.metadata.clone(),
                descriptor: self.descriptor.clone(),
                next_fence: self.next_fence,
                declaration: self.declaration.clone(),
            }
        }
    }

    /// A stable refusal for an upgrade whose validated state changed under it.
    ///
    /// SQLITE_BUSY is the correct class: the only way the preserved metadata can
    /// move between validation and the write reservation is another connection
    /// committing a supported mutation, and the caller can retry the upgrade once
    /// it is quiescent again. The refused SAVEPOINT is rolled back, so no
    /// mutation is observable.
    fn intervening_writer() -> IndexError {
        IndexError::with_code(
            ffi::SQLITE_BUSY as i32,
            "DiskANN upgrade observed an intervening writer; the validated Format-4 state changed, so the upgrade was refused without mutation; retry it",
        )
    }

    /// Refuses when the state re-read under the write reservation differs from
    /// the pre-transaction validated snapshot.
    fn ensure_snapshot_unchanged(
        validated: &PreservedSnapshot,
        reread: &PreservedSnapshot,
    ) -> Result<()> {
        if validated == reread {
            Ok(())
        } else {
            Err(intervening_writer())
        }
    }

    /// Public entry point for vectorlite_index_upgrade(schema, table).
    pub(super) fn run(
        connection: Connection<'static>,
        registration: &Arc<Registration>,
        schema: &str,
        name: &str,
    ) -> Result<String> {
        let (table, declaration_sql, target) = declaration(&connection, schema, name)?;
        let names = StorageNames::new(schema, &table)?;
        let vector_bytes = checked_product(target.vector_space.dim, 4, "vector")?;
        // The carried catalog is Format 4 (8-byte slots); the tables this
        // transaction creates are Format 5 (16-byte slots).
        let legacy_adjacency_bytes = checked_product(
            target.diskann.physical_degree(),
            LEGACY_SLOT_BYTES,
            "adjacency",
        )?;
        let adjacency_bytes = checked_product(target.diskann.physical_degree(), 16, "adjacency")?;
        // The format check runs before any catalog-shape validation so a second
        // invocation on an already-upgraded database refuses at the
        // format_version check with zero mutation.
        if catalog_table_sql(&connection, schema, &names.base("diskann_meta"))?.is_none() {
            return Err(corrupt(
                "DiskANN Format-4 metadata table is missing; nothing to upgrade",
            ));
        }
        let format = format_version(&connection, &names)?;
        if format != LEGACY_FORMAT_VERSION {
            return Err(invalid(format!(
                "DiskANN vectorlite_index_upgrade requires an unupgraded Format-4 database; found format_version {format}"
            )));
        }
        let plan = validate_legacy(
            &connection,
            &names,
            vector_bytes,
            legacy_adjacency_bytes,
            adjacency_bytes,
            &target,
            DeclarationSnapshot {
                name: table.clone(),
                sql: declaration_sql,
            },
        )?;
        // Admission shares the standalone-build contract: exactly the running
        // scalar statement may be busy, the connection must be in autocommit and
        // no schema may report an open transaction. An outer BEGIN/SAVEPOINT or
        // any other live cursor makes the savepoint wrapper ambiguous.
        let guard = BuildGuard::enter(&connection)?;
        require_rollback_journal(&connection, schema)?;
        // A live vtab instance for this table cannot exist for a Format-4 file on
        // this build, but any registered candidate state is forgotten before any
        // shadow table changes so no stale lease can reattach after the fence bump.
        if let Ok(live) = registration.target(schema, &table) {
            registration.forget(&live.binding)?;
        }
        execute_plan(&connection, &plan)?;
        drop(guard);
        Ok(json!({
            "format": FORMAT_VERSION,
            "upgraded": true,
            "schema": schema,
            "table": table,
            "index_uuid": format!("idx:{}", hex(&plan.metadata.instance_id)),
            "revision": plan.metadata.revision,
            "next_entry_id": plan.metadata.next_entry_id,
            "next_job_id": plan.metadata.next_job_id,
            "next_generation_id": plan.metadata.next_generation_id,
            "next_fence": plan.next_fence,
            "canonical_count": plan.metadata.canonical_count,
            "retired_count": plan.metadata.retired_count,
            "cleanup_entry_id": plan.metadata.cleanup_entry_id,
            "abandoned_published_generation": plan.metadata.published_generation,
            "abandoned_candidate_generation": plan.metadata.candidate_generation,
            "abandoned_active_job": plan.metadata.active_job,
        })
        .to_string())
    }

    /// Resolves the target declaration from the schema catalog only. No virtual
    /// table is connected and no registration lease is consulted.
    fn declaration(
        connection: &Connection<'_>,
        schema: &str,
        name: &str,
    ) -> Result<(String, String, DeclarationTarget)> {
        if schema.is_empty() || name.is_empty() || schema.contains('\0') || name.contains('\0') {
            return Err(invalid("nonempty schema/table without NUL required"));
        }
        let Some((stored, ddl)) = catalog_table_sql(connection, schema, name)? else {
            return Err(invalid("DiskANN target table not found"));
        };
        let parsed = parse_declaration(&ddl, schema, &stored, MAX_DECLARATION_BYTES)
            .map_err(|e| invalid(format!("DiskANN target: {e}")))?;
        Ok((stored, ddl, parsed))
    }

    /// Reads one sqlite_schema declaration row by exact schema and base name.
    /// The name comparison is case-insensitive (SQLite identifier rules) and the
    /// caller proves that the returned spelling is the canonical one.
    fn catalog_table_sql(
        connection: &Connection<'_>,
        schema: &str,
        base: &str,
    ) -> Result<Option<(String, String)>> {
        let mut statement = connection.prepare(&format!(
            "SELECT name,typeof(sql),length(CAST(sql AS BLOB)),CASE WHEN typeof(sql)='text' AND length(CAST(sql AS BLOB))<=?2 THEN sql END FROM {}.sqlite_schema WHERE name=?1 COLLATE NOCASE AND type='table' LIMIT 2",
            quote_identifier(schema)?
        ))?;
        statement.bind_text(1, base)?;
        statement.bind_i64(2, MAX_DECLARATION_BYTES as i64)?;
        let result = (|| {
            if statement.step()? != Step::Row {
                return Ok(None);
            }
            let name = statement.column_text(0)?;
            if statement.column_text(1)? != "text" {
                return Err(corrupt("DiskANN table declaration is not TEXT"));
            }
            let length = statement.column_i64(2)?;
            if length < 0 || length as usize > MAX_DECLARATION_BYTES {
                return Err(corrupt("DiskANN table declaration exceeds the byte bound"));
            }
            let sql = statement.column_text(3)?;
            if statement.step()? != Step::Done {
                return Err(corrupt("DiskANN table declaration is ambiguous"));
            }
            Ok(Some((name, sql)))
        })();
        let finish = statement.finish();
        let out = result?;
        finish?;
        Ok(out)
    }

    fn format_version(connection: &Connection<'_>, names: &StorageNames) -> Result<u64> {
        let mut statement = connection.prepare(&format!(
            "SELECT format_version FROM {} WHERE singleton=1",
            names.meta
        ))?;
        let result = (|| {
            if statement.step()? != Step::Row {
                return Err(corrupt("DiskANN metadata row is missing"));
            }
            let version = nonnegative(&statement, 0, "format version")?;
            if statement.step()? != Step::Done {
                return Err(corrupt("DiskANN metadata singleton is not unique"));
            }
            Ok(version)
        })();
        let finish = statement.finish();
        let version = result?;
        finish?;
        Ok(version)
    }

    /// Proves that the carried catalog is exactly the frozen Format-4 shape and
    /// returns the preserved metadata plus the derived Format-5 descriptor.
    fn validate_legacy(
        connection: &Connection<'_>,
        names: &StorageNames,
        vector_bytes: usize,
        legacy_adjacency_bytes: usize,
        adjacency_bytes: usize,
        target: &DeclarationTarget,
        declaration: DeclarationSnapshot,
    ) -> Result<UpgradePlan> {
        // The upgrade creates this Format-5-only table; a pre-existing one means
        // the file is not a canonical Format-4 database.
        if table_exists(
            connection,
            &names.schema,
            &names.base("diskann_generation_retired"),
        )? {
            return Err(corrupt(
                "DiskANN database already contains a Format-5 generation_retired table",
            ));
        }
        for layout in LEGACY_LAYOUTS {
            let base = names.base(layout.suffix);
            let Some((stored, sql)) = catalog_table_sql(connection, &names.schema, &base)? else {
                return Err(corrupt("DiskANN Format-4 shadow table is missing"));
            };
            if stored != base {
                return Err(corrupt(
                    "DiskANN Format-4 shadow table name is not canonical",
                ));
            }
            if sql != expected_legacy_sql(&base, layout, vector_bytes, legacy_adjacency_bytes)? {
                return Err(corrupt(
                    "DiskANN Format-4 shadow table definition is not canonical",
                ));
            }
            validate_legacy_indexes(connection, &names.schema, &base, layout.unique)?;
            validate_legacy_foreign_keys(connection, &names.schema, &base)?;
        }
        validate_legacy_triggers(connection, names)?;
        validate_guard_rows(connection, names)?;
        let (metadata, descriptor, next_fence) =
            validate_legacy_metadata(connection, names, target)?;
        Ok(UpgradePlan {
            names: names.clone(),
            metadata,
            descriptor,
            vector_bytes,
            legacy_adjacency_bytes,
            adjacency_bytes,
            next_fence,
            declaration,
        })
    }

    fn table_exists(connection: &Connection<'_>, schema: &str, base: &str) -> Result<bool> {
        let mut statement = connection.prepare(&format!(
            "SELECT 1 FROM {}.sqlite_schema WHERE type='table' AND name=?1 COLLATE NOCASE LIMIT 1",
            quote_identifier(schema)?
        ))?;
        statement.bind_text(1, base)?;
        let found = statement.step()? == Step::Row;
        statement.finish()?;
        Ok(found)
    }

    /// The exact DDL text SQLite stores for a Format-4 shadow table: the
    /// Format-4 build created it through the schema qualifier, but
    /// sqlite_schema.sql keeps only the quoted base name.
    fn expected_legacy_sql(
        base: &str,
        layout: &LegacyLayout,
        vector_bytes: usize,
        adjacency_bytes: usize,
    ) -> Result<String> {
        let definition = layout
            .definition
            .replace("{vector_bytes}", &vector_bytes.to_string())
            .replace("{adjacency_bytes}", &adjacency_bytes.to_string());
        Ok(format!(
            "CREATE TABLE {} ({}){}",
            quote_identifier(base)?,
            definition,
            if layout.without_rowid {
                " WITHOUT ROWID"
            } else {
                ""
            }
        ))
    }

    /// The Format-4 catalog validation only admitted the unique indexes the
    /// layouts imply; an extra index is not a canonical Format-4 declaration.
    fn validate_legacy_indexes(
        connection: &Connection<'_>,
        schema: &str,
        base: &str,
        expected: &[&[&str]],
    ) -> Result<()> {
        let mut list = connection.prepare(
            "SELECT length(name),name,\"unique\",origin,partial FROM pragma_index_list(?1,?2)",
        )?;
        list.bind_text(1, base)?;
        list.bind_text(2, schema)?;
        let mut seen = vec![false; expected.len()];
        while list.step()? == Step::Row {
            if nonnegative(&list, 0, "index name length")? > (base.len() + 64) as u64
                || nonnegative(&list, 2, "unique flag")? != 1
                || !matches!(list.column_text(3)?.as_str(), "u" | "pk")
                || nonnegative(&list, 4, "partial flag")? != 0
            {
                return Err(corrupt("DiskANN Format-4 shadow index is not canonical"));
            }
            let index = list.column_text(1)?;
            let mut columns = Vec::new();
            let mut info = connection.prepare(
                "SELECT seqno,length(name),name FROM pragma_index_info(?1,?2) ORDER BY seqno",
            )?;
            info.bind_text(1, &index)?;
            info.bind_text(2, schema)?;
            while info.step()? == Step::Row {
                if columns.len() >= 2
                    || nonnegative(&info, 0, "index sequence")? != columns.len() as u64
                    || nonnegative(&info, 1, "index column length")? > 32
                {
                    return Err(corrupt("DiskANN Format-4 shadow index is not canonical"));
                }
                columns.push(info.column_text(2)?);
            }
            info.finish()?;
            let position = expected
                .iter()
                .position(|names| {
                    names.len() == columns.len() && names.iter().zip(&columns).all(|(a, b)| a == b)
                })
                .ok_or_else(|| corrupt("DiskANN Format-4 unique index does not match layout"))?;
            if seen[position] {
                return Err(corrupt("DiskANN Format-4 duplicate layout index"));
            }
            seen[position] = true;
        }
        list.finish()?;
        if seen.iter().any(|value| !*value) {
            return Err(corrupt("DiskANN Format-4 required unique index is missing"));
        }
        Ok(())
    }

    fn validate_legacy_foreign_keys(
        connection: &Connection<'_>,
        schema: &str,
        base: &str,
    ) -> Result<()> {
        let mut keys =
            connection.prepare("SELECT 1 FROM pragma_foreign_key_list(?1,?2) LIMIT 1")?;
        keys.bind_text(1, base)?;
        keys.bind_text(2, schema)?;
        if keys.step()? != Step::Done {
            return Err(corrupt(
                "DiskANN Format-4 shadow tables cannot have foreign keys",
            ));
        }
        keys.finish()
    }

    /// A trigger can rewrite or discard the rows this transaction deletes and
    /// recreates, so any trigger on a shadow table is non-canonical. TEMP
    /// triggers can target main-schema tables and are checked as well.
    fn validate_legacy_triggers(connection: &Connection<'_>, names: &StorageNames) -> Result<()> {
        for layout in LEGACY_LAYOUTS {
            let base = names.base(layout.suffix);
            reject_legacy_triggers(connection, &names.schema, &base)?;
            if !names.schema.eq_ignore_ascii_case("temp") {
                reject_legacy_triggers(connection, "temp", &base)?;
            }
        }
        Ok(())
    }

    fn reject_legacy_triggers(connection: &Connection<'_>, schema: &str, base: &str) -> Result<()> {
        let mut triggers = connection.prepare(&format!(
            "SELECT 1 FROM {}.sqlite_schema WHERE type='trigger' AND tbl_name=?1 COLLATE NOCASE LIMIT 1",
            quote_identifier(schema)?
        ))?;
        triggers.bind_text(1, base)?;
        if triggers.step()? != Step::Done {
            return Err(corrupt(
                "DiskANN Format-4 shadow tables cannot have triggers",
            ));
        }
        triggers.finish()
    }

    /// The two immutable guard rows force SQLite to journal every nested write
    /// of a carrier; they must be exactly the Format-4 initialization rows.
    fn validate_guard_rows(connection: &Connection<'_>, names: &StorageNames) -> Result<()> {
        let mut rows = connection.prepare(&format!(
            "SELECT singleton,value FROM {} ORDER BY singleton LIMIT 3",
            names.txn
        ))?;
        let result = (|| {
            for singleton in [1u64, 2] {
                if rows.step()? != Step::Row
                    || nonnegative(&rows, 0, "atomic guard singleton")? != singleton
                    || nonnegative(&rows, 1, "atomic guard value")? != 0
                {
                    return Err(corrupt(
                        "DiskANN Format-4 atomic guard rows are not canonical",
                    ));
                }
            }
            if rows.step()? != Step::Done {
                return Err(corrupt(
                    "DiskANN Format-4 atomic guard contains unexpected rows",
                ));
            }
            Ok(())
        })();
        let finish = rows.finish();
        result.and(finish)
    }

    /// Validates the carried descriptor verbatim against the Format-4 descriptor
    /// this declaration implies, validates the preserved counters against the
    /// Format-5 metadata invariants, and derives the Format-5 descriptor by
    /// parsing the carried Format-4 descriptor.
    fn validate_legacy_metadata(
        connection: &Connection<'_>,
        names: &StorageNames,
        target: &DeclarationTarget,
    ) -> Result<(Metadata, Vec<u8>, u64)> {
        let mut statement = connection.prepare(&format!(
            "SELECT format_version,CASE WHEN typeof(instance_id)='blob' THEN length(instance_id) ELSE -1 END,instance_id,CASE WHEN typeof(descriptor)='blob' THEN length(descriptor) ELSE -1 END,descriptor,revision,next_entry_id,next_job_id,next_generation_id,canonical_count,retired_count,published_generation,candidate_generation,active_job,next_fence,cleanup_entry_id,(SELECT count(*) FROM (SELECT singleton FROM {meta} LIMIT 2)) FROM {meta} WHERE singleton=1",
            meta = names.meta
        ))?;
        let result = (|| {
            if statement.step()? != Step::Row {
                return Err(corrupt("DiskANN Format-4 metadata row is missing"));
            }
            if nonnegative(&statement, 0, "format version")? != LEGACY_FORMAT_VERSION {
                return Err(corrupt(
                    "DiskANN Format-4 metadata format version changed during validation",
                ));
            }
            if nonnegative(&statement, 16, "metadata count")? != 1 {
                return Err(corrupt("DiskANN Format-4 metadata singleton count"));
            }
            let instance_id: [u8; 16] = bounded_blob(&statement, 1, 2, 16, true, "instance ID")?
                .try_into()
                .map_err(|_| corrupt("DiskANN instance ID"))?;
            let carried =
                bounded_blob(&statement, 3, 4, MAX_DESCRIPTOR_BYTES, false, "descriptor")?;
            let mut expected = codec::descriptor(&target.vector_space, &target.diskann);
            expected["format_version"] = json!(LEGACY_FORMAT_VERSION);
            codec::validate_descriptor(&carried, &expected)?;
            let metadata = Metadata {
                instance_id,
                revision: nonnegative(&statement, 5, "revision")?,
                next_entry_id: nonnegative(&statement, 6, "next entry")?,
                next_job_id: nonnegative(&statement, 7, "next job")?,
                next_generation_id: nonnegative(&statement, 8, "next generation")?,
                next_fence: nonnegative(&statement, 14, "next fence")?,
                cleanup_entry_id: nonnegative(&statement, 15, "cleanup entry ID")?,
                canonical_count: nonnegative(&statement, 9, "canonical count")?,
                retired_count: nonnegative(&statement, 10, "retired count")?,
                published_generation: optional_id(&statement, 11, "published generation")?,
                candidate_generation: optional_id(&statement, 12, "candidate generation")?,
                active_job: optional_id(&statement, 13, "active job")?,
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
                    "DiskANN Format-4 metadata counters/identities are inconsistent",
                ));
            }
            let next_fence = checked_increment(metadata.next_fence, "fence allocator")?;
            let descriptor = derive_format5_descriptor(&carried)?;
            // The derived bytes must be exactly the Format-5 descriptor this build
            // expects for the declaration; a future descriptor extension refuses
            // the upgrade instead of inventing fields.
            codec::validate_descriptor(
                &descriptor,
                &codec::descriptor(&target.vector_space, &target.diskann),
            )?;
            Ok((metadata, descriptor, next_fence))
        })();
        let finish = statement.finish();
        let out = result?;
        finish?;
        Ok(out)
    }

    fn optional_id(statement: &Statement<'_>, column: usize, name: &str) -> Result<Option<u64>> {
        if statement.column_is_null(column)? {
            Ok(None)
        } else {
            Ok(Some(nonnegative(statement, column, name)?))
        }
    }

    /// Parses the carried Format-4 descriptor and changes only format_version.
    /// Nothing else is synthesized: every other field is the carried bytes.
    fn derive_format5_descriptor(carried: &[u8]) -> Result<Vec<u8>> {
        let mut value: Json = serde_json::from_slice(carried)
            .map_err(|_| corrupt("DiskANN descriptor is invalid JSON"))?;
        let object = value
            .as_object_mut()
            .ok_or_else(|| corrupt("DiskANN descriptor is not a JSON object"))?;
        if object.get("format_version").and_then(Json::as_u64) != Some(LEGACY_FORMAT_VERSION) {
            return Err(corrupt("DiskANN descriptor format version is not Format4"));
        }
        object.insert("format_version".to_owned(), json!(FORMAT_VERSION));
        let bytes = serde_json::to_vec(&value).map_err(|e| IndexError::new(e.to_string()))?;
        if bytes.len() > MAX_DESCRIPTOR_BYTES {
            return Err(too_big("DiskANN descriptor too large"));
        }
        Ok(bytes)
    }

    fn require_rollback_journal(connection: &Connection<'_>, schema: &str) -> Result<()> {
        let mut statement = connection.prepare(&format!(
            "PRAGMA {}.journal_mode",
            quote_identifier(schema)?
        ))?;
        let result = (|| {
            if statement.step()? != Step::Row {
                return Err(corrupt("DiskANN could not read SQLite journal_mode"));
            }
            let mode = statement.column_text(0)?;
            if statement.step()? != Step::Done {
                return Err(corrupt("DiskANN journal_mode returned unexpected rows"));
            }
            Ok(mode)
        })();
        let finish = statement.finish();
        let mode = result?;
        finish?;
        codec::validate_journal_mode(&mode)
    }

    /// Runs the whole mutation in one savepoint transaction. RELEASE commits on
    /// success; any failure rolls back to the savepoint and preserves the
    /// primary error, so a failed upgrade leaves the Format-4 file untouched.
    fn execute_plan(connection: &Connection<'_>, plan: &UpgradePlan) -> Result<()> {
        connection.execute(&format!("SAVEPOINT {SAVEPOINT}"))?;
        let body = (|| {
            // Take the database write reservation before consuming one cached
            // plan value. Rewriting the validated singleton row dirties its page
            // and acquires the write lock (RESERVED in rollback-journal modes, the
            // single WAL writer lock in WAL mode); from here to RELEASE no other
            // connection can commit, so the re-read below is authoritative.
            connection.execute(&format!(
                "UPDATE {} SET revision=revision WHERE singleton=1",
                plan.names.qualified("diskann_meta")?
            ))?;
            if connection.changes() != 1 {
                return Err(corrupt("DiskANN upgrade metadata row changed"));
            }
            // Re-read the declaration and re-validate the preserved catalog and
            // metadata state under the write reservation. The pre-transaction
            // validation ran before any lock, so a supported writer could have
            // committed meanwhile; any difference must refuse without mutation
            // instead of rewriting Format-5 metadata from stale counters.
            let (reread_name, reread_sql, reread_target) =
                declaration(connection, &plan.names.schema, &plan.names.table)?;
            let reread_declaration = DeclarationSnapshot {
                name: reread_name,
                sql: reread_sql,
            };
            if plan.declaration != reread_declaration {
                return Err(intervening_writer());
            }
            let reread = validate_legacy(
                connection,
                &plan.names,
                plan.vector_bytes,
                plan.legacy_adjacency_bytes,
                plan.adjacency_bytes,
                &reread_target,
                reread_declaration,
            )?;
            ensure_snapshot_unchanged(&plan.preserved(), &reread.preserved())?;
            // The durable job/generation/snapshot state is abandoned whole: the
            // fence bump makes every stale lease unattachable, and the user
            // republishes through the normal fenced job machinery.
            for suffix in [
                "diskann_generations",
                "diskann_jobs",
                "diskann_snapshot_map",
            ] {
                connection.execute(&format!("DELETE FROM {}", plan.names.qualified(suffix)?))?;
            }
            connection.execute(&format!(
                "DROP TABLE {}",
                plan.names.qualified("diskann_meta")?
            ))?;
            connection.execute(&v5_create_sql(plan, "diskann_meta")?)?;
            connection.execute(&format!(
                "DROP TABLE {}",
                plan.names.qualified("diskann_graph_nodes")?
            ))?;
            connection.execute(&v5_create_sql(plan, "diskann_graph_nodes")?)?;
            connection.execute(&v5_create_sql(plan, "diskann_generation_retired")?)?;
            insert_meta_row(connection, plan)?;
            verify_upgraded(connection, plan)?;
            Ok(())
        })();
        if let Err(error) = body {
            rollback_savepoint(connection);
            return Err(error);
        }
        if let Err(error) = connection.execute(&format!("RELEASE {SAVEPOINT}")) {
            rollback_savepoint(connection);
            return Err(error);
        }
        Ok(())
    }

    /// Best-effort rollback that never replaces the primary error. A successful
    /// ROLLBACK TO discards every statement effect since SAVEPOINT; RELEASE then
    /// removes the savepoint so the connection returns to autocommit.
    fn rollback_savepoint(connection: &Connection<'_>) {
        let _ = connection.execute(&format!("ROLLBACK TO {SAVEPOINT}"));
        let _ = connection.execute(&format!("RELEASE {SAVEPOINT}"));
    }

    fn v5_create_sql(plan: &UpgradePlan, suffix: &str) -> Result<String> {
        format5_create_table_sql(&plan.names, suffix, plan.vector_bytes, plan.adjacency_bytes)?
            .ok_or_else(|| invalid(format!("DiskANN missing Format-5 layout {suffix}")))
    }

    /// Inserts the single Format-5 metadata row, preserving every carried
    /// allocator/counter/identity and abandoning the published/candidate/job
    /// pointers. next_fence was incremented before the transaction started.
    fn insert_meta_row(connection: &Connection<'_>, plan: &UpgradePlan) -> Result<()> {
        let mut insert = connection.prepare(&format!(
            "INSERT INTO {}(singleton,format_version,instance_id,descriptor,revision,next_entry_id,next_job_id,next_generation_id,canonical_count,retired_count,published_generation,candidate_generation,active_job,next_fence,cleanup_entry_id) VALUES(1,?1,?2,?3,?4,?5,?6,?7,?8,?9,NULL,NULL,NULL,?10,?11)",
            plan.names.meta
        ))?;
        insert.bind_i64(1, FORMAT_VERSION)?;
        insert.bind_blob(2, &plan.metadata.instance_id)?;
        insert.bind_blob(3, &plan.descriptor)?;
        insert.bind_i64(4, checked_id(plan.metadata.revision)?)?;
        insert.bind_i64(5, checked_id(plan.metadata.next_entry_id)?)?;
        insert.bind_i64(6, checked_id(plan.metadata.next_job_id)?)?;
        insert.bind_i64(7, checked_id(plan.metadata.next_generation_id)?)?;
        insert.bind_i64(8, checked_id(plan.metadata.canonical_count)?)?;
        insert.bind_i64(9, checked_id(plan.metadata.retired_count)?)?;
        insert.bind_i64(10, checked_id(plan.next_fence)?)?;
        insert.bind_i64(11, checked_id(plan.metadata.cleanup_entry_id)?)?;
        while insert.step()? == Step::Row {}
        insert.finish()?;
        if connection.changes() != 1 {
            return Err(corrupt("DiskANN upgrade metadata insertion count"));
        }
        Ok(())
    }

    /// Proves the new metadata row and the empty derived tables before RELEASE,
    /// so a partially applied rewrite can never commit.
    fn verify_upgraded(connection: &Connection<'_>, plan: &UpgradePlan) -> Result<()> {
        let mut metadata = connection.prepare(&format!(
            "SELECT format_version,instance_id,descriptor,(SELECT count(*) FROM (SELECT singleton FROM {meta} LIMIT 2)),revision,next_entry_id,next_job_id,next_generation_id,canonical_count,retired_count,published_generation,candidate_generation,active_job,next_fence,cleanup_entry_id FROM {meta} WHERE singleton=1",
            meta = plan.names.meta
        ))?;
        let result = (|| {
            if metadata.step()? != Step::Row {
                return Err(corrupt("DiskANN upgrade metadata row is missing"));
            }
            if nonnegative(&metadata, 0, "format version")? != FORMAT_VERSION as u64
                || nonnegative(&metadata, 3, "metadata count")? != 1
            {
                return Err(corrupt("DiskANN upgrade metadata row is not Format5"));
            }
            if metadata.column_type(1)? != ffi::SQLITE_BLOB as i32 {
                return Err(corrupt("DiskANN upgrade instance ID is not a BLOB"));
            }
            let instance = metadata.column_blob_owned(1)?;
            if instance.len() != 16 || instance != plan.metadata.instance_id {
                return Err(corrupt("DiskANN upgrade instance ID changed"));
            }
            if metadata.column_type(2)? != ffi::SQLITE_BLOB as i32 {
                return Err(corrupt("DiskANN upgrade descriptor is not a BLOB"));
            }
            let descriptor = metadata.column_blob_owned(2)?;
            if descriptor.len() > MAX_DESCRIPTOR_BYTES || descriptor != plan.descriptor {
                return Err(corrupt("DiskANN upgrade descriptor changed"));
            }
            if nonnegative(&metadata, 4, "revision")? != plan.metadata.revision
                || nonnegative(&metadata, 5, "next entry")? != plan.metadata.next_entry_id
                || nonnegative(&metadata, 6, "next job")? != plan.metadata.next_job_id
                || nonnegative(&metadata, 7, "next generation")? != plan.metadata.next_generation_id
                || nonnegative(&metadata, 8, "canonical count")? != plan.metadata.canonical_count
                || nonnegative(&metadata, 9, "retired count")? != plan.metadata.retired_count
                || nonnegative(&metadata, 13, "next fence")? != plan.next_fence
                || nonnegative(&metadata, 14, "cleanup entry ID")? != plan.metadata.cleanup_entry_id
                || !metadata.column_is_null(10)?
                || !metadata.column_is_null(11)?
                || !metadata.column_is_null(12)?
            {
                return Err(corrupt("DiskANN upgrade metadata values changed"));
            }
            if metadata.step()? != Step::Done {
                return Err(corrupt("DiskANN upgrade metadata singleton is not unique"));
            }
            Ok(())
        })();
        let finish = metadata.finish();
        result.and(finish)?;
        let mut counts = connection.prepare(&format!(
            "SELECT (SELECT count(*) FROM {}),(SELECT count(*) FROM {}),(SELECT count(*) FROM {}),(SELECT count(*) FROM {}),(SELECT count(*) FROM {})",
            plan.names.generations,
            plan.names.jobs,
            plan.names.snapshot_map,
            plan.names.graph_nodes,
            plan.names.generation_retired
        ))?;
        let result = (|| {
            if counts.step()? != Step::Row {
                return Err(corrupt("DiskANN upgrade derived table counts are missing"));
            }
            for column in 0..5 {
                if nonnegative(&counts, column, "derived table row count")? != 0 {
                    return Err(corrupt("DiskANN upgrade derived tables are not empty"));
                }
            }
            if counts.step()? != Step::Done {
                return Err(corrupt(
                    "DiskANN upgrade derived table counts are not unique",
                ));
            }
            Ok(())
        })();
        let finish = counts.finish();
        result.and(finish)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::vector_space::{DistanceType, NamedVectorSpace, VectorType};

        fn space() -> NamedVectorSpace {
            NamedVectorSpace {
                vector_name: "embedding".into(),
                dim: 3,
                distance_type: DistanceType::Cosine,
                vector_type: VectorType::Float32,
            }
        }

        #[test]
        fn frozen_format4_layouts_cover_the_legacy_shadow_set() {
            assert_eq!(LEGACY_LAYOUTS.len(), 8);
            for layout in LEGACY_LAYOUTS {
                assert!(layout.definition.contains("CHECK("));
                assert!(!layout.suffix.contains("generation_retired"));
            }
            let meta = LEGACY_LAYOUTS
                .iter()
                .find(|layout| layout.suffix == "diskann_meta")
                .unwrap();
            assert!(meta.definition.contains("CHECK(format_version=4)"));
            let graph = LEGACY_LAYOUTS
                .iter()
                .find(|layout| layout.suffix == "diskann_graph_nodes")
                .unwrap();
            assert!(graph.without_rowid);
            assert_eq!(graph.unique.len(), 2);
            assert!(graph.definition.contains("frozen_vector BLOB"));
            assert!(graph.definition.contains("length(neighbors)%8=0"));
            let sql = expected_legacy_sql("v_diskann_meta", meta, 12, 104).unwrap();
            assert!(sql.starts_with("CREATE TABLE \"v_diskann_meta\" ("));
            assert!(!sql.contains("{vector_bytes}"));
            assert!(!sql.contains("{adjacency_bytes}"));
            let graph_sql = expected_legacy_sql("v_diskann_graph_nodes", graph, 12, 104).unwrap();
            assert!(graph_sql.ends_with(") WITHOUT ROWID"));
            assert!(graph_sql.contains("length(frozen_vector)=12"));
            assert!(graph_sql.contains("length(neighbors)<=104"));
        }

        #[test]
        fn derived_format5_descriptor_only_changes_the_version() {
            let space = space();
            let options = crate::index_options::DiskAnnOptions::default();
            let mut format4 = codec::descriptor(&space, &options);
            format4["format_version"] = json!(LEGACY_FORMAT_VERSION);
            let carried = serde_json::to_vec(&format4).unwrap();
            let derived = derive_format5_descriptor(&carried).unwrap();
            codec::validate_descriptor(&derived, &codec::descriptor(&space, &options)).unwrap();
            let value: Json = serde_json::from_slice(&derived).unwrap();
            assert_eq!(value["format_version"], json!(FORMAT_VERSION));
            assert_eq!(value["dimension"], json!(3));
            assert_eq!(value["metric"], json!("cosine"));
            assert!(derive_format5_descriptor(b"not-json").is_err());
            let wrong_version = serde_json::to_vec(&json!({"format_version": 6})).unwrap();
            assert!(derive_format5_descriptor(&wrong_version).is_err());
            let not_object = serde_json::to_vec(&json!([1, 2, 3])).unwrap();
            assert!(derive_format5_descriptor(&not_object).is_err());
        }

        fn preserved(revision: u64) -> PreservedSnapshot {
            PreservedSnapshot {
                metadata: Metadata {
                    instance_id: [7; 16],
                    revision,
                    next_entry_id: 10,
                    next_job_id: 2,
                    next_generation_id: 3,
                    next_fence: 5,
                    cleanup_entry_id: 0,
                    canonical_count: 2,
                    retired_count: 0,
                    published_generation: Some(1),
                    candidate_generation: None,
                    active_job: None,
                },
                descriptor: b"{\"format_version\":5}".to_vec(),
                next_fence: 6,
                declaration: DeclarationSnapshot {
                    name: "v".to_owned(),
                    sql: "CREATE VIRTUAL TABLE v USING vectorlite(embedding float32[3] cosine,diskann())"
                        .to_owned(),
                },
            }
        }

        #[test]
        fn intervening_writer_snapshot_change_is_refused_with_busy() {
            let validated = preserved(3);
            // The no-writer case keeps the existing behaviour: the re-read equals
            // the validated snapshot and the upgrade proceeds.
            assert!(ensure_snapshot_unchanged(&validated, &validated).is_ok());
            // Every preserved value the SAVEPOINT body consumes must refuse when
            // it moved: a supported INSERT bumps revision, next_entry_id and
            // canonical_count, so rewriting Format-5 metadata from the stale plan
            // would mint a revision the new entry already cites.
            let mutations: [fn(&mut PreservedSnapshot); 14] = [
                |snapshot| snapshot.metadata.revision += 1,
                |snapshot| snapshot.metadata.next_entry_id += 1,
                |snapshot| snapshot.metadata.canonical_count += 1,
                |snapshot| snapshot.metadata.retired_count += 1,
                |snapshot| snapshot.metadata.next_job_id += 1,
                |snapshot| snapshot.metadata.next_generation_id += 1,
                |snapshot| snapshot.metadata.next_fence += 1,
                |snapshot| snapshot.metadata.cleanup_entry_id += 1,
                |snapshot| snapshot.metadata.instance_id[0] ^= 1,
                |snapshot| snapshot.metadata.published_generation = None,
                |snapshot| snapshot.descriptor.push(b' '),
                |snapshot| snapshot.next_fence += 1,
                |snapshot| snapshot.declaration.name = "w".to_owned(),
                |snapshot| snapshot.declaration.sql.push(' '),
            ];
            for mutate in mutations {
                let mut reread = validated.clone();
                mutate(&mut reread);
                let error = ensure_snapshot_unchanged(&validated, &reread)
                    .expect_err("changed preserved state must refuse the upgrade");
                assert_eq!(error.code, ffi::SQLITE_BUSY as i32);
                assert!(error.message.contains("intervening writer"));
            }
        }
    }
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
