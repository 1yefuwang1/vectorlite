//! Connection-owned ephemeral authority. No SQLite handles or SQL progress live here.
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::diskann_worker::{
    self as worker, AttemptIdentity, CandidateHandle, CandidateSessions, ConnectionNonce,
    IndexBinding, IndexUuid, SchemaNonce, TableLifetimeNonce,
};
use crate::ffi;
use crate::index_error::IndexError;
use crate::index_options::DiskAnnOptions;
use crate::vector_space::NamedVectorSpace;

type Result<T> = std::result::Result<T, IndexError>;
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn nonce() -> [u8; 16] {
    let mut bytes = [0; 16];
    // SAFETY: the initialized host fills this owned buffer synchronously; no pointer escapes.
    unsafe { ffi::randomness(16, bytes.as_mut_ptr().cast()) };
    bytes
}
pub(crate) fn worker_error(error: worker::WorkerError) -> IndexError {
    use worker::WorkerErrorKind::*;
    let code = match error.kind() {
        Allocation => ffi::SQLITE_NOMEM,
        BudgetExceeded | CapacityExhausted | TooManyIndexes => ffi::SQLITE_TOOBIG,
        ConcurrentBuild | RebuildNeeded | TicketStale => ffi::SQLITE_BUSY,
        Cancelled => ffi::SQLITE_ABORT,
        InvalidTopology | DigestMismatch | MetadataMismatch => ffi::SQLITE_CORRUPT,
        _ => ffi::SQLITE_ERROR,
    };
    IndexError::with_code(
        code as i32,
        format!("DiskANN rebuild-needed/session: {error}"),
    )
}
#[derive(Clone)]
pub(crate) struct TargetMetadata {
    pub(crate) schema: String,
    pub(crate) table: String,
    pub(crate) space: NamedVectorSpace,
    pub(crate) options: DiskAnnOptions,
    pub(crate) instance: [u8; 16],
    pub(crate) binding: IndexBinding,
}
#[derive(Clone)]
struct AttemptRecord {
    job: u64,
    fence: u64,
    identity: Option<AttemptIdentity>,
    handle: Option<CandidateHandle>,
    statistics: Option<String>,
}
struct TargetEntry {
    metadata: TargetMetadata,
    leases: usize,
}
struct SchemaLife {
    nonce: SchemaNonce,
    live: usize,
}
#[derive(Default)]
struct OwnedState {
    schemas: HashMap<String, SchemaLife>,
    targets: HashMap<(String, String), TargetEntry>,
    attempts: HashMap<IndexBinding, AttemptRecord>,
    sessions: Option<Arc<CandidateSessions>>,
}
pub(crate) struct Registration {
    pub(crate) owner: [u8; 16],
    connection: ConnectionNonce,
    state: Mutex<OwnedState>,
}
fn key(schema: &str, table: &str) -> (String, String) {
    (schema.to_ascii_lowercase(), table.to_ascii_lowercase())
}
impl Registration {
    pub(crate) fn new() -> Result<Arc<Self>> {
        let owner = nonce();
        let connection =
            ConnectionNonce::parse(&format!("conn:{}", hex(&owner))).map_err(worker_error)?;
        Ok(Arc::new(Self {
            owner,
            connection,
            state: Mutex::new(OwnedState::default()),
        }))
    }
    fn lock(&self) -> Result<MutexGuard<'_, OwnedState>> {
        self.state
            .lock()
            .map_err(|_| IndexError::new("DiskANN owned registration poisoned"))
    }
    pub(crate) fn connection_nonce(&self) -> &ConnectionNonce {
        &self.connection
    }
    /// Called only after this module's xCreate/xConnect validated the current store.
    pub(crate) fn connect(
        self: &Arc<Self>,
        schema: &str,
        table: &str,
        space: NamedVectorSpace,
        options: DiskAnnOptions,
        instance: [u8; 16],
    ) -> Result<TargetLease> {
        if schema.len() > 1024 || table.len() > 1024 || space.vector_name.len() > 1024 {
            return Err(IndexError::with_code(
                ffi::SQLITE_TOOBIG as i32,
                "DiskANN declaration identity byte bound",
            ));
        }
        let k = key(schema, table);
        let mut s = self.lock()?;
        if s.targets.len() >= 256 && !s.targets.contains_key(&k) {
            return Err(IndexError::with_code(
                ffi::SQLITE_TOOBIG as i32,
                "DiskANN target registration limit",
            ));
        }
        // SQLite can xConnect a replacement wrapper while an older wrapper still
        // lives during schema reparses. Exact owned descriptor+instance equality
        // shares the runtime lifetime, never a SQL handle. Last lease retires it.
        let reusable = s
            .targets
            .get(&k)
            .filter(|e| {
                let t = &e.metadata;
                t.instance == instance
                    && t.options == options
                    && t.space.dim == space.dim
                    && t.space.vector_name == space.vector_name
                    && t.space.distance_type == space.distance_type
                    && t.space.vector_type == space.vector_type
            })
            .map(|e| e.metadata.binding.clone());
        let replaced = if reusable.is_none() {
            s.targets.remove(&k)
        } else {
            None
        };
        let schema_key = k.0.clone();
        let sch = s.schemas.entry(schema_key).or_insert_with(|| SchemaLife {
            nonce: SchemaNonce::parse(&format!("sch:{}", hex(&nonce()))).expect("generated nonce"),
            live: 0,
        });
        sch.live += 1;
        let binding = match reusable {
            Some(binding) => binding,
            None => IndexBinding::new(
                sch.nonce.clone(),
                TableLifetimeNonce::parse(&format!("life:{}", hex(&nonce())))
                    .map_err(worker_error)?,
                IndexUuid::parse(&format!("idx:{}", hex(&instance))).map_err(worker_error)?,
            ),
        };
        let metadata = TargetMetadata {
            schema: schema.into(),
            table: table.into(),
            space,
            options,
            instance,
            binding: binding.clone(),
        };
        s.targets
            .entry(k.clone())
            .and_modify(|e| e.leases += 1)
            .or_insert(TargetEntry {
                metadata,
                leases: 1,
            });
        let sessions = s.sessions.clone();
        let old_identity = replaced
            .as_ref()
            .and_then(|old| s.attempts.remove(&old.metadata.binding))
            .and_then(|r| r.identity);
        drop(s);
        if let (Some(identity), Some(sessions)) = (old_identity, sessions) {
            sessions.cancel(&identity).map_err(worker_error)?;
        }
        Ok(TargetLease {
            registration: Arc::clone(self),
            key: k,
            binding,
        })
    }
    pub(crate) fn target(&self, schema: &str, table: &str) -> Result<TargetMetadata> {
        self.lock()?
            .targets
            .get(&key(schema, table))
            .map(|e| e.metadata.clone())
            .ok_or_else(|| IndexError::new("DiskANN target has no live vectorlite xConnect lease"))
    }
    pub(crate) fn verify_target(&self, target: &TargetMetadata) -> Result<()> {
        if self.target(&target.schema, &target.table)?.binding != target.binding {
            return Err(IndexError::new(
                "DiskANN rebuild-needed: target incarnation changed",
            ));
        }
        Ok(())
    }
    /// Conservative owned frontend metadata coexistence estimate, not RSS enforcement.
    pub(crate) fn metadata_reservation(&self) -> Result<usize> {
        let s = self.lock()?;
        let mut bytes = 64usize * 1024;
        let mut add = |n: usize| -> Result<()> {
            bytes = bytes.checked_add(n).ok_or_else(|| {
                IndexError::with_code(
                    ffi::SQLITE_TOOBIG as i32,
                    "frontend metadata reservation overflow",
                )
            })?;
            Ok(())
        };
        add(s
            .targets
            .capacity()
            .saturating_mul(std::mem::size_of::<TargetMetadata>() + 256))?;
        add(s.schemas.capacity().saturating_mul(256))?;
        add(s
            .attempts
            .capacity()
            .saturating_mul(std::mem::size_of::<AttemptRecord>() + 2048))?;
        for ((schema, table), entry) in &s.targets {
            let t = &entry.metadata;
            add(schema.capacity())?;
            add(table.capacity())?;
            add(t.schema.capacity())?;
            add(t.table.capacity())?;
            add(t.space.vector_name.capacity())?;
        }
        for r in s.attempts.values() {
            if let Some(stats) = &r.statistics {
                add(stats.capacity())?;
            }
        }
        Ok(bytes)
    }
    pub(crate) fn existing_sessions(&self) -> Result<Option<Arc<CandidateSessions>>> {
        Ok(self.lock()?.sessions.clone())
    }
    pub(crate) fn sessions(&self, budget: Option<u64>) -> Result<Arc<CandidateSessions>> {
        let mut s = self.lock()?;
        if let Some(sessions) = &s.sessions {
            return Ok(Arc::clone(sessions));
        }
        let budget = budget
            .ok_or_else(|| IndexError::new("DiskANN rebuild-needed: candidate RAM unavailable"))?;
        let sessions = Arc::new(
            CandidateSessions::new(self.connection.clone(), budget).map_err(worker_error)?,
        );
        s.sessions = Some(Arc::clone(&sessions));
        Ok(sessions)
    }
    /// Pure claim identity only. SQL ownership is always reread independently.
    pub(crate) fn claimed(&self, binding: &IndexBinding, job: u64, fence: u64) -> Result<()> {
        let mut s = self.lock()?;
        if let Some(old) = s.attempts.get(binding) {
            if old.job == job && old.fence == fence {
                return Ok(());
            }
        }
        let sessions = s.sessions.clone();
        let old_identity = s.attempts.get(binding).and_then(|r| r.identity.clone());
        s.attempts.insert(
            binding.clone(),
            AttemptRecord {
                job,
                fence,
                identity: None,
                handle: None,
                statistics: None,
            },
        );
        drop(s);
        if let (Some(sessions), Some(identity)) = (sessions, old_identity) {
            sessions.cancel(&identity).map_err(worker_error)?;
        }
        Ok(())
    }
    pub(crate) fn ready(
        &self,
        binding: &IndexBinding,
        job: u64,
        fence: u64,
    ) -> Result<Option<CandidateHandle>> {
        let s = self.lock()?;
        let record = s
            .attempts
            .get(binding)
            .ok_or_else(|| IndexError::new("DiskANN rebuild-needed: claim target/session lost"))?;
        if record.job != job || record.fence != fence {
            return Err(IndexError::new("DiskANN rebuild-needed: stale claim"));
        }
        if record.identity.is_some() && record.handle.is_none() {
            return Err(IndexError::new(
                "DiskANN rebuild-needed: previous native attempt lost; restart with fresh fence",
            ));
        }
        Ok(record.handle.clone())
    }
    pub(crate) fn starting(&self, identity: &AttemptIdentity, job: u64, fence: u64) -> Result<()> {
        let mut s = self.lock()?;
        let record = s
            .attempts
            .get_mut(identity.binding())
            .ok_or_else(|| IndexError::new("DiskANN claim missing"))?;
        if record.job != job || record.fence != fence || record.identity.is_some() {
            return Err(IndexError::new(
                "DiskANN rebuild-needed: attempt already started",
            ));
        }
        record.identity = Some(identity.clone());
        Ok(())
    }
    pub(crate) fn completed(&self, handle: CandidateHandle, statistics: String) -> Result<()> {
        let mut s = self.lock()?;
        let record = s
            .attempts
            .get_mut(handle.identity().binding())
            .ok_or_else(|| IndexError::new("DiskANN target lost after compute"))?;
        if record.identity.as_ref() != Some(handle.identity()) {
            return Err(IndexError::new("DiskANN attempt lost after compute"));
        }
        record.handle = Some(handle);
        record.statistics = Some(statistics);
        Ok(())
    }
    pub(crate) fn statistics(&self, handle: &CandidateHandle) -> Result<Option<String>> {
        Ok(self
            .lock()?
            .attempts
            .get(handle.identity().binding())
            .filter(|r| r.identity.as_ref() == Some(handle.identity()))
            .and_then(|r| r.statistics.clone()))
    }
    pub(crate) fn candidate(
        &self,
        binding: &IndexBinding,
        job: u64,
        fence: u64,
        token: &str,
    ) -> Result<CandidateHandle> {
        let handle = self
            .ready(binding, job, fence)?
            .ok_or_else(|| IndexError::new("DiskANN rebuild-needed: no ready native candidate"))?;
        let parsed = worker::CandidateToken::parse(token).map_err(worker_error)?;
        let sessions = self.sessions(None)?;
        sessions
            .resume(handle.identity().clone(), parsed)
            .map_err(worker_error)
    }
    pub(crate) fn forget(&self, binding: &IndexBinding) -> Result<()> {
        let mut s = self.lock()?;
        let identity = s.attempts.remove(binding).and_then(|r| r.identity);
        let sessions = s.sessions.clone();
        drop(s);
        if let (Some(sessions), Some(identity)) = (sessions, identity) {
            sessions.cancel(&identity).map_err(worker_error)?;
        }
        Ok(())
    }
}
/// No pointer authority: a lease retains only declaration identity and liveness.
pub(crate) struct TargetLease {
    registration: Arc<Registration>,
    key: (String, String),
    binding: IndexBinding,
}
impl Drop for TargetLease {
    fn drop(&mut self) {
        if let Ok(mut s) = self.registration.lock() {
            let retired = if let Some(entry) = s.targets.get_mut(&self.key) {
                if entry.metadata.binding == self.binding {
                    entry.leases -= 1;
                    entry.leases == 0
                } else {
                    false
                }
            } else {
                false
            };
            let identity = if retired {
                s.targets.remove(&self.key);
                s.attempts.remove(&self.binding).and_then(|r| r.identity)
            } else {
                None
            };
            if let Some(sch) = s.schemas.get_mut(&self.key.0) {
                sch.live -= 1;
                if sch.live == 0 {
                    s.schemas.remove(&self.key.0);
                }
            }
            let sessions = s.sessions.clone();
            drop(s);
            if let (Some(sessions), Some(identity)) = (sessions, identity) {
                let _ = sessions.cancel(&identity);
            }
        }
    }
}
impl Drop for Registration {
    fn drop(&mut self) {
        if let Ok(state) = self.state.get_mut() {
            if let Some(sessions) = &state.sessions {
                let _ = sessions.close();
            }
        }
    }
}
