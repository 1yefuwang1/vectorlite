//! Scoped, per-connection candidate bookkeeping for deferred DiskANN publication.
//!
//! This module is the T4a owned-RAM seam between the qualified native OneShot
//! builder and the SQL owner. It is deliberately independent of the SQL schema:
//! it stores no SQLite handle, statement, value, rowid, progress counter, source
//! vector or SQL callback, and it performs no SQL calls. The SQL owner supplies
//! opaque identity tokens and native ordinal/version metadata; the module owns
//! only an immutable native topology, that metadata and bounded identity labels.
//!
//! Trust boundary. A candidate is authorized by the exact combination of:
//! worker-connection nonce, schema nonce, table-lifetime nonce, index UUID,
//! committed job id, fence, attempt, and expected input/config digests. A
//! copied database UUID alone is never authority, and no process-global map is
//! keyed by a database address. A schema/table lifetime nonce must be generated
//! by the SQL owner for the actual open/attach/create incarnation; it must not
//! be a value read back from a copied database file. This module rejects the
//! same index UUID observed under a different schema/lifetime binding and
//! exposes [CandidateSessions::invalidate_index] for DROP/recreate/RENAME and
//! attached-copy invalidation hooks.
//!
//! Ownership. One current build/candidate is allowed per exact index binding
//! per connection. Native work receives only owned data. A ready candidate is
//! an immutable [Arc] topology plus owned metadata; access returns owned copies
//! and never borrows candidate memory across a caller SQL boundary. Locks are
//! held only for map/state inspection, candidate [Arc] cloning, token generation
//! and bounded accounting; native topology validation and chunk/materialization
//! copies happen outside the lock, and no SQL call can occur under it.
//!
//! Cancellation. [CandidateSessions::cancel] signals the native
//! [Cancellation] and releases retained RAM; it is cooperative and cannot
//! preempt arbitrary native CPU. The SQL owner must independently validate its
//! own committed fence before treating a candidate as authoritative.
//!
//! Memory. Admission of native input/provider/graph/workspace is the native
//! module's job. This module bounds the retained candidate estimate: native
//! admission retained upper bound plus metadata Vec capacity, identity/token
//! label bytes and a fixed candidate overhead. These are estimates, not RSS
//! limits; fallible allocations report [WorkerError::Allocation] separately from
//! an admitted-budget rejection ([WorkerError::BudgetExceeded]).
//!
//! This is partial T4a: root registration, the Format 4 candidate SQL carrier,
//! import/publication and the final transaction-free wrapper are separate work.
#![forbid(unsafe_code)]
#![allow(dead_code)] // T4a seam: root registration lands after storage integration.

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem::size_of;
use std::sync::{Arc, Mutex};

use crate::diskann_build::{BuildMetric, Cancellation, OwnedTopology};

/// Maximum native nodes returned by one [CandidateSessions::copy_chunk] call.
pub const MAX_CHUNK_NODES: u32 = 256;
/// Maximum distinct index bindings retained by one connection session.
pub const MAX_INDEX_SLOTS: usize = 64;
/// Maximum opaque token body length after its role prefix.
pub const MAX_TOKEN_BODY_BYTES: usize = 64;
/// Conservative fixed candidate bookkeeping overhead included in the estimate.
const CANDIDATE_OVERHEAD_BYTES: u64 = 512;

type WorkerResult<T> = Result<T, WorkerError>;

// ---------------------------------------------------------------------------
// Opaque text tokens
// ---------------------------------------------------------------------------

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn validate_token_body(role: &'static str, body: &str) -> WorkerResult<()> {
    if body.is_empty() || body.len() > MAX_TOKEN_BODY_BYTES {
        return Err(WorkerError::InvalidToken { role });
    }
    if !body
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(WorkerError::InvalidToken { role });
    }
    Ok(())
}

fn build_token(prefix: &str, body: &str) -> WorkerResult<String> {
    let mut text = String::new();
    text.try_reserve_exact(prefix.len().saturating_add(body.len()))
        .map_err(|_| WorkerError::Allocation {
            what: "identity token",
        })?;
    text.push_str(prefix);
    text.push_str(body);
    Ok(text)
}

fn split_token<'a>(
    role: &'static str,
    prefix: &'static str,
    text: &'a str,
) -> WorkerResult<&'a str> {
    let body = text
        .strip_prefix(prefix)
        .ok_or(WorkerError::InvalidToken { role })?;
    validate_token_body(role, body)?;
    Ok(body)
}

macro_rules! role_token {
    ($name:ident, $role:expr, $prefix:expr) => {
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            /// Parses one canonical opaque token; prefixes and characters are strict.
            pub fn parse(text: &str) -> WorkerResult<Self> {
                let body = split_token($role, $prefix, text)?;
                Self::from_body(body)
            }

            fn from_body(body: &str) -> WorkerResult<Self> {
                validate_token_body($role, body)?;
                Ok(Self(build_token($prefix, body)?))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            fn label_bytes(&self) -> u64 {
                self.0.capacity() as u64
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }
    };
}

role_token!(ConnectionNonce, "connection nonce", "conn:");
role_token!(SchemaNonce, "schema nonce", "sch:");
role_token!(TableLifetimeNonce, "table lifetime nonce", "life:");
role_token!(IndexUuid, "index UUID", "idx:");
role_token!(JobId, "job id", "job:");
role_token!(FenceToken, "fence token", "fence:");
role_token!(AttemptToken, "attempt token", "attempt:");
role_token!(CandidateToken, "candidate token", "cand:");

impl CandidateToken {
    fn generate(connection: &ConnectionNonce, generation: u64) -> WorkerResult<Self> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        connection.as_str().hash(&mut hasher);
        let tag = hasher.finish();
        let mut body = String::new();
        body.try_reserve_exact(36)
            .map_err(|_| WorkerError::Allocation {
                what: "candidate token",
            })?;
        use std::fmt::Write as _;
        write!(&mut body, "{generation:016x}-{tag:016x}").map_err(|_| WorkerError::Allocation {
            what: "candidate token",
        })?;
        Self::from_body(&body)
    }
}

/// Strict 32-byte digest rendered as exactly 64 lowercase hexadecimal characters.
///
/// This is a comparison token, not a signature: callers obtain the expected
/// values from their committed snapshot/config identity or from the qualified
/// native builder. No cryptographic dependency is introduced here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Digest32([u8; 32]);

impl Digest32 {
    pub const ZERO: Self = Self([0; 32]);

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Parses exactly 64 lowercase hexadecimal characters.
    pub fn parse(text: &str) -> WorkerResult<Self> {
        let bytes = text.as_bytes();
        if bytes.len() != 64 {
            return Err(WorkerError::InvalidDigest);
        }
        let mut output = [0u8; 32];
        for (index, slot) in output.iter_mut().enumerate() {
            let high = hex_value(bytes[index * 2]).ok_or(WorkerError::InvalidDigest)?;
            let low = hex_value(bytes[index * 2 + 1]).ok_or(WorkerError::InvalidDigest)?;
            *slot = (high << 4) | low;
        }
        Ok(Self(output))
    }

    fn label_bytes(&self) -> u64 {
        size_of::<Self>() as u64
    }
}

impl fmt::Display for Digest32 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/// Exact schema/table incarnation binding for one index instance.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IndexBinding {
    schema: SchemaNonce,
    table_lifetime: TableLifetimeNonce,
    index: IndexUuid,
}

impl IndexBinding {
    pub fn new(schema: SchemaNonce, table_lifetime: TableLifetimeNonce, index: IndexUuid) -> Self {
        Self {
            schema,
            table_lifetime,
            index,
        }
    }

    pub fn schema(&self) -> &SchemaNonce {
        &self.schema
    }

    pub fn table_lifetime(&self) -> &TableLifetimeNonce {
        &self.table_lifetime
    }

    pub fn index(&self) -> &IndexUuid {
        &self.index
    }

    fn label_bytes(&self) -> u64 {
        self.schema
            .label_bytes()
            .saturating_add(self.table_lifetime.label_bytes())
            .saturating_add(self.index.label_bytes())
    }
}

/// Full fenced attempt identity supplied by the SQL owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttemptIdentity {
    connection: ConnectionNonce,
    binding: IndexBinding,
    job: JobId,
    fence: FenceToken,
    attempt: AttemptToken,
    input_digest: Digest32,
    config_digest: Digest32,
}

impl AttemptIdentity {
    pub fn new(
        connection: ConnectionNonce,
        binding: IndexBinding,
        job: JobId,
        fence: FenceToken,
        attempt: AttemptToken,
        input_digest: Digest32,
        config_digest: Digest32,
    ) -> Self {
        Self {
            connection,
            binding,
            job,
            fence,
            attempt,
            input_digest,
            config_digest,
        }
    }

    pub fn connection(&self) -> &ConnectionNonce {
        &self.connection
    }

    pub fn binding(&self) -> &IndexBinding {
        &self.binding
    }

    pub fn job(&self) -> &JobId {
        &self.job
    }

    pub fn fence(&self) -> &FenceToken {
        &self.fence
    }

    pub fn attempt(&self) -> &AttemptToken {
        &self.attempt
    }

    pub fn input_digest(&self) -> Digest32 {
        self.input_digest
    }

    pub fn config_digest(&self) -> Digest32 {
        self.config_digest
    }

    fn label_bytes(&self) -> u64 {
        self.binding
            .label_bytes()
            .saturating_add(self.connection.label_bytes())
            .saturating_add(self.job.label_bytes())
            .saturating_add(self.fence.label_bytes())
            .saturating_add(self.attempt.label_bytes())
            .saturating_add(self.input_digest.label_bytes())
            .saturating_add(self.config_digest.label_bytes())
    }
}

// ---------------------------------------------------------------------------
// Snapshot metadata supplied by the SQL owner
// ---------------------------------------------------------------------------

/// One immutable snapshot ordinal/version identity. No rowid or SQL type is
/// interpreted here; the SQL owner resolves entries to canonical rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SnapshotNode {
    pub entry: u64,
    pub version: u64,
}

impl SnapshotNode {
    pub const fn new(entry: u64, version: u64) -> Self {
        Self { entry, version }
    }
}

/// One node identity per native ordinal 0..population-1. The frozen root native
/// ordinal (population) is not present; it is represented by the topology root
/// vector and has no source entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotMeta {
    nodes: Vec<SnapshotNode>,
}

impl SnapshotMeta {
    pub fn new(population: u32, nodes: Vec<SnapshotNode>) -> WorkerResult<Self> {
        let expected = usize::try_from(population).map_err(|_| WorkerError::MetadataMismatch {
            detail: "population exceeds usize",
        })?;
        if nodes.len() != expected {
            return Err(WorkerError::MetadataMismatch {
                detail: "snapshot metadata length differs from population",
            });
        }
        Ok(Self { nodes })
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn node(&self, ordinal: u32) -> Option<SnapshotNode> {
        self.nodes.get(usize::try_from(ordinal).ok()?).copied()
    }

    fn retained_bytes(&self) -> u64 {
        (self.nodes.capacity() as u64).saturating_mul(size_of::<SnapshotNode>() as u64)
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Stable error category for the SQL owner's SQLite/status mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerErrorKind {
    InvalidToken,
    InvalidDigest,
    InvalidBudget,
    InvalidConfig,
    OtherConnection,
    OtherBinding,
    ConcurrentBuild,
    RebuildNeeded,
    Cancelled,
    Closed,
    TooManyIndexes,
    CapacityExhausted,
    BudgetExceeded,
    Allocation,
    InvalidTopology,
    DigestMismatch,
    MetadataMismatch,
    ChunkRange,
    TicketStale,
    Poisoned,
}

/// Why a candidate can no longer be used and a fresh fenced attempt is needed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RebuildCause {
    StaleAttempt,
    StaleBinding,
    LostCandidate,
    Invalidated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbandonReport {
    ReleasedBuild,
    NotCurrent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CancelReport {
    pub had_in_flight: bool,
    pub released_candidate: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidationReport {
    pub cancelled_builds: usize,
    pub released_candidates: usize,
}

/// Granular, dependency-free error returned by this module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkerError {
    InvalidToken { role: &'static str },
    InvalidDigest,
    InvalidBudget,
    InvalidConfig { detail: &'static str },
    OtherConnection,
    OtherBinding { detail: &'static str },
    ConcurrentBuild,
    RebuildNeeded { cause: RebuildCause },
    Cancelled,
    Closed,
    TooManyIndexes { limit: usize },
    CapacityExhausted { what: &'static str },
    BudgetExceeded { requested: u64, limit: u64 },
    Allocation { what: &'static str },
    InvalidTopology { message: String },
    DigestMismatch,
    MetadataMismatch { detail: &'static str },
    ChunkRange { start: u32, count: u32, nodes: u32 },
    TicketStale,
    Poisoned,
}

impl WorkerError {
    pub fn kind(&self) -> WorkerErrorKind {
        match self {
            Self::InvalidToken { .. } => WorkerErrorKind::InvalidToken,
            Self::InvalidDigest => WorkerErrorKind::InvalidDigest,
            Self::InvalidBudget => WorkerErrorKind::InvalidBudget,
            Self::InvalidConfig { .. } => WorkerErrorKind::InvalidConfig,
            Self::OtherConnection => WorkerErrorKind::OtherConnection,
            Self::OtherBinding { .. } => WorkerErrorKind::OtherBinding,
            Self::ConcurrentBuild => WorkerErrorKind::ConcurrentBuild,
            Self::RebuildNeeded { .. } => WorkerErrorKind::RebuildNeeded,
            Self::Cancelled => WorkerErrorKind::Cancelled,
            Self::Closed => WorkerErrorKind::Closed,
            Self::TooManyIndexes { .. } => WorkerErrorKind::TooManyIndexes,
            Self::CapacityExhausted { .. } => WorkerErrorKind::CapacityExhausted,
            Self::BudgetExceeded { .. } => WorkerErrorKind::BudgetExceeded,
            Self::Allocation { .. } => WorkerErrorKind::Allocation,
            Self::InvalidTopology { .. } => WorkerErrorKind::InvalidTopology,
            Self::DigestMismatch => WorkerErrorKind::DigestMismatch,
            Self::MetadataMismatch { .. } => WorkerErrorKind::MetadataMismatch,
            Self::ChunkRange { .. } => WorkerErrorKind::ChunkRange,
            Self::TicketStale => WorkerErrorKind::TicketStale,
            Self::Poisoned => WorkerErrorKind::Poisoned,
        }
    }

    pub fn rebuild_cause(&self) -> Option<RebuildCause> {
        match self {
            Self::RebuildNeeded { cause } => Some(*cause),
            _ => None,
        }
    }
}

impl fmt::Display for WorkerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidToken { role } => write!(formatter, "invalid {role} token"),
            Self::InvalidDigest => {
                formatter.write_str("digest must be 64 lowercase hex characters")
            }
            Self::InvalidBudget => {
                formatter.write_str("retained candidate budget must be positive")
            }
            Self::InvalidConfig { detail } => {
                write!(formatter, "invalid worker configuration: {detail}")
            }
            Self::OtherConnection => {
                formatter.write_str("attempt identity belongs to another worker connection")
            }
            Self::OtherBinding { detail } => {
                write!(formatter, "attempt identity binding mismatch: {detail}")
            }
            Self::ConcurrentBuild => {
                formatter.write_str("another owned build is already in flight for this index")
            }
            Self::RebuildNeeded { cause } => match cause {
                RebuildCause::StaleAttempt => {
                    formatter.write_str("candidate belongs to a stale attempt; rebuild is required")
                }
                RebuildCause::StaleBinding => formatter.write_str(
                    "index UUID observed under another schema/table lifetime; rebuild is required",
                ),
                RebuildCause::LostCandidate => {
                    formatter.write_str("owned candidate was released or lost; rebuild is required")
                }
                RebuildCause::Invalidated => formatter.write_str(
                    "owned candidate was invalidated by schema change; rebuild is required",
                ),
            },
            Self::Cancelled => formatter.write_str("owned candidate build was cancelled"),
            Self::Closed => formatter.write_str("worker session is closed"),
            Self::TooManyIndexes { limit } => write!(
                formatter,
                "worker session reached its {limit} index slot limit"
            ),
            Self::CapacityExhausted { what } => {
                write!(formatter, "worker session exhausted {what}")
            }
            Self::BudgetExceeded { requested, limit } => write!(
                formatter,
                "owned candidate requires {requested} retained bytes; session budget is {limit}"
            ),
            Self::Allocation { what } => {
                write!(formatter, "owned candidate allocation failed: {what}")
            }
            Self::InvalidTopology { message } => {
                write!(formatter, "invalid native-owned topology: {message}")
            }
            Self::DigestMismatch => {
                formatter.write_str("native topology digests do not match the attempt identity")
            }
            Self::MetadataMismatch { detail } => {
                write!(formatter, "snapshot metadata mismatch: {detail}")
            }
            Self::ChunkRange {
                start,
                count,
                nodes,
            } => write!(
                formatter,
                "candidate chunk range start={start} count={count} is outside node_count={nodes}"
            ),
            Self::TicketStale => {
                formatter.write_str("build ticket is no longer current for this index")
            }
            Self::Poisoned => formatter.write_str("worker session lock was poisoned"),
        }
    }
}

impl std::error::Error for WorkerError {}

// ---------------------------------------------------------------------------
// Candidate data
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Candidate {
    identity: AttemptIdentity,
    token: CandidateToken,
    topology: Arc<OwnedTopology>,
    meta: SnapshotMeta,
    retained_bytes: u64,
}

#[derive(Debug)]
struct BuildSlot {
    identity: AttemptIdentity,
    ticket: u64,
    cancellation: Cancellation,
}

#[derive(Debug)]
struct ReadySlot {
    identity: AttemptIdentity,
    token: CandidateToken,
    candidate: Arc<Candidate>,
}

#[derive(Debug)]
enum Slot {
    Building(BuildSlot),
    Ready(ReadySlot),
}

#[derive(Debug)]
struct SessionState {
    closed: bool,
    retained_bytes: u64,
    next_ticket: u64,
    next_token: u64,
    slots: HashMap<IndexBinding, Slot>,
}

/// The one in-flight build ticket. Not cloneable: exactly one completion or
/// abandonment is accepted per build.
#[derive(Debug)]
pub struct BuildTicket {
    identity: AttemptIdentity,
    ticket: u64,
    cancellation: Cancellation,
}

impl BuildTicket {
    pub fn identity(&self) -> &AttemptIdentity {
        &self.identity
    }

    /// Clone the native cooperative cancellation signal for diskann_build::build.
    pub fn cancellation(&self) -> Cancellation {
        self.cancellation.clone()
    }
}

/// Opaque, non-authoritative handle for one ready candidate. The SQL owner may
/// persist [CandidateHandle::token] text; authority is revalidated on every call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateHandle {
    identity: AttemptIdentity,
    token: CandidateToken,
}

impl CandidateHandle {
    pub fn identity(&self) -> &AttemptIdentity {
        &self.identity
    }

    pub fn token(&self) -> &CandidateToken {
        &self.token
    }
}

#[derive(Debug)]
pub enum BeginOutcome {
    Proceed(BuildTicket),
    Replay(CandidateHandle),
}

/// Owned bounded manifest; safe to serialize or inspect outside any lock.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateManifest {
    pub population: u32,
    pub node_count: u32,
    pub root: Option<u32>,
    pub medoid_ordinal: Option<u32>,
    pub dimension: u32,
    pub metric: BuildMetric,
    pub max_degree: u32,
    pub construction_l: u32,
    pub edge_count: u64,
    pub input_digest: Digest32,
    pub config_digest: Digest32,
    pub topology_digest: Digest32,
    pub retained_bytes: u64,
}

/// One owned node copy. The immutable candidate remains in module ownership;
/// this chunk carries no borrow and the SQL owner may run a transaction with a
/// fresh SQLite cursor after the call returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkNode {
    pub ordinal: u32,
    pub entry: Option<SnapshotNode>,
    pub offset_start: u64,
    pub offset_end: u64,
    pub neighbors: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateChunk {
    pub start: u32,
    pub nodes: Vec<ChunkNode>,
    pub next: Option<u32>,
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// Owns all candidate/build state for exactly one SQLite worker connection.
///
/// The SQL owner keeps this under shared registration ownership (for example an
/// [Arc]); standard synchronization makes sharing safe, with no unsafe Send/Sync
/// implementation. Closing the connection must call [Self::close] or drop the
/// last owner; both signal in-flight cancellation and release every candidate.
#[derive(Debug)]
pub struct CandidateSessions {
    connection: ConnectionNonce,
    retained_budget_bytes: u64,
    state: Mutex<SessionState>,
}

impl CandidateSessions {
    pub fn new(connection: ConnectionNonce, retained_budget_bytes: u64) -> WorkerResult<Self> {
        if retained_budget_bytes == 0 {
            return Err(WorkerError::InvalidBudget);
        }
        Ok(Self {
            connection,
            retained_budget_bytes,
            state: Mutex::new(SessionState {
                closed: false,
                retained_bytes: 0,
                next_ticket: 1,
                next_token: 1,
                slots: HashMap::new(),
            }),
        })
    }

    pub fn connection(&self) -> &ConnectionNonce {
        &self.connection
    }

    pub fn retained_budget_bytes(&self) -> u64 {
        self.retained_budget_bytes
    }

    pub fn retained_bytes(&self) -> WorkerResult<u64> {
        let state = self.lock()?;
        if state.closed {
            return Ok(0);
        }
        Ok(state.retained_bytes)
    }

    pub fn is_closed(&self) -> WorkerResult<bool> {
        Ok(self.lock()?.closed)
    }

    /// Begins one fenced build or replays a completed identical attempt.
    ///
    /// No native work is started here and no SQL is touched. A caller that
    /// receives [BeginOutcome::Proceed] must later call [Self::complete] or
    /// [Self::abandon]; cancellation may be requested from any thread through
    /// [Self::cancel]. A completed identical identity returns the same handle
    /// without recomputation or additional retained bytes.
    pub fn begin(&self, identity: AttemptIdentity) -> WorkerResult<BeginOutcome> {
        if identity.connection != self.connection {
            return Err(WorkerError::OtherConnection);
        }
        let mut state = self.lock()?;
        if state.closed {
            return Err(WorkerError::Closed);
        }
        // A copied/attached/recreated database can present the same stored index
        // UUID under another schema or ephemeral lifetime. That is never a new
        // parallel candidate: the SQL owner must invalidate and rebuild.
        for binding in state.slots.keys() {
            if binding.index == identity.binding.index && binding != &identity.binding {
                return Err(WorkerError::RebuildNeeded {
                    cause: RebuildCause::StaleBinding,
                });
            }
        }
        match state.slots.get(&identity.binding) {
            Some(Slot::Building(_)) => Err(WorkerError::ConcurrentBuild),
            Some(Slot::Ready(ready)) => {
                if ready.identity == identity {
                    Ok(BeginOutcome::Replay(CandidateHandle {
                        identity: ready.identity.clone(),
                        token: ready.token.clone(),
                    }))
                } else {
                    Err(WorkerError::RebuildNeeded {
                        cause: RebuildCause::StaleAttempt,
                    })
                }
            }
            None => {
                if state.slots.len() >= MAX_INDEX_SLOTS {
                    return Err(WorkerError::TooManyIndexes {
                        limit: MAX_INDEX_SLOTS,
                    });
                }
                let ticket = state.next_ticket;
                state.next_ticket =
                    ticket
                        .checked_add(1)
                        .ok_or(WorkerError::CapacityExhausted {
                            what: "build ticket identifiers",
                        })?;
                let cancellation = Cancellation::default();
                state.slots.insert(
                    identity.binding.clone(),
                    Slot::Building(BuildSlot {
                        identity: identity.clone(),
                        ticket,
                        cancellation: cancellation.clone(),
                    }),
                );
                Ok(BeginOutcome::Proceed(BuildTicket {
                    identity,
                    ticket,
                    cancellation,
                }))
            }
        }
    }

    /// Validates and installs one fully owned native topology for a current ticket.
    ///
    /// Topology validation and digest/metadata checks are performed outside the
    /// module lock. Installation rechecks the ticket and cancellation state under
    /// the lock, so a concurrent cancel/close never produces a candidate. No SQL
    /// is executed and no candidate memory is exposed by reference.
    pub fn complete(
        &self,
        ticket: BuildTicket,
        topology: OwnedTopology,
        meta: SnapshotMeta,
    ) -> WorkerResult<CandidateHandle> {
        let identity = ticket.identity.clone();
        if identity.connection != self.connection {
            return Err(WorkerError::OtherConnection);
        }
        if ticket.cancellation.is_cancelled() {
            let _ = self.abandon_internal(&identity, ticket.ticket);
            return Err(WorkerError::Cancelled);
        }
        // This may be O(N+E) for the native validator; it must not hold the lock.
        if let Err(error) = validate_owned_topology(&identity, &topology, &meta) {
            let _ = self.abandon_internal(&identity, ticket.ticket);
            return Err(error);
        }
        let mut state = self.lock()?;
        if state.closed {
            return Err(WorkerError::Closed);
        }
        let current = match state.slots.get(&identity.binding) {
            Some(Slot::Building(slot)) => slot,
            Some(Slot::Ready(_)) => {
                return Err(WorkerError::TicketStale);
            }
            None => return Err(WorkerError::TicketStale),
        };
        if current.ticket != ticket.ticket || current.identity != identity {
            return Err(WorkerError::TicketStale);
        }
        if current.cancellation.is_cancelled() || ticket.cancellation.is_cancelled() {
            state.slots.remove(&identity.binding);
            return Err(WorkerError::Cancelled);
        }
        let token = match CandidateToken::generate(&self.connection, state.next_token) {
            Ok(token) => token,
            Err(error) => {
                state.slots.remove(&identity.binding);
                return Err(error);
            }
        };
        let next_token = match state.next_token.checked_add(1) {
            Some(next) => next,
            None => {
                state.slots.remove(&identity.binding);
                return Err(WorkerError::CapacityExhausted {
                    what: "candidate token identifiers",
                });
            }
        };
        state.next_token = next_token;
        let retained = match candidate_retained_bytes(
            &topology,
            &meta,
            &identity,
            &token,
            self.retained_budget_bytes,
        ) {
            Ok(retained) => retained,
            Err(error) => {
                state.slots.remove(&identity.binding);
                return Err(error);
            }
        };
        let total =
            state
                .retained_bytes
                .checked_add(retained)
                .ok_or(WorkerError::BudgetExceeded {
                    requested: u64::MAX,
                    limit: self.retained_budget_bytes,
                })?;
        if total > self.retained_budget_bytes {
            state.slots.remove(&identity.binding);
            return Err(WorkerError::BudgetExceeded {
                requested: total,
                limit: self.retained_budget_bytes,
            });
        }
        let candidate = Arc::new(Candidate {
            identity: identity.clone(),
            token: token.clone(),
            topology: Arc::new(topology),
            meta,
            retained_bytes: retained,
        });
        state.slots.insert(
            identity.binding.clone(),
            Slot::Ready(ReadySlot {
                identity: identity.clone(),
                token: token.clone(),
                candidate,
            }),
        );
        state.retained_bytes = total;
        Ok(CandidateHandle { identity, token })
    }

    /// Releases a current in-flight ticket after a native build error.
    ///
    /// Cleanup is intentionally forgiving: an already-cancelled, abandoned or
    /// completed ticket reports [AbandonReport::NotCurrent] instead of retaining
    /// a zombie slot. The SQL owner records its own durable failure separately.
    pub fn abandon(&self, ticket: BuildTicket) -> WorkerResult<AbandonReport> {
        if ticket.identity.connection != self.connection {
            return Err(WorkerError::OtherConnection);
        }
        self.abandon_internal(&ticket.identity, ticket.ticket)
    }

    fn abandon_internal(
        &self,
        identity: &AttemptIdentity,
        ticket: u64,
    ) -> WorkerResult<AbandonReport> {
        let mut state = self.lock()?;
        if state.closed {
            return Err(WorkerError::Closed);
        }
        match state.slots.get(&identity.binding) {
            Some(Slot::Building(slot)) if slot.ticket == ticket && slot.identity == *identity => {
                slot.cancellation.cancel();
                state.slots.remove(&identity.binding);
                Ok(AbandonReport::ReleasedBuild)
            }
            _ => Ok(AbandonReport::NotCurrent),
        }
    }

    /// Cancels the exact current attempt for one index binding.
    ///
    /// An in-flight build is signalled cooperatively and removed from the session;
    /// a ready candidate is released, releasing its retained RAM estimate. A
    /// different current attempt is rejected as stale rather than silently cancelled.
    pub fn cancel(&self, identity: &AttemptIdentity) -> WorkerResult<CancelReport> {
        if identity.connection != self.connection {
            return Err(WorkerError::OtherConnection);
        }
        let mut state = self.lock()?;
        if state.closed {
            return Err(WorkerError::Closed);
        }
        let Some(slot) = state.slots.get(&identity.binding) else {
            return Ok(CancelReport {
                had_in_flight: false,
                released_candidate: false,
            });
        };
        if slot_identity(slot) != identity {
            return Err(WorkerError::RebuildNeeded {
                cause: RebuildCause::StaleAttempt,
            });
        }
        match state.slots.remove(&identity.binding) {
            Some(Slot::Building(slot)) => {
                slot.cancellation.cancel();
                Ok(CancelReport {
                    had_in_flight: true,
                    released_candidate: false,
                })
            }
            Some(Slot::Ready(ready)) => {
                state.retained_bytes = state
                    .retained_bytes
                    .saturating_sub(ready.candidate.retained_bytes);
                Ok(CancelReport {
                    had_in_flight: false,
                    released_candidate: true,
                })
            }
            None => Ok(CancelReport {
                had_in_flight: false,
                released_candidate: false,
            }),
        }
    }

    /// Explicitly releases one ready candidate after successful import/publication
    /// or an abandoned campaign. The immutable [Arc] may still be used by an
    /// in-function copy already in progress; module ownership ends here. SQLite
    /// rollback cannot restore RAM released here or destroyed in the native
    /// worker, so a lost candidate always takes the fresh fenced rebuild path.
    pub fn release(&self, handle: &CandidateHandle) -> WorkerResult<()> {
        if handle.identity.connection != self.connection {
            return Err(WorkerError::OtherConnection);
        }
        let mut state = self.lock()?;
        if state.closed {
            return Ok(());
        }
        let Some(Slot::Ready(ready)) = state.slots.get(&handle.identity.binding) else {
            return Err(WorkerError::RebuildNeeded {
                cause: RebuildCause::LostCandidate,
            });
        };
        if ready.identity != handle.identity || ready.token != handle.token {
            return Err(WorkerError::RebuildNeeded {
                cause: RebuildCause::LostCandidate,
            });
        }
        let retained = ready.candidate.retained_bytes;
        state.slots.remove(&handle.identity.binding);
        state.retained_bytes = state.retained_bytes.saturating_sub(retained);
        Ok(())
    }

    /// Reconstructs a handle from a persisted opaque token and exact identity.
    /// This is the SQL text-token entry point; unknown/stale tokens require rebuild.
    pub fn resume(
        &self,
        identity: AttemptIdentity,
        token: CandidateToken,
    ) -> WorkerResult<CandidateHandle> {
        if identity.connection != self.connection {
            return Err(WorkerError::OtherConnection);
        }
        let state = self.lock()?;
        if state.closed {
            return Err(WorkerError::Closed);
        }
        let Some(Slot::Ready(ready)) = state.slots.get(&identity.binding) else {
            return Err(WorkerError::RebuildNeeded {
                cause: RebuildCause::LostCandidate,
            });
        };
        if ready.identity != identity || ready.token != token {
            return Err(WorkerError::RebuildNeeded {
                cause: RebuildCause::LostCandidate,
            });
        }
        Ok(CandidateHandle {
            identity: ready.identity.clone(),
            token: ready.token.clone(),
        })
    }

    /// Owned bounded manifest for a ready candidate.
    pub fn manifest(&self, handle: &CandidateHandle) -> WorkerResult<CandidateManifest> {
        let candidate = self.ready_candidate(handle)?;
        let topology = &candidate.topology;
        let config = topology.config();
        let dimension =
            u32::try_from(config.dimension).map_err(|_| WorkerError::InvalidTopology {
                message: "native dimension does not fit u32".to_owned(),
            })?;
        let max_degree =
            u32::try_from(config.max_degree).map_err(|_| WorkerError::InvalidTopology {
                message: "native max_degree does not fit u32".to_owned(),
            })?;
        let construction_l =
            u32::try_from(config.construction_l).map_err(|_| WorkerError::InvalidTopology {
                message: "native construction_l does not fit u32".to_owned(),
            })?;
        let node_count =
            u32::try_from(topology.node_count()).map_err(|_| WorkerError::InvalidTopology {
                message: "native node_count does not fit u32".to_owned(),
            })?;
        Ok(CandidateManifest {
            population: topology.population(),
            node_count,
            root: topology.root(),
            medoid_ordinal: topology.medoid_ordinal(),
            dimension,
            metric: config.metric,
            max_degree,
            construction_l,
            edge_count: topology.adjacency().len() as u64,
            input_digest: Digest32::from_bytes(topology.snapshot_digest()),
            config_digest: Digest32::from_bytes(topology.config_digest()),
            topology_digest: Digest32::from_bytes(topology.topology_digest()),
            retained_bytes: candidate.retained_bytes,
        })
    }

    /// Copies one bounded immutable node range. This is the SQL owner's chunk
    /// source: the call returns fully owned data and the caller runs its SQLite
    /// transaction with a fresh cursor after the call, never inside this module.
    pub fn copy_chunk(
        &self,
        handle: &CandidateHandle,
        start: u32,
        max_nodes: u32,
    ) -> WorkerResult<CandidateChunk> {
        let candidate = self.ready_candidate(handle)?;
        let topology = &candidate.topology;
        let node_count =
            u32::try_from(topology.node_count()).map_err(|_| WorkerError::InvalidTopology {
                message: "native node_count does not fit u32".to_owned(),
            })?;
        if max_nodes == 0 || max_nodes > MAX_CHUNK_NODES || start > node_count {
            return Err(WorkerError::ChunkRange {
                start,
                count: max_nodes,
                nodes: node_count,
            });
        }
        let remaining = node_count - start;
        let count = remaining.min(max_nodes);
        let mut nodes: Vec<ChunkNode> = Vec::new();
        nodes
            .try_reserve_exact(usize::try_from(count).map_err(|_| WorkerError::Allocation {
                what: "candidate chunk nodes",
            })?)
            .map_err(|_| WorkerError::Allocation {
                what: "candidate chunk nodes",
            })?;
        let offsets = topology.offsets();
        for ordinal in start..start.saturating_add(count) {
            let index = usize::try_from(ordinal).map_err(|_| WorkerError::Allocation {
                what: "candidate chunk node ordinal",
            })?;
            let next = index.checked_add(1).ok_or(WorkerError::InvalidTopology {
                message: "native offset index overflow".to_owned(),
            })?;
            let offset_start = *offsets.get(index).ok_or(WorkerError::InvalidTopology {
                message: "native CSR offset is missing".to_owned(),
            })?;
            let offset_end = *offsets.get(next).ok_or(WorkerError::InvalidTopology {
                message: "native CSR offset is missing".to_owned(),
            })?;
            let neighbors = topology
                .neighbors(ordinal)
                .ok_or(WorkerError::InvalidTopology {
                    message: "native adjacency node is missing".to_owned(),
                })?;
            let mut owned: Vec<u32> = Vec::new();
            owned
                .try_reserve_exact(neighbors.len())
                .map_err(|_| WorkerError::Allocation {
                    what: "candidate chunk adjacency",
                })?;
            owned.extend_from_slice(neighbors);
            nodes.push(ChunkNode {
                ordinal,
                entry: candidate.meta.node(ordinal),
                offset_start,
                offset_end,
                neighbors: owned,
            });
        }
        let next_ordinal = start.checked_add(count);
        Ok(CandidateChunk {
            start,
            nodes,
            next: match next_ordinal {
                Some(next) if next < node_count => Some(next),
                _ => None,
            },
        })
    }

    /// Owned copy of the frozen root vector; empty for explicit empty input.
    pub fn root_vector(&self, handle: &CandidateHandle) -> WorkerResult<Vec<f32>> {
        let candidate = self.ready_candidate(handle)?;
        let source = candidate.topology.root_vector();
        let mut output = Vec::new();
        output
            .try_reserve_exact(source.len())
            .map_err(|_| WorkerError::Allocation {
                what: "candidate root vector",
            })?;
        output.extend_from_slice(source);
        Ok(output)
    }

    fn ready_candidate(&self, handle: &CandidateHandle) -> WorkerResult<Arc<Candidate>> {
        if handle.identity.connection != self.connection {
            return Err(WorkerError::OtherConnection);
        }
        let state = self.lock()?;
        if state.closed {
            return Err(WorkerError::Closed);
        }
        let Some(Slot::Ready(ready)) = state.slots.get(&handle.identity.binding) else {
            return Err(WorkerError::RebuildNeeded {
                cause: RebuildCause::LostCandidate,
            });
        };
        if ready.identity != handle.identity || ready.token != handle.token {
            return Err(WorkerError::RebuildNeeded {
                cause: RebuildCause::LostCandidate,
            });
        }
        Ok(ready.candidate.clone())
    }

    /// Invalidates every candidate/build sharing one index UUID, across schema and
    /// table-lifetime bindings. This is the explicit DROP/recreate/RENAME and
    /// attached-copy hook: stale handles become rebuild-needed and in-flight work
    /// is cooperatively cancelled.
    pub fn invalidate_index(&self, index: &IndexUuid) -> WorkerResult<InvalidationReport> {
        let mut state = self.lock()?;
        if state.closed {
            return Err(WorkerError::Closed);
        }
        let mut cancelled_builds = 0usize;
        let mut released_candidates = 0usize;
        let mut retained_released = 0u64;
        state.slots.retain(|binding, slot| {
            if binding.index != *index {
                return true;
            }
            match slot {
                Slot::Building(build) => {
                    build.cancellation.cancel();
                    cancelled_builds += 1;
                }
                Slot::Ready(ready) => {
                    retained_released =
                        retained_released.saturating_add(ready.candidate.retained_bytes);
                    released_candidates += 1;
                }
            }
            false
        });
        state.retained_bytes = state.retained_bytes.saturating_sub(retained_released);
        Ok(InvalidationReport {
            cancelled_builds,
            released_candidates,
        })
    }

    /// Explicit connection-close release. Idempotent; in-flight builds are
    /// signalled cooperatively and every candidate is dropped immediately.
    pub fn close(&self) -> WorkerResult<()> {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if state.closed {
            return Ok(());
        }
        state.closed = true;
        state.retained_bytes = 0;
        for (_, slot) in state.slots.drain() {
            if let Slot::Building(build) = slot {
                build.cancellation.cancel();
            }
        }
        Ok(())
    }

    fn lock(&self) -> WorkerResult<std::sync::MutexGuard<'_, SessionState>> {
        self.state.lock().map_err(|_| WorkerError::Poisoned)
    }
}

impl Drop for CandidateSessions {
    fn drop(&mut self) {
        let state = match self.state.get_mut() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.closed = true;
        state.retained_bytes = 0;
        for (_, slot) in state.slots.drain() {
            if let Slot::Building(build) = slot {
                build.cancellation.cancel();
            }
        }
    }
}

fn slot_identity(slot: &Slot) -> &AttemptIdentity {
    match slot {
        Slot::Building(build) => &build.identity,
        Slot::Ready(ready) => &ready.identity,
    }
}

fn candidate_retained_bytes(
    topology: &OwnedTopology,
    meta: &SnapshotMeta,
    identity: &AttemptIdentity,
    token: &CandidateToken,
    limit: u64,
) -> WorkerResult<u64> {
    let admission_bytes =
        u64::try_from(topology.admission().retained_upper_bytes).map_err(|_| {
            WorkerError::BudgetExceeded {
                requested: u64::MAX,
                limit,
            }
        })?;
    let mut total = checked_retained(0, admission_bytes, limit)?;
    total = checked_retained(total, meta.retained_bytes(), limit)?;
    total = checked_retained(total, identity.label_bytes(), limit)?;
    total = checked_retained(total, token.label_bytes(), limit)?;
    checked_retained(total, CANDIDATE_OVERHEAD_BYTES, limit)
}

fn checked_retained(mut total: u64, extra: u64, limit: u64) -> WorkerResult<u64> {
    total = total
        .checked_add(extra)
        .ok_or(WorkerError::BudgetExceeded {
            requested: u64::MAX,
            limit,
        })?;
    if total > limit {
        return Err(WorkerError::BudgetExceeded {
            requested: total,
            limit,
        });
    }
    Ok(total)
}

fn validate_owned_topology(
    identity: &AttemptIdentity,
    topology: &OwnedTopology,
    meta: &SnapshotMeta,
) -> WorkerResult<()> {
    topology
        .validate()
        .map_err(|error| WorkerError::InvalidTopology {
            message: error.to_string(),
        })?;
    if topology.snapshot_digest() != *identity.input_digest.as_bytes() {
        return Err(WorkerError::DigestMismatch);
    }
    if topology.config_digest() != *identity.config_digest.as_bytes() {
        return Err(WorkerError::DigestMismatch);
    }
    let population =
        usize::try_from(topology.population()).map_err(|_| WorkerError::InvalidTopology {
            message: "native population does not fit usize".to_owned(),
        })?;
    if meta.len() != population {
        return Err(WorkerError::MetadataMismatch {
            detail: "snapshot metadata length differs from native population",
        });
    }
    let nodes = topology.node_count();
    let config = topology.config();
    if population == 0 {
        if nodes != 0
            || topology.root().is_some()
            || topology.medoid_ordinal().is_some()
            || !topology.root_vector().is_empty()
            || !topology.adjacency().is_empty()
        {
            return Err(WorkerError::InvalidTopology {
                message: "empty native topology carries graph/root state".to_owned(),
            });
        }
    } else {
        if nodes != population.saturating_add(1) {
            return Err(WorkerError::InvalidTopology {
                message: "native node_count is not population plus frozen root".to_owned(),
            });
        }
        if topology.root() != Some(topology.population()) {
            return Err(WorkerError::InvalidTopology {
                message: "native frozen root is not the population ID".to_owned(),
            });
        }
        if topology.root_vector().len() != config.dimension {
            return Err(WorkerError::InvalidTopology {
                message: "native root vector dimension mismatch".to_owned(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "diskann_worker/tests.rs"]
mod tests;
