//! Pure module qualification tests for the T4a candidate session seam.
//!
//! Candidates are produced only by the qualified registry-native small toy
//! builder; there is no numeric baseline, recall or performance claim here.

use super::*;
use crate::diskann_build::{
    build, BuildConfig, BuildLimits, BuildMetric, Cancellation, CheckedSnapshot,
};

fn connection() -> ConnectionNonce {
    ConnectionNonce::parse("conn:worker-connection-1").expect("connection token")
}

fn schema(name: &str) -> SchemaNonce {
    SchemaNonce::parse(&format!("sch:{name}")).expect("schema token")
}

fn lifetime(name: &str) -> TableLifetimeNonce {
    TableLifetimeNonce::parse(&format!("life:{name}")).expect("lifetime token")
}

fn index(name: &str) -> IndexUuid {
    IndexUuid::parse(&format!("idx:{name}")).expect("index token")
}

fn job() -> JobId {
    JobId::parse("job:job-1").expect("job token")
}

fn fence() -> FenceToken {
    FenceToken::parse("fence:7").expect("fence token")
}

fn attempt(name: &str) -> AttemptToken {
    AttemptToken::parse(&format!("attempt:{name}")).expect("attempt token")
}

fn binding() -> IndexBinding {
    IndexBinding::new(schema("main"), lifetime("lifetime-1"), index("index-1"))
}

fn toy_config(dimension: usize) -> BuildConfig {
    BuildConfig {
        dimension,
        metric: BuildMetric::L2Squared,
        max_degree: 32,
        construction_l: 100,
        alpha: 1.2,
        seed: 42,
        workers: 1,
    }
}

fn toy_limits() -> BuildLimits {
    BuildLimits {
        memory_budget_bytes: 512 * 1024 * 1024,
        external_reservation_bytes: 8 * 1024 * 1024,
    }
}

fn toy_topology(population: usize, dimension: usize) -> OwnedTopology {
    let mut values = Vec::with_capacity(population * dimension);
    for index in 0..population * dimension {
        values.push((((index * 13 + 7) % 29) as f32) * 0.1 - 1.4);
    }
    let snapshot = CheckedSnapshot::new(population, dimension, BuildMetric::L2Squared, values)
        .expect("checked toy snapshot");
    build(
        snapshot,
        toy_config(dimension),
        toy_limits(),
        Cancellation::default(),
    )
    .expect("native toy build")
}

fn identity_with(
    binding: &IndexBinding,
    attempt_name: &str,
    input_digest: Digest32,
    config_digest: Digest32,
) -> AttemptIdentity {
    AttemptIdentity::new(
        connection(),
        binding.clone(),
        job(),
        fence(),
        attempt(attempt_name),
        input_digest,
        config_digest,
    )
}

fn identity_for(
    binding: &IndexBinding,
    attempt_name: &str,
    topology: &OwnedTopology,
) -> AttemptIdentity {
    identity_with(
        binding,
        attempt_name,
        Digest32::from_bytes(topology.snapshot_digest()),
        Digest32::from_bytes(topology.config_digest()),
    )
}

fn meta(population: u32) -> SnapshotMeta {
    SnapshotMeta::new(
        population,
        (0..population)
            .map(|ordinal| SnapshotNode::new(u64::from(ordinal) + 10, 1))
            .collect(),
    )
    .expect("snapshot metadata")
}

fn sessions() -> CandidateSessions {
    CandidateSessions::new(connection(), 512 * 1024 * 1024).expect("worker session")
}

fn proceed(sessions: &CandidateSessions, identity: AttemptIdentity) -> BuildTicket {
    match sessions.begin(identity).expect("begin") {
        BeginOutcome::Proceed(ticket) => ticket,
        BeginOutcome::Replay(_) => panic!("expected a new build, got replay"),
    }
}

fn ready(sessions: &CandidateSessions, attempt_name: &str) -> (CandidateHandle, AttemptIdentity) {
    let binding = binding();
    let topology = toy_topology(3, 2);
    let identity = identity_for(&binding, attempt_name, &topology);
    let ticket = proceed(sessions, identity.clone());
    let handle = sessions
        .complete(ticket, topology, meta(3))
        .expect("completed candidate");
    (handle, identity)
}

#[test]
fn token_parsing_is_strict() {
    assert!(ConnectionNonce::parse("conn:abc-123._").is_ok());
    assert!(SchemaNonce::parse("sch:main").is_ok());
    assert!(TableLifetimeNonce::parse("life:lifetime-1").is_ok());
    assert!(IndexUuid::parse("idx:index-1").is_ok());
    assert!(JobId::parse("job:job-1").is_ok());
    assert!(FenceToken::parse("fence:7").is_ok());
    assert!(AttemptToken::parse("attempt:1").is_ok());
    assert!(CandidateToken::parse("cand:abc").is_ok());

    assert_eq!(
        ConnectionNonce::parse("worker:abc").unwrap_err().kind(),
        WorkerErrorKind::InvalidToken
    );
    assert_eq!(
        ConnectionNonce::parse("conn:").unwrap_err().kind(),
        WorkerErrorKind::InvalidToken
    );
    assert_eq!(
        JobId::parse("job:has space").unwrap_err().kind(),
        WorkerErrorKind::InvalidToken
    );
    assert_eq!(
        JobId::parse("job:has:colon").unwrap_err().kind(),
        WorkerErrorKind::InvalidToken
    );
    let too_long = format!("job:{}", "a".repeat(MAX_TOKEN_BODY_BYTES + 1));
    assert_eq!(
        JobId::parse(&too_long).unwrap_err().kind(),
        WorkerErrorKind::InvalidToken
    );
    assert_eq!(
        JobId::parse("fence:7").unwrap_err().kind(),
        WorkerErrorKind::InvalidToken
    );
    let parsed = ConnectionNonce::parse("conn:abc").expect("token");
    assert_eq!(parsed.as_str(), "conn:abc");
    assert_eq!(parsed.to_string(), "conn:abc");
}

#[test]
fn digest_parsing_is_strict_and_round_trips() {
    let digest = Digest32::from_bytes([0xab_u8; 32]);
    let text = digest.to_string();
    assert_eq!(text.len(), 64);
    assert_eq!(Digest32::parse(&text).expect("digest"), digest);
    assert_eq!(
        Digest32::parse(&text.to_uppercase()).unwrap_err().kind(),
        WorkerErrorKind::InvalidDigest
    );
    assert_eq!(
        Digest32::parse("00").unwrap_err().kind(),
        WorkerErrorKind::InvalidDigest
    );
    let mut non_hex = text.clone();
    non_hex.replace_range(0..1, "g");
    assert_eq!(
        Digest32::parse(&non_hex).unwrap_err().kind(),
        WorkerErrorKind::InvalidDigest
    );
}

#[test]
fn invalid_budget_rejected() {
    let error = CandidateSessions::new(connection(), 0).expect_err("zero budget");
    assert_eq!(error.kind(), WorkerErrorKind::InvalidBudget);
}

#[test]
fn begin_complete_replay_and_resume() {
    let sessions = sessions();
    let (handle, identity) = ready(&sessions, "attempt-1");
    assert!(sessions.retained_bytes().expect("retained") > 0);

    match sessions.begin(identity.clone()).expect("replay") {
        BeginOutcome::Replay(replayed) => {
            assert_eq!(replayed.token(), handle.token());
            assert_eq!(replayed.identity(), handle.identity());
        }
        BeginOutcome::Proceed(_) => panic!("completed identical attempt must replay"),
    }
    let resumed = sessions
        .resume(identity.clone(), handle.token().clone())
        .expect("resume");
    assert_eq!(resumed.identity(), handle.identity());
    assert_eq!(sessions.retained_bytes().expect("retained"), {
        let manifest = sessions.manifest(&handle).expect("manifest");
        manifest.retained_bytes
    });
    sessions.release(&handle).expect("release");
    assert_eq!(sessions.retained_bytes().expect("retained"), 0);
}

#[test]
fn duplicate_inflight_begin_is_rejected() {
    let sessions = sessions();
    let binding = binding();
    let topology = toy_topology(3, 2);
    let identity = identity_for(&binding, "attempt-1", &topology);
    let _ticket = proceed(&sessions, identity.clone());
    let error = sessions.begin(identity).expect_err("duplicate build");
    assert_eq!(error.kind(), WorkerErrorKind::ConcurrentBuild);
}

#[test]
fn other_connection_identity_is_rejected() {
    let sessions = sessions();
    let other = ConnectionNonce::parse("conn:another-connection").expect("token");
    let identity = AttemptIdentity::new(
        other,
        binding(),
        job(),
        fence(),
        attempt("attempt-1"),
        Digest32::ZERO,
        Digest32::ZERO,
    );
    let error = sessions.begin(identity).expect_err("other connection");
    assert_eq!(error.kind(), WorkerErrorKind::OtherConnection);
}

#[test]
fn different_ready_attempt_requires_rebuild() {
    let sessions = sessions();
    let (_handle, identity) = ready(&sessions, "attempt-1");
    let rebind = identity_with(
        &binding(),
        "attempt-2",
        identity.input_digest(),
        identity.config_digest(),
    );
    let error = sessions.begin(rebind).expect_err("stale attempt");
    assert_eq!(error.kind(), WorkerErrorKind::RebuildNeeded);
    assert_eq!(error.rebuild_cause(), Some(RebuildCause::StaleAttempt));
}

#[test]
fn same_uuid_other_binding_and_invalidate() {
    let sessions = sessions();
    let (handle, _identity) = ready(&sessions, "attempt-1");
    let other_binding = IndexBinding::new(schema("main"), lifetime("lifetime-2"), index("index-1"));
    let other = identity_with(
        &other_binding,
        "attempt-2",
        Digest32::from_bytes([1; 32]),
        Digest32::from_bytes([2; 32]),
    );
    let error = sessions.begin(other).expect_err("stale binding");
    assert_eq!(error.kind(), WorkerErrorKind::RebuildNeeded);
    assert_eq!(error.rebuild_cause(), Some(RebuildCause::StaleBinding));

    let report = sessions
        .invalidate_index(handle.identity().binding().index())
        .expect("invalidate");
    assert_eq!(report.released_candidates, 1);
    assert_eq!(sessions.retained_bytes().expect("retained"), 0);
    let error = sessions
        .copy_chunk(&handle, 0, 1)
        .expect_err("released candidate");
    assert_eq!(error.kind(), WorkerErrorKind::RebuildNeeded);
    assert_eq!(error.rebuild_cause(), Some(RebuildCause::LostCandidate));

    let retry = identity_with(
        &other_binding,
        "attempt-2",
        Digest32::from_bytes([1; 32]),
        Digest32::from_bytes([2; 32]),
    );
    let ticket = proceed(&sessions, retry);
    assert!(sessions.abandon(ticket).is_ok());
}

#[test]
fn cancel_in_flight_signals_and_ticket_stales() {
    let sessions = sessions();
    let binding = binding();
    let topology = toy_topology(3, 2);
    let identity = identity_for(&binding, "attempt-1", &topology);
    let ticket = proceed(&sessions, identity.clone());
    assert!(!ticket.cancellation().is_cancelled());
    let report = sessions.cancel(&identity).expect("cancel");
    assert!(report.had_in_flight);
    assert!(!report.released_candidate);
    assert!(ticket.cancellation().is_cancelled());
    let error = sessions
        .complete(ticket, topology, meta(3))
        .expect_err("cancelled completion");
    assert_eq!(error.kind(), WorkerErrorKind::Cancelled);
    assert_eq!(sessions.retained_bytes().expect("retained"), 0);
}

#[test]
fn cancel_with_other_attempt_is_rejected() {
    let sessions = sessions();
    let binding = binding();
    let topology = toy_topology(3, 2);
    let identity = identity_for(&binding, "attempt-1", &topology);
    let ticket = proceed(&sessions, identity.clone());
    let other = identity_with(
        &binding,
        "attempt-2",
        identity.input_digest(),
        identity.config_digest(),
    );
    let error = sessions.cancel(&other).expect_err("stale cancel");
    assert_eq!(error.kind(), WorkerErrorKind::RebuildNeeded);
    sessions
        .complete(ticket, topology, meta(3))
        .expect("original ticket remains current");
}

#[test]
fn abandon_releases_build_and_allows_retry() {
    let sessions = sessions();
    let binding = binding();
    let topology = toy_topology(3, 2);
    let identity = identity_for(&binding, "attempt-1", &topology);
    let ticket = proceed(&sessions, identity.clone());
    assert_eq!(
        sessions.abandon(ticket).expect("abandon"),
        AbandonReport::ReleasedBuild
    );
    let second = proceed(&sessions, identity);
    assert_eq!(
        sessions.abandon(second).expect("abandon"),
        AbandonReport::ReleasedBuild
    );
}

#[test]
fn digest_mismatch_releases_slot_and_candidate() {
    let sessions = sessions();
    let binding = binding();
    let topology = toy_topology(3, 2);
    let identity = identity_with(&binding, "attempt-1", Digest32::ZERO, Digest32::ZERO);
    let ticket = proceed(&sessions, identity.clone());
    let error = sessions
        .complete(ticket, topology, meta(3))
        .expect_err("digest mismatch");
    assert_eq!(error.kind(), WorkerErrorKind::DigestMismatch);
    assert_eq!(sessions.retained_bytes().expect("retained"), 0);
    let retry = proceed(&sessions, identity);
    assert!(sessions.abandon(retry).is_ok());
}

#[test]
fn metadata_mismatch_rejected() {
    let sessions = sessions();
    let binding = binding();
    let topology = toy_topology(3, 2);
    let identity = identity_for(&binding, "attempt-1", &topology);
    let ticket = proceed(&sessions, identity);
    let error = sessions
        .complete(ticket, topology, meta(2))
        .expect_err("metadata mismatch");
    assert_eq!(error.kind(), WorkerErrorKind::MetadataMismatch);
    assert_eq!(
        SnapshotMeta::new(2, vec![SnapshotNode::new(1, 1)])
            .unwrap_err()
            .kind(),
        WorkerErrorKind::MetadataMismatch
    );
}

#[test]
fn budget_rejection_releases_slot() {
    let sessions = CandidateSessions::new(connection(), 1).expect("session");
    let binding = binding();
    let topology = toy_topology(3, 2);
    let identity = identity_for(&binding, "attempt-1", &topology);
    let ticket = proceed(&sessions, identity.clone());
    let error = sessions
        .complete(ticket, topology, meta(3))
        .expect_err("budget");
    assert_eq!(error.kind(), WorkerErrorKind::BudgetExceeded);
    assert_eq!(sessions.retained_bytes().expect("retained"), 0);
    let retry = proceed(&sessions, identity);
    assert!(sessions.abandon(retry).is_ok());
}

#[test]
fn release_updates_retained_and_lost_handles_need_rebuild() {
    let sessions = sessions();
    let (handle, _identity) = ready(&sessions, "attempt-1");
    let manifest = sessions.manifest(&handle).expect("manifest");
    assert!(manifest.retained_bytes > 0);
    assert_eq!(
        sessions.retained_bytes().expect("retained"),
        manifest.retained_bytes
    );
    sessions.release(&handle).expect("release");
    assert_eq!(sessions.retained_bytes().expect("retained"), 0);
    let error = sessions.manifest(&handle).expect_err("released candidate");
    assert_eq!(error.kind(), WorkerErrorKind::RebuildNeeded);
    assert_eq!(error.rebuild_cause(), Some(RebuildCause::LostCandidate));
    let error = sessions.release(&handle).expect_err("double release");
    assert_eq!(error.kind(), WorkerErrorKind::RebuildNeeded);
}

#[test]
fn chunk_and_root_vector_are_owned_and_bounded() {
    let sessions = sessions();
    let binding = binding();
    let topology = toy_topology(3, 2);
    let node_count = topology.node_count();
    assert_eq!(node_count, 4);
    let expected_neighbors: Vec<Vec<u32>> = (0..node_count as u32)
        .map(|ordinal| topology.neighbors(ordinal).expect("neighbors").to_vec())
        .collect();
    let expected_offsets: Vec<(u64, u64)> = (0..node_count as u32)
        .map(|ordinal| {
            (
                topology.offsets()[ordinal as usize],
                topology.offsets()[ordinal as usize + 1],
            )
        })
        .collect();
    let expected_root = topology.root_vector().to_vec();
    let identity = identity_for(&binding, "attempt-1", &topology);
    let ticket = proceed(&sessions, identity);
    let handle = sessions
        .complete(ticket, topology, meta(3))
        .expect("candidate");
    assert_eq!(sessions.root_vector(&handle).expect("root"), expected_root);

    let first = sessions.copy_chunk(&handle, 0, 2).expect("first chunk");
    assert_eq!(first.start, 0);
    assert_eq!(first.nodes.len(), 2);
    assert_eq!(first.next, Some(2));
    for (position, node) in first.nodes.iter().enumerate() {
        assert_eq!(node.ordinal as usize, position);
        assert_eq!(&node.neighbors, &expected_neighbors[position]);
        assert_eq!(
            (node.offset_start, node.offset_end),
            expected_offsets[position]
        );
        assert_eq!(
            node.entry,
            Some(SnapshotNode::new(u64::from(node.ordinal) + 10, 1))
        );
    }
    let second = sessions.copy_chunk(&handle, 2, 2).expect("second chunk");
    assert_eq!(second.nodes.len(), 2);
    assert_eq!(second.next, None);
    assert_eq!(second.nodes[1].ordinal, 3);
    assert_eq!(
        second.nodes[1].entry, None,
        "frozen root has no source entry"
    );

    assert_eq!(
        sessions.copy_chunk(&handle, 0, 0).unwrap_err().kind(),
        WorkerErrorKind::ChunkRange
    );
    assert_eq!(
        sessions
            .copy_chunk(&handle, 0, MAX_CHUNK_NODES + 1)
            .unwrap_err()
            .kind(),
        WorkerErrorKind::ChunkRange
    );
    let tail = sessions
        .copy_chunk(&handle, 4, 1)
        .expect("empty tail chunk");
    assert!(tail.nodes.is_empty());
    assert_eq!(tail.next, None);
    assert_eq!(
        sessions.copy_chunk(&handle, 5, 1).unwrap_err().kind(),
        WorkerErrorKind::ChunkRange
    );
    sessions.release(&handle).expect("release");
}

#[test]
fn empty_candidate_is_explicit_and_chunkable() {
    let sessions = sessions();
    let binding = binding();
    let topology = toy_topology(0, 2);
    let identity = identity_for(&binding, "attempt-empty", &topology);
    let ticket = proceed(&sessions, identity);
    let handle = sessions
        .complete(ticket, topology, meta(0))
        .expect("empty candidate");
    let manifest = sessions.manifest(&handle).expect("manifest");
    assert_eq!(manifest.population, 0);
    assert_eq!(manifest.node_count, 0);
    assert_eq!(manifest.root, None);
    assert_eq!(manifest.medoid_ordinal, None);
    assert_eq!(
        sessions.root_vector(&handle).expect("root"),
        Vec::<f32>::new()
    );
    let chunk = sessions.copy_chunk(&handle, 0, 1).expect("empty chunk");
    assert!(chunk.nodes.is_empty());
    assert_eq!(chunk.next, None);
    sessions.release(&handle).expect("release");
}

#[test]
fn manifest_reflects_native_topology_and_digests() {
    let sessions = sessions();
    let binding = binding();
    let topology = toy_topology(3, 2);
    let expected_digests = (
        Digest32::from_bytes(topology.snapshot_digest()),
        Digest32::from_bytes(topology.config_digest()),
    );
    let identity = identity_for(&binding, "attempt-1", &topology);
    let ticket = proceed(&sessions, identity.clone());
    let handle = sessions
        .complete(ticket, topology, meta(3))
        .expect("candidate");
    let manifest = sessions.manifest(&handle).expect("manifest");
    assert_eq!(manifest.population, 3);
    assert_eq!(manifest.node_count, 4);
    assert_eq!(manifest.root, Some(3));
    assert_eq!(manifest.dimension, 2);
    assert_eq!(manifest.metric, BuildMetric::L2Squared);
    assert_eq!(manifest.max_degree, 32);
    assert_eq!(manifest.construction_l, 100);
    assert_eq!(manifest.input_digest, expected_digests.0);
    assert_eq!(manifest.config_digest, expected_digests.1);
    assert_eq!(manifest.input_digest, identity.input_digest());
    sessions.release(&handle).expect("release");
}

#[test]
fn resume_with_wrong_token_requires_rebuild() {
    let sessions = sessions();
    let (handle, identity) = ready(&sessions, "attempt-1");
    let wrong = CandidateToken::parse("cand:0000000000000000-0000000000000000").expect("token");
    let error = sessions.resume(identity, wrong).expect_err("wrong token");
    assert_eq!(error.kind(), WorkerErrorKind::RebuildNeeded);
    assert_eq!(error.rebuild_cause(), Some(RebuildCause::LostCandidate));
    assert!(sessions
        .resume(handle.identity().clone(), handle.token().clone())
        .is_ok());
}

#[test]
fn too_many_index_slots_rejected() {
    let sessions = sessions();
    for slot in 0..MAX_INDEX_SLOTS {
        let binding = IndexBinding::new(
            schema("main"),
            lifetime("lifetime-1"),
            index(&format!("slot-{slot}")),
        );
        let identity = identity_with(&binding, "attempt-1", Digest32::ZERO, Digest32::ZERO);
        assert!(matches!(
            sessions.begin(identity).expect("begin"),
            BeginOutcome::Proceed(_)
        ));
    }
    let binding = IndexBinding::new(
        schema("main"),
        lifetime("lifetime-1"),
        index("slot-overflow"),
    );
    let identity = identity_with(&binding, "attempt-1", Digest32::ZERO, Digest32::ZERO);
    let error = sessions.begin(identity).expect_err("slot limit");
    assert_eq!(
        error,
        WorkerError::TooManyIndexes {
            limit: MAX_INDEX_SLOTS
        }
    );
}

#[test]
fn close_signals_build_and_is_idempotent() {
    let sessions = sessions();
    let binding = binding();
    let identity = identity_with(&binding, "attempt-1", Digest32::ZERO, Digest32::ZERO);
    let ticket = proceed(&sessions, identity.clone());
    sessions.close().expect("close");
    assert!(ticket.cancellation().is_cancelled());
    sessions.close().expect("second close");
    assert!(sessions.is_closed().expect("closed"));
    assert_eq!(sessions.retained_bytes().expect("retained"), 0);
    let error = sessions.begin(identity).expect_err("closed session");
    assert_eq!(error.kind(), WorkerErrorKind::Closed);
}

#[test]
fn shared_arc_session_across_threads() {
    let sessions = Arc::new(sessions());
    let binding = binding();
    let topology = toy_topology(3, 2);
    let identity = identity_for(&binding, "attempt-shared", &topology);
    let threads = {
        let sessions = Arc::clone(&sessions);
        let identity = identity.clone();
        std::thread::spawn(move || {
            let ticket = proceed(&sessions, identity);
            sessions
                .complete(ticket, topology, meta(3))
                .expect("thread candidate")
        })
    };
    let handle = threads.join().expect("thread");
    let resumed = sessions
        .resume(identity, handle.token().clone())
        .expect("resume");
    let manifest = sessions.manifest(&resumed).expect("manifest");
    assert_eq!(manifest.population, 3);
    sessions.release(&resumed).expect("release");
    assert_eq!(sessions.retained_bytes().expect("retained"), 0);
}

#[test]
fn two_indexes_account_retained_independently() {
    let sessions = sessions();
    let binding_a = binding();
    let binding_b = IndexBinding::new(schema("main"), lifetime("lifetime-1"), index("index-2"));
    let topology_a = toy_topology(3, 2);
    let topology_b = toy_topology(2, 2);
    let identity_a = identity_for(&binding_a, "attempt-a", &topology_a);
    let identity_b = identity_for(&binding_b, "attempt-b", &topology_b);
    let ticket_a = proceed(&sessions, identity_a);
    let ticket_b = proceed(&sessions, identity_b);
    let handle_a = sessions
        .complete(ticket_a, topology_a, meta(3))
        .expect("candidate a");
    let handle_b = sessions
        .complete(ticket_b, topology_b, meta(2))
        .expect("candidate b");
    let retained_a = sessions
        .manifest(&handle_a)
        .expect("manifest a")
        .retained_bytes;
    let retained_b = sessions
        .manifest(&handle_b)
        .expect("manifest b")
        .retained_bytes;
    assert_eq!(
        sessions.retained_bytes().expect("retained"),
        retained_a + retained_b
    );
    sessions.release(&handle_a).expect("release a");
    assert_eq!(sessions.retained_bytes().expect("retained"), retained_b);
    sessions.release(&handle_b).expect("release b");
    assert_eq!(sessions.retained_bytes().expect("retained"), 0);
}
