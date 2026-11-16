//! Durable SQL-only frozen-input jobs. Caller owns the atomic_write carrier.
use super::*;
use crate::diskann_build::{BuildConfig, BuildMetric};
use crate::diskann_build_guard::SnapshotReaders;
use crate::diskann_worker::SnapshotNode;
use sha2::{Digest, Sha256};

pub(crate) const CAPTURING: u64 = 0;
pub(crate) const SEALED: u64 = 1;
pub(crate) const BUILDING: u64 = 2;
pub(crate) const VALIDATING: u64 = 3;
pub(crate) const VALIDATED: u64 = 4;
pub(crate) const COMPLETED: u64 = 5;
pub(crate) const ABANDONED: u64 = 6;

/// Validated bounded work, including examined history (not merely emitted rows).
#[derive(Clone, Copy, Debug)]
pub(crate) struct StepOptions {
    pub(crate) max_records: u32,
    pub(crate) max_bytes: usize,
}
impl StepOptions {
    pub(crate) fn new(max_records: u32, max_bytes: usize) -> Result<Self> {
        if max_records == 0 || max_records > 256 || max_bytes == 0 {
            return Err(IndexError::new(
                "DiskANN step requires 1..256 records and positive byte budget",
            ));
        }
        Ok(Self {
            max_records,
            max_bytes,
        })
    }
}
/// Root-created authority, never deserialized wholesale from SQL caller JSON.
#[derive(Clone, Debug)]
pub(crate) struct BuildBinding {
    pub(crate) instance_id: [u8; 16],
    pub(crate) job_id: u64,
    pub(crate) generation_id: u64,
    pub(crate) fence: u64,
    pub(crate) owner_nonce: [u8; 16],
    pub(crate) source_hash: [u8; 32],
    pub(crate) config_hash: [u8; 32],
}
pub(crate) struct MaterializedSnapshot {
    pub(crate) vectors: Vec<f32>,
    pub(crate) nodes: Vec<SnapshotNode>,
    pub(crate) source_hash: [u8; 32],
    pub(crate) config: BuildConfig,
}
#[derive(Debug)]
pub(crate) struct Status {
    pub(crate) metadata: Metadata,
    pub(crate) job: Option<Job>,
}

pub(crate) fn stale(message: &str) -> IndexError {
    IndexError::with_code(
        ffi::SQLITE_BUSY as i32,
        format!("DiskANN stale/rebuild-needed: {message}"),
    )
}
pub(crate) fn changed(table: &DiskAnnTable, name: &str) -> Result<()> {
    if table.connection().changes() != 1 {
        return Err(stale(name));
    }
    Ok(())
}
pub(crate) fn metadata(table: &DiskAnnTable) -> Result<Metadata> {
    let (m, anchor) = table.read_metadata()?;
    anchor.finish()?;
    Ok(m)
}
pub(crate) fn status(table: &DiskAnnTable) -> Result<Status> {
    // Keep the metadata snapshot until its job has been read, explicitly finalize.
    let (m, anchor) = table.read_metadata()?;
    let result = m.active_job.map(|id| table.read_job(id)).transpose();
    let finish = anchor.finish();
    let j = result?;
    finish?;
    Ok(Status {
        metadata: m,
        job: j,
    })
}
pub(crate) fn job(table: &DiskAnnTable, id: u64) -> Result<Job> {
    let (_, anchor) = table.read_metadata()?;
    let result = table.read_job(id);
    let finish = anchor.finish();
    let j = result?;
    finish?;
    Ok(j)
}
pub(crate) fn build_config(table: &DiskAnnTable) -> BuildConfig {
    BuildConfig {
        dimension: table.space().dim,
        metric: match table.space().distance_type {
            DistanceType::Cosine => BuildMetric::CosineNormalized,
            _ => BuildMetric::L2Squared,
        },
        max_degree: table.options().degree,
        construction_l: table.options().build_list_size,
        alpha: table.options().alpha,
        seed: 42,
        workers: 1,
    }
}
/// Persist only a public digest output, never SHA implementation-private state.
pub(crate) fn chain_header(config: [u8; 32], revision: u64, cutoff: u64, count: u64) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"vectorlite/deferred/source-chain/v1");
    h.update(config);
    h.update(revision.to_le_bytes());
    h.update(cutoff.to_le_bytes());
    h.update(count.to_le_bytes());
    h.finalize().into()
}
pub(crate) fn chain_row(
    previous: [u8; 32],
    ordinal: u64,
    entry: u64,
    rowid: u64,
    bytes: &[u8],
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"vectorlite/deferred/source-row/v1");
    h.update(previous);
    h.update(ordinal.to_le_bytes());
    h.update(entry.to_le_bytes());
    h.update(rowid.to_le_bytes());
    h.update((bytes.len() as u64).to_le_bytes());
    h.update(bytes);
    h.finalize().into()
}

pub(crate) fn prepare(table: &DiskAnnTable) -> Result<Job> {
    let m = metadata(table)?;
    if m.active_job.is_some() {
        return Err(stale("active job must be cancelled first"));
    }
    if m.canonical_count > u64::from(u32::MAX - 1) {
        return Err(too_big("DiskANN native population exceeds capacity"));
    }
    let next_job = checked_increment(m.next_job_id, "job allocator")?;
    let next_gen = checked_increment(m.next_generation_id, "generation allocator")?;
    let next_fence = checked_increment(m.next_fence, "fence allocator")?;
    let cutoff = m.next_entry_id - 1;
    let config = build_config(table).digest();
    let chain = chain_header(config, m.revision, cutoff, m.canonical_count);
    let mut g = table.connection().prepare(&format!("INSERT INTO {}(generation_id,job_id,state,source_revision,source_hash,config_hash,total_count,eligible_count,root_local_id) VALUES(?1,?2,0,?3,NULL,?4,?5,0,?6)",table.names().generations))?;
    g.bind_i64(1, checked_id(m.next_generation_id)?)?;
    g.bind_i64(2, checked_id(m.next_job_id)?)?;
    g.bind_i64(3, checked_id(m.revision)?)?;
    g.bind_blob(4, &config)?;
    g.bind_i64(5, checked_id(m.canonical_count)?)?;
    if m.canonical_count == 0 {
        /* Fresh prepared parameters default to SQL NULL; statement is never reused. */
    } else {
        g.bind_i64(6, checked_id(m.canonical_count)?)?;
    }
    done(g)?;
    changed(table, "generation allocation")?;
    let mut j = table.connection().prepare(&format!("INSERT INTO {}(job_id,generation_id,parent_generation,cutoff_revision,fence,owner_nonce,phase,expected_count,scan_entry_id,captured_count,import_cursor,validation_cursor,cleanup_cursor,source_hash,config_hash,candidate_hash,failure_code,cutoff_entry_id,snapshot_chain) VALUES(?1,?2,?3,?4,?5,NULL,0,?6,0,0,0,0,0,NULL,?7,NULL,NULL,?8,?9)",table.names().jobs))?;
    j.bind_i64(1, checked_id(m.next_job_id)?)?;
    j.bind_i64(2, checked_id(m.next_generation_id)?)?;
    if let Some(p) = m.published_generation {
        j.bind_i64(3, checked_id(p)?)?;
    } else {
        /* Fresh prepared parameters default to SQL NULL; statement is never reused. */
    }
    j.bind_i64(4, checked_id(m.revision)?)?;
    j.bind_i64(5, checked_id(m.next_fence)?)?;
    j.bind_i64(6, checked_id(m.canonical_count)?)?;
    j.bind_blob(7, &config)?;
    j.bind_i64(8, checked_id(cutoff)?)?;
    j.bind_blob(9, &chain)?;
    done(j)?;
    changed(table, "job allocation")?;
    let mut s=table.connection().prepare(&format!("UPDATE {} SET next_job_id=?1,next_generation_id=?2,next_fence=?3,active_job=?4,candidate_generation=?5 WHERE singleton=1 AND active_job IS NULL AND next_job_id=?4 AND next_generation_id=?5",table.names().meta))?;
    for (i, v) in [
        (1, next_job),
        (2, next_gen),
        (3, next_fence),
        (4, m.next_job_id),
        (5, m.next_generation_id),
    ] {
        s.bind_i64(i, checked_id(v)?)?;
    }
    done(s)?;
    changed(table, "prepare fence")?;
    table.read_job(m.next_job_id)
}

pub(crate) fn active(table: &DiskAnnTable, id: u64) -> Result<Job> {
    let m = metadata(table)?;
    let j = table.read_job(id)?;
    if m.active_job != Some(id)
        || m.candidate_generation != Some(j.generation_id)
        || m.published_generation != j.parent_generation
        || j.phase >= COMPLETED
    {
        return Err(stale("inactive job or parent publication changed"));
    }
    Ok(j)
}
pub(crate) fn bound(table: &DiskAnnTable, b: &BuildBinding) -> Result<Job> {
    let m = metadata(table)?;
    let j = active(table, b.job_id)?;
    if m.instance_id != b.instance_id
        || j.generation_id != b.generation_id
        || j.fence != b.fence
        || j.owner_nonce != Some(b.owner_nonce)
        || j.source_hash != Some(b.source_hash)
        || j.config_hash != b.config_hash
        || build_config(table).digest() != b.config_hash
    {
        return Err(stale("owner/fence/generation/source/config mismatch"));
    }
    let g = table.read_generation(j.generation_id)?;
    if g.state != 0
        || g.job_id != j.job_id
        || g.source_hash != j.source_hash
        || g.config_hash != j.config_hash
        || g.total_count != j.expected_count
        || g.source_revision != j.cutoff_revision
    {
        return Err(corrupt("DiskANN candidate generation binding"));
    }
    Ok(j)
}
pub(crate) fn snapshot_step(table: &DiskAnnTable, id: u64, opts: StepOptions) -> Result<Job> {
    let j = active(table, id)?;
    if j.phase == SEALED {
        return Ok(j);
    }
    if j.phase != CAPTURING {
        return Err(stale("snapshot phase"));
    }
    let per = table
        .vector_bytes()
        .checked_add(40)
        .ok_or_else(|| too_big("snapshot row bytes"))?;
    if opts.max_bytes < per {
        return Err(too_big("snapshot step byte budget below one row"));
    }
    let limit = (opts.max_records as usize).min(opts.max_bytes / per);
    // LIMIT applies to scanned entry records, including retired versions.
    let mut s = table.connection().prepare(&format!(
        "SELECT entry_id FROM {} WHERE entry_id>?1 AND entry_id<=?2 ORDER BY entry_id LIMIT ?3",
        table.names().entries
    ))?;
    s.bind_i64(1, checked_id(j.scan_entry_id)?)?;
    s.bind_i64(2, checked_id(j.cutoff_entry_id)?)?;
    s.bind_i64(3, limit as i64)?;
    let mut ids = Vec::new();
    reserve(&mut ids, limit)?;
    while s.step()? == Step::Row {
        ids.push(nonnegative(&s, 0, "scan entry")?);
    }
    s.finish()?;
    let exhausted = ids.len() < limit;
    let mut scan = j.scan_entry_id;
    let mut count = j.captured_count;
    let mut chain = j.snapshot_chain;
    for id in ids {
        if count == j.expected_count {
            break;
        }
        let e = table.read_entry(id)?;
        scan = id;
        if e.created_revision <= j.cutoff_revision
            && e.retired_revision.is_none_or(|r| r > j.cutoff_revision)
        {
            let mut map = table.connection().prepare(&format!(
                "INSERT INTO {} VALUES(?1,?2,?3,?4)",
                table.names().snapshot_map
            ))?;
            for (i, v) in [(1, j.job_id), (2, count), (3, id), (4, e.public_rowid)] {
                map.bind_i64(i, checked_id(v)?)?;
            }
            done(map)?;
            changed(table, "snapshot map insert")?;
            chain = chain_row(chain, count, id, e.public_rowid, &e.vector);
            count += 1;
        }
    }
    if count < j.expected_count && (exhausted || scan == j.cutoff_entry_id) {
        return Err(corrupt("DiskANN snapshot early exhaustion"));
    }
    let sealed = count == j.expected_count;
    let mut s=table.connection().prepare(&format!("UPDATE {} SET scan_entry_id=?1,captured_count=?2,snapshot_chain=?3,source_hash=?4,phase=?5 WHERE job_id=?6 AND fence=?7 AND phase=0 AND captured_count=?8",table.names().jobs))?;
    s.bind_i64(1, checked_id(scan)?)?;
    s.bind_i64(2, checked_id(count)?)?;
    s.bind_blob(3, &chain)?;
    if sealed {
        s.bind_blob(4, &chain)?;
    } else {
        /* Fresh prepared parameters default to SQL NULL; statement is never reused. */
    }
    s.bind_i64(
        5,
        if sealed {
            SEALED as i64
        } else {
            CAPTURING as i64
        },
    )?;
    s.bind_i64(6, checked_id(j.job_id)?)?;
    s.bind_i64(7, checked_id(j.fence)?)?;
    s.bind_i64(8, checked_id(j.captured_count)?)?;
    done(s)?;
    changed(table, "snapshot progress")?;
    if sealed {
        let mut g=table.connection().prepare(&format!("UPDATE {} SET source_hash=?1 WHERE generation_id=?2 AND state=0 AND source_hash IS NULL",table.names().generations))?;
        g.bind_blob(1, &chain)?;
        g.bind_i64(2, checked_id(j.generation_id)?)?;
        done(g)?;
        changed(table, "snapshot seal")?;
    }
    table.read_job(id)
}
pub(crate) fn claim_build(
    table: &DiskAnnTable,
    id: u64,
    owner_nonce: [u8; 16],
) -> Result<BuildBinding> {
    let j = active(table, id)?;
    if j.phase == BUILDING
        && j.owner_nonce == Some(owner_nonce)
        && j.import_cursor == 0
        && j.candidate_hash.is_none()
    {
        // Replays within the same session are resolved by CandidateSessions; lost RAM
        // requires cancel/restart, never reset/rebuild into a partly imported generation.
    } else if j.phase == SEALED && j.owner_nonce.is_none() {
        let mut s=table.connection().prepare(&format!("UPDATE {} SET owner_nonce=?1,phase=2 WHERE job_id=?2 AND fence=?3 AND phase=1 AND owner_nonce IS NULL",table.names().jobs))?;
        s.bind_blob(1, &owner_nonce)?;
        s.bind_i64(2, checked_id(id)?)?;
        s.bind_i64(3, checked_id(j.fence)?)?;
        done(s)?;
        changed(table, "claim ownership")?;
    } else {
        return Err(stale("claim requires sealed unowned job"));
    }
    Ok(BuildBinding {
        instance_id: metadata(table)?.instance_id,
        job_id: id,
        generation_id: j.generation_id,
        fence: j.fence,
        owner_nonce,
        source_hash: j
            .source_hash
            .ok_or_else(|| corrupt("DiskANN sealed source missing"))?,
        config_hash: j.config_hash,
    })
}

/// Read under a tracked SQL snapshot; the guard finalizes ALL readers before CPU.
/// Returns canonical floats without cosine renormalization. Logical SHA chain is
/// recomputed once while materializing; native CheckedSnapshot owns its raw digest.
pub(crate) fn materialize_snapshot(
    table: &DiskAnnTable,
    b: &BuildBinding,
    readers: &mut SnapshotReaders<'_, '_>,
    max_bytes: usize,
) -> Result<MaterializedSnapshot> {
    let anchor = readers.prepare(&format!(
        "SELECT singleton FROM {} WHERE singleton=1",
        table.names().meta
    ))?;
    if anchor.step()? != Step::Row {
        return Err(corrupt("DiskANN materializer metadata anchor"));
    }
    let j = bound(table, b)?;
    if j.phase != BUILDING || j.captured_count != j.expected_count {
        return Err(stale("materialize phase"));
    }
    let n = usize::try_from(j.expected_count).map_err(|_| too_big("population"))?;
    let bytes = n
        .checked_mul(table.vector_bytes() + std::mem::size_of::<SnapshotNode>())
        .ok_or_else(|| too_big("snapshot materialization bytes"))?;
    if bytes > max_bytes {
        return Err(too_big("DiskANN snapshot owned-input budget exceeded"));
    }
    let mut vectors = Vec::new();
    reserve(
        &mut vectors,
        checked_product(n, table.space().dim, "snapshot floats")?,
    )?;
    let mut nodes = Vec::new();
    reserve(&mut nodes, n)?;
    let sql=format!("SELECT m.ordinal,m.entry_id,m.public_rowid,e.public_rowid,e.created_revision,e.retired_revision,{} FROM {} m LEFT JOIN {} e ON e.entry_id=m.entry_id WHERE m.job_id=?1 ORDER BY m.ordinal LIMIT ?2",blob_projection("e.vector",table.vector_bytes(),true),table.names().snapshot_map,table.names().entries);
    let s = readers.prepare(&sql)?;
    s.bind_i64(1, checked_id(j.job_id)?)?;
    s.bind_i64(2, checked_id(j.expected_count + 1)?)?;
    let mut chain = chain_header(
        j.config_hash,
        j.cutoff_revision,
        j.cutoff_entry_id,
        j.expected_count,
    );
    let mut count = 0;
    while s.step()? == Step::Row {
        let ordinal = nonnegative(s, 0, "snapshot ordinal")?;
        let entry = nonnegative(s, 1, "snapshot entry")?;
        let rowid = nonnegative(s, 2, "snapshot rowid")?;
        let created = nonnegative(s, 4, "snapshot creation")?;
        let retired = optional_id(s, 5, "snapshot retirement")?;
        if ordinal != count
            || count >= j.expected_count
            || entry == 0
            || entry > j.cutoff_entry_id
            || rowid != nonnegative(s, 3, "entry rowid")?
            || created == 0
            || created > j.cutoff_revision
            || retired.is_some_and(|r| r <= j.cutoff_revision)
        {
            return Err(corrupt("DiskANN frozen snapshot identity/interval"));
        }
        let blob = bounded_blob(s, 6, 7, table.vector_bytes(), true, "snapshot vector")?;
        with_decoded_vector(&blob, table.space().dim, &mut Vec::new(), &mut |v| {
            vectors.extend_from_slice(v.as_slice());
            Ok(())
        })?;
        nodes.push(SnapshotNode::new(entry, created));
        chain = chain_row(chain, count, entry, rowid, &blob);
        count += 1;
    }
    if count != j.expected_count || Some(chain) != j.source_hash || chain != j.snapshot_chain {
        return Err(corrupt(
            "DiskANN sealed logical source digest/count mismatch",
        ));
    }
    Ok(MaterializedSnapshot {
        vectors,
        nodes,
        source_hash: chain,
        config: build_config(table),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_chain_is_chunk_independent_and_identity_sensitive() {
        let h = chain_header([7; 32], 12, 25, 2);
        let a = chain_row(h, 0, 3, 99, &[0, 1, 2, 3]);
        let b = chain_row(a, 1, 8, 99, &[4, 5, 6, 7]);
        assert_eq!(
            b,
            chain_row(
                chain_row(h, 0, 3, 99, &[0, 1, 2, 3]),
                1,
                8,
                99,
                &[4, 5, 6, 7]
            )
        );
        assert_ne!(b, chain_row(a, 1, 9, 99, &[4, 5, 6, 7]));
        assert_ne!(h, chain_header([7; 32], 12, 26, 2));
    }
    #[test]
    fn step_bounds_are_explicit() {
        assert!(StepOptions::new(0, 1).is_err());
        assert!(StepOptions::new(257, 1).is_err());
        assert!(StepOptions::new(1, 0).is_err());
        assert!(StepOptions::new(256, 4096).is_ok());
    }
}
