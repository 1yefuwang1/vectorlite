//! Fenced SQL lifecycle and bounded reclamation, with no RAM side effects.
use super::jobs::{self, StepOptions, ABANDONED};
use super::*;

/// Force is a root-validated explicit takeover decision, not an owner credential.
/// This helper returns the OLD job with its NEW invalidating fence. The root must
/// perform ephemeral cancellation separately, after SQL success, and recognize
/// that an outer explicit transaction may still roll back.
pub(crate) fn cancel(
    table: &DiskAnnTable,
    id: u64,
    expected_fence: u64,
    owner: Option<[u8; 16]>,
    force: bool,
    failure_code: Option<i32>,
) -> Result<Job> {
    let m = jobs::metadata(table)?;
    let j = jobs::active(table, id)?;
    if j.fence != expected_fence || (!force && j.owner_nonce != owner) {
        return Err(jobs::stale("cancel owner/fence"));
    }
    let next = checked_increment(m.next_fence, "fence allocator")?;
    let mut s=table.connection().prepare(&format!("UPDATE {} SET phase=6,fence=?1,owner_nonce=NULL,failure_code=?2 WHERE job_id=?3 AND fence=?4 AND phase<5",table.names().jobs))?;
    s.bind_i64(1, checked_id(m.next_fence)?)?;
    if let Some(c) = failure_code {
        s.bind_i64(2, i64::from(c))?;
    } else {
        /* Fresh prepared parameters default to SQL NULL; statement is never reused. */
    }
    s.bind_i64(3, checked_id(id)?)?;
    s.bind_i64(4, checked_id(expected_fence)?)?;
    done(s)?;
    jobs::changed(table, "abandon job")?;
    let mut s = table.connection().prepare(&format!(
        "UPDATE {} SET state=2 WHERE generation_id=?1 AND job_id=?2 AND state=0",
        table.names().generations
    ))?;
    s.bind_i64(1, checked_id(j.generation_id)?)?;
    s.bind_i64(2, checked_id(id)?)?;
    done(s)?;
    jobs::changed(table, "abandon candidate generation")?;
    let mut s=table.connection().prepare(&format!("UPDATE {} SET active_job=NULL,candidate_generation=NULL,next_fence=?1 WHERE singleton=1 AND active_job=?2 AND candidate_generation=?3 AND next_fence=?4",table.names().meta))?;
    s.bind_i64(1, checked_id(next)?)?;
    s.bind_i64(2, checked_id(id)?)?;
    s.bind_i64(3, checked_id(j.generation_id)?)?;
    s.bind_i64(4, checked_id(m.next_fence)?)?;
    done(s)?;
    jobs::changed(table, "cancel metadata fence")?;
    let cancelled = table.read_job(id)?;
    debug_assert_eq!(cancelled.phase, ABANDONED);
    Ok(cancelled)
}
/// A new job, generation AND fence; never reset partial import into the old graph.
pub(crate) fn restart(
    table: &DiskAnnTable,
    id: u64,
    fence: u64,
    owner: Option<[u8; 16]>,
    force: bool,
) -> Result<Job> {
    cancel(table, id, fence, owner, force, None)?;
    jobs::prepare(table)
}
#[derive(Default, Debug)]
pub(crate) struct CleanupResult {
    pub(crate) graph_nodes: u64,
    pub(crate) snapshot_rows: u64,
    pub(crate) generations: u64,
    pub(crate) jobs: u64,
    pub(crate) historical_entries: u64,
    pub(crate) scanned_entries: u64,
}
fn deletion_count(table: &DiskAnnTable) -> Result<u64> {
    u64::try_from(table.connection().changes())
        .map_err(|_| corrupt("DiskANN negative delete count"))
}
/// Bounded PK-in-SELECT-LIMIT deletion works at SQLite floor without optional
/// SQLITE_ENABLE_UPDATE_DELETE_LIMIT. Protect current, published/staged graph and active map references,
/// and cutoff input even BEFORE snapshot_map capture completes.
pub(crate) fn cleanup_step(table: &DiskAnnTable, opts: StepOptions) -> Result<CleanupResult> {
    let m = jobs::metadata(table)?;
    let mut result = CleanupResult::default();
    let per = table
        .vector_bytes()
        .checked_add(table.options().physical_degree() * 8 + 96)
        .ok_or_else(|| too_big("cleanup budget overflow"))?;
    if opts.max_bytes < per {
        return Err(too_big("cleanup byte budget below one record"));
    }
    let mut remaining = (opts.max_records as usize).min(opts.max_bytes / per) as u64;
    let mut s = table.connection().prepare(&format!(
        "SELECT generation_id,job_id FROM {} WHERE state=2 ORDER BY generation_id LIMIT 1",
        table.names().generations
    ))?;
    let garbage = if s.step()? == Step::Row {
        Some((
            nonnegative(&s, 0, "garbage generation")?,
            nonnegative(&s, 1, "garbage job")?,
        ))
    } else {
        None
    };
    // Always finish this bounded anchor, including the ROW case.
    s.finish()?;
    if let Some((generation, job)) = garbage {
        if m.published_generation == Some(generation)
            || m.candidate_generation == Some(generation)
            || m.active_job == Some(job)
        {
            return Err(corrupt("DiskANN garbage references active state"));
        }
        let mut s=table.connection().prepare(&format!("DELETE FROM {} WHERE (generation_id,native_local_id) IN (SELECT generation_id,native_local_id FROM {} WHERE generation_id=?1 ORDER BY native_local_id LIMIT ?2)",table.names().graph_nodes,table.names().graph_nodes))?;
        s.bind_i64(1, checked_id(generation)?)?;
        s.bind_i64(2, checked_id(remaining)?)?;
        done(s)?;
        result.graph_nodes = deletion_count(table)?;
        remaining = remaining
            .checked_sub(result.graph_nodes)
            .ok_or_else(|| corrupt("DiskANN GC graph delete count"))?;
        if remaining > 0 {
            let mut s=table.connection().prepare(&format!("DELETE FROM {} WHERE (job_id,ordinal) IN (SELECT job_id,ordinal FROM {} WHERE job_id=?1 ORDER BY ordinal LIMIT ?2)",table.names().snapshot_map,table.names().snapshot_map))?;
            s.bind_i64(1, checked_id(job)?)?;
            s.bind_i64(2, checked_id(remaining)?)?;
            done(s)?;
            result.snapshot_rows = deletion_count(table)?;
            remaining = remaining
                .checked_sub(result.snapshot_rows)
                .ok_or_else(|| corrupt("DiskANN GC snapshot delete count"))?;
        }
        if remaining >= 1 {
            let mut s=table.connection().prepare(&format!("DELETE FROM {} WHERE generation_id=?1 AND state=2 AND NOT EXISTS(SELECT 1 FROM {} WHERE generation_id=?1 LIMIT 1) AND NOT EXISTS(SELECT 1 FROM {} WHERE job_id=?2 LIMIT 1)",table.names().generations,table.names().graph_nodes,table.names().snapshot_map))?;
            s.bind_i64(1, checked_id(generation)?)?;
            s.bind_i64(2, checked_id(job)?)?;
            done(s)?;
            result.generations = deletion_count(table)?;
            // Generation/job bookkeeping is at most two fixed rows, independent
            // of the bounded data-row budget (plus the singleton progress row).
            if result.generations == 1 {
                let mut s=table.connection().prepare(&format!("DELETE FROM {} WHERE job_id=?1 AND phase IN (5,6) AND NOT EXISTS(SELECT 1 FROM {} WHERE generation_id=?2 LIMIT 1)",table.names().jobs,table.names().generations))?;
                s.bind_i64(1, checked_id(job)?)?;
                s.bind_i64(2, checked_id(generation)?)?;
                done(s)?;
                result.jobs = deletion_count(table)?;
                if result.jobs != 1 {
                    return Err(corrupt("DiskANN garbage job lifecycle"));
                }
            }
        }
    }
    if remaining == 0 {
        return Ok(result);
    }
    let active = m.active_job.map(|id| table.read_job(id)).transpose()?;
    let mut s = table.connection().prepare(&format!(
        "SELECT entry_id FROM {} WHERE entry_id>?1 ORDER BY entry_id LIMIT ?2",
        table.names().entries
    ))?;
    s.bind_i64(1, checked_id(m.cleanup_entry_id)?)?;
    s.bind_i64(2, checked_id(remaining)?)?;
    let mut entries = Vec::new();
    reserve(&mut entries, remaining as usize)?;
    while s.step()? == Step::Row {
        entries.push(nonnegative(&s, 0, "cleanup entry")?);
    }
    s.finish()?;
    let scanned = entries.len() as u64;
    let mut cursor = m.cleanup_entry_id;
    for entry in entries {
        cursor = entry;
        let mut s=table.connection().prepare(&format!("DELETE FROM {} WHERE entry_id=?1 AND retired_revision IS NOT NULL AND NOT EXISTS(SELECT 1 FROM {} WHERE entry_id=?1) AND NOT EXISTS(SELECT 1 FROM {} WHERE generation_id IN (?4,?5) AND entry_id=?1) AND NOT EXISTS(SELECT 1 FROM {} WHERE job_id=?6 AND entry_id=?1) AND NOT (?2 IS NOT NULL AND entry_id<=?3 AND created_revision<=?2 AND retired_revision>?2)",table.names().entries,table.names().current,table.names().graph_nodes,table.names().snapshot_map))?;
        s.bind_i64(1, checked_id(entry)?)?;
        if let Some(j) = &active {
            s.bind_i64(2, checked_id(j.cutoff_revision)?)?;
            s.bind_i64(3, checked_id(j.cutoff_entry_id)?)?;
        } else {
            /* Fresh prepared parameters default to SQL NULL; statement is never reused. */
            /* Fresh prepared parameters default to SQL NULL; statement is never reused. */
        }
        if let Some(g) = m.published_generation {
            s.bind_i64(4, checked_id(g)?)?;
        } else {
            /* Fresh prepared parameters default to SQL NULL; statement is never reused. */
        }
        if let Some(g) = m.candidate_generation {
            s.bind_i64(5, checked_id(g)?)?;
        } else {
            /* Fresh prepared parameters default to SQL NULL; statement is never reused. */
        }
        if let Some(id) = m.active_job {
            s.bind_i64(6, checked_id(id)?)?;
        } else {
            /* Fresh prepared parameters default to SQL NULL; statement is never reused. */
        }
        done(s)?;
        let deleted = deletion_count(table)?;
        if deleted > 1 {
            return Err(corrupt("DiskANN historical deletion count"));
        }
        result.historical_entries += deleted;
    }
    // Wrap only after a bounded exhausted scan; future retirements behind the
    // cursor are reclaimed on a later cycle, never mistaken for complete GC.
    if scanned < remaining {
        cursor = 0;
    }
    let mut s=table.connection().prepare(&format!("UPDATE {} SET cleanup_entry_id=?1,retired_count=retired_count-?2 WHERE singleton=1 AND cleanup_entry_id=?3 AND retired_count>=?2",table.names().meta))?;
    s.bind_i64(1, checked_id(cursor)?)?;
    s.bind_i64(2, checked_id(result.historical_entries)?)?;
    s.bind_i64(3, checked_id(m.cleanup_entry_id)?)?;
    done(s)?;
    jobs::changed(table, "cleanup progress/retained retirement count")?;
    result.scanned_entries = scanned;
    Ok(result)
}
