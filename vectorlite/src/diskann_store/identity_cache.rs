//! Bounded reuse of checked identities within one no-write interval.
//!
//! This cache owns values, never SQLite rows or scoped provider handles. Its
//! caller must invalidate around every write and seed only successful reads.

use crate::diskann_core::NodeState;

pub(super) type Identity = (NodeState, Option<u64>);

const SET_BITS: u32 = 6;
const SETS: usize = 1 << SET_BITS;
const WAYS: usize = 4;
const HASH_MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;

#[derive(Clone, Copy)]
struct Entry {
    id: u64,
    identity: Identity,
}

const EMPTY: Entry = Entry {
    id: 0,
    identity: (NodeState::Frozen, None),
};

/// Four-way per-set LRU with no allocation or generation-counter overflow.
/// Valid prefix lengths, not sentinel IDs, distinguish an empty slot from node 0.
pub(super) struct IdentityCache {
    entries: [[Entry; WAYS]; SETS],
    lengths: [u8; SETS],
}

impl IdentityCache {
    pub(super) const fn new() -> Self {
        Self {
            entries: [[EMPTY; WAYS]; SETS],
            lengths: [0; SETS],
        }
    }

    fn set(id: u64) -> usize {
        // High bits distribute contiguous and stride-64 IDs without relying on
        // randomized hash state or adding any graph-algorithm RNG consumption.
        (id.wrapping_mul(HASH_MULTIPLIER) >> (u64::BITS - SET_BITS)) as usize
    }

    pub(super) fn get(&mut self, id: u64) -> Option<Identity> {
        let set = Self::set(id);
        let length = usize::from(self.lengths[set]);
        let entries = &mut self.entries[set];
        let position = entries[..length].iter().position(|entry| entry.id == id)?;
        let entry = entries[position];
        entries.copy_within(..position, 1);
        entries[0] = entry;
        Some(entry.identity)
    }

    /// Remember a fully checked, successfully completed SQLite read.
    pub(super) fn insert(&mut self, id: u64, identity: Identity) {
        let set = Self::set(id);
        let length = usize::from(self.lengths[set]);
        let entries = &mut self.entries[set];
        let existing = entries[..length].iter().position(|entry| entry.id == id);
        let shift = existing.unwrap_or(length.min(WAYS - 1));
        entries.copy_within(..shift, 1);
        entries[0] = Entry { id, identity };
        if existing.is_none() && length < WAYS {
            self.lengths[set] += 1;
        }
    }

    pub(super) fn clear(&mut self) {
        self.lengths.fill(0);
    }

    /// Invalidate before AND after an action, including an error result.
    /// Capturing the result before the second clear prevents early `?` exits
    /// from retaining identities across failed prepare/bind/step/cleanup paths.
    pub(super) fn with_invalidated<T>(&mut self, action: impl FnOnce(&mut Self) -> T) -> T {
        self.clear();
        let result = action(self);
        self.clear();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::mem::size_of;

    fn identity(id: u64) -> Identity {
        if id == 0 {
            (NodeState::Frozen, None)
        } else {
            (NodeState::Live, Some(id - 1))
        }
    }

    fn collisions() -> [u64; WAYS + 1] {
        let mut ids = [0; WAYS + 1];
        let mut found = 0;
        for id in 0..4096 {
            if IdentityCache::set(id) == 0 {
                ids[found] = id;
                found += 1;
                if found == ids.len() {
                    return ids;
                }
            }
        }
        panic!("not enough collision fixtures");
    }

    #[test]
    fn empty_slots_do_not_alias_frozen_zero() {
        let mut cache = IdentityCache::new();
        assert_eq!(cache.get(0), None);
        cache.insert(0, (NodeState::Frozen, None));
        cache.insert(1, (NodeState::Live, Some(0)));
        assert_eq!(cache.get(0), Some((NodeState::Frozen, None)));
        assert_eq!(cache.get(1), Some((NodeState::Live, Some(0))));
    }

    #[test]
    fn preserves_every_state_and_large_sparse_identity() {
        let mut cache = IdentityCache::new();
        let id = i64::MAX as u64;
        for value in [
            (NodeState::Live, Some(id)),
            (NodeState::Deleted, None),
            (NodeState::Frozen, None),
        ] {
            cache.insert(id, value);
            assert_eq!(cache.get(id), Some(value));
        }
        assert_eq!(
            cache.lengths.iter().map(|&n| usize::from(n)).sum::<usize>(),
            1
        );
    }

    #[test]
    fn full_keys_and_hit_promotion_control_set_eviction() {
        let ids = collisions();
        let mut cache = IdentityCache::new();
        for &id in &ids[..WAYS] {
            cache.insert(id, identity(id));
        }
        assert_eq!(cache.get(ids[WAYS]), None);
        assert_eq!(cache.get(ids[0]), Some(identity(ids[0])));
        cache.insert(ids[WAYS], identity(ids[WAYS]));
        assert_eq!(cache.get(ids[1]), None);
        for &id in &[ids[0], ids[2], ids[3], ids[4]] {
            assert_eq!(cache.get(id), Some(identity(id)));
        }
        assert_eq!(usize::from(cache.lengths[0]), WAYS);
    }

    #[test]
    fn replacing_an_entry_does_not_duplicate_it() {
        let ids = collisions();
        let mut cache = IdentityCache::new();
        for &id in &ids[..WAYS] {
            cache.insert(id, identity(id));
        }
        cache.insert(ids[0], (NodeState::Deleted, None));
        cache.insert(ids[WAYS], identity(ids[WAYS]));
        assert_eq!(cache.get(ids[0]), Some((NodeState::Deleted, None)));
        assert_eq!(cache.get(ids[1]), None);
        assert_eq!(usize::from(cache.lengths[0]), WAYS);
    }

    #[test]
    fn capacity_and_workspace_are_fixed() {
        let mut cache = IdentityCache::new();
        for id in 0..4096 {
            cache.insert(id, identity(id));
            assert_eq!(cache.get(id), Some(identity(id)));
            assert!(cache
                .lengths
                .iter()
                .all(|&length| usize::from(length) <= WAYS));
        }
        assert_eq!(
            cache.lengths.iter().map(|&n| usize::from(n)).sum::<usize>(),
            SETS * WAYS
        );
        assert_eq!(
            size_of::<IdentityCache>(),
            size_of::<Entry>() * SETS * WAYS + SETS
        );
        assert!(size_of::<IdentityCache>() < 16 * 1024);
        let stride_sets: BTreeSet<_> = (0..64).map(|id| IdentityCache::set(id * 64)).collect();
        assert!(stride_sets.len() > SETS / 2);
    }

    #[test]
    fn clear_and_reuse_never_expose_old_prefix_entries() {
        let ids = collisions();
        let mut cache = IdentityCache::new();
        for &id in &ids[..WAYS] {
            cache.insert(id, identity(id));
        }
        cache.clear();
        cache.clear();
        for &id in &ids {
            assert_eq!(cache.get(id), None);
        }
        cache.insert(ids[WAYS], identity(ids[WAYS]));
        for &id in &ids[..WAYS] {
            assert_eq!(cache.get(id), None);
        }
        assert_eq!(cache.get(ids[WAYS]), Some(identity(ids[WAYS])));
    }

    #[test]
    fn write_bracket_clears_before_and_after_success() {
        let mut cache = IdentityCache::new();
        cache.insert(1, identity(1));
        let result = cache.with_invalidated(|during| {
            assert_eq!(during.get(1), None);
            during.insert(2, identity(2));
            17
        });
        assert_eq!(result, 17);
        assert_eq!(cache.get(1), None);
        assert_eq!(cache.get(2), None);
    }

    #[test]
    fn write_bracket_preserves_error_and_clears_new_values() {
        let mut cache = IdentityCache::new();
        cache.insert(1, identity(1));
        let result = cache.with_invalidated(|during| {
            assert_eq!(during.get(1), None);
            during.insert(2, identity(2));
            Err::<(), _>("original SQLite failure")
        });
        assert_eq!(result, Err("original SQLite failure"));
        assert_eq!(cache.get(1), None);
        assert_eq!(cache.get(2), None);
    }
}
