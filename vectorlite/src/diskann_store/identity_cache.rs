//! Bounded reuse of checked identities within one no-write interval.
//!
//! This cache owns values, never SQLite rows or scoped provider handles. Its
//! caller must invalidate around every write (and any read failure or detected
//! reentrant DML) and never caches vector payloads or adjacency lists.

use std::mem::size_of;

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

/// Inline 256-entry 4-way set-associative cache with optional budget-admitted
/// 16,384-entry overflow table for larger graphs.
///
/// Keeping the inline footprint identical to the baseline `IdentityCache`
/// preserves exact pre-DDL `cache_bytes` admission boundaries on tiny budgets
/// while eliminating set-conflict misses when ample `cache_bytes` is configured.
pub(super) struct IdentityCache {
    entries: [[Entry; WAYS]; SETS],
    lengths: [u8; SETS],
    large: Option<LargeIdentityCache>,
}

struct LargeIdentityCache {
    sets: Vec<[Entry; WAYS]>,
    lengths: Vec<u8>,
    set_bits: u32,
}

impl LargeIdentityCache {
    fn try_new(set_bits: u32) -> Option<Self> {
        let count = 1usize.checked_shl(set_bits)?;
        let mut sets = Vec::new();
        sets.try_reserve_exact(count).ok()?;
        sets.resize(count, [EMPTY; WAYS]);
        let mut lengths = Vec::new();
        lengths.try_reserve_exact(count).ok()?;
        lengths.resize(count, 0);
        Some(Self {
            sets,
            lengths,
            set_bits,
        })
    }

    #[inline]
    fn index(&self, id: u64) -> usize {
        (id.wrapping_mul(HASH_MULTIPLIER) >> (u64::BITS - self.set_bits)) as usize
    }

    #[inline]
    fn get(&mut self, id: u64) -> Option<Identity> {
        let set = self.index(id);
        let length = usize::from(self.lengths[set]);
        let entries = &mut self.sets[set];
        let position = entries[..length].iter().position(|entry| entry.id == id)?;
        let entry = entries[position];
        entries.copy_within(..position, 1);
        entries[0] = entry;
        Some(entry.identity)
    }

    #[inline]
    fn insert(&mut self, id: u64, identity: Identity) {
        let set = self.index(id);
        let length = usize::from(self.lengths[set]);
        let entries = &mut self.sets[set];
        let existing = entries[..length].iter().position(|entry| entry.id == id);
        let shift = existing.unwrap_or(length.min(WAYS - 1));
        entries.copy_within(..shift, 1);
        entries[0] = Entry { id, identity };
        if existing.is_none() && length < WAYS {
            self.lengths[set] += 1;
        }
    }

    fn clear(&mut self) {
        self.lengths.fill(0);
    }
}

impl IdentityCache {
    /// Inline byte footprint reserved upfront in `transient_workspace_bytes`.
    pub(super) const INLINE_BYTES: usize =
        size_of::<[[Entry; WAYS]; SETS]>() + size_of::<[u8; SETS]>();

    pub(super) const fn new() -> Self {
        Self {
            entries: [[EMPTY; WAYS]; SETS],
            lengths: [0; SETS],
            large: None,
        }
    }

    /// Upgrade to a 16,384-entry (`4096 * 4`) set-associative table when the
    /// caller has explicitly reserved `LARGE_BYTES` from surplus `cache_bytes`.
    pub(super) const LARGE_SET_BITS: u32 = 12;
    pub(super) const LARGE_BYTES: usize =
        (1usize << Self::LARGE_SET_BITS) * (size_of::<[Entry; WAYS]>() + size_of::<u8>()) + 256;

    pub(super) fn enable_large(&mut self) {
        if self.large.is_none() {
            self.large = LargeIdentityCache::try_new(Self::LARGE_SET_BITS);
        }
    }

    #[inline]
    fn set(id: u64) -> usize {
        // High bits distribute contiguous and stride-64 IDs without relying on
        // randomized hash state or adding any graph-algorithm RNG consumption.
        (id.wrapping_mul(HASH_MULTIPLIER) >> (u64::BITS - SET_BITS)) as usize
    }

    #[inline]
    pub(super) fn get(&mut self, id: u64) -> Option<Identity> {
        if let Some(large) = &mut self.large {
            return large.get(id);
        }
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
    #[inline]
    pub(super) fn insert(&mut self, id: u64, identity: Identity) {
        if let Some(large) = &mut self.large {
            large.insert(id, identity);
            return;
        }
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
        if let Some(large) = &mut self.large {
            large.clear();
        }
    }

    /// Invalidate before AND after an action, including an error result.
    /// Capturing the result before the second clear prevents early `?` exits
    /// from retaining identities across failed prepare/bind/step/cleanup paths.
    #[cfg(test)]
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
        assert_eq!(cache.get(0), Some((NodeState::Frozen, None)));
        cache.clear();
        assert_eq!(cache.get(0), None);
    }

    #[test]
    fn per_set_lru_evicts_least_recent_way_and_refreshes_hits() {
        let mut cache = IdentityCache::new();
        let ids = collisions();
        for &id in &ids[..WAYS] {
            cache.insert(id, identity(id));
        }
        // Touch the oldest entry so the second-oldest becomes the victim.
        assert_eq!(cache.get(ids[0]), Some(identity(ids[0])));
        cache.insert(ids[WAYS], identity(ids[WAYS]));
        assert_eq!(cache.get(ids[1]), None);
        for &id in &[ids[0], ids[2], ids[3], ids[4]] {
            assert_eq!(cache.get(id), Some(identity(id)));
        }
    }

    #[test]
    fn updating_existing_id_does_not_duplicate_or_overflow_set() {
        let mut cache = IdentityCache::new();
        let ids = collisions();
        for &id in &ids[..WAYS] {
            cache.insert(id, (NodeState::Live, Some(id)));
        }
        for round in 0..8 {
            cache.insert(ids[2], (NodeState::Deleted, None));
            assert_eq!(cache.get(ids[2]), Some((NodeState::Deleted, None)));
            cache.insert(ids[2], (NodeState::Live, Some(100 + round)));
            assert_eq!(
                cache.get(ids[2]),
                Some((NodeState::Live, Some(100 + round)))
            );
        }
        assert!(cache.lengths.iter().all(|&len| usize::from(len) <= WAYS));
    }

    #[test]
    fn invalidation_clears_before_and_after_success_or_failure() {
        let mut cache = IdentityCache::new();
        cache.insert(1, (NodeState::Live, Some(0)));
        let result: Result<(), &'static str> = cache.with_invalidated(|inner| {
            assert_eq!(inner.get(1), None);
            inner.insert(2, (NodeState::Live, Some(1)));
            Err("write failed after reading inside action")
        });
        assert!(result.is_err());
        assert_eq!(cache.get(1), None);
        assert_eq!(cache.get(2), None);
    }

    #[test]
    fn hash_distributes_contiguous_and_power_of_two_strides() {
        let contiguous: BTreeSet<_> = (0..64).map(IdentityCache::set).collect();
        let strided: BTreeSet<_> = (0..64).map(|i| IdentityCache::set(i * 64)).collect();
        assert!(contiguous.len() > SETS / 2);
        assert!(strided.len() > SETS / 2);
    }

    #[test]
    fn cache_footprint_stays_small_and_heap_free_by_default() {
        assert_eq!(
            IdentityCache::INLINE_BYTES,
            size_of::<Entry>() * SETS * WAYS + SETS
        );
        const { assert!(IdentityCache::INLINE_BYTES < 16 * 1024) };
        let mut large = IdentityCache::new();
        large.enable_large();
        for id in 0..2048 {
            large.insert(id, identity(id));
        }
        assert_eq!(large.get(0), Some((NodeState::Frozen, None)));
        assert_eq!(large.get(1024), Some(identity(1024)));
        large.clear();
        assert_eq!(large.get(0), None);
    }
}
