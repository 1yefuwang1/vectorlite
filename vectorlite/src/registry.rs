//! Per-connection registry of live in-memory indexes.
//!
//! The registry outlives the short-lived
//! virtual-table objects so the in-memory HNSW index survives schema reparses
//! (VACUUM, ALTER TABLE, foreign DDL, RENAME). It is owned by SQLite as the
//! module's `pAux` and is dropped when the module is unregistered.

use std::collections::HashMap;
use std::rc::Rc;

use crate::core::Index;
use crate::vector_space::NamedVectorSpace;

/// (schema_name, table_name) uniquely identifies a table within a connection.
pub type RegistryKey = (String, String);

/// The stateful core of a vectorlite table: the index plus the exact module
/// arguments that defined it (used to detect a table-name collision on connect).
pub struct IndexEntry {
    pub index: Index,
    pub space: NamedVectorSpace,
    pub vector_space_str: String,
    pub index_options_str: String,
}

#[derive(Default)]
pub struct Registry {
    // Each connected VTab retains an Rc too. Disconnect can therefore release
    // the wrapper without deleting the index, and callbacks need no map lookup.
    // SQLite serializes this connection's callbacks; no cross-thread Rust
    // ownership or Send/Sync implementation is needed for the registry.
    handles: HashMap<RegistryKey, Rc<IndexEntry>>,
}

impl Registry {
    pub fn new() -> Self {
        Registry {
            handles: HashMap::new(),
        }
    }

    pub fn find(&self, key: &RegistryKey) -> Option<Rc<IndexEntry>> {
        self.handles.get(key).cloned()
    }

    /// Stores `entry` under `key`, returning the retained handle for its VTab.
    /// Existing wrappers retain their own entry if this name is replaced.
    pub fn insert(&mut self, key: RegistryKey, entry: IndexEntry) -> Rc<IndexEntry> {
        let entry = Rc::new(entry);
        self.handles.insert(key, Rc::clone(&entry));
        entry
    }

    pub fn erase(&mut self, key: &RegistryKey) {
        self.handles.remove(key);
    }

    /// Moves the entry from `old_key` to `new_key`. Any existing entry at
    /// `new_key` is replaced. No-op if `old_key` is absent or equals `new_key`.
    pub fn rename(&mut self, old_key: &RegistryKey, new_key: RegistryKey) {
        if old_key == &new_key {
            return;
        }
        if let Some(entry) = self.handles.remove(old_key) {
            self.handles.insert(new_key, entry);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_space::{DistanceType, VectorType};

    fn key(schema: &str, table: &str) -> RegistryKey {
        (schema.to_owned(), table.to_owned())
    }

    fn entry(rowid: u64) -> IndexEntry {
        let index = Index::create(
            2,
            DistanceType::L2,
            VectorType::Float32,
            8,
            16,
            100,
            42,
            true,
        )
        .unwrap();
        index.add(&[1., 2.], rowid).unwrap();
        IndexEntry {
            index,
            space: NamedVectorSpace {
                vector_name: "embedding".to_owned(),
                dim: 2,
                distance_type: DistanceType::L2,
                vector_type: VectorType::Float32,
            },
            vector_space_str: "embedding float32[2]".to_owned(),
            index_options_str: "hnsw(max_elements=8)".to_owned(),
        }
    }

    #[test]
    fn keys_distinguish_schemas_and_tables() {
        let mut registry = Registry::new();
        assert!(registry.find(&key("main", "v")).is_none());
        let main = registry.insert(key("main", "v"), entry(1));
        let temp = registry.insert(key("temp", "v"), entry(2));
        let other = registry.insert(key("main", "other"), entry(3));
        assert!(Rc::ptr_eq(
            &registry.find(&key("main", "v")).unwrap(),
            &main
        ));
        assert!(Rc::ptr_eq(
            &registry.find(&key("temp", "v")).unwrap(),
            &temp
        ));
        assert!(Rc::ptr_eq(
            &registry.find(&key("main", "other")).unwrap(),
            &other
        ));
        assert!(!Rc::ptr_eq(&main, &temp));
    }

    #[test]
    fn replacement_retains_existing_live_handle() {
        let mut registry = Registry::new();
        let old = registry.insert(key("main", "v"), entry(1));
        let new = registry.insert(key("main", "v"), entry(2));
        assert!(Rc::ptr_eq(&registry.find(&key("main", "v")).unwrap(), &new));
        assert!(!Rc::ptr_eq(&old, &new));
        assert!(old.index.contains(1));
        assert!(!old.index.contains(2));
        assert!(new.index.contains(2));
    }

    #[test]
    fn erase_does_not_invalidate_live_handle() {
        let mut registry = Registry::new();
        let live = registry.insert(key("main", "v"), entry(1));
        registry.erase(&key("main", "v"));
        registry.erase(&key("main", "missing"));
        assert!(registry.find(&key("main", "v")).is_none());
        assert_eq!(live.index.get_vector(1), Some(vec![1., 2.]));
    }

    #[test]
    fn rename_preserves_identity_and_replaces_occupied_destination() {
        let mut registry = Registry::new();
        let source = registry.insert(key("main", "old"), entry(1));
        let occupied = registry.insert(key("main", "new"), entry(2));
        registry.rename(&key("main", "old"), key("main", "new"));
        assert!(registry.find(&key("main", "old")).is_none());
        assert!(Rc::ptr_eq(
            &registry.find(&key("main", "new")).unwrap(),
            &source
        ));
        assert!(source.index.contains(1));
        assert!(occupied.index.contains(2));
    }

    #[test]
    fn rename_missing_or_same_key_is_a_noop() {
        let mut registry = Registry::new();
        let live = registry.insert(key("main", "v"), entry(1));
        registry.rename(&key("main", "missing"), key("main", "v"));
        registry.rename(&key("main", "v"), key("main", "v"));
        assert!(Rc::ptr_eq(
            &registry.find(&key("main", "v")).unwrap(),
            &live
        ));
    }
}
