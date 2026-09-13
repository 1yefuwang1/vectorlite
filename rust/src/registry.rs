//! Per-connection registry of live in-memory indexes.
//!
//! Mirrors `index_registry.{h,cpp}`. The registry outlives the short-lived
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
