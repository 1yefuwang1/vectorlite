#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "hnswlib/hnswlib.h"
#include "vector_space.h"

namespace vectorlite {

// The stateful core of a vectorlite table. Owned by an IndexRegistry so that it
// outlives the short-lived VirtualTable object across schema reparses. The
// space and index are kept together because the hnswlib index caches a pointer
// into the space (see design doc).
struct IndexHandle {
  NamedVectorSpace space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
  // Runtime-only hnswlib flag (not serialized); retained to reapply on load.
  bool allow_replace_deleted;
  // The exact module-argument strings that defined this table. Used to detect a
  // table-name collision on xConnect.
  std::string vector_space_str;
  std::string index_options_str;
};

// (schema_name, table_name) uniquely identifies a table within a connection.
using RegistryKey = std::pair<std::string, std::string>;

// A per-connection map of live in-memory indexes. Not thread-safe; SQLite
// serializes access to a single connection.
class IndexRegistry {
 public:
  // Returns the handle for `key`, or nullptr if absent.
  IndexHandle* Find(const RegistryKey& key);

  // Stores `handle` under `key`, replacing any existing entry, and returns a
  // stable pointer to the stored handle.
  IndexHandle* Insert(const RegistryKey& key, IndexHandle handle);

  // Removes the entry for `key` if present.
  void Erase(const RegistryKey& key);

  // Moves the entry from `old_key` to `new_key`, preserving the handle's
  // address (and thus any references held into it). Any existing entry at
  // `new_key` is replaced. No-op if `old_key` is absent or equals `new_key`.
  void Rename(const RegistryKey& old_key, const RegistryKey& new_key);

 private:
  std::map<RegistryKey, std::unique_ptr<IndexHandle>> handles_;
};

}  // namespace vectorlite
