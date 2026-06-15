#include "index_registry.h"

#include <memory>
#include <utility>

namespace vectorlite {

IndexHandle* IndexRegistry::Find(const RegistryKey& key) {
  auto it = handles_.find(key);
  return it == handles_.end() ? nullptr : it->second.get();
}

IndexHandle* IndexRegistry::Insert(const RegistryKey& key, IndexHandle handle) {
  auto stored = std::make_unique<IndexHandle>(std::move(handle));
  IndexHandle* ptr = stored.get();
  handles_[key] = std::move(stored);
  return ptr;
}

void IndexRegistry::Erase(const RegistryKey& key) { handles_.erase(key); }

void IndexRegistry::Rename(const RegistryKey& old_key,
                           const RegistryKey& new_key) {
  if (old_key == new_key) {
    return;
  }
  auto it = handles_.find(old_key);
  if (it == handles_.end()) {
    return;
  }
  // Moving the unique_ptr transfers ownership without moving the IndexHandle
  // itself, so its address stays stable for references held by a live
  // VirtualTable.
  handles_[new_key] = std::move(it->second);
  handles_.erase(it);
}

}  // namespace vectorlite
