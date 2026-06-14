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

}  // namespace vectorlite
