// Implementation of the C ABI declared in core_shim.h. Mirrors the behaviour of
// the original C++ virtual table's numeric paths (vector_space.cpp,
// virtual_table.cpp, constraint.cpp, quantization.cpp) but exposes a flat C
// interface so the Rust port can own the SQLite glue.
#include "core_shim.h"

#include <hnswlib/hnswlib.h>
#include <hwy/base.h>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_set>
#include <vector>

#include "ops/ops.h"

// Bring in the vectorlite hnswlib space implementations (header-only). These
// depend on ops/ops.h and hnswlib only.
#include "distance.h"

namespace {

struct VlIndexImpl {
  std::unique_ptr<hnswlib::SpaceInterface<float>> space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
  size_t dim = 0;
  int distance_type = VL_DISTANCE_L2;
  int vector_type = VL_VECTOR_F32;
  bool normalize = false;
  bool allow_replace_deleted = true;
};

char* DupError(const std::string& msg) {
  char* out = static_cast<char*>(std::malloc(msg.size() + 1));
  if (out != nullptr) {
    std::memcpy(out, msg.c_str(), msg.size() + 1);
  }
  return out;
}

void SetError(char** err, const std::string& msg) {
  if (err != nullptr) {
    *err = DupError(msg);
  }
}

std::unique_ptr<hnswlib::SpaceInterface<float>> CreateSpace(size_t dim,
                                                            int distance_type,
                                                            int vector_type) {
  // Cosine uses the inner-product space (vectors are normalized on the way in).
  bool inner_product = (distance_type == VL_DISTANCE_IP) ||
                       (distance_type == VL_DISTANCE_COSINE);
  if (inner_product) {
    switch (vector_type) {
      case VL_VECTOR_F32:
        return std::make_unique<vectorlite::InnerProductSpace>(dim);
      case VL_VECTOR_BF16:
        return std::make_unique<vectorlite::InnerProductSpaceBF16>(dim);
      case VL_VECTOR_F16:
        return std::make_unique<vectorlite::InnerProductSpaceF16>(dim);
      default:
        return nullptr;
    }
  }
  switch (vector_type) {
    case VL_VECTOR_F32:
      return std::make_unique<vectorlite::L2Space>(dim);
    case VL_VECTOR_BF16:
      return std::make_unique<vectorlite::L2SpaceBF16>(dim);
    case VL_VECTOR_F16:
      return std::make_unique<vectorlite::L2SpaceF16>(dim);
    default:
      return nullptr;
  }
}

// Produces the stored-type representation of an f32 vector (quantize + optional
// normalize) into `storage`, returning a pointer to the contiguous data to feed
// to hnswlib. Mirrors InsertOrUpdateVector / QueryExecutor::Execute.
const void* MakeStoredVector(const VlIndexImpl* impl, const float* data,
                             size_t dim, std::vector<float>& f32_storage,
                             std::vector<hwy::bfloat16_t>& bf16_storage,
                             std::vector<hwy::float16_t>& f16_storage) {
  switch (impl->vector_type) {
    case VL_VECTOR_F32: {
      if (!impl->normalize) {
        return data;
      }
      f32_storage.assign(data, data + dim);
      vectorlite::ops::Normalize(f32_storage.data(), dim);
      return f32_storage.data();
    }
    case VL_VECTOR_BF16: {
      bf16_storage.resize(dim);
      vectorlite::ops::QuantizeF32ToBF16(data, bf16_storage.data(), dim);
      if (impl->normalize) {
        vectorlite::ops::Normalize(bf16_storage.data(), dim);
      }
      return bf16_storage.data();
    }
    case VL_VECTOR_F16: {
      f16_storage.resize(dim);
      vectorlite::ops::QuantizeF32ToF16(data, f16_storage.data(), dim);
      if (impl->normalize) {
        vectorlite::ops::Normalize(f16_storage.data(), dim);
      }
      return f16_storage.data();
    }
    default:
      return nullptr;
  }
}

class RowidInFilter : public hnswlib::BaseFilterFunctor {
 public:
  explicit RowidInFilter(std::unordered_set<hnswlib::labeltype> ids)
      : ids_(std::move(ids)) {}
  bool operator()(hnswlib::labeltype id) override {
    return ids_.find(id) != ids_.end();
  }

 private:
  std::unordered_set<hnswlib::labeltype> ids_;
};

class RowidEqualsFilter : public hnswlib::BaseFilterFunctor {
 public:
  explicit RowidEqualsFilter(hnswlib::labeltype id) : id_(id) {}
  bool operator()(hnswlib::labeltype id) override { return id == id_; }

 private:
  hnswlib::labeltype id_;
};

// Mirrors vectorlite::IsRowidInIndex.
bool RowidInIndex(const hnswlib::HierarchicalNSW<float>& index,
                  hnswlib::labeltype rowid) {
  std::unique_lock<std::mutex> lock_label(index.getLabelOpMutex(rowid));
  std::unique_lock<std::mutex> lock_table(index.label_lookup_lock);
  auto search = index.label_lookup_.find(rowid);
  if (search == index.label_lookup_.end() ||
      index.isMarkedDeleted(search->second)) {
    return false;
  }
  return true;
}

}  // namespace

extern "C" {

VlIndex* vl_index_create(size_t dim, int distance_type, int vector_type,
                         size_t max_elements, size_t M, size_t ef_construction,
                         size_t random_seed, int allow_replace_deleted,
                         char** err) {
  if (dim == 0) {
    SetError(err, "Dimension must be greater than 0");
    return nullptr;
  }
  auto space = CreateSpace(dim, distance_type, vector_type);
  if (!space) {
    SetError(err, "Invalid distance/vector type");
    return nullptr;
  }
  try {
    auto impl = std::make_unique<VlIndexImpl>();
    impl->dim = dim;
    impl->distance_type = distance_type;
    impl->vector_type = vector_type;
    impl->normalize = (distance_type == VL_DISTANCE_COSINE);
    impl->allow_replace_deleted = allow_replace_deleted != 0;
    impl->index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space.get(), max_elements, M, ef_construction, random_seed,
        impl->allow_replace_deleted);
    impl->space = std::move(space);
    return reinterpret_cast<VlIndex*>(impl.release());
  } catch (const std::exception& ex) {
    SetError(err, ex.what());
    return nullptr;
  }
}

void vl_index_free(VlIndex* index) {
  delete reinterpret_cast<VlIndexImpl*>(index);
}

size_t vl_index_dim(const VlIndex* index) {
  return reinterpret_cast<const VlIndexImpl*>(index)->dim;
}

size_t vl_index_data_size(const VlIndex* index) {
  auto impl = reinterpret_cast<const VlIndexImpl*>(index);
  return impl->space->get_data_size();
}

int vl_index_add(VlIndex* index, const float* data, size_t len, uint64_t rowid,
                 char** err) {
  auto impl = reinterpret_cast<VlIndexImpl*>(index);
  if (len != impl->dim) {
    SetError(err, "Dimension mismatch");
    return 1;
  }
  std::vector<float> f32_storage;
  std::vector<hwy::bfloat16_t> bf16_storage;
  std::vector<hwy::float16_t> f16_storage;
  const void* stored = MakeStoredVector(impl, data, impl->dim, f32_storage,
                                        bf16_storage, f16_storage);
  if (stored == nullptr) {
    SetError(err, "Unrecognized vector type");
    return 1;
  }
  try {
    impl->index->addPoint(stored, static_cast<hnswlib::labeltype>(rowid),
                          impl->index->allow_replace_deleted_);
  } catch (const std::exception& ex) {
    SetError(err, ex.what());
    return 1;
  }
  return 0;
}

int vl_index_mark_delete(VlIndex* index, uint64_t rowid, char** err) {
  auto impl = reinterpret_cast<VlIndexImpl*>(index);
  try {
    impl->index->markDelete(static_cast<hnswlib::labeltype>(rowid));
  } catch (const std::exception& ex) {
    SetError(err, ex.what());
    return 1;
  }
  return 0;
}

int vl_index_contains(const VlIndex* index, uint64_t rowid) {
  auto impl = reinterpret_cast<const VlIndexImpl*>(index);
  return RowidInIndex(*impl->index, static_cast<hnswlib::labeltype>(rowid)) ? 1
                                                                            : 0;
}

int vl_index_get_vector(const VlIndex* index, uint64_t rowid, float* out) {
  auto impl = reinterpret_cast<const VlIndexImpl*>(index);
  auto label = static_cast<hnswlib::labeltype>(rowid);
  try {
    switch (impl->vector_type) {
      case VL_VECTOR_F32: {
        std::vector<float> vec = impl->index->getDataByLabel<float>(label);
        std::memcpy(out, vec.data(), vec.size() * sizeof(float));
        return 0;
      }
      case VL_VECTOR_BF16: {
        std::vector<hwy::bfloat16_t> stored =
            impl->index->getDataByLabel<hwy::bfloat16_t>(label);
        vectorlite::ops::BF16ToF32(stored.data(), out, stored.size());
        return 0;
      }
      case VL_VECTOR_F16: {
        std::vector<hwy::float16_t> stored =
            impl->index->getDataByLabel<hwy::float16_t>(label);
        vectorlite::ops::F16ToF32(stored.data(), out, stored.size());
        return 0;
      }
      default:
        return -1;
    }
  } catch (const std::exception&) {
    return -1;
  }
}

int vl_index_search(VlIndex* index, const float* query, size_t len, size_t k,
                    size_t ef_override, int filter_kind,
                    const uint64_t* filter_rowids, size_t filter_count,
                    float* out_distances, uint64_t* out_rowids, char** err) {
  auto impl = reinterpret_cast<VlIndexImpl*>(index);
  if (len != impl->dim) {
    SetError(err, "query vector's dimension doesn't match table's dimension");
    return -1;
  }

  std::unique_ptr<hnswlib::BaseFilterFunctor> filter;
  if (filter_kind == VL_FILTER_IN) {
    std::unordered_set<hnswlib::labeltype> ids;
    ids.reserve(filter_count);
    for (size_t i = 0; i < filter_count; i++) {
      ids.insert(static_cast<hnswlib::labeltype>(filter_rowids[i]));
    }
    filter = std::make_unique<RowidInFilter>(std::move(ids));
  } else if (filter_kind == VL_FILTER_EQ) {
    hnswlib::labeltype id =
        filter_count > 0 ? static_cast<hnswlib::labeltype>(filter_rowids[0]) : 0;
    filter = std::make_unique<RowidEqualsFilter>(id);
  }

  std::vector<float> f32_storage;
  std::vector<hwy::bfloat16_t> bf16_storage;
  std::vector<hwy::float16_t> f16_storage;
  const void* stored = MakeStoredVector(impl, query, impl->dim, f32_storage,
                                        bf16_storage, f16_storage);
  if (stored == nullptr) {
    SetError(err, "Unrecognized vector type");
    return -1;
  }

  // setEf mutates shared state on the index; restore it afterwards.
  const size_t original_ef = impl->index->ef_;
  if (ef_override != 0) {
    impl->index->setEf(ef_override);
  }
  int count = 0;
  try {
    auto result =
        impl->index->searchKnnCloserFirst(stored, k, filter.get());
    for (const auto& pair : result) {
      out_distances[count] = pair.first;
      out_rowids[count] = static_cast<uint64_t>(pair.second);
      count++;
    }
  } catch (const std::exception& ex) {
    impl->index->setEf(original_ef);
    SetError(err, ex.what());
    return -1;
  }
  impl->index->setEf(original_ef);
  return count;
}

int vl_index_save(VlIndex* index, const char* path, char** err) {
  auto impl = reinterpret_cast<VlIndexImpl*>(index);
  if (path == nullptr || path[0] == '\0') {
    SetError(err, "path must not be empty");
    return 1;
  }
  try {
    impl->index->saveIndex(path);
  } catch (const std::exception& ex) {
    SetError(err, ex.what());
    return 1;
  }
  return 0;
}

int vl_index_load(VlIndex* index, const char* path, char** err) {
  auto impl = reinterpret_cast<VlIndexImpl*>(index);
  if (path == nullptr || path[0] == '\0') {
    SetError(err, "path must not be empty");
    return 1;
  }
  if (!std::filesystem::exists(path)) {
    SetError(err, std::string("index file does not exist: ") + path);
    return 1;
  }

  std::unique_ptr<hnswlib::HierarchicalNSW<float>> new_index;
  try {
    new_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        impl->space.get(), std::string(path), /*nmslib=*/false,
        impl->index->max_elements_, impl->allow_replace_deleted);
  } catch (const std::exception& ex) {
    SetError(err, ex.what());
    return 1;
  }

  size_t file_data_size = new_index->label_offset_ - new_index->offsetData_;
  if (file_data_size != impl->space->get_data_size()) {
    SetError(err, "index data size mismatch: file has " +
                      std::to_string(file_data_size) +
                      " bytes per vector, table expects " +
                      std::to_string(impl->space->get_data_size()));
    return 1;
  }

  impl->index = std::move(new_index);
  return 0;
}

int vl_distance(const float* a, const float* b, size_t dim, int distance_type,
                float* out) {
  switch (distance_type) {
    case VL_DISTANCE_L2:
      *out = vectorlite::ops::L2DistanceSquared(a, b, dim);
      return 0;
    case VL_DISTANCE_IP:
      *out = vectorlite::ops::InnerProductDistance(a, b, dim);
      return 0;
    case VL_DISTANCE_COSINE: {
      std::vector<float> na(a, a + dim);
      std::vector<float> nb(b, b + dim);
      vectorlite::ops::Normalize(na.data(), dim);
      vectorlite::ops::Normalize(nb.data(), dim);
      *out = vectorlite::ops::InnerProductDistance(na.data(), nb.data(), dim);
      return 0;
    }
    default:
      return 1;
  }
}

const char* vl_best_target(void) { return vectorlite::ops::GetBestTarget(); }

void vl_free_err(char* err) { std::free(err); }

}  // extern "C"
