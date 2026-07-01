// Implementation of the pure hnswlib+ops C ABI declared in core_shim.h.
// Contains only generic glue: forwarders to `ops`, an hnswlib SpaceInterface
// adapter around a caller-supplied distance function, an hnswlib filter adapter
// around a caller-supplied predicate, and thin wrappers over HierarchicalNSW.
// No virtual-table policy lives here.
#include "core_shim.h"

#include <hnswlib/hnswlib.h>
#include <hwy/base.h>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ops/ops.h"

namespace {

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

inline const hwy::bfloat16_t* AsBf16(const uint16_t* p) {
  return reinterpret_cast<const hwy::bfloat16_t*>(p);
}
inline hwy::bfloat16_t* AsBf16(uint16_t* p) {
  return reinterpret_cast<hwy::bfloat16_t*>(p);
}
inline const hwy::float16_t* AsF16(const uint16_t* p) {
  return reinterpret_cast<const hwy::float16_t*>(p);
}
inline hwy::float16_t* AsF16(uint16_t* p) {
  return reinterpret_cast<hwy::float16_t*>(p);
}

// Adapts a caller-supplied distance function into hnswlib's SpaceInterface.
class SpaceAdapter : public hnswlib::SpaceInterface<float> {
 public:
  SpaceAdapter(VlDistFunc func, size_t dim, size_t data_size)
      : func_(func), dim_(dim), data_size_(data_size) {}

  size_t get_data_size() override { return data_size_; }
  hnswlib::DISTFUNC<float> get_dist_func() override { return func_; }
  void* get_dist_func_param() override { return &dim_; }

 private:
  VlDistFunc func_;
  size_t dim_;
  size_t data_size_;
};

// Adapts a caller-supplied predicate into hnswlib's BaseFilterFunctor.
class CallbackFilter : public hnswlib::BaseFilterFunctor {
 public:
  CallbackFilter(VlFilterFunc func, void* ctx) : func_(func), ctx_(ctx) {}
  bool operator()(hnswlib::labeltype id) override {
    return func_(ctx_, static_cast<uint64_t>(id)) != 0;
  }

 private:
  VlFilterFunc func_;
  void* ctx_;
};

}  // namespace

extern "C" {

// ------------------------------------------------------------------ ops FFI --

float vl_ops_l2_sq_f32(const float* a, const float* b, size_t n) {
  return vectorlite::ops::L2DistanceSquared(a, b, n);
}
float vl_ops_l2_sq_bf16(const uint16_t* a, const uint16_t* b, size_t n) {
  return vectorlite::ops::L2DistanceSquared(AsBf16(a), AsBf16(b), n);
}
float vl_ops_l2_sq_f16(const uint16_t* a, const uint16_t* b, size_t n) {
  return vectorlite::ops::L2DistanceSquared(AsF16(a), AsF16(b), n);
}
float vl_ops_ip_dist_f32(const float* a, const float* b, size_t n) {
  return vectorlite::ops::InnerProductDistance(a, b, n);
}
float vl_ops_ip_dist_bf16(const uint16_t* a, const uint16_t* b, size_t n) {
  return vectorlite::ops::InnerProductDistance(AsBf16(a), AsBf16(b), n);
}
float vl_ops_ip_dist_f16(const uint16_t* a, const uint16_t* b, size_t n) {
  return vectorlite::ops::InnerProductDistance(AsF16(a), AsF16(b), n);
}

void vl_ops_normalize_f32(float* inout, size_t n) {
  vectorlite::ops::Normalize(inout, n);
}
void vl_ops_normalize_bf16(uint16_t* inout, size_t n) {
  vectorlite::ops::Normalize(AsBf16(inout), n);
}
void vl_ops_normalize_f16(uint16_t* inout, size_t n) {
  vectorlite::ops::Normalize(AsF16(inout), n);
}

void vl_ops_quantize_f32_to_bf16(const float* in, uint16_t* out, size_t n) {
  vectorlite::ops::QuantizeF32ToBF16(in, AsBf16(out), n);
}
void vl_ops_quantize_f32_to_f16(const float* in, uint16_t* out, size_t n) {
  vectorlite::ops::QuantizeF32ToF16(in, AsF16(out), n);
}
void vl_ops_bf16_to_f32(const uint16_t* in, float* out, size_t n) {
  vectorlite::ops::BF16ToF32(AsBf16(in), out, n);
}
void vl_ops_f16_to_f32(const uint16_t* in, float* out, size_t n) {
  vectorlite::ops::F16ToF32(AsF16(in), out, n);
}

const char* vl_ops_best_target(void) {
  return vectorlite::ops::GetBestTarget();
}

// -------------------------------------------------------------- hnswlib FFI --

VlSpace* vl_hnsw_space_create(VlDistFunc distfunc, size_t dim,
                              size_t data_size) {
  return reinterpret_cast<VlSpace*>(
      new SpaceAdapter(distfunc, dim, data_size));
}

void vl_hnsw_space_free(VlSpace* space) {
  delete reinterpret_cast<SpaceAdapter*>(space);
}

VlHnsw* vl_hnsw_create(VlSpace* space, size_t max_elements, size_t M,
                       size_t ef_construction, size_t random_seed,
                       int allow_replace_deleted, char** err) {
  auto* s = reinterpret_cast<SpaceAdapter*>(space);
  try {
    auto* index = new hnswlib::HierarchicalNSW<float>(
        s, max_elements, M, ef_construction, random_seed,
        allow_replace_deleted != 0);
    return reinterpret_cast<VlHnsw*>(index);
  } catch (const std::exception& ex) {
    SetError(err, ex.what());
    return nullptr;
  }
}

VlHnsw* vl_hnsw_load(VlSpace* space, const char* path, size_t max_elements,
                     int allow_replace_deleted, char** err) {
  auto* s = reinterpret_cast<SpaceAdapter*>(space);
  try {
    auto* index = new hnswlib::HierarchicalNSW<float>(
        s, std::string(path), /*nmslib=*/false, max_elements,
        allow_replace_deleted != 0);
    return reinterpret_cast<VlHnsw*>(index);
  } catch (const std::exception& ex) {
    SetError(err, ex.what());
    return nullptr;
  }
}

void vl_hnsw_free(VlHnsw* index) {
  delete reinterpret_cast<hnswlib::HierarchicalNSW<float>*>(index);
}

int vl_hnsw_add_point(VlHnsw* index, const void* data, uint64_t label,
                      int replace_deleted, char** err) {
  auto* idx = reinterpret_cast<hnswlib::HierarchicalNSW<float>*>(index);
  try {
    idx->addPoint(data, static_cast<hnswlib::labeltype>(label),
                  replace_deleted != 0);
  } catch (const std::exception& ex) {
    SetError(err, ex.what());
    return 1;
  }
  return 0;
}

int vl_hnsw_mark_delete(VlHnsw* index, uint64_t label, char** err) {
  auto* idx = reinterpret_cast<hnswlib::HierarchicalNSW<float>*>(index);
  try {
    idx->markDelete(static_cast<hnswlib::labeltype>(label));
  } catch (const std::exception& ex) {
    SetError(err, ex.what());
    return 1;
  }
  return 0;
}

int vl_hnsw_contains(VlHnsw* index, uint64_t label) {
  auto* idx = reinterpret_cast<hnswlib::HierarchicalNSW<float>*>(index);
  auto id = static_cast<hnswlib::labeltype>(label);
  std::unique_lock<std::mutex> lock_label(idx->getLabelOpMutex(id));
  std::unique_lock<std::mutex> lock_table(idx->label_lookup_lock);
  auto search = idx->label_lookup_.find(id);
  if (search == idx->label_lookup_.end() ||
      idx->isMarkedDeleted(search->second)) {
    return 0;
  }
  return 1;
}

int vl_hnsw_get_data(VlHnsw* index, uint64_t label, void* out, size_t nbytes) {
  auto* idx = reinterpret_cast<hnswlib::HierarchicalNSW<float>*>(index);
  auto id = static_cast<hnswlib::labeltype>(label);
  try {
    // Mirror hnswlib::getDataByLabel's lookup/deleted checks, then copy the
    // full per-vector byte blob (nbytes) from internal storage. Using
    // getDataByLabel<char> would instead copy only `dim` bytes, not the full
    // dim * element_size stored vector.
    std::unique_lock<std::mutex> lock_label(idx->getLabelOpMutex(id));
    std::unique_lock<std::mutex> lock_table(idx->label_lookup_lock);
    auto search = idx->label_lookup_.find(id);
    if (search == idx->label_lookup_.end() ||
        idx->isMarkedDeleted(search->second)) {
      return -1;
    }
    hnswlib::tableint internal_id = search->second;
    lock_table.unlock();
    std::memcpy(out, idx->getDataByInternalId(internal_id), nbytes);
    return 0;
  } catch (const std::exception&) {
    return -1;
  }
}

int vl_hnsw_search(VlHnsw* index, const void* query, size_t k,
                   VlFilterFunc filter, void* filter_ctx, float* out_dist,
                   uint64_t* out_label, char** err) {
  auto* idx = reinterpret_cast<hnswlib::HierarchicalNSW<float>*>(index);
  std::unique_ptr<CallbackFilter> functor;
  if (filter != nullptr) {
    functor = std::make_unique<CallbackFilter>(filter, filter_ctx);
  }
  int count = 0;
  try {
    auto result = idx->searchKnnCloserFirst(query, k, functor.get());
    for (const auto& pair : result) {
      out_dist[count] = pair.first;
      out_label[count] = static_cast<uint64_t>(pair.second);
      count++;
    }
  } catch (const std::exception& ex) {
    SetError(err, ex.what());
    return -1;
  }
  return count;
}

int vl_hnsw_save(VlHnsw* index, const char* path, char** err) {
  auto* idx = reinterpret_cast<hnswlib::HierarchicalNSW<float>*>(index);
  try {
    idx->saveIndex(std::string(path));
  } catch (const std::exception& ex) {
    SetError(err, ex.what());
    return 1;
  }
  return 0;
}

size_t vl_hnsw_get_ef(VlHnsw* index) {
  return reinterpret_cast<hnswlib::HierarchicalNSW<float>*>(index)->ef_;
}

void vl_hnsw_set_ef(VlHnsw* index, size_t ef) {
  reinterpret_cast<hnswlib::HierarchicalNSW<float>*>(index)->setEf(ef);
}

size_t vl_hnsw_per_vector_data_size(VlHnsw* index) {
  auto* idx = reinterpret_cast<hnswlib::HierarchicalNSW<float>*>(index);
  return idx->label_offset_ - idx->offsetData_;
}

void vl_free_err(char* err) { std::free(err); }

}  // extern "C"
