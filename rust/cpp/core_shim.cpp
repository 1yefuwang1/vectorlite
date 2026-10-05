// Implementation of the pure hnswlib+ops C ABI declared in core_shim.h.
// Contains only generic glue: forwarders to `ops`, an hnswlib SpaceInterface
// adapter around a caller-supplied distance function, an hnswlib filter adapter
// around a caller-supplied predicate, and thin wrappers over HierarchicalNSW.
// No virtual-table policy lives here.
#include "core_shim.h"

#include <hnswlib/hnswlib.h>
#include <hwy/base.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "ops/ops.h"

namespace {

// Error reporting must remain nonthrowing even while handling bad_alloc.
void SetError(char** err, const char* msg) noexcept {
  if (err == nullptr) return;
  const size_t size = std::strlen(msg) + 1;
  auto* out = static_cast<char*>(std::malloc(size));
  if (out != nullptr) std::memcpy(out, msg, size);
  *err = out;
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

// Validate the raw native layout before hnswlib reads offsets and graph links.
// The file is a private temporary snapshot prepared by Rust, so validation and
// loading observe the same bytes. Never hand unchecked serialized offsets to
// hnswlib's pointer-based loader.
size_t CheckedAdd(size_t a, size_t b) {
  if (b > std::numeric_limits<size_t>::max() - a)
    throw std::runtime_error("index layout addition overflow");
  return a + b;
}

size_t CheckedMultiply(size_t a, size_t b) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
    throw std::runtime_error("index layout multiplication overflow");
  return a * b;
}

void ValidateCapacity(size_t capacity, size_t stride) {
  if (capacity == 0 ||
      capacity > std::numeric_limits<hnswlib::tableint>::max() ||
      CheckedMultiply(capacity, stride) >
          static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
    throw std::runtime_error(
        "HNSW capacity or allocation size is out of range");
}

template <typename T>
T ReadNative(std::ifstream& input) {
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(value));
  return value;
}

void Seek(std::ifstream& input, size_t offset) {
  if (offset > static_cast<size_t>(std::numeric_limits<std::streamoff>::max()))
    throw std::runtime_error("index file offset is out of range");
  input.seekg(static_cast<std::streamoff>(offset));
}

void ValidateLinks(std::ifstream& input, size_t maximum, size_t count,
                   size_t level, const std::vector<size_t>& levels) {
  const auto header = ReadNative<hnswlib::linklistsizeint>(input);
  uint16_t links;
  std::memcpy(&links, &header, sizeof(links));
  if (links > maximum) throw std::runtime_error("invalid index neighbor count");
  for (size_t j = 0; j < links; ++j) {
    const auto neighbor = ReadNative<hnswlib::tableint>(input);
    if (neighbor >= count || (level != 0 && levels[neighbor] < level))
      throw std::runtime_error("invalid index neighbor reference");
  }
}

size_t ValidateFile(const char* path, size_t data_size,
                    size_t requested_capacity) {
  std::ifstream input;
  input.exceptions(std::ios::failbit | std::ios::badbit);
  input.open(path, std::ios::binary);
  input.seekg(0, std::ios::end);
  const auto length = input.tellg();
  if (length < 0 ||
      static_cast<uint64_t>(length) > std::numeric_limits<size_t>::max())
    throw std::runtime_error("invalid index file length");
  const auto file_size = static_cast<size_t>(length);
  Seek(input, 0);
  const auto offset_level0 = ReadNative<size_t>(input);
  const auto saved_capacity = ReadNative<size_t>(input);
  const auto count = ReadNative<size_t>(input);
  const auto stride = ReadNative<size_t>(input);
  const auto label_offset = ReadNative<size_t>(input);
  const auto data_offset = ReadNative<size_t>(input);
  const auto max_level = ReadNative<int>(input);
  const auto entry = ReadNative<hnswlib::tableint>(input);
  const auto max_m = ReadNative<size_t>(input);
  const auto max_m0 = ReadNative<size_t>(input);
  const auto m = ReadNative<size_t>(input);
  const auto multiplier = ReadNative<double>(input);
  const auto ef_construction = ReadNative<size_t>(input);
  if (m < 2 || m > 10000 || max_m != m || max_m0 != m * 2 ||
      !std::isfinite(multiplier) || multiplier <= 0 || ef_construction < m ||
      std::abs(multiplier - 1.0 / std::log(static_cast<double>(m))) >
          4 * std::numeric_limits<double>::epsilon())
    throw std::runtime_error("invalid HNSW parameters in index");
  const auto links0 =
      CheckedAdd(CheckedMultiply(max_m0, sizeof(hnswlib::tableint)),
                 sizeof(hnswlib::linklistsizeint));
  const auto expected_label = CheckedAdd(links0, data_size);
  if (offset_level0 != 0 || data_offset != links0 ||
      label_offset != expected_label ||
      stride != CheckedAdd(expected_label, sizeof(hnswlib::labeltype)))
    throw std::runtime_error("index layout does not match the vector space");
  ValidateCapacity(saved_capacity, stride);
  if (count > saved_capacity)
    throw std::runtime_error("index count exceeds capacity");
  // The saved capacity is not proportional to file size and is controlled by
  // the file. Allocate only what the table requested or the loaded rows need.
  const size_t load_capacity =
      requested_capacity < count ? count : requested_capacity;
  ValidateCapacity(load_capacity, stride);
  if ((count == 0 &&
       (max_level != -1 ||
        entry != std::numeric_limits<hnswlib::tableint>::max())) ||
      (count != 0 && (max_level < 0 || entry >= count)))
    throw std::runtime_error("invalid index entry point");
  const auto header_size = static_cast<size_t>(input.tellg());
  const auto data_end = CheckedAdd(header_size, CheckedMultiply(count, stride));
  if (CheckedAdd(data_end, CheckedMultiply(count, sizeof(uint32_t))) >
      file_size)
    throw std::runtime_error("truncated index data");
  const auto level_stride =
      CheckedAdd(CheckedMultiply(m, sizeof(hnswlib::tableint)),
                 sizeof(hnswlib::linklistsizeint));
  std::vector<size_t> levels(count);
  std::vector<size_t> offsets(count);
  size_t offset = data_end;
  for (size_t i = 0; i < count; ++i) {
    Seek(input, offset);
    const auto size = ReadNative<uint32_t>(input);
    offset = CheckedAdd(offset, sizeof(uint32_t));
    offsets[i] = offset;
    if (size % level_stride != 0)
      throw std::runtime_error("invalid index level size");
    levels[i] = size / level_stride;
    if (levels[i] > static_cast<size_t>(max_level))
      throw std::runtime_error("invalid index level count");
    offset = CheckedAdd(offset, size);
    if (offset > file_size) throw std::runtime_error("truncated index links");
  }
  if (offset != file_size ||
      (count != 0 && levels[entry] != static_cast<size_t>(max_level)))
    throw std::runtime_error("inconsistent index payload size or entry level");
  for (size_t i = 0; i < count; ++i) {
    Seek(input, header_size + i * stride);
    ValidateLinks(input, max_m0, count, 0, levels);
    for (size_t level = 1; level <= levels[i]; ++level) {
      Seek(input, offsets[i] + (level - 1) * level_stride);
      ValidateLinks(input, m, count, level, levels);
    }
  }
  return load_capacity;
}

void SaveChecked(hnswlib::HierarchicalNSW<float>& index, const char* path) {
  std::ofstream output;
  output.exceptions(std::ios::failbit | std::ios::badbit);
  output.open(path, std::ios::binary | std::ios::trunc);
  using hnswlib::writeBinaryPOD;
  const size_t count = index.getCurrentElementCount();
  writeBinaryPOD(output, index.offsetLevel0_);
  writeBinaryPOD(output, index.max_elements_);
  writeBinaryPOD(output, count);
  writeBinaryPOD(output, index.size_data_per_element_);
  writeBinaryPOD(output, index.label_offset_);
  writeBinaryPOD(output, index.offsetData_);
  writeBinaryPOD(output, index.maxlevel_);
  writeBinaryPOD(output, index.enterpoint_node_);
  writeBinaryPOD(output, index.maxM_);
  writeBinaryPOD(output, index.maxM0_);
  writeBinaryPOD(output, index.M_);
  writeBinaryPOD(output, index.mult_);
  writeBinaryPOD(output, index.ef_construction_);
  output.write(index.data_level0_memory_,
               CheckedMultiply(count, index.size_data_per_element_));
  for (size_t i = 0; i < count; ++i) {
    const size_t bytes = index.element_levels_[i] > 0
                             ? CheckedMultiply(index.size_links_per_element_,
                                               index.element_levels_[i])
                             : 0;
    if (bytes > std::numeric_limits<uint32_t>::max())
      throw std::runtime_error("index level payload is too large");
    const auto size = static_cast<uint32_t>(bytes);
    writeBinaryPOD(output, size);
    if (size != 0) output.write(index.linkLists_[i], size);
  }
  output.flush();
  output.close();
}

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
  try {
    return reinterpret_cast<VlSpace*>(
        new SpaceAdapter(distfunc, dim, data_size));
  } catch (...) {
    // Never let a C++ exception (e.g. std::bad_alloc) unwind across the C ABI
    // into Rust; the caller treats NULL as an allocation failure.
    return nullptr;
  }
}

void vl_hnsw_space_free(VlSpace* space) {
  delete reinterpret_cast<SpaceAdapter*>(space);
}

VlHnsw* vl_hnsw_create(VlSpace* space, size_t max_elements, size_t M,
                       size_t ef_construction, size_t random_seed,
                       int allow_replace_deleted, char** err) {
  auto* s = reinterpret_cast<SpaceAdapter*>(space);
  try {
    if (M < 2 || M > 10000 || ef_construction == 0)
      throw std::runtime_error("invalid HNSW construction parameters");
    const auto stride = CheckedAdd(
        CheckedAdd(CheckedAdd(CheckedMultiply(M, 8), 4), s->get_data_size()),
        sizeof(hnswlib::labeltype));
    ValidateCapacity(max_elements, stride);
    auto* index = new hnswlib::HierarchicalNSW<float>(
        s, max_elements, M, ef_construction, random_seed,
        allow_replace_deleted != 0);
    return reinterpret_cast<VlHnsw*>(index);
  } catch (const std::exception& ex) {
    SetError(err, ex.what());
    return nullptr;
  } catch (...) {
    SetError(err, "unknown native construction failure");
    return nullptr;
  }
}

VlHnsw* vl_hnsw_load(VlSpace* space, const char* path, size_t max_elements,
                     int allow_replace_deleted, char** err) {
  auto* s = reinterpret_cast<SpaceAdapter*>(space);
  try {
    const size_t load_capacity =
        ValidateFile(path, s->get_data_size(), max_elements);
    auto* index = new hnswlib::HierarchicalNSW<float>(
        s, std::string(path), /*nmslib=*/false, load_capacity,
        allow_replace_deleted != 0);
    return reinterpret_cast<VlHnsw*>(index);
  } catch (const std::exception& ex) {
    SetError(err, ex.what());
    return nullptr;
  } catch (...) {
    SetError(err, "unknown native load failure");
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
  } catch (...) {
    SetError(err, "unknown native operation failure");
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
  } catch (...) {
    SetError(err, "unknown native operation failure");
    return 1;
  }
  return 0;
}

int vl_hnsw_contains(VlHnsw* index, uint64_t label) {
  auto* idx = reinterpret_cast<hnswlib::HierarchicalNSW<float>*>(index);
  auto id = static_cast<hnswlib::labeltype>(label);
  try {
    std::unique_lock<std::mutex> lock_label(idx->getLabelOpMutex(id));
    std::unique_lock<std::mutex> lock_table(idx->label_lookup_lock);
    auto search = idx->label_lookup_.find(id);
    if (search == idx->label_lookup_.end() ||
        idx->isMarkedDeleted(search->second)) {
      return 0;
    }
    return 1;
  } catch (...) {
    // A lock/lookup failure must not unwind across the C ABI into Rust.
    return 0;
  }
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
    if (nbytes != idx->data_size_) return -1;
    std::memcpy(out, idx->getDataByInternalId(internal_id), nbytes);
    return 0;
  } catch (...) {
    return -1;
  }
}

int vl_hnsw_search(VlHnsw* index, const void* query, size_t k,
                   VlFilterFunc filter, void* filter_ctx, VlSearchResult* out,
                   size_t* count, char** err) {
  auto* idx = reinterpret_cast<hnswlib::HierarchicalNSW<float>*>(index);
  *count = 0;
  try {
    CallbackFilter functor(filter, filter_ctx);
    auto result =
        idx->searchKnn(query, k, filter == nullptr ? nullptr : &functor);
    size_t remaining = result.size();
    if (remaining > k)
      throw std::runtime_error("native search exceeded result capacity");
    *count = remaining;
    while (!result.empty()) {
      const auto pair = result.top();
      out[--remaining] =
          VlSearchResult{pair.first, static_cast<uint64_t>(pair.second)};
      result.pop();
    }
  } catch (const std::exception& ex) {
    SetError(err, ex.what());
    return -1;
  } catch (...) {
    SetError(err, "unknown native search failure");
    return -1;
  }
  return 0;
}

int vl_hnsw_save(VlHnsw* index, const char* path, char** err) {
  auto* idx = reinterpret_cast<hnswlib::HierarchicalNSW<float>*>(index);
  try {
    SaveChecked(*idx, path);
  } catch (const std::exception& ex) {
    SetError(err, ex.what());
    return 1;
  } catch (...) {
    SetError(err, "unknown native operation failure");
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

size_t vl_hnsw_current_count(VlHnsw* index) {
  return reinterpret_cast<hnswlib::HierarchicalNSW<float>*>(index)
      ->getCurrentElementCount();
}

void vl_free_err(char* err) { std::free(err); }

}  // extern "C"
