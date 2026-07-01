// Pure C ABI over ONLY two things: hnswlib and vectorlite's SIMD `ops`.
//
// This file deliberately contains no virtual-table / business logic. It exposes
// generic hnswlib primitives (an index whose distance function and rowid filter
// are supplied by the caller as C callbacks) and thin forwarders to `ops`. All
// policy — which distance/space to use, quantization, normalization, the filter
// predicate, the load data-size check, save/load orchestration — lives in Rust.
#ifndef VECTORLITE_CORE_SHIM_H
#define VECTORLITE_CORE_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ------------------------------------------------------------------ ops FFI --
// Thin forwarders to vectorlite::ops. bf16/f16 buffers are passed as uint16_t*
// (the ops functions take hwy::bfloat16_t*/float16_t*, which are 2-byte structs
// wrapping a uint16_t, so the pointer reinterpret is layout-compatible).

float vl_ops_l2_sq_f32(const float* a, const float* b, size_t n);
float vl_ops_l2_sq_bf16(const uint16_t* a, const uint16_t* b, size_t n);
float vl_ops_l2_sq_f16(const uint16_t* a, const uint16_t* b, size_t n);
float vl_ops_ip_dist_f32(const float* a, const float* b, size_t n);
float vl_ops_ip_dist_bf16(const uint16_t* a, const uint16_t* b, size_t n);
float vl_ops_ip_dist_f16(const uint16_t* a, const uint16_t* b, size_t n);

void vl_ops_normalize_f32(float* inout, size_t n);
void vl_ops_normalize_bf16(uint16_t* inout, size_t n);
void vl_ops_normalize_f16(uint16_t* inout, size_t n);

void vl_ops_quantize_f32_to_bf16(const float* in, uint16_t* out, size_t n);
void vl_ops_quantize_f32_to_f16(const float* in, uint16_t* out, size_t n);
void vl_ops_bf16_to_f32(const uint16_t* in, float* out, size_t n);
void vl_ops_f16_to_f32(const uint16_t* in, float* out, size_t n);

const char* vl_ops_best_target(void);

// -------------------------------------------------------------- hnswlib FFI --

// Distance function, matching hnswlib::DISTFUNC<float>: (a, b, dist_func_param).
// The third argument is the pointer returned by the space's
// get_dist_func_param(); this shim makes it point at the dimension (size_t).
typedef float (*VlDistFunc)(const void*, const void*, const void*);

// Rowid filter predicate. Returns non-zero to keep the candidate `label`.
typedef int (*VlFilterFunc)(void* ctx, uint64_t label);

typedef struct VlSpace VlSpace;
typedef struct VlHnsw VlHnsw;

// Wraps a caller-supplied distance function into an hnswlib SpaceInterface. The
// space must outlive any index built from it (hnswlib caches its param pointer).
VlSpace* vl_hnsw_space_create(VlDistFunc distfunc, size_t dim, size_t data_size);
void vl_hnsw_space_free(VlSpace* space);

// Creates an empty index over `space`. Returns NULL on failure (sets *err).
VlHnsw* vl_hnsw_create(VlSpace* space, size_t max_elements, size_t M,
                       size_t ef_construction, size_t random_seed,
                       int allow_replace_deleted, char** err);

// Loads an index from `path` using `space`. Returns NULL on failure (sets *err).
VlHnsw* vl_hnsw_load(VlSpace* space, const char* path, size_t max_elements,
                     int allow_replace_deleted, char** err);

void vl_hnsw_free(VlHnsw* index);

// Adds/replaces a point whose stored bytes are `data` (already in the index's
// element type/normalization, prepared by the caller). Returns 0 on success.
int vl_hnsw_add_point(VlHnsw* index, const void* data, uint64_t label,
                      int replace_deleted, char** err);

int vl_hnsw_mark_delete(VlHnsw* index, uint64_t label, char** err);

// Returns 1 if `label` is present and not marked deleted, else 0.
int vl_hnsw_contains(VlHnsw* index, uint64_t label);

// Copies the stored bytes for `label` into `out` (exactly `nbytes`). Returns 0
// on success, -1 if the label is absent. The caller dequantizes as needed.
int vl_hnsw_get_data(VlHnsw* index, uint64_t label, void* out, size_t nbytes);

// k-NN search returning up to `k` results closer-first. `filter` may be NULL.
// Uses the index's current ef (the caller sets/restores ef via vl_hnsw_*_ef).
// Returns the number of results written, or -1 on error (sets *err).
int vl_hnsw_search(VlHnsw* index, const void* query, size_t k,
                   VlFilterFunc filter, void* filter_ctx, float* out_dist,
                   uint64_t* out_label, char** err);

int vl_hnsw_save(VlHnsw* index, const char* path, char** err);

size_t vl_hnsw_get_ef(VlHnsw* index);
void vl_hnsw_set_ef(VlHnsw* index, size_t ef);

// Per-vector data size recorded in the index's memory layout
// (label_offset_ - offsetData_). Used by the caller to detect a load mismatch.
size_t vl_hnsw_per_vector_data_size(VlHnsw* index);

void vl_free_err(char* err);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VECTORLITE_CORE_SHIM_H
