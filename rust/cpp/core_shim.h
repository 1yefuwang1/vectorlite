// C ABI over the vectorlite numeric core (hnswlib + vectorlite spaces + ops +
// quantization). This is intentionally a thin wrapper: all algorithm/SIMD code
// lives in the existing C++ (`vectorlite/ops`) and hnswlib; the Rust side owns
// the SQLite virtual-table glue and calls into these functions.
#ifndef VECTORLITE_CORE_SHIM_H
#define VECTORLITE_CORE_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Distance types. Must match Rust DistanceType discriminants.
enum VlDistanceType {
  VL_DISTANCE_L2 = 0,
  VL_DISTANCE_IP = 1,
  VL_DISTANCE_COSINE = 2,
};

// Vector element types. Must match Rust VectorType discriminants.
enum VlVectorType {
  VL_VECTOR_F32 = 0,
  VL_VECTOR_BF16 = 1,
  VL_VECTOR_F16 = 2,
};

// Rowid filter kinds for search.
enum VlFilterKind {
  VL_FILTER_NONE = 0,
  VL_FILTER_IN = 1,
  VL_FILTER_EQ = 2,
};

typedef struct VlIndex VlIndex;

// Creates an index. On failure returns NULL and, if `err` is non-NULL, sets
// *err to a malloc'd message (free with vl_free_err).
VlIndex* vl_index_create(size_t dim, int distance_type, int vector_type,
                         size_t max_elements, size_t M, size_t ef_construction,
                         size_t random_seed, int allow_replace_deleted,
                         char** err);

void vl_index_free(VlIndex* index);

size_t vl_index_dim(const VlIndex* index);

// Bytes stored per vector (dim * sizeof(element_type)).
size_t vl_index_data_size(const VlIndex* index);

// Adds/replaces a vector (f32 input of length `len`, must equal dim). The shim
// quantizes and/or normalizes internally according to the index's space.
// Returns 0 on success, non-zero on failure (sets *err).
int vl_index_add(VlIndex* index, const float* data, size_t len, uint64_t rowid,
                 char** err);

// Marks a rowid deleted. Returns 0 on success, non-zero on failure (sets *err).
int vl_index_mark_delete(VlIndex* index, uint64_t rowid, char** err);

// Returns 1 if rowid is present and not marked deleted, else 0.
int vl_index_contains(const VlIndex* index, uint64_t rowid);

// Reads a vector back, dequantized to f32 into `out` (must hold dim floats).
// Returns 0 on success, -1 if rowid not found.
int vl_index_get_vector(const VlIndex* index, uint64_t rowid, float* out);

// k-NN search. `query` is f32 of length `len` (must equal dim). `ef_override`
// of 0 means "use the index default". Results (closer-first) are written to
// out_distances/out_rowids, each of capacity `k`. Returns the number of results
// written (>=0), or -1 on error (sets *err).
int vl_index_search(VlIndex* index, const float* query, size_t len, size_t k,
                    size_t ef_override, int filter_kind,
                    const uint64_t* filter_rowids, size_t filter_count,
                    float* out_distances, uint64_t* out_rowids, char** err);

// Serializes the index to `path`. Returns 0 on success, non-zero on failure.
int vl_index_save(VlIndex* index, const char* path, char** err);

// Loads an index from `path`, replacing the in-memory index on success. The
// table's configured max_elements and allow_replace_deleted are preserved. The
// per-vector data size in the file must match this index's data size, otherwise
// the load is rejected and the current index is left unchanged. Returns 0 on
// success, non-zero on failure (sets *err).
int vl_index_load(VlIndex* index, const char* path, char** err);

// Computes the distance between two f32 vectors of length `dim`. Cosine
// normalizes internally. Returns 0 on success and writes *out; non-zero on
// failure.
int vl_distance(const float* a, const float* b, size_t dim, int distance_type,
                float* out);

// Returns the best SIMD target chosen by Highway at runtime (static string).
const char* vl_best_target(void);

// Frees an error string returned through an `err` out-parameter.
void vl_free_err(char* err);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VECTORLITE_CORE_SHIM_H
