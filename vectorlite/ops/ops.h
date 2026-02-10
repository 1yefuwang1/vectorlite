#pragma once

#include <vector>

#include "hwy/base.h"

// This file implements vector operations using google's Highway SIMD library.
// Based on the benchmark on my PC(i5-12600KF with AVX2 support),
// InnerProductDistance is 1.5x-3x faster than HNSWLIB's SIMD implementation
// when dealing with vectors with 256 elements or more. The performance gain is
// mainly due to Highway can leverage fused Multiply-Add instructions of AVX2
// while HNSWLIB can't(HNSWLIB uses Multiply-Add for AVX512 though). Due to
// using dynamic dispatch, the performance gain is not as good when dealing with
// vectors with less than 256 elements. Because the overhead of dynamic dispatch
// is not negligible.
namespace vectorlite {
namespace ops {

// v1 and v2 MUST not be nullptr but can point to the same array.
HWY_DLLEXPORT float InnerProduct(const float* v1, const float* v2,
                                 size_t num_elements);
HWY_DLLEXPORT float InnerProduct(const hwy::bfloat16_t* v1,
                                 const hwy::bfloat16_t* v2,
                                 size_t num_elements);
HWY_DLLEXPORT float InnerProduct(const hwy::float16_t* v1,
                                 const hwy::float16_t* v2,
                                 size_t num_elements);
HWY_DLLEXPORT float InnerProductDistance(const float* v1, const float* v2,
                                         size_t num_elements);
HWY_DLLEXPORT float InnerProductDistance(const hwy::bfloat16_t* v1,
                                         const hwy::bfloat16_t* v2,
                                         size_t num_elements);
HWY_DLLEXPORT float InnerProductDistance(const hwy::float16_t* v1,
                                         const hwy::float16_t* v2,
                                         size_t num_elements);

// v1 and v2 MUST not be nullptr but can point to the same array.
HWY_DLLEXPORT float L2DistanceSquared(const float* v1, const float* v2,
                                      size_t num_elements);

// v1 and v2 MUST not be nullptr but can point to the same array.
HWY_DLLEXPORT float L2DistanceSquared(const hwy::bfloat16_t* v1,
                                      const hwy::bfloat16_t* v2,
                                      size_t num_elements);

// v1 and v2 MUST not be nullptr but can point to the same array.
HWY_DLLEXPORT float L2DistanceSquared(const hwy::float16_t* v1,
                                      const hwy::float16_t* v2,
                                      size_t num_elements);

// v1 and v2 MUST not be nullptr and MUST not point to the same array.
HWY_DLLEXPORT float L2DistanceSquared(const float* HWY_RESTRICT v1,
                                      const hwy::bfloat16_t* HWY_RESTRICT v2,
                                      size_t num_elements);

// Nornalize the input vector in place.
HWY_DLLEXPORT void Normalize(float* HWY_RESTRICT inout, size_t num_elements);
HWY_DLLEXPORT void Normalize(hwy::float16_t* HWY_RESTRICT inout,
                             size_t num_elements);
HWY_DLLEXPORT void Normalize(hwy::bfloat16_t* HWY_RESTRICT inout,
                             size_t num_elements);

// Normalize the input vector in place. Implemented using non-SIMD code for
// testing and benchmarking purposes.
HWY_DLLEXPORT void Normalize_Scalar(float* HWY_RESTRICT inout,
                                    size_t num_elements);

// Normalize the input vector in place. Implemented using non-SIMD code for
// testing and benchmarking purposes.
HWY_DLLEXPORT void Normalize_Scalar(hwy::bfloat16_t* HWY_RESTRICT inout,
                                    size_t num_elements);

// Normalize the input vector in place. Implemented using non-SIMD code for
// testing and benchmarking purposes.
HWY_DLLEXPORT void Normalize_Scalar(hwy::float16_t* HWY_RESTRICT inout,
                                    size_t num_elements);

// Get supported SIMD target name strings.
HWY_DLLEXPORT std::vector<const char*> GetSuppportedTargets();

// in and out should not be nullptr and points to valid memory of required size.
HWY_DLLEXPORT void QuantizeF32ToF16(const float* HWY_RESTRICT in,
                                    hwy::float16_t* HWY_RESTRICT out,
                                    size_t num_elements);
HWY_DLLEXPORT void QuantizeF32ToBF16(const float* HWY_RESTRICT in,
                                     hwy::bfloat16_t* HWY_RESTRICT out,
                                     size_t num_elements);

// Convert fp16/bf16 to fp32, useful for json serde
HWY_DLLEXPORT void F16ToF32(const hwy::float16_t* HWY_RESTRICT in,
                            float* HWY_RESTRICT out, size_t num_elements);
HWY_DLLEXPORT void BF16ToF32(const hwy::bfloat16_t* HWY_RESTRICT in,
                             float* HWY_RESTRICT out, size_t num_elements);

}  // namespace ops
}  // namespace vectorlite
