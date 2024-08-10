#ifndef VECTORLITE_DISTANCE_DISTANCE_H
#define VECTORLITE_DISTANCE_DISTANCE_H

#include <string_view>

#include "hwy/base.h"

namespace vectorlite {
namespace distance {

// v1 and v2 MUST not be nullptr but can point to the same array.
HWY_DLLEXPORT float InnerProduct(const float* v1, const float* v2, size_t size);

// Detect best available SIMD target to ensure future dynamic dispatch avoids the overhead of CPU detection. 
HWY_DLLEXPORT std::string_view DetectTarget();

} // namespace distance
} // namespace vectorlite

#endif // VECTORLITE_DISTANCE_DISTANCE_H