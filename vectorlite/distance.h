#pragma once

#include "hnswlib/hnswlib.h"
#include "hwy/base.h"
#include "macros.h"
#include "ops/ops.h"

// This file implements hnswlib::SpaceInterface<float> using vectorlite
// implemented SIMD distance functions, which uses google's Highway SIMD
// library. Vectorlite's implementation is 1.5x-3x faster than HNSWLIB's on my
// PC(i5-12600KF with AVX2 support)
namespace vectorlite {

template <class T, VECTORLITE_IF_FLOAT_SUPPORTED(T)>
class GenericInnerProductSpace : public hnswlib::SpaceInterface<float> {
 public:
  explicit GenericInnerProductSpace(size_t dim)
      : dim_(dim), func_(GenericInnerProductSpace::InnerProductDistanceFunc) {}

  size_t get_data_size() override { return dim_ * sizeof(T); }

  void* get_dist_func_param() override { return &dim_; }

  hnswlib::DISTFUNC<float> get_dist_func() override { return func_; }

 private:
  size_t dim_;
  hnswlib::DISTFUNC<float> func_;

  static float InnerProductDistanceFunc(const void* v1, const void* v2,
                                        const void* dim) {
    return ops::InnerProductDistance(static_cast<const T*>(v1),
                                     static_cast<const T*>(v2),
                                     *reinterpret_cast<const size_t*>(dim));
  }
};

using InnerProductSpace = GenericInnerProductSpace<float>;
using InnerProductSpaceBF16 = GenericInnerProductSpace<hwy::bfloat16_t>;
using InnerProductSpaceF16 = GenericInnerProductSpace<hwy::float16_t>;

template <class T, VECTORLITE_IF_FLOAT_SUPPORTED(T)>
class GenericL2Space : public hnswlib::SpaceInterface<float> {
 public:
  explicit GenericL2Space(size_t dim)
      : dim_(dim), func_(GenericL2Space::L2DistanceSquaredFunc) {}

  size_t get_data_size() override { return dim_ * sizeof(T); }

  void* get_dist_func_param() override { return &dim_; }

  hnswlib::DISTFUNC<float> get_dist_func() override { return func_; }

 private:
  size_t dim_;
  hnswlib::DISTFUNC<float> func_;

  static float L2DistanceSquaredFunc(const void* v1, const void* v2,
                                     const void* dim) {
    return ops::L2DistanceSquared(static_cast<const T*>(v1),
                                  static_cast<const T*>(v2),
                                  *reinterpret_cast<const size_t*>(dim));
  }
};

using L2Space = GenericL2Space<float>;
using L2SpaceBF16 = GenericL2Space<hwy::bfloat16_t>;
using L2SpaceF16 = GenericL2Space<hwy::float16_t>;

}  // namespace vectorlite