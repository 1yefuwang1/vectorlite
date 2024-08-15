#pragma once

#include "hnswlib/hnswlib.h"
#include "ops/ops.h"

// This file implements hnswlib::SpaceInterface<float> using vectorlite
// implemented SIMD distance functions, which uses google's Highway SIMD
// library. Vectorlite's implementation is 1.5x-3x faster than HNSWLIB's on my
// PC(i5-12600KF with AVX2 support)
namespace vectorlite {

class InnerProductSpace : public hnswlib::SpaceInterface<float> {
 public:
  explicit InnerProductSpace(size_t dim)
      : dim_(dim), func_(InnerProductSpace::InnerProductDistanceFunc) {}

  size_t get_data_size() override { return dim_ * sizeof(float); }

  void* get_dist_func_param() override { return &dim_; }

  hnswlib::DISTFUNC<float> get_dist_func() override { return func_; }

 private:
  size_t dim_;
  hnswlib::DISTFUNC<float> func_;

  static float InnerProductDistanceFunc(const void* v1, const void* v2,
                                        const void* dim) {
    return ops::InnerProductDistance(static_cast<const float*>(v1),
                                     static_cast<const float*>(v2),
                                     *reinterpret_cast<const size_t*>(dim));
  }
};

class L2Space : public hnswlib::SpaceInterface<float> {
 public:
  explicit L2Space(size_t dim)
      : dim_(dim), func_(L2Space::L2DistanceSquaredFunc) {}

  size_t get_data_size() override { return dim_ * sizeof(float); }

  void* get_dist_func_param() override { return &dim_; }

  hnswlib::DISTFUNC<float> get_dist_func() override { return func_; }

 private:
  size_t dim_;
  hnswlib::DISTFUNC<float> func_;

  static float L2DistanceSquaredFunc(const void* v1, const void* v2,
                                     const void* dim) {
    return ops::L2DistanceSquared(static_cast<const float*>(v1),
                                  static_cast<const float*>(v2),
                                  *reinterpret_cast<const size_t*>(dim));
  }
};

}  // namespace vectorlite