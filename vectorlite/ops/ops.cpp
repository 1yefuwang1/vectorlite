#include "ops.h"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include "hwy/base.h"
// >>>> for dynamic dispatch only, skip if you want static dispatch

// For dynamic dispatch, specify the name of the current file (unfortunately
// __FILE__ is not reliable) so that foreach_target.h can re-include it.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ops.cpp"
// Generates code for each enabled target by re-including this source file.
#include "hwy/foreach_target.h"  // IWYU pragma: keep

// <<<< end of dynamic dispatch

// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/contrib/algo/transform-inl.h"
#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/highway.h"
#include "hwy/targets.h"

// Optional, can instead add HWY_ATTR to all functions.
HWY_BEFORE_NAMESPACE();

// This namespace name is unique per target, which allows code for multiple
// targets to co-exist in the same translation unit. Required when using dynamic
// dispatch, otherwise optional.
namespace HWY_NAMESPACE {

// Highway ops reside here; ADL does not find templates nor builtins.
namespace hn = hwy::HWY_NAMESPACE;

template <class D, typename T = hn::TFromD<D>>
static float SquaredSumVectorized(const D d, const T* v, size_t num_elements) {
  static_assert(hwy::IsFloat<T>(), "MulAdd requires float type");
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);
  HWY_DASSERT(num_elements >= N && num_elements % N == 0);

  V sum0 = Zero(d);
  V sum1 = Zero(d);
  V sum2 = Zero(d);
  V sum3 = Zero(d);

  size_t i = 0;
  // Main loop: unrolled
  for (; i + 4 * N <= num_elements; /* i += 4 * N */) {  // incr in loop
    const auto a0 = LoadU(d, v + i);
    i += N;
    sum0 = MulAdd(a0, a0, sum0);
    const auto a1 = LoadU(d, v + i);
    i += N;
    sum1 = MulAdd(a1, a1, sum1);
    const auto a2 = LoadU(d, v + i);
    i += N;
    sum2 = MulAdd(a2, a2, sum2);
    const auto a3 = LoadU(d, v + i);
    i += N;
    sum3 = MulAdd(a3, a3, sum3);
  }

  // Up to 3 iterations of whole vectors
  for (; i + N <= num_elements; i += N) {
    const auto a = LoadU(d, v + i);
    sum0 = MulAdd(a, a, sum0);
  }

  // Reduction tree: sum of all accumulators by pairs, then across lanes.
  sum0 = Add(sum0, sum1);
  sum2 = Add(sum2, sum3);
  sum0 = Add(sum0, sum2);

  return hn::ReduceSum(d, sum0);
}

template <class D, HWY_IF_BF16_D(D)>
static float SquaredSumVectorized(const D d, const hwy::bfloat16_t* v,
                                  size_t num_elements) {
  const hn::Repartition<float, D> df32;

  using V = decltype(Zero(df32));
  const size_t N = Lanes(d);

  size_t i = 0;
  // See comment in the hwy::Dot::Compute() overload. Unroll 2x, but we need
  // twice as many sums for ReorderWidenMulAccumulate.
  V sum0 = Zero(df32);
  V sum1 = Zero(df32);
  V sum2 = Zero(df32);
  V sum3 = Zero(df32);

  // Main loop: unrolled
  for (; i + 2 * N <= num_elements; /* i += 2 * N */) {  // incr in loop
    const auto a0 = LoadU(d, v + i);
    i += N;
    sum0 = ReorderWidenMulAccumulate(df32, a0, a0, sum0, sum1);
    const auto a1 = LoadU(d, v + i);
    i += N;
    sum2 = ReorderWidenMulAccumulate(df32, a1, a1, sum2, sum3);
  }

  // Possibly one more iteration of whole vectors
  if (i + N <= num_elements) {
    const auto a0 = LoadU(d, v + i);
    i += N;
    sum0 = ReorderWidenMulAccumulate(df32, a0, a0, sum0, sum1);
  }

  // Reduction tree: sum of all accumulators by pairs, then across lanes.
  sum0 = Add(sum0, sum1);
  sum2 = Add(sum2, sum3);
  sum0 = Add(sum0, sum2);
  return ReduceSum(df32, sum0);
}

template <class D, typename T = hn::TFromD<D>>
static float InnerProductImplVectorized(const D d, const T* v1, const T* v2,
                                        size_t num_elements) {
  const size_t N = hn::Lanes(d);
  HWY_DASSERT(num_elements >= N && num_elements % N == 0);

  constexpr int assumption =
      hn::Dot::kAtLeastOneVector | hn::Dot::kMultipleOfVector;
  if (v1 != v2) {
    return hn::Dot::Compute<assumption>(d, v1, v2, num_elements);
  } else {
    return SquaredSumVectorized(d, v1, num_elements);
  }
}

template <class D, typename T = hn::TFromD<D>>
static float InnerProductImpl(const D d, const T* v1, const T* v2,
                              size_t num_elements) {
  const size_t N = hn::Lanes(d);

  const size_t leftover = num_elements % N;

  float result = 0;

  if (num_elements >= N) {
    result = InnerProductImplVectorized(d, v1, v2, num_elements - leftover);
  }

  if (leftover > 0) {
    // Manually 2x unroll the loop
    float sum0 = 0;
    float sum1 = 0;
    size_t i = num_elements - leftover;
    for (; i + 2 <= num_elements; i += 2) {
      sum0 += hwy::ConvertScalarTo<float>(v1[i]) *
              hwy::ConvertScalarTo<float>(v2[i]);
      sum1 += hwy::ConvertScalarTo<float>(v1[i + 1]) *
              hwy::ConvertScalarTo<float>(v2[i + 1]);
    }

    if (i < num_elements) {
      sum0 += hwy::ConvertScalarTo<float>(v1[i]) *
              hwy::ConvertScalarTo<float>(v2[i]);
    }
    return result + sum0 + sum1;
  } else {
    return result;
  }
}

template <class D, typename T = hn::TFromD<D>>
static float L2DistanceSquaredImplVectorized(const D d,
                                             const T* HWY_RESTRICT v1,
                                             const T* HWY_RESTRICT v2,
                                             size_t num_elements) {
  static_assert(hwy::IsFloat<T>(), "MulAdd requires float type");
  const size_t N = hn::Lanes(d);
  HWY_DASSERT(num_elements >= N && num_elements % N == 0);
  using V = hn::Vec<decltype(d)>;

  V sum0 = Zero(d);
  V sum1 = Zero(d);
  V sum2 = Zero(d);
  V sum3 = Zero(d);

  size_t i = 0;
  // Main loop: unrolled
  for (; i + 4 * N <= num_elements; /* i += 4 * N */) {  // incr in loop
    const auto diff0 = hn::Sub(LoadU(d, v1 + i), LoadU(d, v2 + i));
    i += N;
    sum0 = MulAdd(diff0, diff0, sum0);
    const auto diff1 = hn::Sub(LoadU(d, v1 + i), LoadU(d, v2 + i));
    i += N;
    sum1 = MulAdd(diff1, diff1, sum1);
    const auto diff2 = hn::Sub(LoadU(d, v1 + i), LoadU(d, v2 + i));
    i += N;
    sum2 = MulAdd(diff2, diff2, sum2);
    const auto diff3 = hn::Sub(LoadU(d, v1 + i), LoadU(d, v2 + i));
    i += N;
    sum3 = MulAdd(diff3, diff3, sum3);
  }

  // Up to 3 iterations of whole vectors
  for (; i + N <= num_elements; i += N) {
    const auto diff = hn::Sub(LoadU(d, v1 + i), LoadU(d, v2 + i));
    sum0 = MulAdd(diff, diff, sum0);
  }
  // Reduction tree: sum of all accumulators by pairs, then across lanes.
  sum0 = Add(sum0, sum1);
  sum2 = Add(sum2, sum3);
  sum0 = Add(sum0, sum2);

  return hwy::ConvertScalarTo<float>(hn::ReduceSum(d, sum0));
}

template <class D, typename T = hn::TFromD<D>>
static float L2DistanceSquaredImpl(const D d, const T* HWY_RESTRICT v1,
                                   const T* HWY_RESTRICT v2,
                                   size_t num_elements) {
  const size_t N = hn::Lanes(d);

  const size_t leftover = num_elements % N;
  float result = 0;
  if (num_elements >= N) {
    result =
        L2DistanceSquaredImplVectorized(d, v1, v2, num_elements - leftover);
  }

  if (leftover > 0) {
    // Manually 2x unroll the loop
    float sum0 = 0;
    float sum1 = 0;
    size_t i = num_elements - leftover;
    for (; i + 2 <= num_elements; i += 2) {
      float diff0 = hwy::ConvertScalarTo<float>(v1[i]) -
                    hwy::ConvertScalarTo<float>(v2[i]);
      sum0 += diff0 * diff0;
      float diff1 = hwy::ConvertScalarTo<float>(v1[i + 1]) -
                    hwy::ConvertScalarTo<float>(v2[i + 1]);
      sum1 += diff1 * diff1;
    }

    if (i < num_elements) {
      float diff = hwy::ConvertScalarTo<float>(v1[i]) -
                   hwy::ConvertScalarTo<float>(v2[i]);
      sum0 += diff * diff;
    }

    return result + sum0 + sum1;
  } else {
    return result;
  }
}

// A vectorized implementation following
// https://github.com/nmslib/hnswlib/blob/v0.8.0/python_bindings/bindings.cpp#L241
template <class D, typename T = hn::TFromD<D>>
static void NormalizeImpl(const D d, T* HWY_RESTRICT inout,
                          size_t num_elements) {
  const float squared_sum = InnerProductImpl(d, inout, inout, num_elements);
  const float norm =
      hwy::ConvertScalarTo<float>(1.0f / (sqrtf(squared_sum) + 1e-30f));
  hn::Transform(d, inout, num_elements, [norm](D d, hn::Vec<D> v) HWY_ATTR {
    return hn::Mul(v, hn::Set(d, norm));
  });
}

template <class D, HWY_IF_BF16_D(D)>
static void NormalizeImpl(const D d, hwy::bfloat16_t* HWY_RESTRICT inout,
                          size_t num_elements) {
  const float squared_sum = InnerProductImpl(d, inout, inout, num_elements);
  const float norm =
      hwy::ConvertScalarTo<float>(1.0f / (sqrtf(squared_sum) + 1e-30f));
  hn::Transform(d, inout, num_elements, [norm](D d, hn::Vec<D> v) HWY_ATTR {
    const hn::RepartitionToWide<D> df32;
    const auto norm_vector = hn::Set(df32, norm);
    const auto lower = hn::Mul(hn::PromoteLowerTo(df32, v), norm_vector);
    const auto upper = hn::Mul(hn::PromoteUpperTo(df32, v), norm_vector);
    return hn::OrderedDemote2To(d, lower, upper);
  });
}

template <class HalfFloat, HWY_IF_SPECIAL_FLOAT(HalfFloat)>
static void QuantizeF32ToHalf(const float* HWY_RESTRICT in,
                              HalfFloat* HWY_RESTRICT out, size_t size) {
  static_assert(sizeof(float) / sizeof(HalfFloat) == 2,
                "HalfFloat must be 16-bit");
  const hn::ScalableTag<float> df32;
  // f16 here refers to the 16-bit floating point type, including float16_t and
  // bfloat16_t
  const hn::Repartition<HalfFloat, decltype(df32)> df16;
  const size_t NF = hn::Lanes(df32);
  using VF = hn::Vec<decltype(df32)>;
  using VBF = hn::Vec<decltype(df16)>;
  const hn::Half<decltype(df16)> df16h;
  constexpr bool is_bfloat16 = std::is_same<HalfFloat, hwy::bfloat16_t>::value;

  size_t i = 0;
  if (size >= 2 * NF) {
    for (; i <= size - 2 * NF; i += 2 * NF) {
      const VF v0 = hn::LoadU(df32, in + i);
      const VF v1 = hn::LoadU(df32, in + i + NF);
      if constexpr (is_bfloat16) {
        const VBF bf = hn::OrderedDemote2To(df16, v0, v1);
        hn::StoreU(bf, df16, out + i);
      } else {
        static_assert(std::is_same<HalfFloat, hwy::float16_t>::value,
                      "Unsupported HalfFloat type");
        // todo: use OrderedDemote2To once it's implemented for float16_t
        const VBF bf =
            hn::Combine(df16, hn::DemoteTo(df16h, v1), hn::DemoteTo(df16h, v0));
        hn::StoreU(bf, df16, out + i);
      }
    }
  }
  if (size - i >= NF) {
    const VF v0 = hn::LoadU(df32, in + i);
    const hn::Vec<decltype(df16h)> bfh = hn::DemoteTo(df16h, v0);
    hn::StoreU(bfh, df16h, out + i);
    i += NF;
  }

  if (i != size) {
    const size_t remaining = size - i;
    const VF v0 = hn::LoadN(df32, in + i, remaining);
    const hn::Vec<decltype(df16h)> bfh = hn::DemoteTo(df16h, v0);
    hn::StoreN(bfh, df16h, out + i, remaining);
  }
}

template <class HalfFloat, HWY_IF_SPECIAL_FLOAT(HalfFloat)>
static void HalfFloatToF32(const HalfFloat* HWY_RESTRICT in,
                           float* HWY_RESTRICT out, size_t size) {
  static_assert(sizeof(float) / sizeof(HalfFloat) == 2,
                "HalfFloat must be 16-bit");
  const hn::ScalableTag<float> df32;
  // f16 here refers to the 16-bit floating point type, including float16_t and
  // bfloat16_t
  const hn::Repartition<HalfFloat, decltype(df32)> df16;
  const size_t NF = hn::Lanes(df32);
  using VF = hn::Vec<decltype(df32)>;
  using VBF = hn::Vec<decltype(df16)>;
  const hn::Half<decltype(df16)> df16h;

  size_t i = 0;
  if (size >= NF) {
    for (; i <= size - NF; i += NF) {
      const auto v = hn::LoadU(df16h, in + i);
      hn::StoreU(hn::PromoteTo(df32, v), df32, out + i);
    }
  }

  if (i != size) {
    const size_t remaining = size - i;
    const auto v = hn::LoadN(df16h, in + i, remaining);
    hn::StoreN(hn::PromoteTo(df32, v), df32, out + i, remaining);
  }
}

static void QuantizeF32ToBF16Impl(const float* HWY_RESTRICT in,
                                  hwy::bfloat16_t* HWY_RESTRICT out,
                                  size_t size) {
  QuantizeF32ToHalf(in, out, size);
}

static void QuantizeF32ToF16Impl(const float* HWY_RESTRICT in,
                                 hwy::float16_t* HWY_RESTRICT out,
                                 size_t size) {
  QuantizeF32ToHalf(in, out, size);
}

static float InnerProductImplF32(const float* v1, const float* v2,
                                 size_t num_elements) {
  return InnerProductImpl(hn::ScalableTag<float>(), v1, v2, num_elements);
}

static float L2DistanceSquaredImplF32(const float* v1, const float* v2,
                                      size_t num_elements) {
  return L2DistanceSquaredImpl(hn::ScalableTag<float>(), v1, v2, num_elements);
}

static void NormalizeImplF32(float* HWY_RESTRICT inout, size_t num_elements) {
  return NormalizeImpl(hn::ScalableTag<float>(), inout, num_elements);
}

// static void NormalizeImplF16(hwy::float16_t* HWY_RESTRICT inout, size_t
// num_elements) {
//   return NormalizeImpl(hn::Half<hn::ScalableTag<hwy::float16_t>>(), inout,
//   num_elements);
// }

static void NormalizeImplBF16(hwy::bfloat16_t* HWY_RESTRICT inout,
                              size_t num_elements) {
  return NormalizeImpl(hn::ScalableTag<hwy::bfloat16_t>(), inout, num_elements);
}

static void F16ToF32Impl(const hwy::float16_t* HWY_RESTRICT in,
                         float* HWY_RESTRICT out, size_t num_elements) {
  HalfFloatToF32(in, out, num_elements);
}
static void BF16ToF32Impl(const hwy::bfloat16_t* HWY_RESTRICT in,
                          float* HWY_RESTRICT out, size_t num_elements) {
  HalfFloatToF32(in, out, num_elements);
}

}  // namespace HWY_NAMESPACE

HWY_AFTER_NAMESPACE();

// The table of pointers to the various implementations in HWY_NAMESPACE must
// be compiled only once (foreach_target #includes this file multiple times).
// HWY_ONCE is true for only one of these 'compilation passes'.
#if HWY_ONCE

namespace vectorlite {
namespace ops {

// This macro declares a static array used for dynamic dispatch; it resides in
// the same outer namespace that contains FloorLog2.
HWY_EXPORT(InnerProductImplF32);
HWY_EXPORT(L2DistanceSquaredImplF32);
HWY_EXPORT(QuantizeF32ToF16Impl);
HWY_EXPORT(QuantizeF32ToBF16Impl);
HWY_EXPORT(F16ToF32Impl);
HWY_EXPORT(BF16ToF32Impl);

HWY_EXPORT(NormalizeImplF32);
// HWY_EXPORT(NormalizeImplF16);
HWY_EXPORT(NormalizeImplBF16);

HWY_DLLEXPORT float InnerProduct(const float* v1, const float* v2,
                                 size_t num_elements) {
  return HWY_DYNAMIC_DISPATCH(InnerProductImplF32)(v1, v2, num_elements);
}

HWY_DLLEXPORT float InnerProductDistance(const float* v1, const float* v2,
                                         size_t num_elements) {
  return 1.0f - InnerProduct(v1, v2, num_elements);
}

HWY_DLLEXPORT void Normalize(float* HWY_RESTRICT inout, size_t size) {
  HWY_DYNAMIC_DISPATCH(NormalizeImplF32)(inout, size);
  return;
}

// HWY_DLLEXPORT void Normalize(hwy::float16_t* HWY_RESTRICT inout, size_t size)
// {
//   HWY_DYNAMIC_DISPATCH(NormalizeImplF16)(inout, size);
//   return;
// }

HWY_DLLEXPORT void Normalize(hwy::bfloat16_t* HWY_RESTRICT inout, size_t size) {
  HWY_DYNAMIC_DISPATCH(NormalizeImplBF16)(inout, size);
  return;
}

HWY_DLLEXPORT float L2DistanceSquared(const float* v1, const float* v2,
                                      size_t num_elements) {
  if (HWY_UNLIKELY(v1 == v2)) {
    return 0.0f;
  }

  return HWY_DYNAMIC_DISPATCH(L2DistanceSquaredImplF32)(v1, v2, num_elements);
}

// Implementation follows
// https://github.com/nmslib/hnswlib/blob/v0.8.0/python_bindings/bindings.cpp#L241
// Not sure whether compiler will do auto-vectorization for this function.
HWY_DLLEXPORT void Normalize_Scalar(float* HWY_RESTRICT inout, size_t size) {
  float norm = 0.0f;
  for (int i = 0; i < size; i++) {
    float data = inout[i];
    norm += data * data;
  }
  norm = 1.0f / (sqrtf(norm) + 1e-30f);
  for (int i = 0; i < size; i++) {
    inout[i] = inout[i] * norm;
  }
  return;
}

HWY_DLLEXPORT void Normalize_Scalar(hwy::bfloat16_t* HWY_RESTRICT inout,
                                    size_t size) {
  float norm = 0.0f;
  for (int i = 0; i < size; i++) {
    float data = hwy::F32FromBF16(inout[i]);
    norm += data * data;
  }
  norm = 1.0f / (sqrtf(norm) + 1e-30f);
  for (int i = 0; i < size; i++) {
    inout[i] = hwy::BF16FromF32(hwy::F32FromBF16(inout[i]) * norm);
  }
  return;
}

HWY_DLLEXPORT std::vector<const char*> GetSupportedTargets() {
  std::vector<int64_t> targets = hwy::SupportedAndGeneratedTargets();
  std::vector<const char*> target_names(targets.size());
  std::transform(targets.cbegin(), targets.cend(), target_names.begin(),
                 [](int64_t target) { return hwy::TargetName(target); });
  return target_names;
}

HWY_DLLEXPORT void QuantizeF32ToF16(const float* HWY_RESTRICT in,
                                    hwy::float16_t* HWY_RESTRICT out,
                                    size_t num_elements) {
  HWY_DYNAMIC_DISPATCH(QuantizeF32ToF16Impl)(in, out, num_elements);
}

HWY_DLLEXPORT void QuantizeF32ToBF16(const float* HWY_RESTRICT in,
                                     hwy::bfloat16_t* HWY_RESTRICT out,
                                     size_t num_elements) {
  HWY_DYNAMIC_DISPATCH(QuantizeF32ToBF16Impl)(in, out, num_elements);
}

HWY_DLLEXPORT void F16ToF32(const hwy::float16_t* HWY_RESTRICT in,
                            float* HWY_RESTRICT out, size_t num_elements) {
  HWY_DYNAMIC_DISPATCH(F16ToF32Impl)(in, out, num_elements);
}
HWY_DLLEXPORT void BF16ToF32(const hwy::bfloat16_t* HWY_RESTRICT in,
                             float* HWY_RESTRICT out, size_t num_elements) {
  HWY_DYNAMIC_DISPATCH(BF16ToF32Impl)(in, out, num_elements);
}

}  // namespace ops
}  // namespace vectorlite

#endif  // HWY_ONCE