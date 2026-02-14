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
// VECTORLITE_CLANGD is defined via .clangd config. foreach_target.h re-includes
// the main file, which breaks clangd's preamble builder.
#ifdef VECTORLITE_CLANGD
#include "hwy/detect_targets.h"
#undef HWY_ONCE
#define HWY_ONCE 1
#undef HWY_TARGET
#define HWY_TARGET HWY_STATIC_TARGET
// Force HWY_IDE so highway.h uses simplified single-target HWY_EXPORT/DISPATCH.
#undef HWY_IDE
#define HWY_IDE 1
#else
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#endif

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

  V sum0 = hn::Zero(d);
  V sum1 = hn::Zero(d);
  V sum2 = hn::Zero(d);
  V sum3 = hn::Zero(d);

  size_t i = 0;
  // Main loop: unrolled
  for (; i + 4 * N <= num_elements; /* i += 4 * N */) {  // incr in loop
    const auto a0 = hn::LoadU(d, v + i);
    i += N;
    sum0 = hn::MulAdd(a0, a0, sum0);
    const auto a1 = hn::LoadU(d, v + i);
    i += N;
    sum1 = hn::MulAdd(a1, a1, sum1);
    const auto a2 = hn::LoadU(d, v + i);
    i += N;
    sum2 = hn::MulAdd(a2, a2, sum2);
    const auto a3 = hn::LoadU(d, v + i);
    i += N;
    sum3 = hn::MulAdd(a3, a3, sum3);
  }

  // Up to 3 iterations of whole vectors
  for (; i + N <= num_elements; i += N) {
    const auto a = hn::LoadU(d, v + i);
    sum0 = hn::MulAdd(a, a, sum0);
  }

  // Reduction tree: sum of all accumulators by pairs, then across lanes.
  sum0 = hn::Add(sum0, sum1);
  sum2 = hn::Add(sum2, sum3);
  sum0 = hn::Add(sum0, sum2);

  return hn::ReduceSum(d, sum0);
}

template <class D, HWY_IF_BF16_D(D)>
static float SquaredSumVectorized(const D d, const hwy::bfloat16_t* v,
                                  size_t num_elements) {
  const hn::Repartition<float, D> df32;

  using V = decltype(hn::Zero(df32));
  const size_t N = hn::Lanes(d);

  size_t i = 0;
  // See comment in the hwy::Dot::Compute() overload. Unroll 2x, but we need
  // twice as many sums for ReorderWidenMulAccumulate.
  V sum0 = hn::Zero(df32);
  V sum1 = hn::Zero(df32);
  V sum2 = hn::Zero(df32);
  V sum3 = hn::Zero(df32);

  // Main loop: unrolled
  for (; i + 2 * N <= num_elements; /* i += 2 * N */) {  // incr in loop
    const auto a0 = hn::LoadU(d, v + i);
    i += N;
    sum0 = hn::ReorderWidenMulAccumulate(df32, a0, a0, sum0, sum1);
    const auto a1 = hn::LoadU(d, v + i);
    i += N;
    sum2 = hn::ReorderWidenMulAccumulate(df32, a1, a1, sum2, sum3);
  }

  // Possibly one more iteration of whole vectors
  if (i + N <= num_elements) {
    const auto a0 = hn::LoadU(d, v + i);
    i += N;
    sum0 = hn::ReorderWidenMulAccumulate(df32, a0, a0, sum0, sum1);
  }

  // Reduction tree: sum of all accumulators by pairs, then across lanes.
  sum0 = hn::Add(sum0, sum1);
  sum2 = hn::Add(sum2, sum3);
  sum0 = hn::Add(sum0, sum2);
  return hn::ReduceSum(df32, sum0);
}

// When float16 is not natively supported, we need to widen to f32 for
// arithmetic. When HWY_HAVE_FLOAT16 is true (e.g. Apple Silicon M4), the
// generic MulAdd-based SquaredSumVectorized above handles float16_t directly.
#if !HWY_HAVE_FLOAT16
template <class D, HWY_IF_F16_D(D)>
static float SquaredSumVectorized(const D d, const hwy::float16_t* v,
                                  size_t num_elements) {
  const hn::Repartition<float, D> df32;

  using V = decltype(hn::Zero(df32));
  const size_t N = hn::Lanes(d);

  size_t i = 0;
  V sum0 = hn::Zero(df32);
  V sum1 = hn::Zero(df32);
  V sum2 = hn::Zero(df32);
  V sum3 = hn::Zero(df32);

  // Main loop: unrolled
  for (; i + 2 * N <= num_elements; /* i += 2 * N */) {  // incr in loop
    const auto a0 = hn::LoadU(d, v + i);
    i += N;
    sum0 = hn::ReorderWidenMulAccumulate(df32, a0, a0, sum0, sum1);
    const auto a1 = hn::LoadU(d, v + i);
    i += N;
    sum2 = hn::ReorderWidenMulAccumulate(df32, a1, a1, sum2, sum3);
  }

  // Possibly one more iteration of whole vectors
  if (i + N <= num_elements) {
    const auto a0 = hn::LoadU(d, v + i);
    i += N;
    sum0 = hn::ReorderWidenMulAccumulate(df32, a0, a0, sum0, sum1);
  }

  // Reduction tree: sum of all accumulators by pairs, then across lanes.
  sum0 = hn::Add(sum0, sum1);
  sum2 = hn::Add(sum2, sum3);
  sum0 = hn::Add(sum0, sum2);
  return hn::ReduceSum(df32, sum0);
}
#endif  // !HWY_HAVE_FLOAT16

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

// When float16 is not natively supported, we need a custom inner product that
// widens to f32. When HWY_HAVE_FLOAT16 is true, the generic
// InnerProductImplVectorized above (which uses Dot::Compute/MulAdd) works.
#if !HWY_HAVE_FLOAT16
template <class D, HWY_IF_F16_D(D)>
static float InnerProductImplVectorized(const D d, const hwy::float16_t* v1,
                                        const hwy::float16_t* v2,
                                        size_t num_elements) {
  if (v1 == v2) {
    return SquaredSumVectorized(d, v1, num_elements);
  }

  const hn::Repartition<float, D> df32;

  using V = decltype(hn::Zero(df32));
  const size_t N = hn::Lanes(d);
  HWY_DASSERT(num_elements >= N && num_elements % N == 0);

  size_t i = 0;
  V sum0 = hn::Zero(df32);
  V sum1 = hn::Zero(df32);
  V sum2 = hn::Zero(df32);
  V sum3 = hn::Zero(df32);

  // Main loop: unrolled
  for (; i + 2 * N <= num_elements; /* i += 2 * N */) {
    const auto a0 = hn::LoadU(d, v1 + i);
    const auto b0 = hn::LoadU(d, v2 + i);
    i += N;
    sum0 = hn::ReorderWidenMulAccumulate(df32, a0, b0, sum0, sum1);
    const auto a1 = hn::LoadU(d, v1 + i);
    const auto b1 = hn::LoadU(d, v2 + i);
    i += N;
    sum2 = hn::ReorderWidenMulAccumulate(df32, a1, b1, sum2, sum3);
  }

  // Possibly one more iteration of whole vectors
  if (i + N <= num_elements) {
    const auto a0 = hn::LoadU(d, v1 + i);
    const auto b0 = hn::LoadU(d, v2 + i);
    i += N;
    sum0 = hn::ReorderWidenMulAccumulate(df32, a0, b0, sum0, sum1);
  }

  // Reduction tree: sum of all accumulators by pairs, then across lanes.
  sum0 = hn::Add(sum0, sum1);
  sum2 = hn::Add(sum2, sum3);
  sum0 = hn::Add(sum0, sum2);
  return hn::ReduceSum(df32, sum0);
}
#endif  // !HWY_HAVE_FLOAT16

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

template <class D, HWY_IF_BF16_D(D)>
static float L2DistanceSquaredImplVectorized(
    const D d, const hwy::bfloat16_t* HWY_RESTRICT v1,
    const hwy::bfloat16_t* HWY_RESTRICT v2, size_t num_elements) {
  const hn::Repartition<float, D> df32;

  using V = decltype(hn::Zero(df32));
  const size_t N = hn::Lanes(d);
  HWY_DASSERT(num_elements >= N && num_elements % N == 0);

  size_t i = 0;

  V sum0 = hn::Zero(df32);
  V sum1 = hn::Zero(df32);
  V sum2 = hn::Zero(df32);
  V sum3 = hn::Zero(df32);

  // Main loop: unrolled
  for (; i + 2 * N <= num_elements; /* i += 2 * N */) {  // incr in loop
    const auto a0 = hn::LoadU(d, v1 + i);
    const auto a0_lower = hn::PromoteLowerTo(df32, a0);
    const auto a0_upper = hn::PromoteUpperTo(df32, a0);
    const auto a1 = hn::LoadU(d, v2 + i);
    const auto a1_lower = hn::PromoteLowerTo(df32, a1);
    const auto a1_upper = hn::PromoteUpperTo(df32, a1);
    const auto diff_a_lower = hn::Sub(a0_lower, a1_lower);
    const auto diff_a_upper = hn::Sub(a0_upper, a1_upper);
    i += N;
    sum0 = hn::MulAdd(diff_a_lower, diff_a_lower, sum0);
    sum1 = hn::MulAdd(diff_a_upper, diff_a_upper, sum1);

    const auto b0 = hn::LoadU(d, v1 + i);
    const auto b0_lower = hn::PromoteLowerTo(df32, b0);
    const auto b0_upper = hn::PromoteUpperTo(df32, b0);
    const auto b1 = hn::LoadU(d, v2 + i);
    const auto b1_lower = hn::PromoteLowerTo(df32, b1);
    const auto b1_upper = hn::PromoteUpperTo(df32, b1);
    const auto diff_b_lower = hn::Sub(b0_lower, b1_lower);
    const auto diff_b_upper = hn::Sub(b0_upper, b1_upper);
    i += N;
    sum2 = hn::MulAdd(diff_b_lower, diff_b_lower, sum2);
    sum3 = hn::MulAdd(diff_b_upper, diff_b_upper, sum3);
  }

  // Up to 1 iterations of whole vectors
  for (; i + N <= num_elements; i += N) {
    const auto a0 = hn::LoadU(d, v1 + i);
    const auto a0_lower = hn::PromoteLowerTo(df32, a0);
    const auto a0_upper = hn::PromoteUpperTo(df32, a0);
    const auto a1 = hn::LoadU(d, v2 + i);
    const auto a1_lower = hn::PromoteLowerTo(df32, a1);
    const auto a1_upper = hn::PromoteUpperTo(df32, a1);
    const auto diff_a_lower = hn::Sub(a0_lower, a1_lower);
    const auto diff_a_upper = hn::Sub(a0_upper, a1_upper);
    i += N;
    sum0 = hn::MulAdd(diff_a_lower, diff_a_lower, sum0);
    sum1 = hn::MulAdd(diff_a_upper, diff_a_upper, sum1);
  }
  // Reduction tree: sum of all accumulators by pairs, then across lanes.
  sum0 = hn::Add(sum0, sum1);
  sum2 = hn::Add(sum2, sum3);
  sum0 = hn::Add(sum0, sum2);

  return hwy::ConvertScalarTo<float>(hn::ReduceSum(df32, sum0));
}

// When float16 is not natively supported, we need to promote to f32 for
// Sub/MulAdd. When HWY_HAVE_FLOAT16 is true, the generic
// L2DistanceSquaredImplVectorized above (which uses Sub+MulAdd on float16_t
// directly) works.
#if !HWY_HAVE_FLOAT16
template <class D, HWY_IF_F16_D(D)>
static float L2DistanceSquaredImplVectorized(
    const D d, const hwy::float16_t* HWY_RESTRICT v1,
    const hwy::float16_t* HWY_RESTRICT v2, size_t num_elements) {
  const hn::Repartition<float, D> df32;

  using V = decltype(hn::Zero(df32));
  const size_t N = hn::Lanes(d);
  HWY_DASSERT(num_elements >= N && num_elements % N == 0);

  size_t i = 0;

  V sum0 = hn::Zero(df32);
  V sum1 = hn::Zero(df32);
  V sum2 = hn::Zero(df32);
  V sum3 = hn::Zero(df32);

  // Main loop: unrolled
  for (; i + 2 * N <= num_elements; /* i += 2 * N */) {  // incr in loop
    const auto a0 = hn::LoadU(d, v1 + i);
    const auto a0_lower = hn::PromoteLowerTo(df32, a0);
    const auto a0_upper = hn::PromoteUpperTo(df32, a0);
    const auto a1 = hn::LoadU(d, v2 + i);
    const auto a1_lower = hn::PromoteLowerTo(df32, a1);
    const auto a1_upper = hn::PromoteUpperTo(df32, a1);
    const auto diff_a_lower = hn::Sub(a0_lower, a1_lower);
    const auto diff_a_upper = hn::Sub(a0_upper, a1_upper);
    i += N;
    sum0 = hn::MulAdd(diff_a_lower, diff_a_lower, sum0);
    sum1 = hn::MulAdd(diff_a_upper, diff_a_upper, sum1);

    const auto b0 = hn::LoadU(d, v1 + i);
    const auto b0_lower = hn::PromoteLowerTo(df32, b0);
    const auto b0_upper = hn::PromoteUpperTo(df32, b0);
    const auto b1 = hn::LoadU(d, v2 + i);
    const auto b1_lower = hn::PromoteLowerTo(df32, b1);
    const auto b1_upper = hn::PromoteUpperTo(df32, b1);
    const auto diff_b_lower = hn::Sub(b0_lower, b1_lower);
    const auto diff_b_upper = hn::Sub(b0_upper, b1_upper);
    i += N;
    sum2 = hn::MulAdd(diff_b_lower, diff_b_lower, sum2);
    sum3 = hn::MulAdd(diff_b_upper, diff_b_upper, sum3);
  }

  // Up to 1 iterations of whole vectors
  for (; i + N <= num_elements; i += N) {
    const auto a0 = hn::LoadU(d, v1 + i);
    const auto a0_lower = hn::PromoteLowerTo(df32, a0);
    const auto a0_upper = hn::PromoteUpperTo(df32, a0);
    const auto a1 = hn::LoadU(d, v2 + i);
    const auto a1_lower = hn::PromoteLowerTo(df32, a1);
    const auto a1_upper = hn::PromoteUpperTo(df32, a1);
    const auto diff_a_lower = hn::Sub(a0_lower, a1_lower);
    const auto diff_a_upper = hn::Sub(a0_upper, a1_upper);
    i += N;
    sum0 = hn::MulAdd(diff_a_lower, diff_a_lower, sum0);
    sum1 = hn::MulAdd(diff_a_upper, diff_a_upper, sum1);
  }
  // Reduction tree: sum of all accumulators by pairs, then across lanes.
  sum0 = hn::Add(sum0, sum1);
  sum2 = hn::Add(sum2, sum3);
  sum0 = hn::Add(sum0, sum2);

  return hwy::ConvertScalarTo<float>(hn::ReduceSum(df32, sum0));
}
#endif  // !HWY_HAVE_FLOAT16

template <class D, HWY_IF_F32_D(D)>
static float L2DistanceSquaredImplVectorized(
    const D df, const float* HWY_RESTRICT v1,
    const hwy::bfloat16_t* HWY_RESTRICT v2, size_t num_elements) {
  const hn::Repartition<hwy::bfloat16_t, D> dbf;
  using VBF = decltype(hn::Zero(dbf));
  const hn::Half<decltype(dbf)> dbfh;
  using VF = decltype(hn::Zero(df));

  const size_t NF = hn::Lanes(df);
  HWY_DASSERT(num_elements >= NF && num_elements % NF == 0);

  size_t i = 0;

  VF sum0 = hn::Zero(df);
  VF sum1 = hn::Zero(df);
  VF sum2 = hn::Zero(df);
  VF sum3 = hn::Zero(df);

  // Main loop: unrolled
  for (; i + 4 * NF <= num_elements; /* i += 4 * NF */) {
    const VF a0 = hn::LoadU(df, v1 + i);
    const VBF b0 = hn::LoadU(dbf, v2 + i);
    i += NF;
    const VF b0_lower = hn::PromoteLowerTo(df, b0);
    const VF diff0 = hn::Sub(a0, b0_lower);
    sum0 = hn::MulAdd(diff0, diff0, sum0);

    const VF a1 = hn::LoadU(df, v1 + i);
    i += NF;
    const VF b0_upper = hn::PromoteUpperTo(df, b0);
    const VF diff1 = hn::Sub(a1, b0_upper);
    sum1 = hn::MulAdd(diff1, diff1, sum1);

    const VF a2 = hn::LoadU(df, v1 + i);
    const VBF b2 = hn::LoadU(dbf, v2 + i);
    i += NF;
    const VF b2_lower = hn::PromoteLowerTo(df, b2);
    const VF diff2 = hn::Sub(a2, b2_lower);
    sum2 = hn::MulAdd(diff2, diff2, sum2);

    const VF a3 = hn::LoadU(df, v1 + i);
    i += NF;
    const VF b2_upper = hn::PromoteUpperTo(df, b2);
    const VF diff3 = hn::Sub(a3, b2_upper);
    sum3 = hn::MulAdd(diff3, diff3, sum3);
  }

  // Up to 3 iterations of whole vectors
  for (; i + NF <= num_elements; i += NF) {
    const VF a = hn::LoadU(df, v1 + i);
    const VF b = hn::PromoteTo(df, hn::LoadU(dbfh, v2 + i));
    const VF diff = hn::Sub(a, b);
    sum0 = hn::MulAdd(diff, diff, sum0);
  }
  // Reduction tree: sum of all accumulators by pairs, then across lanes.
  sum0 = hn::Add(sum0, sum1);
  sum2 = hn::Add(sum2, sum3);
  sum0 = hn::Add(sum0, sum2);

  return hwy::ConvertScalarTo<float>(hn::ReduceSum(df, sum0));
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

  V sum0 = hn::Zero(d);
  V sum1 = hn::Zero(d);
  V sum2 = hn::Zero(d);
  V sum3 = hn::Zero(d);

  size_t i = 0;
  // Main loop: unrolled
  for (; i + 4 * N <= num_elements; /* i += 4 * N */) {  // incr in loop
    const auto diff0 = hn::Sub(hn::LoadU(d, v1 + i), hn::LoadU(d, v2 + i));
    i += N;
    sum0 = hn::MulAdd(diff0, diff0, sum0);
    const auto diff1 = hn::Sub(hn::LoadU(d, v1 + i), hn::LoadU(d, v2 + i));
    i += N;
    sum1 = hn::MulAdd(diff1, diff1, sum1);
    const auto diff2 = hn::Sub(hn::LoadU(d, v1 + i), hn::LoadU(d, v2 + i));
    i += N;
    sum2 = hn::MulAdd(diff2, diff2, sum2);
    const auto diff3 = hn::Sub(hn::LoadU(d, v1 + i), hn::LoadU(d, v2 + i));
    i += N;
    sum3 = hn::MulAdd(diff3, diff3, sum3);
  }

  // Up to 3 iterations of whole vectors
  for (; i + N <= num_elements; i += N) {
    const auto diff = hn::Sub(hn::LoadU(d, v1 + i), hn::LoadU(d, v2 + i));
    sum0 = hn::MulAdd(diff, diff, sum0);
  }
  // Reduction tree: sum of all accumulators by pairs, then across lanes.
  sum0 = hn::Add(sum0, sum1);
  sum2 = hn::Add(sum2, sum3);
  sum0 = hn::Add(sum0, sum2);

  return hwy::ConvertScalarTo<float>(hn::ReduceSum(d, sum0));
}

template <class D, typename T1 = hn::TFromD<D>, typename T2 = T1>
static float L2DistanceSquaredImpl(const D d, const T1* HWY_RESTRICT v1,
                                   const T2* HWY_RESTRICT v2,
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
  const T norm =
      hwy::ConvertScalarTo<T>(1.0f / (sqrtf(squared_sum) + 1e-30f));

  using V = hn::Vec<D>;
  const size_t N = hn::Lanes(d);
  const V norm_vec = hn::Set(d, norm);

  size_t i = 0;
  // Main loop: 4x unrolled
  for (; i + 4 * N <= num_elements; i += 4 * N) {
    const V v0 = hn::Mul(hn::LoadU(d, inout + i), norm_vec);
    const V v1 = hn::Mul(hn::LoadU(d, inout + i + N), norm_vec);
    const V v2 = hn::Mul(hn::LoadU(d, inout + i + 2 * N), norm_vec);
    const V v3 = hn::Mul(hn::LoadU(d, inout + i + 3 * N), norm_vec);
    hn::StoreU(v0, d, inout + i);
    hn::StoreU(v1, d, inout + i + N);
    hn::StoreU(v2, d, inout + i + 2 * N);
    hn::StoreU(v3, d, inout + i + 3 * N);
  }

  // Up to 3 remaining whole vectors
  for (; i + N <= num_elements; i += N) {
    hn::StoreU(hn::Mul(hn::LoadU(d, inout + i), norm_vec), d, inout + i);
  }

  // Remaining elements
  if (i != num_elements) {
    const size_t remaining = num_elements - i;
    const V v = hn::LoadN(d, inout + i, remaining);
    hn::StoreN(hn::Mul(v, norm_vec), d, inout + i, remaining);
  }
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

// When float16 is not natively supported, we need to promote to f32 for Mul
// and demote back. When HWY_HAVE_FLOAT16 is true, the generic NormalizeImpl
// above (which uses Set/Mul on float16_t directly) works.
#if !HWY_HAVE_FLOAT16
template <class D, HWY_IF_F16_D(D)>
static void NormalizeImpl(const D d, hwy::float16_t* HWY_RESTRICT inout,
                          size_t num_elements) {
  const float squared_sum = InnerProductImpl(d, inout, inout, num_elements);
  const float norm =
      hwy::ConvertScalarTo<float>(1.0f / (sqrtf(squared_sum) + 1e-30f));
  hn::Transform(d, inout, num_elements, [norm](D d, hn::Vec<D> v) HWY_ATTR {
    const hn::RepartitionToWide<D> df32;
    const hn::Half<D> dfh;
    const auto norm_vector = hn::Set(df32, norm);
    const auto lower = hn::Mul(hn::PromoteLowerTo(df32, v), norm_vector);
    const auto upper = hn::Mul(hn::PromoteUpperTo(df32, v), norm_vector);
    return hn::Combine(d, hn::DemoteTo(dfh, upper), hn::DemoteTo(dfh, lower));
  });
}
#endif  // !HWY_HAVE_FLOAT16

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
  if (size >= 2 * NF) {
    for (; i <= size - 2 * NF; i += 2 * NF) {
      const auto v0 = hn::LoadU(df16h, in + i);
      const auto v1 = hn::LoadU(df16h, in + i + NF);
      hn::StoreU(hn::PromoteTo(df32, v0), df32, out + i);
      hn::StoreU(hn::PromoteTo(df32, v1), df32, out + i + NF);
    }
  }
  if (size - i >= NF) {
    const auto v = hn::LoadU(df16h, in + i);
    hn::StoreU(hn::PromoteTo(df32, v), df32, out + i);
    i += NF;
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

static float InnerProductImplBF16(const hwy::bfloat16_t* v1,
                                  const hwy::bfloat16_t* v2,
                                  size_t num_elements) {
  return InnerProductImpl(hn::ScalableTag<hwy::bfloat16_t>(), v1, v2,
                          num_elements);
}

static float InnerProductImplF16(const hwy::float16_t* v1,
                                 const hwy::float16_t* v2,
                                 size_t num_elements) {
  return InnerProductImpl(hn::ScalableTag<hwy::float16_t>(), v1, v2,
                          num_elements);
}

static float L2DistanceSquaredImplF32(const float* v1, const float* v2,
                                      size_t num_elements) {
  return L2DistanceSquaredImpl(hn::ScalableTag<float>(), v1, v2, num_elements);
}

static float L2DistanceSquaredImplBF16(const hwy::bfloat16_t* v1,
                                       const hwy::bfloat16_t* v2,
                                       size_t num_elements) {
  return L2DistanceSquaredImpl(hn::ScalableTag<hwy::bfloat16_t>(), v1, v2,
                               num_elements);
}

static float L2DistanceSquaredImplF16(const hwy::float16_t* v1,
                                      const hwy::float16_t* v2,
                                      size_t num_elements) {
  return L2DistanceSquaredImpl(hn::ScalableTag<hwy::float16_t>(), v1, v2,
                               num_elements);
}

static float L2DistanceSquaredImplF32BF16(const float* v1,
                                          const hwy::bfloat16_t* v2,
                                          size_t num_elements) {
  return L2DistanceSquaredImpl(hn::ScalableTag<float>(), v1, v2, num_elements);
}

static void NormalizeImplF32(float* HWY_RESTRICT inout, size_t num_elements) {
  return NormalizeImpl(hn::ScalableTag<float>(), inout, num_elements);
}

static void NormalizeImplF16(hwy::float16_t* HWY_RESTRICT inout,
                            size_t num_elements) {
  return NormalizeImpl(hn::ScalableTag<hwy::float16_t>(), inout, num_elements);
}

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

namespace {

auto RuntimeTargetForName(int) -> decltype(hwy::SupportedTarget()) {
  return hwy::SupportedTarget();
}

auto RuntimeTargetForName(long) -> decltype(hwy::DispatchedTarget()) {
  return hwy::DispatchedTarget();
}

int64_t RuntimeTargetForName(...) { return hwy::SupportedTargets(); }

}  // namespace

// This macro declares a static array used for dynamic dispatch; it resides in
// the same outer namespace that contains FloorLog2.
HWY_EXPORT(InnerProductImplF32);
HWY_EXPORT(InnerProductImplBF16);
HWY_EXPORT(InnerProductImplF16);
HWY_EXPORT(L2DistanceSquaredImplF32);
HWY_EXPORT(L2DistanceSquaredImplBF16);
HWY_EXPORT(L2DistanceSquaredImplF16);
HWY_EXPORT(L2DistanceSquaredImplF32BF16);
HWY_EXPORT(QuantizeF32ToF16Impl);
HWY_EXPORT(QuantizeF32ToBF16Impl);
HWY_EXPORT(F16ToF32Impl);
HWY_EXPORT(BF16ToF32Impl);

HWY_EXPORT(NormalizeImplF32);
HWY_EXPORT(NormalizeImplF16);
HWY_EXPORT(NormalizeImplBF16);

HWY_DLLEXPORT float InnerProduct(const float* v1, const float* v2,
                                 size_t num_elements) {
  return HWY_DYNAMIC_DISPATCH(InnerProductImplF32)(v1, v2, num_elements);
}

HWY_DLLEXPORT float InnerProduct(const hwy::bfloat16_t* v1,
                                 const hwy::bfloat16_t* v2,
                                 size_t num_elements) {
  return HWY_DYNAMIC_DISPATCH(InnerProductImplBF16)(v1, v2, num_elements);
}

HWY_DLLEXPORT float InnerProduct(const hwy::float16_t* v1,
                                 const hwy::float16_t* v2,
                                 size_t num_elements) {
  return HWY_DYNAMIC_DISPATCH(InnerProductImplF16)(v1, v2, num_elements);
}

HWY_DLLEXPORT float InnerProductDistance(const float* v1, const float* v2,
                                         size_t num_elements) {
  return 1.0f - InnerProduct(v1, v2, num_elements);
}

HWY_DLLEXPORT float InnerProductDistance(const hwy::bfloat16_t* v1,
                                         const hwy::bfloat16_t* v2,
                                         size_t num_elements) {
  return 1.0f - InnerProduct(v1, v2, num_elements);
}

HWY_DLLEXPORT float InnerProductDistance(const hwy::float16_t* v1,
                                         const hwy::float16_t* v2,
                                         size_t num_elements) {
  return 1.0f - InnerProduct(v1, v2, num_elements);
}

HWY_DLLEXPORT void Normalize(float* HWY_RESTRICT inout, size_t size) {
  HWY_DYNAMIC_DISPATCH(NormalizeImplF32)(inout, size);
  return;
}

HWY_DLLEXPORT void Normalize(hwy::float16_t* HWY_RESTRICT inout, size_t size) {
  HWY_DYNAMIC_DISPATCH(NormalizeImplF16)(inout, size);
  return;
}

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

HWY_DLLEXPORT float L2DistanceSquared(const hwy::bfloat16_t* v1,
                                      const hwy::bfloat16_t* v2,
                                      size_t num_elements) {
  if (HWY_UNLIKELY(v1 == v2)) {
    return 0.0f;
  }

  return HWY_DYNAMIC_DISPATCH(L2DistanceSquaredImplBF16)(v1, v2, num_elements);
}

HWY_DLLEXPORT float L2DistanceSquared(const hwy::float16_t* v1,
                                      const hwy::float16_t* v2,
                                      size_t num_elements) {
  if (HWY_UNLIKELY(v1 == v2)) {
    return 0.0f;
  }

  return HWY_DYNAMIC_DISPATCH(L2DistanceSquaredImplF16)(v1, v2, num_elements);
}

// v1 and v2 MUST not be nullptr but **cannot** point to the same array.
HWY_DLLEXPORT float L2DistanceSquared(const float* v1,
                                      const hwy::bfloat16_t* v2,
                                      size_t num_elements) {
  return HWY_DYNAMIC_DISPATCH(L2DistanceSquaredImplF32BF16)(v1, v2,
                                                            num_elements);
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

HWY_DLLEXPORT void Normalize_Scalar(hwy::float16_t* HWY_RESTRICT inout,
                                    size_t size) {
  float norm = 0.0f;
  for (int i = 0; i < size; i++) {
    float data = hwy::F32FromF16(inout[i]);
    norm += data * data;
  }
  norm = 1.0f / (sqrtf(norm) + 1e-30f);
  for (int i = 0; i < size; i++) {
    inout[i] = hwy::F16FromF32(hwy::F32FromF16(inout[i]) * norm);
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

HWY_DLLEXPORT const char* GetRuntimeTarget() {
  return hwy::TargetName(RuntimeTargetForName(0));
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
