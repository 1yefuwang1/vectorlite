#include "distance.h"
// >>>> for dynamic dispatch only, skip if you want static dispatch

// For dynamic dispatch, specify the name of the current file (unfortunately
// __FILE__ is not reliable) so that foreach_target.h can re-include it.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "distance.cpp"
// Generates code for each enabled target by re-including this source file.
#include "hwy/foreach_target.h"  // IWYU pragma: keep

// <<<< end of dynamic dispatch

// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/highway.h"
#include "hwy/targets.h"
#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/contrib/math/math-inl.h"

// Optional, can instead add HWY_ATTR to all functions.
HWY_BEFORE_NAMESPACE();

// This namespace name is unique per target, which allows code for multiple
// targets to co-exist in the same translation unit. Required when using dynamic
// dispatch, otherwise optional.
namespace HWY_NAMESPACE {

// Highway ops reside here; ADL does not find templates nor builtins.
namespace hn = hwy::HWY_NAMESPACE;

static float SquaredSum(const float* v, size_t num_elements) {
	const hn::ScalableTag<float> d;
	using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);
	HWY_ASSERT(num_elements >= N && num_elements % N == 0);

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

static float InnerProductImpl(const float* v1, const float* v2, size_t num_elements) {
	const hn::ScalableTag<float> d;
	const size_t N = hn::Lanes(d);
	HWY_ASSERT(num_elements >= N && num_elements % N == 0);

	constexpr int assumption = hn::Dot::kAtLeastOneVector | hn::Dot::kMultipleOfVector;
	if (HWY_LIKELY(v1 != v2)) {
		return hn::Dot::Compute<assumption>(d, v1, v2, num_elements);
	} else {
		return SquaredSum(v1, num_elements);
	}
}

} // namespace HWY_NAMESPACE

HWY_AFTER_NAMESPACE();


// The table of pointers to the various implementations in HWY_NAMESPACE must
// be compiled only once (foreach_target #includes this file multiple times).
// HWY_ONCE is true for only one of these 'compilation passes'.
#if HWY_ONCE

namespace vectorlite {
namespace distance {

// This macro declares a static array used for dynamic dispatch; it resides in
// the same outer namespace that contains FloorLog2.
HWY_EXPORT(InnerProductImpl);

namespace hn = hwy::HWY_NAMESPACE;

HWY_DLLEXPORT float InnerProduct(const float* v1, const float* v2, size_t num_elements) {
	const hn::ScalableTag<float> d;
  const size_t N = hn::Lanes(d);

	const size_t leftover = num_elements % N;

	float result = 0;
	if (num_elements >= N) {
		const auto ptr = HWY_DYNAMIC_POINTER(InnerProductImpl);
		result = ptr(v1, v2, num_elements - leftover);
	}

	if (leftover > 0) {
		// Manually 2x unroll the loop
		float sum0 = 0;
		float sum1 = 0;
		size_t i = num_elements - leftover;
		for (; i + 2 <= num_elements; i += 2) {
			sum0 += v1[i] * v2[i];
			sum1 += v1[i + 1] * v2[i + 1];
		}

		if (i < num_elements) {
			sum0 += v1[i] * v2[i];
		}
		return result + sum0 + sum1;
	} else {
		return result;
	}
}

HWY_DLLEXPORT std::string_view DetectTarget() {
	uint64_t supported_targets = HWY_SUPPORTED_TARGETS;
	hwy::GetChosenTarget().Update(supported_targets);
	return hwy::TargetName(supported_targets);
}

} // namespace distance
} // namespace vectorlite

#endif