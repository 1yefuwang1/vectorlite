#include "quantization.h"

#include <vector>

#include "hwy/base.h"
#include "ops/ops.h"
#include "vector.h"
#include "vector_view.h"

namespace vectorlite {

BF16Vector Quantize(VectorView v) {
  std::vector<hwy::bfloat16_t> quantized(v.dim());
  ops::QuantizeF32ToBF16(v.data().data(), quantized.data(), v.dim());

  return BF16Vector(std::move(quantized));
}

F16Vector QuantizeToF16(VectorView v) {
  std::vector<hwy::float16_t> quantized(v.dim());
  ops::QuantizeF32ToF16(v.data().data(), quantized.data(), v.dim());

  return F16Vector(std::move(quantized));
}

}  // namespace vectorlite