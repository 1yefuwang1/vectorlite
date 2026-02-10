#pragma once

#include "vector.h"
#include "vector_view.h"

namespace vectorlite {

BF16Vector Quantize(VectorView v);
F16Vector QuantizeToF16(VectorView v);

}  // namespace vectorlite