#pragma once

#include "vector.h"
#include "vector_view.h"

namespace vectorlite {

BF16Vector Quantize(VectorView v);

}  // namespace vectorlite