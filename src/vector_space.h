#pragma once

#include <memory>
#include <optional>
#include <string_view>

#include "hnswlib/hnswlib.h"
#include "absl/status/statusor.h"

namespace sqlite_vector {

enum class SpaceType {
    L2,
    InnerProduct,
    Cosine,
};

std::optional<SpaceType> ParseSpaceType(std::string_view space_type); 

struct VectorSpace {
    bool normalize;
    std::unique_ptr<hnswlib::SpaceInterface<float>> space;
};

absl::StatusOr<VectorSpace> CreateVectorSpace(size_t dim, SpaceType space_type);

}  // namespace sqlite_vector