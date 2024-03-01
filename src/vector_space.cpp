#include "vector_space.h"
#include <absl/strings/str_format.h>

namespace sqlite_vector {

std::optional<SpaceType> ParseSpaceType(std::string_view space_type) {
  if (space_type == "l2") {
    return SpaceType::L2;
  } else if (space_type == "ip") {
    return SpaceType::InnerProduct;
  } else if (space_type == "cosine") {
    return SpaceType::Cosine;
  }
  return std::nullopt;
}

absl::StatusOr<VectorSpace> CreateVectorSpace(size_t dim, SpaceType space_type) {
	if (dim == 0) {
		return absl::InvalidArgumentError("Dimension must be greater than 0");
	}

  VectorSpace result;
  result.normalize = space_type == SpaceType::Cosine;
  switch (space_type) {
    case SpaceType::L2:
      result.space = std::make_unique<hnswlib::L2Space>(dim);
      break;
    case SpaceType::InnerProduct:
      result.space = std::make_unique<hnswlib::InnerProductSpace>(dim);
      break;
    case SpaceType::Cosine:
      result.space = std::make_unique<hnswlib::InnerProductSpace>(dim);
      break;
    default:
			std::string err_msg = absl::StrFormat("Invalid space type: %d", space_type);
      return absl::InvalidArgumentError(err_msg);
  }

  return result;
}

} // end namespace sqlite_vector