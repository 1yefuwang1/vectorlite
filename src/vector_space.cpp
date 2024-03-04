#include "vector_space.h"

#include <regex>
#include <string_view>

#include "absl/strings/str_format.h"
#include "util.h"

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

absl::StatusOr<VectorSpace> CreateVectorSpace(size_t dim, SpaceType space_type,
                                              std::string_view vector_name) {
  if (dim == 0) {
    return absl::InvalidArgumentError("Dimension must be greater than 0");
  }

  VectorSpace result;
  result.type = space_type;
  result.vector_name = std::string(vector_name);
  switch (space_type) {
    case SpaceType::L2:
      result.space = std::make_unique<hnswlib::L2Space>(dim);
      break;
    case SpaceType::InnerProduct:
      result.space = std::make_unique<hnswlib::InnerProductSpace>(dim);
      break;
    case SpaceType::Cosine:
      result.space = std::make_unique<hnswlib::InnerProductSpace>(dim);
      result.normalize = true;
      break;
    default:
      std::string err_msg =
          absl::StrFormat("Invalid space type: %d", space_type);
      return absl::InvalidArgumentError(err_msg);
  }

  return result;
}

absl::StatusOr<VectorSpace> VectorSpace::FromString(const std::string& space_str) {
  static const std::regex reg("([\\w\\d]+)\\((d+),\\s*\"([\\w\\d]+)\"\\)");
  std::smatch match;

  if (std::regex_match(space_str, match, reg)) {
    if (match.size() != 4) {
      return absl::InvalidArgumentError("Invalid vector space string");
    }

    size_t dim;
    const std::string dim_str = match[2].str();
    if (!absl::SimpleAtoi(dim_str, &dim)) {
      std::string error = absl::StrFormat("Cannot parse dimension: %s", dim_str);
      return absl::InvalidArgumentError(error);
    }

    const std::string space_type_str = match[3].str();
    auto space_type = ParseSpaceType(space_type_str);
    if (!space_type) {
      std::string error = absl::StrFormat("Invalid space type: %s", space_type_str);
      return absl::InvalidArgumentError(error);
    }

    const std::string vector_name = match[1].str();
    if (!IsValidColumnName(vector_name)) {
      std::string error = absl::StrFormat("Invalid vector name: %s", vector_name);
      return absl::InvalidArgumentError(error);
    }
    return CreateVectorSpace(dim, *space_type, vector_name);
  }
  return absl::InvalidArgumentError("Unable to parse vector space");
}

size_t VectorSpace::dimension() const {
  return *reinterpret_cast<size_t*>(space->get_dist_func_param());
}

}  // end namespace sqlite_vector