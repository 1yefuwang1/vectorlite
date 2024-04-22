#include "vector_space.h"

#include <string_view>

#include "absl/strings/str_format.h"
#include "re2/re2.h"
#include "util.h"

namespace vectorlite {

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

absl::StatusOr<VectorSpace> VectorSpace::Create(size_t dim,
                                                SpaceType space_type) {
  if (dim == 0) {
    return absl::InvalidArgumentError("Dimension must be greater than 0");
  }

  VectorSpace result;
  result.type = space_type;
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
      std::string err_msg =
          absl::StrFormat("Invalid space type: %d", space_type);
      return absl::InvalidArgumentError(err_msg);
  }

  return result;
}

absl::StatusOr<NamedVectorSpace> CreateNamedVectorSpace(
    size_t dim, SpaceType space_type, std::string_view vector_name) {
  auto result = VectorSpace::Create(dim, space_type);

  if (!result.ok()) {
    return result.status();
  }

  NamedVectorSpace named_vector_space(std::move(*result));
  named_vector_space.vector_name = vector_name;
  return named_vector_space;
}

absl::StatusOr<NamedVectorSpace> NamedVectorSpace::FromString(
    std::string_view space_str) {
  static const re2::RE2 reg("([\\w]+)\\((\\d+),\\s*\"([\\w]+)\"\\)");

  std::string vector_name;
  std::string dim_str;
  std::string space_type_str;
  if (re2::RE2::FullMatch(space_str, reg, &vector_name, &dim_str,
                          &space_type_str)) {
    size_t dim;
    if (!absl::SimpleAtoi(dim_str, &dim)) {
      std::string error =
          absl::StrFormat("Cannot parse dimension: %s", dim_str);
      return absl::InvalidArgumentError(error);
    }

    auto space_type = ParseSpaceType(space_type_str);
    if (!space_type) {
      std::string error =
          absl::StrFormat("Invalid space type: %s", space_type_str);
      return absl::InvalidArgumentError(error);
    }

    if (!IsValidColumnName(vector_name)) {
      std::string error =
          absl::StrFormat("Invalid vector name: %s", vector_name);
      return absl::InvalidArgumentError(error);
    }
    return CreateNamedVectorSpace(dim, *space_type, vector_name);
  }
  return absl::InvalidArgumentError("Unable to parse vector space");
}

size_t VectorSpace::dimension() const {
  return *reinterpret_cast<size_t*>(space->get_dist_func_param());
}

}  // end namespace vectorlite