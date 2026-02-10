#include "vector_space.h"

#include <hnswlib/hnswlib.h>

#include <memory>
#include <optional>
#include <string_view>

#include "absl/base/optimization.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "distance.h"
#include "macros.h"
#include "re2/re2.h"
#include "util.h"

namespace vectorlite {

std::optional<DistanceType> ParseDistanceType(std::string_view distance_type) {
  if (distance_type == "l2") {
    return DistanceType::L2;
  } else if (distance_type == "ip") {
    return DistanceType::InnerProduct;
  } else if (distance_type == "cosine") {
    return DistanceType::Cosine;
  }
  return std::nullopt;
}

std::optional<VectorType> ParseVectorType(std::string_view vector_type) {
  if (vector_type == "float32") {
    return VectorType::Float32;
  }

  if (vector_type == "bfloat16") {
    return VectorType::BFloat16;
  }

  if (vector_type == "float16") {
    return VectorType::Float16;
  }

  return std::nullopt;
}

static std::unique_ptr<hnswlib::SpaceInterface<float>> CreateL2Space(
    size_t dim, VectorType vector_type) {
  switch (vector_type) {
    case VectorType::Float32:
      return std::make_unique<vectorlite::L2Space>(dim);
    case VectorType::BFloat16:
      return std::make_unique<vectorlite::L2SpaceBF16>(dim);
    case VectorType::Float16:
      return std::make_unique<vectorlite::L2SpaceF16>(dim);
    default:
      // This should never happen, but we include it for completeness
      ABSL_UNREACHABLE();
      return nullptr;
  }
}

static std::unique_ptr<hnswlib::SpaceInterface<float>> CreateInnerProductSpace(
    size_t dim, VectorType vector_type) {
  switch (vector_type) {
    case VectorType::Float32:
      return std::make_unique<vectorlite::InnerProductSpace>(dim);
    case VectorType::BFloat16:
      return std::make_unique<vectorlite::InnerProductSpaceBF16>(dim);
    case VectorType::Float16:
      return std::make_unique<vectorlite::InnerProductSpaceF16>(dim);
    default:
      // This should never happen, but we include it for completeness
      ABSL_UNREACHABLE();
      return nullptr;
  }
}

absl::StatusOr<VectorSpace> VectorSpace::Create(size_t dim,
                                                DistanceType distance_type,
                                                VectorType vector_type) {
  if (dim == 0) {
    return absl::InvalidArgumentError("Dimension must be greater than 0");
  }

  VectorSpace result;
  result.distance_type = distance_type;
  result.normalize = distance_type == DistanceType::Cosine;
  result.vector_type = vector_type;
  switch (distance_type) {
    case DistanceType::L2:
      result.space = CreateL2Space(dim, vector_type);
      break;
    case DistanceType::InnerProduct:
      result.space = CreateInnerProductSpace(dim, vector_type);
      break;
    case DistanceType::Cosine:
      result.space = CreateInnerProductSpace(dim, vector_type);
      break;
    default:
      std::string err_msg =
          absl::StrFormat("Invalid space type: %d", distance_type);
      return absl::InvalidArgumentError(err_msg);
  }

  return result;
}

absl::StatusOr<NamedVectorSpace> CreateNamedVectorSpace(
    size_t dim, DistanceType distance_type, std::string_view vector_name,
    VectorType vector_type) {
  auto result = VectorSpace::Create(dim, distance_type, vector_type);

  if (!result.ok()) {
    return result.status();
  }

  NamedVectorSpace named_vector_space(std::move(*result));
  named_vector_space.vector_name = vector_name;
  return named_vector_space;
}

absl::StatusOr<NamedVectorSpace> NamedVectorSpace::FromString(
    std::string_view space_str) {
  static const re2::RE2 reg(
      "^(?<vector_name>\\w+)\\s+(?<vector_type>\\w+)\\[(?<dim>\\d+)\\]\\s*(?<"
      "distance_type>\\w+)?\\s*$");
  VECTORLITE_ASSERT(reg.ok());

  std::string_view vector_name;
  std::string_view vector_type_str;
  size_t dim = 0;
  std::optional<std::string_view> distance_type_str;
  if (re2::RE2::FullMatch(space_str, reg, &vector_name, &vector_type_str, &dim,
                          &distance_type_str)) {
    if (!IsValidColumnName(vector_name)) {
      std::string error =
          absl::StrFormat("Invalid vector name: %s", vector_name);
      return absl::InvalidArgumentError(error);
    }

    auto vector_type = ParseVectorType(vector_type_str);
    if (!vector_type) {
      std::string error =
          absl::StrFormat("Invalid vector type: %s", vector_type_str);
      return absl::InvalidArgumentError(error);
    }

    DistanceType distance_type = DistanceType::L2;
    if (distance_type_str) {
      auto maybe_distance_type = ParseDistanceType(*distance_type_str);
      if (!maybe_distance_type) {
        std::string error =
            absl::StrFormat("Invalid distance type: %s", *distance_type_str);
        return absl::InvalidArgumentError(error);
      }
      distance_type = *maybe_distance_type;
    }

    return CreateNamedVectorSpace(dim, distance_type, vector_name,
                                  *vector_type);
  }
  return absl::InvalidArgumentError("Unable to parse vector space");
}

size_t VectorSpace::dimension() const {
  return *reinterpret_cast<size_t*>(space->get_dist_func_param());
}

}  // end namespace vectorlite