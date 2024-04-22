#pragma once

#include <memory>
#include <optional>
#include <string_view>

#include "absl/status/statusor.h"
#include "hnswlib/hnswlib.h"

namespace vectorlite {

enum class SpaceType {
  L2,
  InnerProduct,
  Cosine,
};

std::optional<SpaceType> ParseSpaceType(std::string_view space_type);

struct VectorSpace {
  SpaceType type;
  bool normalize;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space;

  size_t dimension() const;

  static absl::StatusOr<VectorSpace> Create(size_t dim, SpaceType space_type);
};

struct NamedVectorSpace : public VectorSpace {
  std::string vector_name;

  NamedVectorSpace(VectorSpace&& other) : VectorSpace(std::move(other)) {}

  // Parses a string into NamedVectorSpace.
  // This input is usually from the CREATE VIRTUAL TABLE statement.
  // e.g. CREATE VIRTUAL TABLE my_vectors using vectorlite(my_vector(384,
  // "l2"), "hnsw(max_elements=1000)") The `vector(384, "l2")` is the vector
  // space string. Supported space type are "l2", "cos", "ip"
  static absl::StatusOr<NamedVectorSpace> FromString(
      std::string_view space_str);
};

absl::StatusOr<NamedVectorSpace> CreateNamedVectorSpace(
    size_t dim, SpaceType space_type, std::string_view vector_name);

}  // namespace vectorlite