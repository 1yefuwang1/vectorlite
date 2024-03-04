#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

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
  SpaceType type;
  std::string vector_name;
  bool normalize;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space;

  size_t dimension() const;

  // Parses a string into VectorSpace.
  // This input is usually from the CREATE VIRTUAL TABLE statement.
  // e.g. CREATE VIRTUAL TABLE my_vectors using sqlite_vector(my_vector(384, "l2"), "hnsw(max_elements=1000)")
  // The `vector(384, "l2")` is the vector space string. Supported space type are "l2", "cos", "ip"
  // The second parameter is optional and defaults to "l2" if not specified.
  static absl::StatusOr<VectorSpace> FromString(const std::string& space_str);
};

absl::StatusOr<VectorSpace> CreateVectorSpace(size_t dim, SpaceType space_type, std::string_view vector_name);

}  // namespace sqlite_vector