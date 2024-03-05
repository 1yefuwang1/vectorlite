#pragma once

#include <string_view>

#include "absl/status/statusor.h"

namespace sqlite_vector {

struct IndexOptions {
  size_t max_elements;
  size_t M = 16;
  size_t ef_construction = 200;
  size_t random_seed = 100;
  bool allow_replace_deleted = false;

  // Parses a string into IndexOptions.
  // This input is usually from the CREATE VIRTUAL TABLE statement.
  // e.g. CREATE VIRTUAL TABLE my_vectors using sqlite_vector(my_vector(384, "l2"),
  // "hnsw(max_elements=1000,M=16,ef_construction=200,random_seed=100,allow_replace_deleted=false)")
  // The second parameter to sqlite_vector() is the index options string.
  // All parameters except max_elemnts are optional, default values are used
  // if not specified.
  static absl::StatusOr<IndexOptions> FromString(std::string_view index_options);
};

}  // namespace sqlite_vector