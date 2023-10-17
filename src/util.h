#pragma once

#include <string>
#include <string_view>

#include "vector.h"

#include "absl/status/statusor.h"


namespace sqlite_vector {

// Tests whether the given string is a valid column name in SQLite.
// Requirements are:
// - It must begin with a letter or underscore.
// - It can be followed by any combination of letters, underscores, digits, or dollar signs.
// - It must not be a reserved keyword.
// The input is of string type because built-in regex doesn't work with string_view
bool IsValidColumnName(const std::string& name);

absl::StatusOr<Vector> ParseVector(std::string_view json);
} // end namespace sqlite_vector