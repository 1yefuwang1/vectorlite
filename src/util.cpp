#include "util.h"
#include <absl/status/status.h>

#include <regex>

#include "sqlite3.h"
#include "vector.h"

namespace sqlite_vector {

bool IsValidColumnName(const std::string& name) {
  if (name.empty() || sqlite3_keyword_check(name.c_str(), name.size()) != 0) {
    return false;
  }

  static const std::regex kColumnNameRegex("^[a-zA-Z_][a-zA-Z0-9_\\$]*$");

  return std::regex_match(name, kColumnNameRegex);
}


absl::StatusOr<Vector> ParseVector(std::string_view json) {
  Vector v;
  auto result = Vector::FromJSON(json, &v);
  if (result == Vector::ParseResult::kOk) {
    return v;
  } else if (result == Vector::ParseResult::kParseFailed) {
    return absl::InvalidArgumentError("Failed to parse JSON");
  } else if (result == Vector::ParseResult::kInvalidElementType) {
    return absl::InvalidArgumentError("Invalid element type in JSON");
  } else if (result == Vector::ParseResult::kInvalidJSONType) {
    return absl::InvalidArgumentError("Invalid JSON type");
  }

  return absl::InternalError("Unknown error");
}

} // end namespace sqlite_vector