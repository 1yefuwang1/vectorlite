#pragma once

#include <utility>
#include <optional>
#include <string_view>

namespace sqlite_vector {

// Tests whether the given string is a valid column name in SQLite.
// Requirements are:
// - It must begin with a letter or underscore.
// - It can be followed by any combination of letters, underscores, digits, or
// dollar signs.
// - It must not be a reserved keyword.
// The input is of string type because built-in regex doesn't work with
// string_view
bool IsValidColumnName(std::string_view name);

// Returns which SIMD instruction set is used at build time.
// e.g. SSE, AVX, AVX512
std::optional<std::string_view> DetectSIMD();

// Checks whether the minimum required version of SQLite3 is met.
// If met, returns (true, "")
// If not met, returns false and a human readable explanation.
std::pair<bool, std::string_view> IsMinimumSqlite3VersionMet();

}  // end namespace sqlite_vector
