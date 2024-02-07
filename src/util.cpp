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

} // end namespace sqlite_vector