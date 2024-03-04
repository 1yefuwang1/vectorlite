#include "index_options.h"

#include <regex>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"

namespace sqlite_vector {

absl::StatusOr<IndexOptions> IndexOptions::FromString(
    const std::string& index_options) {
  IndexOptions options;
  static const std::regex reg("(\\w+)=([\\w\\d]+)");
  std::smatch match;

  bool has_max_elements = false;

  while (std::regex_search(index_options, match, reg)) {
    if (match.size() != 3) {
      return absl::InvalidArgumentError("Invalid index options string");
    }

    if (match[1] == "max_elements") {
      const std::string& max_elements_string = match[2].str();
      if (!absl::SimpleAtoi<size_t>(max_elements_string,
                                    &options.max_elements)) {
        std::string error = absl::StrFormat("Cannot parse max_elements: %s",
                                            max_elements_string);
        return absl::InvalidArgumentError(error);
      }
      has_max_elements = true;
    } else if (match[1] == "M") {
      const std::string& m_string = match[2].str();
      if (!absl::SimpleAtoi<size_t>(m_string, &options.M)) {
        std::string error = absl::StrFormat("Cannot parse M: %s", m_string);
        return absl::InvalidArgumentError(error);
      }
    } else if (match[1] == "ef_construction") {
      const std::string& ef_construction_string = match[2].str();
      if (!absl::SimpleAtoi<size_t>(ef_construction_string,
                                    &options.ef_construction)) {
        std::string error = absl::StrFormat("Cannot parse ef_construction: %s",
                                            ef_construction_string);
        return absl::InvalidArgumentError(error);
      }
    } else if (match[1] == "random_seed") {
      const std::string& random_seed_string = match[2].str();
      if (!absl::SimpleAtoi<size_t>(random_seed_string, &options.random_seed)) {
        std::string error =
            absl::StrFormat("Cannot parse random_seed: %s", random_seed_string);
        return absl::InvalidArgumentError(error);
      }
    } else if (match[1] == "allow_replace_deleted") {
      const std::string& allow_replace_deleted_string = match[2].str();
      if (!absl::SimpleAtob(allow_replace_deleted_string,
                            &options.allow_replace_deleted)) {
        std::string error =
            absl::StrFormat("Cannot parse allow_replace_deleted: %s",
                            allow_replace_deleted_string);
        return absl::InvalidArgumentError(error);
      }
    } else {
      std::string error =
          absl::StrFormat("Invalid index option: %s", match[1].str());
      return absl::InvalidArgumentError(error);
    }
  }

  if (!has_max_elements) {
    return absl::InvalidArgumentError(
        "max_elements is required but not provided");
  }
  return options;
}

}  // namespace sqlite_vector