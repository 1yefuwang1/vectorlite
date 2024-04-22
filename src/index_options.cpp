#include "index_options.h"

#include <absl/status/status.h>

#include <string_view>

#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "re2/re2.h"

namespace vectorlite {

absl::StatusOr<IndexOptions> IndexOptions::FromString(
    std::string_view index_options) {
  static const re2::RE2 hnsw_reg("^hnsw\\((.*)\\)$");
  std::string key_value;
  if (!re2::RE2::FullMatch(index_options, hnsw_reg, &key_value)) {
    return absl::InvalidArgumentError(
        "Invalid index option. Only hnsw is supported");
  }

  IndexOptions options;
  static const re2::RE2 kv_reg("([\\w]+)=([\\w]+)");

  bool has_max_elements = false;
  std::string key;
  std::string value;

  std::string_view input(key_value);
  while (re2::RE2::FindAndConsume(&input, kv_reg, &key, &value)) {
    if (key == "max_elements") {
      if (!absl::SimpleAtoi<size_t>(value, &options.max_elements)) {
        std::string error =
            absl::StrFormat("Cannot parse max_elements: %s", value);
        return absl::InvalidArgumentError(error);
      }
      has_max_elements = true;
    } else if (key == "M") {
      if (!absl::SimpleAtoi<size_t>(value, &options.M)) {
        std::string error = absl::StrFormat("Cannot parse M: %s", value);
        return absl::InvalidArgumentError(error);
      }
    } else if (key == "ef_construction") {
      if (!absl::SimpleAtoi<size_t>(value, &options.ef_construction)) {
        std::string error =
            absl::StrFormat("Cannot parse ef_construction: %s", value);
        return absl::InvalidArgumentError(error);
      }
    } else if (key == "random_seed") {
      if (!absl::SimpleAtoi<size_t>(value, &options.random_seed)) {
        std::string error =
            absl::StrFormat("Cannot parse random_seed: %s", value);
        return absl::InvalidArgumentError(error);
      }
    } else if (key == "allow_replace_deleted") {
      if (!absl::SimpleAtob(value, &options.allow_replace_deleted)) {
        std::string error =
            absl::StrFormat("Cannot parse allow_replace_deleted: %s", value);
        return absl::InvalidArgumentError(error);
      }
    } else {
      std::string error = absl::StrFormat("Invalid index option: %s", key);
      return absl::InvalidArgumentError(error);
    }
  }

  if (!has_max_elements) {
    return absl::InvalidArgumentError(
        "max_elements is required but not provided");
  }
  return options;
}

}  // namespace vectorlite