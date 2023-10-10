#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "macros.h"

namespace sqlite_vector {

class Vector {
 public:
  Vector() = default;
  Vector(const Vector&) = default;
  Vector(Vector&&) = default;

  Vector(std::vector<float>&& data) : data_(std::move(data)) {}
  Vector(const std::vector<float>& data) : data_(data) {}

  enum class ParseResult {
    kOk,
    kParseFailed,
    kInvalidElementType,
    kInvalidJSONType,
  };

  // Pasrse a JSON string into [out]. [out] should not be nullptr.
  // and should points to an empty vector.
  static ParseResult FromJSON(std::string_view json, Vector* out);

  std::string ToJSON() const;

  const std::vector<float>& get_data() const { return data_; }

 private:
  std::vector<float> data_;
};

}  // namespace sqlite_vector
