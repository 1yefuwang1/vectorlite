#pragma once

#include <cstddef>
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

  explicit Vector(std::vector<float>&& data) : data_(std::move(data)) {}
  explicit Vector(const std::vector<float>& data) : data_(data) {}

  enum class ParseResult {
    kOk,
    kParseFailed,
    kInvalidElementType,
    kInvalidJSONType,
  };

  Vector& operator=(const Vector&) = default;
  Vector& operator=(Vector&&) = default;

  // Pasrse a JSON string into [out]. [out] should not be nullptr.
  // and should points to an empty vector.
  static ParseResult FromJSON(std::string_view json, Vector* out);

  std::string ToJSON() const;

  const std::vector<float>& data() const { return data_; }

  std::size_t dim() const { return data_.size(); }

 private:
  std::vector<float> data_;
};

// Calculate the L2 distance between two vectors.
// v1 and v2 must have the same dimension.
float L2Distance(const Vector& v1, const Vector& v2);

}  // namespace sqlite_vector
