#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "macros.h"

namespace sqlite_vector {

class Vector {
 public:
  Vector() = default;
  Vector(const Vector&) = default;
  Vector(Vector&&) = default;

  explicit Vector(std::vector<float>&& data) : data_(std::move(data)) {}
  explicit Vector(const std::vector<float>& data) : data_(data) {}

  Vector& operator=(const Vector&) = default;
  Vector& operator=(Vector&&) = default;

  static absl::StatusOr<Vector> FromJSON(std::string_view json);

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
