#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "macros.h"
#include "vector_space.h"

namespace vectorlite {

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

  static absl::StatusOr<Vector> FromMsgPack(std::string_view json);

  static absl::StatusOr<Vector> FromBlob(std::string_view blob);

  std::string ToJSON() const;

  std::string ToMsgPack() const;

  std::string_view ToBlob() const;

  const std::vector<float>& data() const { return data_; }

  std::size_t dim() const { return data_.size(); }

  Vector Normalize() const;

 private:
  std::vector<float> data_;
};

// Calculate the distance between two vectors.
absl::StatusOr<float> Distance(const Vector& v1, const Vector& v2,
                               SpaceType space);

}  // namespace vectorlite
