#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "macros.h"
#include "vector_space.h"
#include "vector_view.h"

namespace vectorlite {

class Vector {
 public:
  Vector() = default;
  Vector(const Vector&) = default;
  Vector(Vector&&) = default;

  explicit Vector(std::vector<float>&& data) : data_(std::move(data)) {}
  explicit Vector(const std::vector<float>& data) : data_(data) {}
  explicit Vector(VectorView vector_view)
      : data_(vector_view.data().begin(), vector_view.data().end()) {}

  Vector& operator=(const Vector&) = default;
  Vector& operator=(Vector&&) = default;

  static absl::StatusOr<Vector> FromJSON(std::string_view json);

  static absl::StatusOr<Vector> FromBlob(std::string_view blob);

  std::string ToJSON() const;

  std::string_view ToBlob() const;

  const std::vector<float>& data() const { return data_; }

  std::size_t dim() const { return data_.size(); }

  Vector Normalize() const;

  static Vector Normalize(VectorView vector_view);

 private:
  std::vector<float> data_;
};

// Calculate the distance between two vectors.
absl::StatusOr<float> Distance(VectorView v1, VectorView v2,
                               DistanceType space);

}  // namespace vectorlite
