#pragma once

#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace vectorlite {

class Vector;

// VectorView is a read-only view of a vector, like std::string_view is to
// std::string.
class VectorView {
 public:
  VectorView() = default;
  VectorView(const VectorView&) = default;
  VectorView(VectorView&&) = default;

  VectorView(const Vector& vector);
  explicit VectorView(absl::Span<const float> data) : data_(data) {}

  VectorView& operator=(const VectorView&) = default;
  VectorView& operator=(VectorView&&) = default;

  static absl::StatusOr<VectorView> FromBlob(std::string_view blob);

  std::string ToJSON() const;

  std::string_view ToBlob() const;

  std::size_t dim() const { return data_.size(); }

  absl::Span<const float> data() const { return data_; }

 private:
  absl::Span<const float> data_;
};

}  // namespace vectorlite