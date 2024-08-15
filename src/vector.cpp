#include "vector.h"

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include <cstddef>
#include <memory>
#include <string_view>

#include "hnswlib/hnswlib.h"
#include "hnswlib/space_l2.h"
#include "macros.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "vector_space.h"
#include "vector_view.h"
#include "ops/ops.h"

namespace vectorlite {

absl::StatusOr<Vector> Vector::FromJSON(std::string_view json) {
  rapidjson::Document doc;
  doc.Parse(json.data(), json.size());
  auto err = doc.GetParseError();
  if (err != rapidjson::ParseErrorCode::kParseErrorNone) {
    return absl::InvalidArgumentError(rapidjson::GetParseError_En(err));
  }

  Vector result;

  if (doc.IsArray()) {
    for (auto& v : doc.GetArray()) {
      if (v.IsNumber()) {
        result.data_.push_back(v.GetFloat());
      } else {
        return absl::InvalidArgumentError(
            "JSON array contains non-numeric value.");
      }
    }
    return result;
  }

  return absl::InvalidArgumentError("Input JSON is not an array.");
}

absl::StatusOr<Vector> Vector::FromBlob(std::string_view blob) {
  auto vector_view = VectorView::FromBlob(blob);
  if (vector_view.ok()) {
    return Vector(*vector_view);
  }
  return vector_view.status();
}

std::string Vector::ToJSON() const {
  VectorView vector_view(*this);

  return vector_view.ToJSON();
}

absl::StatusOr<float> Distance(VectorView v1, VectorView v2,
                               DistanceType distance_type) {
  if (v1.dim() != v2.dim()) {
    std::string err =
        absl::StrFormat("Dimension mismatch: %d != %d", v1.dim(), v2.dim());
    return absl::InvalidArgumentError(err);
  }

  ops::DistanceFunc distance_func = nullptr;
  
  switch (distance_type) {
    case DistanceType::L2:
      distance_func = ops::L2DistanceSquared;
      break;
    case DistanceType::InnerProduct:
      distance_func = ops::InnerProductDistance;
      break;
    case DistanceType::Cosine:
      distance_func = ops::InnerProductDistance;
      break;
    default:
      return absl::InvalidArgumentError("Invalid distance type");
  }

  bool normalize = distance_type == DistanceType::Cosine;

  if (!normalize) {
    return distance_func(v1.data().data(), v2.data().data(), v1.dim());
  }

  Vector lhs = Vector::Normalize(v1);
  Vector rhs = Vector::Normalize(v2);
  return distance_func(lhs.data().data(), rhs.data().data(), v1.dim());
}

std::string_view Vector::ToBlob() const {
  VectorView vector_view(*this);

  return vector_view.ToBlob();
}

Vector Vector::Normalize() const {
  VectorView vector_view(*this);

  return Vector::Normalize(vector_view);
}

Vector Vector::Normalize(VectorView vector_view) {
  Vector v(vector_view);
  ops::Normalize(v.data_.data(), vector_view.dim());
  return v;
}

}  // namespace vectorlite