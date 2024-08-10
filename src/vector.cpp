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

  auto vector_space =
      VectorSpace::Create(v1.dim(), distance_type, VectorType::Float32);
  if (!vector_space.ok()) {
    return vector_space.status();
  }

  if (!vector_space->normalize) {
    return vector_space->space->get_dist_func()(
        v1.data().data(), v2.data().data(),
        vector_space->space->get_dist_func_param());
  }

  VECTORLITE_ASSERT(vector_space->normalize);

  Vector lhs = Vector::Normalize(v1);
  Vector rhs = Vector::Normalize(v2);
  return vector_space->space->get_dist_func()(
      lhs.data().data(), rhs.data().data(),
      vector_space->space->get_dist_func_param());
}

std::string_view Vector::ToBlob() const {
  VectorView vector_view(*this);

  return vector_view.ToBlob();
}

Vector Vector::Normalize() const {
  VectorView vector_view(*this);

  return Vector::Normalize(vector_view);
}

// Implementation follows
// https://github.com/nmslib/hnswlib/blob/v0.8.0/python_bindings/bindings.cpp#L241
Vector Vector::Normalize(VectorView vector_view) {
  std::vector<float> normalized(vector_view.data().size());
  float norm = 0.0f;
  for (float data : vector_view.data()) {
    norm += data * data;
  }
  norm = 1.0f / (sqrtf(norm) + 1e-30f);
  for (int i = 0; i < vector_view.data().size(); i++) {
    normalized[i] = vector_view.data()[i] * norm;
  }
  return Vector(std::move(normalized));
}

}  // namespace vectorlite