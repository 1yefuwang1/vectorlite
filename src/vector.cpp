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
  std::vector<float> result;
  if (blob.size() % sizeof(float) != 0) {
    return absl::InvalidArgumentError("Blob size is not a multiple of 4.");
  }
  result.resize(blob.size() / sizeof(float));
  std::memcpy(result.data(), blob.data(), blob.size());
  return Vector(std::move(result));
}

std::string Vector::ToJSON() const {
  rapidjson::Document doc;
  doc.SetArray();

  auto& allocator = doc.GetAllocator();
  for (float v : data_) {
    doc.PushBack(v, allocator);
  }

  rapidjson::StringBuffer buf;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
  doc.Accept(writer);

  return buf.GetString();
}

absl::StatusOr<float> Distance(const Vector& v1, const Vector& v2,
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

  const Vector& lhs = vector_space->normalize ? v1.Normalize() : v1;
  const Vector& rhs = vector_space->normalize ? v2.Normalize() : v2;
  return vector_space->space->get_dist_func()(
      lhs.data().data(), rhs.data().data(),
      vector_space->space->get_dist_func_param());
}

std::string_view Vector::ToBlob() const {
  return std::string_view(reinterpret_cast<const char*>(data_.data()),
                          data_.size() * sizeof(float));
}

// Implementation follows
// https://github.com/nmslib/hnswlib/blob/v0.8.0/python_bindings/bindings.cpp#L241
Vector Vector::Normalize() const {
  std::vector<float> normalized(data_.size());
  float norm = 0.0f;
  for (float data : data_) {
    norm += data * data;
  }
  norm = 1.0f / (sqrtf(norm) + 1e-30f);
  for (int i = 0; i < data_.size(); i++) {
    normalized[i] = data_[i] * norm;
  }
  return Vector(std::move(normalized));
}

}  // namespace vectorlite