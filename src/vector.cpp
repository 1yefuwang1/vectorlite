#include "vector.h"
#include <cstddef>
#include <string_view>

#include "hnswlib/hnswlib.h"
#include "hnswlib/space_l2.h"
#include "macros.h"
#include "msgpack.hpp"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace sqlite_vector {

std::string Vector::ToMsgPack() const {
  msgpack::sbuffer sbuf;
  msgpack::pack(sbuf, data_);
  return std::string(sbuf.data(), sbuf.size());
}

absl::StatusOr<Vector> Vector::FromMsgPack(std::string_view json) {
  auto handle = msgpack::unpack(json.data(), json.size());
  auto obj = handle.get();
  try {
    std::vector<float> result = obj.as<std::vector<float>>();
    return Vector(std::move(result));
  } catch (const msgpack::type_error& e) {
    return absl::InvalidArgumentError(e.what());
  }
}

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

absl::StatusOr<Vector> Vector::FromBlob(std::string_view binary) {
  std::vector<float> result;
  if (binary.size() % sizeof(float) != 0) {
    return absl::InvalidArgumentError("Binary size is not a multiple of 4.");
  }
  result.resize(binary.size() / sizeof(float));
  std::memcpy(result.data(), binary.data(), binary.size());
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

float L2Distance(const Vector& v1, const Vector& v2) {
  SQLITE_VECTOR_ASSERT(v1.dim() == v2.dim());
  hnswlib::L2Space space(v1.dim());
  return space.get_dist_func()(v1.data().data(), v2.data().data(),
                               space.get_dist_func_param());
}

std::string_view Vector::ToBlob() const {
  return std::string_view(reinterpret_cast<const char*>(data_.data()),
                          data_.size() * sizeof(float));
}


// Implementation follows https://github.com/nmslib/hnswlib/blob/v0.8.0/python_bindings/bindings.cpp#L241
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

}  // namespace sqlite_vector