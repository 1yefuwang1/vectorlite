#include "vector.h"
#include <msgpack/v3/unpack_decl.hpp>

#include "hnswlib/hnswlib.h"
#include "hnswlib/space_l2.h"
#include "macros.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "rapidjson/error/en.h"
#include "msgpack.hpp"

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
        return absl::InvalidArgumentError("JSON array contains non-numeric value.");
      }
    }
    return result;
  }

  return absl::InvalidArgumentError("Input JSON is not an array.");
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

}  // namespace sqlite_vector