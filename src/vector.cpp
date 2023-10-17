#include "vector.h"

#include "hnswlib/hnswlib.h"
#include "hnswlib/space_l2.h"
#include "macros.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace sqlite_vector {

Vector::ParseResult Vector::FromJSON(std::string_view json, Vector* out) {
  SQLITE_VECTOR_ASSERT(out != nullptr);

  rapidjson::Document doc;
  doc.Parse(json.data(), json.size());
  auto err = doc.GetParseError();
  if (err != rapidjson::ParseErrorCode::kParseErrorNone) {
    return Vector::ParseResult::kParseFailed;
  }

  if (!out->data_.empty()) {
    out->data_.clear();
  }

  if (doc.IsArray()) {
    for (auto& v : doc.GetArray()) {
      if (v.IsNumber()) {
        out->data_.push_back(v.GetFloat());
      } else {
        out->data_.clear();
        return Vector::ParseResult::kInvalidElementType;
      }
    }
    return Vector::ParseResult::kOk;
  }

  return Vector::ParseResult::kInvalidJSONType;
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