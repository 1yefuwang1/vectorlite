#include "vector.h"

namespace sqlite_vector {

int Vector::FromJSON(std::string_view json, Vector* out) {
  SQLITE_VECTOR_ASSERT(out != nullptr);

  rapidjson::Document doc;
  doc.Parse(json.data(), json.size());
  auto err = doc.GetParseError();
  if (err != rapidjson::ParseErrorCode::kParseErrorNone) {
    return -1;
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
        return -1;
      }
    }
    return 0;
  }

  return -1;
}

}  // namespace sqlite_vector