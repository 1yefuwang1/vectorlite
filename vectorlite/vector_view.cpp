#include "vector_view.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "vector.h"

namespace vectorlite {

VectorView::VectorView(const Vector& vector) : data_(vector.data()) {}

absl::StatusOr<VectorView> VectorView::FromBlob(std::string_view blob) {
  if (blob.size() % sizeof(float) != 0) {
    return absl::InvalidArgumentError("Blob size is not a multiple of float");
  }
  return VectorView(absl::MakeSpan(reinterpret_cast<const float*>(blob.data()),
                                   blob.size() / sizeof(float)));
}

std::string VectorView::ToJSON() const {
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

std::string_view VectorView::ToBlob() const {
  return std::string_view(reinterpret_cast<const char*>(data_.data()),
                          data_.size() * sizeof(float));
}

}  // namespace vectorlite