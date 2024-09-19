#pragma once

#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "hwy/base.h"
#include "macros.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "util.h"

namespace vectorlite {

template <class T, VECTORLITE_IF_FLOAT_SUPPORTED_FWD_DECL(T)>
class GenericVector;

// GenericVectorView is a read-only view of a vector, like what std::string_view
// is to std::string.
template <class T, VECTORLITE_IF_FLOAT_SUPPORTED(T)>
class GenericVectorView : private CopyAssignBase<T>,
                          private CopyCtorBase<T>,
                          private MoveCtorBase<T>,
                          private MoveAssignBase<T> {
 public:
  GenericVectorView() = default;
  GenericVectorView(const GenericVectorView&) = default;
  GenericVectorView(GenericVectorView&&) = default;

  GenericVectorView(const GenericVector<T, nullptr>& vector)
      : data_(vector.data()) {}
  explicit GenericVectorView(absl::Span<const T> data) : data_(data) {}

  GenericVectorView& operator=(const GenericVectorView&) = default;
  GenericVectorView& operator=(GenericVectorView&&) = default;

  static absl::StatusOr<GenericVectorView<T>> FromBlob(std::string_view blob) {
    if (blob.size() % sizeof(T) != 0) {
      return absl::InvalidArgumentError("Blob size is not a multiple of float");
    }
    return GenericVectorView(absl::MakeSpan(
        reinterpret_cast<const T*>(blob.data()), blob.size() / sizeof(T)));
  };

  std::string ToJSON() const {
    rapidjson::Document doc;
    doc.SetArray();

    auto& allocator = doc.GetAllocator();
    for (T v : data_) {
      doc.PushBack(hwy::ConvertScalarTo<float>(v), allocator);
    }

    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
    doc.Accept(writer);

    return buf.GetString();
  };

  std::string_view ToBlob() const {
    return std::string_view(reinterpret_cast<const char*>(data_.data()),
                            data_.size() * sizeof(T));
  };

  std::size_t dim() const { return data_.size(); }

  absl::Span<const T> data() const { return data_; }

 private:
  absl::Span<const T> data_;
};

using VectorView = GenericVectorView<float>;
using BF16VectorView = GenericVectorView<hwy::bfloat16_t>;

}  // namespace vectorlite