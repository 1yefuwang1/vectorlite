#pragma once

#include <hwy/base.h>

#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "macros.h"
#include "ops/ops.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "util.h"
#include "vector_space.h"
#include "vector_view.h"

namespace vectorlite {

template <class T, VECTORLITE_IF_FLOAT_SUPPORTED(T)>
class GenericVector : private CopyAssignBase<T>,
                      private CopyCtorBase<T>,
                      private MoveCtorBase<T>,
                      private MoveAssignBase<T> {
 public:
  GenericVector() = default;
  GenericVector(const GenericVector&) = default;
  GenericVector(GenericVector&&) = default;

  explicit GenericVector(std::vector<T>&& data) : data_(std::move(data)) {}
  explicit GenericVector(const std::vector<T>& data) : data_(data) {}
  explicit GenericVector(GenericVectorView<T> vector_view)
      : data_(vector_view.data().begin(), vector_view.data().end()) {}

  GenericVector& operator=(const GenericVector&) = default;
  GenericVector& operator=(GenericVector&&) = default;

  static absl::StatusOr<GenericVector<T>> FromJSON(std::string_view json) {
    rapidjson::Document doc;
    doc.Parse(json.data(), json.size());
    auto err = doc.GetParseError();
    if (err != rapidjson::ParseErrorCode::kParseErrorNone) {
      return absl::InvalidArgumentError(rapidjson::GetParseError_En(err));
    }

    GenericVector<T> result;

    if (doc.IsArray()) {
      for (auto& v : doc.GetArray()) {
        if (v.IsNumber()) {
          result.data_.push_back(hwy::ConvertScalarTo<T>(v.GetFloat()));
        } else {
          return absl::InvalidArgumentError(
              "JSON array contains non-numeric value.");
        }
      }
      return result;
    }

    return absl::InvalidArgumentError("Input JSON is not an array.");
  }

  static absl::StatusOr<GenericVector<T>> FromBlob(std::string_view blob) {
    auto vector_view = GenericVectorView<T>::FromBlob(blob);
    if (vector_view.ok()) {
      return GenericVector<T>(*vector_view);
    }
    return vector_view.status();
  }

  std::string ToJSON() const {
    GenericVectorView<T> vector_view(*this);

    return vector_view.ToJSON();
  }

  std::string_view ToBlob() const {
    GenericVectorView<T> vector_view(*this);

    return vector_view.ToBlob();
  };

  const std::vector<T>& data() const { return data_; }

  std::size_t dim() const { return data_.size(); }

  GenericVector<T> Normalize() const {
    GenericVectorView<T> vector_view(*this);

    return GenericVector<T>::Normalize(vector_view);
  }

  static GenericVector<T> Normalize(GenericVectorView<T> vector_view) {
    GenericVector<T> v(vector_view);
    ops::Normalize(v.data_.data(), vector_view.dim());
    return v;
  }

 private:
  std::vector<T> data_;
};

template <class T, VECTORLITE_IF_FLOAT_SUPPORTED(T)>
using DistanceFunc = float (*)(const T*, const T*, size_t);

// Calculate the distance between two vectors.
template <class T, VECTORLITE_IF_FLOAT_SUPPORTED(T)>
absl::StatusOr<float> Distance(GenericVectorView<T> v1, GenericVectorView<T> v2,
                               DistanceType distance_type) {
  if (v1.dim() != v2.dim()) {
    std::string err =
        absl::StrFormat("Dimension mismatch: %d != %d", v1.dim(), v2.dim());
    return absl::InvalidArgumentError(err);
  }

  DistanceFunc<T> distance_func = nullptr;

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

  GenericVector<T> lhs = GenericVector<T>::Normalize(v1);
  GenericVector<T> rhs = GenericVector<T>::Normalize(v2);
  return distance_func(lhs.data().data(), rhs.data().data(), v1.dim());
}

using Vector = GenericVector<float>;
using BF16Vector = GenericVector<hwy::bfloat16_t>;

}  // namespace vectorlite
