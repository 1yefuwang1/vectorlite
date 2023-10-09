#pragma once

#include <hnswlib/hnswlib.h>
#include <rapidjson/document.h>

#include <string_view>
#include <string>
#include <vector>

#include "macros.h"

namespace sqlite_vector {

class Vector {
 public:
  Vector() = default;

  static int FromJSON(std::string_view json, Vector* out);

  std::string ToJSON() const;

 private:
  std::vector<float> data_;
};

}  // namespace sqlite_vector
