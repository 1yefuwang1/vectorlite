#include "util.h"

#include <optional>
#include <string_view>

#include "absl/status/status.h"
#include "hnswlib/hnswlib.h"
#include "re2/re2.h"
#include "sqlite3.h"
#include "vector.h"

namespace vectorlite {

bool IsValidColumnName(std::string_view name) {
  if (name.empty() || sqlite3_keyword_check(name.data(), name.size()) != 0) {
    return false;
  }

  static const re2::RE2 kColumnNameRegex("^[a-zA-Z_][a-zA-Z0-9_\\$]*$");

  return re2::RE2::FullMatch(name, kColumnNameRegex);
}

std::optional<std::string_view> DetectSIMD() {
#ifdef USE_SSE
  return "SSE";
#elif defined(USE_AVX)
  return "AVX";
#elif defined(USE_AVX512)
  return "AVX512";
#else
  return std::nullopt;
#endif
}


bool IsRowidInIndex(const hnswlib::HierarchicalNSW<float>& index, hnswlib::labeltype rowid) {
  std::unique_lock<std::mutex> lock_label(index.getLabelOpMutex(rowid));
  std::unique_lock<std::mutex> lock_table(index.label_lookup_lock);
  auto search = index.label_lookup_.find(rowid);
  if (search == index.label_lookup_.end() ||
      index.isMarkedDeleted(search->second)) {
    return false;
  }
  return true;
}

}  // end namespace vectorlite