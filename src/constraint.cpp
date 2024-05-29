#include "constraint.h"
#include <hnswlib/hnswalg.h>

#include <algorithm>
#include <memory>
#include <mutex>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "hnswlib/hnswlib.h"
#include "macros.h"
#include "sqlite3ext.h"
#include "util.h"

namespace vectorlite {

absl::Status RowIdIn::DoMaterialize(const sqlite3_api_routines* sqlite3_api,
                                    sqlite3_value* arg) {
  VECTORLITE_ASSERT(sqlite3_api != nullptr);
  VECTORLITE_ASSERT(arg != nullptr);
  int rc = SQLITE_OK;
  sqlite3_value* rowid_value = nullptr;
  for (rc = sqlite3_vtab_in_first(arg, &rowid_value); rc == SQLITE_OK;
       rc = sqlite3_vtab_in_next(arg, &rowid_value)) {
    if (ABSL_PREDICT_FALSE(sqlite3_value_type(rowid_value) != SQLITE_INTEGER)) {
      return absl::InvalidArgumentError("rowid must be of type INTEGER");
    }
    hnswlib::labeltype rowid =
        static_cast<hnswlib::labeltype>(sqlite3_value_int64(rowid_value));
    rowids_.insert(rowid);
  }
  return absl::OkStatus();
}

absl::Status KnnSearchConstraint::DoMaterialize(
    const sqlite3_api_routines* sqlite3_api, sqlite3_value* arg) {
  VECTORLITE_ASSERT(sqlite3_api != nullptr);
  VECTORLITE_ASSERT(arg != nullptr);

  auto knn_param =
      static_cast<KnnParam*>(sqlite3_value_pointer(arg, kKnnParamType.data()));
  if (!knn_param) {
    return absl::InvalidArgumentError(
        "knn_param() should be used for the 2nd param of knn_search()");
  }
  knn_param_ = knn_param;
  return absl::OkStatus();
}

void QueryExecutor::Visit(const KnnSearchConstraint& constraint) {
  if (!constraint.materialized()) {
    status_ = absl::FailedPreconditionError("knn_search not materialized");
    return;
  }
  if (!status_.ok()) {
    return;
  }

  if (knn_param_) {
    status_ =
        absl::InvalidArgumentError("only one knn_search constraint is allowed");
    return;
  }

  knn_param_ = constraint.get_knn_param();
}

void QueryExecutor::Visit(const RowIdIn& constraint) {
  if (!constraint.materialized()) {
    status_ = absl::FailedPreconditionError("rowid_in not materialized");
    return;
  }
  if (!status_.ok()) {
    return;
  }

  rowid_in_.push_back(constraint.get_rowids());
}

void QueryExecutor::Visit(const RowIdEquals& constraint) {
  if (!status_.ok()) {
    return;
  }

  if (rowid_equals_ && *rowid_equals_ != constraint.rowid()) {
    status_ = absl::InvalidArgumentError(
        "only one rowid_equals constraint is allowed");
    return;
  }

  rowid_equals_ = constraint.rowid();
}

namespace {

class RowidInFilter : public hnswlib::BaseFilterFunctor {
 public:
  RowidInFilter(
      const std::vector<const absl::flat_hash_set<hnswlib::labeltype>*>&
          rowid_in)
      : rowid_in_(rowid_in) {}
  virtual bool operator()(hnswlib::labeltype id) override {
    return std::all_of(
        rowid_in_.begin(), rowid_in_.end(),
        [id](const absl::flat_hash_set<hnswlib::labeltype>* rowids) {
          return rowids->contains(id);
        });
  }

 private:
  const std::vector<const absl::flat_hash_set<hnswlib::labeltype>*>& rowid_in_;
};

class RowidEqualsFilter : public hnswlib::BaseFilterFunctor {
 public:
  RowidEqualsFilter(hnswlib::labeltype rowid) : rowid_(rowid) {}
  virtual bool operator()(hnswlib::labeltype id) override {
    return id == rowid_;
  }

 private:
  hnswlib::labeltype rowid_;
};

class RowidInAndEqualsFilter : public hnswlib::BaseFilterFunctor {
 public:
  RowidInAndEqualsFilter(
      const std::vector<const absl::flat_hash_set<hnswlib::labeltype>*>&
          rowid_in,
      hnswlib::labeltype rowid)
      : rowid_in_(rowid_in), rowid_(rowid) {}
  virtual bool operator()(hnswlib::labeltype id) override {
    return std::all_of(
               rowid_in_.begin(), rowid_in_.end(),
               [id](const absl::flat_hash_set<hnswlib::labeltype>* rowids) {
                 return rowids->contains(id);
               }) &&
           id == rowid_;
  }

 private:
  const std::vector<const absl::flat_hash_set<hnswlib::labeltype>*>& rowid_in_;
  hnswlib::labeltype rowid_;
};

std::unique_ptr<hnswlib::BaseFilterFunctor> MakeRowidFilter(
    const std::vector<const absl::flat_hash_set<hnswlib::labeltype>*>& rowid_in,
    std::optional<hnswlib::labeltype> rowid_equals) {
  if (rowid_in.empty() && !rowid_equals) {
    return nullptr;
  }

  if (rowid_in.empty()) {
    return std::make_unique<RowidEqualsFilter>(*rowid_equals);
  }

  if (!rowid_equals) {
    return std::make_unique<RowidInFilter>(rowid_in);
  }

  return std::make_unique<RowidInAndEqualsFilter>(rowid_in, *rowid_equals);
}

}  // namespace

absl::StatusOr<QueryExecutor::QueryResult> QueryExecutor::Execute() const {
  if (!status_.ok()) {
    return status_;
  }
  if (knn_param_) {
    // we are doing a vector search
    if (space_.dimension() != knn_param_->query_vector.dim()) {
      std::string error = absl::StrFormat(
          "query vector's dimension(%d) doesn't match %s's dimension: %d",
          knn_param_->query_vector.dim(), space_.vector_name,
          space_.dimension());
      return absl::InvalidArgumentError(error);
    }

    auto filter = MakeRowidFilter(rowid_in_, rowid_equals_);
    auto result = index_.searchKnnCloserFirst(
        knn_param_->query_vector.data().data(), knn_param_->k, filter.get());
    return result;
  } else {
    // we are doing a rowid search without using hnsw index
    QueryExecutor::QueryResult result;
    auto isIdAllowed = MakeRowidFilter(rowid_in_, rowid_equals_);
    if (rowid_equals_) {
      if (IsRowidInIndex(index_, *rowid_equals_) && (*isIdAllowed)(*rowid_equals_)) {
        result.push_back({0.0f, *rowid_equals_});
      }
    }

    for (const auto& s : rowid_in_) {
      for (auto rowid : *s) {
        if (IsRowidInIndex(index_, rowid) && (*isIdAllowed)(rowid)) {
          result.push_back({0.0f, rowid});
        }
      }
    }
    return result;
  }
}

std::string ConstraintsToDebugString(
    const std::vector<std::unique_ptr<Constraint>>& constraints) {
  std::vector<std::string> constraint_strings;
  std::transform(constraints.begin(), constraints.end(),
                 std::back_inserter(constraint_strings),
                 [](const std::unique_ptr<Constraint>& constraint) {
                   return constraint->ToDebugString();
                 });
  return absl::StrJoin(constraint_strings, ", ");
}

}  // namespace vectorlite