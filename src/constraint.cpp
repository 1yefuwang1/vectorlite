#include "constraint.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>

#include "absl/base/optimization.h"
#include "absl/functional/overload.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "hnswlib/hnswlib.h"
#include "macros.h"
#include "sqlite3ext.h"
#include "util.h"

namespace vectorlite {

absl::Status RowIdEquals::DoMaterialize(const sqlite3_api_routines* sqlite3_api,
                                        sqlite3_value* arg) {
  VECTORLITE_ASSERT(sqlite3_api != nullptr);
  VECTORLITE_ASSERT(arg != nullptr);
  if (sqlite3_value_type(arg) != SQLITE_INTEGER) {
    return absl::InvalidArgumentError("rowid must be of type INTEGER");
  }

  // TODO: handle rowid out of range
  hnswlib::labeltype rowid =
      static_cast<hnswlib::labeltype>(sqlite3_value_int64(arg));
  rowid_ = rowid;

  return absl::OkStatus();
}

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
    // TODO: handle rowid out of range
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

  if (vector_constraint_) {
    status_ =
        absl::AlreadyExistsError("only one knn_search constraint is allowed");
    return;
  }

  vector_constraint_ = &constraint;
}

void QueryExecutor::Visit(const RowIdIn& constraint) {
  if (!constraint.materialized()) {
    status_ = absl::FailedPreconditionError("rowid_in not materialized");
    return;
  }
  if (!status_.ok()) {
    return;
  }

  if (rowid_constraint_) {
    status_ =
        absl::InvalidArgumentError("only one rowid constraint is allowed");
    return;
  }

  rowid_constraint_ = &constraint;
}

void QueryExecutor::Visit(const RowIdEquals& constraint) {
  if (!constraint.materialized()) {
    status_ = absl::FailedPreconditionError("rowid_eq not materialized");
    return;
  }
  if (!status_.ok()) {
    return;
  }

  if (rowid_constraint_) {
    status_ =
        absl::InvalidArgumentError("only one rowid constraint is allowed");
    return;
  }

  rowid_constraint_ = &constraint;
}

namespace {

class RowidInFilter : public hnswlib::BaseFilterFunctor {
 public:
  explicit RowidInFilter(
      const absl::flat_hash_set<hnswlib::labeltype>& rowid_in)
      : rowid_in_(rowid_in) {}
  virtual bool operator()(hnswlib::labeltype id) override {
    return rowid_in_.contains(id);
  }

 private:
  const absl::flat_hash_set<hnswlib::labeltype>& rowid_in_;
};

class RowidEqualsFilter : public hnswlib::BaseFilterFunctor {
 public:
  explicit RowidEqualsFilter(hnswlib::labeltype rowid) : rowid_(rowid) {}
  virtual bool operator()(hnswlib::labeltype id) override {
    return id == rowid_;
  }

 private:
  hnswlib::labeltype rowid_;
};

std::unique_ptr<hnswlib::BaseFilterFunctor> MakeRowidFilter(
    std::optional<absl::variant<const RowIdIn*, const RowIdEquals*>>
        row_id_constraint) {
  if (!row_id_constraint) {
    return nullptr;
  }

  return absl::visit(
      absl::Overload(
          [](const RowIdIn* rowid_in)
              -> std::unique_ptr<hnswlib::BaseFilterFunctor> {
            return std::make_unique<RowidInFilter>(rowid_in->get_rowids());
          },
          [](const RowIdEquals* rowid_equals)
              -> std::unique_ptr<hnswlib::BaseFilterFunctor> {
            return std::make_unique<RowidEqualsFilter>(rowid_equals->rowid());
          }),
      *row_id_constraint);
}

}  // namespace

absl::StatusOr<QueryExecutor::QueryResult> QueryExecutor::Execute() const {
  if (!status_.ok()) {
    return status_;
  }

  if (vector_constraint_) {
    // we are doing a vector search
    const KnnParam* knn_param = vector_constraint_->knn_param();
    VECTORLITE_ASSERT(knn_param != nullptr);

    if (space_.dimension() != knn_param->query_vector.dim()) {
      std::string error = absl::StrFormat(
          "query vector's dimension(%d) doesn't match %s's dimension: %d",
          knn_param->query_vector.dim(), space_.vector_name,
          space_.dimension());
      return absl::InvalidArgumentError(error);
    }

    auto rowid_filter = MakeRowidFilter(rowid_constraint_);
    if (knn_param->ef_search.has_value()) {
      index_.setEf(*knn_param->ef_search);
    }
    auto result = index_.searchKnnCloserFirst(
        space_.normalize ? knn_param->query_vector.Normalize().data().data()
                         : knn_param->query_vector.data().data(),
        knn_param->k, rowid_filter.get());
    return result;
  } else {
    QueryExecutor::QueryResult result;
    if (rowid_constraint_) {
      // we are doing a rowid search without using hnsw index
      absl::visit(absl::Overload(
                      [&result, this](const RowIdIn* rowid_in) {
                        for (const auto& rowid : rowid_in->get_rowids()) {
                          // TODO: IsRowidInIndex takes a lock on
                          // index.label_lookup_ on each invoke, Lock once in
                          // the future.
                          if (IsRowidInIndex(index_, rowid)) {
                            result.emplace_back(0.0f, rowid);
                          }
                        }
                      },
                      [&result, this](const RowIdEquals* rowid_equals) {
                        if (IsRowidInIndex(index_, rowid_equals->rowid())) {
                          result.emplace_back(0.0f, rowid_equals->rowid());
                        }
                      }),
                  *rowid_constraint_);
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

absl::StatusOr<std::vector<std::unique_ptr<Constraint>>>
ParseConstraintsFromShortNames(std::string_view constraint_str) {
  if (constraint_str.size() % 2 != 0) {
    return absl::InvalidArgumentError("constraint_str's size() must be even");
  }
  std::vector<std::unique_ptr<Constraint>> constraints;
  for (size_t i = 0; i < constraint_str.size(); i += 2) {
    std::string_view short_name = constraint_str.substr(i, 2);
    if (short_name == RowIdIn::kShortName) {
      constraints.push_back(std::make_unique<RowIdIn>());
    } else if (short_name == RowIdEquals::kShortName) {
      constraints.push_back(std::make_unique<RowIdEquals>());
    } else if (short_name == KnnSearchConstraint::kShortName) {
      constraints.push_back(std::make_unique<KnnSearchConstraint>());
    } else {
      return absl::InvalidArgumentError(
          absl::StrFormat("unknown constraint short name: %s", short_name));
    }
  }

  return constraints;
}

}  // namespace vectorlite