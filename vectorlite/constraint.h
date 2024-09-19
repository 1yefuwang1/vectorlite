#pragma once

#include <memory>
#include <optional>
#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/variant.h"
#include "hnswlib/hnswlib.h"
#include "macros.h"
#include "sqlite3.h"
#include "vector_space.h"
#include "vector_view.h"

namespace vectorlite {

struct KnnParam {
  VectorView query_vector;
  uint32_t k;
  std::optional<uint32_t> ef_search;
};

// Used to identify pointer type for sqlite_result_pointer/sqlite_value_pointer
constexpr std::string_view kKnnParamType = "vectorlite_knn_param";

class KnnSearchConstraint;
class RowIdIn;
class RowIdEquals;

class ConstraintVisitor {
 public:
  virtual ~ConstraintVisitor() = default;

  virtual void Visit(const KnnSearchConstraint& constraint) = 0;
  virtual void Visit(const RowIdIn& constraint) = 0;
  virtual void Visit(const RowIdEquals& constraint) = 0;
};

class QueryExecutor : public ConstraintVisitor {
 public:
  using QueryResult = std::vector<std::pair<float, hnswlib::labeltype>>;

  QueryExecutor(hnswlib::HierarchicalNSW<float>& index,
                const NamedVectorSpace& space)
      : index_(index), space_(space) {}
  virtual ~QueryExecutor() = default;

  // Should only be called iff IsOk() returns true.
  absl::StatusOr<QueryResult> Execute() const;

  void Visit(const KnnSearchConstraint& constraint) override;
  void Visit(const RowIdIn& constraint) override;
  void Visit(const RowIdEquals& constraint) override;

  bool ok() const { return status_.ok(); }

  // If IsOk() is false, status() returns the error message.
  // The returned pointer is valid until the QueryExecutor object is destroyed.
  const char* message() const {
    VECTORLITE_ASSERT(!ok());
    return absl::StatusMessageAsCStr(status_);
  }

 private:
  // setting ef when querying the index is allowed. So index_ cannot be marked
  // as const.
  hnswlib::HierarchicalNSW<float>& index_;
  const NamedVectorSpace& space_;
  absl::Status status_;

  // there can at most one KnnParam constraint
  const KnnSearchConstraint* vector_constraint_ = nullptr;

  // there can be at most one vector constraint
  std::optional<absl::variant<const RowIdIn*, const RowIdEquals*>>
      rowid_constraint_;
};

class Constraint {
 public:
  virtual ~Constraint() = default;

  // Constraints can only get its required data inside xFilter.
  // Materialize should be only be called in xFliter and before calling
  // Accept(), otherwise the behavior is undefined.
  absl::Status Materialize(const sqlite3_api_routines* sqlite3_api,
                           sqlite3_value* arg) {
    if (!materialized_) {
      auto status = DoMaterialize(sqlite3_api, arg);
      if (status.ok()) {
        materialized_ = true;
      }
      return status;
    }

    return absl::OkStatus();
  }

  virtual void Accept(ConstraintVisitor* visitor) = 0;

  virtual std::string ToDebugString() const = 0;

  bool materialized() const { return materialized_; }

 private:
  virtual absl::Status DoMaterialize(const sqlite3_api_routines* sqlite3_api,
                                     sqlite3_value* arg) = 0;
  bool materialized_ = false;
};

class KnnSearchConstraint : public Constraint {
 public:
  // Name used in idxStr that is created in xBestIndex and then passed to
  // xFilter
  constexpr static std::string_view kShortName = "ks";

  KnnSearchConstraint() : knn_param_(nullptr) {}

  void Accept(ConstraintVisitor* visitor) override { visitor->Visit(*this); }

  const KnnParam* knn_param() const { return knn_param_; }

  std::string ToDebugString() const override {
    if (materialized()) {
      return absl::StrFormat("knn_parm(vector of dim %d, %d)",
                             knn_param_->query_vector.dim(), knn_param_->k);
    }

    return absl::StrFormat("knn_param(?)");
  }

 private:
  absl::Status DoMaterialize(const sqlite3_api_routines* sqlite3_api,
                             sqlite3_value* arg) override;
  const KnnParam* knn_param_;
};

class RowIdIn : public Constraint {
 public:
  // Name used in idxStr that is created in xBestIndex and then passed to
  // xFilter
  constexpr static std::string_view kShortName = "in";

  RowIdIn() : rowids_() {}

  void Accept(ConstraintVisitor* visitor) override { visitor->Visit(*this); }

  const absl::flat_hash_set<hnswlib::labeltype>& get_rowids() const {
    return rowids_;
  }

 private:
  virtual absl::Status DoMaterialize(const sqlite3_api_routines* sqlite3_api,
                                     sqlite3_value* arg) override;

  std::string ToDebugString() const override {
    if (materialized()) {
      return absl::StrFormat("rowid in (%d rowids...)", rowids_.size());
    }

    return absl::StrFormat("rowid in (?)");
  }
  absl::flat_hash_set<hnswlib::labeltype> rowids_;
};

class RowIdEquals : public Constraint {
 public:
  // Name used in idxStr that is created in xBestIndex and then passed to
  // xFilter
  constexpr static std::string_view kShortName = "eq";

  explicit RowIdEquals() : rowid_() {}

  void Accept(ConstraintVisitor* visitor) override { visitor->Visit(*this); }

  hnswlib::labeltype rowid() const { return rowid_; }

 private:
  virtual absl::Status DoMaterialize(const sqlite3_api_routines* sqlite3_api,
                                     sqlite3_value* arg) override;

  std::string ToDebugString() const override {
    if (materialized()) {
      return absl::StrFormat("rowid = %d", rowid_);
    }

    return "rowid = ?";
  }

  hnswlib::labeltype rowid_;
};

std::string ConstraintsToDebugString(
    const std::vector<std::unique_ptr<Constraint>>& constraints);

absl::StatusOr<std::vector<std::unique_ptr<Constraint>>>
ParseConstraintsFromShortNames(std::string_view constraint_str);

}  // namespace vectorlite