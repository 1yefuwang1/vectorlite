#pragma once

#include <absl/strings/str_format.h>
#include <hnswlib/hnswalg.h>

#include <optional>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hnswlib/hnswlib.h"
#include "macros.h"
#include "sqlite3ext.h"
#include "vector.h"
#include "vector_space.h"

namespace vectorlite {

struct KnnParam {
  Vector query_vector;
  uint32_t k;
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

  QueryExecutor(const hnswlib::HierarchicalNSW<float>& index,
                const NamedVectorSpace& space)
      : index_(index), space_(space), status_() {}
  virtual ~QueryExecutor() = default;

  // Should only be called iff IsOk() returns true.
  absl::StatusOr<QueryResult> Execute() const;

  void Visit(const KnnSearchConstraint& constraint) override;
  void Visit(const RowIdIn& constraint) override;
  void Visit(const RowIdEquals& constraint) override;

  bool IsOk() const { return status_.ok(); }

  // If IsOk() is false, get_status() returns the error message.
  // The returned pointer is valid until the QueryExecutor object is destroyed.
  const char* get_message() const {
    VECTORLITE_ASSERT(!IsOk());
    return absl::StatusMessageAsCStr(status_);
  }

 private:
  bool IsRowidInIndex(hnswlib::labeltype rowid) const;

  const hnswlib::HierarchicalNSW<float>& index_;
  const NamedVectorSpace& space_;
  absl::Status status_;

  // there can at most one KnnParam constraint, at most one "rowid = xx"  but 0
  // or multiple rowid constraints.
  const KnnParam* knn_param_ = nullptr;
  std::vector<const absl::flat_hash_set<hnswlib::labeltype>*> rowid_in_;
  std::optional<hnswlib::labeltype> rowid_equals_;
};

class Constraint {
 public:
  virtual ~Constraint() = default;

  // Some constraints can only get its required data inside xFilter.
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
  KnnSearchConstraint() : knn_param_(nullptr) {}

  void Accept(ConstraintVisitor* visitor) override { visitor->Visit(*this); }

  const KnnParam* get_knn_param() const { return knn_param_; }

  std::string ToDebugString() const override {
    if (materialized()) {
      return absl::StrFormat("knn_parm(vector of dim %d, %d)",
                             knn_param_->query_vector.dim(), knn_param_->k);
    }

    return absl::StrFormat("knn_param: unmaterialized");
  }

 private:
  absl::Status DoMaterialize(const sqlite3_api_routines* sqlite3_api,
                             sqlite3_value* arg) override;
  const KnnParam* knn_param_;
};

class RowIdIn : public Constraint {
 public:
  RowIdIn() : rowids_() {}

  void Accept(ConstraintVisitor* visitor) override { visitor->Visit(*this); }

  const absl::flat_hash_set<hnswlib::labeltype>* get_rowids() const {
    return &rowids_;
  }

 private:
  virtual absl::Status DoMaterialize(const sqlite3_api_routines* sqlite3_api,
                                     sqlite3_value* arg) override;

  std::string ToDebugString() const override {
    if (materialized()) {
      return absl::StrFormat("rowid in (%d rowids...)", rowids_.size());
    }

    return absl::StrFormat("rowid in: unmaterialized");
  }
  absl::flat_hash_set<hnswlib::labeltype> rowids_;
};

class RowIdEquals : public Constraint {
 public:
  explicit RowIdEquals(hnswlib::labeltype rowid) : rowid_(rowid) {}

  void Accept(ConstraintVisitor* visitor) override { visitor->Visit(*this); }

  hnswlib::labeltype rowid() const { return rowid_; }

 private:
  // The right hand side of the equality constraint can be determined in
  // xBestIndex. So we don't need to materialize it in xFilter.
  virtual absl::Status DoMaterialize(const sqlite3_api_routines* sqlite3_api,
                                     sqlite3_value* arg) override {
    return absl::OkStatus();
  }

  std::string ToDebugString() const override {
    return absl::StrFormat("rowid = %d", rowid_);
  }

  hnswlib::labeltype rowid_;
};

std::string ConstraintsToDebugString(
    const std::vector<std::unique_ptr<Constraint>>& constraints);

}  // namespace vectorlite