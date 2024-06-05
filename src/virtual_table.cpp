#include "virtual_table.h"

#include <sqlite3.h>

#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "constraint.h"
#include "hnswlib/hnswlib.h"
#include "index_options.h"
#include "macros.h"
#include "sqlite3ext.h"
#include "util.h"
#include "vector_space.h"

extern const sqlite3_api_routines* sqlite3_api;

namespace vectorlite {

enum ColumnIndexInTable {
  kColumnIndexVector,
  kColumnIndexDistance,
};

enum IndexConstraintUsage {
  kVector = 1,
  kRowid,
};

enum FunctionConstraint {
  kFunctionConstraintVectorSearchKnn = SQLITE_INDEX_CONSTRAINT_FUNCTION,
  kFunctionConstraintVectorMatch = SQLITE_INDEX_CONSTRAINT_FUNCTION + 1,
};

// A helper function to reduce boilerplate code when setting zErrMsg.
static void SetZErrMsg(char** pzErr, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  if (*pzErr) {
    sqlite3_free(*pzErr);
  }
  *pzErr = sqlite3_vmprintf(fmt, args);

  va_end(args);
}

int VirtualTable::Create(sqlite3* db, void* pAux, int argc,
                         const char* const* argv, sqlite3_vtab** ppVTab,
                         char** pzErr) {
  int rc = sqlite3_vtab_config(db, SQLITE_VTAB_CONSTRAINT_SUPPORT, 1);
  if (rc != SQLITE_OK) {
    return rc;
  }

  // The first string, argv[0], is the name of the module being invoked. The
  // module name is the name provided as the second argument to
  // sqlite3_create_module() and as the argument to the USING clause of the
  // CREATE VIRTUAL TABLE statement that is running. The second, argv[1], is the
  // name of the database in which the new virtual table is being created. The
  // database name is "main" for the primary database, or "temp" for TEMP
  // database, or the name given at the end of the ATTACH statement for attached
  // databases. The third element of the array, argv[2], is the name of the new
  // virtual table, as specified following the TABLE keyword in the CREATE
  // VIRTUAL TABLE statement. If present, the fourth and subsequent strings in
  // the argv[] array report the arguments to the module name in the CREATE
  // VIRTUAL TABLE statement.
  constexpr int kModuleParamOffset = 3;

  if (argc != 2 + kModuleParamOffset) {
    *pzErr = sqlite3_mprintf("Expected 3 argument, got %d",
                             argc - kModuleParamOffset);
    return SQLITE_ERROR;
  }

  std::string_view vector_space_str = argv[0 + kModuleParamOffset];
  DLOG(INFO) << "vector_space_str: " << vector_space_str;
  auto vector_space = NamedVectorSpace::FromString(vector_space_str);
  if (!vector_space.ok()) {
    *pzErr = sqlite3_mprintf("Invalid vector space: %s. Reason: %s",
                             argv[0 + kModuleParamOffset],
                             absl::StatusMessageAsCStr(vector_space.status()));
    return SQLITE_ERROR;
  }

  std::string_view index_options_str = argv[1 + kModuleParamOffset];
  DLOG(INFO) << "index_options_str: " << index_options_str;
  auto index_options = IndexOptions::FromString(index_options_str);
  if (!index_options.ok()) {
    *pzErr = sqlite3_mprintf("Invalid index_options %s. Reason: %s",
                             argv[1 + kModuleParamOffset],
                             absl::StatusMessageAsCStr(index_options.status()));
    return SQLITE_ERROR;
  }

  std::string sql = absl::StrFormat("CREATE TABLE X(%s, distance REAL hidden)",
                                    vector_space->vector_name);
  rc = sqlite3_declare_vtab(db, sql.c_str());
  DLOG(INFO) << "vtab declared: " << sql.c_str() << ", rc=" << rc;
  if (rc != SQLITE_OK) {
    return rc;
  }

  try {
    *ppVTab = new VirtualTable(std::move(*vector_space), *index_options);
  } catch (const std::exception& ex) {
    *pzErr = sqlite3_mprintf("Failed to create virtual table: %s", ex.what());
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

VirtualTable::~VirtualTable() {
  if (zErrMsg) {
    sqlite3_free(zErrMsg);
  }
}

int VirtualTable::Destroy(sqlite3_vtab* pVTab) {
  VECTORLITE_ASSERT(pVTab != nullptr);
  delete static_cast<VirtualTable*>(pVTab);
  return SQLITE_OK;
}

int VirtualTable::Open(sqlite3_vtab* pVtab, sqlite3_vtab_cursor** ppCursor) {
  DLOG(INFO) << "Open called";
  VECTORLITE_ASSERT(pVtab != nullptr);
  VECTORLITE_ASSERT(ppCursor != nullptr);
  *ppCursor = new Cursor(static_cast<VirtualTable*>(pVtab));
  DLOG(INFO) << "Open end";
  return SQLITE_OK;
}

int VirtualTable::Close(sqlite3_vtab_cursor* pCursor) {
  DLOG(INFO) << "Close called";
  VECTORLITE_ASSERT(pCursor != nullptr);
  delete static_cast<Cursor*>(pCursor);
  return SQLITE_OK;
}

int VirtualTable::Rowid(sqlite3_vtab_cursor* pCur, sqlite_int64* pRowid) {
  DLOG(INFO) << "Rowid called";
  VECTORLITE_ASSERT(pCur != nullptr);
  VECTORLITE_ASSERT(pRowid != nullptr);

  Cursor* cursor = static_cast<Cursor*>(pCur);
  if (cursor->current_row != cursor->result.cend()) {
    *pRowid = cursor->current_row->second;
    return SQLITE_OK;
  } else {
    return SQLITE_ERROR;
  }
}

int VirtualTable::Eof(sqlite3_vtab_cursor* pCur) {
  DLOG(INFO) << "Eof called";
  VECTORLITE_ASSERT(pCur != nullptr);

  Cursor* cursor = static_cast<Cursor*>(pCur);
  return cursor->current_row == cursor->result.cend();
}

int VirtualTable::Next(sqlite3_vtab_cursor* pCur) {
  DLOG(INFO) << "Next called";
  VECTORLITE_ASSERT(pCur != nullptr);

  Cursor* cursor = static_cast<Cursor*>(pCur);
  if (cursor->current_row != cursor->result.cend()) {
    ++cursor->current_row;
  }

  return SQLITE_OK;
}

absl::StatusOr<Vector> VirtualTable::GetVectorByRowid(int64_t rowid) const {
  try {
    // TODO: handle cases where sizeof(rowid) != sizeof(hnswlib::labeltype)
    std::vector<float> vec =
        index_->getDataByLabel<float>(static_cast<hnswlib::labeltype>(rowid));
    VECTORLITE_ASSERT(vec.size() == dimension());
    return Vector(std::move(vec));
  } catch (const std::runtime_error& ex) {
    return absl::Status(absl::StatusCode::kNotFound, ex.what());
  }
}

int VirtualTable::Column(sqlite3_vtab_cursor* pCur, sqlite3_context* pCtx,
                         int N) {
  VECTORLITE_ASSERT(pCur != nullptr);
  VECTORLITE_ASSERT(pCtx != nullptr);
  DLOG(INFO) << "Column called with N=" << N;

  Cursor* cursor = static_cast<Cursor*>(pCur);
  if (cursor->current_row == cursor->result.cend()) {
    return SQLITE_ERROR;
  }

  if (kColumnIndexDistance == N) {
    sqlite3_result_double(pCtx,
                          static_cast<double>(cursor->current_row->first));
    return SQLITE_OK;
  } else if (kColumnIndexVector == N) {
    Cursor::Rowid rowid = cursor->current_row->second;
    VirtualTable* vtab = static_cast<VirtualTable*>(pCur->pVtab);
    auto vector = vtab->GetVectorByRowid(rowid);
    if (vector.ok()) {
      std::string_view blob = vector->ToBlob();
      sqlite3_result_blob(pCtx, blob.data(), blob.size(), SQLITE_TRANSIENT);
      return SQLITE_OK;
    } else {
      std::string err =
          absl::StrFormat("Can't find vector with rowid %d", rowid);
      sqlite3_result_text(pCtx, err.c_str(), err.size(), SQLITE_TRANSIENT);
      return SQLITE_NOTFOUND;
    }
  } else {
    std::string err = absl::StrFormat("Invalid column index: %d", N);
    sqlite3_result_text(pCtx, err.c_str(), err.size(), SQLITE_TRANSIENT);
    return SQLITE_ERROR;
  }
}

// Checks whether the minimum required version of SQLite3 is met.
// If met, returns (version, "")
// If not met, returns version and a human readable explanation.
static std::pair<int, std::string_view> IsMinimumSqlite3VersionMet() {
  int version = sqlite3_libversion_number();
  // Checks whether sqlite3_vtab_in() is available.
  if (version < 3038000) {
    return {version, "sqlite version 3.38.0 or higher is required."};
  }
  return {version, ""};
}

using Constraints = std::vector<std::unique_ptr<Constraint>>;

int VirtualTable::BestIndex(sqlite3_vtab* vtab,
                            sqlite3_index_info* index_info) {
  VECTORLITE_ASSERT(vtab != nullptr);
  // VirtualTable* virtual_table = static_cast<VirtualTable*>(vtab);
  VECTORLITE_ASSERT(index_info != nullptr);

  int argvIndex = 0;

  std::vector<std::string_view> constraint_short_names;
  constraint_short_names.reserve(index_info->nConstraint);

  DLOG(INFO) << "BestIndex called with " << index_info->nConstraint
             << " constraints";

  for (int i = 0; i < index_info->nConstraint; i++) {
    const auto& constraint = index_info->aConstraint[i];
    if (!constraint.usable) {
      DLOG(INFO) << i << "-th constraint is not usable. iColumn: "
                 << constraint.iColumn
                 << ", op: " << static_cast<int>(constraint.op);
      continue;
    }
    int column = constraint.iColumn;
    if (constraint.op == kFunctionConstraintVectorSearchKnn &&
        column == kColumnIndexVector) {
      DLOG(INFO) << "Found knn_search constraint";
      index_info->aConstraintUsage[i].argvIndex = ++argvIndex;
      index_info->aConstraintUsage[i].omit = 1;
      constraint_short_names.push_back(KnnSearchConstraint::kShortName);
      index_info->estimatedCost = 100;
    } else if (column == -1) {
      // in this case the constraint is on rowid
      DLOG(INFO) << "rowid constraint found: "
                 << static_cast<int>(constraint.op);
      auto [version, notMetReason] = IsMinimumSqlite3VersionMet();
      if (!notMetReason.empty()) {
        SetZErrMsg(&vtab->zErrMsg, "SQLite version is too old: %s",
                   notMetReason.data());
        return SQLITE_ERROR;
      }

      DLOG(INFO) << "sqlite3 version check passed: " << version;

      if (constraint.op == SQLITE_INDEX_CONSTRAINT_EQ) {
        // For more details, check https://sqlite.org/c3ref/vtab_in.html
        bool can_be_processed_vtab_in = sqlite3_vtab_in(index_info, i, 1);
        index_info->aConstraintUsage[i].argvIndex = ++argvIndex;
        index_info->aConstraintUsage[i].omit = 1;
        if (can_be_processed_vtab_in) {
          DLOG(INFO) << i << "-th constraint can be processed with vtab in";
          constraint_short_names.push_back(RowIdIn::kShortName);
          index_info->estimatedCost = 200;
        } else {
          DLOG(INFO) << i << "-th constraint cannot be processed with vtab in";
          constraint_short_names.push_back(RowIdEquals::kShortName);
          index_info->estimatedCost = 100;
        }
      }
    } else {
      DLOG(INFO) << "Unknown constraint iColumn=" << column
                 << ", op=" << static_cast<int>(constraint.op);
    }
  }

  DLOG(INFO) << "Picked " << constraint_short_names.size() << " constraints";

  if (constraint_short_names.empty()) {
    SetZErrMsg(&vtab->zErrMsg, "No valid constraint found in where clause");
    return SQLITE_CONSTRAINT;
  }

  std::string index_str = absl::StrJoin(constraint_short_names, "");

  char* p = sqlite3_mprintf("%s", index_str.c_str());

  if (p == nullptr) {
    SetZErrMsg(&vtab->zErrMsg, "Failed to allocate memory for idxStr");
    return SQLITE_NOMEM;
  }

  index_info->idxStr = p;
  index_info->needToFreeIdxStr = 1;
  // idxNum is the length of idxStr
  index_info->idxNum = constraint_short_names.size() * 2;

  return SQLITE_OK;
}

int VirtualTable::Filter(sqlite3_vtab_cursor* pCur, int idxNum,
                         const char* idxStr, int argc, sqlite3_value** argv) {
  DLOG(INFO) << "Filter begins: " << (int*)(idxStr);
  VECTORLITE_ASSERT(pCur != nullptr);
  Cursor* cursor = static_cast<Cursor*>(pCur);

  VECTORLITE_ASSERT(pCur->pVtab != nullptr);
  VirtualTable* vtab = static_cast<VirtualTable*>(pCur->pVtab);

  VECTORLITE_ASSERT(idxStr != nullptr);
  std::string_view index_str(idxStr, idxNum);

  DLOG(INFO) << "Filter called with idxNum=" << idxNum
             << ", idxStr=" << index_str << ", argc=" << argc;

  auto constraints = ParseConstraintsFromShortNames(index_str);

  if (!constraints.ok()) {
    SetZErrMsg(&vtab->zErrMsg, "Failed to parse constraints: %s",
               absl::StatusMessageAsCStr(constraints.status()));
    return SQLITE_ERROR;
  }

  DLOG(INFO) << "constraints: " << ConstraintsToDebugString(*constraints);
  auto executor = QueryExecutor(*vtab->index_, vtab->space_);
  int n = constraints->size();
  for (int i = 0; i < n; i++) {
    auto status = (*constraints)[i]->Materialize(sqlite3_api, argv[i]);
    if (status.ok()) {
      (*constraints)[i]->Accept(&executor);
    } else {
      SetZErrMsg(&vtab->zErrMsg,
                 "Failed to materialize constraint %s due to %s",
                 (*constraints)[i]->ToDebugString().c_str(),
                 absl::StatusMessageAsCStr(status));
      return SQLITE_ERROR;
    }
  }

  DLOG(INFO) << "Materialized constraints: "
             << ConstraintsToDebugString(*constraints);

  if (!executor.ok()) {
    SetZErrMsg(&vtab->zErrMsg, "Failed to execute query due to: %s",
               executor.message());
    return SQLITE_ERROR;
  }

  auto result = executor.Execute();

  if (result.ok()) {
    cursor->result = std::move(*result);
    cursor->current_row = cursor->result.cbegin();
    DLOG(INFO) << "Found " << cursor->result.size() << " rows";
    return SQLITE_OK;
  } else {
    SetZErrMsg(&vtab->zErrMsg, "Failed to execute query due to: %s",
               absl::StatusMessageAsCStr(result.status()));
    return SQLITE_ERROR;
  }
}

// a marker function with empty implementation
void KnnSearch(sqlite3_context* context, int argc, sqlite3_value** argv) {}

void KnnParamDeleter(void* param) {
  KnnParam* p = static_cast<KnnParam*>(param);
  delete p;
}

void KnnParamFunc(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
  if (argc != 2 && argc != 3) {
    sqlite3_result_error(
        ctx, "invalid number of paramters to knn_param(). 2 or 3 is expected",
        -1);
    return;
  }

  if (sqlite3_value_type(argv[0]) != SQLITE_BLOB) {
    sqlite3_result_error(
        ctx, "vector(1st param of knn_param) should be of type Blob", -1);
    return;
  }

  if (sqlite3_value_type(argv[1]) != SQLITE_INTEGER) {
    sqlite3_result_error(
        ctx, "k(2nd param of knn_param) should be of type INTEGER", -1);
    return;
  }

  if (argc == 3 && sqlite3_value_type(argv[2]) != SQLITE_INTEGER) {
    sqlite3_result_error(
        ctx, "ef(3rd param of knn_param) should be of type INTEGER", -1);
    return;
  }

  std::string_view vector_blob(
      reinterpret_cast<const char*>(sqlite3_value_blob(argv[0])),
      sqlite3_value_bytes(argv[0]));
  auto vec = Vector::FromBlob(vector_blob);
  if (!vec.ok()) {
    std::string err = absl::StrFormat("Failed to parse vector due to: %s",
                                      vec.status().message());
    sqlite3_result_error(ctx, err.c_str(), -1);
    return;
  }

  int32_t k = sqlite3_value_int(argv[1]);
  if (k <= 0) {
    sqlite3_result_error(ctx, "k should be greater than 0", -1);
    return;
  }

  std::optional<uint32_t> ef_search;
  if (argc == 3) {
    int32_t ef = sqlite3_value_int(argv[2]);
    if (ef <= 0) {
      sqlite3_result_error(ctx, "ef should be greater than 0", -1);
      return;
    }
    ef_search = ef;
  }

  KnnParam* param = new KnnParam();
  param->query_vector = std::move(*vec);
  param->k = static_cast<uint32_t>(k);
  param->ef_search = std::move(ef_search);

  sqlite3_result_pointer(ctx, param, kKnnParamType.data(), KnnParamDeleter);
  return;
}

int VirtualTable::FindFunction(sqlite3_vtab* pVtab, int nArg, const char* zName,
                               void (**pxFunc)(sqlite3_context*, int,
                                               sqlite3_value**),
                               void** ppArg) {
  VECTORLITE_ASSERT(pVtab != nullptr);
  if (std::string_view(zName) == "knn_search") {
    *pxFunc = KnnSearch;
    *ppArg = nullptr;
    return kFunctionConstraintVectorSearchKnn;
  }

  return 0;
}

constexpr bool IsRowidOutOfRange(sqlite3_int64 rowid) {
  // VirtualTable::Cursor::Rowid is hnswlib::labeltype which is size_t(4 bytes
  // or 8 bytes) Check the invariant here for future proof in case one day it is
  // changed in hnswlib.
  static_assert(sizeof(VirtualTable::Cursor::Rowid) <= sizeof(rowid));

  // VirtualTable::Cursor::Rowid is also 8 bytes
  if constexpr (sizeof(VirtualTable::Cursor::Rowid) == sizeof(rowid)) {
    return rowid < static_cast<sqlite3_int64>(
                       std::numeric_limits<VirtualTable::Cursor::Rowid>::min());
  }
  // VirtualTable::Cursor::Rowid is 4 bytes
  return rowid < static_cast<sqlite3_int64>(
                     std::numeric_limits<VirtualTable::Cursor::Rowid>::min()) ||
         rowid > static_cast<sqlite3_int64>(
                     std::numeric_limits<VirtualTable::Cursor::Rowid>::max());
}

// Only insert is supported for now
int VirtualTable::Update(sqlite3_vtab* pVTab, int argc, sqlite3_value** argv,
                         sqlite_int64* pRowid) {
  VirtualTable* vtab = static_cast<VirtualTable*>(pVTab);
  auto argv0_type = sqlite3_value_type(argv[0]);
  if (argc > 1 && argv0_type == SQLITE_NULL) {
    // Insert with a new row
    if (sqlite3_value_type(argv[1]) == SQLITE_NULL) {
      SetZErrMsg(&vtab->zErrMsg, "rowid must be specified during insertion");
      return SQLITE_ERROR;
    }
    sqlite3_int64 raw_rowid = sqlite3_value_int64(argv[1]);
    // This limitation comes from the fact that rowid is used as the label in
    // hnswlib(hnswlib::labeltype), whose type is size_t which could be 4 or 8
    // bytes depending on the platform and the compiler. But rowid in sqlite3
    // has type int64.
    if (IsRowidOutOfRange(raw_rowid)) {
      SetZErrMsg(&vtab->zErrMsg, "rowid %lld out of range", raw_rowid);
      return SQLITE_ERROR;
    }

    Cursor::Rowid rowid = static_cast<Cursor::Rowid>(raw_rowid);
    *pRowid = rowid;

    if (sqlite3_value_type(argv[2]) != SQLITE_BLOB) {
      SetZErrMsg(&vtab->zErrMsg, "vector must be of type Blob");
      return SQLITE_ERROR;
    }

    auto vector = Vector::FromBlob(std::string_view(
        reinterpret_cast<const char*>(sqlite3_value_blob(argv[2])),
        sqlite3_value_bytes(argv[2])));
    if (vector.ok()) {
      if (vector->dim() != vtab->dimension()) {
        SetZErrMsg(&vtab->zErrMsg,
                   "Dimension mismatch: vector's "
                   "dimension %d, table's "
                   "dimension %d",
                   vector->dim(), vtab->dimension());
        return SQLITE_ERROR;
      }

      try {
        vtab->index_->addPoint(vtab->space_.normalize
                                   ? vector->Normalize().data().data()
                                   : vector->data().data(),
                               rowid, true);

      } catch (const std::runtime_error& e) {
        SetZErrMsg(&vtab->zErrMsg, "Failed to insert row %lld due to: %s",
                   rowid, e.what());
        return SQLITE_ERROR;
      }
      return SQLITE_OK;
    } else {
      SetZErrMsg(&vtab->zErrMsg, "Failed to perform insertion due to: %s",
                 absl::StatusMessageAsCStr(vector.status()));
      return SQLITE_ERROR;
    }
  } else if (argc == 1 && argv0_type != SQLITE_NULL) {
    // Delete a single row
    DLOG(INFO) << "Delete a single row";
    sqlite3_int64 raw_rowid = sqlite3_value_int64(argv[0]);
    if (IsRowidOutOfRange(raw_rowid)) {
      SetZErrMsg(&vtab->zErrMsg, "rowid %lld out of range", raw_rowid);
      return SQLITE_ERROR;
    }
    Cursor::Rowid rowid = static_cast<Cursor::Rowid>(raw_rowid);
    try {
      vtab->index_->markDelete(rowid);
    } catch (const std::runtime_error& ex) {
      SetZErrMsg(&vtab->zErrMsg, "Delete failed with rowid %lld: %s", raw_rowid,
                 ex.what());
      return SQLITE_ERROR;
    }
    return SQLITE_OK;
  } else if (argc > 1 && argv0_type != SQLITE_NULL) {
    DLOG(INFO) << "Update a single row";
    // Update a single row
    if (argv0_type != SQLITE_INTEGER) {
      SetZErrMsg(&vtab->zErrMsg, "rowid must be of type INTEGER");
      return SQLITE_ERROR;
    }
    if (sqlite3_value_type(argv[1]) != SQLITE_INTEGER) {
      SetZErrMsg(&vtab->zErrMsg, "target rowid must be of type INTEGER");
      return SQLITE_ERROR;
    }
    sqlite3_int64 source_rowid = sqlite3_value_int64(argv[0]);
    sqlite3_int64 target_rowid = sqlite3_value_int64(argv[1]);
    if (source_rowid != target_rowid) {
      SetZErrMsg(&vtab->zErrMsg, "rowid cannot be changed");
      return SQLITE_ERROR;
    }

    if (IsRowidOutOfRange(source_rowid)) {
      SetZErrMsg(&vtab->zErrMsg, "rowid %lld out of range", source_rowid);
      return SQLITE_ERROR;
    }

    Cursor::Rowid rowid = static_cast<Cursor::Rowid>(source_rowid);
    if (!IsRowidInIndex(*(vtab->index_), rowid)) {
      SetZErrMsg(&vtab->zErrMsg, "rowid %lld not found", source_rowid);
      return SQLITE_ERROR;
    }

    if (sqlite3_value_type(argv[2]) != SQLITE_BLOB) {
      SetZErrMsg(&vtab->zErrMsg, "vector must be of type Blob");
      return SQLITE_ERROR;
    }
    auto vector = Vector::FromBlob(std::string_view(
        reinterpret_cast<const char*>(sqlite3_value_blob(argv[2])),
        sqlite3_value_bytes(argv[2])));

    if (vector.ok()) {
      if (vector->dim() != vtab->dimension()) {
        SetZErrMsg(&vtab->zErrMsg,
                   "Dimension mismatch: vector's "
                   "dimension %d, table's "
                   "dimension %d",
                   vector->dim(), vtab->dimension());
        return SQLITE_ERROR;
      }

      try {
        vtab->index_->addPoint(vtab->space_.normalize
                                   ? vector->Normalize().data().data()
                                   : vector->data().data(),
                               rowid, vtab->index_->allow_replace_deleted_);

      } catch (const std::runtime_error& e) {
        SetZErrMsg(&vtab->zErrMsg, "Failed to update row %lld due to: %s",
                   rowid, e.what());
        return SQLITE_ERROR;
      }

      return SQLITE_OK;
    } else {
      SetZErrMsg(&vtab->zErrMsg, "Failed to perform row %lld due to: %s", rowid,
                 absl::StatusMessageAsCStr(vector.status()));
      return SQLITE_ERROR;
    }

  } else {
    SetZErrMsg(&vtab->zErrMsg, "Operation not supported for now");
    return SQLITE_ERROR;
  }
}

}  // end namespace vectorlite
