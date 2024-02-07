#include "absl/strings/str_format.h"
#include "sqlite3.h"

#include <string_view>

#include "macros.h"
#include "sqlite3ext.h"
#include "vector.h"
#include "virtual_table.h"

SQLITE_EXTENSION_INIT1;

// L2distance takes two vector json and outputs their l2distance
static void L2distance(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  if (argc != 2 || (sqlite3_value_type(argv[0]) != SQLITE_TEXT) ||
      (sqlite3_value_type(argv[1]) != SQLITE_TEXT)) {
    sqlite3_result_error(ctx, "Invalid argument", -1);
    return;
  }

  std::string_view json1(
      reinterpret_cast<const char *>(sqlite3_value_text(argv[0])),
      sqlite3_value_bytes(argv[0]));

  auto v1 = sqlite_vector::Vector::FromJSON(json1);
  if (!v1.ok()) {
    std::string err = absl::StrFormat("Failed to parse 1st vector due to: %s", v1.status().message());
    sqlite3_result_error(ctx, err.c_str(), -1);
    return;
  }

  std::string_view json2(
      reinterpret_cast<const char *>(sqlite3_value_text(argv[1])),
      sqlite3_value_bytes(argv[1]));
  auto v2 = sqlite_vector::Vector::FromJSON(json2);
  if (!v2.ok()) {
    std::string err = absl::StrFormat("Failed to parse 2nd vector due to: %s", v2.status().message());
    sqlite3_result_error(ctx, err.c_str(), -1);
    return;
  }

  if (v1->dim() != v2->dim()) {
    std::string err = absl::StrFormat("Dimension mismatch: %d != %d", v1->dim(), v2->dim());
    sqlite3_result_error(ctx, err.c_str(), -1);
    return;
  }

  float distance = sqlite_vector::L2Distance(*v1, *v2);
  sqlite3_result_double(ctx, static_cast<double>(distance));
}

using sqlite_vector::VirtualTable;

static sqlite3_module vector_search_module = {
    /* iVersion    */ 3,
    /* xCreate     */ VirtualTable::Create,
    /* xConnect    */ VirtualTable::Create,
    /* xBestIndex  */ VirtualTable::BestIndex,
    /* xDisconnect */ VirtualTable::Destroy,
    /* xDestroy    */ VirtualTable::Destroy,
    /* xOpen       */ VirtualTable::Open,
    /* xClose      */ VirtualTable::Close,
    /* xFilter     */ VirtualTable::Filter,
    /* xNext       */ VirtualTable::Next,
    /* xEof        */ VirtualTable::Eof,
    /* xColumn     */ VirtualTable::Column,
    /* xRowid      */ VirtualTable::Rowid,
    /* xUpdate     */ VirtualTable::Update,
    /* xBegin      */ 0,
    /* xSync       */ 0,
    /* xCommit     */ 0,
    /* xRollback   */ 0,
    /* xFindFunction */ VirtualTable::FindFunction,
    /* xRename     */ 0,
    /* xSavepoint  */ 0,
    /* xRelease    */ 0,
    /* xRollbackTo */ 0,
    /* xShadowName */ 0};

#ifdef __cplusplus
extern "C" {
#endif

SQLITE_VECTOR_EXPORT int sqlite3_extension_init(
    sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi) {
  int rc = SQLITE_OK;
  SQLITE_EXTENSION_INIT2(pApi);

  rc = sqlite3_create_function(db, "l2distance", 2, SQLITE_UTF8, nullptr,
                               L2distance, nullptr, nullptr);

  if (rc != SQLITE_OK) {
    *pzErrMsg = sqlite3_mprintf("Failed to create l2distance function: %s",
                                sqlite3_errstr(rc));
    return rc;
  }

  rc = sqlite3_create_function(db, "knn_search", 2, SQLITE_UTF8, nullptr,
                               sqlite_vector::KnnSearch, nullptr,
                               nullptr);
  if (rc != SQLITE_OK) {
    *pzErrMsg = sqlite3_mprintf("Failed to create knn_search function: %s",
                                sqlite3_errstr(rc));
    return rc;
  }

  rc = sqlite3_create_function(db, "knn_param", 2, SQLITE_UTF8, nullptr,
                               sqlite_vector::KnnParamFunc, nullptr, nullptr);

  if (rc != SQLITE_OK) {
    *pzErrMsg = sqlite3_mprintf("Failed to create knn_param function: %s",
                                sqlite3_errstr(rc));
    return rc;
  }

  rc = sqlite3_create_module(db, "vector_search", &vector_search_module,
                             nullptr);
  if (rc != SQLITE_OK) {
    *pzErrMsg = sqlite3_mprintf("Failed to create module vector_search: %s",
                                sqlite3_errstr(rc));
    return rc;
  }

  return rc;
}

#ifdef __cplusplus
}  // end extern "C"
#endif