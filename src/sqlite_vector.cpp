#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "macros.h"
#include "sqlite3.h"
#include "sqlite3ext.h"
#include "sqlite_functions.h"
#include "util.h"
#include "vector.h"
#include "vector_space.h"
#include "virtual_table.h"

SQLITE_EXTENSION_INIT1;

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

  rc = sqlite3_create_function(db, "vector_distance", 3, SQLITE_UTF8, nullptr,
                               sqlite_vector::VectorDistance, nullptr, nullptr);

  if (rc != SQLITE_OK) {
    *pzErrMsg = sqlite3_mprintf("Failed to create function vector_distance: %s",
                                sqlite3_errstr(rc));
    return rc;
  }

  rc = sqlite3_create_function(db, "vector_from_json", 1, SQLITE_UTF8, nullptr,
                               sqlite_vector::VectorFromJson, nullptr, nullptr);

  if (rc != SQLITE_OK) {
    *pzErrMsg = sqlite3_mprintf(
        "Failed to create function vector_from_json: %s", sqlite3_errstr(rc));
    return rc;
  }

  rc = sqlite3_create_function(db, "vector_from_msgpack", 1, SQLITE_UTF8,
                               nullptr, sqlite_vector::VectorFromMsgPack,
                               nullptr, nullptr);
  if (rc != SQLITE_OK) {
    *pzErrMsg =
        sqlite3_mprintf("Failed to create function vector_from_msgpack: %s",
                        sqlite3_errstr(rc));
    return rc;
  }

  rc = sqlite3_create_function(db, "knn_search", 2, SQLITE_UTF8, nullptr,
                               sqlite_vector::KnnSearch, nullptr, nullptr);
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

  rc =
      sqlite3_create_function(db, "sqlite_vector_info", 0, SQLITE_UTF8, nullptr,
                              sqlite_vector::ShowInfo, nullptr, nullptr);
  if (rc != SQLITE_OK) {
    *pzErrMsg = sqlite3_mprintf(
        "Failed to create sqlite_vector_info function: %s", sqlite3_errstr(rc));
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