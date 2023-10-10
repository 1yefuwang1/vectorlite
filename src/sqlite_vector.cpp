#include <sqlite3.h>
#include <string_view>

#include "macros.h"
#include "sqlite3ext.h"
#include "vector.h"

SQLITE_EXTENSION_INIT1;
/* Hello World function */
static void l2distance(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  if (argc != 2 || (sqlite3_value_type(argv[0]) != SQLITE_TEXT) ||
      (sqlite3_value_type(argv[1]) != SQLITE_TEXT)) {
    sqlite3_result_error(ctx, "Invalid argument", -1);
  }

  std::string_view json1(
      reinterpret_cast<const char *>(sqlite3_value_text(argv[0])),
      sqlite3_value_bytes(argv[0]));

  sqlite_vector::Vector v1;
  auto parse_result = sqlite_vector::Vector::FromJSON(json1, &v1);
  if (parse_result != sqlite_vector::Vector::ParseResult::kOk) {
    sqlite3_result_error(ctx, "Failed to parse JSON", -1);
    return;
  }
  
  std::string_view json2(
      reinterpret_cast<const char *>(sqlite3_value_text(argv[1])),
      sqlite3_value_bytes(argv[1]));
  sqlite_vector::Vector v2;
  parse_result = sqlite_vector::Vector::FromJSON(json2, &v2);
  if (parse_result != sqlite_vector::Vector::ParseResult::kOk) {
    sqlite3_result_error(ctx, "Failed to parse JSON", -1);
    return;
  }

  if (v1.get_dim() != v2.get_dim()) {
    sqlite3_result_error(ctx, "Dimension mismatch", -1);
    return;
  }

  float distance = sqlite_vector::L2Distance(v1, v2);
  sqlite3_result_double(ctx, static_cast<double>(distance));
}

#ifdef __cplusplus
extern "C" {
#endif

SQLITE_VECTOR_EXPORT int sqlite3_extension_init(
    sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi) {
  int rc = SQLITE_OK;
  SQLITE_EXTENSION_INIT2(pApi);

  rc = sqlite3_create_function(db, "l2distance", 2, SQLITE_UTF8, nullptr, l2distance,
                               nullptr, nullptr);

  if (rc != SQLITE_OK) {
    *pzErrMsg = sqlite3_mprintf("Failed to create hello_world function: %s",
                                sqlite3_errstr(rc));
  }

  return rc;
}

#ifdef __cplusplus
}  // end extern "C"
#endif