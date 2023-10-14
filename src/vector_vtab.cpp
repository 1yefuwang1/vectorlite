#include "vector_vtab.h"
#include <absl/strings/str_cat.h>

#include <charconv>
#include <exception>
#include <optional>
#include <string_view>


#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "macros.h"
#include "sqlite3ext.h"

#include "util.h"

extern const sqlite3_api_routines* sqlite3_api;

namespace sqlite_vector {

static absl::StatusOr<size_t> ParseNumber(std::string_view s) {
  size_t value = 0;
  auto result = std::from_chars(s.data(), s.data() + s.size(), value);
  if (result.ec == std::errc::invalid_argument || result.ec == std::errc::result_out_of_range) {
    return absl::ErrnoToStatus(static_cast<int>(result.ec), absl::StrCat("Failed to parse number: ", s));
  }

  return value;
}

int VectorVTable::Create(sqlite3* db, void* pAux, int argc, char* const* argv,
                         sqlite3_vtab** ppVTab, char** pzErr) {
  int rc = sqlite3_vtab_config(db, SQLITE_VTAB_CONSTRAINT_SUPPORT, 1);
  if (rc != SQLITE_OK) {
    return rc;
  }

  if (argc != 3) {
    *pzErr = sqlite3_mprintf("Expected 3 argument, got %d", argc);
    return SQLITE_ERROR;
  }

  std::string col_name = argv[0];
  if (!IsValidColumnName(col_name)) {
    *pzErr = sqlite3_mprintf("Invalid column name: %s", argv[0]);
    return SQLITE_ERROR;
  }

  auto dim = ParseNumber(argv[1]);
  if (!dim.ok()) {
    *pzErr = sqlite3_mprintf("Invalid dimension %s. Reason: %s", argv[1], absl::StatusMessageAsCStr(dim.status()));
    return SQLITE_ERROR;
  }

  auto max_elements = ParseNumber(argv[2]);
  if (!max_elements.ok()) {
    *pzErr = sqlite3_mprintf("Invalid max_elements: %s. Reason: %s", argv[2], absl::StatusMessageAsCStr(max_elements.status()));
    return SQLITE_ERROR;
  }

  rc = sqlite3_declare_vtab(db, "CREATE TABLE vector(distance INTEGER hidden)");
  if (rc != SQLITE_OK) {
    return rc;
  }

  try {
    *ppVTab = new VectorVTable(col_name, *dim, *max_elements);
  } catch (const std::exception& ex) {
    *pzErr = sqlite3_mprintf("Failed to create virtual table: %s", ex.what());
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

int VectorVTable::Destroy(sqlite3_vtab* pVTab) {
  SQLITE_VECTOR_ASSERT(pVTab != nullptr);
  delete static_cast<VectorVTable*>(pVTab);
  return SQLITE_OK;
}

int VectorVTable::Open(sqlite3_vtab* pVtab, sqlite3_vtab_cursor** ppCursor) {
  SQLITE_VECTOR_ASSERT(pVtab != nullptr);
  SQLITE_VECTOR_ASSERT(ppCursor != nullptr);
  *ppCursor = new Cursor(static_cast<VectorVTable*>(pVtab));
  return SQLITE_OK;
}

int VectorVTable::Close(sqlite3_vtab_cursor* pCursor) {
  SQLITE_VECTOR_ASSERT(pCursor != nullptr);
  delete static_cast<Cursor*>(pCursor);
  return SQLITE_OK;
}

}  // end namespace sqlite_vector
