#include "vector_vtab.h"

#include <charconv>
#include <optional>
#include <string_view>

#include "sqlite3ext.h"

extern const sqlite3_api_routines* sqlite3_api;

namespace sqlite_vector {

static std::optional<size_t> ParseNumber(std::string_view s) {
  size_t value = 0;
  auto result = std::from_chars(s.data(), s.data() + s.size(),value);
  if (result.ec == std::errc::invalid_argument || result.ec == std::errc::result_out_of_range) {
    return std::nullopt;
  }

  return value;
}

int VectorVTable::Create(sqlite3* db, void* pAux, int argc, char* const* argv,
                         sqlite3_vtab** ppVTab, char** pzErr) {
  int rc = sqlite3_vtab_config(db, SQLITE_VTAB_CONSTRAINT_SUPPORT, 1);
  if (rc != SQLITE_OK) {
    return rc;
  }

  if (argc != 2) {
    *pzErr = sqlite3_mprintf("Expected 2 argument, got %d", argc);
    return SQLITE_ERROR;
  }

  rc = sqlite3_declare_vtab(db, "CREATE TABLE vector(distance INTEGER hidden)");
  if (rc != SQLITE_OK) {
    return rc;
  }

  *ppVTab = new VectorVTable(db, 3, 1000000);
  return SQLITE_OK;
}

}  // end namespace sqlite_vector
