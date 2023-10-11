#include "vector_vtab.h"
#include "sqlite3ext.h"

extern const sqlite3_api_routines* sqlite3_api;

namespace sqlite_vector {

int Create(sqlite3* db, void* pAux, int argc, char* const* argv,
           sqlite3_vtab** ppVTab, char** pzErr) {
  sqlite3_vtab_config(db, SQLITE_VTAB_CONSTRAINT_SUPPORT, 1);

  *ppVTab = new VectorVTable(db, 3, 1000000);
  return SQLITE_OK;
}

}  // end namespace sqlite_vector
