#pragma once

#include <cstddef>
#include <memory>
#include <set>
#include <utility>  // std::pair

#include "absl/status/statusor.h"
#include "hnswlib/hnswlib.h"
#include "index_options.h"
#include "macros.h"
#include "sqlite3ext.h"
#include "vector.h"
#include "vector_space.h"

namespace vectorlite {

// Note there shouldn't be any virtual functions in this class.
// Because VirtualTable* is expected to be static_cast-ed to sqlite3_vtab*.
// vptr could cause UB.
class VirtualTable : public sqlite3_vtab {
 public:
  // No virtual function
  struct Cursor : public sqlite3_vtab_cursor {
    using Distance = float;
    // A rowid is used as a label in hnswlib index to identify a vector.
    // rowid in sqlite is of type int64_t, size_t is used here to align with
    // hnswlib's labeltype, so that a query result can be directed moved instead
    // of copied.
    static_assert(sizeof(size_t) == sizeof(hnswlib::labeltype));
    using Rowid = size_t;
    using ResultSet = std::vector<std::pair<Distance, Rowid>>;
    using ResultSetIter =
        std::vector<std::pair<Distance, Rowid>>::const_iterator;

    Cursor(VirtualTable* vtab)
        : result(), current_row(result.cend()), query_vector() {
      VECTORLITE_ASSERT(vtab != nullptr);
      pVtab = vtab;
    }

    ResultSet result;           // result rowid set, pair is (distance, rowid)
    ResultSetIter current_row;  // points to current row
    Vector query_vector;        // query vector
  };

  ~VirtualTable();

  VirtualTable(NamedVectorSpace space, const IndexOptions& options)
      : space_(std::move(space)),
        index_(std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_.space.get(), options.max_elements, options.M,
            options.ef_construction, options.random_seed,
            options.allow_replace_deleted)) {
    VECTORLITE_ASSERT(space_.space != nullptr);
    VECTORLITE_ASSERT(index_ != nullptr);
  }

  size_t dimension() const { return space_.dimension(); }

  // Implementation of the virtual table goes below.
  // For more info on what each function does, please check
  // https://www.sqlite.org/vtab.html

  static int Create(sqlite3* db, void* pAux, int argc, const char* const* argv,
                    sqlite3_vtab** ppVTab, char** pzErr);
  static int Destroy(sqlite3_vtab* pVTab);
  // Connect and Disconnect are not implemented because right now the vtab is
  // memory-only. So xCreate and xConnect points to the same function. So are
  // xDestroy and xDisconnect.
  /*
  static int Connect(sqlite3*, void* pAux, int argc, char* const* argv,
                     sqlite3_vtab** ppVTab, char** pzErr);
  static int Disconnect(sqlite3_vtab* pVTab);
  */

  static int BestIndex(sqlite3_vtab* pVTab, sqlite3_index_info*);
  static int Open(sqlite3_vtab* pVtab, sqlite3_vtab_cursor** ppCursor);
  static int Close(sqlite3_vtab_cursor* pCur);
  static int Eof(sqlite3_vtab_cursor* pCur);
  static int Filter(sqlite3_vtab_cursor*, int idxNum, const char* idxStr,
                    int argc, sqlite3_value** argv);
  static int Next(sqlite3_vtab_cursor* pCur);
  static int Column(sqlite3_vtab_cursor* pCur, sqlite3_context* pCtx, int N);
  static int Rowid(sqlite3_vtab_cursor* pCur, sqlite_int64* pRowid);
  static int Update(sqlite3_vtab* pVTab, int argc, sqlite3_value** argv,
                    sqlite_int64* pRowid);
  static int FindFunction(sqlite3_vtab* pVtab, int nArg, const char* zName,
                          void (**pxFunc)(sqlite3_context*, int,
                                          sqlite3_value**),
                          void** ppArg);

 private:
  absl::StatusOr<Vector> GetVectorByRowid(int64_t rowid) const;

  NamedVectorSpace space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
};

// Just a marker function that tells BestIndex that this is a vector search
void KnnSearch(sqlite3_context* context, int argc, sqlite3_value** argv);

// Returns a sqlite3 value pointer that points to KnnSearch's second parameter.
// including inpupt vector, k
void KnnParamFunc(sqlite3_context* context, int argc, sqlite3_value** argv);

}  // end namespace vectorlite
