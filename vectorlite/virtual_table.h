#pragma once

#include <cstddef>
#include <memory>
#include <set>
#include <string_view>
#include <utility>  // std::pair

#include "absl/status/statusor.h"
#include "hnswlib/hnswlib.h"
#include "index_options.h"
#include "index_registry.h"
#include "macros.h"
#include "sqlite3ext.h"
#include "vector.h"
#include "vector_space.h"

namespace vectorlite {

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

  // `registry` and `handle` are owned by the connection's IndexRegistry, not by
  // this object. `handle` must already live in `registry` under `key`.
  VirtualTable(IndexRegistry* registry, RegistryKey key, IndexHandle* handle)
      : registry_(registry),
        key_(std::move(key)),
        handle_(handle),
        space_(handle->space),
        index_(handle->index),
        allow_replace_deleted_(handle->allow_replace_deleted) {
    VECTORLITE_ASSERT(registry_ != nullptr);
    VECTORLITE_ASSERT(handle_ != nullptr);
    VECTORLITE_ASSERT(space_.space != nullptr);
    VECTORLITE_ASSERT(index_ != nullptr);
  }

  // Serialize the in-memory index to `path`, overwriting any existing file.
  absl::Status SaveTo(const std::string& path);

  // Replace the in-memory index with one loaded from `path`. On any error the
  // current index is left unchanged.
  absl::Status LoadFrom(const std::string& path);

  size_t dimension() const { return space_.dimension(); }

  // Implementation of the virtual table goes below.
  // For more info on what each function does, please check
  // https://www.sqlite.org/vtab.html

  static int Create(sqlite3* db, void* pAux, int argc, const char* const* argv,
                    sqlite3_vtab** ppVTab, char** pzErr);
  static int Destroy(sqlite3_vtab* pVTab);
  static int Connect(sqlite3* db, void* pAux, int argc, const char* const* argv,
                     sqlite3_vtab** ppVTab, char** pzErr);
  static int Disconnect(sqlite3_vtab* pVTab);

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
  int InsertOrUpdateVector(VectorView vector, Cursor::Rowid rowid);
  // Handles an INSERT carrying a non-NULL `operation` column.
  int ExecutePersistenceCommand(sqlite3_value** argv);

  IndexRegistry* registry_;  // not owned
  RegistryKey key_;          // this table's (schema, name)
  IndexHandle* handle_;      // owned by registry_
  // References into *handle_, so existing member-access call sites are
  // unchanged. The handle has a stable address (owned via unique_ptr in the
  // registry map), so these references stay valid until the entry is erased.
  NamedVectorSpace& space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>>& index_;
  bool& allow_replace_deleted_;
};

// Just a marker function that tells BestIndex that this is a vector search
void KnnSearch(sqlite3_context* context, int argc, sqlite3_value** argv);

// Returns a sqlite3 value pointer that points to KnnSearch's second parameter.
// including inpupt vector, k
void KnnParamFunc(sqlite3_context* context, int argc, sqlite3_value** argv);

}  // end namespace vectorlite
