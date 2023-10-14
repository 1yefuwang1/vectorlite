#include <memory>
#include <set>

#include "hnswlib/hnswlib.h"
#include "macros.h"
#include "sqlite3ext.h"
#include "vector.h"

namespace sqlite_vector {

// Note there shouldn't be any virtual functions in this class.
// Because VectorVTable* is expected to be static_cast-ed to sqlite3_vtab*.
class VectorVTable : public sqlite3_vtab {
 public:
  class Cursor : public sqlite3_vtab_cursor {
   public:
    Cursor(VectorVTable* vtab) : rowid_(0) {
      SQLITE_VECTOR_ASSERT(vtab != nullptr);
      pVtab = vtab;
    }

   private:
    uint64_t rowid_;
  };

  VectorVTable(std::string_view col_name, size_t dim, size_t max_elements)
      : col_name_(col_name),
        space_(std::make_unique<hnswlib::L2Space>(dim)),
        index_(std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_.get(), max_elements)) {
    SQLITE_VECTOR_ASSERT(space_ != nullptr);
    SQLITE_VECTOR_ASSERT(index_ != nullptr);
  }

  int dimension() const { return space_->get_data_size(); }

  // Implementation of the virtual table goes below.
  // For more info on what each function does, please check
  // https://www.sqlite.org/vtab.html

  static int Create(sqlite3* db, void* pAux, int argc, char* const* argv,
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
  static int Close(sqlite3_vtab_cursor* cur);
  static int Eof(sqlite3_vtab_cursor* cur);
  static int Filter(sqlite3_vtab_cursor*, int idxNum, const char* idxStr,
                    int argc, sqlite3_value** argv);
  static int Next(sqlite3_vtab_cursor*);
  static int Column(sqlite3_vtab_cursor*, sqlite3_context*, int N);
  static int Rowid(sqlite3_vtab_cursor* pCur, sqlite_int64* pRowid);
  static int Update(sqlite3_vtab* pVTab, int argc, sqlite3_value** argv,
                    sqlite_int64* pRowid);
  static int xFindFunction(sqlite3_vtab* pVtab, int nArg, const char* zName,
                           void (**pxFunc)(sqlite3_context*, int,
                                           sqlite3_value**),
                           void** ppArg);

 private:
  std::string col_name_;  // vector column name
  std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
  std::set<int64_t> rowids_;
};

}  // end namespace sqlite_vector
