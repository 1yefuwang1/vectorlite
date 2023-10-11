#include <memory>

#include "hnswlib/hnswalg.h"
#include "hnswlib/hnswlib.h"
#include "hnswlib/space_l2.h"
#include "macros.h"
#include "sqlite3.h"
#include "sqlite3ext.h"
#include "vector.h"

namespace sqlite_vector {

// Note there shouldn't be any virtual functions in this class.
// Because VectorVTable* is expected to be static_cast-ed to sqlite3_vtab*.
class VectorVTable : sqlite3_vtab {
 public:
  VectorVTable(sqlite3* db, size_t dim, size_t max_elements)
      : db_(db),
        space_(std::make_unique<hnswlib::L2Space>(hnswlib::L2Space(dim))),
        index_(std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_.get(), max_elements)) {
    SQLITE_VECTOR_ASSERT(db != nullptr);
    SQLITE_VECTOR_ASSERT(space_ != nullptr);
    SQLITE_VECTOR_ASSERT(index_ != nullptr);
  }

 private:
  sqlite3* db_;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
};

int Create(sqlite3* db, void* pAux, int argc, char* const* argv,
           sqlite3_vtab** ppVTab, char** pzErr);

}  // end namespace sqlite_vector
