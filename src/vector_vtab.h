#include <memory>
#include <set>

#include "hnswlib/hnswlib.h"
#include "macros.h"
#include "sqlite3ext.h"
#include "vector.h"

namespace sqlite_vector {

struct VTableColumn {
  std::string name;
};

// Note there shouldn't be any virtual functions in this class.
// Because VectorVTable* is expected to be static_cast-ed to sqlite3_vtab*.
class VectorVTable : public sqlite3_vtab {
 public:
  VectorVTable(sqlite3* db, size_t dim, size_t max_elements)
      : space_(std::make_unique<hnswlib::L2Space>(hnswlib::L2Space(dim))),
        index_(std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_.get(), max_elements)) {
    SQLITE_VECTOR_ASSERT(db != nullptr);
    SQLITE_VECTOR_ASSERT(space_ != nullptr);
    SQLITE_VECTOR_ASSERT(index_ != nullptr);
  }

  int dimension() const { return space_->get_data_size(); }

  static int Create(sqlite3* db, void* pAux, int argc, char* const* argv,
                    sqlite3_vtab** ppVTab, char** pzErr);

 private:
  std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
  std::set<int64_t> rowids;
};

}  // end namespace sqlite_vector
