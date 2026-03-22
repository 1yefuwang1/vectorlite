#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <utility>  // std::pair

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hnswlib/hnswlib.h"
#include "index_options.h"
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

  VirtualTable(NamedVectorSpace space, const IndexOptions& options,
               std::string_view file_path, sqlite3* db,
               std::string table_name)
      : space_(std::move(space)),
        index_(std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_.space.get(), options.max_elements, options.M,
            options.ef_construction, options.random_seed,
            options.allow_replace_deleted)),
        file_path_(),
        db_(db),
        table_name_(std::move(table_name)),
        index_options_(options),
        use_shadow_tables_(file_path.empty()) {
    VECTORLITE_ASSERT(space_.space != nullptr);
    VECTORLITE_ASSERT(index_ != nullptr);
    if (!file_path.empty()) {
      // might throw
      file_path_ = file_path;
    }
  }

  // Load index from file_path_.
  absl::Status LoadIndexFromFile();

  absl::Status DeleteIndexFile();

  absl::Status SaveIndexToFile();

  // Initialize storage: for shadow table mode, create/load tables.
  // For file mode, load from file. is_create distinguishes CREATE vs CONNECT.
  absl::Status InitStorage(bool is_create);

  bool use_shadow_tables() const { return use_shadow_tables_; }

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

  // Returns 1 if the given name is a recognized shadow table suffix.
  static int ShadowName(const char* name);

 private:
  absl::StatusOr<Vector> GetVectorByRowid(int64_t rowid) const;
  int InsertOrUpdateVector(VectorView vector, Cursor::Rowid rowid);

  // Execute a command received via the hidden 'command' column.
  int ExecuteCommand(const char* command);

  // Shadow table methods.
  absl::Status CreateShadowTables();
  absl::Status DropShadowTables();

  // Serialize the in-memory index to a byte buffer.
  absl::StatusOr<std::string> SerializeIndex() const;
  // Deserialize a byte buffer into the in-memory index, replacing its state.
  absl::Status DeserializeIndex(const std::string& data);

  // Save serialized index to the _index shadow table (chunked).
  absl::Status SaveIndexToShadowTable();
  // Load and deserialize from the _index shadow table.
  absl::Status LoadIndexFromShadowTable();

  // Append an operation to the _wal shadow table.
  absl::Status AppendToWal(int op, hnswlib::labeltype label, const void* data,
                           size_t data_size);
  // Replay all WAL entries on the current in-memory index.
  absl::Status ReplayWal();

  // Serialize current index and clear WAL (fast).
  absl::Status Compact();
  // Rebuild index without deleted nodes, then compact (slow).
  absl::Status Rebuild();

  // Helper to construct shadow table names.
  std::string IndexTableName() const;
  std::string WalTableName() const;

  NamedVectorSpace space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
  std::filesystem::path file_path_;
  sqlite3* db_ = nullptr;
  std::string table_name_;
  IndexOptions index_options_;
  bool use_shadow_tables_ = false;

  // Cached prepared statement for WAL append.
  sqlite3_stmt* wal_insert_stmt_ = nullptr;
};

// Just a marker function that tells BestIndex that this is a vector search
void KnnSearch(sqlite3_context* context, int argc, sqlite3_value** argv);

// Returns a sqlite3 value pointer that points to KnnSearch's second parameter.
// including inpupt vector, k
void KnnParamFunc(sqlite3_context* context, int argc, sqlite3_value** argv);

}  // end namespace vectorlite
