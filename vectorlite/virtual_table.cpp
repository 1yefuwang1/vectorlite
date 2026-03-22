#include "virtual_table.h"

#include <sqlite3.h>

#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "constraint.h"
#include "hnswlib/hnswlib.h"
#include "index_options.h"
#include "macros.h"
#include "quantization.h"
#include "sqlite3ext.h"
#include "util.h"
#include "vector.h"
#include "vector_space.h"
#include "vector_view.h"

extern const sqlite3_api_routines* sqlite3_api;

namespace vectorlite {

enum ColumnIndexInTable {
  kColumnIndexVector,
  kColumnIndexDistance,
  kColumnIndexCommand,
};

enum IndexConstraintUsage {
  kVector = 1,
  kRowid,
};

enum FunctionConstraint {
  kFunctionConstraintVectorSearchKnn = SQLITE_INDEX_CONSTRAINT_FUNCTION,
  kFunctionConstraintVectorMatch = SQLITE_INDEX_CONSTRAINT_FUNCTION + 1,
};

// A helper function to reduce boilerplate code when setting zErrMsg.
static void SetZErrMsg(char** pzErr, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  if (*pzErr) {
    sqlite3_free(*pzErr);
  }
  *pzErr = sqlite3_vmprintf(fmt, args);

  va_end(args);
}

// Shared by Create and Connect
static int InitVirtualTable(bool is_create, sqlite3* db, void* pAux,
                            int argc, const char* const* argv,
                            sqlite3_vtab** ppVTab, char** pzErr) {
  int rc = sqlite3_vtab_config(db, SQLITE_VTAB_CONSTRAINT_SUPPORT, 1);
  if (rc != SQLITE_OK) {
    return rc;
  }

  // The first string, argv[0], is the name of the module being invoked. The
  // module name is the name provided as the second argument to
  // sqlite3_create_module() and as the argument to the USING clause of the
  // CREATE VIRTUAL TABLE statement that is running. The second, argv[1], is the
  // name of the database in which the new virtual table is being created. The
  // database name is "main" for the primary database, or "temp" for TEMP
  // database, or the name given at the end of the ATTACH statement for attached
  // databases. The third element of the array, argv[2], is the name of the new
  // virtual table, as specified following the TABLE keyword in the CREATE
  // VIRTUAL TABLE statement. If present, the fourth and subsequent strings in
  // the argv[] array report the arguments to the module name in the CREATE
  // VIRTUAL TABLE statement.
  constexpr int kModuleParamOffset = 3;

  if (argc != 2 + kModuleParamOffset && argc != 3 + kModuleParamOffset) {
    *pzErr = sqlite3_mprintf("vectorlite expects 2 or 3 arguments, got %d",
                             argc - kModuleParamOffset);
    return SQLITE_ERROR;
  }

  std::string_view vector_space_str = argv[0 + kModuleParamOffset];
  DLOG(INFO) << "vector_space_str: " << vector_space_str;
  auto vector_space = NamedVectorSpace::FromString(vector_space_str);
  if (!vector_space.ok()) {
    *pzErr = sqlite3_mprintf("Invalid vector space: %s. Reason: %s",
                             argv[0 + kModuleParamOffset],
                             absl::StatusMessageAsCStr(vector_space.status()));
    return SQLITE_ERROR;
  }

  std::string_view index_options_str = argv[1 + kModuleParamOffset];
  DLOG(INFO) << "index_options_str: " << index_options_str;
  auto index_options = IndexOptions::FromString(index_options_str);
  if (!index_options.ok()) {
    *pzErr = sqlite3_mprintf("Invalid index_options %s. Reason: %s",
                             argv[1 + kModuleParamOffset],
                             absl::StatusMessageAsCStr(index_options.status()));
    return SQLITE_ERROR;
  }

  std::string_view index_file_path;
  if (argc == 3 + kModuleParamOffset) {
    index_file_path = argv[2 + kModuleParamOffset];
    int size = index_file_path.size();
    // Handle cases where the index_file_path is enclosed in double/single
    // quotes. It is necessary for windows paths, because they contain ':', that
    // must be quoted for sqlite to parse correctly.
    if (size > 2) {
      if ((index_file_path[0] == '\"' && index_file_path[size - 1] == '\"') ||
          (index_file_path[0] == '\'' && index_file_path[size - 1] == '\'')) {
        index_file_path = index_file_path.substr(1, size - 2);
      }
    }
  }

  std::string sql = absl::StrFormat(
      "CREATE TABLE X(%s, distance REAL hidden, command TEXT hidden)",
      vector_space->vector_name);
  rc = sqlite3_declare_vtab(db, sql.c_str());
  DLOG(INFO) << "vtab declared: " << sql.c_str() << ", rc=" << rc;
  if (rc != SQLITE_OK) {
    return rc;
  }

  std::string table_name(argv[2]);

  try {
    auto vtab = new VirtualTable(std::move(*vector_space), *index_options,
                                 index_file_path, db, table_name);
    *ppVTab = vtab;

    auto status = vtab->InitStorage(is_create);
    if (!status.ok()) {
      *pzErr = sqlite3_mprintf("Failed to initialize storage: %s",
                               absl::StatusMessageAsCStr(status));
      return SQLITE_ERROR;
    }

  } catch (const std::exception& ex) {
    *pzErr = sqlite3_mprintf("Failed to create virtual table: %s", ex.what());
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

absl::Status VirtualTable::LoadIndexFromFile() {
  VECTORLITE_ASSERT(index_ != nullptr);
  if (!file_path_.empty() && std::filesystem::exists(file_path_)) {
    try {
      index_->loadIndex(file_path_.string(), space_.space.get(),
                        index_->max_elements_);
    } catch (const std::runtime_error& ex) {
      return absl::Status(absl::StatusCode::kInternal, ex.what());
    } catch (const std::exception& ex) {
      return absl::Status(absl::StatusCode::kUnknown, ex.what());
    }
  }

  return absl::OkStatus();
}

absl::Status VirtualTable::DeleteIndexFile() {
  if (!file_path_.empty()) {
    try {
      std::filesystem::remove(file_path_);
    } catch (const std::filesystem::filesystem_error& ex) {
      return absl::Status(absl::StatusCode::kInternal, ex.what());
    }
  }
  return absl::OkStatus();
}

absl::Status VirtualTable::SaveIndexToFile() {
  VECTORLITE_ASSERT(index_ != nullptr);
  if (!file_path_.empty()) {
    try {
      index_->saveIndex(file_path_.string());
    } catch (const std::runtime_error& ex) {
      return absl::Status(absl::StatusCode::kInternal, ex.what());
    }
  }

  return absl::OkStatus();
}

// Maximum BLOB chunk size for shadow table storage (256 MB).
static constexpr size_t kMaxChunkSize = 256 * 1024 * 1024;

std::string VirtualTable::IndexTableName() const {
  return table_name_ + "_index";
}

std::string VirtualTable::WalTableName() const {
  return table_name_ + "_wal";
}

int VirtualTable::ShadowName(const char* name) {
  if (std::strcmp(name, "index") == 0 || std::strcmp(name, "wal") == 0) {
    return 1;
  }
  return 0;
}

absl::Status VirtualTable::CreateShadowTables() {
  VECTORLITE_ASSERT(db_ != nullptr);
  std::string sql = absl::StrFormat(
      "CREATE TABLE IF NOT EXISTS \"%s\"(chunk_id INTEGER PRIMARY KEY, data "
      "BLOB NOT NULL)",
      IndexTableName());
  char* err_msg = nullptr;
  int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_msg);
  if (rc != SQLITE_OK) {
    std::string err = err_msg ? err_msg : "unknown error";
    sqlite3_free(err_msg);
    return absl::InternalError(
        absl::StrCat("Failed to create index shadow table: ", err));
  }

  sql = absl::StrFormat(
      "CREATE TABLE IF NOT EXISTS \"%s\"(seq INTEGER PRIMARY KEY "
      "AUTOINCREMENT, op INTEGER NOT NULL, label INTEGER NOT NULL, vector BLOB)",
      WalTableName());
  rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_msg);
  if (rc != SQLITE_OK) {
    std::string err = err_msg ? err_msg : "unknown error";
    sqlite3_free(err_msg);
    return absl::InternalError(
        absl::StrCat("Failed to create WAL shadow table: ", err));
  }

  return absl::OkStatus();
}

absl::Status VirtualTable::DropShadowTables() {
  VECTORLITE_ASSERT(db_ != nullptr);
  std::string sql =
      absl::StrFormat("DROP TABLE IF EXISTS \"%s\"", IndexTableName());
  char* err_msg = nullptr;
  int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_msg);
  if (rc != SQLITE_OK) {
    std::string err = err_msg ? err_msg : "unknown error";
    sqlite3_free(err_msg);
    return absl::InternalError(
        absl::StrCat("Failed to drop index shadow table: ", err));
  }

  sql = absl::StrFormat("DROP TABLE IF EXISTS \"%s\"", WalTableName());
  rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_msg);
  if (rc != SQLITE_OK) {
    std::string err = err_msg ? err_msg : "unknown error";
    sqlite3_free(err_msg);
    return absl::InternalError(
        absl::StrCat("Failed to drop WAL shadow table: ", err));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::string> VirtualTable::SerializeIndex() const {
  VECTORLITE_ASSERT(index_ != nullptr);
  const auto& idx = *index_;

  std::ostringstream output(std::ios::binary);
  hnswlib::writeBinaryPOD(output, idx.offsetLevel0_);
  hnswlib::writeBinaryPOD(output, idx.max_elements_);
  hnswlib::writeBinaryPOD(output, idx.cur_element_count);
  hnswlib::writeBinaryPOD(output, idx.size_data_per_element_);
  hnswlib::writeBinaryPOD(output, idx.label_offset_);
  hnswlib::writeBinaryPOD(output, idx.offsetData_);
  hnswlib::writeBinaryPOD(output, idx.maxlevel_);
  hnswlib::writeBinaryPOD(output, idx.enterpoint_node_);
  hnswlib::writeBinaryPOD(output, idx.maxM_);
  hnswlib::writeBinaryPOD(output, idx.maxM0_);
  hnswlib::writeBinaryPOD(output, idx.M_);
  hnswlib::writeBinaryPOD(output, idx.mult_);
  hnswlib::writeBinaryPOD(output, idx.ef_construction_);

  size_t count = idx.cur_element_count;
  output.write(idx.data_level0_memory_,
               count * idx.size_data_per_element_);

  for (size_t i = 0; i < count; i++) {
    unsigned int link_list_size =
        idx.element_levels_[i] > 0
            ? idx.size_links_per_element_ * idx.element_levels_[i]
            : 0;
    hnswlib::writeBinaryPOD(output, link_list_size);
    if (link_list_size > 0) {
      output.write(idx.linkLists_[i], link_list_size);
    }
  }

  if (!output.good()) {
    return absl::InternalError("Failed to serialize index to buffer");
  }

  return output.str();
}

absl::Status VirtualTable::DeserializeIndex(const std::string& data) {
  VECTORLITE_ASSERT(index_ != nullptr);

  std::istringstream input(data, std::ios::binary);
  auto& idx = *index_;

  idx.clear();

  hnswlib::readBinaryPOD(input, idx.offsetLevel0_);
  hnswlib::readBinaryPOD(input, idx.max_elements_);
  hnswlib::readBinaryPOD(input, idx.cur_element_count);

  size_t max_elements = idx.max_elements_;
  if (max_elements < idx.cur_element_count) {
    max_elements = idx.cur_element_count;
  }
  idx.max_elements_ = max_elements;

  hnswlib::readBinaryPOD(input, idx.size_data_per_element_);
  hnswlib::readBinaryPOD(input, idx.label_offset_);
  hnswlib::readBinaryPOD(input, idx.offsetData_);
  hnswlib::readBinaryPOD(input, idx.maxlevel_);
  hnswlib::readBinaryPOD(input, idx.enterpoint_node_);
  hnswlib::readBinaryPOD(input, idx.maxM_);
  hnswlib::readBinaryPOD(input, idx.maxM0_);
  hnswlib::readBinaryPOD(input, idx.M_);
  hnswlib::readBinaryPOD(input, idx.mult_);
  hnswlib::readBinaryPOD(input, idx.ef_construction_);

  idx.data_size_ = space_.space->get_data_size();
  idx.fstdistfunc_ = space_.space->get_dist_func();
  idx.dist_func_param_ = space_.space->get_dist_func_param();

  idx.data_level0_memory_ =
      (char*)malloc(max_elements * idx.size_data_per_element_);
  if (idx.data_level0_memory_ == nullptr) {
    return absl::ResourceExhaustedError(
        "Not enough memory to allocate level0 data");
  }
  input.read(idx.data_level0_memory_,
             idx.cur_element_count * idx.size_data_per_element_);

  idx.size_links_per_element_ =
      idx.maxM_ * sizeof(hnswlib::tableint) + sizeof(hnswlib::linklistsizeint);
  idx.size_links_level0_ =
      idx.maxM0_ * sizeof(hnswlib::tableint) + sizeof(hnswlib::linklistsizeint);

  std::vector<std::mutex>(max_elements).swap(idx.link_list_locks_);
  std::vector<std::mutex>(hnswlib::HierarchicalNSW<float>::MAX_LABEL_OPERATION_LOCKS)
      .swap(idx.label_op_locks_);
  idx.visited_list_pool_.reset(
      new hnswlib::VisitedListPool(1, max_elements));

  idx.linkLists_ = (char**)malloc(sizeof(void*) * max_elements);
  if (idx.linkLists_ == nullptr) {
    return absl::ResourceExhaustedError(
        "Not enough memory to allocate linklists");
  }
  idx.element_levels_ = std::vector<int>(max_elements);
  idx.revSize_ = 1.0 / idx.mult_;
  idx.ef_ = 10;

  for (size_t i = 0; i < idx.cur_element_count; i++) {
    idx.label_lookup_[idx.getExternalLabel(i)] = i;
    unsigned int link_list_size;
    hnswlib::readBinaryPOD(input, link_list_size);
    if (link_list_size == 0) {
      idx.element_levels_[i] = 0;
      idx.linkLists_[i] = nullptr;
    } else {
      idx.element_levels_[i] =
          link_list_size / idx.size_links_per_element_;
      idx.linkLists_[i] = (char*)malloc(link_list_size);
      if (idx.linkLists_[i] == nullptr) {
        return absl::ResourceExhaustedError(
            "Not enough memory to allocate linklist");
      }
      input.read(idx.linkLists_[i], link_list_size);
    }
  }

  for (size_t i = 0; i < idx.cur_element_count; i++) {
    if (idx.isMarkedDeleted(i)) {
      idx.num_deleted_ += 1;
      if (idx.allow_replace_deleted_) {
        idx.deleted_elements.insert(i);
      }
    }
  }

  if (!input.good()) {
    return absl::DataLossError("Failed to deserialize index from buffer");
  }

  return absl::OkStatus();
}

absl::Status VirtualTable::SaveIndexToShadowTable() {
  VECTORLITE_ASSERT(db_ != nullptr);

  auto serialized = SerializeIndex();
  if (!serialized.ok()) {
    return serialized.status();
  }

  const std::string& buf = *serialized;

  // Delete existing chunks.
  std::string sql =
      absl::StrFormat("DELETE FROM \"%s\"", IndexTableName());
  char* err_msg = nullptr;
  int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_msg);
  if (rc != SQLITE_OK) {
    std::string err = err_msg ? err_msg : "unknown error";
    sqlite3_free(err_msg);
    return absl::InternalError(
        absl::StrCat("Failed to clear index table: ", err));
  }

  // Insert chunks.
  sql = absl::StrFormat(
      "INSERT INTO \"%s\"(chunk_id, data) VALUES(?, ?)", IndexTableName());
  sqlite3_stmt* stmt = nullptr;
  rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    return absl::InternalError(absl::StrCat(
        "Failed to prepare insert for index table: ", sqlite3_errmsg(db_)));
  }

  size_t offset = 0;
  int chunk_id = 0;
  while (offset < buf.size()) {
    size_t chunk_size = std::min(kMaxChunkSize, buf.size() - offset);

    sqlite3_bind_int(stmt, 1, chunk_id);
    sqlite3_bind_blob(stmt, 2, buf.data() + offset,
                      static_cast<int>(chunk_size), SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
      sqlite3_finalize(stmt);
      return absl::InternalError(absl::StrCat(
          "Failed to insert index chunk: ", sqlite3_errmsg(db_)));
    }
    sqlite3_reset(stmt);

    offset += chunk_size;
    chunk_id++;
  }

  sqlite3_finalize(stmt);
  return absl::OkStatus();
}

absl::Status VirtualTable::LoadIndexFromShadowTable() {
  VECTORLITE_ASSERT(db_ != nullptr);

  std::string sql = absl::StrFormat(
      "SELECT data FROM \"%s\" ORDER BY chunk_id", IndexTableName());
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    return absl::InternalError(absl::StrCat(
        "Failed to prepare select for index table: ", sqlite3_errmsg(db_)));
  }

  std::string buf;
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    const void* blob = sqlite3_column_blob(stmt, 0);
    int blob_size = sqlite3_column_bytes(stmt, 0);
    if (blob != nullptr && blob_size > 0) {
      buf.append(static_cast<const char*>(blob), blob_size);
    }
  }

  sqlite3_finalize(stmt);

  if (rc != SQLITE_DONE) {
    return absl::InternalError(absl::StrCat(
        "Failed to read index chunks: ", sqlite3_errmsg(db_)));
  }

  if (buf.empty()) {
    // No saved index — start with empty index.
    return absl::OkStatus();
  }

  return DeserializeIndex(buf);
}

absl::Status VirtualTable::AppendToWal(int op, hnswlib::labeltype label,
                                       const void* data, size_t data_size) {
  VECTORLITE_ASSERT(db_ != nullptr);

  // Lazily prepare the statement.
  if (wal_insert_stmt_ == nullptr) {
    std::string sql = absl::StrFormat(
        "INSERT INTO \"%s\"(op, label, vector) VALUES(?, ?, ?)",
        WalTableName());
    int rc =
        sqlite3_prepare_v2(db_, sql.c_str(), -1, &wal_insert_stmt_, nullptr);
    if (rc != SQLITE_OK) {
      return absl::InternalError(absl::StrCat(
          "Failed to prepare WAL insert: ", sqlite3_errmsg(db_)));
    }
  }

  sqlite3_bind_int(wal_insert_stmt_, 1, op);
  sqlite3_bind_int64(wal_insert_stmt_, 2, static_cast<sqlite3_int64>(label));
  if (data != nullptr && data_size > 0) {
    sqlite3_bind_blob(wal_insert_stmt_, 3, data,
                      static_cast<int>(data_size), SQLITE_STATIC);
  } else {
    sqlite3_bind_null(wal_insert_stmt_, 3);
  }

  int rc = sqlite3_step(wal_insert_stmt_);
  sqlite3_reset(wal_insert_stmt_);

  if (rc != SQLITE_DONE) {
    return absl::InternalError(
        absl::StrCat("Failed to append to WAL: ", sqlite3_errmsg(db_)));
  }

  return absl::OkStatus();
}

absl::Status VirtualTable::ReplayWal() {
  VECTORLITE_ASSERT(db_ != nullptr);
  VECTORLITE_ASSERT(index_ != nullptr);

  std::string sql = absl::StrFormat(
      "SELECT op, label, vector FROM \"%s\" ORDER BY seq", WalTableName());
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    return absl::InternalError(absl::StrCat(
        "Failed to prepare WAL replay: ", sqlite3_errmsg(db_)));
  }

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    int op = sqlite3_column_int(stmt, 0);
    hnswlib::labeltype label =
        static_cast<hnswlib::labeltype>(sqlite3_column_int64(stmt, 1));

    if (op == 0) {
      // INSERT_OR_UPDATE
      const void* vec_data = sqlite3_column_blob(stmt, 2);
      if (vec_data == nullptr) {
        sqlite3_finalize(stmt);
        return absl::DataLossError("WAL INSERT entry has NULL vector");
      }
      try {
        index_->addPoint(vec_data, label, index_->allow_replace_deleted_);
      } catch (const std::exception& ex) {
        sqlite3_finalize(stmt);
        return absl::InternalError(
            absl::StrCat("WAL replay addPoint failed: ", ex.what()));
      }
    } else if (op == 1) {
      // DELETE
      try {
        index_->markDelete(label);
      } catch (const std::exception& ex) {
        sqlite3_finalize(stmt);
        return absl::InternalError(
            absl::StrCat("WAL replay markDelete failed: ", ex.what()));
      }
    } else {
      sqlite3_finalize(stmt);
      return absl::DataLossError(
          absl::StrFormat("Unknown WAL op: %d", op));
    }
  }

  sqlite3_finalize(stmt);

  if (rc != SQLITE_DONE) {
    return absl::InternalError(absl::StrCat(
        "Failed to replay WAL: ", sqlite3_errmsg(db_)));
  }

  return absl::OkStatus();
}

absl::Status VirtualTable::Compact() {
  VECTORLITE_ASSERT(db_ != nullptr);

  auto status = SaveIndexToShadowTable();
  if (!status.ok()) {
    return status;
  }

  std::string sql =
      absl::StrFormat("DELETE FROM \"%s\"", WalTableName());
  char* err_msg = nullptr;
  int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_msg);
  if (rc != SQLITE_OK) {
    std::string err = err_msg ? err_msg : "unknown error";
    sqlite3_free(err_msg);
    return absl::InternalError(
        absl::StrCat("Failed to clear WAL: ", err));
  }

  return absl::OkStatus();
}

absl::Status VirtualTable::Rebuild() {
  VECTORLITE_ASSERT(index_ != nullptr);

  // Collect all non-deleted vectors.
  struct VectorEntry {
    hnswlib::labeltype label;
    std::vector<char> data;
  };
  std::vector<VectorEntry> entries;

  size_t data_size = index_->data_size_;
  for (size_t i = 0; i < index_->cur_element_count; i++) {
    if (!index_->isMarkedDeleted(i)) {
      VectorEntry entry;
      entry.label = index_->getExternalLabel(i);
      const char* ptr =
          static_cast<const char*>(index_->getDataByInternalId(i));
      entry.data.assign(ptr, ptr + data_size);
      entries.push_back(std::move(entry));
    }
  }

  // Create a new index with the same parameters.
  size_t max_elements = std::max(entries.size(), index_options_.max_elements);
  auto new_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      space_.space.get(), max_elements, index_options_.M,
      index_options_.ef_construction, index_options_.random_seed,
      index_options_.allow_replace_deleted);

  // Re-insert all non-deleted vectors.
  for (const auto& entry : entries) {
    try {
      new_index->addPoint(entry.data.data(), entry.label);
    } catch (const std::exception& ex) {
      return absl::InternalError(
          absl::StrCat("Rebuild addPoint failed: ", ex.what()));
    }
  }

  // Swap in the new index.
  index_ = std::move(new_index);

  // Compact: serialize and clear WAL.
  return Compact();
}

int VirtualTable::ExecuteCommand(const char* command) {
  if (std::strcmp(command, "compact") == 0) {
    auto status = Compact();
    if (!status.ok()) {
      SetZErrMsg(&this->zErrMsg, "Compact failed: %s",
                 absl::StatusMessageAsCStr(status));
      return SQLITE_ERROR;
    }
    return SQLITE_OK;
  } else if (std::strcmp(command, "rebuild") == 0) {
    auto status = Rebuild();
    if (!status.ok()) {
      SetZErrMsg(&this->zErrMsg, "Rebuild failed: %s",
                 absl::StatusMessageAsCStr(status));
      return SQLITE_ERROR;
    }
    return SQLITE_OK;
  } else {
    SetZErrMsg(&this->zErrMsg, "Unknown command: %s", command);
    return SQLITE_ERROR;
  }
}

absl::Status VirtualTable::InitStorage(bool is_create) {
  if (use_shadow_tables_) {
    if (is_create) {
      auto status = CreateShadowTables();
      if (!status.ok()) return status;
      return SaveIndexToShadowTable();
    } else {
      auto status = LoadIndexFromShadowTable();
      if (!status.ok()) return status;
      return ReplayWal();
    }
  } else {
    return LoadIndexFromFile();
  }
}

int VirtualTable::Create(sqlite3* db, void* pAux, int argc,
                         const char* const* argv, sqlite3_vtab** ppVTab,
                         char** pzErr) {
  return InitVirtualTable(true, db, pAux, argc, argv, ppVTab, pzErr);
}

VirtualTable::~VirtualTable() {
  if (wal_insert_stmt_) {
    sqlite3_finalize(wal_insert_stmt_);
    wal_insert_stmt_ = nullptr;
  }
  if (zErrMsg) {
    sqlite3_free(zErrMsg);
  }
}

int VirtualTable::Destroy(sqlite3_vtab* pVTab) {
  DLOG(INFO) << "Destroy called";
  VECTORLITE_ASSERT(pVTab != nullptr);
  VirtualTable* vtab = static_cast<VirtualTable*>(pVTab);
  if (vtab->use_shadow_tables_) {
    auto status = vtab->DropShadowTables();
    if (!status.ok()) {
      SetZErrMsg(&vtab->zErrMsg, "Failed to drop shadow tables: %s",
                 absl::StatusMessageAsCStr(status));
      return SQLITE_ERROR;
    }
  } else {
    auto status = vtab->DeleteIndexFile();
    if (!status.ok()) {
      SetZErrMsg(&vtab->zErrMsg, "Failed to delete index file: %s",
                 absl::StatusMessageAsCStr(status));
      return SQLITE_ERROR;
    }
  }
  delete vtab;
  return SQLITE_OK;
}

int VirtualTable::Connect(sqlite3* db, void* pAux, int argc,
                          const char* const* argv, sqlite3_vtab** ppVTab,
                          char** pzErr) {
  return InitVirtualTable(false, db, pAux, argc, argv, ppVTab, pzErr);
}

int VirtualTable::Disconnect(sqlite3_vtab* pVTab) {
  DLOG(INFO) << "Disconnect called";
  VECTORLITE_ASSERT(pVTab != nullptr);
  VirtualTable* vtab = static_cast<VirtualTable*>(pVTab);
  if (!vtab->use_shadow_tables_) {
    // File-based mode: save on disconnect.
    auto status = vtab->SaveIndexToFile();
    if (!status.ok()) {
      SetZErrMsg(&vtab->zErrMsg, "Failed to save index to file: %s",
                 absl::StatusMessageAsCStr(status));
      return SQLITE_ERROR;
    }
  }
  // Shadow table mode: no save on disconnect. WAL persists.
  delete vtab;
  return SQLITE_OK;
}

int VirtualTable::Open(sqlite3_vtab* pVtab, sqlite3_vtab_cursor** ppCursor) {
  DLOG(INFO) << "Open called";
  VECTORLITE_ASSERT(pVtab != nullptr);
  VECTORLITE_ASSERT(ppCursor != nullptr);
  *ppCursor = new Cursor(static_cast<VirtualTable*>(pVtab));
  DLOG(INFO) << "Open end";
  return SQLITE_OK;
}

int VirtualTable::Close(sqlite3_vtab_cursor* pCursor) {
  DLOG(INFO) << "Close called";
  VECTORLITE_ASSERT(pCursor != nullptr);
  delete static_cast<Cursor*>(pCursor);
  return SQLITE_OK;
}

int VirtualTable::Rowid(sqlite3_vtab_cursor* pCur, sqlite_int64* pRowid) {
  DLOG(INFO) << "Rowid called";
  VECTORLITE_ASSERT(pCur != nullptr);
  VECTORLITE_ASSERT(pRowid != nullptr);

  Cursor* cursor = static_cast<Cursor*>(pCur);
  if (cursor->current_row != cursor->result.cend()) {
    *pRowid = cursor->current_row->second;
    return SQLITE_OK;
  } else {
    return SQLITE_ERROR;
  }
}

int VirtualTable::Eof(sqlite3_vtab_cursor* pCur) {
  DLOG(INFO) << "Eof called";
  VECTORLITE_ASSERT(pCur != nullptr);

  Cursor* cursor = static_cast<Cursor*>(pCur);
  return cursor->current_row == cursor->result.cend();
}

int VirtualTable::Next(sqlite3_vtab_cursor* pCur) {
  DLOG(INFO) << "Next called";
  VECTORLITE_ASSERT(pCur != nullptr);

  Cursor* cursor = static_cast<Cursor*>(pCur);
  if (cursor->current_row != cursor->result.cend()) {
    ++cursor->current_row;
  }

  return SQLITE_OK;
}

absl::StatusOr<Vector> VirtualTable::GetVectorByRowid(int64_t rowid) const {
  try {
    // TODO: handle cases where sizeof(rowid) != sizeof(hnswlib::labeltype)
    std::vector<float> vec =
        index_->getDataByLabel<float>(static_cast<hnswlib::labeltype>(rowid));
    VECTORLITE_ASSERT(vec.size() == dimension());
    return Vector(std::move(vec));
  } catch (const std::runtime_error& ex) {
    return absl::Status(absl::StatusCode::kNotFound, ex.what());
  }
}

int VirtualTable::Column(sqlite3_vtab_cursor* pCur, sqlite3_context* pCtx,
                         int N) {
  VECTORLITE_ASSERT(pCur != nullptr);
  VECTORLITE_ASSERT(pCtx != nullptr);
  DLOG(INFO) << "Column called with N=" << N;

  Cursor* cursor = static_cast<Cursor*>(pCur);
  if (cursor->current_row == cursor->result.cend()) {
    return SQLITE_ERROR;
  }

  if (kColumnIndexDistance == N) {
    sqlite3_result_double(pCtx,
                          static_cast<double>(cursor->current_row->first));
    return SQLITE_OK;
  } else if (kColumnIndexVector == N) {
    Cursor::Rowid rowid = cursor->current_row->second;
    VirtualTable* vtab = static_cast<VirtualTable*>(pCur->pVtab);
    auto vector = vtab->GetVectorByRowid(rowid);
    if (vector.ok()) {
      std::string_view blob = vector->ToBlob();
      sqlite3_result_blob(pCtx, blob.data(), blob.size(), SQLITE_TRANSIENT);
      return SQLITE_OK;
    } else {
      std::string err =
          absl::StrFormat("Can't find vector with rowid %d", rowid);
      sqlite3_result_text(pCtx, err.c_str(), err.size(), SQLITE_TRANSIENT);
      return SQLITE_NOTFOUND;
    }
  } else if (kColumnIndexCommand == N) {
    sqlite3_result_null(pCtx);
    return SQLITE_OK;
  } else {
    std::string err = absl::StrFormat("Invalid column index: %d", N);
    sqlite3_result_text(pCtx, err.c_str(), err.size(), SQLITE_TRANSIENT);
    return SQLITE_ERROR;
  }
}

// Checks whether the minimum required version of SQLite3 is met.
// If met, returns (version, "")
// If not met, returns version and a human readable explanation.
static std::pair<int, std::string_view> IsMinimumSqlite3VersionMet() {
  int version = sqlite3_libversion_number();
  // Checks whether sqlite3_vtab_in() is available.
  if (version < 3038000) {
    return {version, "sqlite version 3.38.0 or higher is required."};
  }
  return {version, ""};
}

using Constraints = std::vector<std::unique_ptr<Constraint>>;

int VirtualTable::BestIndex(sqlite3_vtab* vtab,
                            sqlite3_index_info* index_info) {
  VECTORLITE_ASSERT(vtab != nullptr);
  // VirtualTable* virtual_table = static_cast<VirtualTable*>(vtab);
  VECTORLITE_ASSERT(index_info != nullptr);

  int argvIndex = 0;

  std::vector<std::string_view> constraint_short_names;
  constraint_short_names.reserve(index_info->nConstraint);

  DLOG(INFO) << "BestIndex called with " << index_info->nConstraint
             << " constraints";

  for (int i = 0; i < index_info->nConstraint; i++) {
    const auto& constraint = index_info->aConstraint[i];
    if (!constraint.usable) {
      DLOG(INFO) << i << "-th constraint is not usable. iColumn: "
                 << constraint.iColumn
                 << ", op: " << static_cast<int>(constraint.op);
      continue;
    }
    int column = constraint.iColumn;
    if (constraint.op == kFunctionConstraintVectorSearchKnn &&
        column == kColumnIndexVector) {
      DLOG(INFO) << "Found knn_search constraint";
      index_info->aConstraintUsage[i].argvIndex = ++argvIndex;
      index_info->aConstraintUsage[i].omit = 1;
      constraint_short_names.push_back(KnnSearchConstraint::kShortName);
      index_info->estimatedCost = 100;
    } else if (column == -1) {
      // in this case the constraint is on rowid
      DLOG(INFO) << "rowid constraint found: "
                 << static_cast<int>(constraint.op);
      auto [version, notMetReason] = IsMinimumSqlite3VersionMet();
      if (!notMetReason.empty()) {
        SetZErrMsg(&vtab->zErrMsg, "SQLite version is too old: %s",
                   notMetReason.data());
        return SQLITE_ERROR;
      }

      DLOG(INFO) << "sqlite3 version check passed: " << version;

      if (constraint.op == SQLITE_INDEX_CONSTRAINT_EQ) {
        // For more details, check https://sqlite.org/c3ref/vtab_in.html
        bool can_be_processed_vtab_in = sqlite3_vtab_in(index_info, i, 1);
        index_info->aConstraintUsage[i].argvIndex = ++argvIndex;
        index_info->aConstraintUsage[i].omit = 1;
        if (can_be_processed_vtab_in) {
          DLOG(INFO) << i << "-th constraint can be processed with vtab in";
          constraint_short_names.push_back(RowIdIn::kShortName);
          index_info->estimatedCost = 200;
        } else {
          DLOG(INFO) << i << "-th constraint cannot be processed with vtab in";
          constraint_short_names.push_back(RowIdEquals::kShortName);
          index_info->estimatedCost = 100;
        }
      }
    } else {
      DLOG(INFO) << "Unknown constraint iColumn=" << column
                 << ", op=" << static_cast<int>(constraint.op);
    }
  }

  DLOG(INFO) << "Picked " << constraint_short_names.size() << " constraints";

  if (constraint_short_names.empty()) {
    SetZErrMsg(&vtab->zErrMsg, "No valid constraint found in where clause");
    return SQLITE_CONSTRAINT;
  }

  std::string index_str = absl::StrJoin(constraint_short_names, "");

  char* p = sqlite3_mprintf("%s", index_str.c_str());

  if (p == nullptr) {
    SetZErrMsg(&vtab->zErrMsg, "Failed to allocate memory for idxStr");
    return SQLITE_NOMEM;
  }

  index_info->idxStr = p;
  index_info->needToFreeIdxStr = 1;
  // idxNum is the length of idxStr
  index_info->idxNum = constraint_short_names.size() * 2;

  return SQLITE_OK;
}

int VirtualTable::Filter(sqlite3_vtab_cursor* pCur, int idxNum,
                         const char* idxStr, int argc, sqlite3_value** argv) {
  DLOG(INFO) << "Filter begins: " << (int*)(idxStr);
  VECTORLITE_ASSERT(pCur != nullptr);
  Cursor* cursor = static_cast<Cursor*>(pCur);

  VECTORLITE_ASSERT(pCur->pVtab != nullptr);
  VirtualTable* vtab = static_cast<VirtualTable*>(pCur->pVtab);

  VECTORLITE_ASSERT(idxStr != nullptr);
  std::string_view index_str(idxStr, idxNum);

  DLOG(INFO) << "Filter called with idxNum=" << idxNum
             << ", idxStr=" << index_str << ", argc=" << argc;

  auto constraints = ParseConstraintsFromShortNames(index_str);

  if (!constraints.ok()) {
    SetZErrMsg(&vtab->zErrMsg, "Failed to parse constraints: %s",
               absl::StatusMessageAsCStr(constraints.status()));
    return SQLITE_ERROR;
  }

  DLOG(INFO) << "constraints: " << ConstraintsToDebugString(*constraints);
  auto executor = QueryExecutor(*vtab->index_, vtab->space_);
  int n = constraints->size();
  for (int i = 0; i < n; i++) {
    auto status = (*constraints)[i]->Materialize(sqlite3_api, argv[i]);
    if (status.ok()) {
      (*constraints)[i]->Accept(&executor);
    } else {
      SetZErrMsg(&vtab->zErrMsg,
                 "Failed to materialize constraint %s due to %s",
                 (*constraints)[i]->ToDebugString().c_str(),
                 absl::StatusMessageAsCStr(status));
      return SQLITE_ERROR;
    }
  }

  DLOG(INFO) << "Materialized constraints: "
             << ConstraintsToDebugString(*constraints);

  if (!executor.ok()) {
    SetZErrMsg(&vtab->zErrMsg, "Failed to execute query due to: %s",
               executor.message());
    return SQLITE_ERROR;
  }

  auto result = executor.Execute();

  if (result.ok()) {
    cursor->result = std::move(*result);
    cursor->current_row = cursor->result.cbegin();
    DLOG(INFO) << "Found " << cursor->result.size() << " rows";
    return SQLITE_OK;
  } else {
    SetZErrMsg(&vtab->zErrMsg, "Failed to execute query due to: %s",
               absl::StatusMessageAsCStr(result.status()));
    return SQLITE_ERROR;
  }
}

// a marker function with empty implementation
void KnnSearch(sqlite3_context* context, int argc, sqlite3_value** argv) {}

void KnnParamDeleter(void* param) {
  KnnParam* p = static_cast<KnnParam*>(param);
  delete p;
}

void KnnParamFunc(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
  if (argc != 2 && argc != 3) {
    sqlite3_result_error(
        ctx, "invalid number of paramters to knn_param(). 2 or 3 is expected",
        -1);
    return;
  }

  if (sqlite3_value_type(argv[0]) != SQLITE_BLOB) {
    sqlite3_result_error(
        ctx, "vector(1st param of knn_param) should be of type Blob", -1);
    return;
  }

  if (sqlite3_value_type(argv[1]) != SQLITE_INTEGER) {
    sqlite3_result_error(
        ctx, "k(2nd param of knn_param) should be of type INTEGER", -1);
    return;
  }

  if (argc == 3 && sqlite3_value_type(argv[2]) != SQLITE_INTEGER) {
    sqlite3_result_error(
        ctx, "ef(3rd param of knn_param) should be of type INTEGER", -1);
    return;
  }

  std::string_view vector_blob(
      reinterpret_cast<const char*>(sqlite3_value_blob(argv[0])),
      sqlite3_value_bytes(argv[0]));
  auto vec = VectorView::FromBlob(vector_blob);
  if (!vec.ok()) {
    std::string err = absl::StrFormat("Failed to parse vector due to: %s",
                                      vec.status().message());
    sqlite3_result_error(ctx, err.c_str(), -1);
    return;
  }

  int32_t k = sqlite3_value_int(argv[1]);
  if (k <= 0) {
    sqlite3_result_error(ctx, "k should be greater than 0", -1);
    return;
  }

  std::optional<uint32_t> ef_search;
  if (argc == 3) {
    int32_t ef = sqlite3_value_int(argv[2]);
    if (ef <= 0) {
      sqlite3_result_error(ctx, "ef should be greater than 0", -1);
      return;
    }
    ef_search = ef;
  }

  KnnParam* param = new KnnParam();
  param->query_vector = *vec;
  param->k = static_cast<uint32_t>(k);
  param->ef_search = std::move(ef_search);

  sqlite3_result_pointer(ctx, param, kKnnParamType.data(), KnnParamDeleter);
  return;
}

int VirtualTable::FindFunction(sqlite3_vtab* pVtab, int nArg, const char* zName,
                               void (**pxFunc)(sqlite3_context*, int,
                                               sqlite3_value**),
                               void** ppArg) {
  VECTORLITE_ASSERT(pVtab != nullptr);
  if (std::string_view(zName) == "knn_search") {
    *pxFunc = KnnSearch;
    *ppArg = nullptr;
    return kFunctionConstraintVectorSearchKnn;
  }

  return 0;
}

constexpr bool IsRowidOutOfRange(sqlite3_int64 rowid) {
  // VirtualTable::Cursor::Rowid is hnswlib::labeltype which is size_t(4 bytes
  // or 8 bytes) Check the invariant here for future proof in case one day it is
  // changed in hnswlib.
  static_assert(sizeof(VirtualTable::Cursor::Rowid) <= sizeof(rowid));

  // VirtualTable::Cursor::Rowid is also 8 bytes
  if constexpr (sizeof(VirtualTable::Cursor::Rowid) == sizeof(rowid)) {
    return rowid < static_cast<sqlite3_int64>(
                       std::numeric_limits<VirtualTable::Cursor::Rowid>::min());
  }
  // VirtualTable::Cursor::Rowid is 4 bytes
  return rowid < static_cast<sqlite3_int64>(
                     std::numeric_limits<VirtualTable::Cursor::Rowid>::min()) ||
         rowid > static_cast<sqlite3_int64>(
                     std::numeric_limits<VirtualTable::Cursor::Rowid>::max());
}

int VirtualTable::InsertOrUpdateVector(VectorView vector, Cursor::Rowid rowid) {
  try {
    if (space_.vector_type == vectorlite::VectorType::Float32) {
      if (!space_.normalize) {
        index_->addPoint(vector.data().data(), rowid,
                         index_->allow_replace_deleted_);
      } else {
        Vector normalized_vector = Vector::Normalize(vector);
        index_->addPoint(normalized_vector.data().data(), rowid,
                         index_->allow_replace_deleted_);
      }
    } else if (space_.vector_type == vectorlite::VectorType::BFloat16) {
      BF16Vector bf16_vector = Quantize(vector);
      if (!space_.normalize) {
        index_->addPoint(bf16_vector.data().data(), rowid,
                         index_->allow_replace_deleted_);
      } else {
        BF16Vector normalized_vector = bf16_vector.Normalize();
        index_->addPoint(normalized_vector.data().data(), rowid,
                         index_->allow_replace_deleted_);
      }

    } else if (space_.vector_type == vectorlite::VectorType::Float16) {
      F16Vector f16_vector = QuantizeToF16(vector);
      if (!space_.normalize) {
        index_->addPoint(f16_vector.data().data(), rowid,
                         index_->allow_replace_deleted_);
      } else {
        F16Vector normalized_vector = f16_vector.Normalize();
        index_->addPoint(normalized_vector.data().data(), rowid,
                         index_->allow_replace_deleted_);
      }

    } else {
      SetZErrMsg(&this->zErrMsg, "Unrecognized vector type %d",
                 space_.vector_type);
      return SQLITE_ERROR;
    }

  } catch (const std::runtime_error& e) {
    SetZErrMsg(&this->zErrMsg, "Failed to insert row %lld due to: %s", rowid,
               e.what());
    return SQLITE_ERROR;
  }

  // Append to WAL if in shadow table mode.
  if (use_shadow_tables_) {
    // Read the processed vector data back from the index.
    auto it = index_->label_lookup_.find(rowid);
    if (it != index_->label_lookup_.end()) {
      const void* stored_data = index_->getDataByInternalId(it->second);
      auto status =
          AppendToWal(0, rowid, stored_data, index_->data_size_);
      if (!status.ok()) {
        SetZErrMsg(&this->zErrMsg, "Failed to append to WAL: %s",
                   absl::StatusMessageAsCStr(status));
        return SQLITE_ERROR;
      }
    }
  }

  return SQLITE_OK;
}

int VirtualTable::Update(sqlite3_vtab* pVTab, int argc, sqlite3_value** argv,
                         sqlite_int64* pRowid) {
  VirtualTable* vtab = static_cast<VirtualTable*>(pVTab);
  auto argv0_type = sqlite3_value_type(argv[0]);
  if (argc > 1 && argv0_type == SQLITE_NULL) {
    // Insert with a new row.

    // Check for command-only insert: INSERT INTO vtab(command) VALUES('...')
    // In this case, argv[2] (vector) is NULL and argv[4] (command) is TEXT.
    if (argc > 4 && sqlite3_value_type(argv[2]) == SQLITE_NULL &&
        sqlite3_value_type(argv[4]) == SQLITE_TEXT) {
      const char* command =
          reinterpret_cast<const char*>(sqlite3_value_text(argv[4]));
      return vtab->ExecuteCommand(command);
    }

    if (sqlite3_value_type(argv[1]) == SQLITE_NULL) {
      SetZErrMsg(&vtab->zErrMsg, "rowid must be specified during insertion");
      return SQLITE_ERROR;
    }
    sqlite3_int64 raw_rowid = sqlite3_value_int64(argv[1]);
    // This limitation comes from the fact that rowid is used as the label in
    // hnswlib(hnswlib::labeltype), whose type is size_t which could be 4 or 8
    // bytes depending on the platform and the compiler. But rowid in sqlite3
    // has type int64.
    if (IsRowidOutOfRange(raw_rowid)) {
      SetZErrMsg(&vtab->zErrMsg, "rowid %lld out of range", raw_rowid);
      return SQLITE_ERROR;
    }

    Cursor::Rowid rowid = static_cast<Cursor::Rowid>(raw_rowid);
    *pRowid = rowid;

    if (IsRowidInIndex(*vtab->index_, rowid)) {
      SetZErrMsg(&vtab->zErrMsg, "row %u already exists", rowid);
      return SQLITE_ERROR;
    }

    if (sqlite3_value_type(argv[2]) != SQLITE_BLOB) {
      SetZErrMsg(&vtab->zErrMsg, "vector must be of type Blob");
      return SQLITE_ERROR;
    }

    auto vector = VectorView::FromBlob(std::string_view(
        reinterpret_cast<const char*>(sqlite3_value_blob(argv[2])),
        sqlite3_value_bytes(argv[2])));
    if (vector.ok()) {
      if (vector->dim() != vtab->dimension()) {
        SetZErrMsg(&vtab->zErrMsg,
                   "Dimension mismatch: vector's "
                   "dimension %d, table's "
                   "dimension %d",
                   vector->dim(), vtab->dimension());
        return SQLITE_ERROR;
      }

      return vtab->InsertOrUpdateVector(*vector, rowid);
    } else {
      SetZErrMsg(&vtab->zErrMsg, "Failed to perform insertion due to: %s",
                 absl::StatusMessageAsCStr(vector.status()));
      return SQLITE_ERROR;
    }
  } else if (argc == 1 && argv0_type != SQLITE_NULL) {
    // Delete a single row
    DLOG(INFO) << "Delete a single row";
    sqlite3_int64 raw_rowid = sqlite3_value_int64(argv[0]);
    if (IsRowidOutOfRange(raw_rowid)) {
      SetZErrMsg(&vtab->zErrMsg, "rowid %lld out of range", raw_rowid);
      return SQLITE_ERROR;
    }
    Cursor::Rowid rowid = static_cast<Cursor::Rowid>(raw_rowid);
    try {
      vtab->index_->markDelete(rowid);
    } catch (const std::runtime_error& ex) {
      SetZErrMsg(&vtab->zErrMsg, "Delete failed with rowid %lld: %s", raw_rowid,
                 ex.what());
      return SQLITE_ERROR;
    }
    // Append to WAL if in shadow table mode.
    if (vtab->use_shadow_tables_) {
      auto status = vtab->AppendToWal(1, rowid, nullptr, 0);
      if (!status.ok()) {
        SetZErrMsg(&vtab->zErrMsg, "Failed to append delete to WAL: %s",
                   absl::StatusMessageAsCStr(status));
        return SQLITE_ERROR;
      }
    }
    return SQLITE_OK;
  } else if (argc > 1 && argv0_type != SQLITE_NULL) {
    DLOG(INFO) << "Update a single row";
    // Update a single row
    if (argv0_type != SQLITE_INTEGER) {
      SetZErrMsg(&vtab->zErrMsg, "rowid must be of type INTEGER");
      return SQLITE_ERROR;
    }
    if (sqlite3_value_type(argv[1]) != SQLITE_INTEGER) {
      SetZErrMsg(&vtab->zErrMsg, "target rowid must be of type INTEGER");
      return SQLITE_ERROR;
    }
    sqlite3_int64 source_rowid = sqlite3_value_int64(argv[0]);
    sqlite3_int64 target_rowid = sqlite3_value_int64(argv[1]);
    if (source_rowid != target_rowid) {
      SetZErrMsg(&vtab->zErrMsg, "rowid cannot be changed");
      return SQLITE_ERROR;
    }

    if (IsRowidOutOfRange(source_rowid)) {
      SetZErrMsg(&vtab->zErrMsg, "rowid %lld out of range", source_rowid);
      return SQLITE_ERROR;
    }

    Cursor::Rowid rowid = static_cast<Cursor::Rowid>(source_rowid);
    if (!IsRowidInIndex(*(vtab->index_), rowid)) {
      SetZErrMsg(&vtab->zErrMsg, "rowid %lld not found", source_rowid);
      return SQLITE_ERROR;
    }

    if (sqlite3_value_type(argv[2]) != SQLITE_BLOB) {
      SetZErrMsg(&vtab->zErrMsg, "vector must be of type Blob");
      return SQLITE_ERROR;
    }
    auto vector = VectorView::FromBlob(std::string_view(
        reinterpret_cast<const char*>(sqlite3_value_blob(argv[2])),
        sqlite3_value_bytes(argv[2])));

    if (vector.ok()) {
      if (vector->dim() != vtab->dimension()) {
        SetZErrMsg(&vtab->zErrMsg,
                   "Dimension mismatch: vector's "
                   "dimension %d, table's "
                   "dimension %d",
                   vector->dim(), vtab->dimension());
        return SQLITE_ERROR;
      }

      return vtab->InsertOrUpdateVector(*vector, rowid);
    } else {
      SetZErrMsg(&vtab->zErrMsg, "Failed to perform row %lld due to: %s", rowid,
                 absl::StatusMessageAsCStr(vector.status()));
      return SQLITE_ERROR;
    }

  } else {
    SetZErrMsg(&vtab->zErrMsg, "Operation not supported for now");
    return SQLITE_ERROR;
  }
}

}  // end namespace vectorlite
