#include <memory>
#include <iostream>
#include <random>
#include <vector>
#include <cassert>

#include "sqlite3.h"

static const int kDim = 16;
static const int kNumVector = 10;
// static const int kNumVector = 100;
static const int kTopK = 10;

static std::vector<std::vector<float>> GenerateRandomVectors() {
  static std::vector<std::vector<float>> data;
  if (data.size() == kNumVector) {
    return data;
  }

  data.reserve(kNumVector);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-100.0f, 100.0f);

  for (int i = 0; i < kNumVector; ++i) {
    std::vector<float> vec;
    vec.reserve(kDim);
    for (int j = 0; j < kDim; ++j) {
      vec.push_back(dis(gen));
    }
    data.push_back(vec);
  }

  std::cout << "random vectors generated" << std::endl;

  return data;
}

int main(int argc, char* argv[]) {
  sqlite3* db;
  char* zErrMsg;
  int rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK) {
    return -1;
  }

  const auto& vectors = GenerateRandomVectors();
  rc = sqlite3_enable_load_extension(db, 1);
  assert(rc == SQLITE_OK);
  rc = sqlite3_load_extension(db, "build/dev/vectorlite.so", "sqlite3_extension_init", &zErrMsg);
  if (rc != SQLITE_OK) {
    std::cerr << "load extension failed: " << zErrMsg << std::endl;
    sqlite3_free(zErrMsg);
    return -1;
  }

  rc = sqlite3_exec(db, "CREATE VIRTUAL TABLE x USING vectorlite(vec(100, \"l2\"), hnsw(max_elements=10000))", nullptr, nullptr, &zErrMsg);
  assert(rc == SQLITE_OK);
  std::cout << "virtual table created" << std::endl;

  int i = 0;
  for (const auto& v : vectors) {
    sqlite3_stmt* stmt;
    rc = sqlite3_prepare(db, "INSERT INTO x(rowid, vec) VALUES(?, ?)", -1, &stmt, nullptr);
    assert(rc == SQLITE_OK);
    
    rc = sqlite3_bind_int64(stmt, 1, i);
    assert(rc == SQLITE_OK);
    rc = sqlite3_bind_blob(stmt, 2, v.data(), kDim * sizeof(float), SQLITE_TRANSIENT);
    assert(rc == SQLITE_OK);

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
      std::cerr << "insert vector failed: " << " errcode: " << rc << std::endl;
      sqlite3_free(zErrMsg);
      return -1;
    }
    sqlite3_finalize(stmt);
    i++;
  }

  rc = sqlite3_exec(db, "select rowid, vector_to_json(vec) from x where rowid = 1", nullptr, nullptr, &zErrMsg);
  assert(rc == SQLITE_OK);
  std::cout << "select 1" << std::endl;
  rc = sqlite3_exec(db, "delete from x where rowid = 1", nullptr, nullptr, &zErrMsg);
  assert(rc == SQLITE_OK);
  std::cout << "delete 1" << std::endl;
  rc = sqlite3_exec(db, "select rowid, vector_to_json(vec) from x where rowid = 1", nullptr, nullptr, &zErrMsg);
  assert(rc == SQLITE_OK);
  std::cout << "select 1 again" << std::endl;

  return 0;
}