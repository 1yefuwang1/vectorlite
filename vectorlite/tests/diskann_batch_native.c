/* Native host regression tests for tagged-pointer DiskANN batch INSERT. */
#include <sqlite3.h>
#include "vectorlite_batch.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(condition) do { \
  if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    exit(1); \
  } \
} while (0)

_Static_assert(sizeof(BatchF32V1) == 40, "batch ABI size");
_Static_assert(offsetof(BatchF32V1, abi_version) == 0, "batch ABI version offset");
_Static_assert(offsetof(BatchF32V1, struct_size) == 4, "batch ABI size offset");
_Static_assert(offsetof(BatchF32V1, count) == 8, "batch ABI count offset");
_Static_assert(offsetof(BatchF32V1, dimension) == 16, "batch ABI dimension offset");
_Static_assert(offsetof(BatchF32V1, vectors) == 24, "batch ABI vectors offset");
_Static_assert(offsetof(BatchF32V1, rowids) == 32, "batch ABI rowids offset");

static int checks = 0;

static void execute(sqlite3 *db, const char *sql) {
  char *error = NULL;
  int rc = sqlite3_exec(db, sql, NULL, NULL, &error);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "SQL failed (%d): %s\n%s\n", rc,
            error ? error : sqlite3_errmsg(db), sql);
    sqlite3_free(error);
    exit(1);
  }
}

static sqlite3 *open_host(const char *extension, const char *database) {
  sqlite3 *db = NULL;
  REQUIRE(sqlite3_open(database, &db) == SQLITE_OK);
  REQUIRE(sqlite3_extended_result_codes(db, 1) == SQLITE_OK);
  REQUIRE(sqlite3_enable_load_extension(db, 1) == SQLITE_OK);
  char *error = NULL;
  int rc = sqlite3_load_extension(db, extension, NULL, &error);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "Cannot load extension (%d): %s\n", rc,
            error ? error : sqlite3_errmsg(db));
    sqlite3_free(error);
    exit(1);
  }
  REQUIRE(sqlite3_enable_load_extension(db, 0) == SQLITE_OK);
  return db;
}

static void create_index(sqlite3 *db, const char *metric) {
  char sql[512];
  snprintf(sql, sizeof(sql),
           "CREATE VIRTUAL TABLE v USING vectorlite(embedding float32[2] %s,"
           "diskann(degree=8,build_list_size=32,search_list_size=32,max_visits=65536))",
           metric);
  execute(db, sql);
}

static sqlite3_int64 integer(sqlite3 *db, const char *sql) {
  sqlite3_stmt *stmt = NULL;
  REQUIRE(sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK);
  REQUIRE(sqlite3_step(stmt) == SQLITE_ROW);
  sqlite3_int64 value = sqlite3_column_int64(stmt, 0);
  REQUIRE(sqlite3_step(stmt) == SQLITE_DONE);
  REQUIRE(sqlite3_finalize(stmt) == SQLITE_OK);
  return value;
}

static BatchF32V1 descriptor(const float *vectors, const int64_t *rowids,
                             uint64_t count) {
  BatchF32V1 batch = {VECTORLITE_BATCH_F32_V1_ABI_VERSION,
                     (uint32_t)sizeof(BatchF32V1), count, 2, vectors, rowids};
  return batch;
}

static int run_batch(sqlite3 *db, const char *sql, const BatchF32V1 *batch,
                     const char *tag, const char *options,
                     char *error, size_t error_bytes) {
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
  if (rc == SQLITE_OK) {
    rc = sqlite3_bind_pointer(stmt, 1, (void *)batch, tag, NULL);
  }
  if (rc == SQLITE_OK && sqlite3_bind_parameter_count(stmt) >= 2) {
    rc = options ? sqlite3_bind_text(stmt, 2, options, -1, SQLITE_TRANSIENT)
                 : sqlite3_bind_null(stmt, 2);
  }
  if (rc == SQLITE_OK) {
    do { rc = sqlite3_step(stmt); } while (rc == SQLITE_ROW);
    if (rc == SQLITE_DONE) rc = SQLITE_OK;
  }
  if (error && error_bytes) {
    snprintf(error, error_bytes, "%s", sqlite3_errmsg(db));
  }
  int finish = sqlite3_finalize(stmt);
  if (rc == SQLITE_OK && finish != SQLITE_OK) rc = finish;
  return rc;
}

static int batch_insert(sqlite3 *db, const BatchF32V1 *batch,
                        const char *options) {
  return run_batch(db,
      "INSERT INTO v(operation,embedding,path) VALUES('insert_batch',?1,?2)",
      batch, VECTORLITE_BATCH_F32_V1_TAG, options, NULL, 0);
}

/* Store byte-exact shadow snapshots in private ordinary tables. */
static void save_snapshot(sqlite3 *db) {
  execute(db, "DROP TABLE IF EXISTS snap_meta;DROP TABLE IF EXISTS snap_nodes;"
              "DROP TABLE IF EXISTS snap_txn;DROP TABLE IF EXISTS snap_rebuild;"
              "CREATE TABLE snap_meta AS SELECT * FROM v_diskann_meta;"
              "CREATE TABLE snap_nodes AS SELECT * FROM v_diskann_nodes;"
              "CREATE TABLE snap_txn AS SELECT * FROM v_diskann_txn;"
              "CREATE TABLE snap_rebuild AS SELECT * FROM v_diskann_rebuild;");
}

static void assert_snapshot(sqlite3 *db) {
  const char *suffixes[] = {"meta", "nodes", "txn", "rebuild"};
  for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); ++i) {
    char sql[512];
    snprintf(sql, sizeof(sql),
        "SELECT (SELECT count(*) FROM (SELECT * FROM v_diskann_%s EXCEPT SELECT * FROM snap_%s))"
        "+(SELECT count(*) FROM (SELECT * FROM snap_%s EXCEPT SELECT * FROM v_diskann_%s))",
        suffixes[i], suffixes[i], suffixes[i], suffixes[i]);
    REQUIRE(integer(db, sql) == 0);
  }
}

static void read_vector(sqlite3 *db, int64_t rowid, float *vector) {
  sqlite3_stmt *stmt = NULL;
  REQUIRE(sqlite3_prepare_v2(db, "SELECT embedding FROM v WHERE rowid=?1", -1,
                             &stmt, NULL) == SQLITE_OK);
  REQUIRE(sqlite3_bind_int64(stmt, 1, rowid) == SQLITE_OK);
  REQUIRE(sqlite3_step(stmt) == SQLITE_ROW);
  REQUIRE(sqlite3_column_type(stmt, 0) == SQLITE_BLOB);
  REQUIRE(sqlite3_column_bytes(stmt, 0) == 2 * (int)sizeof(float));
  memcpy(vector, sqlite3_column_blob(stmt, 0), 2 * sizeof(float));
  REQUIRE(sqlite3_step(stmt) == SQLITE_DONE);
  REQUIRE(sqlite3_finalize(stmt) == SQLITE_OK);
}

static int64_t nearest(sqlite3 *db, const float *query, float *distance) {
  sqlite3_stmt *stmt = NULL;
  REQUIRE(sqlite3_prepare_v2(db,
      "SELECT rowid,distance FROM v WHERE knn_search(embedding,knn_param(?1,1))",
      -1, &stmt, NULL) == SQLITE_OK);
  REQUIRE(sqlite3_bind_blob(stmt, 1, query, 2 * sizeof(float), SQLITE_TRANSIENT) == SQLITE_OK);
  REQUIRE(sqlite3_step(stmt) == SQLITE_ROW);
  int64_t rowid = sqlite3_column_int64(stmt, 0);
  *distance = (float)sqlite3_column_double(stmt, 1);
  REQUIRE(sqlite3_step(stmt) == SQLITE_DONE);
  REQUIRE(sqlite3_finalize(stmt) == SQLITE_OK);
  return rowid;
}

static void generate(float *vectors, int64_t *rowids, size_t count, int64_t base) {
  for (size_t i = 0; i < count; ++i) {
    rowids[i] = base + (int64_t)i;
    vectors[2 * i] = (float)i + 1.0f;
    vectors[2 * i + 1] = (float)((i * 7) % 11) * 0.1f;
  }
}

static void test_insert_readback_query(const char *extension) {
  sqlite3 *db = open_host(extension, ":memory:");
  create_index(db, "l2");
  float vectors[130];
  int64_t rowids[65];
  generate(vectors, rowids, 65, 1000);
  BatchF32V1 batch = descriptor(vectors, rowids, 65);
  REQUIRE(batch_insert(db, &batch, NULL) == SQLITE_OK);
  REQUIRE(integer(db, "SELECT live_count FROM v_diskann_meta") == 65);
  REQUIRE(integer(db, "SELECT deleted_count FROM v_diskann_meta") == 0);
  REQUIRE(integer(db, "SELECT count(*) FROM v_diskann_nodes WHERE state=0") == 65);
  REQUIRE(integer(db, "SELECT count(*) FROM v_diskann_txn WHERE value=0") == 2);
  for (size_t i = 0; i < 65; ++i) {
    float readback[2], distance = -1.0f;
    read_vector(db, rowids[i], readback);
    REQUIRE(memcmp(readback, vectors + 2 * i, sizeof(readback)) == 0);
    REQUIRE(nearest(db, vectors + 2 * i, &distance) == rowids[i]);
    REQUIRE(fabsf(distance) < 1e-4f);
  }
  execute(db, "DELETE FROM v WHERE rowid=1000;"
              "INSERT INTO v(operation) VALUES('consolidate')");
  REQUIRE(integer(db, "SELECT live_count FROM v_diskann_meta") == 64);
  REQUIRE(integer(db, "SELECT deleted_count FROM v_diskann_meta") == 0);
  REQUIRE(sqlite3_close(db) == SQLITE_OK);
  ++checks;
}

static void test_chunk_sizes(const char *extension) {
  const char *options[] = {"{\"batch_size\":1}", "{\"batch_size\":8}",
                           "{\"batch_size\":32}"};
  for (size_t option = 0; option < 3; ++option) {
    sqlite3 *db = open_host(extension, ":memory:");
    create_index(db, "l2");
    float vectors[34]; int64_t rowids[17];
    generate(vectors, rowids, 17, 0);
    BatchF32V1 batch = descriptor(vectors, rowids, 17);
    REQUIRE(batch_insert(db, &batch, options[option]) == SQLITE_OK);
    REQUIRE(integer(db, "SELECT live_count FROM v_diskann_meta") == 17);
    REQUIRE(integer(db, "SELECT count(*) FROM v_diskann_nodes WHERE public_rowid=0 AND state=0") == 1);
    REQUIRE(sqlite3_close(db) == SQLITE_OK);
    ++checks;
  }
}

static void test_cosine_and_zero(const char *extension) {
  sqlite3 *db = open_host(extension, ":memory:");
  create_index(db, "cosine");
  const float vectors[] = {3.0f, 4.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 2.0f};
  const int64_t rowids[] = {5, 7, 9, 11};
  BatchF32V1 batch = descriptor(vectors, rowids, 4);
  REQUIRE(batch_insert(db, &batch, "{\"batch_size\":8}") == SQLITE_OK);
  float vector[2];
  read_vector(db, 5, vector);
  REQUIRE(fabsf(vector[0] - 0.6f) < 1e-6f && fabsf(vector[1] - 0.8f) < 1e-6f);
  read_vector(db, 7, vector);
  REQUIRE(vector[0] == 0.0f && vector[1] == 0.0f);
  REQUIRE(sqlite3_close(db) == SQLITE_OK);
  ++checks;
}

static void test_empty(const char *extension) {
  sqlite3 *db = open_host(extension, ":memory:");
  create_index(db, "l2");
  save_snapshot(db);
  BatchF32V1 empty = descriptor(NULL, NULL, 0);
  REQUIRE(batch_insert(db, &empty, NULL) == SQLITE_OK);
  assert_snapshot(db);
  REQUIRE(sqlite3_close(db) == SQLITE_OK);
  ++checks;
}

static void test_transaction_and_savepoint(const char *extension) {
  sqlite3 *db = open_host(extension, ":memory:");
  create_index(db, "l2");
  execute(db, "CREATE TABLE app_meta(value);INSERT INTO app_meta VALUES('committed')");
  save_snapshot(db);
  float vectors[18]; int64_t rowids[9]; generate(vectors, rowids, 9, 20);
  BatchF32V1 batch = descriptor(vectors, rowids, 9);
  execute(db, "BEGIN;INSERT INTO app_meta VALUES('prior');SAVEPOINT before_batch");
  REQUIRE(batch_insert(db, &batch, "{\"batch_size\":8}") == SQLITE_OK);
  REQUIRE(integer(db, "SELECT live_count FROM v_diskann_meta") == 9);
  execute(db, "ROLLBACK TO before_batch;RELEASE before_batch");
  assert_snapshot(db);
  REQUIRE(integer(db, "SELECT count(*) FROM app_meta") == 2);
  REQUIRE(batch_insert(db, &batch, NULL) == SQLITE_OK);
  execute(db, "ROLLBACK");
  assert_snapshot(db);
  REQUIRE(integer(db, "SELECT count(*) FROM app_meta") == 1);
  REQUIRE(sqlite3_close(db) == SQLITE_OK);
  ++checks;
}

static void test_duplicates_and_late_invalid(const char *extension) {
  sqlite3 *db = open_host(extension, ":memory:");
  create_index(db, "l2");
  float vectors[82]; int64_t rowids[41]; generate(vectors, rowids, 41, 100);
  BatchF32V1 batch = descriptor(vectors, rowids, 41);
  save_snapshot(db);
  rowids[40] = rowids[0];
  REQUIRE((batch_insert(db, &batch, "{\"batch_size\":8}") & 0xff) == SQLITE_CONSTRAINT);
  assert_snapshot(db);
  rowids[40] = -1;
  REQUIRE(batch_insert(db, &batch, NULL) == SQLITE_RANGE);
  assert_snapshot(db);
  rowids[40] = 140;
  rowids[2] = rowids[1];
  REQUIRE((batch_insert(db, &batch, "{\"batch_size\":8}") & 0xff) == SQLITE_CONSTRAINT);
  assert_snapshot(db);
  rowids[2] = 102;
  vectors[80] = INFINITY;
  REQUIRE(batch_insert(db, &batch, NULL) != SQLITE_OK);
  assert_snapshot(db);
  vectors[80] = NAN;
  REQUIRE(batch_insert(db, &batch, NULL) != SQLITE_OK);
  assert_snapshot(db);
  vectors[80] = 41.0f;
  REQUIRE(batch_insert(db, &batch, NULL) == SQLITE_OK);
  save_snapshot(db);
  const float duplicate_vector[] = {3.0f, 7.0f};
  const int64_t duplicate_id[] = {100};
  BatchF32V1 duplicate = descriptor(duplicate_vector, duplicate_id, 1);
  REQUIRE((batch_insert(db, &duplicate, NULL) & 0xff) == SQLITE_CONSTRAINT);
  assert_snapshot(db);
  REQUIRE(sqlite3_close(db) == SQLITE_OK);
  ++checks;
}

static void test_late_failure_atomic(const char *extension, const char *raise_mode,
                                     const char *outer_policy) {
  sqlite3 *db = open_host(extension, ":memory:");
  create_index(db, "l2");
  execute(db, "INSERT INTO v(rowid,embedding) VALUES(1,X'0000803F00000000');"
              "CREATE TABLE app_meta(value)");
  char trigger[512], sql[256], error[256];
  snprintf(trigger, sizeof(trigger),
      "CREATE TRIGGER late_batch_failure BEFORE UPDATE OF neighbors ON v_diskann_nodes "
      "WHEN (SELECT live_count FROM v_diskann_meta)>=20 "
      "BEGIN SELECT RAISE(%s,'late batch edge failure'); END", raise_mode);
  execute(db, trigger);
  save_snapshot(db);
  float vectors[82]; int64_t rowids[41]; generate(vectors, rowids, 41, 100);
  BatchF32V1 batch = descriptor(vectors, rowids, 41);
  snprintf(sql, sizeof(sql),
      "INSERT %s INTO v(operation,embedding,path) VALUES('insert_batch',?1,?2)", outer_policy);
  execute(db, "BEGIN;INSERT INTO app_meta VALUES('prior')");
  int rc = run_batch(db, sql, &batch, VECTORLITE_BATCH_F32_V1_TAG,
                     "{\"batch_size\":8}", error, sizeof(error));
  REQUIRE(rc == SQLITE_CONSTRAINT_TRIGGER);
  REQUIRE(strstr(error, "late batch edge failure") != NULL);
  REQUIRE(sqlite3_get_autocommit(db) == 0);
  assert_snapshot(db);
  REQUIRE(integer(db, "SELECT count(*) FROM app_meta") == 1);
  execute(db, "ROLLBACK;DROP TRIGGER late_batch_failure");
  REQUIRE(batch_insert(db, &batch, NULL) == SQLITE_OK);
  REQUIRE(integer(db, "SELECT live_count FROM v_diskann_meta") == 42);
  REQUIRE(sqlite3_close(db) == SQLITE_OK);
  ++checks;
}

/* Native observations survive SQL rollback, proving which phase was reached. */
typedef struct PhaseTrace {
  sqlite3_int64 live_count;
  int assignments;
  int backedges;
  int completed_earlier_chunk;
} PhaseTrace;

static void trace_phase(sqlite3_context *context, int argc, sqlite3_value **argv) {
  REQUIRE(argc == 2);
  PhaseTrace *trace = (PhaseTrace *)sqlite3_user_data(context);
  sqlite3_int64 live = sqlite3_value_int64(argv[0]);
  if (live != trace->live_count) {
    if (trace->assignments && trace->backedges) trace->completed_earlier_chunk = 1;
    trace->live_count = live;
    trace->assignments = trace->backedges = 0;
  }
  if (sqlite3_value_int(argv[1]) == 1) ++trace->assignments;
  else ++trace->backedges;
  sqlite3_result_int(context, 0);
}

static void test_late_phase_failure(const char *extension, int backedge) {
  sqlite3 *db = open_host(extension, ":memory:");
  create_index(db, "l2");
  execute(db, "INSERT INTO v(rowid,embedding) VALUES(1,X'0000803F00000000');"
              "CREATE TABLE app_meta(value)");
  PhaseTrace trace = {0, 0, 0, 0};
  REQUIRE(sqlite3_create_function_v2(db, "batch_trace_phase", 2, SQLITE_UTF8,
      &trace, trace_phase, NULL, NULL, NULL) == SQLITE_OK);
  /* Each full chunk allocates eight contiguous internal IDs before assignment.
   * Only these new IDs receive outgoing assignment; older IDs are backedges.
   * No exact adjacency or hash-map iteration order is assumed. */
  execute(db,
      "CREATE TRIGGER trace_assignment AFTER UPDATE OF neighbors ON v_diskann_nodes "
      "WHEN OLD.node_id>=(SELECT next_node_id-8 FROM v_diskann_meta) "
      "AND length(OLD.neighbors)=0 BEGIN "
      "SELECT batch_trace_phase((SELECT live_count FROM v_diskann_meta),1); END;"
      "CREATE TRIGGER trace_backedge BEFORE UPDATE OF neighbors ON v_diskann_nodes "
      "WHEN OLD.node_id<(SELECT next_node_id-8 FROM v_diskann_meta) BEGIN "
      "SELECT batch_trace_phase((SELECT live_count FROM v_diskann_meta),2); END");
  char sql[768];
  snprintf(sql, sizeof(sql),
      "CREATE TRIGGER fail_phase BEFORE UPDATE OF neighbors ON v_diskann_nodes "
      "WHEN (SELECT live_count FROM v_diskann_meta)>=17 AND %s BEGIN "
      "SELECT RAISE(FAIL,'targeted late phase failure'); END",
      backedge ? "OLD.node_id<(SELECT next_node_id-8 FROM v_diskann_meta)"
               : "OLD.node_id>=(SELECT next_node_id-8 FROM v_diskann_meta) "
                 "AND length(OLD.neighbors)=0");
  execute(db, sql);
  save_snapshot(db);
  float vectors[50]; int64_t rowids[25]; generate(vectors, rowids, 25, 100);
  BatchF32V1 batch = descriptor(vectors, rowids, 25);
  execute(db, "BEGIN;INSERT INTO app_meta VALUES('prior')");
  char error[256];
  REQUIRE(run_batch(db,
      "INSERT OR IGNORE INTO v(operation,embedding,path) VALUES('insert_batch',?1,?2)",
      &batch, VECTORLITE_BATCH_F32_V1_TAG, "{\"batch_size\":8}",
      error, sizeof(error)) == SQLITE_CONSTRAINT_TRIGGER);
  REQUIRE(strstr(error, "targeted late phase failure") != NULL);
  REQUIRE(sqlite3_get_autocommit(db) == 0);
  assert_snapshot(db);
  REQUIRE(integer(db, "SELECT count(*) FROM app_meta") == 1);
  REQUIRE(trace.assignments == 8);
  if (backedge) {
    /* All eight outgoing assignments in the failed chunk finished first. */
    REQUIRE(trace.live_count == 17);
    REQUIRE(trace.completed_earlier_chunk);
  } else {
    /* The first completed chunk had both assignment and backedge writes; the
     * second chunk's first outgoing assignment failed before its AFTER trace. */
    REQUIRE(trace.live_count == 9);
    REQUIRE(trace.backedges > 0);
  }
  execute(db, "ROLLBACK;DROP TRIGGER fail_phase;DROP TRIGGER trace_assignment;"
              "DROP TRIGGER trace_backedge");
  REQUIRE(batch_insert(db, &batch, "{\"batch_size\":8}") == SQLITE_OK);
  REQUIRE(sqlite3_close(db) == SQLITE_OK);
  ++checks;
}

static void test_late_write_effects(const char *extension, int failure) {
  const char *triggers[] = {
      "CREATE TRIGGER late_write BEFORE INSERT ON v_diskann_nodes "
      "WHEN NEW.public_rowid=124 BEGIN SELECT RAISE(FAIL,'late allocation failure'); END",
      "CREATE TRIGGER late_write BEFORE INSERT ON v_diskann_nodes "
      "WHEN NEW.public_rowid=124 BEGIN SELECT RAISE(IGNORE); END",
      "CREATE TRIGGER late_write AFTER INSERT ON v_diskann_nodes "
      "WHEN NEW.public_rowid=124 BEGIN DELETE FROM v_diskann_txn WHERE singleton=2; END",
      "CREATE TRIGGER late_write BEFORE UPDATE OF revision ON v_diskann_meta "
      "BEGIN SELECT RAISE(FAIL,'late revision failure'); END",
      "CREATE TRIGGER late_write BEFORE UPDATE OF neighbors ON v_diskann_nodes "
      "WHEN (SELECT live_count FROM v_diskann_meta)>=17 "
      "BEGIN SELECT RAISE(IGNORE); END"};
  sqlite3 *db = open_host(extension, ":memory:");
  create_index(db, "l2");
  execute(db, "INSERT INTO v(rowid,embedding) VALUES(1,X'0000803F00000000');"
              "CREATE TABLE app_meta(value)");
  execute(db, triggers[failure]);
  save_snapshot(db);
  float vectors[66]; int64_t rowids[33]; generate(vectors, rowids, 33, 100);
  BatchF32V1 batch = descriptor(vectors, rowids, 33);
  /* Also exercise draining count-change rows on an unsuccessful carrier. */
  execute(db, "PRAGMA count_changes=ON;BEGIN;INSERT INTO app_meta VALUES('prior')");
  char error[256];
  int rc = run_batch(db,
      "INSERT OR FAIL INTO v(operation,embedding,path) VALUES('insert_batch',?1,?2)",
      &batch, VECTORLITE_BATCH_F32_V1_TAG, "{\"batch_size\":8}", error, sizeof(error));
  if (failure == 0 || failure == 3) {
    REQUIRE(rc == SQLITE_CONSTRAINT_TRIGGER);
    REQUIRE(strstr(error, failure == 0 ? "late allocation failure" : "late revision failure") != NULL);
  } else {
    REQUIRE(rc == SQLITE_CORRUPT);
  }
  REQUIRE(sqlite3_get_autocommit(db) == 0);
  assert_snapshot(db);
  REQUIRE(integer(db, "SELECT count(*) FROM app_meta") == 1);
  /* CORRUPT may make the transaction read-only; end it before recovery writes. */
  execute(db, "ROLLBACK;PRAGMA count_changes=OFF;DROP TRIGGER late_write");
  REQUIRE(batch_insert(db, &batch, "{\"batch_size\":8}") == SQLITE_OK);
  REQUIRE(sqlite3_close(db) == SQLITE_OK);
  ++checks;
}

static void test_invalid_descriptor_and_options(const char *extension) {
  sqlite3 *db = open_host(extension, ":memory:");
  create_index(db, "l2");
  const float vectors[] = {1.0f, 0.0f}; const int64_t rowids[] = {1};
  BatchF32V1 batch = descriptor(vectors, rowids, 1);
  save_snapshot(db);
  REQUIRE(run_batch(db, "INSERT INTO v(operation,embedding) VALUES('insert_batch',?1)",
                    &batch, "wrong.batch.tag", NULL, NULL, 0) == SQLITE_MISUSE);
  REQUIRE(run_batch(db, "INSERT INTO v(operation,embedding) VALUES('insert_batch',?1)",
                    NULL, VECTORLITE_BATCH_F32_V1_TAG, NULL, NULL, 0) == SQLITE_MISUSE);
  REQUIRE(run_batch(db, "INSERT INTO v(operation,embedding) VALUES('insert_batch',?1)",
                    (const BatchF32V1 *)(uintptr_t)1, VECTORLITE_BATCH_F32_V1_TAG,
                    NULL, NULL, 0) == SQLITE_MISUSE);
  BatchF32V1 invalid = batch;
  invalid.abi_version = 2;
  REQUIRE(batch_insert(db, &invalid, NULL) == SQLITE_MISMATCH);
  invalid = batch; invalid.struct_size = 39;
  REQUIRE(batch_insert(db, &invalid, NULL) == SQLITE_MISMATCH);
  invalid = batch; invalid.dimension = 3;
  REQUIRE(batch_insert(db, &invalid, NULL) == SQLITE_MISMATCH);
  invalid = batch; invalid.vectors = NULL;
  REQUIRE(batch_insert(db, &invalid, NULL) == SQLITE_MISUSE);
  invalid = batch; invalid.rowids = NULL;
  REQUIRE(batch_insert(db, &invalid, NULL) == SQLITE_MISUSE);
  invalid = batch; invalid.rowids = (const int64_t *)(uintptr_t)1;
  REQUIRE(batch_insert(db, &invalid, NULL) == SQLITE_MISUSE);
  invalid = descriptor(NULL, NULL, 0); invalid.dimension = 0;
  REQUIRE(batch_insert(db, &invalid, NULL) == SQLITE_MISMATCH);
  invalid.dimension = 3;
  REQUIRE(batch_insert(db, &invalid, NULL) == SQLITE_MISMATCH);
  invalid = batch; invalid.count = UINT64_MAX;
  REQUIRE(batch_insert(db, &invalid, NULL) == SQLITE_RANGE);
  invalid = batch; invalid.count = (uint64_t)INT64_MAX;
  REQUIRE(batch_insert(db, &invalid, NULL) == SQLITE_TOOBIG);
  invalid = batch; invalid.vectors = (const float *)(uintptr_t)1;
  REQUIRE(batch_insert(db, &invalid, NULL) == SQLITE_MISUSE);
  const char *bad_options[] = {"{} trailing", "[]", "null", "{\"batch_size\":0}",
      "{\"batch_size\":33}", "{\"batch_size\":1.5}", "{\"batch_size\":-1}",
      "{\"batch_size\":true}", "{\"unknown\":8}",
      "{\"batch_size\":8,\"batch_size\":16}"};
  for (size_t i = 0; i < sizeof(bad_options)/sizeof(bad_options[0]); ++i) {
    REQUIRE(batch_insert(db, &batch, bad_options[i]) != SQLITE_OK);
    assert_snapshot(db);
  }
  assert_snapshot(db);
  REQUIRE(run_batch(db, "INSERT INTO v(rowid,operation,embedding) VALUES(99,'insert_batch',?1)",
                    &batch, VECTORLITE_BATCH_F32_V1_TAG, NULL, NULL, 0) != SQLITE_OK);
  REQUIRE(run_batch(db, "INSERT INTO v(distance,operation,embedding) VALUES(1,'insert_batch',?1)",
                    &batch, VECTORLITE_BATCH_F32_V1_TAG, NULL, NULL, 0) != SQLITE_OK);
  assert_snapshot(db);
  REQUIRE(batch_insert(db, &batch, "{}") == SQLITE_OK);
  REQUIRE(sqlite3_close(db) == SQLITE_OK);
  ++checks;
}

static void test_guard_and_count_changes(const char *extension) {
  sqlite3 *db = open_host(extension, ":memory:");
  create_index(db, "l2");
  float vectors[18]; int64_t rowids[9]; generate(vectors, rowids, 9, 10);
  BatchF32V1 batch = descriptor(vectors, rowids, 9);
  execute(db, "PRAGMA count_changes=ON");
  REQUIRE(batch_insert(db, &batch, "{\"batch_size\":8}") == SQLITE_OK);
  execute(db, "PRAGMA count_changes=OFF");
  save_snapshot(db);
  execute(db, "CREATE TEMP TRIGGER corrupt_guard AFTER UPDATE ON v_diskann_txn "
              "BEGIN SELECT RAISE(IGNORE); END");
  REQUIRE(batch_insert(db, &batch, NULL) != SQLITE_OK);
  assert_snapshot(db);
  execute(db, "DROP TRIGGER corrupt_guard");
  REQUIRE(sqlite3_close(db) == SQLITE_OK);
  ++checks;
}

static void test_journal_off_and_budget(const char *extension) {
  sqlite3 *db = open_host(extension, ":memory:");
  create_index(db, "l2");
  save_snapshot(db);
  execute(db, "PRAGMA journal_mode=OFF");
  const float vectors[] = {1.0f, 0.0f}; const int64_t rowids[] = {1};
  BatchF32V1 batch = descriptor(vectors, rowids, 1);
  REQUIRE(batch_insert(db, &batch, NULL) != SQLITE_OK);
  assert_snapshot(db);
  execute(db, "PRAGMA journal_mode=MEMORY");
  REQUIRE(sqlite3_close(db) == SQLITE_OK);
  db = open_host(extension, ":memory:");
  execute(db, "CREATE VIRTUAL TABLE v USING vectorlite(embedding float32[2] l2,"
              "diskann(degree=2,build_list_size=2,search_list_size=2,cache_bytes=262144,max_visits=16))");
  save_snapshot(db);
  const float budget_vectors[] = {1.0f, 0.0f, 2.0f, 0.0f};
  const int64_t budget_rowids[] = {1, 2};
  BatchF32V1 budget_batch = descriptor(budget_vectors, budget_rowids, 2);
  /* The seed fits, but the second point requires admitted runtime/batch scratch. */
  REQUIRE(batch_insert(db, &budget_batch, NULL) == SQLITE_TOOBIG);
  assert_snapshot(db);
  REQUIRE(sqlite3_close(db) == SQLITE_OK);
  ++checks;
}

static int destroy_count = 0;
static void destroy_batch(void *pointer) {
  BatchF32V1 *batch = (BatchF32V1 *)pointer;
  ++destroy_count;
  free((void *)batch->vectors);
  free((void *)batch->rowids);
  free(batch);
}

static void test_binding_lifetime(const char *extension) {
  sqlite3 *db = open_host(extension, ":memory:");
  create_index(db, "l2");
  BatchF32V1 *batch = (BatchF32V1 *)malloc(sizeof(*batch));
  float *vectors = (float *)malloc(4 * sizeof(float));
  int64_t *rowids = (int64_t *)malloc(2 * sizeof(int64_t));
  REQUIRE(batch && vectors && rowids);
  generate(vectors, rowids, 2, 1);
  *batch = descriptor(vectors, rowids, 2);
  sqlite3_stmt *stmt = NULL;
  REQUIRE(sqlite3_prepare_v2(db,
      "INSERT INTO v(operation,embedding) VALUES('insert_batch',?1)",
      -1, &stmt, NULL) == SQLITE_OK);
  int before = destroy_count;
  REQUIRE(sqlite3_bind_pointer(stmt, 1, batch, VECTORLITE_BATCH_F32_V1_TAG,
                              destroy_batch) == SQLITE_OK);
  REQUIRE(sqlite3_step(stmt) == SQLITE_DONE);
  REQUIRE(destroy_count == before);
  REQUIRE(sqlite3_reset(stmt) == SQLITE_OK);
  REQUIRE(destroy_count == before);
  REQUIRE(sqlite3_clear_bindings(stmt) == SQLITE_OK);
  REQUIRE(destroy_count == before + 1);
  REQUIRE(sqlite3_finalize(stmt) == SQLITE_OK);
  REQUIRE(destroy_count == before + 1);
  REQUIRE(sqlite3_close(db) == SQLITE_OK);
  ++checks;
}

static void test_wal_reopen_interactive(const char *extension, const char *database) {
  /* The wrapper supplies a fresh absolute path in its unique test directory.
   * Exclusive creation refuses existing files; the wrapper owns checked cleanup. */
  REQUIRE(database[0] == '/' || (strlen(database) > 2 && database[1] == ':'));
  FILE *reserved = fopen(database, "wx");
  REQUIRE(reserved != NULL);
  REQUIRE(fclose(reserved) == 0);
  sqlite3 *writer = open_host(extension, database);
  execute(writer, "PRAGMA journal_mode=WAL");
  create_index(writer, "l2");
  execute(writer, "INSERT INTO v(rowid,embedding) VALUES(1,X'0000803F00000000')");
  save_snapshot(writer);
  sqlite3 *reader = open_host(extension, database);
  execute(reader, "BEGIN");
  assert_snapshot(reader); /* Establish the pre-batch WAL read snapshot. */
  float vectors[34]; int64_t rowids[17]; generate(vectors, rowids, 17, 100);
  BatchF32V1 batch = descriptor(vectors, rowids, 17);
  REQUIRE(batch_insert(writer, &batch, "{\"batch_size\":8}") == SQLITE_OK);
  REQUIRE(integer(writer, "SELECT live_count FROM v_diskann_meta") == 18);
  REQUIRE(integer(writer, "SELECT revision FROM v_diskann_meta") == 2);
  assert_snapshot(reader);
  REQUIRE(integer(reader, "SELECT live_count FROM v_diskann_meta") == 1);
  REQUIRE(integer(reader, "SELECT count(*) FROM v WHERE rowid=116") == 0);
  execute(reader, "COMMIT");
  REQUIRE(integer(reader, "SELECT live_count FROM v_diskann_meta") == 18);
  float readback[2], distance;
  read_vector(reader, 116, readback);
  REQUIRE(memcmp(readback, vectors + 32, sizeof(readback)) == 0);
  REQUIRE(nearest(reader, vectors + 32, &distance) == 116);
  REQUIRE(sqlite3_close(reader) == SQLITE_OK);
  save_snapshot(writer);
  REQUIRE(sqlite3_close(writer) == SQLITE_OK);

  writer = open_host(extension, database);
  assert_snapshot(writer); /* Byte-exact committed shadows survive reconnect. */
  REQUIRE(nearest(writer, vectors + 32, &distance) == 116);
  execute(writer, "INSERT INTO v(rowid,embedding) VALUES(200,X'000048430000803F');"
                  "UPDATE v SET embedding=X'0000803F00000040' WHERE rowid=100;"
                  "DELETE FROM v WHERE rowid=101");
  read_vector(writer, 100, readback);
  REQUIRE(readback[0] == 1.0f && readback[1] == 2.0f);
  REQUIRE(integer(writer, "SELECT count(*) FROM v WHERE rowid=101") == 0);
  REQUIRE(integer(writer, "SELECT live_count FROM v_diskann_meta") == 18);
  REQUIRE(integer(writer, "SELECT deleted_count FROM v_diskann_meta") == 2);
  /* A second native batch after interactive UPDATE/DELETE keeps retired nodes
   * legal and resets every private runtime/store resource from the first batch. */
  const float more_vectors[] = {300.0f, 1.0f, 301.0f, 1.0f};
  const int64_t more_rowids[] = {300, 301};
  BatchF32V1 more = descriptor(more_vectors, more_rowids, 2);
  REQUIRE(batch_insert(writer, &more, "{\"batch_size\":1}") == SQLITE_OK);
  execute(writer, "INSERT INTO v(operation) VALUES('consolidate')");
  REQUIRE(integer(writer, "SELECT live_count FROM v_diskann_meta") == 20);
  REQUIRE(integer(writer, "SELECT deleted_count FROM v_diskann_meta") == 0);
  REQUIRE(nearest(writer, more_vectors, &distance) == 300);
  save_snapshot(writer);
  REQUIRE(sqlite3_close(writer) == SQLITE_OK);
  reader = open_host(extension, database);
  assert_snapshot(reader);
  REQUIRE(nearest(reader, more_vectors + 2, &distance) == 301);
  REQUIRE(sqlite3_close(reader) == SQLITE_OK);
  ++checks;
}

int main(int argc, char **argv) {
  REQUIRE(argc == 3);
  REQUIRE(sqlite3_libversion_number() >= 3038000);
  test_insert_readback_query(argv[1]);
  test_chunk_sizes(argv[1]);
  test_cosine_and_zero(argv[1]);
  test_empty(argv[1]);
  test_transaction_and_savepoint(argv[1]);
  test_duplicates_and_late_invalid(argv[1]);
  test_late_failure_atomic(argv[1], "ABORT", "");
  test_late_failure_atomic(argv[1], "FAIL", "OR IGNORE");
  test_late_failure_atomic(argv[1], "FAIL", "OR FAIL");
  test_late_failure_atomic(argv[1], "ABORT", "OR REPLACE");
  test_invalid_descriptor_and_options(argv[1]);
  test_guard_and_count_changes(argv[1]);
  test_journal_off_and_budget(argv[1]);
  test_binding_lifetime(argv[1]);
  test_late_phase_failure(argv[1], 0);
  test_late_phase_failure(argv[1], 1);
  for (int failure = 0; failure < 5; ++failure) test_late_write_effects(argv[1], failure);
  test_wal_reopen_interactive(argv[1], argv[2]);
  printf("PASS: %d native pointer batch checks (SQLite %s)\n", checks,
         sqlite3_libversion());
  return 0;
}
