/* Native CTest gate for the Debug-only DiskANN write statement-shape counters.
 *
 * This host is an ordinary SQLite application: it links the host SQLite library,
 * loads the built extension with sqlite3_load_extension, and drives the public
 * deferred-indexing lifecycle with plain SQL. It resolves the three Debug-only
 * instrumentation entry points it needs from the loaded module (dlsym /
 * GetProcAddress) and includes no Vectorlite header.
 *
 * Plan 2026-11-17-diskann-write-path-statement-elimination, criterion C3:
 *   - fixture: one 10,000-row, 128-dimension DiskANN table driven to publication
 *     with the same public control SQL the deferred test suites use;
 *   - warm-up: 100 single-row INSERTs after publication, so the first measured
 *     row is at steady state with a published generation present;
 *   - single-row shape: the counters are reset and read across five consecutive
 *     individually instrumented INSERTs; every row must issue <= 6
 *     extension-side sqlite3_prepare_v2 calls;
 *   - multi-row shape: one 128-row INSERT measured as a whole must issue
 *     <= 6 * 128 + 32 = 800 extension-side prepares;
 *   - attempted prepares (vectorlite_test_prepare_attempts) are recorded
 *     separately from successful prepares and finalizes for every measurement.
 *
 * The counters are incremented only by statements the extension itself prepares
 * and finalizes, so this host's own sqlite3_prepare_v2 calls (the single-row
 * INSERT, the multi-row INSERT and the lifecycle control SQL) are not counted.
 * The carrier UPDATE stays callback-scoped (design D4) and the three DML
 * statements of a row (insert entry, insert current, metadata update) are
 * prepared per use, because retaining a DML statement across the carrier
 * callback trips SQLite's BTREE_SAVEPOSITION assertion (plan section 12).
 * Those four are the residual per-row prepares: 4 observed against a limit
 * of 6.
 *
 * Gate failures are recorded with the bound and the observed count and make the
 * process exit non-zero with a clear message. Fixture and lifecycle invariants
 * fail fast through REQUIRE.
 */
#include <sqlite3.h>

#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
typedef HMODULE SymbolHandle;
#else
#include <dlfcn.h>
typedef void *SymbolHandle;
#endif

#define REQUIRE(condition) do { \
  if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    exit(1); \
  } \
} while (0)

enum {
  /* Declared vector space and the required published population. */
  DIMENSION = 128,
  PUBLISHED_ROWS = 10000,
  /* C3 protocol: warm-up, measured single rows, measured multi-row statement. */
  WARMUP_ROWS = 100,
  MEASURED_ROWS = 5,
  MULTI_ROWS = 128,
  /* C3 statement-shape limits. */
  PREPARE_LIMIT_PER_ROW = 6,
  MULTI_PREPARE_LIMIT = 6 * MULTI_ROWS + 32,
  /* Bounded public lifecycle steps, as in the existing query-counts host. */
  STEP_RECORDS = 256,
  STEP_BYTES = 65536,
  MAX_LIFECYCLE_STEPS = 100000,
  OPTIONS_BYTES = 2048,
  MULTI_SQL_BYTES = 16384,
  MULTI_PARAMETERS = 2 * MULTI_ROWS
};

static const uint64_t BUILD_BUDGET_BYTES = 536870912ULL;
static const char TABLE[] = "shape";
static const char OPTIONS[] =
    "degree=8, build_list_size=16, search_list_size=8, max_visits=65536, "
    "cache_bytes=67108864";

/* --- Debug-only instrumentation entry points ----------------------------- */

typedef struct TestApi {
  void (*reset_counters)(void);
  void (*read_counters)(uint64_t *out);
  uint64_t (*prepare_attempts)(void);
} TestApi;

/* Gate verdict checks are non-fatal: every measurement is printed before the
 * host exits nonzero, so a failing gate still records its machine-checkable
 * summary. Setup and lifecycle invariants above use REQUIRE and fail fast. */
#define CHECK(condition) do { \
  if (condition) { \
    ++checks; \
  } else { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    ++failures; \
  } \
} while (0)

static TestApi test_api;
static int checks = 0;
static int failures = 0;

typedef struct Counters {
  uint64_t prepares;
  uint64_t finalizes;
} Counters;

/* --- SQLite host helpers -------------------------------------------------- */

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

static char *query_text(sqlite3 *db, const char *sql) {
  sqlite3_stmt *statement = NULL;
  REQUIRE(sqlite3_prepare_v2(db, sql, -1, &statement, NULL) == SQLITE_OK);
  REQUIRE(sqlite3_step(statement) == SQLITE_ROW);
  REQUIRE(sqlite3_column_type(statement, 0) == SQLITE_TEXT);
  char *text = sqlite3_mprintf("%s", sqlite3_column_text(statement, 0));
  REQUIRE(text != NULL);
  REQUIRE(sqlite3_step(statement) == SQLITE_DONE);
  REQUIRE(sqlite3_finalize(statement) == SQLITE_OK);
  return text;
}

/* --- Debug-only symbol resolution ---------------------------------------- */

static SymbolHandle loaded_module(const char *extension) {
#if defined(_WIN32)
  HMODULE module = GetModuleHandleA(extension);
  if (module == NULL) module = LoadLibraryA(extension);
  return module;
#else
#ifdef RTLD_NOLOAD
  void *module = dlopen(extension, RTLD_LAZY | RTLD_NOLOAD);
  if (module == NULL) module = dlopen(extension, RTLD_LAZY);
  return module;
#else
  return dlopen(extension, RTLD_LAZY);
#endif
#endif
}

static void *module_symbol(SymbolHandle module, const char *name) {
#if defined(_WIN32)
  return (void *)GetProcAddress(module, name);
#else
  return dlsym(module, name);
#endif
}

static void *require_symbol(SymbolHandle module, const char *name) {
  void *symbol = module_symbol(module, name);
  if (symbol == NULL) {
    fprintf(stderr, "FAIL missing Debug-only instrumentation symbol: %s\n",
            name);
    exit(1);
  }
  return symbol;
}

/* Every symbol below is compiled out of Release builds, so a missing export is a
 * configuration error, never a skipped check. */
static TestApi resolve_test_api(const char *extension) {
  TestApi api;
  SymbolHandle module = loaded_module(extension);
  if (module == NULL) {
    fprintf(stderr, "FAIL cannot inspect the loaded extension module: %s\n",
            extension);
    exit(1);
  }
  api.reset_counters = (void (*)(void))require_symbol(
      module, "vectorlite_test_reset_counters");
  api.read_counters = (void (*)(uint64_t *))require_symbol(
      module, "vectorlite_test_read_counters");
  api.prepare_attempts = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_prepare_attempts");
  printf("SYMBOLS resolved 3 Debug-only instrumentation exports from %s\n",
         extension);
  return api;
}

static Counters read_counters(void) {
  uint64_t out[2] = {0, 0};
  Counters counters;
  test_api.read_counters(out);
  counters.prepares = out[0];
  counters.finalizes = out[1];
  return counters;
}

/* --- fixed-shape JSON scanners ------------------------------------------- */

/* The public control/build results are small fixed-shape JSON documents. The
 * host scans them for one exact key, so it needs no JSON library. */
static const char *json_value(const char *json, const char *key) {
  const char *found = strstr(json, key);
  return found == NULL ? NULL : found + strlen(key);
}

static int json_u64(const char *json, const char *key, uint64_t *out) {
  const char *value = json_value(json, key);
  uint64_t parsed = 0;
  if (value == NULL || *value < '0' || *value > '9') return 0;
  while (*value >= '0' && *value <= '9') {
    if (parsed > (UINT64_MAX - 9) / 10) return 0;
    parsed = parsed * 10 + (uint64_t)(*value - '0');
    ++value;
  }
  *out = parsed;
  return 1;
}

/* key ends with the opening quote of the string value. */
static int json_text(const char *json, const char *key, char *out,
                     size_t capacity) {
  const char *value = json_value(json, key);
  size_t length = 0;
  if (value == NULL) return 0;
  while (value[length] != '\0' && value[length] != '"') ++length;
  if (value[length] != '"' || length + 1 > capacity) return 0;
  memcpy(out, value, length);
  out[length] = '\0';
  return 1;
}

static char *status_json(sqlite3 *db, const char *table) {
  char sql[256];
  int written = snprintf(sql, sizeof(sql),
                         "SELECT vectorlite_index_status('main','%s')", table);
  REQUIRE(written > 0 && (size_t)written < sizeof(sql));
  return query_text(db, sql);
}

typedef struct Job {
  uint64_t job_id;
  uint64_t fence;
  uint64_t phase;
} Job;

static Job read_job(sqlite3 *db, const char *table) {
  Job job;
  char *json = status_json(db, table);
  REQUIRE(json_u64(json, "\"job_id\":", &job.job_id));
  REQUIRE(json_u64(json, "\"fence\":", &job.fence));
  REQUIRE(json_u64(json, "\"phase\":", &job.phase));
  sqlite3_free(json);
  return job;
}

static uint64_t status_u64(sqlite3 *db, const char *table, const char *key) {
  uint64_t value = 0;
  char *json = status_json(db, table);
  REQUIRE(json_u64(json, key, &value));
  sqlite3_free(json);
  return value;
}

/* --- public deferred lifecycle ------------------------------------------- */

static void control(sqlite3 *db, const char *table, const char *operation,
                    const char *options) {
  char sql[OPTIONS_BYTES + 128];
  int written = snprintf(sql, sizeof(sql),
                         "INSERT INTO %s(operation,path) VALUES('%s','%s')",
                         table, operation, options);
  REQUIRE(written > 0 && (size_t)written < sizeof(sql));
  execute(db, sql);
}

static void ids_options(char *out, size_t capacity, Job job) {
  int written = snprintf(out, capacity,
                         "{\"job_id\":%" PRIu64 ",\"fence\":%" PRIu64 "}",
                         job.job_id, job.fence);
  REQUIRE(written > 0 && (size_t)written < capacity);
}

static void step_options(char *out, size_t capacity, Job job,
                         const char *token) {
  int written;
  if (token == NULL) {
    written = snprintf(out, capacity,
                       "{\"job_id\":%" PRIu64 ",\"fence\":%" PRIu64
                       ",\"max_records\":%d,\"max_bytes\":%d}",
                       job.job_id, job.fence, STEP_RECORDS, STEP_BYTES);
  } else {
    written = snprintf(out, capacity,
                       "{\"job_id\":%" PRIu64 ",\"fence\":%" PRIu64
                       ",\"candidate_token\":\"%s\",\"max_records\":%d,"
                       "\"max_bytes\":%d}",
                       job.job_id, job.fence, token, STEP_RECORDS, STEP_BYTES);
  }
  REQUIRE(written > 0 && (size_t)written < capacity);
}

static void publish_options(char *out, size_t capacity, Job job,
                            const char *token) {
  int written = snprintf(out, capacity,
                         "{\"job_id\":%" PRIu64 ",\"fence\":%" PRIu64
                         ",\"candidate_token\":\"%s\"}",
                         job.job_id, job.fence, token);
  REQUIRE(written > 0 && (size_t)written < capacity);
}

/* Opaque candidate tokens are only ever embedded in a single-quoted SQL literal,
 * so an unexpected byte must fail the test rather than reach the parser. */
static int candidate_token_valid(const char *token) {
  if (strncmp(token, "cand:", 5) != 0 || token[5] == '\0') return 0;
  for (const char *byte = token + 5; *byte != '\0'; ++byte) {
    unsigned char value = (unsigned char)*byte;
    if (!((value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
          (value >= '0' && value <= '9') || value == ':' || value == '.' ||
          value == '_' || value == '-')) {
      return 0;
    }
  }
  return 1;
}

/* Drives prepare -> snapshot -> claim -> build -> import -> validate -> publish
 * with the same public control SQL the deferred suites use. */
static uint64_t publish_index(sqlite3 *db, const char *table,
                              int64_t expected_rows) {
  char options[OPTIONS_BYTES];
  char token[256];
  char sql[512];
  char *json;
  uint64_t population = 0;
  Job job;
  int steps;
  int phase_index;

  REQUIRE(sqlite3_get_autocommit(db) != 0);
  control(db, table, "index_prepare", "{}");
  for (steps = 0;; ++steps) {
    REQUIRE(steps < MAX_LIFECYCLE_STEPS);
    job = read_job(db, table);
    if (job.phase != 0) break; /* CAPTURING */
    step_options(options, sizeof(options), job, NULL);
    control(db, table, "index_snapshot_step", options);
  }
  REQUIRE(job.phase == 1); /* SEALED */

  ids_options(options, sizeof(options), job);
  control(db, table, "index_claim_build", options);
  int written = snprintf(
      sql, sizeof(sql),
      "SELECT vectorlite_index_build('main','%s','{\"job_id\":%" PRIu64
      ",\"fence\":%" PRIu64 ",\"memory_budget_bytes\":%" PRIu64 "}')",
      table, job.job_id, job.fence, BUILD_BUDGET_BYTES);
  REQUIRE(written > 0 && (size_t)written < sizeof(sql));
  json = query_text(db, sql);
  REQUIRE(json_text(json, "\"candidate_token\":\"", token, sizeof(token)));
  REQUIRE(candidate_token_valid(token));
  REQUIRE(json_u64(json, "\"population\":", &population));
  sqlite3_free(json);
  REQUIRE(population == (uint64_t)expected_rows);

  /* BUILDING (2) -> VALIDATING (3) -> VALIDATED (4), one bounded step each. */
  for (phase_index = 0; phase_index < 2; ++phase_index) {
    const uint64_t wanted = phase_index == 0 ? 2 : 3;
    const char *operation =
        phase_index == 0 ? "index_import_step" : "index_validate_step";
    for (steps = 0;; ++steps) {
      REQUIRE(steps < MAX_LIFECYCLE_STEPS);
      job = read_job(db, table);
      if (job.phase != wanted) break;
      step_options(options, sizeof(options), job, token);
      control(db, table, operation, options);
    }
    REQUIRE(job.phase == wanted + 1);
  }
  REQUIRE(job.phase == 4); /* VALIDATED */

  publish_options(options, sizeof(options), job, token);
  control(db, table, "index_publish", options);
  json = status_json(db, table);
  REQUIRE(strstr(json, "\"current_job\":null") != NULL);
  sqlite3_free(json);
  REQUIRE(status_u64(db, table, "\"pending_count\":") == 0);
  REQUIRE(status_u64(db, table, "\"published_count\":") ==
          (uint64_t)expected_rows);
  return population;
}

/* --- population and measured writes -------------------------------------- */

static void generate_vector(int64_t rowid, float *out) {
  for (int index = 0; index < DIMENSION; ++index) {
    /* Deterministic distinct vectors: v[j] = sin(rowid*0.01 + j*0.1). */
    out[index] = (float)sin((double)rowid * 0.01 + (double)index * 0.1);
  }
}

static void single_insert_sql(char *out, size_t capacity) {
  int written = snprintf(
      out, capacity, "INSERT INTO %s(rowid,embedding) VALUES(?1,?2)", TABLE);
  REQUIRE(written > 0 && (size_t)written < capacity);
}

static void prepare_single_insert(sqlite3 *db, sqlite3_stmt **out) {
  char sql[256];
  single_insert_sql(sql, sizeof(sql));
  REQUIRE(sqlite3_prepare_v2(db, sql, -1, out, NULL) == SQLITE_OK);
}

/* Bulk fixture population: one transaction, so the measured protocol is not
 * dominated by 10,000 autocommit fsyncs. */
static void populate(sqlite3 *db, int64_t rows) {
  sqlite3_stmt *statement = NULL;
  prepare_single_insert(db, &statement);
  execute(db, "BEGIN");
  for (int64_t rowid = 1; rowid <= rows; ++rowid) {
    float vector[DIMENSION];
    generate_vector(rowid, vector);
    REQUIRE(sqlite3_bind_int64(statement, 1, rowid) == SQLITE_OK);
    REQUIRE(sqlite3_bind_blob(statement, 2, vector, (int)sizeof(vector),
                              SQLITE_TRANSIENT) == SQLITE_OK);
    REQUIRE(sqlite3_step(statement) == SQLITE_DONE);
    REQUIRE(sqlite3_reset(statement) == SQLITE_OK);
  }
  REQUIRE(sqlite3_finalize(statement) == SQLITE_OK);
  execute(db, "COMMIT");
  REQUIRE(sqlite3_get_autocommit(db) != 0);
}

static void require_insert_done(sqlite3 *db, int rc, const char *shape,
                                int64_t rowid) {
  if (rc != SQLITE_DONE) {
    fprintf(stderr, "FAIL %s rowid=%" PRId64 " step returned %d: %s\n", shape,
            rowid, rc, sqlite3_errmsg(db));
    exit(1);
  }
}

/* One autocommit single-row INSERT at steady state. The counters are reset
 * immediately before sqlite3_step and read immediately after it, so both values
 * describe exactly this one row. */
typedef struct RowMeasurement {
  int64_t rowid;
  Counters counters;
  uint64_t attempts;
} RowMeasurement;

static RowMeasurement measure_single_row(sqlite3 *db, sqlite3_stmt *insert,
                                         int64_t rowid, const float *vector) {
  RowMeasurement measurement;
  int rc;
  memset(&measurement, 0, sizeof(measurement));
  measurement.rowid = rowid;
  REQUIRE(sqlite3_bind_int64(insert, 1, rowid) == SQLITE_OK);
  REQUIRE(sqlite3_bind_blob(insert, 2, vector, (int)sizeof(float) * DIMENSION,
                            SQLITE_TRANSIENT) == SQLITE_OK);
  test_api.reset_counters();
  rc = sqlite3_step(insert);
  measurement.counters = read_counters();
  measurement.attempts = test_api.prepare_attempts();
  require_insert_done(db, rc, "single-row INSERT", rowid);
  REQUIRE(sqlite3_reset(insert) == SQLITE_OK);
  return measurement;
}

/* Warm-up: WARMUP_ROWS autocommit single-row INSERTs after publication, same SQL
 * shape as the measured statements, counted by nothing. */
static void warm_up(sqlite3 *db, sqlite3_stmt *insert, int64_t first_rowid) {
  for (int index = 0; index < WARMUP_ROWS; ++index) {
    int64_t rowid = first_rowid + index;
    float vector[DIMENSION];
    generate_vector(rowid, vector);
    REQUIRE(sqlite3_bind_int64(insert, 1, rowid) == SQLITE_OK);
    REQUIRE(sqlite3_bind_blob(insert, 2, vector,
                              (int)sizeof(float) * DIMENSION,
                              SQLITE_TRANSIENT) == SQLITE_OK);
    require_insert_done(db, sqlite3_step(insert), "warm-up INSERT", rowid);
    REQUIRE(sqlite3_reset(insert) == SQLITE_OK);
  }
}

static void multi_row_sql(char *out, size_t capacity, int rows) {
  int used = snprintf(out, capacity, "INSERT INTO %s(rowid,embedding) VALUES",
                      TABLE);
  REQUIRE(used > 0 && (size_t)used < capacity);
  for (int index = 0; index < rows; ++index) {
    int written = snprintf(out + used, capacity - (size_t)used,
                           "%s(?%d,?%d)", index == 0 ? "" : ",", 2 * index + 1,
                           2 * index + 2);
    REQUIRE(written > 0 && (size_t)written < capacity - (size_t)used);
    used += written;
  }
}

typedef struct MultiMeasurement {
  int64_t first_rowid;
  Counters counters;
  uint64_t attempts;
} MultiMeasurement;

/* One 128-row multi-row INSERT measured as a whole. The counter interval starts
 * before the host statement is prepared, so extension-side preparation triggered
 * by planning the statement is counted too, and ends after finalization. */
static MultiMeasurement measure_multi_row(sqlite3 *db, int64_t first_rowid) {
  MultiMeasurement measurement;
  sqlite3_stmt *statement = NULL;
  char sql[MULTI_SQL_BYTES];
  int rc;
  memset(&measurement, 0, sizeof(measurement));
  measurement.first_rowid = first_rowid;
  multi_row_sql(sql, sizeof(sql), MULTI_ROWS);
  test_api.reset_counters();
  REQUIRE(sqlite3_prepare_v2(db, sql, -1, &statement, NULL) == SQLITE_OK);
  for (int index = 0; index < MULTI_ROWS; ++index) {
    float vector[DIMENSION];
    generate_vector(first_rowid + index, vector);
    REQUIRE(sqlite3_bind_int64(statement, 2 * index + 1,
                               first_rowid + index) == SQLITE_OK);
    REQUIRE(sqlite3_bind_blob(statement, 2 * index + 2, vector,
                              (int)sizeof(float) * DIMENSION,
                              SQLITE_TRANSIENT) == SQLITE_OK);
  }
  rc = sqlite3_step(statement);
  require_insert_done(db, rc, "multi-row INSERT", first_rowid);
  REQUIRE(sqlite3_finalize(statement) == SQLITE_OK);
  measurement.counters = read_counters();
  measurement.attempts = test_api.prepare_attempts();
  REQUIRE(sqlite3_get_autocommit(db) != 0);
  return measurement;
}

/* --- verdict helpers ----------------------------------------------------- */

static void check_at_most(const char *what, uint64_t observed, uint64_t limit) {
  if (observed <= limit) {
    ++checks;
    return;
  }
  fprintf(stderr,
          "FAIL: %s observed %" PRIu64 " extension-side prepares, C3 gate is "
          "<= %" PRIu64 "\n",
          what, observed, limit);
  ++failures;
}

static void append_u64(char *out, size_t capacity, size_t *used,
                       uint64_t value) {
  int written = snprintf(out + *used, capacity - *used, "%s%" PRIu64,
                         *used == 0 ? "" : ",", value);
  REQUIRE(written > 0 && (size_t)written < capacity - *used);
  *used += (size_t)written;
}

/* --- test body ------------------------------------------------------------ */

int main(int argc, char **argv) {
  const char *extension;
  const char *database;
  sqlite3 *db;
  sqlite3_stmt *insert = NULL;
  int64_t population;
  int64_t first_warmup_rowid;
  int64_t first_measured_rowid;
  int64_t first_multi_rowid;
  RowMeasurement rows[MEASURED_ROWS];
  MultiMeasurement multi;
  uint64_t max_single_prepares = 0;
  uint64_t total_measured_prepares = 0;
  uint64_t pending_rows;
  uint64_t canonical_rows;
  uint64_t published_rows;
  char single_prepares[256];
  char single_finalizes[256];
  char single_attempts[256];
  size_t prepares_used = 0;
  size_t finalizes_used = 0;
  size_t attempts_used = 0;
  FILE *reserved;
  char ddl[512];

  REQUIRE(argc == 3);
  extension = argv[1];
  database = argv[2];
  /* The managed wrapper owns cleanup of exactly these files, so the host must
   * reserve an absolute path and leave it behind. */
  REQUIRE(database[0] == '/' || (strlen(database) > 2 && database[1] == ':'));
  reserved = fopen(database, "wx");
  REQUIRE(reserved != NULL);
  REQUIRE(fclose(reserved) == 0);

  REQUIRE(sqlite3_libversion_number() >= 3038000);
  printf("HOST sqlite=%s extension=%s database=%s\n", sqlite3_libversion(),
         extension, database);

  db = open_host(extension, database);
  test_api = resolve_test_api(extension);

  int written = snprintf(
      ddl, sizeof(ddl),
      "CREATE VIRTUAL TABLE %s USING vectorlite(embedding float32[%d] "
      "l2,diskann(%s))",
      TABLE, DIMENSION, OPTIONS);
  REQUIRE(written > 0 && (size_t)written < sizeof(ddl));
  printf("DDL %s\n", ddl);
  execute(db, ddl);
  populate(db, PUBLISHED_ROWS);
  REQUIRE(status_u64(db, TABLE, "\"pending_count\":") ==
          (uint64_t)PUBLISHED_ROWS);
  population = (int64_t)publish_index(db, TABLE, PUBLISHED_ROWS);
  REQUIRE(population == PUBLISHED_ROWS);
  printf("PUBLISHED table=%s rows=%" PRIu64 " generation=present\n", TABLE,
         (uint64_t)population);
  REQUIRE(sqlite3_get_autocommit(db) != 0);

  /* The published generation is present from here on; the measured rows below
   * only append canonical entries and never rebuild or republish. */
  first_warmup_rowid = PUBLISHED_ROWS + 1;
  first_measured_rowid = first_warmup_rowid + WARMUP_ROWS;
  first_multi_rowid = first_measured_rowid + MEASURED_ROWS;
  prepare_single_insert(db, &insert);
  {
    char single_sql[256];
    single_insert_sql(single_sql, sizeof(single_sql));
    printf("SQL_SINGLE %s\n", single_sql);
  }
  warm_up(db, insert, first_warmup_rowid);
  printf("WARMUP table=%s rows=%d first_rowid=%" PRId64
         " published_generation=present\n",
         TABLE, WARMUP_ROWS, first_warmup_rowid);

  for (int index = 0; index < MEASURED_ROWS; ++index) {
    float vector[DIMENSION];
    int64_t rowid = first_measured_rowid + index;
    generate_vector(rowid, vector);
    rows[index] = measure_single_row(db, insert, rowid, vector);
    append_u64(single_prepares, sizeof(single_prepares), &prepares_used,
               rows[index].counters.prepares);
    append_u64(single_finalizes, sizeof(single_finalizes), &finalizes_used,
               rows[index].counters.finalizes);
    append_u64(single_attempts, sizeof(single_attempts), &attempts_used,
               rows[index].attempts);
    if (rows[index].counters.prepares > max_single_prepares) {
      max_single_prepares = rows[index].counters.prepares;
    }
    total_measured_prepares += rows[index].counters.prepares;
    printf("MEASURED_SINGLE_ROW rowid=%" PRId64 " prepares=%" PRIu64
           " finalizes=%" PRIu64 " attempts=%" PRIu64 " limit=%d\n",
           rows[index].rowid, rows[index].counters.prepares,
           rows[index].counters.finalizes, rows[index].attempts,
           PREPARE_LIMIT_PER_ROW);
  }
  REQUIRE(sqlite3_finalize(insert) == SQLITE_OK);
  REQUIRE(sqlite3_get_autocommit(db) != 0);

  multi = measure_multi_row(db, first_multi_rowid);
  total_measured_prepares += multi.counters.prepares;
  printf("MEASURED_MULTI_ROW rows=%d first_rowid=%" PRId64
         " parameters=%d prepares=%" PRIu64 " finalizes=%" PRIu64
         " attempts=%" PRIu64 " limit=%d\n",
         MULTI_ROWS, multi.first_rowid, MULTI_PARAMETERS,
         multi.counters.prepares, multi.counters.finalizes, multi.attempts,
         MULTI_PREPARE_LIMIT);

  /* Fixture accounting: the publication is untouched and every post-publication
   * write is pending. */
  pending_rows = status_u64(db, TABLE, "\"pending_count\":");
  canonical_rows = status_u64(db, TABLE, "\"canonical_count\":");
  published_rows = status_u64(db, TABLE, "\"published_count\":");
  printf("FIXTURE pending_rows=%" PRIu64 " canonical_rows=%" PRIu64
         " published_rows=%" PRIu64 " expected_pending=%d\n",
         pending_rows, canonical_rows, published_rows,
         WARMUP_ROWS + MEASURED_ROWS + MULTI_ROWS);
  CHECK(pending_rows == (uint64_t)(WARMUP_ROWS + MEASURED_ROWS + MULTI_ROWS));
  CHECK(canonical_rows ==
        (uint64_t)(PUBLISHED_ROWS + WARMUP_ROWS + MEASURED_ROWS + MULTI_ROWS));
  CHECK(published_rows == (uint64_t)PUBLISHED_ROWS);
  REQUIRE(sqlite3_get_autocommit(db) != 0);
  REQUIRE(sqlite3_close(db) == SQLITE_OK);

  for (int index = 0; index < MEASURED_ROWS; ++index) {
    char what[128];
    snprintf(what, sizeof(what), "single-row INSERT rowid=%" PRId64,
             rows[index].rowid);
    check_at_most(what, rows[index].counters.prepares,
                  (uint64_t)PREPARE_LIMIT_PER_ROW);
    CHECK(rows[index].attempts >= rows[index].counters.prepares);
  }
  check_at_most("128-row INSERT", multi.counters.prepares,
                (uint64_t)MULTI_PREPARE_LIMIT);
  CHECK(multi.attempts >= multi.counters.prepares);
  /* A zero total would mean the counters never observed the write path, so the
   * upper-bound gate would be vacuous; the callback-scoped carrier guarantees at
   * least one prepare per inserted row today. */
  CHECK(total_measured_prepares > 0);

  printf("SUMMARY table=%s dimension=%d published_rows=%" PRIu64
         " warmup_rows=%d pending_rows=%" PRIu64 " canonical_rows=%" PRIu64
         " measured_single_rows=%d single_prepares=%s single_finalizes=%s"
         " single_attempts=%s single_max_prepares=%" PRIu64
         " single_limit=%d multi_rows=%d multi_first_rowid=%" PRId64
         " multi_prepares=%" PRIu64 " multi_finalizes=%" PRIu64
         " multi_attempts=%" PRIu64 " multi_limit=%d\n",
         TABLE, DIMENSION, published_rows, WARMUP_ROWS, pending_rows,
         canonical_rows, MEASURED_ROWS, single_prepares, single_finalizes,
         single_attempts, max_single_prepares, PREPARE_LIMIT_PER_ROW, MULTI_ROWS,
         multi.first_rowid, multi.counters.prepares, multi.counters.finalizes,
         multi.attempts, MULTI_PREPARE_LIMIT);
  if (failures != 0) {
    printf("FAIL: %d of %d native write-shape checks failed; C3 requires <= %d "
           "extension-side prepares per single-row INSERT and <= %d for one "
           "%d-row INSERT (SQLite %s)\n",
           failures, failures + checks, PREPARE_LIMIT_PER_ROW,
           MULTI_PREPARE_LIMIT, MULTI_ROWS, sqlite3_libversion());
    return 1;
  }
  printf("PASS: %d native write-shape checks (SQLite %s)\n", checks,
         sqlite3_libversion());
  return 0;
}
