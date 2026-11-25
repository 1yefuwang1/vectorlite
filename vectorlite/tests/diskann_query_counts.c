/* Native CTest gate for the Debug-only DiskANN query statement-shape counters.
 *
 * This host is an ordinary SQLite application: it links the host SQLite library,
 * loads the built extension with sqlite3_load_extension, and drives the public
 * deferred-indexing lifecycle with plain SQL. It resolves the Debug-only
 * instrumentation entry points from the loaded module (dlsym / GetProcAddress)
 * and includes no Vectorlite header.
 *
 * C3 gate measured here:
 *   - one public, unfiltered KNN query over 10,000 published rows performs
 *     <= 12 sqlite3_prepare_v2 calls at L=10 and at L=100;
 *   - the two counts differ by <= 4;
 *   - the instrumentation-reported effective search list equals the requested L;
 *   - at least one reused reader executed and at least one batch override ran;
 *   - prepares == finalizes after the cursor closes, on both the successful
 *     query and a deterministic forced-error query;
 *   - C3 identity on every measured success query: total statement executions
 *     observed by the host trace equal the exported setup reads + expansions +
 *     scored candidates + decision reads + projection reads; three-table
 *     identity-join executions are zero on format-served routes; decision reads
 *     equal the requested L; projection reads and carry hits equal k; the final
 *     top-k loop performs zero SQL while still charging its k visits, so the
 *     owner visit total matches the recorded call-map semantics;
 *   - the D3 three-table fallback fixture (a churned generation whose retired
 *     set cannot be admitted at the pinned minimum budget) exercises the
 *     fallback route: fallback/identity-three-table executions are non-zero, the
 *     result and owner-visit totals match the graph route over the same churned
 *     population, and prepares == finalizes;
 *   - forced demotion: the one-shot demotion hook leaves the L=10 answer
 *     bit-identical while it prepares far more statements than the retained
 *     reader set, and the next query returns to the normal prepare count;
 *   - pair-cache differential: the one-shot force-pair-cache-off hook leaves the
 *     L=10 answer bit-identical (rowids, order and distance bits) and the owner
 *     visit totals unchanged, while every scored id that hit the pair cache on
 *     the normal query performs exactly one extra NodeIdentity read at the same
 *     visit charge and the run reports one cache-disabled operation;
 *   - construction failure: arming a real failed prepare for every reader shape
 *     (and for the NodeIdentity shape alone) surfaces no error, keeps the answer
 *     bit-identical, and attempts more prepares than it completes;
 *   - partial admission: a budget whose scratch floor leaves no reader
 *     allowance still fails with HEAD's SQLITE_TOOBIG scratch message, returns
 *     no rows, finalizes everything, and attempts no lower-priority reader
 *     prepare after admission closes.
 *
 * Debug-only fail-reader-prepare hook mask semantics (ReaderShape::ALL bit
 * index): bit 0 NodeExpand, bit 1 NodeScore, bit 2 NodeIdentity, bit 3
 * ExactUnfiltered, bit 4 ExactPoint.
 *
 * Forced-error fixture: a valid core configuration with max_visits=16 over 600
 * published rows, queried with k=1 and L=8. Its first expansion executes a
 * reader and then deterministically exceeds the visit budget, so the query fails
 * SQLITE_TOOBIG during traversal with no rows returned.
 *
 * Partial-admission fixture: a valid core configuration with cache_bytes=71000
 * over 600 published rows, queried with k=3 and L=4. The table name is kept
 * short so HEAD's unchanged owner reserve still fits inside cache_bytes; the
 * scratch floor then exceeds the whole remainder, admission closes after the
 * first reader prepare, and core rejects the copied allowance exactly as HEAD
 * does. Five header prepares (metadata, generation, root node, retired count,
 * schema stamp) plus the one admission-closing reader try bound the attempts.
 *
 * Fallback fixture: the same 600-row population published at the pinned minimum
 * budget 8,623,280 (the smallest cache_bytes whose head accepts core scratch for
 * this configuration), then churned with ten supported DELETEs. The retired
 * set's load cannot fit the zero reader allowance, so the whole query keeps the
 * format-5-aware three-table fallback reader. A twin table at the default
 * cache_bytes carries the same churn and acts as the graph-route reference.
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
  /* Declared vector space and the published populations. */
  DIMENSION = 8,
  SUCCESS_ROWS = 10000,
  FORCED_ERROR_ROWS = 600,
  PARTIAL_ADMISSION_ROWS = 600,
  FALLBACK_ROWS = 600,
  /* C3 query shapes. */
  SUCCESS_K = 10,
  SUCCESS_LOW_L = 10,
  SUCCESS_HIGH_L = 100,
  FORCED_ERROR_K = 1,
  FORCED_ERROR_L = 8,
  PARTIAL_ADMISSION_K = 3,
  PARTIAL_ADMISSION_L = 4,
  FALLBACK_K = 3,
  FALLBACK_L = 4,
  /* C3 statement-shape limits. */
  PREPARE_LIMIT = 12,
  PREPARE_DELTA_LIMIT = 4,
  /* Demotion must prepare far more than the retained reader set; the bound is
   * well above measurement noise and far below the observed scalar fallback. */
  FORCED_DEMOTE_MIN_EXTRA_PREPARES = 8,
  /* Five header prepares (metadata, generation, root node, retired count,
   * schema stamp) plus at most the one admission-closing reader try. */
  PARTIAL_ADMISSION_MAX_ATTEMPTS = 6,
  /* ReaderShape::ALL and the per-shape Debug-hook masks. Bit index order is
   * NodeExpand, NodeScore, NodeIdentity, ExactUnfiltered, ExactPoint. */
  READER_SHAPE_ALL = 0x1F,
  READER_SHAPE_NODE_EXPAND = 0x01,
  READER_SHAPE_NODE_SCORE = 0x02,
  READER_SHAPE_NODE_IDENTITY = 0x04,
  READER_SHAPE_EXACT_UNFILTERED = 0x08,
  READER_SHAPE_EXACT_POINT = 0x10,
  /* Captured (rowid, distance) rows: the largest measured query is k=10. */
  RESULT_SLOTS = 32,
  /* Bounded public lifecycle steps. */
  STEP_RECORDS = 256,
  STEP_BYTES = 65536,
  MAX_LIFECYCLE_STEPS = 100000,
  OPTIONS_BYTES = 2048
};

static const uint64_t BUILD_BUDGET_BYTES = 536870912ULL;
static const char SUCCESS_TABLE[] = "counts_ok";
static const char FORCED_ERROR_TABLE[] = "counts_err";
/* The short name keeps HEAD's unchanged owner reserve below cache_bytes, so the
 * scratch floor (not the owner estimate) is what closes reader admission. */
static const char PARTIAL_ADMISSION_TABLE[] = "cap";
/* Fixed fallback fixture names: the short name is part of the pinned owner
 * arithmetic behind the 8,623,280-byte minimum cache_bytes constant. */
static const char FALLBACK_TABLE[] = "cf";
static const char FALLBACK_REFERENCE_TABLE[] = "cg";
static const char SUCCESS_OPTIONS[] =
    "degree=8, build_list_size=16, search_list_size=8, max_visits=65536, "
    "cache_bytes=67108864";
static const char FORCED_ERROR_OPTIONS[] =
    "degree=8, build_list_size=16, search_list_size=8, max_visits=16, "
    "cache_bytes=67108864";
static const char PARTIAL_ADMISSION_OPTIONS[] =
    "degree=8, build_list_size=16, search_list_size=8, max_visits=65536, "
    "cache_bytes=71000";
/* T6/T0 pinned minimum: the smallest cache_bytes whose owner head accepts the
 * whole core scratch floor for this configuration, so the reader allowance is
 * exactly zero and the retired shape is declined (three-table fallback). Derived
 * deterministically from the same integer owner/floor arithmetic the extension
 * uses; the twin reference table keeps the default cache budget. */
static const uint64_t FALLBACK_CACHE_BYTES = 8623280ULL;
static const char FALLBACK_OPTIONS[] =
    "degree=8, build_list_size=16, search_list_size=8, max_visits=65536, "
    "cache_bytes=8623280";
static const char FALLBACK_REFERENCE_OPTIONS[] =
    "degree=8, build_list_size=16, search_list_size=8, max_visits=65536, "
    "cache_bytes=67108864";
/* HEAD's exact core workspace rejection for the partial-admission fixture. */
static const char PARTIAL_ADMISSION_MESSAGE[] =
    "DiskANN algorithm scratch exceeds workspace limit";

/* --- Debug-only instrumentation entry points ----------------------------- */

typedef struct TestApi {
  void (*reset_counters)(void);
  void (*read_counters)(uint64_t *out);
  uint64_t (*last_search_l)(void);
  uint64_t (*reader_executions)(void);
  uint64_t (*batch_calls)(void);
  uint64_t (*prepare_attempts)(void);
  void (*force_demote)(void);
  void (*force_carry_off)(void);
  void (*force_pair_cache_off)(void);
  void (*fail_reader_prepare)(uint64_t mask);
  /* C3 reader-shape and setup counters (D4/D8). */
  uint64_t (*identity_3table_executions)(void);
  uint64_t (*fallback_3table_executions)(void);
  uint64_t (*node_expand_executions)(void);
  uint64_t (*node_score_executions)(void);
  uint64_t (*node_identity_executions)(void);
  uint64_t (*setup_reads)(void);
  /* D3 retired-shape counters. */
  uint64_t (*retired_admissions)(void);
  uint64_t (*retired_fallbacks)(void);
  uint64_t (*retired_size)(void);
  /* Traversal work counters (C3 identity terms and visit accounting). */
  uint64_t (*decision_reads)(void);
  uint64_t (*expansions)(void);
  uint64_t (*scored_candidates)(void);
  uint64_t (*projection_reads)(void);
  /* D5 carry counters. */
  uint64_t (*carry_hits)(void);
  uint64_t (*carry_misses)(void);
  uint64_t (*carry_admissions)(void);
  uint64_t (*carry_bytes)(void);
  /* D6 pair-cache counters. */
  uint64_t (*pair_cache_hits)(void);
  uint64_t (*pair_cache_misses)(void);
  uint64_t (*pair_cache_disabled_operations)(void);
  /* Owner visit totals. */
  uint64_t (*owner_visit_total)(void);
  uint64_t (*last_query_visits)(void);
} TestApi;

#define TEST_API_SYMBOL_COUNT 32

/* C3 verdict checks are non-fatal: every measurement is reported before the
 * host exits nonzero, so a failing gate still records its machine-checkable
 * summary. Setup and lifecycle invariants above use REQUIRE and fail fast. */
#define CHECK(condition) do {   if (condition) {     ++checks;   } else {     fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);     ++failures;   } } while (0)

static TestApi test_api;
static float query_vector[DIMENSION];
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
  api.last_search_l = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_last_search_l");
  api.reader_executions = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_reader_executions");
  api.batch_calls = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_batch_calls");
  api.prepare_attempts = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_prepare_attempts");
  api.force_demote = (void (*)(void))require_symbol(
      module, "vectorlite_test_force_demote");
  api.fail_reader_prepare = (void (*)(uint64_t))require_symbol(
      module, "vectorlite_test_fail_reader_prepare");
  api.force_carry_off = (void (*)(void))require_symbol(
      module, "vectorlite_test_force_carry_off");
  api.force_pair_cache_off = (void (*)(void))require_symbol(
      module, "vectorlite_test_force_pair_cache_off");
  api.identity_3table_executions = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_identity_3table_executions");
  api.fallback_3table_executions = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_fallback_3table_executions");
  api.node_expand_executions = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_node_expand_executions");
  api.node_score_executions = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_node_score_executions");
  api.node_identity_executions = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_node_identity_executions");
  api.setup_reads = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_setup_reads");
  api.retired_admissions = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_retired_admissions");
  api.retired_fallbacks = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_retired_fallbacks");
  api.retired_size = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_retired_size");
  api.decision_reads = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_decision_reads");
  api.expansions = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_expansions");
  api.scored_candidates = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_scored_candidates");
  api.projection_reads = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_projection_reads");
  api.carry_hits = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_carry_hits");
  api.carry_misses = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_carry_misses");
  api.carry_admissions = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_carry_admissions");
  api.carry_bytes = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_carry_bytes");
  api.pair_cache_hits = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_pair_cache_hits");
  api.pair_cache_misses = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_pair_cache_misses");
  api.pair_cache_disabled_operations = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_pair_cache_disabled_operations");
  api.owner_visit_total = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_owner_visit_total");
  api.last_query_visits = (uint64_t (*)(void))require_symbol(
      module, "vectorlite_test_last_query_visits");
  printf("SYMBOLS resolved %d Debug-only instrumentation exports from %s\n",
         TEST_API_SYMBOL_COUNT, extension);
  return api;
}

/* --- C3 statement-execution trace and counter snapshots ------------------ */

/* All exported query counters for one measurement window. The host trace
 * observes total statement executions; every other field is a Debug-only
 * exported counter. */
typedef struct QueryCounters {
  uint64_t identity_3table;
  uint64_t fallback_3table;
  uint64_t node_expand;
  uint64_t node_score;
  uint64_t node_identity;
  uint64_t setup;
  uint64_t retired_admissions;
  uint64_t retired_fallbacks;
  uint64_t retired_size;
  uint64_t decision;
  uint64_t expansions;
  uint64_t scored;
  uint64_t projection;
  uint64_t carry_hits;
  uint64_t carry_misses;
  uint64_t carry_admissions;
  uint64_t carry_bytes;
  uint64_t pair_hits;
  uint64_t pair_misses;
  uint64_t pair_disabled;
  uint64_t owner_visits;
  uint64_t last_query_visits;
  uint64_t executions;
} QueryCounters;

static QueryCounters read_query_counters(void) {
  QueryCounters counters;
  counters.identity_3table = test_api.identity_3table_executions();
  counters.fallback_3table = test_api.fallback_3table_executions();
  counters.node_expand = test_api.node_expand_executions();
  counters.node_score = test_api.node_score_executions();
  counters.node_identity = test_api.node_identity_executions();
  counters.setup = test_api.setup_reads();
  counters.retired_admissions = test_api.retired_admissions();
  counters.retired_fallbacks = test_api.retired_fallbacks();
  counters.retired_size = test_api.retired_size();
  counters.decision = test_api.decision_reads();
  counters.expansions = test_api.expansions();
  counters.scored = test_api.scored_candidates();
  counters.projection = test_api.projection_reads();
  counters.carry_hits = test_api.carry_hits();
  counters.carry_misses = test_api.carry_misses();
  counters.carry_admissions = test_api.carry_admissions();
  counters.carry_bytes = test_api.carry_bytes();
  counters.pair_hits = test_api.pair_cache_hits();
  counters.pair_misses = test_api.pair_cache_misses();
  counters.pair_disabled = test_api.pair_cache_disabled_operations();
  counters.owner_visits = test_api.owner_visit_total();
  counters.last_query_visits = test_api.last_query_visits();
  counters.executions = 0;
  return counters;
}

static int trace_counting;
static uint64_t trace_statements;

/* Counts every statement SQLite starts executing while a measured query runs.
 * The host's own public KNN statement is the only non-extension execution and is
 * excluded by its SQL text, so the remainder is exactly the extension's
 * per-shape statement work (setup reads, expansions, scored candidates,
 * decisions and projections). */
static int trace_callback(unsigned type, void *unused, void *statement,
                          void *sql_text) {
  const char *sql = (const char *)sql_text;
  (void)unused;
  (void)statement;
  if (type == SQLITE_TRACE_STMT && trace_counting) {
    /* The host's own public KNN statement is the single non-extension
     * execution in the window; extension statements are prepared inside the
     * callbacks and never begin with "SELECT rowid". */
    if (sql == NULL || strncmp(sql, "SELECT rowid", 12) != 0) {
      ++trace_statements;
    }
  }
  return 0;
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
  REQUIRE(status_u64(db, table, "\"published_count\":") == (uint64_t)expected_rows);
  return population;
}

/* --- population ----------------------------------------------------------- */

static void generate_vector(int64_t rowid, float *out) {
  for (int index = 0; index < DIMENSION; ++index) {
    /* Deterministic distinct vectors: v[j] = sin(rowid*0.01 + j). */
    out[index] = (float)sin((double)rowid * 0.01 + (double)index);
  }
}

static void populate(sqlite3 *db, const char *table, int64_t rows) {
  char sql[256];
  sqlite3_stmt *statement = NULL;
  int written = snprintf(sql, sizeof(sql),
                         "INSERT INTO %s(rowid,embedding) VALUES(?1,?2)",
                         table);
  REQUIRE(written > 0 && (size_t)written < sizeof(sql));
  REQUIRE(sqlite3_prepare_v2(db, sql, -1, &statement, NULL) == SQLITE_OK);
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

/* --- supported churn ------------------------------------------------------ */

/* Deletes [first, last] through the public virtual table so retire() maintains
 * the canonical and published state exactly as an application DELETE would. */
static void churn_rows(sqlite3 *db, const char *table, int64_t first,
                       int64_t last) {
  char sql[256];
  sqlite3_stmt *statement = NULL;
  int written = snprintf(sql, sizeof(sql), "DELETE FROM %s WHERE rowid=?1",
                         table);
  REQUIRE(written > 0 && (size_t)written < sizeof(sql));
  REQUIRE(sqlite3_prepare_v2(db, sql, -1, &statement, NULL) == SQLITE_OK);
  for (int64_t rowid = first; rowid <= last; ++rowid) {
    REQUIRE(sqlite3_bind_int64(statement, 1, rowid) == SQLITE_OK);
    REQUIRE(sqlite3_step(statement) == SQLITE_DONE);
    REQUIRE(sqlite3_reset(statement) == SQLITE_OK);
  }
  REQUIRE(sqlite3_finalize(statement) == SQLITE_OK);
}

/* --- measured public KNN queries ----------------------------------------- */

static void knn_sql(char *out, size_t capacity, const char *table, int k, int l) {
  int written = snprintf(
      out, capacity,
      "SELECT rowid FROM %s WHERE knn_search(embedding, "
      "knn_param(?1,%d,'{\"search_list_size\":%d}'))",
      table, k, l);
  REQUIRE(written > 0 && (size_t)written < capacity);
}

typedef struct SuccessMeasurement {
  int k;
  int l;
  size_t rows;
  Counters counters;
  uint64_t last_search_l;
  uint64_t reader_executions_delta;
  uint64_t batch_calls_delta;
  /* Per-window delta of every exported query counter, plus the host-observed
   * statement executions for the C3 identity. */
  QueryCounters query;
  char sql[512];
} SuccessMeasurement;

static QueryCounters counter_delta(QueryCounters before, QueryCounters after) {
  QueryCounters delta;
  delta.identity_3table = after.identity_3table - before.identity_3table;
  delta.fallback_3table = after.fallback_3table - before.fallback_3table;
  delta.node_expand = after.node_expand - before.node_expand;
  delta.node_score = after.node_score - before.node_score;
  delta.node_identity = after.node_identity - before.node_identity;
  delta.setup = after.setup - before.setup;
  delta.retired_admissions = after.retired_admissions - before.retired_admissions;
  delta.retired_fallbacks = after.retired_fallbacks - before.retired_fallbacks;
  delta.retired_size = after.retired_size;
  delta.decision = after.decision - before.decision;
  delta.expansions = after.expansions - before.expansions;
  delta.scored = after.scored - before.scored;
  delta.projection = after.projection - before.projection;
  delta.carry_hits = after.carry_hits - before.carry_hits;
  delta.carry_misses = after.carry_misses - before.carry_misses;
  delta.carry_admissions = after.carry_admissions - before.carry_admissions;
  delta.carry_bytes = after.carry_bytes;
  delta.pair_hits = after.pair_hits - before.pair_hits;
  delta.pair_misses = after.pair_misses - before.pair_misses;
  delta.pair_disabled = after.pair_disabled - before.pair_disabled;
  delta.owner_visits = after.owner_visits - before.owner_visits;
  delta.last_query_visits = after.last_query_visits;
  delta.executions = after.executions - before.executions;
  return delta;
}

/* Resets the statement counters, runs exactly the public C3 query, fetches and
 * discards every row, finalizes the host statement and reads the counters. */
static SuccessMeasurement measure_success(sqlite3 *db, const char *table,
                                          int k, int l) {
  SuccessMeasurement measurement;
  sqlite3_stmt *statement = NULL;
  uint64_t readers_before;
  uint64_t batches_before;
  QueryCounters before;
  int rc;
  memset(&measurement, 0, sizeof(measurement));
  measurement.k = k;
  measurement.l = l;
  knn_sql(measurement.sql, sizeof(measurement.sql), table, k, l);
  test_api.reset_counters();
  readers_before = test_api.reader_executions();
  batches_before = test_api.batch_calls();
  before = read_query_counters();
  trace_statements = 0;
  trace_counting = 1;
  REQUIRE(sqlite3_trace_v2(db, SQLITE_TRACE_STMT, trace_callback, NULL) ==
          SQLITE_OK);
  REQUIRE(sqlite3_prepare_v2(db, measurement.sql, -1, &statement, NULL) ==
          SQLITE_OK);
  REQUIRE(sqlite3_bind_blob(statement, 1, query_vector,
                            (int)sizeof(query_vector), SQLITE_STATIC) ==
          SQLITE_OK);
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) ++measurement.rows;
  REQUIRE(rc == SQLITE_DONE);
  REQUIRE(sqlite3_finalize(statement) == SQLITE_OK);
  trace_counting = 0;
  REQUIRE(sqlite3_trace_v2(db, 0, NULL, NULL) == SQLITE_OK);
  measurement.counters = read_counters();
  measurement.last_search_l = test_api.last_search_l();
  measurement.reader_executions_delta =
      test_api.reader_executions() - readers_before;
  measurement.batch_calls_delta = test_api.batch_calls() - batches_before;
  measurement.query = counter_delta(before, read_query_counters());
  measurement.query.executions = trace_statements;
  return measurement;
}

typedef struct ErrorMeasurement {
  int k;
  int l;
  size_t rows;
  int step_code;
  int finalize_code;
  int primary_code;
  int extended_code;
  char message[256];
  Counters before_finalize;
  Counters after_finalize;
  uint64_t last_search_l;
  uint64_t reader_executions_delta;
  uint64_t batch_calls_delta;
  char sql[512];
} ErrorMeasurement;

static ErrorMeasurement measure_forced_error(sqlite3 *db, const char *table,
                                             int k, int l) {
  ErrorMeasurement measurement;
  sqlite3_stmt *statement = NULL;
  uint64_t readers_before;
  uint64_t batches_before;
  int rc;
  memset(&measurement, 0, sizeof(measurement));
  measurement.k = k;
  measurement.l = l;
  knn_sql(measurement.sql, sizeof(measurement.sql), table, k, l);
  test_api.reset_counters();
  readers_before = test_api.reader_executions();
  batches_before = test_api.batch_calls();
  REQUIRE(sqlite3_prepare_v2(db, measurement.sql, -1, &statement, NULL) ==
          SQLITE_OK);
  REQUIRE(sqlite3_bind_blob(statement, 1, query_vector,
                            (int)sizeof(query_vector), SQLITE_STATIC) ==
          SQLITE_OK);
  rc = sqlite3_step(statement);
  while (rc == SQLITE_ROW) {
    ++measurement.rows;
    rc = sqlite3_step(statement);
  }
  measurement.step_code = rc;
  measurement.primary_code = sqlite3_errcode(db);
  measurement.extended_code = sqlite3_extended_errcode(db);
  snprintf(measurement.message, sizeof(measurement.message), "%s",
           sqlite3_errmsg(db));
  measurement.before_finalize = read_counters();
  /* Finalizing a failed statement repeats its error; the counters are what
   * matters, so the finalize status is recorded rather than required OK. */
  measurement.finalize_code = sqlite3_finalize(statement);
  measurement.after_finalize = read_counters();
  measurement.last_search_l = test_api.last_search_l();
  measurement.reader_executions_delta =
      test_api.reader_executions() - readers_before;
  measurement.batch_calls_delta = test_api.batch_calls() - batches_before;
  return measurement;
}

/* --- result-capturing measurements for the Debug-hook checks -------------- */

/* The relative-drift checks compare the exact returned answer, so the host keeps
 * both the public rowid and the distance bits. Distance is an f32 widened by the
 * extension to a REAL, so the widening is lossless and a byte comparison of the
 * doubles is a bit comparison of the extension's answers. */
typedef struct KnnResult {
  int64_t rowid;
  double distance;
} KnnResult;

typedef struct KnnList {
  size_t count;
  KnnResult rows[RESULT_SLOTS];
} KnnList;

static int knn_lists_identical(const KnnList *expected, const KnnList *actual) {
  if (expected->count != actual->count) return 0;
  for (size_t index = 0; index < expected->count; ++index) {
    if (expected->rows[index].rowid != actual->rows[index].rowid) return 0;
    if (memcmp(&expected->rows[index].distance, &actual->rows[index].distance,
               sizeof(double)) != 0) {
      return 0;
    }
  }
  return 1;
}

typedef struct KnnMeasurement {
  size_t rows;
  /* SQLITE_DONE on success; the failing step code otherwise. */
  int step_code;
  int extended_code;
  int finalize_code;
  char message[256];
  /* Read after the host statement is finalized, so the balance covers the
   * cursor close. */
  Counters after_finalize;
  uint64_t prepare_attempts;
  uint64_t last_search_l;
  uint64_t reader_executions_delta;
  QueryCounters query;
  KnnList list;
  char sql[512];
} KnnMeasurement;

static void knn_rows_sql(char *out, size_t capacity, const char *table, int k,
                         int l) {
  int written = snprintf(
      out, capacity,
      "SELECT rowid, distance FROM %s WHERE knn_search(embedding, "
      "knn_param(?1,%d,'{\"search_list_size\":%d}'))",
      table, k, l);
  REQUIRE(written > 0 && (size_t)written < capacity);
}

/* Resets the statement counters, runs exactly the public result-capturing query,
 * records every returned (rowid, distance) pair even on a failing step, and
 * reads the final counters after the host statement is finalized, so lifecycle
 * balance is measured after the cursor closes. A failing step is recorded rather
 * than required, because residual-error checks are machine-checked verdicts. */
static KnnMeasurement measure_knn_rows(sqlite3 *db, const char *table, int k,
                                       int l) {
  KnnMeasurement measurement;
  sqlite3_stmt *statement = NULL;
  uint64_t readers_before;
  QueryCounters before;
  int rc;
  memset(&measurement, 0, sizeof(measurement));
  REQUIRE(k > 0 && k <= RESULT_SLOTS);
  knn_rows_sql(measurement.sql, sizeof(measurement.sql), table, k, l);
  test_api.reset_counters();
  readers_before = test_api.reader_executions();
  before = read_query_counters();
  trace_statements = 0;
  trace_counting = 1;
  REQUIRE(sqlite3_trace_v2(db, SQLITE_TRACE_STMT, trace_callback, NULL) ==
          SQLITE_OK);
  REQUIRE(sqlite3_prepare_v2(db, measurement.sql, -1, &statement, NULL) ==
          SQLITE_OK);
  REQUIRE(sqlite3_bind_blob(statement, 1, query_vector,
                            (int)sizeof(query_vector), SQLITE_STATIC) ==
          SQLITE_OK);
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
    KnnResult *result;
    REQUIRE(measurement.list.count < RESULT_SLOTS);
    result = &measurement.list.rows[measurement.list.count];
    result->rowid = sqlite3_column_int64(statement, 0);
    result->distance = sqlite3_column_double(statement, 1);
    ++measurement.list.count;
  }
  measurement.rows = measurement.list.count;
  measurement.step_code = rc;
  measurement.extended_code = sqlite3_extended_errcode(db);
  snprintf(measurement.message, sizeof(measurement.message), "%s",
           sqlite3_errmsg(db));
  /* Finalizing a failed statement repeats its error; the counters are what
   * matters, so the finalize status is recorded rather than required OK. */
  measurement.finalize_code = sqlite3_finalize(statement);
  trace_counting = 0;
  REQUIRE(sqlite3_trace_v2(db, 0, NULL, NULL) == SQLITE_OK);
  measurement.after_finalize = read_counters();
  measurement.prepare_attempts = test_api.prepare_attempts();
  measurement.last_search_l = test_api.last_search_l();
  measurement.reader_executions_delta =
      test_api.reader_executions() - readers_before;
  measurement.query = counter_delta(before, read_query_counters());
  measurement.query.executions = trace_statements;
  return measurement;
}

/* --- C3 identity checks --------------------------------------------------- */

/* Asserts the C3 instrumented identity and the recorded call-map semantics on
 * one measured success query. The host trace supplies total statement
 * executions; every other term is an exported Debug-only counter. */
static void check_c3_identity(const char *label, const SuccessMeasurement *m) {
  const QueryCounters *q = &m->query;
  uint64_t terms = q->setup + q->expansions + q->scored + q->decision +
                   q->projection;
  uint64_t owner_visits = q->expansions + 2 * q->scored + q->decision +
                          2 * (uint64_t)m->k;
  printf("C3_IDENTITY %s executions=%" PRIu64 " terms=%" PRIu64
         " setup=%" PRIu64 " expansions=%" PRIu64 " scored=%" PRIu64
         " decision=%" PRIu64 " projection=%" PRIu64
         " identity_3table=%" PRIu64 " fallback_3table=%" PRIu64
         " carry_hits=%" PRIu64 " carry_misses=%" PRIu64
         " carry_admissions=%" PRIu64 " carry_bytes=%" PRIu64
         " pair_hits=%" PRIu64 " pair_misses=%" PRIu64
         " pair_disabled=%" PRIu64
         " last_query_visits=%" PRIu64 " owner_visits_expected=%" PRIu64
         " retired_admissions=%" PRIu64 " retired_fallbacks=%" PRIu64
         " retired_size=%" PRIu64 "\n",
         label, q->executions, terms, q->setup, q->expansions, q->scored,
         q->decision, q->projection, q->identity_3table, q->fallback_3table,
         q->carry_hits, q->carry_misses, q->carry_admissions, q->carry_bytes,
         q->pair_hits, q->pair_misses, q->pair_disabled, q->last_query_visits,
         owner_visits, q->retired_admissions, q->retired_fallbacks,
         q->retired_size);
  CHECK(q->identity_3table == 0);
  CHECK(q->fallback_3table == 0);
  /* Fixture-qualified constant (C3/D8): metadata, generation, root node, retired
   * count, root identity, root state, root decision, stamp capture, stamp
   * verify. Other routes keep their own instrumented counts. */
  CHECK(q->setup == 9);
  CHECK(q->executions == terms);
  CHECK(q->decision == (uint64_t)m->l);
  CHECK(q->projection == (uint64_t)m->k);
  CHECK(q->carry_admissions == 1);
  CHECK(q->carry_bytes > 0);
  /* The final top-k loop is resolved from the carry: exactly k hits with zero
   * additional SQL, so the miss count stays at the decision/setup scale. */
  CHECK(q->carry_hits == (uint64_t)m->k);
  /* The pair cache is admitted on this fixture and the pinned call map scores
   * each expansion's neighbours before the next expansion, so every scored id
   * after the start point is a hit: no pair-cache miss hides a NodeIdentity read
   * inside the execution identity above. */
  CHECK(q->pair_disabled == 0);
  CHECK(q->pair_misses == 0);
  CHECK(q->pair_hits > 0);
  CHECK(q->retired_admissions == 0);
  CHECK(q->retired_fallbacks == 0);
  CHECK(q->retired_size == 0);
  CHECK(q->last_query_visits == owner_visits);
  CHECK(m->last_search_l == (uint64_t)m->l);
  CHECK(m->counters.prepares == m->counters.finalizes);
}

/* --- test body ------------------------------------------------------------ */

static int64_t create_and_publish(sqlite3 *db, const char *table,
                                  const char *options, int64_t rows) {
  char ddl[512];
  int written = snprintf(
      ddl, sizeof(ddl),
      "CREATE VIRTUAL TABLE %s USING vectorlite(embedding float32[%d] "
      "l2,diskann(%s))",
      table, DIMENSION, options);
  REQUIRE(written > 0 && (size_t)written < sizeof(ddl));
  printf("DDL %s\n", ddl);
  execute(db, ddl);
  populate(db, table, rows);
  REQUIRE(status_u64(db, table, "\"pending_count\":") == (uint64_t)rows);
  return (int64_t)publish_index(db, table, rows);
}

int main(int argc, char **argv) {
  const char *extension;
  const char *database;
  sqlite3 *db;
  int64_t success_population;
  int64_t error_population;
  int64_t partial_population;
  SuccessMeasurement low;
  SuccessMeasurement high;
  ErrorMeasurement forced;
  KnnMeasurement normal_result;
  KnnMeasurement pair_off;
  KnnMeasurement demoted;
  KnnMeasurement recovered;
  KnnMeasurement construction_all;
  KnnMeasurement construction_identity;
  KnnMeasurement partial;
  KnnMeasurement fallback;
  KnnMeasurement reference;
  int64_t fallback_population;
  int64_t reference_population;
  uint64_t prepare_delta;
  FILE *reserved;

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
  memset(query_vector, 0, sizeof(query_vector));
  for (int index = 0; index < DIMENSION; ++index) {
    query_vector[index] = (float)cos((double)index * 0.25);
  }

  /* One application connection drives the public deferred lifecycle (each
   * control call is its own autocommit INSERT) and then the measured queries. */
  db = open_host(extension, database);
  test_api = resolve_test_api(extension);
  success_population =
      create_and_publish(db, SUCCESS_TABLE, SUCCESS_OPTIONS, SUCCESS_ROWS);
  REQUIRE(success_population == SUCCESS_ROWS);
  printf("PUBLISHED table=%s rows=%" PRIu64 "\n", SUCCESS_TABLE,
         (uint64_t)success_population);
  error_population = create_and_publish(db, FORCED_ERROR_TABLE,
                                        FORCED_ERROR_OPTIONS, FORCED_ERROR_ROWS);
  REQUIRE(error_population == FORCED_ERROR_ROWS);
  printf("PUBLISHED table=%s rows=%" PRIu64 "\n", FORCED_ERROR_TABLE,
         (uint64_t)error_population);
  partial_population =
      create_and_publish(db, PARTIAL_ADMISSION_TABLE,
                         PARTIAL_ADMISSION_OPTIONS, PARTIAL_ADMISSION_ROWS);
  REQUIRE(partial_population == PARTIAL_ADMISSION_ROWS);
  printf("PUBLISHED table=%s rows=%" PRIu64 "\n", PARTIAL_ADMISSION_TABLE,
         (uint64_t)partial_population);
  fallback_population = create_and_publish(
      db, FALLBACK_TABLE, FALLBACK_OPTIONS, FALLBACK_ROWS);
  REQUIRE(fallback_population == FALLBACK_ROWS);
  printf("PUBLISHED table=%s rows=%" PRIu64 " fallback_cache_bytes=%" PRIu64
         "\n",
         FALLBACK_TABLE, (uint64_t)fallback_population, FALLBACK_CACHE_BYTES);
  reference_population = create_and_publish(
      db, FALLBACK_REFERENCE_TABLE, FALLBACK_REFERENCE_OPTIONS, FALLBACK_ROWS);
  REQUIRE(reference_population == FALLBACK_ROWS);
  printf("PUBLISHED table=%s rows=%" PRIu64 "\n", FALLBACK_REFERENCE_TABLE,
         (uint64_t)reference_population);
  /* Supported churn: ten application DELETEs on both twins, so the published
   * generation carries exactly ten retired entries and the D3 count invariant
   * holds on both. */
  churn_rows(db, FALLBACK_TABLE, FALLBACK_ROWS - 9, FALLBACK_ROWS);
  churn_rows(db, FALLBACK_REFERENCE_TABLE, FALLBACK_ROWS - 9, FALLBACK_ROWS);
  REQUIRE(status_u64(db, FALLBACK_TABLE, "\"eligible_count\":") ==
          (uint64_t)(FALLBACK_ROWS - 10));
  REQUIRE(status_u64(db, FALLBACK_REFERENCE_TABLE, "\"eligible_count\":") ==
          (uint64_t)(FALLBACK_ROWS - 10));
  REQUIRE(status_u64(db, FALLBACK_TABLE, "\"published_count\":") ==
          (uint64_t)FALLBACK_ROWS);
  REQUIRE(sqlite3_get_autocommit(db) != 0);

  /* Warm the query shape once; the measured statements below are the exact
   * public C3 SQL and are still the only statements counted. */
  {
    SuccessMeasurement warmup =
        measure_success(db, SUCCESS_TABLE, SUCCESS_K, SUCCESS_LOW_L);
    REQUIRE(warmup.rows == (size_t)SUCCESS_K);
  }
  low = measure_success(db, SUCCESS_TABLE, SUCCESS_K, SUCCESS_LOW_L);
  high = measure_success(db, SUCCESS_TABLE, SUCCESS_K, SUCCESS_HIGH_L);
  printf("SQL_L10 %s\n", low.sql);
  printf("SQL_L100 %s\n", high.sql);
  printf("MEASURED_L10 prepares=%" PRIu64 " finalizes=%" PRIu64
         " rows=%zu last_search_l=%" PRIu64 " reader_executions_delta=%" PRIu64
         " batch_calls_delta=%" PRIu64 "\n",
         low.counters.prepares, low.counters.finalizes, low.rows,
         low.last_search_l, low.reader_executions_delta,
         low.batch_calls_delta);
  printf("MEASURED_L100 prepares=%" PRIu64 " finalizes=%" PRIu64
         " rows=%zu last_search_l=%" PRIu64 " reader_executions_delta=%" PRIu64
         " batch_calls_delta=%" PRIu64 "\n",
         high.counters.prepares, high.counters.finalizes, high.rows,
         high.last_search_l, high.reader_executions_delta,
         high.batch_calls_delta);

  CHECK(low.rows == (size_t)SUCCESS_K);
  CHECK(high.rows == (size_t)SUCCESS_K);
  CHECK(low.counters.prepares <= PREPARE_LIMIT);
  CHECK(high.counters.prepares <= PREPARE_LIMIT);
  prepare_delta = low.counters.prepares > high.counters.prepares
                      ? low.counters.prepares - high.counters.prepares
                      : high.counters.prepares - low.counters.prepares;
  CHECK(prepare_delta <= PREPARE_DELTA_LIMIT);
  CHECK(low.counters.prepares == low.counters.finalizes);
  CHECK(high.counters.prepares == high.counters.finalizes);
  CHECK(low.last_search_l == (uint64_t)SUCCESS_LOW_L);
  CHECK(high.last_search_l == (uint64_t)SUCCESS_HIGH_L);
  CHECK(low.reader_executions_delta > 0);
  CHECK(high.reader_executions_delta > 0);
  CHECK(test_api.reader_executions() > 0);
  CHECK(low.batch_calls_delta > 0);
  CHECK(high.batch_calls_delta > 0);
  CHECK(test_api.batch_calls() > 0);

  /* C3 instrumented identity: the host-observed statement executions must equal
   * the exported setup/expansion/scored/decision/projection terms, with zero
   * three-table identity joins, decision reads == L, projection reads == k and
   * the carry resolving the final top-k loop with zero SQL. */
  check_c3_identity("L10", &low);
  check_c3_identity("L100", &high);

  /* --- D3 three-table fallback route: the churned generation at the pinned
   * minimum budget cannot admit its retired set, so the whole query keeps the
   * format-5-aware three-table reader. The twin table at the default budget is
   * the graph-route reference over the same churned population: results, order,
   * distance bits and owner-visit totals must be identical. --- */
  fallback = measure_knn_rows(db, FALLBACK_TABLE, FALLBACK_K, FALLBACK_L);
  reference =
      measure_knn_rows(db, FALLBACK_REFERENCE_TABLE, FALLBACK_K, FALLBACK_L);
  printf("MEASURED_FALLBACK step_code=%d rows=%zu prepares=%" PRIu64
         " finalizes=%" PRIu64 " identity_3table=%" PRIu64
         " fallback_3table=%" PRIu64 " retired_admissions=%" PRIu64
         " retired_fallbacks=%" PRIu64 " retired_size=%" PRIu64
         " last_query_visits=%" PRIu64 " rows_equal_ref=%d\n",
         fallback.step_code, fallback.rows, fallback.after_finalize.prepares,
         fallback.after_finalize.finalizes, fallback.query.identity_3table,
         fallback.query.fallback_3table, fallback.query.retired_admissions,
         fallback.query.retired_fallbacks, fallback.query.retired_size,
         fallback.query.last_query_visits,
         knn_lists_identical(&reference.list, &fallback.list));
  printf("MEASURED_FALLBACK_REFERENCE step_code=%d rows=%zu identity_3table=%"
         PRIu64 " fallback_3table=%" PRIu64 " retired_admissions=%" PRIu64
         " retired_fallbacks=%" PRIu64 " last_query_visits=%" PRIu64 "\n",
         reference.step_code, reference.rows, reference.query.identity_3table,
         reference.query.fallback_3table, reference.query.retired_admissions,
         reference.query.retired_fallbacks, reference.query.last_query_visits);
  CHECK(fallback.step_code == SQLITE_DONE);
  CHECK(fallback.rows == (size_t)FALLBACK_K);
  CHECK(reference.step_code == SQLITE_DONE);
  CHECK(reference.rows == (size_t)FALLBACK_K);
  CHECK(knn_lists_identical(&reference.list, &fallback.list));
  CHECK(fallback.after_finalize.prepares == fallback.after_finalize.finalizes);
  /* The C3 instrumented identity is defined for format-served reader routes;
   * the fallback deliberately counts its three-table executions separately
   * instead, so only the route/visit parity below is asserted here. */
  CHECK(fallback.query.identity_3table > 0);
  CHECK(fallback.query.fallback_3table > 0);
  CHECK(fallback.query.retired_fallbacks == 1);
  CHECK(fallback.query.retired_admissions == 0);
  CHECK(fallback.query.retired_size == 10);
  CHECK(fallback.query.decision == (uint64_t)FALLBACK_L);
  CHECK(fallback.query.projection == (uint64_t)FALLBACK_K);
  CHECK(fallback.query.last_query_visits == reference.query.last_query_visits);
  CHECK(reference.query.identity_3table == 0);
  CHECK(reference.query.fallback_3table == 0);
  CHECK(reference.query.retired_admissions == 1);
  CHECK(reference.query.retired_fallbacks == 0);
  CHECK(reference.after_finalize.prepares ==
        reference.after_finalize.finalizes);

  forced = measure_forced_error(db, FORCED_ERROR_TABLE, FORCED_ERROR_K,
                                FORCED_ERROR_L);
  printf("SQL_FORCED_ERROR %s\n", forced.sql);
  printf("ERROR_POINT step_code=%d extended_code=%d primary_code=%d "
         "finalize_code=%d rows=%zu message=\"%s\"\n",
         forced.step_code, forced.extended_code, forced.primary_code,
         forced.finalize_code, forced.rows, forced.message);
  CHECK(forced.rows == 0);
  CHECK(forced.step_code != SQLITE_DONE);
  CHECK((forced.step_code & 0xff) == SQLITE_TOOBIG);
  CHECK(forced.extended_code == SQLITE_TOOBIG);
  CHECK(forced.primary_code == SQLITE_TOOBIG);
  CHECK(forced.reader_executions_delta > 0);
  CHECK(forced.before_finalize.prepares >= forced.before_finalize.finalizes);
  CHECK(forced.after_finalize.prepares == forced.after_finalize.finalizes);
  printf("MEASURED_FORCED_ERROR prepares=%" PRIu64 " finalizes=%" PRIu64
         " last_search_l=%" PRIu64 " reader_executions_delta=%" PRIu64
         " batch_calls_delta=%" PRIu64 "\n",
         forced.after_finalize.prepares, forced.after_finalize.finalizes,
         forced.last_search_l, forced.reader_executions_delta,
         forced.batch_calls_delta);

  /* --- forced demotion: the next query demotes every admitted reader after
   * its first completed read, so it must answer the identical public query
   * bit-identically while preparing far more statements than the retained reader
   * set. The hook is one-shot: the following query must return to the normal
   * prepare count. --- */
  normal_result =
      measure_knn_rows(db, SUCCESS_TABLE, SUCCESS_K, SUCCESS_LOW_L);
  /* --- D6 pair cache: the Debug-only one-shot hook disables the cache for the
   * next operation. The answer (rowids, order, distance bits) and the owner
   * visit totals must be bit-identical to the cache-on query; every scored id
   * that hit the cache then performs exactly one extra NodeIdentity read at the
   * same charge, and the operation reports the cache as disabled. --- */
  test_api.force_pair_cache_off();
  pair_off = measure_knn_rows(db, SUCCESS_TABLE, SUCCESS_K, SUCCESS_LOW_L);
  printf("MEASURED_PAIR_CACHE step_code=%d rows=%zu prepares=%" PRIu64
         " finalizes=%" PRIu64 " hits_cached=%" PRIu64
         " misses_cached=%" PRIu64 " misses_off=%" PRIu64
         " disabled_cached=%" PRIu64 " disabled_off=%" PRIu64
         " identity_cached=%" PRIu64 " identity_off=%" PRIu64
         " visits_cached=%" PRIu64 " visits_off=%" PRIu64
         " rows_equal=%d\n",
         pair_off.step_code, pair_off.rows, pair_off.after_finalize.prepares,
         pair_off.after_finalize.finalizes, normal_result.query.pair_hits,
         normal_result.query.pair_misses, pair_off.query.pair_misses,
         normal_result.query.pair_disabled, pair_off.query.pair_disabled,
         normal_result.query.node_identity, pair_off.query.node_identity,
         normal_result.query.last_query_visits,
         pair_off.query.last_query_visits,
         knn_lists_identical(&normal_result.list, &pair_off.list));
  CHECK(pair_off.step_code == SQLITE_DONE);
  CHECK(pair_off.rows == (size_t)SUCCESS_K);
  CHECK(knn_lists_identical(&normal_result.list, &pair_off.list));
  CHECK(pair_off.after_finalize.prepares == pair_off.after_finalize.finalizes);
  CHECK(normal_result.query.pair_disabled == 0);
  CHECK(pair_off.query.pair_disabled == 1);
  CHECK(normal_result.query.pair_misses == 0);
  CHECK(pair_off.query.pair_hits == 0);
  /* The disabled operation resolves every scored id through one identity read:
   * exactly the set of ids the cache-on query hit, with no extra visit. */
  CHECK(pair_off.query.pair_misses ==
        normal_result.query.pair_hits + normal_result.query.pair_misses);
  CHECK(pair_off.query.node_identity - normal_result.query.node_identity ==
        normal_result.query.pair_hits);
  CHECK(pair_off.query.owner_visits == normal_result.query.owner_visits);
  CHECK(pair_off.query.last_query_visits ==
        normal_result.query.last_query_visits);
  test_api.force_demote();
  demoted = measure_knn_rows(db, SUCCESS_TABLE, SUCCESS_K, SUCCESS_LOW_L);
  recovered = measure_knn_rows(db, SUCCESS_TABLE, SUCCESS_K, SUCCESS_LOW_L);
  printf("SQL_RESULT_L10 %s\n", normal_result.sql);
  printf("MEASURED_NORMAL_RESULT step_code=%d rows=%zu prepares=%" PRIu64
         " finalizes=%" PRIu64 " attempts=%" PRIu64 "\n",
         normal_result.step_code, normal_result.rows,
         normal_result.after_finalize.prepares,
         normal_result.after_finalize.finalizes, normal_result.prepare_attempts);
  printf("MEASURED_FORCED_DEMOTE step_code=%d rows=%zu prepares=%" PRIu64
         " finalizes=%" PRIu64 " attempts=%" PRIu64
         " reader_executions_delta=%" PRIu64 " rows_equal=%d\n",
         demoted.step_code, demoted.rows, demoted.after_finalize.prepares,
         demoted.after_finalize.finalizes, demoted.prepare_attempts,
         demoted.reader_executions_delta,
         knn_lists_identical(&normal_result.list, &demoted.list));
  printf("MEASURED_RECOVERED step_code=%d rows=%zu prepares=%" PRIu64
         " finalizes=%" PRIu64 " attempts=%" PRIu64 "\n",
         recovered.step_code, recovered.rows,
         recovered.after_finalize.prepares, recovered.after_finalize.finalizes,
         recovered.prepare_attempts);
  CHECK(normal_result.step_code == SQLITE_DONE);
  CHECK(normal_result.rows == (size_t)SUCCESS_K);
  CHECK(demoted.step_code == SQLITE_DONE);
  CHECK(demoted.rows == (size_t)SUCCESS_K);
  CHECK(knn_lists_identical(&normal_result.list, &demoted.list));
  CHECK(demoted.after_finalize.prepares == demoted.after_finalize.finalizes);
  CHECK(demoted.after_finalize.prepares >
        low.counters.prepares + (uint64_t)FORCED_DEMOTE_MIN_EXTRA_PREPARES);
  CHECK(recovered.step_code == SQLITE_DONE &&
        recovered.after_finalize.prepares == low.counters.prepares);

  /* --- construction failure: every selected reader shape fails its next real
   * sqlite3_prepare_v2 dispatch and silently falls back to scalar. The query
   * must surface no error, answer identically, balance finalization, and count
   * the failed attempts. --- */
  test_api.reset_counters();
  test_api.fail_reader_prepare((uint64_t)READER_SHAPE_ALL);
  construction_all =
      measure_knn_rows(db, SUCCESS_TABLE, SUCCESS_K, SUCCESS_LOW_L);
  test_api.reset_counters();
  test_api.fail_reader_prepare((uint64_t)READER_SHAPE_NODE_IDENTITY);
  construction_identity =
      measure_knn_rows(db, SUCCESS_TABLE, SUCCESS_K, SUCCESS_LOW_L);
  printf("MEASURED_CONSTRUCTION_FAILURE mask=0x%02X step_code=%d rows=%zu "
         "prepares=%" PRIu64 " finalizes=%" PRIu64 " attempts=%" PRIu64
         " reader_executions_delta=%" PRIu64 " rows_equal=%d\n",
         (unsigned)READER_SHAPE_ALL, construction_all.step_code,
         construction_all.rows, construction_all.after_finalize.prepares,
         construction_all.after_finalize.finalizes,
         construction_all.prepare_attempts,
         construction_all.reader_executions_delta,
         knn_lists_identical(&normal_result.list, &construction_all.list));
  printf("MEASURED_CONSTRUCTION_FAILURE mask=0x%02X step_code=%d rows=%zu "
         "prepares=%" PRIu64 " finalizes=%" PRIu64 " attempts=%" PRIu64
         " reader_executions_delta=%" PRIu64 " rows_equal=%d\n",
         (unsigned)READER_SHAPE_NODE_IDENTITY, construction_identity.step_code,
         construction_identity.rows,
         construction_identity.after_finalize.prepares,
         construction_identity.after_finalize.finalizes,
         construction_identity.prepare_attempts,
         construction_identity.reader_executions_delta,
         knn_lists_identical(&normal_result.list,
                             &construction_identity.list));
  CHECK(construction_all.step_code == SQLITE_DONE);
  CHECK(construction_all.rows == (size_t)SUCCESS_K);
  CHECK(knn_lists_identical(&normal_result.list, &construction_all.list));
  CHECK(construction_all.after_finalize.prepares ==
        construction_all.after_finalize.finalizes);
  CHECK(construction_all.prepare_attempts >
        construction_all.after_finalize.prepares);
  CHECK(knn_lists_identical(&normal_result.list,
                            &construction_identity.list));
  CHECK(construction_identity.prepare_attempts >=
        construction_identity.after_finalize.prepares + 1);

  /* --- partial admission: the scratch floor exceeds the whole owner remainder,
   * so admission closes on the first reader and no lower-priority shape may be
   * prepared. Core must reject the copied allowance exactly as HEAD does. --- */
  partial = measure_knn_rows(db, PARTIAL_ADMISSION_TABLE, PARTIAL_ADMISSION_K,
                             PARTIAL_ADMISSION_L);
  printf("MEASURED_PARTIAL_ADMISSION attempts=%" PRIu64 " prepares=%" PRIu64
         " finalizes=%" PRIu64 " rows=%zu step_code=%d extended_code=%d "
         "finalize_code=%d last_search_l=%" PRIu64 " message=\"%s\"\n",
         partial.prepare_attempts, partial.after_finalize.prepares,
         partial.after_finalize.finalizes, partial.rows, partial.step_code,
         partial.extended_code, partial.finalize_code, partial.last_search_l,
         partial.message);
  CHECK(partial.rows == 0);
  CHECK((partial.step_code & 0xff) == SQLITE_TOOBIG);
  CHECK(strcmp(partial.message, PARTIAL_ADMISSION_MESSAGE) == 0);
  CHECK(partial.after_finalize.prepares == partial.after_finalize.finalizes);
  /* Exactly five header prepares (metadata, generation, root node, retired
   * count, schema stamp) plus the one admission-closing reader try. */
  CHECK(partial.prepare_attempts == (uint64_t)PARTIAL_ADMISSION_MAX_ATTEMPTS);

  REQUIRE(sqlite3_close(db) == SQLITE_OK);

  printf("SUMMARY success_table=%s success_rows=%" PRIu64
         " prepares_l10=%" PRIu64 " prepares_l100=%" PRIu64
         " finalizes_l10=%" PRIu64 " finalizes_l100=%" PRIu64
         " last_search_l10=%" PRIu64 " last_search_l100=%" PRIu64
         " reader_executions_delta_l10=%" PRIu64
         " reader_executions_delta_l100=%" PRIu64
         " reader_executions_total=%" PRIu64
         " batch_calls_delta_l10=%" PRIu64 " batch_calls_delta_l100=%" PRIu64
         " batch_calls_total=%" PRIu64
         " forced_error_table=%s forced_error_rows=%" PRIu64
         " forced_error_code=%d forced_error_extended_code=%d"
         " forced_error_last_search_l=%" PRIu64
         " forced_error_prepares=%" PRIu64 " forced_error_finalizes=%" PRIu64
         " prepares_eq_finalizes_l10=%d prepares_eq_finalizes_l100=%d"
         " prepares_eq_finalizes_forced_error=%d"
         " normal_result_prepares=%" PRIu64
         " forced_demote_rows=%zu forced_demote_prepares=%" PRIu64
         " forced_demote_finalizes=%" PRIu64
         " forced_demote_attempts=%" PRIu64
         " forced_demote_reader_executions_delta=%" PRIu64
         " forced_demote_rows_equal=%d recovered_prepares=%" PRIu64
         " pair_cache_hits_cached=%" PRIu64
         " pair_cache_misses_off=%" PRIu64
         " pair_cache_identity_extra=%" PRIu64
         " pair_cache_disabled_off=%" PRIu64
         " pair_cache_rows_equal=%d pair_cache_visits_equal=%d"
         " construction_failure_mask=0x%02X"
         " construction_failure_attempts=%" PRIu64
         " construction_failure_prepares=%" PRIu64
         " construction_failure_rows_equal=%d"
         " node_identity_failure_mask=0x%02X"
         " node_identity_failure_attempts=%" PRIu64
         " node_identity_failure_prepares=%" PRIu64
         " node_identity_failure_rows_equal=%d"
         " partial_admission_table=%s partial_admission_rows_published=%" PRIu64
         " partial_admission_k=%d partial_admission_l=%d"
         " partial_admission_attempts=%" PRIu64
         " partial_admission_prepares=%" PRIu64
         " partial_admission_finalizes=%" PRIu64
         " partial_admission_error_code=%d"
         " partial_admission_rows=%zu\n",
         SUCCESS_TABLE, (uint64_t)success_population, low.counters.prepares,
         high.counters.prepares, low.counters.finalizes, high.counters.finalizes,
         low.last_search_l, high.last_search_l, low.reader_executions_delta,
         high.reader_executions_delta, test_api.reader_executions(),
         low.batch_calls_delta, high.batch_calls_delta, test_api.batch_calls(),
         FORCED_ERROR_TABLE, (uint64_t)error_population, forced.step_code,
         forced.extended_code, forced.last_search_l,
         forced.after_finalize.prepares, forced.after_finalize.finalizes,
         low.counters.prepares == low.counters.finalizes,
         high.counters.prepares == high.counters.finalizes,
         forced.after_finalize.prepares == forced.after_finalize.finalizes,
         normal_result.after_finalize.prepares, demoted.rows,
         demoted.after_finalize.prepares, demoted.after_finalize.finalizes,
         demoted.prepare_attempts, demoted.reader_executions_delta,
         knn_lists_identical(&normal_result.list, &demoted.list),
         recovered.after_finalize.prepares, normal_result.query.pair_hits,
         pair_off.query.pair_misses,
         pair_off.query.node_identity - normal_result.query.node_identity,
         pair_off.query.pair_disabled,
         knn_lists_identical(&normal_result.list, &pair_off.list),
         pair_off.query.owner_visits == normal_result.query.owner_visits,
         (unsigned)READER_SHAPE_ALL,
         construction_all.prepare_attempts,
         construction_all.after_finalize.prepares,
         knn_lists_identical(&normal_result.list, &construction_all.list),
         (unsigned)READER_SHAPE_NODE_IDENTITY,
         construction_identity.prepare_attempts,
         construction_identity.after_finalize.prepares,
         knn_lists_identical(&normal_result.list, &construction_identity.list),
         PARTIAL_ADMISSION_TABLE, (uint64_t)partial_population,
         PARTIAL_ADMISSION_K, PARTIAL_ADMISSION_L, partial.prepare_attempts,
         partial.after_finalize.prepares, partial.after_finalize.finalizes,
         partial.step_code, partial.rows);
  if (failures != 0) {
    printf("FAIL: %d of %d native query-count checks failed (SQLite %s)\n",
           failures, failures + checks, sqlite3_libversion());
    return 1;
  }
  printf("PASS: %d native query-count checks (SQLite %s)\n", checks,
         sqlite3_libversion());
  return 0;
}
