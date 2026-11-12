/* Focused native INSERT benchmark; each invocation builds exactly one index.
 * Optional VECTORLITE_BENCH_ environment controls: METRIC (l2/cosine),
 * DATABASE (fresh plain path; default :memory:), SNAPSHOT (fresh backup path),
 * TIMEOUT_SECONDS (default 90), MAX_VISITS (default 65536). No SQL tracing.
 * Dataset loading, queries and snapshot backup are outside the build timer. */
#include <sqlite3.h>
#include "vectorlite_batch.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
static double seconds(void) {
  LARGE_INTEGER counter, frequency;
  QueryPerformanceCounter(&counter);
  QueryPerformanceFrequency(&frequency);
  return (double)counter.QuadPart / (double)frequency.QuadPart;
}
#else
static double seconds(void) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}
#endif

static double deadline = 0.0;
static int interrupt_for_deadline(void *unused) {
  (void)unused;
  return seconds() > deadline;
}

static void require(sqlite3 *db, int status, int expected, const char *operation) {
  if (status != expected) {
    fprintf(stderr, "%s failed: %d %s\n", operation, status, sqlite3_errmsg(db));
    exit(1);
  }
}

static void execute(sqlite3 *db, const char *sql) {
  char *error = NULL;
  int status = sqlite3_exec(db, sql, NULL, NULL, &error);
  if (status != SQLITE_OK) {
    fprintf(stderr, "SQL failed: %d %s\n", status, error ? error : sqlite3_errmsg(db));
    sqlite3_free(error);
    exit(1);
  }
}

static unsigned char *read_exact(const char *path, size_t bytes) {
  FILE *source = fopen(path, "rb");
  if (!source) { perror(path); exit(1); }
  unsigned char *output = (unsigned char *)malloc(bytes);
  if (!output || fread(output, 1, bytes, source) != bytes || fgetc(source) != EOF) {
    fprintf(stderr, "Invalid dataset byte length: %s\n", path);
    exit(1);
  }
  fclose(source);
  return output;
}

/* Reserve with exclusive creation before SQLite opens the empty file. Never
 * point sqlite3_open at an unreserved caller-owned database. */
static int reserve_destination(const char *path) {
  /* SQLite special names/URIs could redirect away from the reserved file. */
  if (!path[0] || strcmp(path, ":memory:") == 0 || strncmp(path, "file:", 5) == 0) {
    fprintf(stderr, "Destination must be a plain filesystem path: %s\n", path);
    return 0;
  }
  FILE *file = fopen(path, "wbx");
  if (!file) { perror(path); return 0; }
  if (fclose(file) != 0) { perror(path); return 0; }
  return 1;
}

static int save_snapshot(sqlite3 *source, const char *path) {
  if (!reserve_destination(path)) return 0;
  sqlite3 *destination = NULL;
  sqlite3_backup *backup = NULL;
  int status = sqlite3_open(path, &destination);
  if (status == SQLITE_OK) {
    backup = sqlite3_backup_init(destination, "main", source, "main");
    if (!backup) status = sqlite3_errcode(destination);
  }
  if (backup) {
    status = sqlite3_backup_step(backup, -1);
    int finished = sqlite3_backup_finish(backup);
    if (status == SQLITE_DONE) status = finished;
    else if (status == SQLITE_OK) status = SQLITE_ERROR; /* Incomplete backup. */
  }
  if (status != SQLITE_OK) {
    fprintf(stderr, "Snapshot failed: %d %s\n", status,
            destination ? sqlite3_errmsg(destination) : "cannot open destination");
  }
  int closed = destination ? sqlite3_close(destination) : SQLITE_OK;
  if (closed != SQLITE_OK) fprintf(stderr, "Snapshot close failed: %d\n", closed);
  return status == SQLITE_OK && closed == SQLITE_OK;
}

static void json_string(const char *text) {
  putchar('"');
  for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
    if (*p == '"' || *p == '\\') printf("\\%c", *p);
    else if (*p < 0x20) printf("\\u%04x", *p);
    else putchar(*p); /* Preserve UTF-8 paths rather than escaping each byte. */
  }
  putchar('"');
}

static uint64_t random_state = 42;
static float random_float(void) {
  random_state ^= random_state << 13;
  random_state ^= random_state >> 7;
  random_state ^= random_state << 17;
  return (float)(random_state >> 40) / 16777216.0f;
}

static int step_done(sqlite3_stmt *statement) {
  int rc;
  do { rc = sqlite3_step(statement); } while (rc == SQLITE_ROW);
  return rc;
}

int main(int argc, char **argv) {
  if (argc != 5 && argc != 8) {
    fprintf(stderr, "usage: %s extension mode count dimension [vectors.bin queries.bin num_queries]\n", argv[0]);
    return 2;
  }
  if (sizeof(float) != 4 || sizeof(int64_t) != 8) return 2;
  const unsigned char endian[] = {1, 0, 0, 0};
  uint32_t endian_value;
  memcpy(&endian_value, endian, 4);
  if (endian_value != 1) { fprintf(stderr, "Benchmark requires little endian\n"); return 2; }
  int count = atoi(argv[3]), dimension = atoi(argv[4]);
  int query_count = argc == 8 ? atoi(argv[7]) : 16;
  if (count < 10 || count > 100000 || dimension <= 0 || dimension > 3000 ||
      query_count <= 0 || query_count > 1000) return 2;
  size_t elements = (size_t)count * (size_t)dimension;
  size_t query_elements = (size_t)query_count * (size_t)dimension;
  float *vectors, *queries;
  if (argc == 8) {
    vectors = (float *)read_exact(argv[5], elements * sizeof(float));
    queries = (float *)read_exact(argv[6], query_elements * sizeof(float));
  } else {
    vectors = (float *)malloc(elements * sizeof(float));
    queries = (float *)malloc(query_elements * sizeof(float));
    if (!vectors || !queries) return 1;
    for (size_t i = 0; i < elements; ++i) vectors[i] = random_float();
    for (size_t i = 0; i < query_elements; ++i) queries[i] = random_float();
  }
  int64_t *rowids = (int64_t *)malloc((size_t)count * sizeof(int64_t));
  if (!rowids) return 1;
  for (int i = 0; i < count; ++i) rowids[i] = i;

  const char *metric = getenv("VECTORLITE_BENCH_METRIC");
  if (!metric) metric = "l2";
  if (strcmp(metric, "l2") && strcmp(metric, "cosine")) {
    fprintf(stderr, "Invalid VECTORLITE_BENCH_METRIC (l2 or cosine)\n");
    return 2;
  }
  const char *database = getenv("VECTORLITE_BENCH_DATABASE");
  if (!database) database = ":memory:";
  const char *snapshot = getenv("VECTORLITE_BENCH_SNAPSHOT");
  double timeout = 90.0;
  const char *timeout_setting = getenv("VECTORLITE_BENCH_TIMEOUT_SECONDS");
  if (timeout_setting) {
    char *end = NULL;
    timeout = strtod(timeout_setting, &end);
    if (!timeout_setting[0] || !end || *end || !isfinite(timeout) || timeout <= 0) return 2;
  }
  if (strcmp(database, ":memory:") && !reserve_destination(database)) return 1;
  sqlite3 *db = NULL;
  require(db, sqlite3_open(database, &db), SQLITE_OK, "open");
  require(db, sqlite3_extended_result_codes(db, 1), SQLITE_OK, "extended errors");
  require(db, sqlite3_enable_load_extension(db, 1), SQLITE_OK, "enable extension");
  char *error = NULL;
  int status = sqlite3_load_extension(db, argv[1], NULL, &error);
  if (status != SQLITE_OK) {
    fprintf(stderr, "Load failed: %s\n", error ? error : sqlite3_errmsg(db));
    sqlite3_free(error);
    return 1;
  }
  require(db, sqlite3_enable_load_extension(db, 0), SQLITE_OK, "disable extension");
  char sql[512];
  unsigned long max_visits = 65536;
  const char *visit_setting = getenv("VECTORLITE_BENCH_MAX_VISITS");
  if (visit_setting) {
    char *end = NULL;
    max_visits = strtoul(visit_setting, &end, 10);
    if (!visit_setting[0] || !end || *end || max_visits < 100 || max_visits > UINT32_MAX) return 2;
  }
  int hnsw = strcmp(argv[2], "hnsw") == 0;
  int batch_size = strcmp(argv[2], "batch8") == 0 ? 8 :
                   strcmp(argv[2], "batch32") == 0 ? 32 : 0;
  if (!hnsw && batch_size == 0 && strcmp(argv[2], "single") != 0) return 2;
  if (hnsw) {
    snprintf(sql, sizeof(sql),
        "CREATE VIRTUAL TABLE v USING vectorlite(embedding float32[%d] %s,"
        "hnsw(max_elements=%d,ef_construction=100,M=30))", dimension, metric, count);
  } else {
    snprintf(sql, sizeof(sql),
        "CREATE VIRTUAL TABLE v USING vectorlite(embedding float32[%d] %s,"
        "diskann(degree=32,build_list_size=100,search_list_size=64,max_visits=%lu))",
        dimension, metric, max_visits);
  }
  execute(db, sql);
  deadline = seconds() + timeout;
  sqlite3_progress_handler(db, 1000, interrupt_for_deadline, NULL);
  double start = seconds();
  execute(db, "BEGIN");
  sqlite3_stmt *insert = NULL;
  if (batch_size) {
    require(db, sqlite3_prepare_v2(db,
        "INSERT INTO v(operation,embedding,path) VALUES('insert_batch',?1,?2)",
        -1, &insert, NULL), SQLITE_OK, "prepare batch");
    BatchF32V1 batch = {VECTORLITE_BATCH_F32_V1_ABI_VERSION,
        (uint32_t)sizeof(BatchF32V1), (uint64_t)count, (uint64_t)dimension, vectors, rowids};
    char options[64];
    snprintf(options, sizeof(options), "{\"batch_size\":%d}", batch_size);
    require(db, sqlite3_bind_pointer(insert, 1, &batch, VECTORLITE_BATCH_F32_V1_TAG, NULL),
            SQLITE_OK, "bind batch pointer");
    require(db, sqlite3_bind_text(insert, 2, options, -1, SQLITE_TRANSIENT), SQLITE_OK, "bind options");
    require(db, step_done(insert), SQLITE_DONE, "batch INSERT");
    require(db, sqlite3_finalize(insert), SQLITE_OK, "finalize batch");
  } else {
    require(db, sqlite3_prepare_v2(db,
        "INSERT INTO v(rowid,embedding) VALUES(?1,?2)", -1, &insert, NULL), SQLITE_OK, "prepare INSERT");
    for (int i = 0; i < count; ++i) {
      require(db, sqlite3_bind_int64(insert, 1, rowids[i]), SQLITE_OK, "bind rowid");
      require(db, sqlite3_bind_blob(insert, 2, vectors + (size_t)i * dimension,
          dimension * (int)sizeof(float), SQLITE_STATIC), SQLITE_OK, "bind vector");
      require(db, step_done(insert), SQLITE_DONE, "INSERT");
      require(db, sqlite3_reset(insert), SQLITE_OK, "reset INSERT");
    }
    require(db, sqlite3_finalize(insert), SQLITE_OK, "finalize INSERT");
  }
  execute(db, "COMMIT");
  double build_seconds = seconds() - start;
  sqlite3_progress_handler(db, 0, NULL, NULL);
  if (!hnsw) {
    sqlite3_stmt *counter = NULL;
    require(db, sqlite3_prepare_v2(db, "SELECT live_count FROM v_diskann_meta", -1, &counter, NULL),
            SQLITE_OK, "prepare count");
    require(db, sqlite3_step(counter), SQLITE_ROW, "read count");
    if (sqlite3_column_int64(counter, 0) != count) return 1;
    require(db, sqlite3_finalize(counter), SQLITE_OK, "finalize count");
  }
  printf("{\"mode\":\"%s\",\"count\":%d,\"dimension\":%d,\"metric\":\"%s\","
         "\"sqlite\":\"%s\",\"build_seconds\":%.9f,\"storage\":\"%s\",\"database\":",
         argv[2], count, dimension, metric, sqlite3_libversion(), build_seconds,
         strcmp(database, ":memory:") ? "file" : "memory");
  json_string(database);
  printf(",\"snapshot\":");
  if (snapshot) json_string(snapshot); else printf("null");
  printf(",\"tracing\":false,\"max_visits\":%lu,\"timeout_seconds\":%.9f,"
         "\"query_windows\":[", max_visits, timeout);
  int windows[] = {10, 50, 100};
  for (int w = 0; w < 3; ++w) {
    int window = windows[w];
    sqlite3_stmt *query = NULL;
    require(db, sqlite3_prepare_v2(db,
        "SELECT rowid,distance FROM v WHERE knn_search(embedding,knn_param(?1,10,?2))",
        -1, &query, NULL), SQLITE_OK, "prepare query");
    char tuning[64];
    snprintf(tuning, sizeof(tuning), "{\"search_list_size\":%d}", window);
    if (w) printf(",");
    printf("{\"window\":%d,\"milliseconds\":[", window);
    int64_t *labels = (int64_t *)malloc((size_t)query_count * 10 * sizeof(int64_t));
    uint32_t *distance_bits = (uint32_t *)malloc((size_t)query_count * 10 * sizeof(uint32_t));
    if (!labels || !distance_bits) return 1;
    /* One untimed warm-up pass, then a single timed pass. */
    for (int pass = 0; pass < 2; ++pass) {
      for (int q = 0; q < query_count; ++q) {
        require(db, sqlite3_bind_blob(query, 1, queries + (size_t)q * dimension,
            dimension * (int)sizeof(float), SQLITE_STATIC), SQLITE_OK, "bind query");
        if (hnsw) require(db, sqlite3_bind_int(query, 2, window), SQLITE_OK, "bind ef");
        else require(db, sqlite3_bind_text(query, 2, tuning, -1, SQLITE_TRANSIENT), SQLITE_OK, "bind L");
        double query_start = seconds();
        int returned = 0, rc;
        while ((rc = sqlite3_step(query)) == SQLITE_ROW) {
          if (returned >= 10) return 1;
          if (pass) {
            size_t ordinal = (size_t)q * 10 + returned;
            labels[ordinal] = sqlite3_column_int64(query, 0);
            float distance = (float)sqlite3_column_double(query, 1);
            if (!isfinite(distance)) return 1;
            memcpy(&distance_bits[ordinal], &distance, sizeof(distance));
          }
          ++returned;
        }
        double milliseconds = (seconds() - query_start) * 1000.0;
        require(db, rc, SQLITE_DONE, "query");
        if (returned != 10) return 1;
        require(db, sqlite3_reset(query), SQLITE_OK, "reset query");
        if (pass) printf("%s%.9f", q ? "," : "", milliseconds);
      }
    }
    printf("],\"labels\":[");
    for (int q = 0; q < query_count; ++q) {
      printf("%s[", q ? "," : "");
      for (int k = 0; k < 10; ++k) printf("%s%lld", k ? "," : "", (long long)labels[(size_t)q * 10 + k]);
      printf("]");
    }
    printf("],\"distance_bits\":[");
    for (int q = 0; q < query_count; ++q) {
      printf("%s[", q ? "," : "");
      for (int k = 0; k < 10; ++k) printf("%s%u", k ? "," : "", (unsigned)distance_bits[(size_t)q * 10 + k]);
      printf("]");
    }
    printf("]}");
    free(distance_bits);
    free(labels);
    require(db, sqlite3_finalize(query), SQLITE_OK, "finalize query");
  }
  if (snapshot && !save_snapshot(db, snapshot)) {
    sqlite3_close(db);
    free(rowids); free(vectors); free(queries);
    return 1;
  }
  printf("]}\n");
  require(db, sqlite3_close(db), SQLITE_OK, "close");
  free(rowids); free(vectors); free(queries);
  return 0;
}
