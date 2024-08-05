#include "sqlite_functions.h"

#include <sqlite3ext.h>

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "util.h"
#include "vector.h"
#include "vector_space.h"
#include "version.h"

// Defined in vectorlite.cpp
extern const sqlite3_api_routines *sqlite3_api;

namespace vectorlite {

void ShowInfo(sqlite3_context *ctx, int, sqlite3_value **) {
  auto simd = vectorlite::DetectSIMD().value_or("SIMD not enabled");
  std::string info =
      absl::StrFormat("vectorlite extension version %s, built with %s",
                      VECTORLITE_VERSION, simd);
  DLOG(INFO) << "ShowInfo called: " << info;
  sqlite3_result_text(ctx, info.c_str(), -1, SQLITE_TRANSIENT);
}

// VectorDistance takes two vectors and and space type, then outputs their
// distance
void VectorDistance(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  if (argc != 3) {
    std::string err = absl::StrFormat(
        "vector_distance expects 3 arguments but %d provided", argc);
    sqlite3_result_error(ctx, err.c_str(), -1);
    return;
  }

  auto type1 = sqlite3_value_type(argv[0]);
  auto type2 = sqlite3_value_type(argv[1]);
  if (type1 != SQLITE_BLOB || type2 != SQLITE_BLOB) {
    DLOG_IF(INFO, type1 != SQLITE_BLOB && type1 == SQLITE_TEXT)
        << "vector1 is of type TEXT" << sqlite3_value_text(argv[0])
        << " with size " << sqlite3_value_bytes(argv[0]);
    DLOG_IF(INFO, type2 != SQLITE_BLOB && type2 == SQLITE_TEXT)
        << "vector2 is of type TEXT" << sqlite3_value_text(argv[1])
        << " with size " << sqlite3_value_bytes(argv[1]);
    std::string err = absl::StrFormat(
        "vector_distance expects vectors of type blob but found %u and %u",
        type1, type2);
    sqlite3_result_error(ctx, err.c_str(), -1);
    return;
  }

  if (sqlite3_value_type(argv[2]) != SQLITE_TEXT) {
    sqlite3_result_error(
        ctx, "vectors_distance expects space type of type text", -1);
    return;
  }

  std::string_view space_type_str(
      reinterpret_cast<const char *>(sqlite3_value_text(argv[2])),
      sqlite3_value_bytes(argv[2]));
  auto distance_type = vectorlite::ParseDistanceType(space_type_str);
  if (!distance_type.has_value()) {
    std::string err =
        absl::StrFormat("Failed to parse space type: %s", space_type_str);
    sqlite3_result_error(ctx, err.c_str(), -1);
    return;
  }

  std::string_view v1_str(
      reinterpret_cast<const char *>(sqlite3_value_text(argv[0])),
      sqlite3_value_bytes(argv[0]));

  auto v1 = vectorlite::Vector::FromBlob(v1_str);
  if (!v1.ok()) {
    std::string err = absl::StrFormat("Failed to parse 1st vector due to: %s",
                                      v1.status().message());
    sqlite3_result_error(ctx, err.c_str(), -1);
    return;
  }

  std::string_view v2_str(
      reinterpret_cast<const char *>(sqlite3_value_text(argv[1])),
      sqlite3_value_bytes(argv[1]));
  auto v2 = vectorlite::Vector::FromBlob(v2_str);
  if (!v2.ok()) {
    std::string err = absl::StrFormat("Failed to parse 2nd vector due to: %s",
                                      v2.status().message());
    sqlite3_result_error(ctx, err.c_str(), -1);
    return;
  }

  auto distance = vectorlite::Distance(*v1, *v2, *distance_type);
  if (!distance.ok()) {
    sqlite3_result_error(ctx, absl::StatusMessageAsCStr(distance.status()), -1);
    return;
  }
  sqlite3_result_double(ctx, static_cast<double>(*distance));
  return;
}

void VectorFromJson(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  if (argc != 1) {
    std::string err = absl::StrFormat(
        "vector_from_json expects 1 argument but %d provided", argc);
    sqlite3_result_error(ctx, err.c_str(), -1);
    return;
  }

  if (sqlite3_value_type(argv[0]) != SQLITE_TEXT) {
    sqlite3_result_error(ctx, "vector_from_json expects a JSON string", -1);
    return;
  }

  std::string_view json_str(
      reinterpret_cast<const char *>(sqlite3_value_text(argv[0])),
      sqlite3_value_bytes(argv[0]));

  auto vector = vectorlite::Vector::FromJSON(json_str);
  if (!vector.ok()) {
    std::string err = absl::StrFormat("Failed to parse vector due to: %s",
                                      vector.status().message());
    sqlite3_result_error(ctx, err.c_str(), -1);
    return;
  }

  sqlite3_result_blob(ctx, vector->ToBlob().data(), vector->ToBlob().size(),
                      SQLITE_TRANSIENT);
  return;
}

void VectorToJson(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  if (argc != 1) {
    std::string err = absl::StrFormat(
        "vector_to_json expects 1 argument but %d provided", argc);
    sqlite3_result_error(ctx, err.c_str(), err.length());
    return;
  }

  if (sqlite3_value_type(argv[0]) != SQLITE_BLOB) {
    sqlite3_result_error(ctx, "vector_to_json expects vector of type blob", -1);
    return;
  }

  std::string_view vector_blob(
      reinterpret_cast<const char *>(sqlite3_value_blob(argv[0])),
      sqlite3_value_bytes(argv[0]));

  // todo: avoid copying by not constructing a new vector. Because we only need
  // read-only access here.
  auto vector = vectorlite::Vector::FromBlob(vector_blob);
  if (!vector.ok()) {
    std::string err = absl::StrFormat("Failed to parse vector due to: %s",
                                      vector.status().message());
    sqlite3_result_error(ctx, err.c_str(), err.length());
    return;
  }

  auto json = vector->ToJSON();
  sqlite3_result_text(ctx, json.data(), json.size(), SQLITE_TRANSIENT);
  return;
}

}  // namespace vectorlite