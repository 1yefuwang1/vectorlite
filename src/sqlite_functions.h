#pragma once

#include "sqlite3ext.h"

namespace sqlite_vector {

// Shows a human-readable string about version, what SIMD instruction is used at
// build time.
void ShowInfo(sqlite3_context* ctx, int, sqlite3_value**);

// VectorDistance takes two vectors and and space type, then outputs their
// distance
void VectorDistance(sqlite3_context* ctx, int argc, sqlite3_value** argv);

void VectorFromJson(sqlite3_context* ctx, int argc, sqlite3_value** argv);

void VectorFromMsgPack(sqlite3_context* ctx, int argc, sqlite3_value** argv);

}  // namespace sqlite_vector