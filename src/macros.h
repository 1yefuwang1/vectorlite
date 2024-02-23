#pragma once

#if defined(_WIN32) || defined(__WIN32__)
#define SQLITE_VECTOR_EXPORT __declspec(dllexport)
#else
#define SQLITE_VECTOR_EXPORT __attribute__((visibility("default")))
#endif

// Use the c style assert by default but can be overridden by the user
#ifndef SQLITE_VECTOR_ASSERT
#include <cassert>
#define SQLITE_VECTOR_ASSERT(x) assert(x)
#endif
