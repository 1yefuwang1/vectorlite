#pragma once

#if defined(_WIN32) || defined(__WIN32__)
#define VECTORLITE_EXPORT __declspec(dllexport)
#else
#define VECTORLITE_EXPORT __attribute__((visibility("default")))
#endif

// Use the c style assert by default but can be overridden by the user
#ifndef VECTORLITE_ASSERT
#include <cassert>
#define VECTORLITE_ASSERT(x) assert(x)
#endif
