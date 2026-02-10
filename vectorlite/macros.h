#pragma once

#include <type_traits>

#include "hwy/base.h"

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

#define VECTORLITE_IF_FLOAT_SUPPORTED(T)       \
  std::enable_if_t<std::is_same_v<T, float> || \
                   std::is_same_v<T, hwy::bfloat16_t> || \
                   std::is_same_v<T, hwy::float16_t>>* = nullptr

#define VECTORLITE_IF_FLOAT_SUPPORTED_FWD_DECL(T) \
  std::enable_if_t<std::is_same_v<T, float> ||    \
                   std::is_same_v<T, hwy::bfloat16_t> || \
                   std::is_same_v<T, hwy::float16_t>>*
