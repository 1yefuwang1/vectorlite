#pragma once

#if defined (_WIN32) || defined (__WIN32__)
  #define SQLITE_VECTOR_EXPORT __declspec(dllexport)
#else
  #define SQLITE_VECTOR_EXPORT __attribute__((visibility("default")))
#endif