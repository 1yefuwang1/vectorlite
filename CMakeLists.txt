cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

if (WIN32)
    set(VCPKG_TARGET_TRIPLET "x64-windows-static-md-release")
    message(STATUS "VCPKG_TARGET_TRIPLET on windows: ${VCPKG_TARGET_TRIPLET}")
endif(WIN32)

project(vectorlite VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(absl CONFIG REQUIRED)
find_package(RapidJSON CONFIG REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_package(hnswlib CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)

find_package(re2 CONFIG REQUIRED)

find_path(RAPIDJSON_INCLUDE_DIRS rapidjson/rapidjson.h)
message(STATUS "RapidJSON include dir: ${RAPIDJSON_INCLUDE_DIRS}")
find_path(HNSWLIB_INCLUDE_DIRS hnswlib/hnswlib.h)
message(STATUS "HNSWLIB include dir: ${HNSWLIB_INCLUDE_DIRS}")

add_subdirectory(vectorlite)
