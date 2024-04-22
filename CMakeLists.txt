cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(vectorlite VERSION 0.1.0 LANGUAGES CXX)

configure_file(src/version.h.in version.h)
message(STATUS "version.h generated to: ${PROJECT_BINARY_DIR}")

SET(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(absl CONFIG REQUIRED)
# find_package(RapidJSON CONFIG REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
# find_package(hnswlib CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

find_package(re2 CONFIG REQUIRED)

find_path(RAPIDJSON_INCLUDE_DIRS rapidjson/rapidjson.h)
message(STATUS "RapidJSON include dir: ${RAPIDJSON_INCLUDE_DIRS}")
find_path(HNSWLIB_INCLUDE_DIRS hnswlib/hnswlib.h)
message(STATUS "HNSWLIB include dir: ${HNSWLIB_INCLUDE_DIRS}")

add_library(vectorlite SHARED src/vectorlite.cpp src/virtual_table.cpp src/vector.cpp src/util.cpp src/vector_space.cpp src/index_options.cpp src/sqlite_functions.cpp)
target_include_directories(vectorlite PUBLIC ${RAPIDJSON_INCLUDE_DIRS} ${HNSWLIB_INCLUDE_DIRS} ${PROJECT_BINARY_DIR})
target_link_libraries(vectorlite PRIVATE unofficial::sqlite3::sqlite3 absl::status absl::statusor absl::strings absl::log re2::re2)

include(GoogleTest)
enable_testing()
file(GLOB TEST_SOURCES src/*.cpp)
add_executable(unit-test ${TEST_SOURCES})
target_include_directories(unit-test PUBLIC ${PROJECT_BINARY_DIR})
target_link_libraries(unit-test PRIVATE GTest::gtest GTest::gtest_main unofficial::sqlite3::sqlite3 absl::status absl::statusor absl::strings absl::log re2::re2)
target_compile_options(unit-test PRIVATE -Wall -fno-omit-frame-pointer -g -O0)
# target_link_options(unit-test PRIVATE -fsanitize=address)

gtest_discover_tests(unit-test)

add_test(NAME unit-test COMMAND unit-test)