cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(sqlite-vector VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(RapidJSON CONFIG REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_package(hnswlib CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

add_library(sqlite-vector-impl SHARED src/vector.cpp)
target_include_directories(sqlite-vector-impl PUBLIC ${RAPIDJSON_INCLUDE_DIRS})

add_library(sqlite-vector SHARED src/sqlite_vector.cpp)
target_link_libraries(sqlite-vector PRIVATE unofficial::sqlite3::sqlite3 sqlite-vector-impl)

include(GoogleTest)
enable_testing()
file(GLOB TEST_SOURCES src/*_test.cpp)
add_executable(unit-test ${TEST_SOURCES})
target_link_libraries(unit-test PRIVATE GTest::gtest GTest::gtest_main sqlite-vector-impl)

gtest_discover_tests(unit-test)

add_test(NAME unit-test COMMAND unit-test)