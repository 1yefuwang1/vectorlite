cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(sqlite-vector VERSION 0.1.0 LANGUAGES CXX)

SET(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(absl CONFIG REQUIRED)
find_package(RapidJSON CONFIG REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_package(hnswlib CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

add_library(sqlite-vector SHARED src/sqlite_vector.cpp src/virtual_table.cpp src/vector.cpp src/util.cpp)
target_include_directories(sqlite-vector PUBLIC ${RAPIDJSON_INCLUDE_DIRS})
target_link_libraries(sqlite-vector PRIVATE unofficial::sqlite3::sqlite3 absl::status absl::statusor absl::strings)

include(GoogleTest)
enable_testing()
file(GLOB TEST_SOURCES src/*.cpp)
add_executable(unit-test ${TEST_SOURCES})
target_link_libraries(unit-test PRIVATE GTest::gtest GTest::gtest_main unofficial::sqlite3::sqlite3 absl::status absl::statusor absl::strings)
target_compile_options(unit-test PRIVATE -Wall -fsanitize=address -fno-omit-frame-pointer -g)
target_link_options(unit-test PRIVATE -fsanitize=address)

gtest_discover_tests(unit-test)

add_test(NAME unit-test COMMAND unit-test)