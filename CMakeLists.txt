cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(vectorlite VERSION 0.1.0 LANGUAGES CXX)

configure_file(src/version.h.in version.h)
message(STATUS "version.h generated to: ${PROJECT_BINARY_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OPTION_USE_AVX OFF)


find_package(absl CONFIG REQUIRED)
find_package(RapidJSON CONFIG REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_package(hnswlib CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

find_package(re2 CONFIG REQUIRED)

find_path(RAPIDJSON_INCLUDE_DIRS rapidjson/rapidjson.h)
message(STATUS "RapidJSON include dir: ${RAPIDJSON_INCLUDE_DIRS}")
find_path(HNSWLIB_INCLUDE_DIRS hnswlib/hnswlib.h)
message(STATUS "HNSWLIB include dir: ${HNSWLIB_INCLUDE_DIRS}")

message(STATUS "Compiling on ${CMAKE_SYSTEM_PROCESSOR}")

# compile with avx for x86_64 and x86. Though SSE would be a safer default.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(OPTION_USE_AVX ON)
endif ()

add_library(vectorlite SHARED src/vectorlite.cpp src/virtual_table.cpp src/vector.cpp src/util.cpp src/vector_space.cpp src/index_options.cpp src/sqlite_functions.cpp src/constraint.cpp)
target_include_directories(vectorlite PUBLIC ${RAPIDJSON_INCLUDE_DIRS} ${HNSWLIB_INCLUDE_DIRS} ${PROJECT_BINARY_DIR})
target_link_libraries(vectorlite PRIVATE unofficial::sqlite3::sqlite3 absl::status absl::statusor absl::strings re2::re2)
# copy the shared library to the python package to make running integration tests easier
add_custom_command(TARGET vectorlite POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:vectorlite> ${PROJECT_SOURCE_DIR}/vectorlite_py/$<TARGET_FILE_NAME:vectorlite>)


include(GoogleTest)
enable_testing()
file(GLOB TEST_SOURCES src/*.cpp)
add_executable(unit-test ${TEST_SOURCES})
target_include_directories(unit-test PUBLIC ${PROJECT_BINARY_DIR})
target_link_libraries(unit-test PRIVATE GTest::gtest GTest::gtest_main unofficial::sqlite3::sqlite3 absl::status absl::statusor absl::strings re2::re2)
# target_compile_options(unit-test PRIVATE -Wall -fno-omit-frame-pointer -g -O0)
# target_link_options(unit-test PRIVATE -fsanitize=address)
if(OPTION_USE_AVX)
    message(STATUS "AVX enabled")
    if (MSVC)
        target_compile_options(vectorlite PRIVATE /arch:AVX)
        target_compile_options(unit-test PRIVATE /arch:AVX)
    else()
        target_compile_options(vectorlite PRIVATE -mavx)
        target_compile_options(unit-test PRIVATE -mavx)
    endif()
endif(OPTION_USE_AVX)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_libraries(vectorlite PRIVATE absl::log)
    target_link_libraries(unit-test PRIVATE absl::log)
endif()

gtest_discover_tests(unit-test)

add_test(NAME unit-test COMMAND unit-test)

add_subdirectory(integration_test)