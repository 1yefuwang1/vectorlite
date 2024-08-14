cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

if (WIN32)
    set(VCPKG_TARGET_TRIPLET "x64-windows-static-md-release")
    message(STATUS "VCPKG_TARGET_TRIPLET on windows: ${VCPKG_TARGET_TRIPLET}")
endif(WIN32)

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
find_package(benchmark CONFIG REQUIRED)

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

add_subdirectory(src/ops)

add_library(vectorlite SHARED src/vectorlite.cpp src/virtual_table.cpp src/vector.cpp src/vector_view.cpp src/util.cpp src/vector_space.cpp src/index_options.cpp src/sqlite_functions.cpp src/constraint.cpp)
# remove the lib prefix to make the shared library name consistent on all platforms.
set_target_properties(vectorlite PROPERTIES PREFIX "")
target_include_directories(vectorlite PUBLIC ${RAPIDJSON_INCLUDE_DIRS} ${HNSWLIB_INCLUDE_DIRS} ${PROJECT_BINARY_DIR})
target_link_libraries(vectorlite PRIVATE unofficial::sqlite3::sqlite3 absl::status absl::statusor absl::strings re2::re2)
# copy the shared library to the python package to make running integration tests easier
add_custom_command(TARGET vectorlite POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:vectorlite> ${PROJECT_SOURCE_DIR}/bindings/python/vectorlite_py/$<TARGET_FILE_NAME:vectorlite>)

include(GoogleTest)
enable_testing()
file(GLOB TEST_SOURCES src/*.cpp)
add_executable(unit_test ${TEST_SOURCES})
target_include_directories(unit_test PUBLIC ${PROJECT_BINARY_DIR})
target_link_libraries(unit_test PRIVATE GTest::gtest GTest::gtest_main unofficial::sqlite3::sqlite3 absl::status absl::statusor absl::strings re2::re2)
# target_compile_options(unit_test PRIVATE -Wall -fno-omit-frame-pointer -g -O0)
# target_link_options(unit_test PRIVATE -fsanitize=address)
if(OPTION_USE_AVX)
    message(STATUS "AVX enabled")
    if (MSVC)
        target_compile_options(vectorlite PRIVATE /arch:AVX)
        target_compile_options(unit_test PRIVATE /arch:AVX)
    else()
        target_compile_options(vectorlite PRIVATE -mavx)
        target_compile_options(unit_test PRIVATE -mavx)
    endif()
endif(OPTION_USE_AVX)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_libraries(vectorlite PRIVATE absl::log)
    target_link_libraries(unit_test PRIVATE absl::log)
endif()

# TODO: For mysterious reason, absl::log symbols are required for even release build on MSVC. Only DLOG are used which should be guarded by NDEBUG and not included in Release build.
if(MSVC)
    target_link_libraries(vectorlite PRIVATE absl::log)
    target_link_libraries(unit_test PRIVATE absl::log)
endif()

gtest_discover_tests(unit_test)

add_test(NAME unit_test COMMAND unit_test)
