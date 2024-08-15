configure_file(version.h.in version.h)
message(STATUS "version.h generated to: ${PROJECT_BINARY_DIR}")

find_path(RAPIDJSON_INCLUDE_DIRS rapidjson/rapidjson.h)
message(STATUS "RapidJSON include dir: ${RAPIDJSON_INCLUDE_DIRS}")
find_path(HNSWLIB_INCLUDE_DIRS hnswlib/hnswlib.h)
message(STATUS "HNSWLIB include dir: ${HNSWLIB_INCLUDE_DIRS}")

message(STATUS "Compiling on ${CMAKE_SYSTEM_PROCESSOR}")

add_subdirectory(ops)

add_library(vectorlite SHARED vectorlite.cpp virtual_table.cpp vector.cpp vector_view.cpp util.cpp vector_space.cpp index_options.cpp sqlite_functions.cpp constraint.cpp)
# remove the lib prefix to make the shared library name consistent on all platforms.
set_target_properties(vectorlite PROPERTIES PREFIX "")
target_include_directories(vectorlite PUBLIC ${RAPIDJSON_INCLUDE_DIRS} ${HNSWLIB_INCLUDE_DIRS} ${PROJECT_BINARY_DIR})
target_link_libraries(vectorlite PRIVATE unofficial::sqlite3::sqlite3 absl::status absl::statusor absl::strings re2::re2 ops)
# copy the shared library to the python package to make running integration tests easier
add_custom_command(TARGET vectorlite POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:vectorlite> ${PROJECT_SOURCE_DIR}/bindings/python/vectorlite_py/$<TARGET_FILE_NAME:vectorlite>)

include(GoogleTest)
enable_testing()
file(GLOB TEST_SOURCES *.cpp)
add_executable(unit_test ${TEST_SOURCES})
target_include_directories(unit_test PUBLIC ${PROJECT_BINARY_DIR})
target_link_libraries(unit_test PRIVATE GTest::gtest GTest::gtest_main unofficial::sqlite3::sqlite3 absl::status absl::statusor absl::strings re2::re2 ops)
# target_compile_options(unit_test PRIVATE -Wall -fno-omit-frame-pointer -g -O0)
# target_link_options(unit_test PRIVATE -fsanitize=address)
if (MSVC)
    target_compile_options(vectorlite PRIVATE /arch:AVX)
    target_compile_options(unit_test PRIVATE /arch:AVX)
endif()

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
