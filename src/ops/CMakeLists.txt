find_package(hwy CONFIG REQUIRED)

find_path(HIGHWAY_INCLUDE_DIRS hwy/highway.h)
message(STATUS "Highway include dir: ${HIGHWAY_INCLUDE_DIRS}")

add_library(distance STATIC distance.cpp)
target_link_libraries(distance PRIVATE hwy::hwy)
target_include_directories(distance PRIVATE ${HIGHWAY_INCLUDE_DIRS} ${CMAKE_CURRENT_LIST_DIR})
# target_compile_definitions(distance PRIVATE HWY_IDE=1)

include(GoogleTest)
enable_testing()

add_executable(distance_test distance_test.cpp)
target_include_directories(distance PRIVATE ${HIGHWAY_INCLUDE_DIRS} ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(distance_test PRIVATE distance GTest::gtest GTest::gtest_main)


gtest_discover_tests(distance_test)

add_test(NAME distance_test COMMAND distance_test)

add_executable(distance_benchmark distance_benchmark.cpp)
target_include_directories(distance_benchmark PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(distance_benchmark PRIVATE distance benchmark::benchmark benchmark::benchmark_main)

if (MSVC)
    target_compile_options(distance_test PRIVATE /arch:AVX)
    target_compile_options(distance_benchmark PRIVATE /arch:AVX)
else()
    target_compile_options(distance_test PRIVATE -march=native)
    target_compile_options(distance_benchmark PRIVATE -march=native)
endif()
