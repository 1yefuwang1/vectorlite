cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(sqlite-vector VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(RapidJSON CONFIG REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_package(hnswlib CONFIG REQUIRED)

add_library(sqlite-vector SHARED src/sqlite_vector.cpp src/vector.cpp)
target_link_libraries(sqlite-vector PRIVATE unofficial::sqlite3::sqlite3)