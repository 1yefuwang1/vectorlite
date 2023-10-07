cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(sqlite-hnsw VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(RapidJSON REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_library(hnswlib CONFIG REQUIRED)

add_library(sqlite-hnsw SHARED src/sqlite_hnsw.cpp)
target_link_libraries(sqlite-hnsw PRIVATE unofficial::sqlite3::sqlite3)