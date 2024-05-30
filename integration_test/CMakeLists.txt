add_executable(delete_test delete_test.cpp)
target_link_libraries(delete_test PRIVATE unofficial::sqlite3::sqlite3)