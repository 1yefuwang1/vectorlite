cmake --preset release && cmake --build build/release -j8 && ctest --test-dir build/release --output-on-failure && pytest vectorlite_py/test
