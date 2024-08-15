cmake --preset release && cmake --build build/release -j8 && ctest --test-dir build/release/vectorlite --output-on-failure && pytest bindings/python/vectorlite_py/test
