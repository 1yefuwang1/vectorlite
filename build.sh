cmake --preset dev && cmake --build build/dev -j8 && ctest --test-dir build/dev --output-on-failure && pytest bindings/python/vectorlite_py/test