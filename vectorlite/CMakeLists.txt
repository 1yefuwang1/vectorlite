# The SQLite extension is implemented in Rust. CMake supplies the native
# dependencies, stages Cargo's cdylib under the public filename, and installs it
# for scikit-build-core. The only C++ implementation retained here is SIMD ops.
if(NOT VCPKG_INSTALLED_DIR OR NOT VCPKG_TARGET_TRIPLET)
    message(FATAL_ERROR "Configure with the vcpkg toolchain to build Vectorlite")
endif()

if(VCPKG_TARGET_TRIPLET MATCHES "^x64-")
    set(_rust_arch x86_64)
elseif(VCPKG_TARGET_TRIPLET MATCHES "^arm64-")
    set(_rust_arch aarch64)
elseif(VCPKG_TARGET_TRIPLET MATCHES "^x86-")
    set(_rust_arch i686)
elseif(VCPKG_TARGET_TRIPLET MATCHES "^arm-")
    set(_rust_arch armv7)
else()
    message(FATAL_ERROR "Unsupported vcpkg architecture: ${VCPKG_TARGET_TRIPLET}")
endif()

if(APPLE)
    list(LENGTH CMAKE_OSX_ARCHITECTURES _osx_arch_count)
    if(_osx_arch_count GREATER 1)
        message(FATAL_ERROR "Build separate arm64/x86_64 artifacts; universal2 is not supported")
    endif()
    set(_rust_target "${_rust_arch}-apple-darwin")
    set(_cargo_library libvectorlite.dylib)
elseif(MSVC)
    set(_rust_target "${_rust_arch}-pc-windows-msvc")
    set(_cargo_library vectorlite.dll)
elseif(MINGW)
    set(_rust_target "${_rust_arch}-pc-windows-gnu")
    set(_cargo_library vectorlite.dll)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(VCPKG_TARGET_TRIPLET MATCHES "-linux-musl")
        set(_rust_target "${_rust_arch}-unknown-linux-musl")
    else()
        set(_rust_target "${_rust_arch}-unknown-linux-gnu")
    endif()
    set(_cargo_library libvectorlite.so)
else()
    message(FATAL_ERROR "Unsupported Vectorlite platform: ${CMAKE_SYSTEM_NAME}")
endif()

set(VECTORLITE_RUST_TARGET "${_rust_target}" CACHE STRING "Cargo target triple")
set(VECTORLITE_CARGO_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo" CACHE PATH
    "Cargo output directory for this CMake build")
set(_cargo_profile "$<IF:$<CONFIG:Debug>,debug,release>")
set(_cargo_artifact
    "${VECTORLITE_CARGO_TARGET_DIR}/${VECTORLITE_RUST_TARGET}/${_cargo_profile}/${_cargo_library}")
set(VECTORLITE_EXTENSION
    "${CMAKE_CURRENT_BINARY_DIR}/vectorlite${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(_cargo_manifest "${PROJECT_SOURCE_DIR}/Cargo.toml")

# Isolated wheel build directories are not part of Cargo's normal discovery.
# Always pass the exact installation chosen by this CMake toolchain.
set(_cargo_environment
    "CARGO_TARGET_DIR=${VECTORLITE_CARGO_TARGET_DIR}"
    "VECTORLITE_VCPKG_TRIPLET_DIR=${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}"
    "CC=${CMAKE_C_COMPILER}"
    "CXX=${CMAKE_CXX_COMPILER}")
if(NOT MSVC)
    list(APPEND _cargo_environment "AR=${CMAKE_AR}")
endif()
if(CMAKE_OSX_DEPLOYMENT_TARGET)
    list(APPEND _cargo_environment
         "MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()

if(SKBUILD)
    set(_deploy_default OFF)
else()
    set(_deploy_default ON)
endif()
option(VECTORLITE_DEPLOY_TO_PYTHON "Deploy the built extension into the source Python package"
       ${_deploy_default})
set(_deploy_commands)
if(VECTORLITE_DEPLOY_TO_PYTHON)
    set(_deploy_commands
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${VECTORLITE_EXTENSION}"
                "${PROJECT_SOURCE_DIR}/bindings/python/vectorlite_py/vectorlite${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

# Cargo performs its own incremental source/header/dependency invalidation, so
# invoke it on every requested build rather than duplicating its dependency graph.
add_custom_target(vectorlite ALL
    COMMAND "${CMAKE_COMMAND}" -E env ${_cargo_environment}
            "${CARGO_EXECUTABLE}" build --locked --manifest-path "${_cargo_manifest}"
            --target "${VECTORLITE_RUST_TARGET}" $<$<NOT:$<CONFIG:Debug>>:--release>
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${_cargo_artifact}" "${VECTORLITE_EXTENSION}"
    ${_deploy_commands}
    BYPRODUCTS "${VECTORLITE_EXTENSION}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMAND_EXPAND_LISTS
    USES_TERMINAL
    VERBATIM)

install(FILES "${VECTORLITE_EXTENSION}" DESTINATION vectorlite_py)

if(BUILD_TESTING)
    add_test(NAME rust_unit_tests
        COMMAND "${CMAKE_COMMAND}" -E env ${_cargo_environment}
                "${CARGO_EXECUTABLE}" test --locked --manifest-path "${_cargo_manifest}"
                --workspace --target "${VECTORLITE_RUST_TARGET}")
    add_test(NAME sqlite_abi
        COMMAND "${CMAKE_COMMAND}" -E env ${_cargo_environment}
                "${CARGO_EXECUTABLE}" test --locked --manifest-path "${_cargo_manifest}"
                -p vectorlite-sqlite-sys --features abi-check
                --target "${VECTORLITE_RUST_TARGET}")
endif()

if(BUILD_TESTING OR VECTORLITE_BUILD_BENCHMARKS)
    add_subdirectory(ops)
endif()
