cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" VECTORLITE_VERSION)

# Native ports are configured separately from this project. Pin their macOS
# floor in a repository-owned triplet, whose contents enter vcpkg's ABI hash.
# Caller overlays retain priority; artifact checks still reject higher floors.
# A previous configure may have cached our appended path through vcpkg. Move
# only that managed path to the end again before considering new caller paths.
list(REMOVE_ITEM VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/triplets")
if(DEFINED ENV{VCPKG_OVERLAY_TRIPLETS})
    cmake_path(CONVERT "$ENV{VCPKG_OVERLAY_TRIPLETS}" TO_CMAKE_PATH_LIST _vectorlite_env_triplets)
    list(APPEND VCPKG_OVERLAY_TRIPLETS ${_vectorlite_env_triplets})
endif()
list(APPEND VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/triplets")
list(REMOVE_DUPLICATES VCPKG_OVERLAY_TRIPLETS)

# Set the project/Cargo floor before compiler initialization, including when an
# older local CMake cache contains an empty deployment target. An explicit newer
# local target is permitted, but is not a macOS 11 release artifact.
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR
   (NOT CMAKE_SYSTEM_NAME AND CMAKE_HOST_APPLE))
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version" FORCE)
    endif()
    if(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS "11.0")
        message(FATAL_ERROR "Supported macOS arm64 builds require macOS 11.0 or newer")
    endif()
endif()

# Source distributions do not contain Git submodules. Fetch the exact pinned
# vcpkg checkout into the build directory when neither the repository toolchain
# nor an explicit external toolchain is available.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" AND
   NOT (CMAKE_TOOLCHAIN_FILE AND EXISTS "${CMAKE_TOOLCHAIN_FILE}"))
    find_package(Git REQUIRED)
    include(FetchContent)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg.json" _vectorlite_manifest)
    string(JSON _vcpkg_revision GET "${_vectorlite_manifest}" builtin-baseline)
    FetchContent_Declare(vectorlite_vcpkg
        GIT_REPOSITORY https://github.com/microsoft/vcpkg.git
        GIT_TAG "${_vcpkg_revision}"
        SOURCE_SUBDIR _vectorlite_no_cmake_project)
    FetchContent_MakeAvailable(vectorlite_vcpkg)
    set(CMAKE_TOOLCHAIN_FILE
        "${vectorlite_vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake"
        CACHE FILEPATH "Pinned vcpkg toolchain" FORCE)
    if(CMAKE_HOST_WIN32)
        set(_vcpkg_bootstrap "${vectorlite_vcpkg_SOURCE_DIR}/bootstrap-vcpkg.bat")
        set(_vcpkg_program "${vectorlite_vcpkg_SOURCE_DIR}/vcpkg.exe")
    else()
        set(_vcpkg_bootstrap "${vectorlite_vcpkg_SOURCE_DIR}/bootstrap-vcpkg.sh")
        set(_vcpkg_program "${vectorlite_vcpkg_SOURCE_DIR}/vcpkg")
    endif()
    if(NOT EXISTS "${_vcpkg_program}")
        execute_process(COMMAND "${_vcpkg_bootstrap}" -disableMetrics
                        WORKING_DIRECTORY "${vectorlite_vcpkg_SOURCE_DIR}"
                        COMMAND_ERROR_IS_FATAL ANY)
    endif()
endif()

# Wheel builds only need the extension. Repository builds also validate the
# retained SIMD kernels and the Rust/SQLite ABI boundary.
if(SKBUILD)
    set(_vectorlite_test_default OFF)
else()
    set(_vectorlite_test_default ON)
endif()
option(BUILD_TESTING "Build and run Vectorlite tests" ${_vectorlite_test_default})
option(VECTORLITE_BUILD_BENCHMARKS "Build native SIMD benchmarks" OFF)

# vcpkg resolves features while processing the toolchain in project(). Keep the
# shipped runtime independent of the old C++ glue and development-only libraries.
set(VCPKG_MANIFEST_NO_DEFAULT_FEATURES ON)
if(BUILD_TESTING)
    list(APPEND VCPKG_MANIFEST_FEATURES tests)
endif()
if(VECTORLITE_BUILD_BENCHMARKS)
    list(APPEND VCPKG_MANIFEST_FEATURES benchmarks)
endif()
list(REMOVE_DUPLICATES VCPKG_MANIFEST_FEATURES)

if((CMAKE_SYSTEM_NAME STREQUAL "Windows" OR
    (NOT CMAKE_SYSTEM_NAME AND CMAKE_HOST_WIN32)) AND
   NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET "x64-windows-static-md" CACHE STRING "vcpkg target triplet")
endif()

project(vectorlite VERSION ${VECTORLITE_VERSION} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()

find_program(CARGO_EXECUTABLE NAMES cargo REQUIRED)
find_package(hwy CONFIG REQUIRED)
find_package(hnswlib CONFIG REQUIRED)

include(CTest)
add_subdirectory(vectorlite)

# Install Python source explicitly so an ignored/stale source-tree .so/.dll/
# .dylib can never override the Cargo-built extension when packaging a wheel.
install(DIRECTORY "${PROJECT_SOURCE_DIR}/bindings/python/vectorlite_py/"
        DESTINATION vectorlite_py
        FILES_MATCHING PATTERN "*.py"
        PATTERN "__pycache__" EXCLUDE)
