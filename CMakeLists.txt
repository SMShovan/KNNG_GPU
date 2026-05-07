cmake_minimum_required(VERSION 3.20)

# --------------------------------------------------------------------------- #
# Project definition
# --------------------------------------------------------------------------- #
project(knng
    VERSION       0.1.0
    DESCRIPTION   "GPU-accelerated approximate K-nearest neighbor graph construction"
    HOMEPAGE_URL  "https://github.com/SMShovan/KNNG_GPU"
    LANGUAGES     CXX
)

# --------------------------------------------------------------------------- #
# Global settings
# --------------------------------------------------------------------------- #
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD           20)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
set(CMAKE_CXX_EXTENSIONS         OFF)

# Default to Release when the generator does not supply a build type.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

# Export compile_commands.json for editors/clang-tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Emit shared libraries + executables into a single build/bin so that
# running binaries directly from the build tree is friction-free.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
include(CompilerWarnings)
include(FindKnngBlas)
include(FindKnngOpenMP)
include(FindKnngMPI)

# Generate include/knng/version.hpp from the template.
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/knng/version.hpp"
    @ONLY
)

# Public header interface target. Other targets link this to pick up
# include paths (public headers + generated headers) without caring where
# they live on disk.
add_library(knng_headers INTERFACE)
target_include_directories(knng_headers INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
)
target_compile_features(knng_headers INTERFACE cxx_std_20)
add_library(knng::headers ALIAS knng_headers)

# --------------------------------------------------------------------------- #
# Library + tool subdirectories
# --------------------------------------------------------------------------- #
# `src/` owns the library targets (currently just `knng::core`, a
# header-only INTERFACE library). `tools/` owns CLI executables that
# link those library targets. Both must be added before `tests/` so
# that test targets can link the aliases they define. `docs/` is
# pulled in unconditionally — its own CMakeLists.txt gates the
# `docs` target on whether Doxygen is installed.
add_subdirectory(src)
if(KNNG_HAVE_MPI)
    add_subdirectory(src/dist)
endif()
add_subdirectory(tools)
add_subdirectory(docs)

# --------------------------------------------------------------------------- #
# Testing
# --------------------------------------------------------------------------- #
# `include(CTest)` defines the `BUILD_TESTING` cache option (default ON) and
# calls `enable_testing()`. The actual GoogleTest download + test targets
# are only wired up when we are the top-level project AND tests are
# requested — downstream consumers pulling knng via FetchContent get the
# library without incurring the GoogleTest clone.
option(KNNG_BUILD_TESTS "Build knng's unit tests" ${PROJECT_IS_TOP_LEVEL})

if(KNNG_BUILD_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        include(FetchGoogleTest)
        add_subdirectory(tests)
    endif()
endif()

# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #
# Opt-in (default OFF) so the standard developer loop and the CI matrix
# don't pay the Google Benchmark FetchContent + build cost unless they
# actually want benchmark numbers. Enable with `-DKNNG_BUILD_BENCHMARKS=ON`.
option(KNNG_BUILD_BENCHMARKS "Build knng's micro and end-to-end benchmarks" OFF)

if(KNNG_BUILD_BENCHMARKS)
    include(FetchGoogleBenchmark)
    add_subdirectory(benchmarks)
endif()

# --------------------------------------------------------------------------- #
# Build summary (helpful when `ccmake` or CI logs are the only feedback).
# --------------------------------------------------------------------------- #
message(STATUS "")
message(STATUS "----- knng build configuration -----")
message(STATUS "  version       : ${PROJECT_VERSION}")
message(STATUS "  build type    : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard  : ${CMAKE_CXX_STANDARD}")
message(STATUS "  compiler      : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  binary dir    : ${CMAKE_BINARY_DIR}")
message(STATUS "  blas (Step 21): ${KNNG_HAVE_BLAS}")
message(STATUS "  openmp (S24+) : ${KNNG_HAVE_OPENMP}")
message(STATUS "  mpi   (S38+)  : ${KNNG_HAVE_MPI}")
message(STATUS "------------------------------------")
message(STATUS "")
