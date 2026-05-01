# FetchGoogleBenchmark.cmake
#
# Fetches and configures Google Benchmark for in-tree micro/end-to-end
# benchmarking. Mirrors the structure and policy of FetchGoogleTest.cmake
# (see that file for the rationale on source-fetching vs system discovery).
#
# Policy:
#   * Version is pinned to a specific release tag so FetchContent is
#     deterministic across developers and CI. Bump explicitly.
#   * Google Benchmark is always built from source. A system-installed
#     `benchmark` package would work in principle but introduces the
#     same shared-library / rpath surprises that motivated the
#     source-only choice for GoogleTest.
#   * Google Benchmark's own self-tests and install rules are disabled —
#     we only need the library targets.
#   * Google Benchmark's compile flags are left alone; we do NOT apply
#     `knng_set_warnings()` to its targets, because upstream code does
#     not compile cleanly under `-Wconversion -Wold-style-cast -Werror`.
#     Our own benchmark binaries still go through `knng_set_warnings()`
#     per the project policy.
#
# Usage:
#   include(FetchGoogleBenchmark)   # once, from the root CMakeLists
#   add_executable(my_bench my_bench.cpp)
#   target_link_libraries(my_bench PRIVATE knng::cpu benchmark::benchmark_main)

include_guard(GLOBAL)

include(FetchContent)

set(KNNG_GOOGLEBENCHMARK_TAG "v1.9.1" CACHE STRING
    "Git tag of Google Benchmark to fetch via FetchContent")
mark_as_advanced(KNNG_GOOGLEBENCHMARK_TAG)

# Disable Google Benchmark's own test suite — we only consume the library.
set(BENCHMARK_ENABLE_TESTING  OFF CACHE BOOL "" FORCE)
# Don't install Google Benchmark when this project's `cmake --install` runs.
set(BENCHMARK_ENABLE_INSTALL  OFF CACHE BOOL "" FORCE)
# Avoid Google Benchmark trying to download its own GoogleTest copy.
set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)
# Ensure the compile-flag policy upstream chooses doesn't fight ours
# (no `-Werror` injected from their build).
set(BENCHMARK_ENABLE_WERROR   OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        ${KNNG_GOOGLEBENCHMARK_TAG}
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(googlebenchmark)
