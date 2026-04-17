# FetchGoogleTest.cmake
#
# Fetches and configures GoogleTest / GoogleMock for in-tree unit tests.
# The filename is deliberately NOT `GoogleTest.cmake` — that would shadow
# CMake's built-in module of the same name (the one that defines the
# `gtest_discover_tests` function), leading to confusing missing-command
# errors when this file is loaded first via `CMAKE_MODULE_PATH`.
#
# Policy:
#   * Version is pinned to a specific release tag so that `FetchContent`
#     is deterministic across developers and CI. Bump explicitly, never
#     track a moving ref.
#   * GoogleTest is always built from source alongside the project, never
#     discovered from the system. A Homebrew / distro GoogleTest install
#     is typically a shared library whose rpath is not picked up by our
#     test binaries, breaking `gtest_discover_tests` at build time.
#     Source-fetching gives us a single, known-good static library and
#     sidesteps an entire class of environment-specific surprises.
#   * GoogleTest's own compile flags are left alone: we do NOT apply
#     `knng_set_warnings()` to its targets, because upstream GoogleTest
#     does not compile cleanly under `-Wconversion -Werror`. Our own test
#     binaries still go through `knng_set_warnings()` per usual.
#   * On Windows, `gtest_force_shared_crt` keeps GoogleTest's CRT setting
#     consistent with the default MSVC runtime; without this, a link-time
#     CRT mismatch is the first thing every new contributor hits.
#
# Usage:
#   include(FetchGoogleTest)   # once, from the root CMakeLists after project()
#   # ...then in tests/CMakeLists.txt:
#   add_executable(my_test my_test.cpp)
#   target_link_libraries(my_test PRIVATE knng::headers GTest::gtest_main)
#   gtest_discover_tests(my_test)

include_guard(GLOBAL)

include(FetchContent)

set(KNNG_GOOGLETEST_TAG "v1.15.2" CACHE STRING
    "Git tag of GoogleTest to fetch via FetchContent")
mark_as_advanced(KNNG_GOOGLETEST_TAG)

# Keep GoogleTest's CRT aligned with the consumer's CRT on MSVC. No effect
# on non-MSVC toolchains.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Do not install GoogleTest as part of `cmake --install` of this project.
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        ${KNNG_GOOGLETEST_TAG}
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(googletest)

# CMake's built-in `GoogleTest` module (distinct from this file — see the
# header comment) provides `gtest_discover_tests`, which registers each
# TEST() with CTest at build time. This is the modern replacement for the
# old `add_test(NAME ... COMMAND gtest)` incantations and works correctly
# with TEST_F / parameterized tests.
include(GoogleTest)
