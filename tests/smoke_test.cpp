/// @file
/// @brief Smoke tests for the `knng` build system and public headers.
///
/// This translation unit intentionally contains no algorithmic logic. Its
/// job is to verify, end-to-end, that:
///   1. GoogleTest is wired up via `FetchContent` and `gtest_discover_tests`.
///   2. `knng::headers` is reachable from a test target.
///   3. The CMake-configured `knng/version.hpp` exposes macros that agree
///      with the `project(knng VERSION ...)` declaration in the root
///      `CMakeLists.txt`.
///
/// If any of these regress, `ctest` on a clean tree will fail before any
/// real algorithmic test has a chance to run — which is the intent.

#include <cstdio>
#include <string>

#include <gtest/gtest.h>

#include "knng/version.hpp"

namespace {

TEST(Version, MacrosAreNonNegative)
{
    EXPECT_GE(KNNG_VERSION_MAJOR, 0);
    EXPECT_GE(KNNG_VERSION_MINOR, 0);
    EXPECT_GE(KNNG_VERSION_PATCH, 0);
}

TEST(Version, StringMatchesComponents)
{
    const std::string expected =
        std::to_string(KNNG_VERSION_MAJOR) + "." +
        std::to_string(KNNG_VERSION_MINOR) + "." +
        std::to_string(KNNG_VERSION_PATCH);

    EXPECT_EQ(std::string{KNNG_VERSION_STRING}, expected);
}

TEST(Version, StringIsNonEmpty)
{
    EXPECT_GT(std::string{KNNG_VERSION_STRING}.size(), 0u);
}

} // namespace
