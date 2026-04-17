/// @file
/// @brief Smoke-test executable for the `knng` build system.
///
/// `hello_knng` exists for one purpose: to confirm, end-to-end, that the
/// build system is producing a runnable binary with the expected compile-
/// time configuration. It links `knng::headers`, reads the generated
/// version header, and prints a short report. It deliberately has no
/// algorithmic content and no runtime dependencies beyond the standard
/// library.
///
/// Intended uses:
///   * After changing anything in `CMakeLists.txt` or `cmake/`, rebuild
///     and run `./build/bin/hello_knng` to verify the configuration.
///   * As a minimal reproducer when reporting build-system issues.

#include <cstdio>
#include <string_view>

#include "knng/version.hpp"

namespace {

/// Return a human-readable label for the host compiler.
///
/// The macros checked here cover every compiler the project currently
/// supports. Unknown compilers fall back to "unknown".
///
/// @return A `std::string_view` with static storage duration.
constexpr std::string_view compiler_id() noexcept
{
#if defined(__clang__)
#  if defined(__apple_build_version__)
    return "AppleClang";
#  else
    return "Clang";
#  endif
#elif defined(__GNUC__)
    return "GCC";
#elif defined(_MSC_VER)
    return "MSVC";
#else
    return "unknown";
#endif
}

/// Return the compile-time build type as a string.
///
/// CMake defines `NDEBUG` for every non-Debug configuration, so we
/// distinguish at source level only between Debug and "optimized"; the
/// precise configuration name (Release vs. RelWithDebInfo vs. MinSizeRel)
/// is injected via the `KNNG_BUILD_TYPE_STRING` macro, which future
/// revisions of `CMakeLists.txt` can populate.
constexpr std::string_view build_type() noexcept
{
#if defined(NDEBUG)
    return "optimized";
#else
    return "debug";
#endif
}

/// Return a label for the host operating system.
constexpr std::string_view platform_id() noexcept
{
#if defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#elif defined(_WIN32)
    return "Windows";
#else
    return "unknown";
#endif
}

} // namespace

/// Program entry point — print the version banner and exit.
int main()
{
    std::printf("knng %s\n", KNNG_VERSION_STRING);
    std::printf("  compiler : %.*s\n",
                static_cast<int>(compiler_id().size()),
                compiler_id().data());
    std::printf("  build    : %.*s\n",
                static_cast<int>(build_type().size()),
                build_type().data());
    std::printf("  platform : %.*s\n",
                static_cast<int>(platform_id().size()),
                platform_id().data());
    std::printf("  C++      : %ldL\n", static_cast<long>(__cplusplus));
    return 0;
}
