/// @file
/// @brief Implementation of the bench-side runtime counters.
///
/// Only `peak_memory_mb` actually lives in this TU; the count of
/// brute-force distance evaluations is an `inline` helper in the
/// header. The split mirrors how Google Benchmark expects counter
/// providers to look — a single non-allocating function the
/// bench TU calls inside its measurement loop.

#include "knng/bench/runtime_counters.hpp"

#include <sys/resource.h>

namespace knng::bench {

double peak_memory_mb() noexcept
{
    struct ::rusage usage{};
    if (::getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0.0;
    }
    // `ru_maxrss` units differ across POSIX:
    //   * Linux: kilobytes
    //   * macOS: bytes
    // Normalise to MB. The branch is on the platform macro because
    // the unit is a build-time property — there is no runtime way
    // to ask the OS what units it returned.
#if defined(__APPLE__)
    constexpr double bytes_per_mb = 1024.0 * 1024.0;
    return static_cast<double>(usage.ru_maxrss) / bytes_per_mb;
#else
    constexpr double kb_per_mb = 1024.0;
    return static_cast<double>(usage.ru_maxrss) / kb_per_mb;
#endif
}

} // namespace knng::bench
