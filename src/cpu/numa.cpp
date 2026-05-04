/// @file
/// @brief Implementation of the first-touch NUMA helper.
///
/// The function is `noexcept` and does not allocate; the OS may
/// page-fault on uncached pages but that is the *purpose* of the
/// pass (to bind each page to the touching thread's NUMA node).
/// Apple Silicon has a single NUMA domain so the redistribution
/// is a no-op; the cache-warm side effect is still useful.

#include "knng/cpu/numa.hpp"

#include <cstddef>
#include <unistd.h>

#if defined(KNNG_HAVE_OPENMP) && KNNG_HAVE_OPENMP
#  include <omp.h>
#endif

namespace knng::cpu {

namespace {

/// Bytes per OS page on the current host. `sysconf(_SC_PAGESIZE)`
/// returns the value at runtime so a future port to a host with
/// non-4 KB pages (e.g. Apple Silicon's M-series uses 16 KB pages,
/// `aarch64` Linux can be 4 KB or 64 KB) does not need code changes.
[[nodiscard]] std::size_t page_size_bytes() noexcept
{
    const long sc = ::sysconf(_SC_PAGESIZE);
    if (sc <= 0) {
        // sysconf() failure is essentially impossible in practice.
        // Fall back to the cross-platform safe value.
        return 4096;
    }
    return static_cast<std::size_t>(sc);
}

} // namespace

void first_touch(float* data, std::size_t n_elements,
                 int num_threads) noexcept
{
    if (data == nullptr || n_elements == 0) {
        return;
    }

    const std::size_t page_floats =
        page_size_bytes() / sizeof(float);
    if (page_floats == 0) {
        return;
    }

#if defined(KNNG_HAVE_OPENMP) && KNNG_HAVE_OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#else
    (void)num_threads;
#endif

    const long long n_signed = static_cast<long long>(n_elements);
    const long long step     = static_cast<long long>(page_floats);

    // The pragma matches Step 24+'s `schedule(static)` so the
    // page-to-thread binding aligns with the later read pattern.
    // The body re-writes each touched cell to itself — preserves
    // contents while forcing the page-fault path that lazy-binds
    // the page on Linux.
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n_signed; i += step) {
        data[i] = data[i];
    }
}

bool is_numa_relevant_platform() noexcept
{
#if defined(__linux__)
    // We do not attempt a runtime detection of how many NUMA nodes
    // the host actually has; libnuma + `numa_available()` would be
    // the right tool but is an extra dependency. Conservative
    // behaviour: assume a Linux build is on a host where NUMA
    // matters. A future refinement can swap the constant for a
    // libnuma probe.
    return true;
#else
    return false;
#endif
}

} // namespace knng::cpu
