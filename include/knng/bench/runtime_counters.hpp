#pragma once

/// @file
/// @brief Runtime counters reported by every `knng` benchmark.
///
/// Google Benchmark already emits per-iteration wall time and CPU
/// time. This header adds the counters every later phase will care
/// about as it tries to defend a speed/quality tradeoff:
///
///   * `recall_at_k`              — quality of the approximate graph
///                                  against an exact reference.
///   * `peak_memory_mb`           — maximum resident set size of the
///                                  benchmark process so far.
///   * `n_distance_computations`  — algorithm-specific count of
///                                  pairwise distance evaluations
///                                  per iteration. For brute-force
///                                  intra-set KNN this is `n*(n-1)`;
///                                  later approximate builders set
///                                  whatever count they actually do.
///
/// The functions here are tiny, allocation-free, and have a stable
/// API so that every bench TU in the project (CPU brute-force today,
/// SIMD distance kernels in Phase 4, GPU kernels in Phase 7+) can
/// emit the same JSON shape. `tools/plot_bench.py` keys on the field
/// names below, so renaming a counter is a binding-format change.

#include <cstddef>
#include <cstdint>

namespace knng::bench {

/// Peak resident set size of the calling process so far, in
/// megabytes (1 MB = 1'048'576 B). Implemented via `getrusage(2)`'s
/// `ru_maxrss`, which on every supported development platform —
/// macOS, Linux on x86_64, Linux on arm64 — reports the maximum
/// RSS the process has ever held since launch. The unit returned
/// by the syscall differs (kB on Linux, bytes on macOS); this
/// helper normalises to MB so callers do not have to special-case.
///
/// Returns `0.0` if `getrusage` fails (which it does not on any
/// realistic POSIX configuration). The value is a `double` rather
/// than an integer because Google Benchmark's counter map is
/// `std::map<std::string, Counter>` and `Counter` only stores
/// `double`.
[[nodiscard]] double peak_memory_mb() noexcept;

/// Number of pairwise distance evaluations a single brute-force
/// KNN call performs over a dataset of `n` points. The `(n-1)`
/// factor reflects that brute-force excludes self-matches per row;
/// this is the same arithmetic the bench harness already used to
/// scale `SetItemsProcessed`, but lifted out here so every bench
/// TU emits an identical `n_distance_computations` counter without
/// re-deriving the formula.
[[nodiscard]] inline std::int64_t brute_force_distance_count(
    std::size_t n) noexcept
{
    if (n < 2) {
        return 0;
    }
    return static_cast<std::int64_t>(n)
         * static_cast<std::int64_t>(n - 1);
}

} // namespace knng::bench
