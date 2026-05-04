/// @file
/// @brief OpenMP-parallel L2 brute-force builder (Step 24).
///
/// Compiled unconditionally; the OpenMP-specific bits live behind
/// `#if KNNG_HAVE_OPENMP` and degrade to a serial loop when the
/// build did not link an OpenMP runtime. The `#pragma omp` is a
/// comment under non-OpenMP compilers, so the source is portable
/// without a separate serial fallback file.
///
/// Algorithmic shape mirrors `brute_force_knn_l2_with_norms`
/// (Step 19) — same identity, same per-query `TopK` heap, same
/// per-pair clamp. The only addition is the `#pragma omp parallel
/// for schedule(static)` on the outer query loop.

#include "knng/cpu/brute_force.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(KNNG_HAVE_OPENMP) && KNNG_HAVE_OPENMP
#  include <omp.h>
#endif

#include "knng/cpu/distance.hpp"

namespace knng::cpu {

Knng brute_force_knn_l2_omp(const Dataset& ds,
                             std::size_t k,
                             int num_threads)
{
    if (ds.n == 0) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_omp: dataset is empty");
    }
    if (k == 0) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_omp: k must be > 0");
    }
    if (k > ds.n - 1) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_omp: k ("
            + std::to_string(k) + ") must be <= ds.n - 1 ("
            + std::to_string(ds.n - 1) + ")");
    }
    assert(ds.is_contiguous());

    std::vector<float> norms;
    compute_norms_squared(ds, norms);

    const float*      base   = ds.data_ptr();
    const std::size_t stride = ds.stride();
    const std::size_t n      = ds.n;

    Knng out(n, k);

#if defined(KNNG_HAVE_OPENMP) && KNNG_HAVE_OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#else
    (void)num_threads;
#endif

    // `schedule(static)` because each query is roughly the same
    // amount of work (n distance evaluations) — `dynamic` would
    // pay the scheduling overhead for no load-balance benefit.
    // The signed int loop variable is required by OpenMP 4.5 / 5.x.
    const long long n_signed = static_cast<long long>(n);
#pragma omp parallel for schedule(static)
    for (long long q_signed = 0; q_signed < n_signed; ++q_signed) {
        const std::size_t q       = static_cast<std::size_t>(q_signed);
        const float*      a       = base + q * stride;
        const float       norm_a  = norms[q];

        TopK heap(k);
        for (std::size_t r = 0; r < n; ++r) {
            if (r == q) {
                continue;
            }
            const float* b      = base + r * stride;
            const float  norm_b = norms[r];
            float dist =
                norm_a + norm_b - 2.0f * dot_product(a, b, stride);
            if (dist < 0.0f) {
                dist = 0.0f;
            }
            heap.push(static_cast<index_t>(r), dist);
        }

        const auto sorted = heap.extract_sorted();
        auto neighbors_row = out.neighbors_of(q);
        auto distances_row = out.distances_of(q);
        for (std::size_t j = 0; j < sorted.size(); ++j) {
            neighbors_row[j] = sorted[j].first;
            distances_row[j] = sorted[j].second;
        }
    }

    return out;
}

namespace {

/// Cache-line size assumed by `alignas` and padding. 64 bytes is
/// correct on every supported development platform: x86_64 (Intel
/// + AMD), arm64 Apple Silicon, arm64 Linux. Defining as a
/// `constexpr` rather than reaching for the C++17
/// `std::hardware_destructive_interference_size` because the
/// latter is `[[experimental]]` on libstdc++ and triggers a
/// warning under `-Wpedantic`.
constexpr std::size_t kCacheLineBytes = 64;

/// Per-thread scratch wrapper. Each instance lives on its own
/// cache line so adjacent worker threads cannot false-share the
/// heap's hot metadata (size, capacity pointer). The trailing
/// `pad` ensures that even if a future `TopK` change grows the
/// type, we still occupy a multiple of the line size.
struct alignas(kCacheLineBytes) ThreadScratch {
    TopK heap;

    explicit ThreadScratch(std::size_t k) : heap{k} {}

private:
    // Pad to the next multiple of the cache line, computed at
    // compile time so a `TopK` larger than the line size still
    // produces a sensible padding (zero in that case).
    static constexpr std::size_t kPadBytes =
        ((sizeof(TopK) + kCacheLineBytes - 1) / kCacheLineBytes)
            * kCacheLineBytes
        - sizeof(TopK);
    [[maybe_unused]] std::array<char,
        (kPadBytes == 0 ? kCacheLineBytes : kPadBytes)> pad{};
};

} // namespace

Knng brute_force_knn_l2_omp_scratch(const Dataset& ds,
                                     std::size_t k,
                                     int num_threads)
{
    if (ds.n == 0) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_omp_scratch: dataset is empty");
    }
    if (k == 0) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_omp_scratch: k must be > 0");
    }
    if (k > ds.n - 1) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_omp_scratch: k ("
            + std::to_string(k) + ") must be <= ds.n - 1 ("
            + std::to_string(ds.n - 1) + ")");
    }
    assert(ds.is_contiguous());

    std::vector<float> norms;
    compute_norms_squared(ds, norms);

    const float*      base   = ds.data_ptr();
    const std::size_t stride = ds.stride();
    const std::size_t n      = ds.n;

    Knng out(n, k);

#if defined(KNNG_HAVE_OPENMP) && KNNG_HAVE_OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    const int requested = (num_threads > 0)
                            ? num_threads
                            : omp_get_max_threads();
#else
    (void)num_threads;
    constexpr int requested = 1;
#endif

    // Pre-allocate one cache-line-aligned heap per worker. The
    // emplace-loop is deliberate — `std::vector<ThreadScratch>(N, ThreadScratch{k})`
    // would copy-construct N times; emplace is one allocation
    // and N in-place constructions.
    std::vector<ThreadScratch> scratch;
    scratch.reserve(static_cast<std::size_t>(requested));
    for (int i = 0; i < requested; ++i) {
        scratch.emplace_back(k);
    }

    const long long n_signed = static_cast<long long>(n);
#pragma omp parallel for schedule(static)
    for (long long q_signed = 0; q_signed < n_signed; ++q_signed) {
        const std::size_t q       = static_cast<std::size_t>(q_signed);
        const float*      a       = base + q * stride;
        const float       norm_a  = norms[q];

#if defined(KNNG_HAVE_OPENMP) && KNNG_HAVE_OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        TopK& heap = scratch[static_cast<std::size_t>(tid)].heap;

        for (std::size_t r = 0; r < n; ++r) {
            if (r == q) {
                continue;
            }
            const float* b      = base + r * stride;
            const float  norm_b = norms[r];
            float dist =
                norm_a + norm_b - 2.0f * dot_product(a, b, stride);
            if (dist < 0.0f) {
                dist = 0.0f;
            }
            heap.push(static_cast<index_t>(r), dist);
        }

        // `extract_sorted` drains the heap into `sorted` and leaves
        // the priority_queue empty (capacity preserved) — ready
        // for the next iteration this thread picks up.
        const auto sorted = heap.extract_sorted();
        auto neighbors_row = out.neighbors_of(q);
        auto distances_row = out.distances_of(q);
        for (std::size_t j = 0; j < sorted.size(); ++j) {
            neighbors_row[j] = sorted[j].first;
            distances_row[j] = sorted[j].second;
        }
    }

    return out;
}

} // namespace knng::cpu
