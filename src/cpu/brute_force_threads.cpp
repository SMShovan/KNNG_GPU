/// @file
/// @brief `std::thread` + atomic-counter work-queue brute-force (Step 27).
///
/// A pedagogical sibling to `src/cpu/brute_force_omp.cpp`. Same
/// algorithm, different parallelism API. The intent is to show
/// the OpenMP version's `#pragma omp parallel for schedule(static)`
/// is not magic — it expands to roughly the body of this file —
/// and to leave the project with both versions side-by-side so
/// future contributors can compare the source-line cost and
/// performance.
///
/// Work-queue shape: a single `std::atomic<std::size_t> next{0}`
/// that every worker `fetch_add`-s to claim one query at a time.
/// Dynamic load balancing for free; no mutex; no condition
/// variables. The atomic counter is the simplest "work queue" that
/// works for an embarrassingly-parallel loop where every iteration
/// is independent and roughly the same cost.

#include "knng/cpu/brute_force.hpp"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "knng/cpu/distance.hpp"

namespace knng::cpu {

Knng brute_force_knn_l2_threaded(const Dataset& ds,
                                  std::size_t k,
                                  int num_threads)
{
    if (ds.n == 0) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_threaded: dataset is empty");
    }
    if (k == 0) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_threaded: k must be > 0");
    }
    if (k > ds.n - 1) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_threaded: k ("
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

    // Worker count. Default to the hardware's reported concurrency;
    // the runtime may report 0 if it cannot determine a value, in
    // which case we fall back to 1 so the single-threaded path
    // still runs.
    int workers = num_threads;
    if (workers <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        workers = static_cast<int>(hw == 0 ? 1u : hw);
    }

    // The atomic work counter. `relaxed` order is sufficient: the
    // counter's only purpose is mutual exclusion of `q` indices;
    // the per-query work writes into disjoint rows of `out`, so no
    // happens-before relationship is needed between threads.
    std::atomic<std::size_t> next{0};

    auto worker_body = [&]() {
        TopK heap(k);
        while (true) {
            const std::size_t q =
                next.fetch_add(1, std::memory_order_relaxed);
            if (q >= n) {
                return;
            }
            const float* a       = base + q * stride;
            const float  norm_a  = norms[q];

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
            // `extract_sorted` left the heap empty for the next
            // iteration; no allocation pressure across queries.
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers));
    for (int t = 0; t < workers; ++t) {
        threads.emplace_back(worker_body);
    }
    for (auto& t : threads) {
        t.join();
    }

    return out;
}

} // namespace knng::cpu
