/// @file
/// @brief Translation unit for `knng::cpu::brute_force_knn` and
///        the Step-19 norms-precompute variant.
///
/// The generic algorithm itself is a function template defined in
/// `include/knng/cpu/brute_force.hpp` (see that file for the
/// contract). This `.cpp` exists for two reasons:
///
///   1. To explicitly instantiate the template for the two built-in
///      distance functors so that downstream callers using `L2Squared`
///      or `NegativeInnerProduct` link against pre-compiled symbols
///      and pay the parsing / instantiation cost exactly once. Other
///      `Distance`-satisfying functors are still implicitly
///      instantiated at the consumer's site (the explicit
///      instantiations do not preclude implicit ones).
///   2. To give the `knng_cpu` static library a real translation unit.
///      Until Step 10 the project shipped only INTERFACE libraries;
///      `knng_cpu` is the first STATIC target.
///
/// Step 19 adds `brute_force_knn_l2_with_norms` here as well — an
/// L2-specific entry point that precomputes the per-row squared
/// norm vector once and replaces each pair's subtract-and-square
/// with the algebraic identity ||a - b||² = ||a||² + ||b||² - 2⟨a,b⟩.

#include "knng/cpu/brute_force.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "knng/cpu/distance.hpp"

namespace knng::cpu {

template Knng brute_force_knn<L2Squared>(
    const Dataset&, std::size_t, L2Squared);

template Knng brute_force_knn<NegativeInnerProduct>(
    const Dataset&, std::size_t, NegativeInnerProduct);

Knng brute_force_knn_l2_tiled(const Dataset& ds,
                               std::size_t k,
                               std::size_t query_tile,
                               std::size_t ref_tile)
{
    if (ds.n == 0) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_tiled: dataset is empty");
    }
    if (k == 0) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_tiled: k must be > 0");
    }
    if (k > ds.n - 1) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_tiled: k ("
            + std::to_string(k) + ") must be <= ds.n - 1 ("
            + std::to_string(ds.n - 1) + ")");
    }
    if (query_tile == 0 || ref_tile == 0) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_tiled: tile sizes must be > 0");
    }
    assert(ds.is_contiguous());

    std::vector<float> norms;
    compute_norms_squared(ds, norms);

    const float*      base   = ds.data_ptr();
    const std::size_t stride = ds.stride();

    Knng out(ds.n, k);

    // Per-tile heap workspace, reused across q_tile iterations.
    // We build TopKs anew each q_tile; the explicit reserve avoids
    // the realloc on the very first push.
    std::vector<TopK> heaps;
    heaps.reserve(query_tile);

    for (std::size_t q_lo = 0; q_lo < ds.n; q_lo += query_tile) {
        const std::size_t q_hi = std::min(q_lo + query_tile, ds.n);
        const std::size_t q_n  = q_hi - q_lo;

        heaps.clear();
        for (std::size_t i = 0; i < q_n; ++i) {
            heaps.emplace_back(k);
        }

        for (std::size_t r_lo = 0; r_lo < ds.n; r_lo += ref_tile) {
            const std::size_t r_hi = std::min(r_lo + ref_tile, ds.n);

            for (std::size_t q = q_lo; q < q_hi; ++q) {
                const float* a      = base + q * stride;
                const float  norm_a = norms[q];
                TopK&        heap   = heaps[q - q_lo];

                for (std::size_t r = r_lo; r < r_hi; ++r) {
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
            }
        }

        for (std::size_t q = q_lo; q < q_hi; ++q) {
            const auto sorted = heaps[q - q_lo].extract_sorted();
            auto neighbors_row = out.neighbors_of(q);
            auto distances_row = out.distances_of(q);
            for (std::size_t j = 0; j < sorted.size(); ++j) {
                neighbors_row[j] = sorted[j].first;
                distances_row[j] = sorted[j].second;
            }
        }
    }

    return out;
}

Knng brute_force_knn_l2_with_norms(const Dataset& ds, std::size_t k)
{
    if (ds.n == 0) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_with_norms: dataset is empty");
    }
    if (k == 0) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_with_norms: k must be > 0");
    }
    if (k > ds.n - 1) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_with_norms: k ("
            + std::to_string(k) + ") must be <= ds.n - 1 ("
            + std::to_string(ds.n - 1) + ")");
    }
    assert(ds.is_contiguous());

    // ||p||² for each row, computed once — amortised over (n-1)
    // distance evaluations per query.
    std::vector<float> norms;
    compute_norms_squared(ds, norms);

    const float*      base   = ds.data_ptr();
    const std::size_t stride = ds.stride();

    Knng out(ds.n, k);
    for (std::size_t q = 0; q < ds.n; ++q) {
        TopK heap(k);
        const float* a       = base + q * stride;
        const float  norm_a  = norms[q];

        for (std::size_t r = 0; r < ds.n; ++r) {
            if (r == q) {
                continue;
            }
            const float* b      = base + r * stride;
            const float  norm_b = norms[r];

            // ||a - b||² via the algebraic identity. The
            // subtraction can produce a small negative result
            // for self-similar points under fp32 cancellation
            // (a == b after rounding). Clamp at zero so the
            // ranking key never goes negative and TopK's ordering
            // is preserved against the canonical path.
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

} // namespace knng::cpu
