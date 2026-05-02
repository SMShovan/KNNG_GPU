#pragma once

/// @file
/// @brief Vectorisation-friendly CPU distance primitives.
///
/// `knng::core::distance` ships the canonical scalar `squared_l2`
/// (`include/knng/core/distance.hpp`); this header adds the two
/// helpers Step 19 needs to rewrite squared-L2 as
///
///     ||a - b||²  =  ||a||²  +  ||b||²  -  2 · ⟨a, b⟩
///
/// for the brute-force builder:
///
///   * `compute_norms_squared(Dataset, out)` — the `n`-element vector
///     of `||row_i||²` precomputed once before the timed loop.
///   * `dot_product(a, b, dim)` — a `(const float*, const float*,
///     std::size_t)`-shaped scalar inner product. Same signature as
///     `squared_l2` so a future SIMD specialisation (Step 27) can
///     overload either function in lockstep.
///
/// Both functions live in `knng::cpu` rather than `knng::core` so
/// the GPU kernels in Phase 7+ can keep `knng::core::distance.hpp`
/// header-only (no transitive `<numeric>` / `<vector>` cost) while
/// the CPU translation units pick this up explicitly.

#include <cstddef>
#include <vector>

#include "knng/core/dataset.hpp"

namespace knng::cpu {

/// Standard scalar dot product over two equal-length float buffers.
///
/// Contract:
///   * `a` and `b` point to at least `dim` valid floats. No bounds
///     checking — this is an inner-loop primitive.
///   * `dim == 0` returns `0.0f` (the empty sum).
///   * No special-casing for zero / NaN / inf inputs — the result
///     is whatever IEEE-754 produces given the inputs.
///
/// The function is `inline` so a hot inner loop can specialise on
/// `dim` and (eventually) on a SIMD width without paying for a
/// translation-unit boundary. Same argument shape as
/// `knng::squared_l2` — keeping the two primitives congruent makes
/// the Step-27 SIMD pass a search-and-replace rather than an API
/// rework.
[[nodiscard]] inline float dot_product(const float* a,
                                       const float* b,
                                       std::size_t dim) noexcept
{
    float acc = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

/// Compute the squared-L2 norm of every row of `ds` and write the
/// result into `out`. After the call, `out.size() == ds.n` and
/// `out[i] == Σ_j ds.row(i)[j] * ds.row(i)[j]`.
///
/// @param ds Dataset whose rows are to be normed. Must satisfy
///        `ds.is_contiguous()`; the function asserts this in debug
///        builds.
/// @param[out] out Destination buffer, resized to `ds.n` floats.
///
/// Cost: `O(n * d)` once per call. Used by Step 19's
/// `brute_force_knn_l2_with_norms` to avoid a per-pair difference
/// computation; the upfront cost is amortised over `n - 1` distance
/// evaluations per row, so the speedup grows with `n`.
void compute_norms_squared(const Dataset& ds, std::vector<float>& out);

} // namespace knng::cpu
