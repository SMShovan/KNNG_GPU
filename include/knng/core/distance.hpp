#pragma once

/// @file
/// @brief Distance metrics and the `Distance` concept that unifies them.
///
/// Every nearest-neighbor algorithm in the library is parameterized on a
/// distance functor satisfying the `knng::Distance` concept. Writing
/// algorithms against the concept — instead of a concrete metric —
/// means the same CPU reference builder, the same GPU kernels, and the
/// same refinement routines can serve L2, inner product, cosine, or
/// any future metric without code duplication.
///
/// Contract of a `Distance` functor:
///   * Callable as `float d(a, b)` where `a` and `b` are equal-length
///     `std::span<const float>`s.
///   * Must be `noexcept`. Distance calls sit in the innermost loop of
///     both search and refinement; throwing would be a latent
///     correctness + performance hazard.
///   * Must be *monotone lower-is-better*: if point x is closer to q
///     than point y under the metric, `d(q, x) < d(q, y)`. This is
///     why `InnerProduct` is exposed as `NegativeInnerProduct` — we
///     keep a single ordering convention across the library.

#include <concepts>
#include <cstddef>
#include <span>

namespace knng {

/// Scalar squared-L2 distance over two equal-length float buffers.
///
/// This is the lower-level C-style cousin of `L2Squared`: it takes raw
/// pointers and an explicit length so that later SIMD intrinsics
/// (Phase 4) and CUDA / HIP kernels (Phase 7+) can specialise the
/// `(const float*, const float*, std::size_t)` signature without
/// reaching through a `std::span` wrapper that those toolchains often
/// inline poorly across ABI boundaries. The functor `L2Squared` below
/// dispatches to this function for its single source of truth.
///
/// Contract:
///   * `a` and `b` point to at least `dim` valid floats. No bounds
///     checking — this is an inner-loop primitive.
///   * `dim == 0` returns `0.0f` (the empty sum). No division, no
///     special cases.
///   * Result is the sum of squared componentwise differences; it is
///     never negative and is finite whenever the inputs are finite.
///
/// @param a first feature vector, length `dim`
/// @param b second feature vector, length `dim`
/// @param dim number of float components in each vector
/// @return Σ (a[i] - b[i])² over i ∈ [0, dim)
[[nodiscard]] inline float squared_l2(const float* a,
                                      const float* b,
                                      std::size_t dim) noexcept
{
    float acc = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        const float delta = a[i] - b[i];
        acc += delta * delta;
    }
    return acc;
}


/// Concept: a callable that scores the dissimilarity of two equal-length
/// float vectors. See file-level documentation for the full contract.
template <class F>
concept Distance = requires(const F& f,
                            std::span<const float> a,
                            std::span<const float> b) {
    { f(a, b) } noexcept -> std::same_as<float>;
};

/// Squared Euclidean distance.
///
/// Returned unrooted for two reasons:
///   1. `sqrt` is monotone on `[0, inf)`, so squared-L2 and L2 produce
///      identical nearest-neighbor orderings. Every algorithm in the
///      library only needs the ordering, not the magnitude.
///   2. `sqrt` is a relatively expensive op on CPU (one FP instruction)
///      and a *very* expensive op on many GPUs (high-latency special
///      function unit). Removing it from the inner loop is measurable.
///
/// Callers that need true Euclidean magnitude for a final report can
/// take the square root once, outside the search loop.
struct L2Squared {
    float operator()(std::span<const float> a,
                     std::span<const float> b) const noexcept
    {
        // Single source of truth for the scalar formula — see
        // `squared_l2` above. Per the concept contract, callers
        // guarantee `a.size() == b.size()`.
        return squared_l2(a.data(), b.data(), a.size());
    }
};
static_assert(Distance<L2Squared>);

/// Negated inner product — lower is more similar.
///
/// Plain inner-product similarity is *higher-is-better*, which would
/// force every search/refinement routine in the library to special-case
/// the ordering. Negating at the metric boundary keeps the rest of the
/// code mono-orientation (smaller value == closer neighbor) and is the
/// same trick FAISS uses for its IP index.
struct NegativeInnerProduct {
    float operator()(std::span<const float> a,
                     std::span<const float> b) const noexcept
    {
        float acc = 0.0f;
        const std::size_t n = a.size();
        for (std::size_t i = 0; i < n; ++i) {
            acc += a[i] * b[i];
        }
        return -acc;
    }
};
static_assert(Distance<NegativeInnerProduct>);

} // namespace knng
