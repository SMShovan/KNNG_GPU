#pragma once

/// @file
/// @brief Naive O(n²·d) brute-force exact K-nearest-neighbor builder.
///
/// `brute_force_knn` is the correctness floor for the entire project.
/// Every later optimisation — vectorised distance kernels, blocked
/// tiling, BLAS-as-distance, GPU brute-force, NN-Descent, multi-GPU
/// distribution — is measured for both wall time and recall against
/// the output of this function. The implementation deliberately makes
/// no concession to performance: a triple loop, a per-query `TopK`
/// buffer, no SIMD, no parallelism, no heuristics.
///
/// Determinism: the output of `brute_force_knn` is a pure function of
/// the input dataset, the chosen `k`, and the chosen `Distance`
/// functor. There is no RNG anywhere; the per-query buffer's
/// equal-distance tie-break by smaller neighbor id (Step 09) carries
/// over so two distance-equivalent neighbors are always emitted in
/// ascending-id order. This is the property every subsequent
/// elementwise-equality regression test will rely on.

#include <cstddef>
#include <stdexcept>
#include <string>

#include "knng/core/dataset.hpp"
#include "knng/core/distance.hpp"
#include "knng/core/graph.hpp"
#include "knng/core/types.hpp"
#include "knng/top_k.hpp"

namespace knng::cpu {

/// L2 brute-force builder using the precomputed-norms identity.
///
/// Mathematically identical to `brute_force_knn(ds, k, L2Squared{})`:
///
///     ||a - b||²  =  ||a||²  +  ||b||²  -  2 · ⟨a, b⟩
///
/// The right-hand side replaces each pair's `O(d)` subtract-and-
/// square-and-sum with a `O(d)` multiply-and-accumulate plus three
/// scalar adds. Under fp32 it is ~1.5× the work per pair on paper
/// (`d` muls + `d` adds vs `d` subs + `d` muls + `d` adds), but the
/// muls fuse with the loads on every modern CPU and the
/// post-`compute_norms_squared` phase pays the per-row cost exactly
/// once, so the inner loop is dominated by the dot product alone.
/// Step 21 swaps the dot product for `cblas_sgemm`; this step is
/// the algebraic predecessor that makes that swap a one-line change.
///
/// Output is bit-identical to the canonical `brute_force_knn` path
/// in fp32 *only* up to the floating-point reordering of the
/// accumulation; the test suite asserts row-equality of neighbor
/// IDs and `EXPECT_NEAR` of distances within a small relative
/// tolerance.
///
/// @param ds Reference / query set.
/// @param k Number of neighbors per point. Same constraints as
///          `brute_force_knn`: `1 <= k <= ds.n - 1`.
/// @return A `Knng` of shape `(ds.n, k)`; rows sorted ascending by
///         distance with ties broken by ascending neighbor index.
/// @throws std::invalid_argument on malformed inputs.
[[nodiscard]] Knng brute_force_knn_l2_with_norms(const Dataset& ds,
                                                 std::size_t k);

/// L2 brute-force builder with `(QUERY_TILE × REF_TILE)` blocking.
///
/// Builds on the algebraic identity from
/// `brute_force_knn_l2_with_norms` and adds an outer-loop tiling
/// scheme designed for L1 residency:
///
/// ```text
///   for each q_tile of QUERY_TILE rows:
///       initialise QUERY_TILE TopK heaps
///       for each r_tile of REF_TILE rows:
///           compute the (QUERY_TILE × REF_TILE) distance block
///           push every (q, r) pair into the matching heap
///       write the q_tile's heaps into the output Knng
/// ```
///
/// The reference tile is reused across `QUERY_TILE` queries before
/// being evicted from L1; the heap state stays in registers /
/// L1 throughout the q_tile. On AppleClang at d=128 this drops the
/// L1 miss rate measurably vs the per-query scan.
///
/// Tile sizes are tunable via the optional parameters but default
/// to `QUERY_TILE = 32`, `REF_TILE = 128` — values chosen so that
/// `QUERY_TILE × REF_TILE × 2 × sizeof(float) =~ 32 KB`, matching
/// a typical x86_64 / arm64 L1 data cache. Step 23's profiling
/// writeup will validate or revise these.
///
/// Output is bit-equivalent (within fp accumulation reordering) to
/// `brute_force_knn(ds, k, L2Squared{})`. Same constraints apply:
/// `1 <= k <= ds.n - 1`, contiguous dataset.
[[nodiscard]] Knng brute_force_knn_l2_tiled(
    const Dataset& ds,
    std::size_t k,
    std::size_t query_tile = 32,
    std::size_t ref_tile = 128);

/// Build an exact K-nearest-neighbor graph by brute force.
///
/// For each row `q` of `ds`, the function scores every other row `r`
/// under the chosen distance functor, retains the `k` smallest, and
/// writes the resulting `(neighbor, distance)` pairs in ascending
/// distance order into the output `Knng`. Self-matches (`r == q`) are
/// excluded.
///
/// @tparam D A type satisfying the `knng::Distance` concept.
/// @param ds Reference / query set (intra-set KNN — every row is both).
/// @param k Number of neighbors per point. Must satisfy
///          `1 <= k <= ds.n - 1`.
/// @param distance Distance functor instance. Default-constructed when
///          `D` is default-constructible.
/// @return A `Knng` of shape `(ds.n, k)`; rows sorted ascending by
///         distance with ties broken by ascending neighbor index.
/// @throws std::invalid_argument if `ds.n == 0`, `k == 0`, or
///         `k > ds.n - 1`.
template <Distance D>
Knng brute_force_knn(const Dataset& ds, std::size_t k, D distance = D{})
{
    if (ds.n == 0) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn: dataset is empty");
    }
    if (k == 0) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn: k must be > 0");
    }
    if (k > ds.n - 1) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn: k (" + std::to_string(k)
            + ") must be <= ds.n - 1 ("
            + std::to_string(ds.n - 1) + ")");
    }

    Knng out(ds.n, k);
    for (std::size_t q = 0; q < ds.n; ++q) {
        TopK heap(k);
        const auto query = ds.row(q);
        for (std::size_t r = 0; r < ds.n; ++r) {
            if (r == q) {
                continue;
            }
            const float d = distance(query, ds.row(r));
            heap.push(static_cast<index_t>(r), d);
        }
        const auto sorted = heap.extract_sorted();
        // The TopK invariant guarantees `sorted.size() == k` here
        // because we offered `ds.n - 1 >= k` distinct candidates.
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
