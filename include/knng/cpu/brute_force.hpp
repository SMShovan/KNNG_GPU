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

/// True iff the BLAS-backed builder
/// `brute_force_knn_l2_blas` is compiled into this build of
/// `knng::cpu`. Wraps the build-time `KNNG_HAVE_BLAS` macro into a
/// compile-time constant callers can check without preprocessor
/// guards in user code.
inline constexpr bool kHasBlasBuiltin =
#if defined(KNNG_HAVE_BLAS) && KNNG_HAVE_BLAS
    true;
#else
    false;
#endif

#if defined(KNNG_HAVE_BLAS) && KNNG_HAVE_BLAS
/// L2 brute-force builder using `cblas_sgemm` for the cross term.
///
/// This is the headline Phase-3 CPU optimisation. The algebraic
/// identity from Step 19,
///
///     ||a - b||²  =  ||a||²  +  ||b||²  -  2 · ⟨a, b⟩
///
/// rewrites into the *matrix* form
///
///     D[i, j]  =  ||x_i||²  +  ||y_j||²  -  2 · (X · Yᵀ)[i, j]
///
/// where `X` and `Y` are tile-sized slices of the dataset. The
/// expensive `(QUERY_TILE × REF_TILE × d)` matmul becomes a single
/// `cblas_sgemm` call against an industrially-tuned BLAS — on
/// macOS the Apple Accelerate framework, on Linux OpenBLAS / MKL —
/// and the per-element norm fold-in is a tiny scalar epilogue.
///
/// This is also the algorithmic preview for Step 55's GPU port:
/// `cublasSgemm` over device-side `X` and `Y` tiles, fused norm
/// epilogue kernel, identical mathematical shape. The day Step 55
/// ships, it is a translation, not a rederivation.
///
/// Output is bit-equivalent (within fp accumulation reordering) to
/// `brute_force_knn(ds, k, L2Squared{})` on every test fixture.
///
/// @param ds Reference / query set.
/// @param k Number of neighbors per point.
/// @param query_tile Number of queries per outer tile (default 64).
/// @param ref_tile Number of references per inner tile (default 256).
/// @return A `Knng` of shape `(ds.n, k)`; rows sorted ascending by
///         distance with ties broken by ascending neighbor index.
/// @throws std::invalid_argument on malformed inputs.
[[nodiscard]] Knng brute_force_knn_l2_blas(
    const Dataset& ds,
    std::size_t k,
    std::size_t query_tile = 64,
    std::size_t ref_tile = 256);
#endif  // KNNG_HAVE_BLAS

/// L2 brute-force builder using `std::partial_sort` instead of the
/// streaming `TopK` heap.
///
/// Uses Step 19's norms-precompute identity for the distance
/// computation, then materialises the full `(n - 1)`-element
/// candidate buffer per query and runs `std::partial_sort` to
/// extract the `k` smallest. The contrast with the heap path:
///
///   * `TopK` is `O(log k)` per push; total `O((n-1) log k)` per
///     query. The streaming admission test branches on every
///     candidate. Cache-friendly only by accident.
///   * `std::partial_sort` over a contiguous `n - 1`-element buffer
///     is `O((n-1) log k)` *worst-case* but with much better
///     constants — the algorithm is a max-heapify over the first k
///     elements followed by a single linear scan with sift-down.
///     The buffer is contiguous; the inner loop branches less.
///
/// Tie-breaking matches the heap path: equal distances are broken
/// by ascending neighbor id, so the output is bit-equivalent (up to
/// fp accumulation reordering of the dot product itself) to
/// `brute_force_knn_l2_with_norms` on every test fixture.
///
/// Memory cost: one `(n - 1)`-element scratch buffer of
/// `(index_t, float)` pairs, allocated once and reused across
/// queries. ~8 bytes per reference.
///
/// @param ds Reference / query set.
/// @param k Number of neighbors per point.
/// @return A `Knng` of shape `(ds.n, k)`; rows sorted ascending by
///         distance with ties broken by ascending neighbor index.
/// @throws std::invalid_argument on malformed inputs.
[[nodiscard]] Knng brute_force_knn_l2_partial_sort(
    const Dataset& ds,
    std::size_t k);

/// True iff this build of `knng::cpu` was compiled against an
/// OpenMP runtime. Wraps the build-time `KNNG_HAVE_OPENMP` macro
/// into a compile-time constant callers can check without
/// preprocessor guards.
inline constexpr bool kHasOpenmpBuiltin =
#if defined(KNNG_HAVE_OPENMP) && KNNG_HAVE_OPENMP
    true;
#else
    false;
#endif

/// L2 brute-force builder, OpenMP-parallel over the outer query
/// loop.
///
/// Algorithmically identical to `brute_force_knn_l2_with_norms`
/// (Step 19): same precomputed norms, same `||a||² + ||b||² - 2⟨a,b⟩`
/// identity, same `TopK` heap admission. The only structural change
/// is `#pragma omp parallel for schedule(static)` on the outer
/// query loop.
///
/// Thread safety: each iteration writes into its own row of the
/// output `Knng` (`out.neighbors_of(q)` and `out.distances_of(q)`
/// are disjoint for distinct `q`) and uses its own `TopK` instance
/// declared inside the loop. No locks, no atomics, no shared state
/// beyond read-only `ds`, `norms`, and `base`.
///
/// Behaves identically to the serial path when the build did not
/// link OpenMP (`!kHasOpenmpBuiltin`): the pragma is a comment and
/// the loop runs single-threaded.
///
/// @param ds Reference / query set.
/// @param k Number of neighbors per point.
/// @param num_threads If > 0, sets `omp_set_num_threads(num_threads)`
///        for this call. If 0 (default), uses the runtime's default
///        (`OMP_NUM_THREADS` or the hardware concurrency).
/// @return A `Knng` of shape `(ds.n, k)`; rows sorted ascending by
///         distance with ties broken by ascending neighbor index.
/// @throws std::invalid_argument on malformed inputs.
[[nodiscard]] Knng brute_force_knn_l2_omp(
    const Dataset& ds,
    std::size_t k,
    int num_threads = 0);

/// L2 brute-force builder, OpenMP-parallel with per-thread scratch.
///
/// Same algorithm as `brute_force_knn_l2_omp` but with two
/// false-sharing-aware refinements:
///
///   1. **Per-thread `TopK` heap pre-allocated once.** Step 24's
///      variant declares the heap inside the parallel-for body, so
///      every iteration paid the heap's `std::priority_queue`'s
///      vector allocation. This variant pre-allocates one heap per
///      OpenMP worker, then `extract_sorted` drains it between
///      iterations — the allocation amortises across all queries.
///   2. **Cache-line padding between per-thread heaps.** Each
///      `ThreadScratch` is `alignas(64)` and padded so two
///      adjacent heaps cannot share a cache line. Without padding,
///      threads writing to neighbouring heaps would ping-pong the
///      shared line through the LLC; with padding, every thread
///      owns its own line cleanly.
///
/// Output is bit-equivalent to `brute_force_knn_l2_omp`. The win
/// shows up at large `n` where the per-query allocation cost is a
/// non-trivial fraction of the iteration body.
///
/// @param ds Reference / query set.
/// @param k Number of neighbors per point.
/// @param num_threads If > 0, sets the OpenMP team size for this
///        call. If 0, uses the runtime's default.
/// @return A `Knng` of shape `(ds.n, k)`.
/// @throws std::invalid_argument on malformed inputs.
[[nodiscard]] Knng brute_force_knn_l2_omp_scratch(
    const Dataset& ds,
    std::size_t k,
    int num_threads = 0);

/// L2 brute-force builder using `std::thread` + atomic work queue.
///
/// A learning-exercise alternative to the OpenMP variants
/// (`brute_force_knn_l2_omp` / `_omp_scratch`). Spawns
/// `num_threads` workers that consume queries from a single shared
/// `std::atomic<std::size_t>` counter via `fetch_add`. Each worker
/// processes whichever query it grabbed, then loops back for the
/// next — giving dynamic load balancing without a mutex-protected
/// queue.
///
/// API contrasts (vs OpenMP path):
///   * **Source-line cost.** OpenMP version is ~30 lines; this is
///     ~50. `std::thread` ergonomics (capture-by-reference for
///     shared state, explicit join, `std::atomic` for the
///     counter) are louder than `#pragma omp parallel for`.
///   * **Wall-time.** Roughly equivalent on this fixture; the
///     atomic-counter contention is invisible at n=1024 because
///     each query takes ~50 µs, far longer than the
///     `fetch_add` cycle. On workloads with sub-microsecond
///     per-query work, the atomic would become a bottleneck.
///   * **Determinism.** Both paths emit bit-identical output for
///     the same input — atomic-counter dispatch only changes
///     *which thread* processes which query, not the output.
///
/// Output is bit-equivalent to `brute_force_knn_l2_omp_scratch`.
///
/// @param ds Reference / query set.
/// @param k Number of neighbors per point.
/// @param num_threads If > 0, spawns this many workers. If 0
///        (default), uses `std::thread::hardware_concurrency()`.
/// @return A `Knng` of shape `(ds.n, k)`.
/// @throws std::invalid_argument on malformed inputs.
[[nodiscard]] Knng brute_force_knn_l2_threaded(
    const Dataset& ds,
    std::size_t k,
    int num_threads = 0);

/// L2 brute-force builder using the hand-vectorised dot-product
/// primitive (`simd_dot_product`).
///
/// Same algorithmic shape as `brute_force_knn_l2_with_norms`
/// (Step 19): same precomputed norms, same algebraic identity,
/// same `TopK` heap. The only structural change is the inner-loop
/// dot product is replaced with `knng::cpu::simd_dot_product`,
/// which compile-time picks the AVX2 / NEON / scalar path
/// available on the build target and runtime-degrades to scalar
/// on x86 binaries running on a non-AVX2 CPU.
///
/// Output is bit-equivalent (within fp accumulation reordering)
/// to the canonical L2 builder. The win shows up at large `d`
/// where the per-pair dot product dominates the inner loop and
/// the compiler's autovectoriser leaves throughput on the table.
[[nodiscard]] Knng brute_force_knn_l2_simd(const Dataset& ds,
                                           std::size_t k);

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
