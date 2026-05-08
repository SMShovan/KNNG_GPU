#pragma once

/// @file
/// @brief GPU brute-force KNN declarations — all three tiers.
///
/// Three progressively better GPU implementations are provided across
/// Steps 48–50, each exposed through a host wrapper with the same
/// signature so that the caller can switch strategies by name:
///
/// | Function                     | Step | Kernel layout       | k limit |
/// |------------------------------|------|---------------------|---------|
/// | `brute_force_thread_knn`     |  48  | 1 thread / query    | ≤ 32    |
/// | `brute_force_block_knn`      |  49  | 1 block  / query    | ≤ 1024  |
/// | `gpu_brute_force_knn`        |  50  | auto-selects block  | any k   |
///
/// All functions share the same CPU reference `cpu_brute_force_knn` which
/// is always compiled (no CUDA dependency) and used as the correctness
/// oracle in unit tests.
///
/// ### Output layout
/// Results are written into a flat `index_t[n*k]` array (neighbor IDs,
/// row-major) and a flat `float[n*k]` array (squared-L2 distances,
/// row-major), both sorted ascending by distance within each row.

#include <knng/core/types.hpp>
#include <knng/core/dataset.hpp>
#include <knng/core/graph.hpp>

#include <cstddef>

namespace knng::gpu {

// ---------------------------------------------------------------------------
// CPU reference (always compiled)
// ---------------------------------------------------------------------------

/// @brief Exact brute-force KNN on the CPU.
///
/// Runs the full `n × n × d` triple loop, then top-k selects the k nearest
/// neighbours for each query.  O(n² d) time, O(nk) output space.
///
/// @param dataset  Input point set (n points, d dimensions).
/// @param k        Number of nearest neighbours to find.
/// @return         Knng with `n` rows of `k` neighbours each.
[[nodiscard]] knng::Knng cpu_brute_force_knn(
    const knng::Dataset& dataset,
    std::size_t          k);

// ---------------------------------------------------------------------------
// GPU kernels (compiled only when CUDA / HIP is available)
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)

/// @brief Thread-per-query GPU brute-force KNN (Step 48).
///
/// Each CUDA thread handles one query: iterates over all `n` references,
/// maintains a register-resident top-k heap of size `k` using insertion
/// sort.  Correct for all k, but only efficient for k ≤ 32 (register
/// spill becomes severe above that).
///
/// @param dataset  Input point set.
/// @param k        Neighbours per query (optimal: k ≤ 32).
/// @return         Knng sorted by distance.
[[nodiscard]] knng::Knng brute_force_thread_knn(
    const knng::Dataset& dataset,
    std::size_t          k);

/// @brief Block-per-query GPU brute-force KNN (Step 49).
///
/// One thread block per query; threads cooperate to scan all references
/// using a shared-memory top-k buffer.  Scales to arbitrary k (bounded
/// only by shared memory: `k × (sizeof(index_t) + sizeof(float))` bytes
/// per block).
///
/// @param dataset  Input point set.
/// @param k        Neighbours per query.
/// @return         Knng sorted by distance.
[[nodiscard]] knng::Knng brute_force_block_knn(
    const knng::Dataset& dataset,
    std::size_t          k);

/// @brief Step-50 end-to-end GPU brute-force KNN (auto-strategy).
///
/// Selects the block-per-query kernel for all k, matching the API of
/// the CPU `knng::cpu::brute_force_knn`.
///
/// @param dataset  Input point set.
/// @param k        Neighbours per query.
/// @return         Knng sorted by distance.
[[nodiscard]] knng::Knng gpu_brute_force_knn(
    const knng::Dataset& dataset,
    std::size_t          k);

#endif // KNNG_HAVE_CUDA || KNNG_HAVE_HIP

} // namespace knng::gpu
