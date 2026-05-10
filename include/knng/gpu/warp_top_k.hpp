#pragma once

/// @file
/// @brief Step 54 — Warp-level top-k using shuffle-based reduction.
///
/// The block-per-query kernel (Step 49) serialises candidate insertions
/// through an `atomicCAS` spinlock.  For large blocks (128+ threads) and
/// small k, the fraction of threads competing for the lock at any given
/// moment is low — but the lock still forces serial execution of all
/// insertions.
///
/// This step replaces the spinlock with a **warp-level shuffle-based**
/// approach:
///
/// 1. Each thread independently maintains a private **partial top-k** list
///    of size `k` in registers (like the Step-48 thread-per-query kernel).
/// 2. After the reference scan, a sequence of `__shfl_xor_sync` operations
///    merges all 32 partial top-k lists within the warp into a single
///    sorted top-k — with zero shared-memory usage and zero locking.
/// 3. Thread 0 of each warp writes the final top-k to global memory.
///
/// Because each thread in the warp operates on a disjoint slice of
/// references, there is no data hazard — no synchronisation is needed
/// during the scan phase.  Only the merge step uses shuffles.
///
/// ### k constraint
/// Each thread holds `k` (id, dist) pairs in registers.  For k=10 that
/// is 20 floats per thread = 640 floats per warp — well within the Volta
/// 255-register budget at `blockDim.x = 32`.  Phase 9 will extend this
/// to device-side neighbour lists that avoid the register constraint.

#include <knng/core/dataset.hpp>
#include <knng/core/graph.hpp>
#include <knng/core/types.hpp>

#include <cstddef>

namespace knng::gpu {

// ---------------------------------------------------------------------------
// CPU reference (always compiled)
// ---------------------------------------------------------------------------

/// @brief Warp-level top-k KNN on the CPU — correctness oracle.
///
/// Simulates the warp-level strategy: divides the reference set among
/// 32 "lane" slots, each lane builds a partial top-k, then all lanes
/// are merged into the final top-k.  Output must match `cpu_brute_force_knn`.
///
/// @param dataset  Input point set.
/// @param k        Neighbours per query (optimal: k ≤ 32).
/// @return         Knng sorted by distance.
[[nodiscard]] knng::Knng cpu_warp_top_k_knn(
    const knng::Dataset& dataset,
    std::size_t          k);

// ---------------------------------------------------------------------------
// GPU path
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)

/// @brief Warp-level top-k brute-force KNN (Step 54).
///
/// One warp per query; each lane scans a strided slice of the references,
/// maintains a register-resident partial top-k, then shuffle-merges with
/// other lanes.
///
/// @param dataset  Input point set.
/// @param k        Neighbours per query (k ≤ 32 for best register budget).
/// @return         Knng sorted by distance.
[[nodiscard]] knng::Knng gpu_warp_top_k_knn(
    const knng::Dataset& dataset,
    std::size_t          k);

#endif

} // namespace knng::gpu
