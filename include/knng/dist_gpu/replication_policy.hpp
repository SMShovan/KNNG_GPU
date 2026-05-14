#pragma once

/// @file
/// @brief Steps 93–95 — NEO-DNND replication policies.
///
/// Each policy reduces the number of remote feature-vector bytes sent per
/// NN-Descent iteration.  Policies are composable: dedup reduces unique IDs,
/// intra-node SHM avoids cross-node traffic, bandwidth-aware replication
/// proactively caches frequently requested IDs.
///
/// CPU-reference implementations operate on host vectors.
/// GPU implementations use device bitsets + CUB prefix-sums.

#if !defined(KNNG_HAVE_MPI) || !KNNG_HAVE_MPI
#  error "replication_policy.hpp requires MPI"
#endif

#include <cstddef>
#include <cstdint>
#include <mpi.h>
#include <vector>
#include <knng/core/types.hpp>

namespace knng::dist_gpu {

// ===========================================================================
// Step 93 — GPU-side bitset dedup
// ===========================================================================

/// Statistics from one deduplication pass.
struct GpuDedupStats {
    std::size_t raw_count;
    std::size_t dedup_count;

    [[nodiscard]] double reduction() const noexcept {
        return raw_count == 0 ? 0.0
            : static_cast<double>(raw_count - dedup_count) /
              static_cast<double>(raw_count);
    }
};

/// Deduplicate a list of global point IDs using a host-side bitset.
///
/// CPU-reference path: sorts and removes duplicates in place.
/// GPU path: uses a device bitset + CUB ExclusiveSum to compact.
///
/// @param requests  Flat list of global IDs to deduplicate. Modified in place.
/// @param global_n  Total number of points (bitset width).
/// @returns         Statistics (raw vs deduplicated count).
[[nodiscard]]
GpuDedupStats cpu_dedup_requests(std::vector<knng::index_t>& requests,
                                 std::size_t                 global_n);

#if defined(KNNG_HAVE_CUDA)
/// GPU version: returns deduplicated device buffer.
/// @param d_requests Device pointer to raw request list (may be unsorted).
/// @param n_raw      Number of raw entries.
/// @param global_n   Bitset width.
/// @param d_out      Output device buffer for unique IDs (must be >= n_raw).
/// @returns          Number of unique IDs written to d_out + stats.
std::pair<std::size_t, GpuDedupStats>
gpu_dedup_requests(const knng::index_t* d_requests,
                   std::size_t          n_raw,
                   std::size_t          global_n,
                   knng::index_t*       d_out);
#endif

// ===========================================================================
// Step 95 — Bandwidth-aware proactive replication policy
// ===========================================================================

/// Policy that proactively replicates a point's feature vector to all ranks
/// that have requested it at least `threshold` times across recent iterations.
///
/// CPU-reference: tracks request frequency in a host map.
struct BandwidthAwarePolicy {
    int threshold = 2; ///< Minimum request count before replication.

    /// Update frequency counters from this iteration's request set.
    /// Returns the set of IDs that should be proactively replicated.
    [[nodiscard]]
    std::vector<knng::index_t>
    update(const std::vector<knng::index_t>& requests, std::size_t global_n);

    /// Reset all frequency counters (call between NND runs).
    void reset() noexcept { freq_.assign(freq_.size(), 0); }

private:
    std::vector<int> freq_;
};

} // namespace knng::dist_gpu
