#pragma once

/// @file
/// @brief Steps 82–85 — Multi-GPU partitioning helpers and algorithm drivers.
///
/// Always-compiled types:
///   - `MultiGpuConfig`    — partition + communicator parameters
///   - `PointShard`        — describes one GPU's row range
///   - CPU reference drivers for point-sharded and tile-sharded brute-force,
///     multi-GPU NN-Descent, and the triple-buffered pipeline.
///
/// GPU variants guarded by `KNNG_HAVE_CUDA`.

#include <knng/core/dataset.hpp>
#include <knng/core/graph.hpp>
#include <knng/gpu/nccl_comm.hpp>
#include <knng/gpu/topology.hpp>

#include <cstddef>
#include <vector>

namespace knng::gpu {

// ===========================================================================
// Shared configuration
// ===========================================================================

/// @brief Parameters controlling how work is partitioned across virtual GPUs.
struct MultiGpuConfig {
    int    n_gpus       = 1;     ///< Number of GPU devices (or virtual "ranks")
    int    k            = 10;    ///< Neighbor-list size
    float  rho          = 0.5f;  ///< NN-Descent sampling fraction
    int    n_iterations = 10;    ///< NN-Descent iteration count
    bool   verbose      = false;
};

/// @brief Half-open row range [begin, end) owned by one GPU rank.
struct PointShard {
    std::size_t begin = 0;  ///< First row index (inclusive)
    std::size_t end   = 0;  ///< Last row index (exclusive)
    int         rank  = 0;  ///< Owning GPU rank

    std::size_t size() const noexcept { return end - begin; }
};

/// @brief Partition n points evenly across cfg.n_gpus ranks.
///
/// The last rank gets any remainder rows.
std::vector<PointShard> partition_points(std::size_t n,
                                          const MultiGpuConfig& cfg);

// ===========================================================================
// Step 82 — Point-sharded brute-force (CPU reference)
// ===========================================================================

/// @brief Multi-GPU point-sharded brute-force KNN (CPU simulation).
///
/// Partitions dataset rows across `cfg.n_gpus` virtual ranks.  Each rank
/// computes distances between its shard and **all** reference vectors
/// (broadcast via `cpu_sim_bcast`), then the partial top-k results are
/// gathered and merged on rank 0.
///
/// @param dataset  Full dataset (all n × d vectors).
/// @param cfg      Partitioning + k parameters.
/// @returns        KNN graph for all n points.
knng::Knng cpu_multi_brute_force_point(const knng::Dataset& dataset,
                                        const MultiGpuConfig& cfg);

// ===========================================================================
// Step 83 — Tile-sharded brute-force (CPU reference)
// ===========================================================================

/// @brief Multi-GPU tile-sharded brute-force KNN (CPU simulation).
///
/// 2-D tile partitioning: both queries and references are split into tiles.
/// Each rank owns one tile of the (query, reference) distance matrix.
/// AllToAll of partial top-k results followed by rank-local merge.
///
/// @param dataset  Full dataset.
/// @param cfg      Partitioning + k parameters.
/// @returns        KNN graph for all n points.
knng::Knng cpu_multi_brute_force_tile(const knng::Dataset& dataset,
                                       const MultiGpuConfig& cfg);

// ===========================================================================
// Step 85 — Multi-GPU NN-Descent (CPU reference)
// ===========================================================================

/// @brief Multi-GPU NN-Descent (CPU simulation).
///
/// Graph partitioned by point ownership (each rank owns n/P points).
/// Neighbor-of-neighbor lookups that cross partition boundaries are resolved
/// via a simulated `allgather` of the full graph state at each iteration.
///
/// @param dataset  Full dataset.
/// @param cfg      Partitioning + algorithm parameters.
/// @param seed     RNG seed for random graph initialisation.
/// @returns        KNN graph for all n points.
knng::Knng cpu_multi_nn_descent(const knng::Dataset& dataset,
                                  const MultiGpuConfig& cfg,
                                  std::uint64_t seed = 42u);

} // namespace knng::gpu
