#pragma once

/// @file
/// @brief Steps 92–96 — Distributed GPU NN-Descent (NEO-DNND family).
///
/// Baseline (Step 92): gather-scatter every iteration.
///   Each rank owns n/P points. At every iteration:
///     1. Each GPU runs local_join on its shard (identifies new candidates).
///     2. Remote neighbor features needed for cross-partition joins are
///        fetched via MPI AllToAllv (one request per remote ID).
///     3. After fetching, each rank updates its local graph.
///     4. MPI AllReduce sums update counts for convergence.
///
/// Optimisations (Steps 93–96) layer on top of the same structure.
///
/// CPU-reference path: host-only, simulates gather-scatter with plain MPI.

#if !defined(KNNG_HAVE_MPI) || !KNNG_HAVE_MPI
#  error "nn_descent.hpp requires MPI — guard with KNNG_HAVE_MPI"
#endif

#include <knng/core/dataset.hpp>
#include <knng/core/graph.hpp>
#include <knng/dist_gpu/topology.hpp>

#include <cstdint>
#include <mpi.h>

namespace knng::dist_gpu {

/// Configuration shared by all distributed NND variants (Steps 92–96).
struct DistNndConfig {
    int      k            = 10;
    float    rho          = 0.5f;    ///< Sampling fraction for candidate pairs.
    int      n_iterations = 20;
    double   delta        = 0.001;   ///< Convergence: stop when updates < delta*n*k.
    bool     use_reverse  = true;    ///< Include reverse-neighbor joins.
    bool     verbose      = false;
    std::uint64_t seed    = 42u;
};

/// CPU-reference distributed NN-Descent baseline (gather-scatter every iter).
///
/// @param dataset  Full dataset on rank 0; ignored on other ranks.
/// @param topo     Pre-built topology descriptor.
/// @param cfg      Algorithm parameters.
/// @param comm     MPI communicator.
/// @returns        KNN graph on rank 0; empty on other ranks.
[[nodiscard]]
knng::Knng cpu_dist_nn_descent(const knng::Dataset& dataset,
                               const DistTopology&  topo,
                               const DistNndConfig& cfg,
                               MPI_Comm             comm);

#if defined(KNNG_HAVE_CUDA)

/// Real GPU distributed NN-Descent baseline (Step 92).
///
/// Each rank runs local_join on GPU; remote features fetched via MPI.
///
/// @param dataset  Full dataset on rank 0.
/// @param topo     Pre-built topology descriptor.
/// @param cfg      Algorithm parameters.
/// @param comm     MPI communicator.
/// @returns        KNN graph on rank 0.
[[nodiscard]]
knng::Knng gpu_dist_nn_descent(const knng::Dataset& dataset,
                               const DistTopology&  topo,
                               const DistNndConfig& cfg,
                               MPI_Comm             comm);

/// GPU distributed NN-Descent with GPU-side dedup (Step 93 — NEO opt 1).
[[nodiscard]]
knng::Knng gpu_dist_nn_descent_dedup(const knng::Dataset& dataset,
                                     const DistTopology&  topo,
                                     const DistNndConfig& cfg,
                                     MPI_Comm             comm);

/// GPU distributed NN-Descent with intra-node NCCL replication (Step 94 — NEO opt 2).
[[nodiscard]]
knng::Knng gpu_dist_nn_descent_shm(const knng::Dataset& dataset,
                                   const DistTopology&  topo,
                                   const DistNndConfig& cfg,
                                   MPI_Comm             comm);

/// GPU distributed NN-Descent with bandwidth-aware proactive replication
/// (Step 95 — NEO opt 3).
[[nodiscard]]
knng::Knng gpu_dist_nn_descent_bw(const knng::Dataset& dataset,
                                  const DistTopology&  topo,
                                  const DistNndConfig& cfg,
                                  MPI_Comm             comm,
                                  int                  replication_threshold = 2);

/// GPU distributed NN-Descent with overlapped comm+compute (Step 96).
[[nodiscard]]
knng::Knng gpu_dist_nn_descent_overlap(const knng::Dataset& dataset,
                                       const DistTopology&  topo,
                                       const DistNndConfig& cfg,
                                       MPI_Comm             comm);

#endif // KNNG_HAVE_CUDA

} // namespace knng::dist_gpu
