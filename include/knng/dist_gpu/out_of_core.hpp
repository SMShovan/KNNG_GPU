#pragma once

/// @file
/// @brief Step 99 — Billion-scale out-of-core streaming for distributed GPU NND.
///
/// For datasets that exceed GPU VRAM (e.g., 1B × 128 float32 ≈ 512 GB),
/// each rank streams its shard through GPU memory in chunks.  The triple-
/// buffer pipeline from Phase 11 (Step 84) is reused to overlap H2D load,
/// GPU local_join, and D2H graph writeback.
///
/// CPU-reference path is always compiled; GPU path gated on KNNG_HAVE_CUDA.

#if !defined(KNNG_HAVE_MPI) || !KNNG_HAVE_MPI
#  error "out_of_core.hpp requires MPI — guard with KNNG_HAVE_MPI"
#endif

#include <knng/core/dataset.hpp>
#include <knng/core/graph.hpp>
#include <knng/dist_gpu/topology.hpp>
#include <knng/dist_gpu/nn_descent.hpp>

#include <cstddef>
#include <mpi.h>

namespace knng::dist_gpu {

/// Configuration for out-of-core streaming NND.
struct OutOfCoreConfig {
    DistNndConfig nnd;               ///< Base NND parameters (k, rho, iters, delta).
    std::size_t   chunk_size = 0;    ///< Points per GPU-memory chunk (0 = all, i.e., in-core).
    bool          verbose    = false;
};

/// CPU-reference distributed out-of-core NN-Descent.
///
/// Simulates billion-scale streaming by slicing each rank's shard into
/// `cfg.chunk_size`-point windows and processing them through a
/// CpuTripleBuffer pipeline.  Results are merged across chunks and the
/// AllGather between iterations uses only delta-compressed updates
/// (changed graph entries only), not the full graph.
///
/// @param dataset  Full dataset on rank 0; scattered to all ranks internally.
/// @param topo     Pre-built topology descriptor.
/// @param cfg      Algorithm + streaming parameters.
/// @param comm     MPI communicator.
/// @returns        KNN graph on rank 0; empty on other ranks.
[[nodiscard]]
knng::Knng cpu_dist_streaming_nn_descent(const knng::Dataset& dataset,
                                         const DistTopology&  topo,
                                         const OutOfCoreConfig& cfg,
                                         MPI_Comm             comm);

#if defined(KNNG_HAVE_CUDA)

/// GPU out-of-core streaming NN-Descent using GpuTriplePipeline.
///
/// Three CUDA streams overlap H2D chunk load, GPU local_join kernel,
/// and D2H graph writeback.  Reduces peak GPU memory from
/// O(N_per_rank × d) to O(chunk_size × d).  Delta-compressed AllGather
/// reduces inter-node traffic to O(Δ × (d + k)) per iteration, where
/// Δ ≪ N is the count of changed graph rows.
///
/// @param dataset  Full dataset on rank 0.
/// @param topo     Pre-built topology descriptor.
/// @param cfg      Algorithm + streaming parameters.
/// @param comm     MPI communicator.
/// @returns        KNN graph on rank 0.
[[nodiscard]]
knng::Knng gpu_dist_streaming_nn_descent(const knng::Dataset&   dataset,
                                         const DistTopology&    topo,
                                         const OutOfCoreConfig& cfg,
                                         MPI_Comm               comm);

#endif // KNNG_HAVE_CUDA

} // namespace knng::dist_gpu
