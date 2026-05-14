#pragma once

/// @file
/// @brief Step 91 — Distributed GPU brute-force KNN (ring-based).
///
/// Each MPI rank owns a shard of query points. Reference vectors circulate
/// through a ring: at step t, rank r holds the reference block originally
/// owned by rank (r+t) mod P, broadcasts it within the node via NCCL, then
/// computes distances on each GPU against its local query shard.
///
/// After P ring steps every rank has seen all N reference vectors and holds
/// the top-k neighbors for each of its own query points.  A final host-side
/// gather on rank 0 produces the full graph.
///
/// CPU-reference path: simulates the ring using host memory and host MPI
/// sends (no NCCL, no CUDA kernels).  Enables tests on Mac.
///
/// GPU path: real ring with intra-node ncclBcast + inter-node MPI_Sendrecv
/// for the reference block, block-per-query kernel for distances.

#if !defined(KNNG_HAVE_MPI) || !KNNG_HAVE_MPI
#  error "brute_force.hpp requires MPI — guard the include with KNNG_HAVE_MPI"
#endif

#include <knng/core/dataset.hpp>
#include <knng/core/graph.hpp>
#include <knng/dist_gpu/topology.hpp>

#include <mpi.h>

namespace knng::dist_gpu {

/// Parameters for distributed brute-force KNN.
struct DistBfConfig {
    int  k            = 10;    ///< Neighbor-list size.
    bool verbose      = false;
};

/// CPU-reference distributed brute-force (ring, host memory, MPI only).
///
/// @param dataset  Full dataset on rank 0; ignored on other ranks.
/// @param topo     Pre-built topology descriptor.
/// @param cfg      k and verbosity.
/// @param comm     MPI communicator.
/// @returns        KNN graph on rank 0; empty on other ranks.
[[nodiscard]]
knng::Knng cpu_dist_brute_force(const knng::Dataset& dataset,
                                const DistTopology&  topo,
                                const DistBfConfig&  cfg,
                                MPI_Comm             comm);

#if defined(KNNG_HAVE_CUDA)

/// Real GPU distributed brute-force (ring, block-per-query kernel).
///
/// Intra-node reference-block sharing uses ncclBcast (when NCCL present);
/// inter-node block handoff uses MPI_Sendrecv on host pinned buffers.
///
/// @param dataset  Full dataset on rank 0.
/// @param topo     Pre-built topology descriptor.
/// @param cfg      k and verbosity.
/// @param comm     MPI communicator.
/// @returns        KNN graph on rank 0.
[[nodiscard]]
knng::Knng gpu_dist_brute_force(const knng::Dataset& dataset,
                                const DistTopology&  topo,
                                const DistBfConfig&  cfg,
                                MPI_Comm             comm);

#endif // KNNG_HAVE_CUDA

} // namespace knng::dist_gpu
