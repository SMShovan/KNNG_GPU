#pragma once

/// @file
/// @brief Step 97 — Distributed GPU NN-Descent benchmark driver declaration.

#if !defined(KNNG_HAVE_MPI) || !KNNG_HAVE_MPI
#  error "benchmark.hpp requires MPI — guard with KNNG_HAVE_MPI"
#endif

#include <knng/core/dataset.hpp>

#include <mpi.h>

namespace knng::dist_gpu {

/// Run the Phase 12 benchmark suite and print a result table on rank 0.
///
/// Times cpu_dist_brute_force, cpu_dist_nn_descent, and (when CUDA is
/// present) gpu_dist_nn_descent, gpu_dist_nn_descent_dedup, and
/// gpu_dist_nn_descent_overlap.  Recall@k is computed against the
/// exact brute-force reference.  Results include a comparison to the
/// published NEO-DNND baseline (Luo et al. 2021).
///
/// @param dataset  Full dataset on rank 0; ignored on other ranks.
/// @param k        Neighbor count.
/// @param n_iters  NN-Descent iteration cap.
/// @param comm     MPI communicator (typically MPI_COMM_WORLD).
void run_dist_benchmark(const knng::Dataset& dataset,
                        int                  k,
                        int                  n_iters,
                        MPI_Comm             comm);

} // namespace knng::dist_gpu
