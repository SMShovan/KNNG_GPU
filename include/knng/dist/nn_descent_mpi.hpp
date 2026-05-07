#pragma once

/// @file
/// @brief Distributed NN-Descent over MPI-sharded datasets.
///
/// Phase 6 builds distributed NN-Descent in two layers:
///
///   **Step 41 (this file's first version) — Gather-scatter baseline.**
///   Neighbor lists live on the owning rank. In each iteration, every rank
///   publishes a minimal set of feature vectors (the ones its neighbors and
///   reverse-neighbors reference) via `MPI_Alltoallv`; every rank can then
///   compute all needed distances locally. Deliberately inefficient:
///   a point's feature vector may be sent to every rank in the worst case,
///   and no deduplication is performed across ranks. The correctness floor.
///
///   **Step 42 — Duplicate-request reduction.**
///   Before sending, each rank deduplicates the set of remote IDs it needs
///   (many will already be cached locally). Measures bytes-sent reduction.
///
/// ## Ownership model
/// Point `p` is *owned* by the rank whose shard contains `p`'s global
/// index. A rank owns its neighbor lists; it does *not* cache feature
/// vectors for points it does not own.
///
/// ## Per-iteration protocol (gather-scatter)
/// 1. Each rank builds the set of *remote* global IDs referenced in its
///    local neighbor lists and reverse-neighbor lists.
/// 2. Ranks exchange those IDs via `MPI_Alltoall` (send count per peer,
///    then `MPI_Alltoallv` for the actual IDs).
/// 3. Each rank responds by sending back the requested feature vectors.
/// 4. All ranks now have every needed feature vector locally; the local
///    join runs exactly as in the CPU NN-Descent.
/// 5. Convergence: each rank counts local updates; a global `MPI_Allreduce`
///    sums them; the driver stops when the fraction < `delta`.

#if !defined(KNNG_HAVE_MPI) || !KNNG_HAVE_MPI
#  error "nn_descent_mpi.hpp included without MPI — guard with KNNG_HAVE_MPI"
#endif

#include <cstddef>
#include <cstdint>
#include <mpi.h>
#include <vector>

#include "knng/core/graph.hpp"
#include "knng/cpu/nn_descent.hpp"
#include "knng/dist/sharded_dataset.hpp"

namespace knng::dist {

/// Configuration knobs for the distributed NN-Descent driver.
struct NnDescentMpiConfig {
    /// Hard cap on iterations.
    std::size_t max_iters  = 50;

    /// Convergence threshold: fraction of `(global_n * k)` updates
    /// below which the graph is declared stable.
    double      delta      = 0.001;

    /// RNG seed for the random graph initialisation on each rank.
    /// Each rank mixes this seed with its rank so initialisation
    /// differs across ranks.
    std::uint64_t seed     = 42;

    /// Use reverse neighbor lists in the local join (recommended ON).
    bool          use_reverse = true;

    /// Sampling rate for the local join candidate set (1.0 = no sampling).
    double        rho      = 1.0;
};

/// Per-iteration statistics collected across all ranks.
struct NnDescentMpiLog {
    std::size_t iteration;
    std::size_t global_updates;
    double      update_fraction; ///< `global_updates / (global_n * k)`.
    std::size_t bytes_sent;      ///< Feature-vector bytes sent this iteration.
};

/// Build an approximate KNN graph on a sharded dataset using
/// gather-scatter NN-Descent.
///
/// **Correctness guarantee:** with a single rank (`shard.size() == 1`)
/// and identical seeds the output is numerically equivalent to
/// `knng::cpu::nn_descent` on the same dataset and config.
///
/// @param shard   Calling rank's slice of the global dataset.
/// @param k       Neighbors per point.
/// @param cfg     Tuning knobs.
/// @param comm    Communicator (typically `MPI_COMM_WORLD`).
/// @return        Local `Knng` of shape `(shard.local_n(), k)` with
///                *global* neighbor indices.
[[nodiscard]] Knng nn_descent_mpi(const ShardedDataset&     shard,
                                  std::size_t               k,
                                  const NnDescentMpiConfig& cfg,
                                  MPI_Comm                  comm);

/// Same as `nn_descent_mpi` but appends a per-iteration log entry to
/// `log_out`. Useful for convergence-curve analysis and bytes-sent
/// profiling.
[[nodiscard]] Knng nn_descent_mpi_with_log(
    const ShardedDataset&       shard,
    std::size_t                 k,
    const NnDescentMpiConfig&   cfg,
    MPI_Comm                    comm,
    std::vector<NnDescentMpiLog>& log_out);

} // namespace knng::dist
