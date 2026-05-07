#pragma once

/// @file
/// @brief Distributed brute-force exact KNN via MPI ring communication.
///
/// **Algorithm (ring shift):**
///
/// Each rank owns `n/P` query rows. The reference data comprises all
/// `n` rows. Rather than AllGathering all `n` rows to every rank (cost:
/// `O(n·d)` bytes per rank, infeasible at scale), the reference columns
/// cycle through ranks using a unidirectional ring:
///
/// ```
///   ring_buf = local_shard          // every rank starts with its shard
///   for step in 0 .. P-1:
///       compute distances(local_queries, ring_buf)
///       update local top-k heaps
///       ring_buf = MPI_Sendrecv(ring_buf → right, ← left)
/// ```
///
/// After `P` steps (back to the starting shard), every rank has seen all
/// `n` reference rows in `P` chunks of size `n/P` and holds the exact
/// top-k for its `n/P` local queries. Peak working memory per rank:
/// two reference buffers (`ring_buf` + the recv staging buffer) of
/// size `≤ (n/P + remainder) × d` floats — `O(n/P)`, independent of `P`.
///
/// **Communication cost:** `P - 1` rounds of `MPI_Sendrecv`, each
/// transferring `≤ (n/P + remainder) × d × 4` bytes. Total bytes
/// received per rank: `n × d × 4` — identical to AllGather, but the
/// peak *in-flight* buffer is `O(n/P)` rather than `O(n)`.
///
/// **Correctness:** self-matches are excluded by comparing the
/// *global* index of the current reference row against the global
/// index of the current query row.
///
/// **Output:** `brute_force_knn_mpi` returns the local `Knng` (rows
/// for the shard's queries). Indices in the output are *global* point
/// indices, consistent with the global `Dataset`'s row numbering.
/// A caller that wants the full global graph must call
/// `ShardedDataset::gather` on the graph (or assemble from shards).

#if !defined(KNNG_HAVE_MPI) || !KNNG_HAVE_MPI
#  error "brute_force_mpi.hpp included without MPI — guard with KNNG_HAVE_MPI"
#endif

#include <cstddef>
#include <mpi.h>

#include "knng/core/graph.hpp"
#include "knng/dist/sharded_dataset.hpp"

namespace knng::dist {

/// Distributed brute-force KNN on a point-sharded dataset.
///
/// @param shard   The calling rank's shard of the global dataset.
/// @param k       Number of nearest neighbours per point.
/// @param comm    Communicator (typically `MPI_COMM_WORLD`).
/// @return        `Knng` of shape `(shard.local_n(), k)` where
///                neighbor indices are *global* point indices.
/// @throws std::invalid_argument on malformed inputs.
[[nodiscard]] Knng brute_force_knn_mpi(const ShardedDataset& shard,
                                       std::size_t           k,
                                       MPI_Comm              comm);

/// Gather the distributed graph from all ranks to `root_rank`.
///
/// On `root_rank`, returns a `Knng` of shape `(global_n, k)` where
/// each row `i` holds the k-nearest neighbors of global point `i`.
/// On non-root ranks returns an empty `Knng`. Useful for correctness
/// checks and end-to-end benchmark reporting.
///
/// @param local_graph  Shard-local graph from `brute_force_knn_mpi`.
/// @param shard        The shard that produced `local_graph`.
/// @param root_rank    Rank that assembles the full result.
/// @param comm         Communicator.
[[nodiscard]] Knng gather_graph(const Knng&           local_graph,
                                const ShardedDataset& shard,
                                int                   root_rank,
                                MPI_Comm              comm);

} // namespace knng::dist
