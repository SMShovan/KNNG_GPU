#pragma once

/// @file
/// @brief Point-sharded dataset view for distributed-memory computation.
///
/// Each MPI rank in Phase 6+ owns a contiguous slice of the global
/// dataset — `n/P` rows (with any remainder assigned to the last rank).
/// `ShardedDataset` wraps this slice together with the global metadata
/// so algorithms can reason about both local and global indices without
/// passing separate bookkeeping alongside the `Dataset`.
///
/// **Sharding convention:**
/// - Rank `r` owns rows `[r * base, r * base + local_n)` where
///   `base = global_n / size` and `local_n = base + (r == size-1 ? remainder : 0)`.
/// - The last rank absorbs any remainder rows (`global_n % size`).
/// - This static balanced partitioning is sufficient for Phase 6.
///   Phase 12 will introduce work-stealing rebalancing when GPU memory
///   constraints create non-uniform shard sizes.
///
/// **Communication pattern:**
/// The distributed brute-force (Step 40) and NN-Descent (Step 41) do
/// *not* AllGather the entire dataset. Instead, the reference columns
/// cycle through ranks via a ring; each rank broadcasts its shard to its
/// left neighbour one tile at a time. `ShardedDataset::local_dataset()`
/// is the primitive these algorithms operate on.

#if !defined(KNNG_HAVE_MPI) || !KNNG_HAVE_MPI
#  error "sharded_dataset.hpp included without MPI — guard with KNNG_HAVE_MPI"
#endif

#include <cstddef>
#include <mpi.h>
#include <vector>

#include "knng/core/dataset.hpp"

namespace knng::dist {

/// Distributed view of a feature-vector dataset.
///
/// Owns the local shard as a `Dataset` value. Constructed by the
/// `scatter` factory or by directly providing a pre-populated
/// `Dataset` slice (useful in tests or when reading from a sharded
/// file).
class ShardedDataset {
public:
    /// Construct from an already-populated local shard.
    ///
    /// @param local_data   Local slice of the global dataset.
    ///                     Must satisfy `local_data.is_contiguous()`.
    /// @param global_n     Total number of points across all ranks.
    /// @param local_start  Global index of the first row in `local_data`.
    /// @param rank         MPI rank of this process.
    /// @param size         Total number of MPI ranks.
    ShardedDataset(Dataset local_data,
                   std::size_t global_n,
                   std::size_t local_start,
                   int rank,
                   int size);

    /// Scatter a root dataset across all ranks in `comm`.
    ///
    /// Rank `root_rank` holds the full dataset in `root_data`; other
    /// ranks may pass a default-constructed `Dataset{}`. On return,
    /// every rank owns its shard.
    ///
    /// The sharding is statically balanced: each rank gets
    /// `global_n / size` rows; the last rank takes the remainder.
    ///
    /// @param root_data  Full dataset on `root_rank`; ignored on others.
    /// @param root_rank  Rank that holds the data to scatter (default 0).
    /// @param comm       Communicator to scatter over.
    /// @return           Populated `ShardedDataset` on every rank.
    [[nodiscard]] static ShardedDataset scatter(const Dataset& root_data,
                                                int root_rank,
                                                MPI_Comm comm);

    /// Gather all shards to `root_rank` and return the full dataset.
    ///
    /// Only the return value on `root_rank` is meaningful; other ranks
    /// receive an empty `Dataset`. Useful for correctness checks and
    /// end-to-end benchmarks.
    ///
    /// @param root_rank Rank that should receive the full dataset.
    /// @param comm      Communicator to gather over.
    [[nodiscard]] Dataset gather(int root_rank, MPI_Comm comm) const;

    /// Read-only access to the locally-owned shard.
    [[nodiscard]] const Dataset& local_dataset() const noexcept
    {
        return local_;
    }

    /// Number of rows owned by this rank.
    [[nodiscard]] std::size_t local_n() const noexcept { return local_.n; }

    /// Total number of rows in the global dataset.
    [[nodiscard]] std::size_t global_n() const noexcept { return global_n_; }

    /// Global index of the first locally-owned row.
    [[nodiscard]] std::size_t local_start() const noexcept
    {
        return local_start_;
    }

    /// Global index one past the last locally-owned row.
    [[nodiscard]] std::size_t local_end() const noexcept
    {
        return local_start_ + local_.n;
    }

    /// Dimensionality of every feature vector.
    [[nodiscard]] std::size_t d() const noexcept { return local_.d; }

    /// Convert a local row index to a global point index.
    [[nodiscard]] std::size_t global_index(std::size_t local_i) const noexcept
    {
        return local_start_ + local_i;
    }

    /// MPI rank of this process.
    [[nodiscard]] int rank() const noexcept { return rank_; }

    /// Total number of MPI ranks.
    [[nodiscard]] int size() const noexcept { return size_; }

private:
    Dataset     local_;
    std::size_t global_n_    = 0;
    std::size_t local_start_ = 0;
    int         rank_        = 0;
    int         size_        = 1;
};

/// Compute the shard bounds for rank `r` given `global_n` and `num_ranks`.
///
/// Returns `{start, count}` where `start` is the first global index
/// owned by rank `r` and `count` is the number of rows it owns.
/// The last rank absorbs any remainder.
struct ShardBounds {
    std::size_t start;
    std::size_t count;
};

[[nodiscard]] inline ShardBounds compute_shard(std::size_t global_n,
                                               int num_ranks,
                                               int rank) noexcept
{
    const std::size_t base      = global_n / static_cast<std::size_t>(num_ranks);
    const std::size_t remainder = global_n % static_cast<std::size_t>(num_ranks);
    const std::size_t start     = static_cast<std::size_t>(rank) * base;
    const std::size_t count     =
        base + (rank == num_ranks - 1 ? remainder : std::size_t{0});
    return {start, count};
}

} // namespace knng::dist
