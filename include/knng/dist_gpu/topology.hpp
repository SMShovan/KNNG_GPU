#pragma once

/// @file
/// @brief Step 90 — Hierarchical topology: ranks → nodes → GPUs.
///
/// `DistTopology` probes the MPI communicator to discover:
///   - Which ranks share the same physical node (same MPI_Get_processor_name).
///   - The intra-node GPU-to-GPU NVLink connectivity (from Topology::probe).
///   - Categorises each rank pair as intra-node (use NCCL) or
///     inter-node (use MPI).
///
/// This information drives algorithm choices in Steps 91–96:
///   - Within a node: use ncclBcast / ncclAllGather over NVLink.
///   - Across nodes: use MPI_Isend/Irecv over the network fabric.

#if !defined(KNNG_HAVE_MPI) || !KNNG_HAVE_MPI
#  error "topology.hpp requires MPI — guard the include with KNNG_HAVE_MPI"
#endif

#include <mpi.h>
#include <cstddef>
#include <string>
#include <vector>

namespace knng::dist_gpu {

/// Per-rank topology record.
struct RankInfo {
    int         rank      = 0;  ///< MPI rank.
    int         node_id   = -1; ///< Node index (same node ↔ same node_id).
    int         local_gpu = -1; ///< GPU index on the node (-1 if no GPU).
    std::string hostname;       ///< MPI_Get_processor_name result.
};

/// Hierarchical topology descriptor for a distributed run.
///
/// Constructed once per process via `DistTopology::build()` and then shared
/// read-only throughout the lifetime of the distributed algorithms.
class DistTopology {
public:
    /// Probe the communicator and return a fully populated topology.
    ///
    /// Collective: all ranks in `comm` must call this simultaneously.
    /// Each rank's GPU index is taken as `rank % gpus_per_node` when
    /// `gpus_per_node > 0`; set to -1 otherwise (CPU-only run).
    ///
    /// @param comm          MPI communicator (typically MPI_COMM_WORLD).
    /// @param gpus_per_node Number of GPUs per node (0 = CPU-only).
    [[nodiscard]] static DistTopology build(MPI_Comm comm,
                                            int gpus_per_node = 0);

    /// Total number of MPI ranks in the communicator.
    [[nodiscard]] int size()      const noexcept { return static_cast<int>(ranks_.size()); }

    /// Rank of the calling process.
    [[nodiscard]] int my_rank()   const noexcept { return my_rank_; }

    /// Number of distinct nodes in the communicator.
    [[nodiscard]] int num_nodes() const noexcept { return num_nodes_; }

    /// Node index of the calling process.
    [[nodiscard]] int my_node()   const noexcept { return ranks_[static_cast<std::size_t>(my_rank_)].node_id; }

    /// All ranks that share the same node as `rank`.
    [[nodiscard]] const std::vector<int>& intra_node_ranks(int rank) const;

    /// True when `a` and `b` reside on the same node.
    [[nodiscard]] bool same_node(int a, int b) const noexcept
    {
        return ranks_[static_cast<std::size_t>(a)].node_id ==
               ranks_[static_cast<std::size_t>(b)].node_id;
    }

    /// True when `a` and `b` are on different nodes.
    [[nodiscard]] bool inter_node(int a, int b) const noexcept
    {
        return !same_node(a, b);
    }

    /// GPU index assigned to `rank` on its node (-1 if CPU-only).
    [[nodiscard]] int gpu_for_rank(int rank) const noexcept
    {
        return ranks_[static_cast<std::size_t>(rank)].local_gpu;
    }

    /// Hostname of `rank`.
    [[nodiscard]] const std::string& hostname(int rank) const
    {
        return ranks_[static_cast<std::size_t>(rank)].hostname;
    }

    /// All per-rank info (indexed by rank).
    [[nodiscard]] const std::vector<RankInfo>& ranks() const noexcept
    {
        return ranks_;
    }

    /// Human-readable summary (rank 0 only — others get an empty string).
    [[nodiscard]] std::string to_string() const;

private:
    std::vector<RankInfo>              ranks_;
    std::vector<std::vector<int>>      node_peers_;  ///< node_id → [ranks]
    int                                my_rank_   = 0;
    int                                num_nodes_ = 0;
};

} // namespace knng::dist_gpu
