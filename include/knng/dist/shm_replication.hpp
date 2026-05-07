#pragma once

/// @file
/// @brief NEO-DNND Optimisation 2 — intra-node shared-memory replication.
///
/// ## The problem
/// Steps 41–42 move feature vectors across MPI ranks, including between
/// ranks that share the same physical node. On a multi-rank-per-node
/// setup (e.g., one rank per CPU socket), feature data sent from rank 0
/// on node A to rank 1 on the same node A travels over shared memory —
/// but only if both sides know about it. The MPI point-to-point path
/// (send / receive) copies the data through the MPI library's internal
/// buffer even when the two ranks could share memory directly. This wastes
/// bandwidth and doubles the buffer footprint.
///
/// ## The fix — MPI-3 Shared Memory Windows (MPI_Win_allocate_shared)
/// MPI-3 provides `MPI_Win_allocate_shared`, which allocates a segment of
/// process-shared memory accessible by all ranks in a shared-memory
/// communicator (`MPI_COMM_TYPE_SHARED`). Ranks on the same node can
/// *directly read* each other's features without any MPI send/recv — a
/// zero-copy intra-node path.
///
/// `ShmRegion` wraps a shared-memory window:
///   * Ranks within the same node (`intra_comm`) collectively allocate a
///     window large enough for each rank's full local shard.
///   * Each rank writes its own shard into its segment of the window.
///   * Any intra-node rank can then read any other intra-node rank's
///     features via `ShmRegion::read_remote_row(rank, local_i, d)`.
///
/// Inter-node communication continues to use the normal MPI Alltoallv
/// path from Step 41. Only intra-node accesses benefit from SHM.
///
/// ## Reference
/// NEO-DNND §3.3 (Luo et al., 2021): "Intra-Node Replication —
/// Ranks on the same node replicate feature vectors into a shared
/// memory segment accessible without inter-process message passing."
///
/// ## Topology note
/// `ShmRegion::intra_size()` returns the number of ranks on this node.
/// When `intra_size() == 1`, this rank is the only one on the node and
/// SHM brings no benefit; the code compiles and runs correctly in this
/// degenerate case (the window covers only the local shard and inter-node
/// communication is used for everything).

#if !defined(KNNG_HAVE_MPI) || !KNNG_HAVE_MPI
#  error "shm_replication.hpp included without MPI — guard with KNNG_HAVE_MPI"
#endif

#include <cstddef>
#include <mpi.h>
#include <vector>

namespace knng::dist {

/// Shared-memory window over the local node's feature data.
///
/// Lifecycle:
///   1. Construct with the local feature buffer, local point count, and
///      dimensionality.
///   2. After construction, every intra-node rank's shard is immediately
///      readable by any other intra-node rank via `read_remote_row`.
///   3. Destructor calls `MPI_Win_free` and `MPI_Comm_free`.
class ShmRegion {
public:
    /// Create an MPI-3 shared-memory window covering all ranks on this node.
    ///
    /// `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` partitions `comm` into
    /// per-node subcommunicators. Within the resulting subcommunicator,
    /// `MPI_Win_allocate_shared` allocates one contiguous region per rank,
    /// all accessible from every other rank in the subcommunicator.
    ///
    /// @param local_data  Pointer to this rank's feature data. Must remain
    ///                    valid for the lifetime of `ShmRegion`.
    /// @param local_n     Number of feature vectors in `local_data`.
    /// @param d           Dimensionality of each feature vector.
    /// @param comm        Parent communicator (typically `MPI_COMM_WORLD`).
    ShmRegion(const float* local_data, std::size_t local_n,
              std::size_t d, MPI_Comm comm);

    ~ShmRegion();

    ShmRegion(const ShmRegion&)            = delete;
    ShmRegion& operator=(const ShmRegion&) = delete;
    ShmRegion(ShmRegion&&)                 = delete;
    ShmRegion& operator=(ShmRegion&&)      = delete;

    /// Number of ranks in this node's subcommunicator.
    [[nodiscard]] int intra_size() const noexcept { return intra_size_; }

    /// This rank's index within the node's subcommunicator.
    [[nodiscard]] int intra_rank() const noexcept { return intra_rank_; }

    /// Read a single feature row from a remote intra-node rank's segment.
    ///
    /// @param intra_r  Rank index within the node's subcommunicator.
    /// @param row_i    Row index within that rank's shard.
    /// @param d        Dimensionality (must match the constructor's `d`).
    /// @return         Pointer to the first float of the requested row.
    ///                 The pointer is valid for `d` contiguous floats and
    ///                 remains valid until the `ShmRegion` is destroyed.
    [[nodiscard]] const float* read_remote_row(int intra_r,
                                               std::size_t row_i,
                                               std::size_t d) const noexcept;

    /// First float of this rank's own segment in the shared window.
    /// Equivalent to `read_remote_row(intra_rank_, 0, d)`.
    [[nodiscard]] const float* local_base() const noexcept
    {
        return local_base_;
    }

    /// Return the per-rank row counts for all intra-node ranks, in
    /// subcommunicator-rank order. Index `r` is the number of feature
    /// vectors owned by intra-rank `r`.
    [[nodiscard]] const std::vector<std::size_t>&
    intra_row_counts() const noexcept
    {
        return row_counts_;
    }

private:
    MPI_Comm intra_comm_  = MPI_COMM_NULL;
    MPI_Win  win_         = MPI_WIN_NULL;
    int      intra_rank_  = 0;
    int      intra_size_  = 1;

    const float*             local_base_ = nullptr;
    std::vector<float*>      segment_ptrs_;   ///< base ptr per intra-rank
    std::vector<std::size_t> row_counts_;     ///< rows per intra-rank
};

} // namespace knng::dist
