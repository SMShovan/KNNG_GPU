#pragma once

/// @file
/// @brief Step 80 — NCCL integration + CPU simulation of collective operations.
///
/// Two layers:
///   1. `knng::gpu::CpuSimCollective` — always compiled; models the math of
///      each collective on plain CPU vectors so Phase 11 algorithms can be
///      unit-tested on Mac without CUDA hardware.
///   2. `knng::gpu::NcclComm` — compiled only when both CUDA and NCCL are
///      available; wraps `ncclComm_t` with RAII and the same operation names.

#include <cstddef>
#include <vector>

#if defined(KNNG_HAVE_CUDA) && defined(KNNG_HAVE_NCCL)
#include <nccl.h>
#include <cuda_runtime.h>
#endif

namespace knng::gpu {

// ===========================================================================
// CPU simulation of collective operations (always compiled)
// ===========================================================================

/// @brief CPU simulation of NCCL-style collectives over n_ranks flat float
/// buffers.
///
/// Each element of `per_rank` represents one GPU's local buffer.  All
/// operations are deterministic and single-threaded (no concurrency needed
/// for correctness tests).  Buffer sizes must match `count` for every rank;
/// undefined behaviour if they differ.
struct CpuSimCollective {
    /// @brief In-place allreduce sum: every rank's buffer becomes the
    /// element-wise sum across all ranks.
    ///
    /// @param per_rank  Vector of n_ranks float buffers, each of length count.
    /// @param count     Number of float elements per rank.
    static void allreduce_sum(std::vector<std::vector<float>>& per_rank,
                               std::size_t count);

    /// @brief Broadcast: copy per_rank[root] to every other rank.
    ///
    /// @param per_rank  Vector of n_ranks float buffers, each of length count.
    /// @param root      Source rank (0-based).
    /// @param count     Number of float elements to broadcast.
    static void bcast(std::vector<std::vector<float>>& per_rank,
                       int root, std::size_t count);

    /// @brief Allgather: result is the concatenation of all per_rank buffers.
    ///
    /// @param per_rank        Input buffers, each of length count_per_rank.
    /// @param result          Output buffer sized n_ranks * count_per_rank.
    /// @param count_per_rank  Elements contributed by each rank.
    static void allgather(const std::vector<std::vector<float>>& per_rank,
                           std::vector<float>& result,
                           std::size_t count_per_rank);

    /// @brief Reduce-scatter: element-wise sum across all ranks, then
    /// distribute chunk r (size count/n_ranks) to rank r.
    ///
    /// Requires count % n_ranks == 0.
    ///
    /// @param per_rank  In/out vector of n_ranks buffers, each of length count.
    /// @param count     Total element count (divisible by n_ranks).
    static void reduce_scatter(std::vector<std::vector<float>>& per_rank,
                                std::size_t count);
};

// ===========================================================================
// Real NCCL communicator (CUDA + NCCL only)
// ===========================================================================

#if defined(KNNG_HAVE_CUDA) && defined(KNNG_HAVE_NCCL)

/// @brief RAII wrapper around `ncclComm_t`.
///
/// Typical use: call `NcclComm::init_all(n_gpus)` to get one communicator
/// per device, then pass each to the corresponding GPU worker.
class NcclComm {
public:
    NcclComm() = default;
    ~NcclComm();

    NcclComm(const NcclComm&)            = delete;
    NcclComm& operator=(const NcclComm&) = delete;
    NcclComm(NcclComm&& o) noexcept;
    NcclComm& operator=(NcclComm&& o) noexcept;

    /// @brief Create communicators for all `n_gpus` devices on this node.
    /// @returns Vector of n_gpus NcclComm objects (index = device id).
    static std::vector<NcclComm> init_all(int n_gpus);

    /// @brief Allreduce sum of count floats on device buffer d_buf.
    void allreduce_sum(float* d_buf, std::size_t count, cudaStream_t stream = nullptr);

    /// @brief Broadcast count floats from root's d_buf to all ranks.
    void bcast(float* d_buf, std::size_t count, int root,
               cudaStream_t stream = nullptr);

    /// @brief Allgather: each rank contributes count floats; d_recvbuf must
    /// be n_ranks * count elements on every rank.
    void allgather(const float* d_sendbuf, float* d_recvbuf,
                   std::size_t count, cudaStream_t stream = nullptr);

    int  rank()    const { return rank_; }
    int  n_ranks() const { return n_ranks_; }
    bool valid()   const { return comm_ != nullptr; }

private:
    ncclComm_t comm_    = nullptr;
    int        rank_    = 0;
    int        n_ranks_ = 0;
};

#endif // KNNG_HAVE_CUDA && KNNG_HAVE_NCCL

} // namespace knng::gpu
