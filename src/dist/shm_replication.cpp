/// @file
/// @brief NEO-DNND Optimisation 2 — MPI-3 shared-memory replication.

#include "knng/dist/shm_replication.hpp"

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace knng::dist {

ShmRegion::ShmRegion(const float* local_data,
                     std::size_t  local_n,
                     std::size_t  d,
                     MPI_Comm     comm)
{
    // Split `comm` into per-node subcommunicators using the standard
    // MPI-3 `MPI_COMM_TYPE_SHARED` key.
    if (MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0,
                            MPI_INFO_NULL, &intra_comm_) != MPI_SUCCESS) {
        throw std::runtime_error{
            "knng::dist::ShmRegion: MPI_Comm_split_type failed"};
    }

    MPI_Comm_rank(intra_comm_, &intra_rank_);
    MPI_Comm_size(intra_comm_, &intra_size_);

    // Allocate shared window. Each rank contributes `local_n * d * sizeof(float)`
    // bytes; MPI stitches them into one contiguous addressable region.
    const MPI_Aint my_bytes =
        static_cast<MPI_Aint>(local_n * d * sizeof(float));

    void* win_base_ptr = nullptr;
    if (MPI_Win_allocate_shared(my_bytes,
                                static_cast<int>(sizeof(float)),
                                MPI_INFO_NULL,
                                intra_comm_,
                                &win_base_ptr,
                                &win_) != MPI_SUCCESS) {
        MPI_Comm_free(&intra_comm_);
        throw std::runtime_error{
            "knng::dist::ShmRegion: MPI_Win_allocate_shared failed"};
    }

    local_base_ = static_cast<float*>(win_base_ptr);

    // Copy this rank's feature data into its segment of the shared window.
    if (local_n > 0 && local_data != nullptr) {
        std::memcpy(local_base_, local_data,
                    local_n * d * sizeof(float));
    }

    // Fence to ensure all ranks have finished writing before any reads.
    MPI_Win_fence(0, win_);

    // Gather segment base pointers and row counts from every intra-node rank.
    segment_ptrs_.resize(static_cast<std::size_t>(intra_size_));
    row_counts_.resize(static_cast<std::size_t>(intra_size_));

    for (int r = 0; r < intra_size_; ++r) {
        MPI_Aint   r_size   = 0;
        int        r_disp   = 0;
        void*      r_base   = nullptr;
        MPI_Win_shared_query(win_, r, &r_size, &r_disp, &r_base);
        segment_ptrs_[static_cast<std::size_t>(r)] =
            static_cast<float*>(r_base);
        const std::size_t r_floats =
            static_cast<std::size_t>(r_size) / sizeof(float);
        row_counts_[static_cast<std::size_t>(r)] =
            (d > 0) ? (r_floats / d) : 0;
    }
}

ShmRegion::~ShmRegion()
{
    if (win_ != MPI_WIN_NULL) {
        MPI_Win_free(&win_);
    }
    if (intra_comm_ != MPI_COMM_NULL) {
        MPI_Comm_free(&intra_comm_);
    }
}

const float* ShmRegion::read_remote_row(int         intra_r,
                                        std::size_t row_i,
                                        std::size_t d) const noexcept
{
    assert(intra_r >= 0 && intra_r < intra_size_);
    assert(row_i < row_counts_[static_cast<std::size_t>(intra_r)]);
    return segment_ptrs_[static_cast<std::size_t>(intra_r)] + row_i * d;
}

} // namespace knng::dist
