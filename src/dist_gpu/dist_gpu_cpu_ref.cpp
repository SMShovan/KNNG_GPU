/// @file
/// @brief CPU-reference implementations for all Phase 12 dist_gpu algorithms.
///
/// Always compiled as part of knng_dist_gpu_ref. Enables tests on CPU-only /
/// Mac machines where no CUDA device is available.
///
/// Each Step's simulation is marked with a comment. The file grows as later
/// steps are added.
///
/// When KNNG_DIST_GPU_CUDA_BUILD is defined (by CMake when CUDA is present),
/// sections that would conflict with the real .cu implementations are omitted.

#include <knng/dist_gpu/cuda_aware_mpi.hpp>

#include <mpi.h>

// ===========================================================================
// Step 89 — CUDA-aware MPI: CPU-reference stubs
// ===========================================================================
// Omitted when the real CUDA path (cuda_aware_mpi.cu) is compiled instead.
#if !defined(KNNG_DIST_GPU_CUDA_BUILD)

namespace knng::dist_gpu {

CudaAwareMpiCapability query_cuda_aware_mpi()
{
    return {false};  // No GPU — never CUDA-aware.
}

void mpi_device_send(const void* buf, std::size_t bytes,
                     int dest, int tag, MPI_Comm comm)
{
    MPI_Send(buf, static_cast<int>(bytes), MPI_BYTE, dest, tag, comm);
}

void mpi_device_recv(void* buf, std::size_t bytes,
                     int src, int tag, MPI_Comm comm)
{
    MPI_Recv(buf, static_cast<int>(bytes), MPI_BYTE,
             src, tag, comm, MPI_STATUS_IGNORE);
}

} // namespace knng::dist_gpu

#endif // !KNNG_DIST_GPU_CUDA_BUILD
