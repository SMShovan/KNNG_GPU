/// @file
/// @brief Step 89 — CUDA-aware MPI: real GPU-path implementations.
///
/// Provides query_cuda_aware_mpi(), mpi_device_send(), and mpi_device_recv()
/// for the CUDA build.  Strategy:
///   - KNNG_CUDA_AWARE_MPI defined (Open MPI ≥ 1.7 / MVAPICH2 with CUDA):
///       pass device pointers directly to MPI.
///   - Fallback (all other CUDA builds):
///       cudaMemcpy D→H into pinned staging buffer, MPI_Send/Recv, free.
///
/// This translation unit is only compiled into knng_dist_gpu (CUDA path).
/// The CPU-reference stubs in dist_gpu_cpu_ref.cpp are suppressed via
/// KNNG_DIST_GPU_CUDA_BUILD when this file is active.

#include <knng/dist_gpu/cuda_aware_mpi.hpp>

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdexcept>
#include <string>

#if defined(KNNG_CUDA_AWARE_MPI)
#  include <mpi-ext.h>
#endif

namespace knng::dist_gpu {

namespace {
inline void cuda_chk(cudaError_t e, const char* where)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(where) + ": " +
                                 cudaGetErrorString(e));
}
} // anonymous namespace

// ---------------------------------------------------------------------------
CudaAwareMpiCapability query_cuda_aware_mpi()
{
#if defined(KNNG_CUDA_AWARE_MPI)
    return {MPIX_Query_cuda_support() != 0};
#else
    return {false};
#endif
}

// ---------------------------------------------------------------------------
void mpi_device_send(const void* d_buf, std::size_t bytes,
                     int dest, int tag, MPI_Comm comm)
{
#if defined(KNNG_CUDA_AWARE_MPI)
    if (MPIX_Query_cuda_support()) {
        // Direct device-pointer transfer — MPI handles the staging internally.
        MPI_Send(d_buf, static_cast<int>(bytes), MPI_BYTE, dest, tag, comm);
        return;
    }
#endif
    // Pinned host staging buffer: D→H copy, send, free.
    void* h = nullptr;
    cuda_chk(cudaMallocHost(&h, bytes),              "cudaMallocHost (send)");
    cuda_chk(cudaMemcpy(h, d_buf, bytes,
                        cudaMemcpyDeviceToHost),      "cudaMemcpy D2H (send)");
    MPI_Send(h, static_cast<int>(bytes), MPI_BYTE, dest, tag, comm);
    cuda_chk(cudaFreeHost(h),                        "cudaFreeHost (send)");
}

// ---------------------------------------------------------------------------
void mpi_device_recv(void* d_buf, std::size_t bytes,
                     int src, int tag, MPI_Comm comm)
{
#if defined(KNNG_CUDA_AWARE_MPI)
    if (MPIX_Query_cuda_support()) {
        MPI_Recv(d_buf, static_cast<int>(bytes), MPI_BYTE,
                 src, tag, comm, MPI_STATUS_IGNORE);
        return;
    }
#endif
    void* h = nullptr;
    cuda_chk(cudaMallocHost(&h, bytes),              "cudaMallocHost (recv)");
    MPI_Recv(h, static_cast<int>(bytes), MPI_BYTE,
             src, tag, comm, MPI_STATUS_IGNORE);
    cuda_chk(cudaMemcpy(d_buf, h, bytes,
                        cudaMemcpyHostToDevice),      "cudaMemcpy H2D (recv)");
    cuda_chk(cudaFreeHost(h),                        "cudaFreeHost (recv)");
}

} // namespace knng::dist_gpu
