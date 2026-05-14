#pragma once

/// @file
/// @brief Step 89 — CUDA-aware MPI probe and device-buffer transfer wrappers.
///
/// Provides:
///   - A runtime probe for whether the linked MPI library supports CUDA-aware
///     device-pointer transfers (`query_cuda_aware_mpi`).
///   - Transparent blocking send/recv wrappers (`mpi_device_send`,
///     `mpi_device_recv`) that pass device pointers directly to MPI when
///     CUDA-aware MPI is available, or stage through a pinned host buffer
///     otherwise.
///
/// CPU-reference path (no CUDA): all functions treat buffers as host pointers
/// and delegate directly to MPI_Send / MPI_Recv.

#if !defined(KNNG_HAVE_MPI) || !KNNG_HAVE_MPI
#  error "cuda_aware_mpi.hpp requires MPI — guard the include with KNNG_HAVE_MPI"
#endif

#include <cstddef>
#include <mpi.h>

namespace knng::dist_gpu {

/// Runtime descriptor for CUDA-aware MPI capability.
struct CudaAwareMpiCapability {
    bool cuda_aware = false; ///< True when device pointers may be passed to MPI.
};

/// Query whether the linked MPI library supports CUDA-aware device transfers.
///
/// GPU path: returns the result of MPIX_Query_cuda_support() when the
///           compile-time probe succeeded; otherwise `{false}`.
/// CPU path: always returns `{false}`.
[[nodiscard]] CudaAwareMpiCapability query_cuda_aware_mpi();

/// Blocking send of `bytes` raw bytes starting at `buf` to rank `dest`.
///
/// GPU path: if CUDA-aware MPI is available, passes the device pointer
///           directly to MPI_Send. Otherwise copies to a pinned staging
///           buffer via cudaMemcpy, sends, then frees the staging buffer.
/// CPU path: `buf` is a host pointer; calls MPI_Send directly.
void mpi_device_send(const void* buf, std::size_t bytes,
                     int dest, int tag, MPI_Comm comm);

/// Blocking receive of `bytes` raw bytes into `buf` from rank `src`.
///
/// GPU path: if CUDA-aware MPI is available, receives directly into the
///           device pointer `buf`. Otherwise receives into a pinned staging
///           buffer and copies to device with cudaMemcpy.
/// CPU path: `buf` is a host pointer; calls MPI_Recv directly.
void mpi_device_recv(void* buf, std::size_t bytes,
                     int src, int tag, MPI_Comm comm);

} // namespace knng::dist_gpu
