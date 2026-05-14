/// @file
/// @brief Phase 12 (Steps 89–100) — Distributed multi-node GPU algorithm tests.
///
/// Structure:
///   - CPU-reference tests always run (require MPI initialisation, not CUDA).
///   - GPU tests are gated at runtime — GTEST_SKIP() when no CUDA device.
///
/// A custom main() initialises MPI so the test binary can be launched by
/// `mpirun -np 1` (or N ranks on a cluster) without modification.

#include <knng/dist_gpu/cuda_aware_mpi.hpp>

#include <gtest/gtest.h>
#include <mpi.h>

#include <cstring>
#include <vector>

// ── helpers ──────────────────────────────────────────────────────────────────

namespace {
int mpi_rank() noexcept
{
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}
} // anonymous namespace

#if defined(KNNG_HAVE_CUDA)
#  include <cuda_runtime.h>
namespace {
int cuda_device_count()
{
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
}
} // anonymous namespace
#  define SKIP_IF_NO_GPU() \
     do { if (cuda_device_count() == 0) GTEST_SKIP() << "no CUDA device"; } while (false)
#else
#  define SKIP_IF_NO_GPU() GTEST_SKIP() << "CUDA not compiled"
#endif

// ===========================================================================
// Step 89 — CUDA-aware MPI
// ===========================================================================

TEST(CudaAwareMpi, QueryCompiles)
{
    // Just verify the probe compiles and returns a valid struct.
    auto cap = knng::dist_gpu::query_cuda_aware_mpi();
    SUCCEED() << "cuda_aware=" << cap.cuda_aware;
}

TEST(CudaAwareMpi, CpuPathNotCudaAware)
{
#if !defined(KNNG_HAVE_CUDA)
    auto cap = knng::dist_gpu::query_cuda_aware_mpi();
    EXPECT_FALSE(cap.cuda_aware)
        << "CPU-reference path must always report cuda_aware=false";
#else
    // On CUDA path the result depends on the MPI build — skip the assertion.
    SUCCEED();
#endif
}

TEST(CudaAwareMpi, SendRecvRoundTrip_HostBuffers)
{
    // Verify mpi_device_send + mpi_device_recv with host buffers (CPU path).
    // Post MPI_Irecv first so the single-rank self-send does not deadlock.
    const std::vector<char> send_data = {'\x01', '\x02', '\x03', '\x04'};
    std::vector<char> recv_data(4, '\x00');

    MPI_Request req;
    MPI_Irecv(recv_data.data(), static_cast<int>(recv_data.size()),
              MPI_BYTE, mpi_rank(), 89, MPI_COMM_WORLD, &req);

    knng::dist_gpu::mpi_device_send(
        send_data.data(), send_data.size(), mpi_rank(), 89, MPI_COMM_WORLD);

    MPI_Wait(&req, MPI_STATUS_IGNORE);

    for (std::size_t i = 0; i < send_data.size(); ++i)
        EXPECT_EQ(recv_data[i], send_data[i]) << "byte mismatch at " << i;
}

TEST(CudaAwareMpi, GpuQueryAndSend_DoesNotCrash)
{
    SKIP_IF_NO_GPU();
#if defined(KNNG_HAVE_CUDA)
    // Verify device-pointer send works (staging path) with a self-send.
    constexpr int N = 16;
    float h_send[N], h_recv[N] = {};
    for (int i = 0; i < N; ++i) h_send[i] = static_cast<float>(i);

    float* d_send = nullptr;
    cudaMalloc(&d_send, N * sizeof(float));
    cudaMemcpy(d_send, h_send, N * sizeof(float), cudaMemcpyHostToDevice);

    // Post host Irecv first (MPI_Irecv with host ptr is always safe).
    MPI_Request req;
    MPI_Irecv(h_recv, N * sizeof(float), MPI_BYTE,
              mpi_rank(), 890, MPI_COMM_WORLD, &req);

    // mpi_device_send stages D→H then calls MPI_Send.
    knng::dist_gpu::mpi_device_send(
        d_send, N * sizeof(float), mpi_rank(), 890, MPI_COMM_WORLD);

    MPI_Wait(&req, MPI_STATUS_IGNORE);
    cudaFree(d_send);

    for (int i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(h_recv[i], h_send[i]) << "float mismatch at " << i;
#endif
}

// ===========================================================================
// main — MPI must be initialised before GoogleTest runs its tests
// ===========================================================================

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
