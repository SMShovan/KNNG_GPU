/// @file
/// @brief Step 45 — GPU backend macro layer unit tests.
///
/// These tests exercise the CPU-stub path of `include/knng/gpu/backend.hpp`.
/// On a machine without CUDA/HIP the stubs delegate to `malloc`/`free`/`memcpy`
/// so the RAII and copy semantics can be verified without a GPU.  When CUDA
/// is enabled the same test binary validates the real API calls.

#include <gtest/gtest.h>
#include <knng/gpu/backend.hpp>

#include <stdexcept>

// ---------------------------------------------------------------------------
// GPU_MALLOC / GPU_FREE
// ---------------------------------------------------------------------------

TEST(GpuBackend, MallocAndFree)
{
    float* ptr = nullptr;
    GPU_MALLOC(ptr, 16 * sizeof(float));
    EXPECT_NE(ptr, nullptr);
    GPU_FREE(ptr);
}

TEST(GpuBackend, MallocZeroBytes)
{
    // Allocating zero bytes is implementation-defined but must not crash.
    void* ptr = nullptr;
    GPU_MALLOC(ptr, 0);
    GPU_FREE(ptr);
}

// ---------------------------------------------------------------------------
// GPU_MEMCPY_H2D / GPU_MEMCPY_D2H round-trip
// ---------------------------------------------------------------------------

TEST(GpuBackend, MemcpyRoundTrip)
{
    constexpr std::size_t kN = 8;
    float host_src[kN] = {1, 2, 3, 4, 5, 6, 7, 8};
    float host_dst[kN] = {};

    float* dev = nullptr;
    GPU_MALLOC(dev, kN * sizeof(float));

    GPU_MEMCPY_H2D(dev, host_src, kN * sizeof(float));
    GPU_MEMCPY_D2H(host_dst, dev, kN * sizeof(float));

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_FLOAT_EQ(host_dst[i], host_src[i]) << "mismatch at index " << i;
    }

    GPU_FREE(dev);
}

TEST(GpuBackend, MemcpyDeviceToDevice)
{
    constexpr std::size_t kN = 4;
    float host[kN] = {10.f, 20.f, 30.f, 40.f};
    float result[kN] = {};

    float* src = nullptr;
    float* dst = nullptr;
    GPU_MALLOC(src, kN * sizeof(float));
    GPU_MALLOC(dst, kN * sizeof(float));

    GPU_MEMCPY_H2D(src, host, kN * sizeof(float));
    GPU_MEMCPY_D2D(dst, src, kN * sizeof(float));
    GPU_MEMCPY_D2H(result, dst, kN * sizeof(float));

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_FLOAT_EQ(result[i], host[i]);
    }

    GPU_FREE(src);
    GPU_FREE(dst);
}

// ---------------------------------------------------------------------------
// GPU_SYNC
// ---------------------------------------------------------------------------

TEST(GpuBackend, SyncDoesNotThrow)
{
    EXPECT_NO_THROW(GPU_SYNC());
}

// ---------------------------------------------------------------------------
// GPU_STREAM_CREATE / GPU_STREAM_DESTROY
// ---------------------------------------------------------------------------

TEST(GpuBackend, StreamLifecycle)
{
    knng::gpu::stream_t s{};
    EXPECT_NO_THROW(GPU_STREAM_CREATE(s));
    EXPECT_NO_THROW(GPU_STREAM_DESTROY(s));
}

// ---------------------------------------------------------------------------
// GPU_CHECK — verifies the error-checking path on the stub (error code != 0)
// ---------------------------------------------------------------------------

TEST(GpuBackend, CheckThrowsOnError)
{
#if !defined(KNNG_HAVE_CUDA) && !defined(KNNG_HAVE_HIP)
    // On the CPU stub, GPU_CHECK(non_zero) must throw.
    EXPECT_THROW(GPU_CHECK(1), std::runtime_error);
    EXPECT_NO_THROW(GPU_CHECK(0));
#else
    // On real backends, cudaSuccess / hipSuccess must not throw.
    EXPECT_NO_THROW(GPU_CHECK(knng::gpu::kSuccess));
#endif
}

// ---------------------------------------------------------------------------
// Compile-time: GPU_HOST_DEVICE, GPU_DEVICE, GPU_GLOBAL qualifiers
// These are just attribute macros; the test is that they compile cleanly.
// ---------------------------------------------------------------------------

GPU_HOST_DEVICE static int add_hd(int a, int b) { return a + b; }

TEST(GpuBackend, HostDeviceCallable)
{
    EXPECT_EQ(add_hd(2, 3), 5);
}
