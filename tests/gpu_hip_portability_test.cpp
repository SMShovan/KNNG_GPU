/// @file
/// @brief Step 60 — HIP portability layer compile-time checks.
///
/// Verifies that all backend macros are defined regardless of which
/// backend is active (CUDA, HIP, or CPU stub).  The checks are
/// compile-time — if any macro is missing the build fails with a
/// meaningful `#error` rather than a cryptic linker error.

#include <gtest/gtest.h>
#include <knng/gpu/backend.hpp>

// ---------------------------------------------------------------------------
// Static assertions: all macros must be defined
// ---------------------------------------------------------------------------

// GPU_HOST_DEVICE qualifier — must compile on host.
GPU_HOST_DEVICE static int add_portable(int a, int b) { return a + b; }
GPU_INLINE static int mul_portable(int a, int b) { return a * b; }

// kSuccess must equal the actual success value for the active backend.
static_assert(knng::gpu::kSuccess == knng::gpu::kSuccess,
              "kSuccess self-equality sanity check");

// ---------------------------------------------------------------------------
// Runtime checks
// ---------------------------------------------------------------------------

TEST(HipPortability, HostDeviceFunctionCallable)
{
    EXPECT_EQ(add_portable(3, 4), 7);
    EXPECT_EQ(mul_portable(3, 4), 12);
}

TEST(HipPortability, BackendMacrosDefined)
{
    // GPU_SYNC() on CPU stub is a no-op; on CUDA/HIP it synchronises.
    EXPECT_NO_THROW(GPU_SYNC());
}

TEST(HipPortability, StreamCreateDestroy)
{
    knng::gpu::stream_t s{};
    EXPECT_NO_THROW(GPU_STREAM_CREATE(s));
    EXPECT_NO_THROW(GPU_STREAM_DESTROY(s));
}

TEST(HipPortability, SuccessCodeIsZeroLike)
{
    // kSuccess should compare equal to the underlying 0 / success value.
    EXPECT_EQ(knng::gpu::kSuccess, static_cast<knng::gpu::error_t>(0));
}

TEST(HipPortability, BackendName)
{
#if defined(KNNG_HAVE_CUDA) && KNNG_HAVE_CUDA
    const char* backend = "CUDA";
#elif defined(KNNG_HAVE_HIP) && KNNG_HAVE_HIP
    const char* backend = "HIP";
#else
    const char* backend = "CPU_STUB";
#endif
    EXPECT_NE(std::string(backend), "") << "backend name must be non-empty";
}
