/// @file
/// @brief Step 57 — fp16 storage, fp32 compute distance tests.

#include <gtest/gtest.h>
#include <knng/gpu/fp16_distance.hpp>
#include <knng/gpu/distance_kernel.hpp>   // cpu_pairwise_l2sq (oracle)

#include <cstdint>
#include <cmath>
#include <vector>

using knng::gpu::f32_to_f16_bits;
using knng::gpu::f16_bits_to_f32;
using knng::gpu::cpu_fp32_to_fp16;
using knng::gpu::cpu_fp16_pairwise_l2sq;
using knng::gpu::cpu_pairwise_l2sq;

// ---------------------------------------------------------------------------
// Software fp16 conversion
// ---------------------------------------------------------------------------

TEST(Fp16, ZeroRoundTrip)
{
    EXPECT_FLOAT_EQ(f16_bits_to_f32(f32_to_f16_bits(0.f)), 0.f);
}

TEST(Fp16, OneRoundTrip)
{
    // 1.0 is exactly representable in fp16.
    EXPECT_FLOAT_EQ(f16_bits_to_f32(f32_to_f16_bits(1.f)), 1.f);
}

TEST(Fp16, SmallValueRoundTrip)
{
    // fp16 has ~3.3 decimal digits of precision; check relative error < 0.1%.
    const float v = 3.14159f;
    const float recovered = f16_bits_to_f32(f32_to_f16_bits(v));
    EXPECT_NEAR(recovered, v, v * 0.002f);
}

TEST(Fp16, NegativeValue)
{
    const float v = -2.718f;
    const float r = f16_bits_to_f32(f32_to_f16_bits(v));
    EXPECT_NEAR(r, v, std::abs(v) * 0.002f);
}

// ---------------------------------------------------------------------------
// CPU fp16 distance matches fp32 distance (within fp16 rounding tolerance)
// ---------------------------------------------------------------------------

static void check_relative(
    const std::vector<float>& ref,
    const std::vector<float>& got,
    float rel_tol = 0.005f)   // 0.5% — fp16 has ~0.1% rounding error per element
{
    ASSERT_EQ(ref.size(), got.size());
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] < 1e-4f) {
            EXPECT_NEAR(got[i], ref[i], 1e-3f) << "at " << i;
        } else {
            EXPECT_NEAR(got[i] / ref[i], 1.f, rel_tol) << "at " << i;
        }
    }
}

TEST(Fp16, DistanceMatchesFp32)
{
    const float q[] = {1.f, 2.f, 3.f, 0.f};
    const float r[] = {4.f, 6.f, 3.f, 1.f};
    constexpr std::size_t NQ=1, NR=1, D=4;

    std::uint16_t q16[4], r16[4];
    cpu_fp32_to_fp16(q, NQ, D, q16);
    cpu_fp32_to_fp16(r, NR, D, r16);

    float fp32_out, fp16_out;
    cpu_pairwise_l2sq(q, r, NQ, NR, D, &fp32_out);
    cpu_fp16_pairwise_l2sq(q16, r16, NQ, NR, D, &fp16_out);

    EXPECT_NEAR(fp16_out, fp32_out, fp32_out * 0.005f + 1e-3f);
}

TEST(Fp16, MatrixMatchesFp32)
{
    constexpr std::size_t NQ=3, NR=4, D=8;
    std::vector<float> q(NQ*D), r(NR*D);
    for (std::size_t i=0;i<NQ*D;++i) q[i]=static_cast<float>(i%5);
    for (std::size_t i=0;i<NR*D;++i) r[i]=static_cast<float>(i%7+1);

    std::vector<std::uint16_t> q16(NQ*D), r16(NR*D);
    cpu_fp32_to_fp16(q.data(), NQ, D, q16.data());
    cpu_fp32_to_fp16(r.data(), NR, D, r16.data());

    std::vector<float> fp32_out(NQ*NR), fp16_out(NQ*NR);
    cpu_pairwise_l2sq(q.data(), r.data(), NQ, NR, D, fp32_out.data());
    cpu_fp16_pairwise_l2sq(q16.data(), r16.data(), NQ, NR, D, fp16_out.data());

    check_relative(fp32_out, fp16_out);
}

TEST(Fp16, SelfDistanceNearZero)
{
    constexpr std::size_t N=4, D=4;
    std::vector<float> pts={1,2,3,4, 5,6,7,8, 1,1,1,1, 0,0,0,0};
    std::vector<std::uint16_t> pts16(N*D);
    cpu_fp32_to_fp16(pts.data(), N, D, pts16.data());
    std::vector<float> out(N*N);
    cpu_fp16_pairwise_l2sq(pts16.data(), pts16.data(), N, N, D, out.data());
    for (std::size_t i=0; i<N; ++i) {
        EXPECT_NEAR(out[i*N+i], 0.f, 1e-2f) << "diagonal " << i;
    }
}

// ---------------------------------------------------------------------------
// GPU vs CPU (CUDA only)
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)
TEST(Fp16, GpuMatchesCpu)
{
    constexpr std::size_t NQ=4, NR=8, D=16;
    std::vector<float> q(NQ*D), r(NR*D);
    for (std::size_t i=0;i<NQ*D;++i) q[i]=static_cast<float>(i%7);
    for (std::size_t i=0;i<NR*D;++i) r[i]=static_cast<float>(i%5+1);

    std::vector<float> cpu_out(NQ*NR);
    cpu_pairwise_l2sq(q.data(), r.data(), NQ, NR, D, cpu_out.data());

    auto gpu_out = knng::gpu::gpu_fp16_pairwise_l2sq(q.data(), r.data(), NQ, NR, D);
    check_relative(cpu_out, gpu_out, 0.005f);
}
#endif
