/// @file
/// @brief Step 52 — Coalesced distance kernel unit tests.

#include <gtest/gtest.h>
#include <knng/gpu/coalesced_distance.hpp>
#include <knng/gpu/distance_kernel.hpp>   // cpu_pairwise_l2sq (oracle)

#include <vector>

using knng::gpu::cpu_pairwise_l2sq;
using knng::gpu::cpu_transpose;
using knng::gpu::cpu_coalesced_pairwise_l2sq;

// ---------------------------------------------------------------------------
// Transpose correctness
// ---------------------------------------------------------------------------

TEST(CoalescedDistance, TransposeSquare)
{
    // 2×3 matrix: row-major → col-major
    const float src[6] = {1, 2, 3,
                           4, 5, 6};
    float dst[6] = {};
    cpu_transpose(src, 2, 3, dst);
    // dst should be [3][2]: column0=(1,4), column1=(2,5), column2=(3,6)
    EXPECT_FLOAT_EQ(dst[0], 1.f); EXPECT_FLOAT_EQ(dst[1], 4.f);
    EXPECT_FLOAT_EQ(dst[2], 2.f); EXPECT_FLOAT_EQ(dst[3], 5.f);
    EXPECT_FLOAT_EQ(dst[4], 3.f); EXPECT_FLOAT_EQ(dst[5], 6.f);
}

TEST(CoalescedDistance, TransposeRoundTrip)
{
    const std::vector<float> original = {1,2,3,4,5,6,7,8};
    std::vector<float> transposed(8), recovered(8);
    cpu_transpose(original.data(), 4, 2, transposed.data());
    cpu_transpose(transposed.data(), 2, 4, recovered.data());
    EXPECT_EQ(original, recovered);
}

// ---------------------------------------------------------------------------
// Coalesced distance matches naive reference
// ---------------------------------------------------------------------------

static void check_near(
    const std::vector<float>& expected,
    const std::vector<float>& got,
    float tol = 1e-5f)
{
    ASSERT_EQ(expected.size(), got.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(expected[i], got[i], tol) << "at index " << i;
    }
}

TEST(CoalescedDistance, MatchesNaive2x3)
{
    const float q[] = {0.f, 0.f,  1.f, 0.f};
    const float r[] = {0.f, 0.f,  1.f, 0.f,  0.f, 1.f};
    constexpr std::size_t NQ = 2, NR = 3, D = 2;

    std::vector<float> naive_out(NQ * NR), col_out(NQ * NR);
    std::vector<float> refs_T(D * NR);

    cpu_pairwise_l2sq(q, r, NQ, NR, D, naive_out.data());
    cpu_transpose(r, NR, D, refs_T.data());
    cpu_coalesced_pairwise_l2sq(q, refs_T.data(), NQ, NR, D, col_out.data());

    check_near(naive_out, col_out);
}

TEST(CoalescedDistance, MatchesNaiveHighDim)
{
    constexpr std::size_t NQ = 4, NR = 8, D = 32;
    std::vector<float> q(NQ * D), r(NR * D);
    for (std::size_t i = 0; i < NQ * D; ++i) q[i] = static_cast<float>(i % 7);
    for (std::size_t i = 0; i < NR * D; ++i) r[i] = static_cast<float>(i % 5 + 1);

    std::vector<float> naive_out(NQ * NR), col_out(NQ * NR);
    std::vector<float> refs_T(D * NR);

    cpu_pairwise_l2sq(q.data(), r.data(), NQ, NR, D, naive_out.data());
    cpu_transpose(r.data(), NR, D, refs_T.data());
    cpu_coalesced_pairwise_l2sq(q.data(), refs_T.data(), NQ, NR, D, col_out.data());

    check_near(naive_out, col_out);
}

TEST(CoalescedDistance, SelfDistanceZero)
{
    constexpr std::size_t N = 5, D = 4;
    std::vector<float> pts(N * D);
    for (std::size_t i = 0; i < N * D; ++i) pts[i] = static_cast<float>(i);

    std::vector<float> refs_T(D * N);
    cpu_transpose(pts.data(), N, D, refs_T.data());

    std::vector<float> out(N * N);
    cpu_coalesced_pairwise_l2sq(pts.data(), refs_T.data(), N, N, D, out.data());

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(out[i * N + i], 0.f) << "diagonal at " << i;
    }
}

// ---------------------------------------------------------------------------
// GPU vs CPU (only when CUDA is present)
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)
TEST(CoalescedDistance, GpuMatchesCpu)
{
    constexpr std::size_t NQ = 8, NR = 16, D = 32;
    std::vector<float> q(NQ * D), r(NR * D);
    for (std::size_t i = 0; i < NQ * D; ++i) q[i] = static_cast<float>(i % 11);
    for (std::size_t i = 0; i < NR * D; ++i) r[i] = static_cast<float>(i % 7 + 1);

    std::vector<float> cpu_out(NQ * NR);
    cpu_pairwise_l2sq(q.data(), r.data(), NQ, NR, D, cpu_out.data());

    auto gpu_out = knng::gpu::gpu_coalesced_pairwise_l2sq(
        q.data(), r.data(), NQ, NR, D);

    check_near(cpu_out, gpu_out, 1e-4f);
}
#endif
