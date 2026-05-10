/// @file
/// @brief Step 58 — WMMA distance tests (CPU reference + GPU on CC≥70).

#include <gtest/gtest.h>
#include <knng/gpu/wmma_distance.hpp>
#include <knng/gpu/distance_kernel.hpp>   // cpu_pairwise_l2sq (oracle)

#include <vector>

using knng::gpu::cpu_wmma_distance;
using knng::gpu::cpu_pairwise_l2sq;

static void check_near(const std::vector<float>& a, const std::vector<float>& b,
                        float tol = 1e-4f)
{
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i=0; i<a.size(); ++i)
        EXPECT_NEAR(a[i], b[i], tol) << "at " << i;
}

// ---------------------------------------------------------------------------
// CPU WMMA reference == CPU naive (they both compute the GEMM decomposition)
// ---------------------------------------------------------------------------

TEST(WmmaDistance, MatchesNaive2x2)
{
    const float q[]={1,0, 0,1};
    const float r[]={1,1, 2,0};
    std::vector<float> naive(4), wmma(4);
    cpu_pairwise_l2sq(q, r, 2, 2, 2, naive.data());
    cpu_wmma_distance(q, r, 2, 2, 2, wmma.data());
    check_near(naive, wmma);
}

TEST(WmmaDistance, MatchesNaiveHighDim)
{
    constexpr std::size_t NQ=4, NR=8, D=16;
    std::vector<float> q(NQ*D), r(NR*D);
    for (std::size_t i=0;i<NQ*D;++i) q[i]=static_cast<float>(i%7);
    for (std::size_t i=0;i<NR*D;++i) r[i]=static_cast<float>(i%5+1);

    std::vector<float> naive(NQ*NR), wmma(NQ*NR);
    cpu_pairwise_l2sq(q.data(), r.data(), NQ, NR, D, naive.data());
    cpu_wmma_distance(q.data(), r.data(), NQ, NR, D, wmma.data());
    check_near(naive, wmma);
}

TEST(WmmaDistance, SelfDistanceZero)
{
    const float pts[]={1,2,3,4};
    float naive_out, wmma_out;
    cpu_pairwise_l2sq(pts, pts, 1, 1, 4, &naive_out);
    cpu_wmma_distance(pts, pts, 1, 1, 4, &wmma_out);
    EXPECT_NEAR(wmma_out, 0.f, 1e-5f);
}

// ---------------------------------------------------------------------------
// GPU tests (CC≥70, CUDA only)
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA)
TEST(WmmaDistance, GpuMatchesCpu)
{
    // nr and d must be multiples of 16 for WMMA.
    constexpr std::size_t NQ=16, NR=16, D=16;
    std::vector<float> q(NQ*D), r(NR*D);
    for (std::size_t i=0;i<NQ*D;++i) q[i]=static_cast<float>(i%7);
    for (std::size_t i=0;i<NR*D;++i) r[i]=static_cast<float>(i%5+1);

    std::vector<float> cpu_out(NQ*NR);
    cpu_pairwise_l2sq(q.data(), r.data(), NQ, NR, D, cpu_out.data());

    // fp16 rounding → slightly looser tolerance than fp32.
    auto gpu_out = knng::gpu::gpu_wmma_distance(q.data(), r.data(), NQ, NR, D);
    check_near(cpu_out, gpu_out, 0.5f);   // fp16 rounding + 128 accumulation steps
}
#endif
