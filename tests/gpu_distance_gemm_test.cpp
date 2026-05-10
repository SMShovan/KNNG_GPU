/// @file
/// @brief Step 56 — GEMM-based distance tests.

#include <gtest/gtest.h>
#include <knng/gpu/distance_gemm.hpp>
#include <knng/gpu/distance_kernel.hpp>   // cpu_pairwise_l2sq (oracle)
#include <vector>

using knng::gpu::cpu_pairwise_l2sq;
using knng::gpu::cpu_compute_norms;
using knng::gpu::cpu_distance_gemm;

static void check_near(const std::vector<float>& a, const std::vector<float>& b,
                        float tol = 1e-4f)
{
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], b[i], tol) << "at " << i;
    }
}

// ---------------------------------------------------------------------------
// cpu_compute_norms
// ---------------------------------------------------------------------------

TEST(GemmDistance, NormsCorrect)
{
    // Points: (1,0), (0,1), (3,4) → norms: 1, 1, 25
    const float pts[] = {1,0, 0,1, 3,4};
    float norms[3] = {};
    cpu_compute_norms(pts, 3, 2, norms);
    EXPECT_FLOAT_EQ(norms[0], 1.f);
    EXPECT_FLOAT_EQ(norms[1], 1.f);
    EXPECT_FLOAT_EQ(norms[2], 25.f);
}

// ---------------------------------------------------------------------------
// cpu_distance_gemm matches cpu_pairwise_l2sq
// ---------------------------------------------------------------------------

TEST(GemmDistance, MatchesNaive2x3)
{
    const float q[] = {0,0, 1,0};
    const float r[] = {0,0, 1,0, 0,1};
    std::vector<float> naive(6), gemm(6);
    cpu_pairwise_l2sq(q, r, 2, 3, 2, naive.data());
    cpu_distance_gemm(q, r, 2, 3, 2, gemm.data());
    check_near(naive, gemm);
}

TEST(GemmDistance, SelfDistance)
{
    const float q[] = {1,2,3};
    float out = 99.f;
    cpu_distance_gemm(q, q, 1, 1, 3, &out);
    EXPECT_NEAR(out, 0.f, 1e-5f);
}

TEST(GemmDistance, MatchesNaiveHighDim)
{
    constexpr std::size_t NQ=4, NR=8, D=32;
    std::vector<float> q(NQ*D), r(NR*D);
    for (std::size_t i=0;i<NQ*D;++i) q[i]=static_cast<float>(i%7);
    for (std::size_t i=0;i<NR*D;++i) r[i]=static_cast<float>(i%5+1);

    std::vector<float> naive(NQ*NR), gemm(NQ*NR);
    cpu_pairwise_l2sq(q.data(), r.data(), NQ, NR, D, naive.data());
    cpu_distance_gemm(q.data(), r.data(), NQ, NR, D, gemm.data());
    check_near(naive, gemm);
}

TEST(GemmDistance, Symmetry)
{
    constexpr std::size_t N=5, D=4;
    std::vector<float> pts(N*D);
    for (std::size_t i=0;i<N*D;++i) pts[i]=static_cast<float>(i%9);

    std::vector<float> ab(N*N), ba(N*N);
    cpu_distance_gemm(pts.data(), pts.data(), N, N, D, ab.data());
    for (std::size_t i=0;i<N;++i) {
        for (std::size_t j=0;j<N;++j) {
            EXPECT_NEAR(ab[i*N+j], ab[j*N+i], 1e-4f);
        }
    }
}

// ---------------------------------------------------------------------------
// GPU vs CPU
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)
TEST(GemmDistance, GpuMatchesCpu)
{
    constexpr std::size_t NQ=8, NR=16, D=16;
    std::vector<float> q(NQ*D), r(NR*D);
    for (std::size_t i=0;i<NQ*D;++i) q[i]=static_cast<float>(i%7);
    for (std::size_t i=0;i<NR*D;++i) r[i]=static_cast<float>(i%5+1);

    std::vector<float> cpu(NQ*NR);
    cpu_pairwise_l2sq(q.data(), r.data(), NQ, NR, D, cpu.data());

    auto gpu = knng::gpu::gpu_distance_gemm(q.data(), r.data(), NQ, NR, D);
    check_near(cpu, gpu, 1e-3f);
}
#endif
