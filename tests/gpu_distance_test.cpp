/// @file
/// @brief Step 47 — GPU distance kernel unit tests.
///
/// All tests run the CPU reference (`cpu_pairwise_l2sq`) which is always
/// available.  When CUDA is present, the same fixtures are also pushed
/// through the GPU kernel and the results compared element-wise against
/// the CPU output (within floating-point tolerance).

#include <gtest/gtest.h>
#include <knng/gpu/distance_kernel.hpp>

#include <vector>

using knng::gpu::cpu_pairwise_l2sq;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void check_matrix(
    const std::vector<float>& expected,
    const std::vector<float>& got,
    float                     abs_tol = 1e-5f)
{
    ASSERT_EQ(expected.size(), got.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(got[i], expected[i], abs_tol)
            << "mismatch at flat index " << i;
    }
}

// ---------------------------------------------------------------------------
// CPU reference: correctness on hand-verified examples
// ---------------------------------------------------------------------------

TEST(GpuDistance, SelfDistanceIsZero)
{
    // A single point queried against itself must have distance 0.
    const float q[] = {1.f, 2.f, 3.f};
    float       out = -1.f;
    cpu_pairwise_l2sq(q, q, 1, 1, 3, &out);
    EXPECT_FLOAT_EQ(out, 0.f);
}

TEST(GpuDistance, TwoPoints1D)
{
    // ||5 - 2||² = 9
    const float q[] = {5.f};
    const float r[] = {2.f};
    float       out = 0.f;
    cpu_pairwise_l2sq(q, r, 1, 1, 1, &out);
    EXPECT_FLOAT_EQ(out, 9.f);
}

TEST(GpuDistance, TwoPoints3D)
{
    // q = (1,2,3), r = (4,6,3)
    // ||(1-4)|² + |(2-6)|² + |(3-3)|² = 9 + 16 + 0 = 25
    const float q[] = {1.f, 2.f, 3.f};
    const float r[] = {4.f, 6.f, 3.f};
    float       out = 0.f;
    cpu_pairwise_l2sq(q, r, 1, 1, 3, &out);
    EXPECT_FLOAT_EQ(out, 25.f);
}

TEST(GpuDistance, FullMatrix2x3)
{
    // nq=2, nr=3, d=2
    // Queries: q0=(0,0), q1=(1,0)
    // Refs:    r0=(0,0), r1=(1,0), r2=(0,1)
    //
    // Expected output (row-major [nq][nr]):
    //   row0: d(q0,r0)=0, d(q0,r1)=1, d(q0,r2)=1
    //   row1: d(q1,r0)=1, d(q1,r1)=0, d(q1,r2)=2
    const float q[] = {0.f, 0.f,  1.f, 0.f};
    const float r[] = {0.f, 0.f,  1.f, 0.f,  0.f, 1.f};
    const std::vector<float> expected = {0.f, 1.f, 1.f,
                                          1.f, 0.f, 2.f};
    std::vector<float> out(6, -1.f);
    cpu_pairwise_l2sq(q, r, 2, 3, 2, out.data());
    check_matrix(expected, out);
}

TEST(GpuDistance, SymmetryProperty)
{
    // d(a,b) == d(b,a) for squared L2.
    const float pts[] = {1.f, 2.f,  3.f, 5.f,  0.f, 0.f};
    constexpr std::size_t N = 3;
    constexpr std::size_t D = 2;

    std::vector<float> ab(N * N), ba(N * N);
    cpu_pairwise_l2sq(pts, pts, N, N, D, ab.data());
    cpu_pairwise_l2sq(pts, pts, N, N, D, ba.data());

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            EXPECT_FLOAT_EQ(ab[i * N + j], ba[j * N + i])
                << "symmetry violation at (" << i << "," << j << ")";
        }
    }
}

TEST(GpuDistance, DiagonalIsZero)
{
    // Every point has zero distance to itself.
    const float pts[] = {1.f, 1.f,  2.f, 3.f,  5.f, 0.f};
    constexpr std::size_t N = 3, D = 2;
    std::vector<float> out(N * N);
    cpu_pairwise_l2sq(pts, pts, N, N, D, out.data());
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(out[i * N + i], 0.f) << "diagonal at " << i;
    }
}

TEST(GpuDistance, HighDimensional)
{
    // 128-d unit vectors along the standard basis: distance between any two
    // distinct basis vectors is ||(e_i - e_j)||² = 2.
    constexpr std::size_t N = 4, D = 128;
    std::vector<float> pts(N * D, 0.f);
    for (std::size_t i = 0; i < N; ++i) {
        pts[i * D + i] = 1.f;   // e_i
    }
    std::vector<float> out(N * N);
    cpu_pairwise_l2sq(pts.data(), pts.data(), N, N, D, out.data());

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float expected = (i == j) ? 0.f : 2.f;
            EXPECT_FLOAT_EQ(out[i * N + j], expected)
                << "at (" << i << "," << j << ")";
        }
    }
}

// ---------------------------------------------------------------------------
// GPU vs CPU comparison (only active when CUDA is present)
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)
TEST(GpuDistance, GpuMatchesCpu)
{
    constexpr std::size_t NQ = 8, NR = 8, D = 16;
    std::vector<float> q(NQ * D), r(NR * D);
    // Fill with deterministic values.
    for (std::size_t i = 0; i < NQ * D; ++i) q[i] = static_cast<float>(i % 7);
    for (std::size_t i = 0; i < NR * D; ++i) r[i] = static_cast<float>(i % 5 + 1);

    std::vector<float> cpu_out(NQ * NR);
    cpu_pairwise_l2sq(q.data(), r.data(), NQ, NR, D, cpu_out.data());

    const auto gpu_out = knng::gpu::gpu_pairwise_l2sq(
        q.data(), r.data(), NQ, NR, D);

    check_matrix(cpu_out, gpu_out, 1e-4f);
}
#endif
