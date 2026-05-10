/// @file
/// @brief Step 54 — Warp-level top-k KNN tests.

#include <gtest/gtest.h>
#include <knng/gpu/warp_top_k.hpp>
#include <knng/gpu/brute_force.hpp>   // cpu_brute_force_knn (oracle)
#include <knng/core/dataset.hpp>

#include <vector>

using knng::Dataset;
using knng::Knng;
using knng::gpu::cpu_brute_force_knn;
using knng::gpu::cpu_warp_top_k_knn;

static Dataset make_ds(std::vector<float> data, std::size_t n, std::size_t d)
{
    Dataset ds; ds.data = std::move(data); ds.n = n; ds.d = d; return ds;
}

static void expect_graphs_equal(const Knng& ref, const Knng& got)
{
    ASSERT_EQ(ref.neighbors.size(), got.neighbors.size());
    for (std::size_t i = 0; i < ref.neighbors.size(); ++i) {
        EXPECT_EQ(ref.neighbors[i], got.neighbors[i]) << "neighbor at " << i;
        EXPECT_NEAR(ref.distances[i], got.distances[i], 1e-5f) << "dist at " << i;
    }
}

// ---------------------------------------------------------------------------
// CPU warp-sim == CPU reference
// ---------------------------------------------------------------------------

TEST(WarpTopK, TwoPoints)
{
    auto ds = make_ds({0.f, 5.f}, 2, 1);
    expect_graphs_equal(cpu_brute_force_knn(ds, 1),
                        cpu_warp_top_k_knn(ds, 1));
}

TEST(WarpTopK, Clusters)
{
    std::vector<float> data = {
        0.f,0.f, 0.1f,0.f, 0.f,0.1f, 0.1f,0.1f,
        9.f,9.f, 9.1f,9.f, 9.f,9.1f, 9.1f,9.1f,
    };
    auto ds = make_ds(data, 8, 2);
    expect_graphs_equal(cpu_brute_force_knn(ds, 3),
                        cpu_warp_top_k_knn(ds, 3));
}

TEST(WarpTopK, MoreThan32Points)
{
    // n=50 forces multiple warp-strides.
    std::vector<float> data(50 * 3);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i % 11);
    auto ds = make_ds(data, 50, 3);
    expect_graphs_equal(cpu_brute_force_knn(ds, 5),
                        cpu_warp_top_k_knn(ds, 5));
}

TEST(WarpTopK, SortedOutput)
{
    std::vector<float> pts(16 * 2);
    for (std::size_t i = 0; i < pts.size(); ++i) pts[i] = static_cast<float>(i);
    auto ds = make_ds(pts, 16, 2);
    Knng g  = cpu_warp_top_k_knn(ds, 4);
    for (std::size_t qi = 0; qi < 16; ++qi) {
        for (std::size_t r = 1; r < 4; ++r) {
            EXPECT_LE(g.distances[qi*4+r-1], g.distances[qi*4+r]);
        }
    }
}

// ---------------------------------------------------------------------------
// GPU vs CPU
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)
TEST(WarpTopK, GpuMatchesCpu)
{
    std::vector<float> data(64 * 4);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i % 7);
    auto ds = make_ds(data, 64, 4);
    expect_graphs_equal(cpu_brute_force_knn(ds, 5),
                        knng::gpu::gpu_warp_top_k_knn(ds, 5));
}
#endif
