/// @file
/// @brief Step 53 — Shared-memory tiled brute-force KNN tests.

#include <gtest/gtest.h>
#include <knng/gpu/tiled_brute_force.hpp>
#include <knng/gpu/brute_force.hpp>    // cpu_brute_force_knn (oracle)
#include <knng/core/dataset.hpp>

#include <vector>

using knng::Dataset;
using knng::Knng;
using knng::gpu::cpu_brute_force_knn;
using knng::gpu::cpu_tiled_brute_force_knn;

static Dataset make_ds(std::vector<float> data, std::size_t n, std::size_t d)
{
    Dataset ds; ds.data = std::move(data); ds.n = n; ds.d = d; return ds;
}

static void check_graphs_equal(const Knng& a, const Knng& b)
{
    ASSERT_EQ(a.neighbors.size(), b.neighbors.size());
    ASSERT_EQ(a.distances.size(), b.distances.size());
    for (std::size_t i = 0; i < a.neighbors.size(); ++i) {
        EXPECT_EQ(a.neighbors[i], b.neighbors[i]) << "neighbor at " << i;
        EXPECT_NEAR(a.distances[i], b.distances[i], 1e-5f) << "dist at " << i;
    }
}

// ---------------------------------------------------------------------------
// CPU tiled == CPU reference (non-tiled)
// ---------------------------------------------------------------------------

TEST(TiledBF, MatchesReferenceTwoPoints)
{
    auto ds = make_ds({0.f, 1.f}, 2, 1);
    check_graphs_equal(cpu_brute_force_knn(ds, 1),
                       cpu_tiled_brute_force_knn(ds, 1));
}

TEST(TiledBF, MatchesReferenceClusters)
{
    std::vector<float> data = {
        0.f,0.f,  0.1f,0.f,  0.f,0.1f,  0.1f,0.1f,
        9.f,9.f,  9.1f,9.f,  9.f,9.1f,  9.1f,9.1f,
    };
    auto ds = make_ds(data, 8, 2);
    check_graphs_equal(cpu_brute_force_knn(ds, 3),
                       cpu_tiled_brute_force_knn(ds, 3));
}

TEST(TiledBF, MatchesReferenceLargeN)
{
    // n=64 > 2*kTileW=64 so at least two full tiles are processed.
    std::vector<float> data(64 * 4);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i % 13);
    }
    auto ds = make_ds(data, 64, 4);
    check_graphs_equal(cpu_brute_force_knn(ds, 5),
                       cpu_tiled_brute_force_knn(ds, 5));
}

TEST(TiledBF, SortedOutput)
{
    std::vector<float> pts(20 * 2);
    for (std::size_t i = 0; i < pts.size(); ++i) pts[i] = static_cast<float>(i);
    auto ds = make_ds(pts, 20, 2);
    Knng g  = cpu_tiled_brute_force_knn(ds, 4);
    for (std::size_t qi = 0; qi < 20; ++qi) {
        for (std::size_t r = 1; r < 4; ++r) {
            EXPECT_LE(g.distances[qi * 4 + r - 1], g.distances[qi * 4 + r]);
        }
    }
}

// ---------------------------------------------------------------------------
// GPU vs CPU
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)
TEST(TiledBF, GpuMatchesCpu)
{
    std::vector<float> data(64 * 4);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i % 7);
    auto ds = make_ds(data, 64, 4);
    check_graphs_equal(cpu_brute_force_knn(ds, 4),
                       knng::gpu::gpu_tiled_brute_force_knn(ds, 4));
}
#endif
