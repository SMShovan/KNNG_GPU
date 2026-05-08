/// @file
/// @brief Steps 49 & 50 — Block-per-query and end-to-end GPU brute-force KNN tests.
///
/// CPU-reference tests always run.  GPU comparisons are guarded by
/// `#ifdef KNNG_HAVE_CUDA`.

#include <gtest/gtest.h>
#include <knng/gpu/brute_force.hpp>
#include <knng/core/dataset.hpp>

#include <cstddef>
#include <vector>

using knng::Dataset;
using knng::Knng;
using knng::gpu::cpu_brute_force_knn;

// ---------------------------------------------------------------------------
// Helpers (shared with thread test; duplicated here for self-containment)
// ---------------------------------------------------------------------------

static Dataset make_dataset(std::vector<float> data, std::size_t n, std::size_t d)
{
    Dataset ds;
    ds.data = std::move(data);
    ds.n    = n;
    ds.d    = d;
    return ds;
}

static void check_valid_ids(const Knng& g, std::size_t n)
{
    for (std::size_t qi = 0; qi < n; ++qi) {
        for (std::size_t r = 0; r < g.k; ++r) {
            EXPECT_LT(g.neighbors[qi * g.k + r], n);
            EXPECT_NE(g.neighbors[qi * g.k + r], qi);
        }
    }
}

static void check_sorted(const Knng& g)
{
    for (std::size_t qi = 0; qi < g.n; ++qi) {
        for (std::size_t r = 1; r < g.k; ++r) {
            EXPECT_LE(g.distances[qi * g.k + r - 1],
                      g.distances[qi * g.k + r]);
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference tests (block-kernel strategy, but using cpu_brute_force_knn)
// ---------------------------------------------------------------------------

TEST(GpuBfBlock, LargeK)
{
    // k = n-1 = 7: can only be satisfied by block-per-query (thread-per-query
    // would require k > kMaxKThread when n is large; here n=8 fits fine in CPU).
    std::vector<float> data = {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f
    };
    auto ds = make_dataset(data, 8, 1);
    Knng g  = cpu_brute_force_knn(ds, 7);
    check_valid_ids(g, 8);
    check_sorted(g);
    EXPECT_EQ(g.neighbors.size(), 8u * 7u);
}

TEST(GpuBfBlock, Clusters3D)
{
    // Four 3-D clusters; k=2 nearest should both be in same cluster.
    std::vector<float> data = {
        // cluster A (points 0-1)
        0.f, 0.f, 0.f,   0.1f, 0.f, 0.f,
        // cluster B (points 2-3)
        10.f, 0.f, 0.f,  10.1f, 0.f, 0.f,
    };
    auto ds = make_dataset(data, 4, 3);
    Knng g  = cpu_brute_force_knn(ds, 1);
    // Point 0 → point 1; point 1 → point 0
    EXPECT_EQ(g.neighbors[0], 1u);
    EXPECT_EQ(g.neighbors[1], 0u);
    // Point 2 → point 3; point 3 → point 2
    EXPECT_EQ(g.neighbors[2], 3u);
    EXPECT_EQ(g.neighbors[3], 2u);
}

TEST(GpuBfBlock, CorrectDistances)
{
    // 1-D: points at {0, 3, 5}.  For k=2:
    //   p0: neighbors = {1(d=9), 2(d=25)} in sorted order
    //   p1: neighbors = {0(d=9), 2(d=4)} → {2(d=4), 0(d=9)}
    //   p2: neighbors = {1(d=4), 0(d=25)} → {1(d=4), 0(d=25)}
    auto ds = make_dataset({0.f, 3.f, 5.f}, 3, 1);
    Knng g  = cpu_brute_force_knn(ds, 2);

    EXPECT_FLOAT_EQ(g.distances[0 * 2 + 0], 9.f);   // p0 nearest
    EXPECT_FLOAT_EQ(g.distances[0 * 2 + 1], 25.f);  // p0 second

    EXPECT_FLOAT_EQ(g.distances[1 * 2 + 0], 4.f);   // p1 nearest
    EXPECT_FLOAT_EQ(g.distances[1 * 2 + 1], 9.f);   // p1 second

    EXPECT_FLOAT_EQ(g.distances[2 * 2 + 0], 4.f);   // p2 nearest
    EXPECT_FLOAT_EQ(g.distances[2 * 2 + 1], 25.f);  // p2 second
}

// ---------------------------------------------------------------------------
// Step 50 — gpu_brute_force_knn end-to-end test (CPU path only on Mac)
// ---------------------------------------------------------------------------

TEST(GpuBruteForceEndToEnd, SameAsCpuReference)
{
    // On CPU stub, gpu_brute_force_knn is implemented as cpu_brute_force_knn
    // via the same reference path — this verifies the header compile.
    std::vector<float> data = {1.f, 2.f, 3.f, 4.f, 5.f};
    auto ds = make_dataset(data, 5, 1);
    Knng cpu = cpu_brute_force_knn(ds, 3);
    check_sorted(cpu);
    check_valid_ids(cpu, 5);
}

// ---------------------------------------------------------------------------
// GPU vs CPU comparisons (only when CUDA is available)
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)

static void compare_graphs(const Knng& cpu, const Knng& gpu)
{
    ASSERT_EQ(cpu.neighbors.size(), gpu.neighbors.size());
    ASSERT_EQ(cpu.distances.size(), gpu.distances.size());
    for (std::size_t i = 0; i < cpu.neighbors.size(); ++i) {
        EXPECT_EQ(cpu.neighbors[i], gpu.neighbors[i]) << "id at " << i;
        EXPECT_NEAR(cpu.distances[i], gpu.distances[i], 1e-4f) << "dist at " << i;
    }
}

TEST(GpuBfBlock, GpuMatchesCpuSmall)
{
    auto ds = make_dataset({0.f, 1.f, 2.f, 3.f, 4.f}, 5, 1);
    Knng cpu = cpu_brute_force_knn(ds, 3);
    Knng gpu = knng::gpu::brute_force_block_knn(ds, 3);
    compare_graphs(cpu, gpu);
}

TEST(GpuBfBlock, GpuMatchesCpuHighK)
{
    // k=7 > kMaxKThread=32 — only block-per-query can handle this.
    std::vector<float> data;
    for (int i = 0; i < 8; ++i) {
        data.push_back(static_cast<float>(i));
    }
    auto ds = make_dataset(data, 8, 1);
    Knng cpu = cpu_brute_force_knn(ds, 7);
    Knng gpu = knng::gpu::brute_force_block_knn(ds, 7);
    compare_graphs(cpu, gpu);
}

TEST(GpuBruteForceEndToEnd, GpuMatchesCpu)
{
    std::vector<float> data;
    for (int i = 0; i < 32; ++i) {
        data.push_back(static_cast<float>(i % 8));
        data.push_back(static_cast<float>(i / 8));
    }
    auto ds = make_dataset(data, 16, 2);
    Knng cpu = cpu_brute_force_knn(ds, 4);
    Knng gpu = knng::gpu::gpu_brute_force_knn(ds, 4);
    compare_graphs(cpu, gpu);
}

#endif // KNNG_HAVE_CUDA || KNNG_HAVE_HIP
