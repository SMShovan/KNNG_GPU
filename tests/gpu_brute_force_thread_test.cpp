/// @file
/// @brief Step 48 — Thread-per-query GPU brute-force KNN tests.
///
/// All tests run `cpu_brute_force_knn` (always available).
/// When CUDA is present the same dataset is also pushed through
/// `brute_force_thread_knn` and results compared.

#include <gtest/gtest.h>
#include <knng/gpu/brute_force.hpp>
#include <knng/core/dataset.hpp>

#include <cstddef>
#include <vector>

using knng::Dataset;
using knng::Knng;
using knng::gpu::cpu_brute_force_knn;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a small Dataset from a flat row-major float array.
static Dataset make_dataset(std::vector<float> data, std::size_t n, std::size_t d)
{
    Dataset ds;
    ds.data = std::move(data);
    ds.n    = n;
    ds.d    = d;
    return ds;
}

/// Check that every neighbor ID is in [0, n) and not the query index itself.
static void check_valid_ids(const Knng& g, std::size_t n)
{
    for (std::size_t qi = 0; qi < n; ++qi) {
        for (std::size_t r = 0; r < g.k; ++r) {
            const auto id = g.neighbors[qi * g.k + r];
            EXPECT_LT(id, n) << "out-of-range id at query " << qi;
            EXPECT_NE(id, qi) << "self-neighbor at query " << qi;
        }
    }
}

/// Check that distances are sorted ascending within each row.
static void check_sorted(const Knng& g)
{
    for (std::size_t qi = 0; qi < g.n; ++qi) {
        for (std::size_t r = 1; r < g.k; ++r) {
            EXPECT_LE(g.distances[qi * g.k + r - 1],
                      g.distances[qi * g.k + r])
                << "unsorted at query " << qi << " rank " << r;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference tests (always run)
// ---------------------------------------------------------------------------

TEST(GpuBfThread, TrivialTwoPoints)
{
    // 2 points in 1-D: only one possible neighbor for each.
    auto ds = make_dataset({0.f, 1.f}, 2, 1);
    Knng g  = cpu_brute_force_knn(ds, 1);
    EXPECT_EQ(g.neighbors[0], 1u);   // nearest to point 0 is point 1
    EXPECT_EQ(g.neighbors[1], 0u);   // nearest to point 1 is point 0
    EXPECT_FLOAT_EQ(g.distances[0], 1.f);
    EXPECT_FLOAT_EQ(g.distances[1], 1.f);
}

TEST(GpuBfThread, EightPointClusters)
{
    // Two tight clusters: {0,1,2,3} near origin, {4,5,6,7} far away.
    // k=3: every point's 3 nearest should all be in its own cluster.
    std::vector<float> data = {
        0.f, 0.f,   0.1f, 0.f,   0.f, 0.1f,   0.1f, 0.1f,  // cluster A
        9.f, 9.f,   9.1f, 9.f,   9.f, 9.1f,   9.1f, 9.1f,  // cluster B
    };
    auto ds = make_dataset(data, 8, 2);
    Knng g  = cpu_brute_force_knn(ds, 3);

    check_valid_ids(g, 8);
    check_sorted(g);

    // Cluster A: points 0-3, cluster B: points 4-7
    for (std::size_t qi = 0; qi < 4; ++qi) {
        for (std::size_t r = 0; r < 3; ++r) {
            EXPECT_LT(g.neighbors[qi * 3 + r], 4u)
                << "cluster A point " << qi << " got cross-cluster neighbor";
        }
    }
    for (std::size_t qi = 4; qi < 8; ++qi) {
        for (std::size_t r = 0; r < 3; ++r) {
            EXPECT_GE(g.neighbors[qi * 3 + r], 4u)
                << "cluster B point " << qi << " got cross-cluster neighbor";
        }
    }
}

TEST(GpuBfThread, SortedDistances)
{
    std::vector<float> pts = {0.f, 3.f, 1.f, 5.f, 2.f};  // 1-D, 5 points
    auto ds = make_dataset(pts, 5, 1);
    Knng g  = cpu_brute_force_knn(ds, 4);
    check_sorted(g);
}

TEST(GpuBfThread, KEqualsNMinusOne)
{
    // k = n-1: every point gets all others as neighbours.
    auto ds = make_dataset({1.f, 2.f, 3.f, 4.f}, 4, 1);
    Knng g  = cpu_brute_force_knn(ds, 3);
    check_valid_ids(g, 4);
    check_sorted(g);
    EXPECT_EQ(g.neighbors.size(), 4u * 3u);
}

// ---------------------------------------------------------------------------
// GPU vs CPU comparison (only when CUDA is present)
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)
TEST(GpuBfThread, GpuMatchesCpu)
{
    std::vector<float> data = {
        0.f, 0.f,   0.1f, 0.f,   0.f, 0.1f,   0.1f, 0.1f,
        9.f, 9.f,   9.1f, 9.f,   9.f, 9.1f,   9.1f, 9.1f,
    };
    auto ds = make_dataset(data, 8, 2);
    constexpr std::size_t K = 3;

    Knng cpu = cpu_brute_force_knn(ds, K);
    Knng gpu = knng::gpu::brute_force_thread_knn(ds, K);

    ASSERT_EQ(cpu.neighbors.size(), gpu.neighbors.size());
    ASSERT_EQ(cpu.distances.size(), gpu.distances.size());

    for (std::size_t i = 0; i < cpu.neighbors.size(); ++i) {
        EXPECT_EQ(cpu.neighbors[i], gpu.neighbors[i]) << "id mismatch at " << i;
        EXPECT_NEAR(cpu.distances[i], gpu.distances[i], 1e-4f)
            << "dist mismatch at " << i;
    }
}
#endif
