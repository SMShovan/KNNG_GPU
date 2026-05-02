/// @file
/// @brief Unit tests for `knng::cpu::brute_force_knn`.
///
/// The algorithm is the project's correctness floor — every later
/// optimisation will assert elementwise equality (within fp tolerance)
/// against this output on small inputs. The cases here pin the
/// per-row contract on a hand-verified 8-point cluster, the strict
/// argument validation, and the deterministic id-based tie-break that
/// keeps output bit-stable across runs without an RNG.

#include <array>
#include <cstddef>
#include <stdexcept>

#include <gtest/gtest.h>

#include "knng/core/dataset.hpp"
#include "knng/core/distance.hpp"
#include "knng/core/graph.hpp"
#include "knng/core/types.hpp"
#include "knng/cpu/brute_force.hpp"

namespace {

/// Two unit-square clusters of four points, separated by a wide gap:
///
///   indices 0..3  → corners of the unit square at the origin
///   indices 4..7  → corners of a unit square at (4, 4)
///
/// Within each cluster, every point's three nearest neighbors are the
/// three other corners of its own square — the inter-cluster distance
/// (>= 18 in squared-L2) is far larger than any intra-cluster
/// distance (<= 2 in squared-L2).
knng::Dataset two_clusters_eight_points()
{
    knng::Dataset ds(8, 2);
    constexpr std::array<std::array<float, 2>, 8> coords{{
        {0.0f, 0.0f},  // 0
        {1.0f, 0.0f},  // 1
        {0.0f, 1.0f},  // 2
        {1.0f, 1.0f},  // 3
        {4.0f, 4.0f},  // 4
        {5.0f, 4.0f},  // 5
        {4.0f, 5.0f},  // 6
        {5.0f, 5.0f},  // 7
    }};
    for (std::size_t i = 0; i < coords.size(); ++i) {
        ds.row(i)[0] = coords[i][0];
        ds.row(i)[1] = coords[i][1];
    }
    return ds;
}

TEST(BruteForceKnn, OutputShapeMatchesArguments)
{
    const auto ds = two_clusters_eight_points();
    const knng::Knng g = knng::cpu::brute_force_knn(ds, std::size_t{3},
                                                    knng::L2Squared{});
    EXPECT_EQ(g.n, std::size_t{8});
    EXPECT_EQ(g.k, std::size_t{3});
    EXPECT_EQ(g.neighbors.size(), std::size_t{24});
    EXPECT_EQ(g.distances.size(), std::size_t{24});
}

TEST(BruteForceKnn, EightPointClusterHandVerifiedRows)
{
    const auto ds = two_clusters_eight_points();
    const knng::Knng g = knng::cpu::brute_force_knn(ds, std::size_t{3},
                                                    knng::L2Squared{});

    // Row 0 = (0,0). dist² to: 1→1, 2→1, 3→2, 4→32, …
    // Tie between (1, 1.0) and (2, 1.0) is broken by smaller id ⇒
    // 1 first, then 2, then 3.
    {
        const auto neighbors = g.neighbors_of(0);
        const auto distances = g.distances_of(0);
        EXPECT_EQ(neighbors[0], knng::index_t{1});
        EXPECT_EQ(neighbors[1], knng::index_t{2});
        EXPECT_EQ(neighbors[2], knng::index_t{3});
        EXPECT_FLOAT_EQ(distances[0], 1.0f);
        EXPECT_FLOAT_EQ(distances[1], 1.0f);
        EXPECT_FLOAT_EQ(distances[2], 2.0f);
    }

    // Row 7 = (5,5). dist² to: 5→1, 6→1, 4→2, …
    // Tie between (5, 1.0) and (6, 1.0) ⇒ 5 first.
    {
        const auto neighbors = g.neighbors_of(7);
        const auto distances = g.distances_of(7);
        EXPECT_EQ(neighbors[0], knng::index_t{5});
        EXPECT_EQ(neighbors[1], knng::index_t{6});
        EXPECT_EQ(neighbors[2], knng::index_t{4});
        EXPECT_FLOAT_EQ(distances[0], 1.0f);
        EXPECT_FLOAT_EQ(distances[1], 1.0f);
        EXPECT_FLOAT_EQ(distances[2], 2.0f);
    }

    // Row 4 = (4,4). dist² to: 5→1, 6→1, 7→2, …
    {
        const auto neighbors = g.neighbors_of(4);
        const auto distances = g.distances_of(4);
        EXPECT_EQ(neighbors[0], knng::index_t{5});
        EXPECT_EQ(neighbors[1], knng::index_t{6});
        EXPECT_EQ(neighbors[2], knng::index_t{7});
        EXPECT_FLOAT_EQ(distances[0], 1.0f);
        EXPECT_FLOAT_EQ(distances[1], 1.0f);
        EXPECT_FLOAT_EQ(distances[2], 2.0f);
    }
}

TEST(BruteForceKnn, SelfIsNeverANeighbor)
{
    const auto ds = two_clusters_eight_points();
    const knng::Knng g = knng::cpu::brute_force_knn(ds, std::size_t{5},
                                                    knng::L2Squared{});
    for (std::size_t q = 0; q < g.n; ++q) {
        for (auto id : g.neighbors_of(q)) {
            EXPECT_NE(static_cast<std::size_t>(id), q);
        }
    }
}

TEST(BruteForceKnn, RowsAreSortedAscendingByDistance)
{
    const auto ds = two_clusters_eight_points();
    const knng::Knng g = knng::cpu::brute_force_knn(ds, std::size_t{5},
                                                    knng::L2Squared{});
    for (std::size_t q = 0; q < g.n; ++q) {
        const auto distances = g.distances_of(q);
        for (std::size_t j = 1; j < distances.size(); ++j) {
            EXPECT_LE(distances[j - 1], distances[j])
                << "row " << q << " column " << j;
        }
    }
}

TEST(BruteForceKnn, KEqualsOneReturnsTheSingleClosest)
{
    const auto ds = two_clusters_eight_points();
    const knng::Knng g = knng::cpu::brute_force_knn(ds, std::size_t{1},
                                                    knng::L2Squared{});
    // Each row must have exactly one neighbor — the smallest-id
    // tie-break makes the answer deterministic for the equidistant
    // pairs (e.g. point 0's nearest is min(1, 2) == 1).
    EXPECT_EQ(g.neighbors_of(0)[0], knng::index_t{1});
    EXPECT_EQ(g.neighbors_of(3)[0], knng::index_t{1});  // (1,1) ↔ (1,0) and (0,1) tie ⇒ 1
    EXPECT_EQ(g.neighbors_of(7)[0], knng::index_t{5});
}

TEST(BruteForceKnn, NegativeInnerProductMetricCompiles)
{
    // Cross-check that the `Distance` template parameter actually
    // dispatches: build the same dataset with `NegativeInnerProduct`
    // and verify the function is callable end-to-end. The numeric
    // answer is metric-dependent and not asserted here — the goal is
    // to ensure both built-in metrics are exercised by ctest.
    const auto ds = two_clusters_eight_points();
    const knng::Knng g = knng::cpu::brute_force_knn(
        ds, std::size_t{3}, knng::NegativeInnerProduct{});
    EXPECT_EQ(g.n, std::size_t{8});
    EXPECT_EQ(g.k, std::size_t{3});
}

TEST(BruteForceKnn, ZeroKThrowsInvalidArgument)
{
    const auto ds = two_clusters_eight_points();
    EXPECT_THROW(
        knng::cpu::brute_force_knn(ds, std::size_t{0}, knng::L2Squared{}),
        std::invalid_argument);
}

TEST(BruteForceKnn, KGreaterThanNMinusOneThrowsInvalidArgument)
{
    const auto ds = two_clusters_eight_points();
    // ds.n - 1 == 7; k == 8 must be rejected.
    EXPECT_THROW(
        knng::cpu::brute_force_knn(ds, std::size_t{8}, knng::L2Squared{}),
        std::invalid_argument);
}

TEST(BruteForceKnn, EmptyDatasetThrowsInvalidArgument)
{
    const knng::Dataset ds;  // 0×0
    EXPECT_THROW(
        knng::cpu::brute_force_knn(ds, std::size_t{1}, knng::L2Squared{}),
        std::invalid_argument);
}

// ---------------------------------------------------------------------
// Step 19: brute_force_knn_l2_with_norms — must agree with
// brute_force_knn(... L2Squared{}) up to fp accumulation reordering.
// ---------------------------------------------------------------------

TEST(BruteForceKnnL2Norms, MatchesCanonicalRowsExactlyOnEightPointFixture)
{
    const auto ds = two_clusters_eight_points();
    const auto baseline = knng::cpu::brute_force_knn(
        ds, std::size_t{3}, knng::L2Squared{});
    const auto norms_path =
        knng::cpu::brute_force_knn_l2_with_norms(ds, std::size_t{3});

    ASSERT_EQ(baseline.n, norms_path.n);
    ASSERT_EQ(baseline.k, norms_path.k);

    // Hand-crafted fixture has well-separated clusters; every row's
    // top-3 neighbors are unique-by-distance, so neighbor IDs must
    // match exactly between the two paths and distances must be
    // numerically very close (the algebraic identity shifts the
    // accumulation order but not the result up to fp reorder).
    for (std::size_t q = 0; q < baseline.n; ++q) {
        const auto base_n  = baseline.neighbors_of(q);
        const auto norms_n = norms_path.neighbors_of(q);
        for (std::size_t j = 0; j < baseline.k; ++j) {
            EXPECT_EQ(base_n[j], norms_n[j]) << "row " << q << " col " << j;
        }
        const auto base_d  = baseline.distances_of(q);
        const auto norms_d = norms_path.distances_of(q);
        for (std::size_t j = 0; j < baseline.k; ++j) {
            EXPECT_NEAR(base_d[j], norms_d[j], 1e-4f)
                << "row " << q << " col " << j;
        }
    }
}

TEST(BruteForceKnnL2Norms, DistancesAreNonNegative)
{
    const auto ds = two_clusters_eight_points();
    const auto g = knng::cpu::brute_force_knn_l2_with_norms(
        ds, std::size_t{5});
    for (std::size_t q = 0; q < g.n; ++q) {
        for (auto d : g.distances_of(q)) {
            EXPECT_GE(d, 0.0f);
        }
    }
}

TEST(BruteForceKnnL2Norms, ZeroKThrowsInvalidArgument)
{
    const auto ds = two_clusters_eight_points();
    EXPECT_THROW(
        { (void)knng::cpu::brute_force_knn_l2_with_norms(
            ds, std::size_t{0}); },
        std::invalid_argument);
}

TEST(BruteForceKnnL2Norms, KGreaterThanNMinusOneThrowsInvalidArgument)
{
    const auto ds = two_clusters_eight_points();
    EXPECT_THROW(
        { (void)knng::cpu::brute_force_knn_l2_with_norms(
            ds, std::size_t{8}); },
        std::invalid_argument);
}

TEST(BruteForceKnnL2Norms, EmptyDatasetThrowsInvalidArgument)
{
    const knng::Dataset ds;
    EXPECT_THROW(
        { (void)knng::cpu::brute_force_knn_l2_with_norms(
            ds, std::size_t{1}); },
        std::invalid_argument);
}

// ---------------------------------------------------------------------
// Step 20: brute_force_knn_l2_tiled — must agree with the canonical
// L2 path for any tile-size choice that is at least 1×1.
// ---------------------------------------------------------------------

TEST(BruteForceKnnL2Tiled, MatchesCanonicalAtDefaultTileSizes)
{
    const auto ds = two_clusters_eight_points();
    const auto baseline = knng::cpu::brute_force_knn(
        ds, std::size_t{3}, knng::L2Squared{});
    const auto tiled = knng::cpu::brute_force_knn_l2_tiled(
        ds, std::size_t{3});  // defaults: 32 × 128

    ASSERT_EQ(baseline.n, tiled.n);
    ASSERT_EQ(baseline.k, tiled.k);
    EXPECT_EQ(baseline.neighbors, tiled.neighbors);
    for (std::size_t i = 0; i < baseline.distances.size(); ++i) {
        EXPECT_NEAR(baseline.distances[i], tiled.distances[i], 1e-4f);
    }
}

TEST(BruteForceKnnL2Tiled, MatchesCanonicalWhenTilesSmallerThanN)
{
    // Tile sizes that force multiple iterations of both outer and
    // inner tile loops on the 8-point fixture (n=8): 3 × 5 splits
    // queries into [0..3, 3..6, 6..8] and references into [0..5, 5..8].
    const auto ds = two_clusters_eight_points();
    const auto baseline = knng::cpu::brute_force_knn(
        ds, std::size_t{3}, knng::L2Squared{});
    const auto tiled = knng::cpu::brute_force_knn_l2_tiled(
        ds, std::size_t{3}, std::size_t{3}, std::size_t{5});

    ASSERT_EQ(baseline.n, tiled.n);
    ASSERT_EQ(baseline.k, tiled.k);
    EXPECT_EQ(baseline.neighbors, tiled.neighbors);
}

TEST(BruteForceKnnL2Tiled, MatchesCanonicalWhenTilesAreOne)
{
    // Degenerate 1 × 1 tiling — exercises the boundary handling
    // at every (q_lo, r_lo) increment. Output must still match.
    const auto ds = two_clusters_eight_points();
    const auto baseline = knng::cpu::brute_force_knn(
        ds, std::size_t{2}, knng::L2Squared{});
    const auto tiled = knng::cpu::brute_force_knn_l2_tiled(
        ds, std::size_t{2}, std::size_t{1}, std::size_t{1});

    EXPECT_EQ(baseline.neighbors, tiled.neighbors);
}

TEST(BruteForceKnnL2Tiled, ZeroTileSizeThrowsInvalidArgument)
{
    const auto ds = two_clusters_eight_points();
    EXPECT_THROW(
        { (void)knng::cpu::brute_force_knn_l2_tiled(
            ds, std::size_t{3}, std::size_t{0}, std::size_t{8}); },
        std::invalid_argument);
    EXPECT_THROW(
        { (void)knng::cpu::brute_force_knn_l2_tiled(
            ds, std::size_t{3}, std::size_t{8}, std::size_t{0}); },
        std::invalid_argument);
}

TEST(BruteForceKnnL2Tiled, ZeroKThrowsInvalidArgument)
{
    const auto ds = two_clusters_eight_points();
    EXPECT_THROW(
        { (void)knng::cpu::brute_force_knn_l2_tiled(ds, std::size_t{0}); },
        std::invalid_argument);
}

TEST(BruteForceKnnL2Tiled, KGreaterThanNMinusOneThrowsInvalidArgument)
{
    const auto ds = two_clusters_eight_points();
    EXPECT_THROW(
        { (void)knng::cpu::brute_force_knn_l2_tiled(ds, std::size_t{8}); },
        std::invalid_argument);
}

#if defined(KNNG_HAVE_BLAS) && KNNG_HAVE_BLAS

// ---------------------------------------------------------------------
// Step 21: brute_force_knn_l2_blas — sgemm-backed cross-term path.
// Must agree with the canonical L2 builder up to fp accumulation
// reordering (BLAS reorders pretty aggressively, so the tolerance is
// looser than for the norms / tiled paths).
// ---------------------------------------------------------------------

TEST(BruteForceKnnL2Blas, MatchesCanonicalNeighborsAtDefaultTileSizes)
{
    const auto ds = two_clusters_eight_points();
    const auto baseline = knng::cpu::brute_force_knn(
        ds, std::size_t{3}, knng::L2Squared{});
    const auto blas_path = knng::cpu::brute_force_knn_l2_blas(
        ds, std::size_t{3});

    ASSERT_EQ(baseline.n, blas_path.n);
    ASSERT_EQ(baseline.k, blas_path.k);

    // Hand-verified fixture has neighbours unique-by-distance, so
    // neighbour IDs match exactly. Distances match within a small
    // absolute tolerance (BLAS may reorder the dot product).
    EXPECT_EQ(baseline.neighbors, blas_path.neighbors);
    for (std::size_t i = 0; i < baseline.distances.size(); ++i) {
        EXPECT_NEAR(baseline.distances[i], blas_path.distances[i], 1e-3f)
            << "i = " << i;
    }
}

TEST(BruteForceKnnL2Blas, MatchesCanonicalWhenTilesSmallerThanN)
{
    const auto ds = two_clusters_eight_points();
    const auto baseline = knng::cpu::brute_force_knn(
        ds, std::size_t{3}, knng::L2Squared{});
    const auto blas_path = knng::cpu::brute_force_knn_l2_blas(
        ds, std::size_t{3}, std::size_t{3}, std::size_t{5});

    EXPECT_EQ(baseline.neighbors, blas_path.neighbors);
}

TEST(BruteForceKnnL2Blas, MatchesCanonicalWhenTilesAreOne)
{
    const auto ds = two_clusters_eight_points();
    const auto baseline = knng::cpu::brute_force_knn(
        ds, std::size_t{2}, knng::L2Squared{});
    const auto blas_path = knng::cpu::brute_force_knn_l2_blas(
        ds, std::size_t{2}, std::size_t{1}, std::size_t{1});

    EXPECT_EQ(baseline.neighbors, blas_path.neighbors);
}

TEST(BruteForceKnnL2Blas, ZeroTileSizeThrowsInvalidArgument)
{
    const auto ds = two_clusters_eight_points();
    EXPECT_THROW(
        { (void)knng::cpu::brute_force_knn_l2_blas(
            ds, std::size_t{3}, std::size_t{0}, std::size_t{8}); },
        std::invalid_argument);
}

TEST(BruteForceKnnL2Blas, ZeroKThrowsInvalidArgument)
{
    const auto ds = two_clusters_eight_points();
    EXPECT_THROW(
        { (void)knng::cpu::brute_force_knn_l2_blas(ds, std::size_t{0}); },
        std::invalid_argument);
}

TEST(BruteForceKnnL2Blas, BuiltinFlagReportsTrue)
{
    static_assert(knng::cpu::kHasBlasBuiltin,
                  "Step 21 was compiled but the builtin flag says no");
    EXPECT_TRUE(knng::cpu::kHasBlasBuiltin);
}

#endif  // KNNG_HAVE_BLAS

} // namespace
