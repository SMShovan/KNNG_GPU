/// @file
/// @brief Unit tests for the NN-Descent public surface (Steps 31+).
///
/// Step 31 contributes the random-graph initialiser tests below.
/// Subsequent Phase-5 steps (32–36) will append to this TU.

#include <array>
#include <cstddef>
#include <set>
#include <stdexcept>

#include <gtest/gtest.h>

#include "knng/core/dataset.hpp"
#include "knng/core/distance.hpp"
#include "knng/core/graph.hpp"
#include "knng/core/types.hpp"
#include "knng/cpu/nn_descent.hpp"

namespace {

/// Cluster fixture from the brute-force tests, reused so the
/// expected-shape assertions read familiarly.
knng::Dataset two_clusters_eight_points()
{
    knng::Dataset ds(8, 2);
    constexpr std::array<std::array<float, 2>, 8> coords{{
        {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f},
        {4.0f, 4.0f}, {5.0f, 4.0f}, {4.0f, 5.0f}, {5.0f, 5.0f},
    }};
    for (std::size_t i = 0; i < coords.size(); ++i) {
        ds.row(i)[0] = coords[i][0];
        ds.row(i)[1] = coords[i][1];
    }
    return ds;
}

// --- NnDescentGraph plumbing -----------------------------------------

TEST(NnDescentGraph, ConstructedShapeIsRespected)
{
    knng::cpu::NnDescentGraph g(/*n=*/4, /*k=*/3);
    EXPECT_EQ(g.n(), std::size_t{4});
    EXPECT_EQ(g.k(), std::size_t{3});
    EXPECT_EQ(g.lists().size(), std::size_t{4});
    for (std::size_t i = 0; i < g.n(); ++i) {
        EXPECT_TRUE(g.at(i).empty());
        EXPECT_EQ(g.at(i).capacity(), std::size_t{3});
    }
}

TEST(NnDescentGraph, ToKnngFillsMissingSlotsWithSentinels)
{
    knng::cpu::NnDescentGraph g(2, 3);
    // Populate row 0 with two entries; leave row 1 empty.
    g.at(0).insert(5, 0.5f, true);
    g.at(0).insert(7, 0.3f, true);

    const auto k = g.to_knng();
    EXPECT_EQ(k.n, std::size_t{2});
    EXPECT_EQ(k.k, std::size_t{3});
    // Row 0: [(7, 0.3), (5, 0.5), (sentinel, +inf)]
    EXPECT_EQ(k.neighbors_of(0)[0], knng::index_t{7});
    EXPECT_EQ(k.neighbors_of(0)[1], knng::index_t{5});
    EXPECT_EQ(k.neighbors_of(0)[2],
              knng::cpu::NeighborList::kEmptyId);
    EXPECT_FLOAT_EQ(k.distances_of(0)[0], 0.3f);
    EXPECT_FLOAT_EQ(k.distances_of(0)[1], 0.5f);
    EXPECT_TRUE(std::isinf(k.distances_of(0)[2]));
    // Row 1: all sentinels.
    for (std::size_t j = 0; j < 3; ++j) {
        EXPECT_EQ(k.neighbors_of(1)[j],
                  knng::cpu::NeighborList::kEmptyId);
        EXPECT_TRUE(std::isinf(k.distances_of(1)[j]));
    }
}

// --- init_random_graph -----------------------------------------------

TEST(InitRandomGraph, OutputShapeMatchesArguments)
{
    const auto ds = two_clusters_eight_points();
    const auto g = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{42}, knng::L2Squared{});
    EXPECT_EQ(g.n(), std::size_t{8});
    EXPECT_EQ(g.k(), std::size_t{3});
    for (std::size_t i = 0; i < g.n(); ++i) {
        EXPECT_EQ(g.at(i).size(), std::size_t{3});
    }
}

TEST(InitRandomGraph, EveryEntryIsNonSelfDistinctAndIsNew)
{
    const auto ds = two_clusters_eight_points();
    const auto g = knng::cpu::init_random_graph(
        ds, std::size_t{4}, std::uint64_t{1234}, knng::L2Squared{});

    for (std::size_t p = 0; p < g.n(); ++p) {
        std::set<knng::index_t> seen;
        for (const auto& nb : g.at(p).view()) {
            EXPECT_NE(static_cast<std::size_t>(nb.id), p);
            EXPECT_TRUE(seen.insert(nb.id).second)
                << "row " << p << " contains duplicate id " << nb.id;
            EXPECT_TRUE(nb.is_new);
            EXPECT_GE(nb.dist, 0.0f);
        }
    }
}

TEST(InitRandomGraph, RowsAreSortedAscendingByDistance)
{
    const auto ds = two_clusters_eight_points();
    const auto g = knng::cpu::init_random_graph(
        ds, std::size_t{4}, std::uint64_t{99}, knng::L2Squared{});

    for (std::size_t p = 0; p < g.n(); ++p) {
        const auto v = g.at(p).view();
        for (std::size_t j = 1; j < v.size(); ++j) {
            EXPECT_LE(v[j - 1].dist, v[j].dist)
                << "row " << p << " column " << j;
        }
    }
}

TEST(InitRandomGraph, SameSeedYieldsIdenticalGraphs)
{
    const auto ds = two_clusters_eight_points();
    const auto a = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{2026}, knng::L2Squared{});
    const auto b = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{2026}, knng::L2Squared{});
    for (std::size_t p = 0; p < a.n(); ++p) {
        const auto va = a.at(p).view();
        const auto vb = b.at(p).view();
        ASSERT_EQ(va.size(), vb.size());
        for (std::size_t j = 0; j < va.size(); ++j) {
            EXPECT_EQ(va[j].id, vb[j].id);
            EXPECT_FLOAT_EQ(va[j].dist, vb[j].dist);
        }
    }
}

TEST(InitRandomGraph, DifferentSeedsYieldDifferentGraphs)
{
    const auto ds = two_clusters_eight_points();
    const auto a = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{1}, knng::L2Squared{});
    const auto b = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{2}, knng::L2Squared{});

    bool any_diff = false;
    for (std::size_t p = 0; p < a.n() && !any_diff; ++p) {
        const auto va = a.at(p).view();
        const auto vb = b.at(p).view();
        for (std::size_t j = 0; j < va.size(); ++j) {
            if (va[j].id != vb[j].id) {
                any_diff = true;
                break;
            }
        }
    }
    EXPECT_TRUE(any_diff)
        << "two distinct seeds produced byte-identical graphs";
}

TEST(InitRandomGraph, DistancesMatchUnderlyingMetric)
{
    const auto ds = two_clusters_eight_points();
    const auto g = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{777}, knng::L2Squared{});
    knng::L2Squared metric;
    for (std::size_t p = 0; p < g.n(); ++p) {
        for (const auto& nb : g.at(p).view()) {
            const float expected = metric(ds.row(p), ds.row(nb.id));
            EXPECT_FLOAT_EQ(nb.dist, expected)
                << "row " << p << " neighbour " << nb.id;
        }
    }
}

TEST(InitRandomGraph, ToKnngRoundTripIsConsistent)
{
    const auto ds = two_clusters_eight_points();
    const auto g = knng::cpu::init_random_graph(
        ds, std::size_t{4}, std::uint64_t{42}, knng::L2Squared{});
    const auto k = g.to_knng();
    EXPECT_EQ(k.n, g.n());
    EXPECT_EQ(k.k, g.k());
    for (std::size_t p = 0; p < g.n(); ++p) {
        const auto v = g.at(p).view();
        const auto neigh = k.neighbors_of(p);
        const auto dist  = k.distances_of(p);
        for (std::size_t j = 0; j < g.k(); ++j) {
            EXPECT_EQ(neigh[j], v[j].id);
            EXPECT_FLOAT_EQ(dist[j], v[j].dist);
        }
    }
}

TEST(InitRandomGraph, NegativeInnerProductMetricCompiles)
{
    const auto ds = two_clusters_eight_points();
    const auto g = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{42},
        knng::NegativeInnerProduct{});
    EXPECT_EQ(g.n(), std::size_t{8});
    EXPECT_EQ(g.k(), std::size_t{3});
}

TEST(InitRandomGraph, ZeroKThrowsInvalidArgument)
{
    const auto ds = two_clusters_eight_points();
    EXPECT_THROW(
        { (void)knng::cpu::init_random_graph(
            ds, std::size_t{0}, std::uint64_t{1},
            knng::L2Squared{}); },
        std::invalid_argument);
}

TEST(InitRandomGraph, KGreaterThanNMinusOneThrowsInvalidArgument)
{
    const auto ds = two_clusters_eight_points();
    EXPECT_THROW(
        { (void)knng::cpu::init_random_graph(
            ds, std::size_t{8}, std::uint64_t{1},
            knng::L2Squared{}); },
        std::invalid_argument);
}

TEST(InitRandomGraph, EmptyDatasetThrowsInvalidArgument)
{
    const knng::Dataset ds;
    EXPECT_THROW(
        { (void)knng::cpu::init_random_graph(
            ds, std::size_t{1}, std::uint64_t{1},
            knng::L2Squared{}); },
        std::invalid_argument);
}

} // namespace
