/// @file
/// @brief Unit tests for `knng::bench::recall_at_k` (Step 15).
///
/// Pins the contract: `[0, 1]`-valued, set-intersection-based, robust
/// against duplicates in the approximate row, intolerant of shape
/// mismatches between `approx` and `truth`.

#include <array>
#include <cstddef>
#include <stdexcept>

#include <gtest/gtest.h>

#include "knng/bench/recall.hpp"
#include "knng/core/dataset.hpp"
#include "knng/core/distance.hpp"
#include "knng/core/graph.hpp"
#include "knng/core/types.hpp"
#include "knng/cpu/brute_force.hpp"

namespace {

/// Build a 4-row, k=3 graph with the given neighbor IDs. Distances
/// are populated as `0.1 * column` so the sorted-by-distance
/// invariant is harmless to maintain — only the IDs matter for
/// recall.
knng::Knng make_graph_4x3(const std::array<std::array<knng::index_t, 3>, 4>& rows)
{
    knng::Knng g(4, 3);
    for (std::size_t i = 0; i < 4; ++i) {
        auto neighbors = g.neighbors_of(i);
        auto distances = g.distances_of(i);
        for (std::size_t j = 0; j < 3; ++j) {
            neighbors[j] = rows[i][j];
            distances[j] = 0.1f * static_cast<float>(j);
        }
    }
    return g;
}

TEST(Recall, ExactMatchYieldsOne)
{
    const auto truth = make_graph_4x3({{
        {{1, 2, 3}}, {{0, 2, 3}}, {{0, 1, 3}}, {{0, 1, 2}},
    }});
    EXPECT_DOUBLE_EQ(knng::bench::recall_at_k(truth, truth), 1.0);
}

TEST(Recall, OrderInsideRowDoesNotMatter)
{
    const auto truth = make_graph_4x3({{
        {{1, 2, 3}}, {{0, 2, 3}}, {{0, 1, 3}}, {{0, 1, 2}},
    }});
    // Same per-row sets, scrambled order.
    const auto approx = make_graph_4x3({{
        {{3, 1, 2}}, {{2, 0, 3}}, {{3, 0, 1}}, {{2, 1, 0}},
    }});
    EXPECT_DOUBLE_EQ(knng::bench::recall_at_k(approx, truth), 1.0);
}

TEST(Recall, ZeroOverlapYieldsZero)
{
    const auto truth = make_graph_4x3({{
        {{1, 2, 3}}, {{0, 2, 3}}, {{0, 1, 3}}, {{0, 1, 2}},
    }});
    // Approx fills every row with the same nonsense IDs that are
    // disjoint from the corresponding truth row.
    const auto approx = make_graph_4x3({{
        {{4, 5, 6}}, {{4, 5, 6}}, {{4, 5, 6}}, {{4, 5, 6}},
    }});
    EXPECT_DOUBLE_EQ(knng::bench::recall_at_k(approx, truth), 0.0);
}

TEST(Recall, PartialOverlapAveragesAcrossRows)
{
    const auto truth = make_graph_4x3({{
        {{1, 2, 3}}, {{0, 2, 3}}, {{0, 1, 3}}, {{0, 1, 2}},
    }});
    // Row 0: 2/3 hits (1, 2 hit; 99 miss).
    // Row 1: 3/3 hits.
    // Row 2: 1/3 hit (0 hit; 88, 99 miss).
    // Row 3: 0/3 hits.
    // Total = 6 / 12 = 0.5.
    const auto approx = make_graph_4x3({{
        {{1, 2, 99}}, {{0, 2, 3}}, {{0, 88, 99}}, {{77, 88, 99}},
    }});
    EXPECT_DOUBLE_EQ(knng::bench::recall_at_k(approx, truth), 0.5);
}

TEST(Recall, DuplicatesInApproxRowDoNotInflate)
{
    const auto truth = make_graph_4x3({{
        {{1, 2, 3}}, {{0, 2, 3}}, {{0, 1, 3}}, {{0, 1, 2}},
    }});
    // A malformed builder that repeats a single correct neighbor in
    // every column should score 1/3 per row, not 3/3.
    const auto approx = make_graph_4x3({{
        {{1, 1, 1}}, {{0, 0, 0}}, {{0, 0, 0}}, {{0, 0, 0}},
    }});
    EXPECT_DOUBLE_EQ(knng::bench::recall_at_k(approx, truth),
                     4.0 / 12.0);
}

TEST(Recall, RowAccessorReturnsIntegerCount)
{
    const auto truth = make_graph_4x3({{
        {{1, 2, 3}}, {{0, 2, 3}}, {{0, 1, 3}}, {{0, 1, 2}},
    }});
    const auto approx = make_graph_4x3({{
        {{1, 2, 99}}, {{0, 2, 3}}, {{0, 88, 99}}, {{77, 88, 99}},
    }});
    EXPECT_EQ(knng::bench::recall_at_k_row(approx, truth, 0), std::size_t{2});
    EXPECT_EQ(knng::bench::recall_at_k_row(approx, truth, 1), std::size_t{3});
    EXPECT_EQ(knng::bench::recall_at_k_row(approx, truth, 2), std::size_t{1});
    EXPECT_EQ(knng::bench::recall_at_k_row(approx, truth, 3), std::size_t{0});
}

TEST(Recall, NMismatchThrows)
{
    knng::Knng a(2, 3);
    knng::Knng b(3, 3);
    EXPECT_THROW(
        { (void)knng::bench::recall_at_k(a, b); },
        std::invalid_argument);
}

TEST(Recall, KMismatchThrows)
{
    knng::Knng a(4, 3);
    knng::Knng b(4, 5);
    EXPECT_THROW(
        { (void)knng::bench::recall_at_k(a, b); },
        std::invalid_argument);
}

TEST(Recall, EmptyGraphReturnsOne)
{
    const knng::Knng a;  // 0×0
    const knng::Knng b;
    EXPECT_DOUBLE_EQ(knng::bench::recall_at_k(a, b), 1.0);
}

TEST(Recall, RowIndexOutOfRangeThrows)
{
    knng::Knng a(4, 3);
    knng::Knng b(4, 3);
    EXPECT_THROW(
        { (void)knng::bench::recall_at_k_row(a, b, std::size_t{4}); },
        std::invalid_argument);
}

TEST(Recall, BruteForceAgainstItselfIsOne)
{
    // End-to-end sanity: brute-force is its own ground truth, so
    // recall against itself must be exactly 1.0 — this catches any
    // future regression where a refactor accidentally drops or
    // permutes a neighbor field.
    knng::Dataset ds(8, 2);
    constexpr std::array<std::array<float, 2>, 8> coords{{
        {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f},
        {4.0f, 4.0f}, {5.0f, 4.0f}, {4.0f, 5.0f}, {5.0f, 5.0f},
    }};
    for (std::size_t i = 0; i < coords.size(); ++i) {
        ds.row(i)[0] = coords[i][0];
        ds.row(i)[1] = coords[i][1];
    }
    const auto g = knng::cpu::brute_force_knn(
        ds, std::size_t{3}, knng::L2Squared{});
    EXPECT_DOUBLE_EQ(knng::bench::recall_at_k(g, g), 1.0);
}

} // namespace
