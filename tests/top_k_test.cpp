/// @file
/// @brief Unit tests for `knng::TopK`.
///
/// Pins the small behavioural contract of the bounded-size top-k
/// container introduced at Step 09: capacity invariant, ordering of
/// the extracted output, deterministic tie-breaking on equal
/// distances, and the degenerate `k == 0` case.

#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "knng/core/types.hpp"
#include "knng/top_k.hpp"

namespace {

TEST(TopK, EmptyExtractIsEmpty)
{
    knng::TopK heap(5);
    EXPECT_TRUE(heap.empty());
    EXPECT_EQ(heap.size(), std::size_t{0});
    EXPECT_EQ(heap.capacity(), std::size_t{5});
    EXPECT_TRUE(heap.extract_sorted().empty());
}

TEST(TopK, FewerThanKAreAllRetainedSortedAscending)
{
    knng::TopK heap(5);
    heap.push(3, 0.30f);
    heap.push(1, 0.10f);
    heap.push(2, 0.20f);

    EXPECT_EQ(heap.size(), std::size_t{3});

    const auto out = heap.extract_sorted();
    ASSERT_EQ(out.size(), std::size_t{3});
    EXPECT_EQ(out[0].first, knng::index_t{1});
    EXPECT_FLOAT_EQ(out[0].second, 0.10f);
    EXPECT_EQ(out[1].first, knng::index_t{2});
    EXPECT_FLOAT_EQ(out[1].second, 0.20f);
    EXPECT_EQ(out[2].first, knng::index_t{3});
    EXPECT_FLOAT_EQ(out[2].second, 0.30f);
    EXPECT_TRUE(heap.empty());
}

TEST(TopK, SizeKInvariantUnderRepeatedInsertion)
{
    constexpr std::size_t k = 3;
    knng::TopK heap(k);

    // Push ten values in arbitrary order; only the three smallest
    // must survive: 0.1, 0.2, 0.3.
    const std::vector<std::pair<knng::index_t, float>> input{
        {0, 0.7f},
        {1, 0.1f},
        {2, 0.9f},
        {3, 0.4f},
        {4, 0.2f},
        {5, 0.8f},
        {6, 0.3f},
        {7, 0.6f},
        {8, 0.5f},
        {9, 0.95f},
    };
    for (const auto& [id, dist] : input) {
        heap.push(id, dist);
        EXPECT_LE(heap.size(), k);
    }

    const auto out = heap.extract_sorted();
    ASSERT_EQ(out.size(), k);
    EXPECT_EQ(out[0].first, knng::index_t{1});
    EXPECT_FLOAT_EQ(out[0].second, 0.10f);
    EXPECT_EQ(out[1].first, knng::index_t{4});
    EXPECT_FLOAT_EQ(out[1].second, 0.20f);
    EXPECT_EQ(out[2].first, knng::index_t{6});
    EXPECT_FLOAT_EQ(out[2].second, 0.30f);
}

TEST(TopK, EqualDistancesAreTieBrokenBySmallerId)
{
    knng::TopK heap(2);
    // Three candidates all at distance 1.0 — only the two with the
    // smallest ids must survive.
    heap.push(7, 1.0f);
    heap.push(2, 1.0f);
    heap.push(5, 1.0f);

    const auto out = heap.extract_sorted();
    ASSERT_EQ(out.size(), std::size_t{2});
    EXPECT_EQ(out[0].first, knng::index_t{2});
    EXPECT_EQ(out[1].first, knng::index_t{5});
    EXPECT_FLOAT_EQ(out[0].second, 1.0f);
    EXPECT_FLOAT_EQ(out[1].second, 1.0f);
}

TEST(TopK, ZeroCapacityRejectsEverything)
{
    knng::TopK heap(0);
    heap.push(1, 0.0f);
    heap.push(2, 0.0f);
    heap.push(3, 0.5f);

    EXPECT_EQ(heap.size(), std::size_t{0});
    EXPECT_TRUE(heap.extract_sorted().empty());
}

TEST(TopK, ExtractDrainsTheBuffer)
{
    knng::TopK heap(3);
    heap.push(1, 0.1f);
    heap.push(2, 0.2f);
    EXPECT_EQ(heap.extract_sorted().size(), std::size_t{2});
    EXPECT_TRUE(heap.empty());
    EXPECT_TRUE(heap.extract_sorted().empty());
}

} // namespace
