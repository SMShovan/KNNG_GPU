/// @file
/// @brief Unit tests for `knng::cpu::NeighborList` (Step 30).
///
/// Pins every behaviour the Step-32+ local-join kernel depends on:
/// sorted-by-distance invariant, deterministic tie-break, duplicate
/// rejection, capacity eviction, `is_new` accounting, the empty /
/// degenerate / full state transitions, and the worst-distance
/// query.

#include <cstddef>
#include <limits>

#include <gtest/gtest.h>

#include "knng/cpu/neighbor_list.hpp"
#include "knng/core/types.hpp"

namespace {

using knng::cpu::NeighborList;

TEST(NeighborList, FreshlyConstructedIsEmpty)
{
    NeighborList list(5);
    EXPECT_EQ(list.size(), std::size_t{0});
    EXPECT_EQ(list.capacity(), std::size_t{5});
    EXPECT_TRUE(list.empty());
    EXPECT_FALSE(list.full());
    EXPECT_TRUE(list.view().empty());
    EXPECT_EQ(list.worst_dist(),
              std::numeric_limits<float>::infinity());
}

TEST(NeighborList, InsertBelowCapacityRetainsAllAndReturnsTrue)
{
    NeighborList list(4);
    EXPECT_TRUE (list.insert(7, 0.7f, true));
    EXPECT_TRUE (list.insert(3, 0.3f, true));
    EXPECT_TRUE (list.insert(5, 0.5f, false));

    ASSERT_EQ(list.size(), std::size_t{3});
    const auto v = list.view();
    EXPECT_EQ(v[0].id, knng::index_t{3}); EXPECT_FLOAT_EQ(v[0].dist, 0.3f);
    EXPECT_EQ(v[1].id, knng::index_t{5}); EXPECT_FLOAT_EQ(v[1].dist, 0.5f);
    EXPECT_EQ(v[2].id, knng::index_t{7}); EXPECT_FLOAT_EQ(v[2].dist, 0.7f);
    EXPECT_TRUE (v[0].is_new);
    EXPECT_FALSE(v[1].is_new);
    EXPECT_TRUE (v[2].is_new);
}

TEST(NeighborList, EqualDistancesAreTieBrokenBySmallerId)
{
    NeighborList list(3);
    list.insert(7, 1.0f, true);
    list.insert(2, 1.0f, true);
    list.insert(5, 1.0f, true);

    const auto v = list.view();
    ASSERT_EQ(v.size(), std::size_t{3});
    EXPECT_EQ(v[0].id, knng::index_t{2});
    EXPECT_EQ(v[1].id, knng::index_t{5});
    EXPECT_EQ(v[2].id, knng::index_t{7});
}

TEST(NeighborList, AtCapacityKeepsBestAndEvictsWorst)
{
    NeighborList list(3);
    list.insert(0, 0.7f, true);
    list.insert(1, 0.5f, true);
    list.insert(2, 0.3f, true);
    EXPECT_TRUE(list.full());
    EXPECT_FLOAT_EQ(list.worst_dist(), 0.7f);

    // Insert better candidate — must displace the worst.
    EXPECT_TRUE(list.insert(3, 0.1f, true));
    const auto v = list.view();
    ASSERT_EQ(v.size(), std::size_t{3});
    EXPECT_EQ(v[0].id, knng::index_t{3});  // 0.1
    EXPECT_EQ(v[1].id, knng::index_t{2});  // 0.3
    EXPECT_EQ(v[2].id, knng::index_t{1});  // 0.5

    // Insert worse candidate — must be rejected, list unchanged.
    EXPECT_FALSE(list.insert(99, 1.0f, true));
    EXPECT_EQ(list.view()[2].id, knng::index_t{1});
}

TEST(NeighborList, EvictionOnDistanceTieFavoursSmallerId)
{
    NeighborList list(2);
    list.insert(5, 1.0f, true);
    list.insert(8, 1.0f, true);
    // List is now [(5, 1.0), (8, 1.0)]. A new candidate at the
    // same dist with smaller id (3) must displace 8.
    EXPECT_TRUE(list.insert(3, 1.0f, true));
    const auto v = list.view();
    EXPECT_EQ(v[0].id, knng::index_t{3});
    EXPECT_EQ(v[1].id, knng::index_t{5});
}

TEST(NeighborList, DuplicateIdWithWorseDistIsIgnored)
{
    NeighborList list(3);
    list.insert(7, 0.5f, true);
    EXPECT_FALSE(list.insert(7, 0.9f, true));
    ASSERT_EQ(list.size(), std::size_t{1});
    EXPECT_FLOAT_EQ(list.view()[0].dist, 0.5f);
}

TEST(NeighborList, DuplicateIdWithBetterDistReplaces)
{
    NeighborList list(3);
    list.insert(7, 0.9f, false);
    EXPECT_TRUE(list.insert(7, 0.4f, true));
    ASSERT_EQ(list.size(), std::size_t{1});
    EXPECT_FLOAT_EQ(list.view()[0].dist, 0.4f);
    EXPECT_TRUE(list.view()[0].is_new);  // takes the new flag
}

TEST(NeighborList, MarkAllOldFlipsEveryIsNew)
{
    NeighborList list(4);
    list.insert(1, 0.1f, true);
    list.insert(2, 0.2f, true);
    list.insert(3, 0.3f, false);
    list.mark_all_old();
    for (const auto& n : list.view()) {
        EXPECT_FALSE(n.is_new);
    }
}

TEST(NeighborList, ContainsScansLinearly)
{
    NeighborList list(4);
    list.insert(10, 0.1f, true);
    list.insert(20, 0.2f, true);
    list.insert(30, 0.3f, true);
    EXPECT_TRUE (list.contains(20));
    EXPECT_TRUE (list.contains(10));
    EXPECT_TRUE (list.contains(30));
    EXPECT_FALSE(list.contains(99));
    EXPECT_FALSE(list.contains(0));
}

TEST(NeighborList, ZeroCapacityRejectsEverything)
{
    NeighborList list(0);
    EXPECT_FALSE(list.insert(1, 0.0f, true));
    EXPECT_FALSE(list.insert(2, 0.5f, true));
    EXPECT_TRUE(list.empty());
    EXPECT_TRUE(list.full());  // 0 == capacity
    EXPECT_TRUE(list.view().empty());
}

TEST(NeighborList, WorstDistTracksTheTrailingElement)
{
    NeighborList list(3);
    EXPECT_EQ(list.worst_dist(),
              std::numeric_limits<float>::infinity());
    list.insert(1, 0.5f, true);
    EXPECT_FLOAT_EQ(list.worst_dist(), 0.5f);
    list.insert(2, 0.2f, true);
    EXPECT_FLOAT_EQ(list.worst_dist(), 0.5f);
    list.insert(3, 0.7f, true);
    EXPECT_FLOAT_EQ(list.worst_dist(), 0.7f);
    list.insert(4, 0.1f, true);  // displaces 0.7
    EXPECT_FLOAT_EQ(list.worst_dist(), 0.5f);
}

TEST(NeighborList, NewlyInsertedAfterMarkAllOldIsAgainNew)
{
    NeighborList list(3);
    list.insert(1, 0.1f, true);
    list.mark_all_old();
    list.insert(2, 0.05f, true);  // freshly inserted ⇒ is_new
    const auto v = list.view();
    ASSERT_EQ(v.size(), std::size_t{2});
    EXPECT_EQ(v[0].id, knng::index_t{2});
    EXPECT_TRUE(v[0].is_new);
    EXPECT_FALSE(v[1].is_new);
}

} // namespace
